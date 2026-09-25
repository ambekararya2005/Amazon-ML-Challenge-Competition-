"""Stage 0: convert the raw TSVs to Parquet in cache/raw/ and report their size.

Also derives cache/raw/train_pairs.parquet (s1_id, other_id, other_source): one
row per positive ground-truth pair.

For each table: read with the mandatory TSV settings, assert no NA / no '\\r',
report rows, per-country counts, empty-field counts and in-memory size, write
Parquet, read it back and assert the round-trip is identical and still NA-free.
Tables are handled one at a time and freed before the next to keep RAM low.

Usage (from code/business_entity_resolution/):
    python -m src.prepare_data            # skip tables already cached
    python -m src.prepare_data --force    # rebuild every table
"""
import argparse
import gc
import json

import pandas as pd
import pyarrow.parquet as pq

from .config import LOG_DIR, SOURCES, SPLITS, TEXT_COLS, set_seeds
from .io_utils import (ground_truth_tsv_path, raw_parquet_path, read_ground_truth_tsv,
                       read_parquet, read_source_tsv, source_tsv_path, write_parquet)
from .logging_utils import MB, StageTimer, get_logger, machine_info

SUMMARY_FILE = "data_summary.json"


def table_jobs() -> list:
    """Return (split, name, tsv_path, loader) for every raw table to convert."""
    jobs = [(split, f"source{k}", source_tsv_path(split, k),
             (lambda s=split, k=k: read_source_tsv(s, k)))
            for split in SPLITS for k in SOURCES]
    jobs.append(("train", "ground_truth", ground_truth_tsv_path("train"),
                 lambda: read_ground_truth_tsv("train")))
    return jobs


def describe(df: pd.DataFrame) -> dict:
    """Return rows, dtypes, in-memory size, per-country counts and empty-field counts of ``df``."""
    info = {
        "rows": len(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "mem_deep_mb": round(df.memory_usage(deep=True).sum() / MB, 1),
        "na_values": int(df.isna().sum().sum()),
        "empty_values": {c: int((df[c] == "").sum()) for c in df.columns},
    }
    if "country" in df.columns:
        info["countries"] = {str(k): int(v) for k, v in df["country"].value_counts().items()}
    return info


def convert_table(split: str, name: str, tsv_path, loader, force: bool, logger) -> dict:
    """Convert one TSV to Parquet (unless cached) and verify the round-trip; return its summary."""
    out = raw_parquet_path(split, name)
    stage = f"prepare_{split}_{name}"
    if out.exists() and not force:
        rows = pq.ParquetFile(out).metadata.num_rows
        logger.info("[%s] cached at %s (%d rows) - skipping (use --force to rebuild)", stage, out, rows)
        return {"rows": rows, "cached": True}

    with StageTimer(stage, logger, tsv=str(tsv_path)):
        df = loader()  # asserts no NA / no '\r' / exact header
        info = describe(df)
        info["tsv_mb"] = round(tsv_path.stat().st_size / MB, 1)
        logger.info("[%s] rows=%d mem_deep=%.1f MB dtypes=%s", stage, info["rows"],
                    info["mem_deep_mb"], info["dtypes"])
        logger.info("[%s] empty fields=%s countries=%s", stage, info["empty_values"],
                    info.get("countries"))

        write_parquet(df, out)
        back = read_parquet(out)  # asserts no NA
        pd.testing.assert_frame_equal(df, back, check_dtype=True)
        text_cols = [c for c in TEXT_COLS if c in df.columns]
        for c in text_cols:
            if int((back[c] == "").sum()) != info["empty_values"][c]:
                raise AssertionError(f"{stage}: empty-string count changed after Parquet round-trip in {c}")
        info["parquet_mb"] = round(out.stat().st_size / MB, 1)
        info["parquet_roundtrip"] = "identical, NA-free"
        logger.info("[%s] parquet %.1f MB, round-trip identical and NA-free", stage, info["parquet_mb"])
        del df, back
        gc.collect()
    return info


def build_train_pairs(force: bool, logger) -> dict:
    """Explode the train ground truth into one row per positive pair; return its summary.

    Writes cache/raw/train_pairs.parquet with columns s1_id, other_id,
    other_source ("S2"/"S3"). Asserts that every other_id is non-empty, has an
    S2-/S3- prefix, appears only once (the one-to-one property) and exists in
    its source file.
    """
    out = raw_parquet_path("train", "pairs")
    stage = "prepare_train_pairs"
    if out.exists() and not force:
        rows = pq.ParquetFile(out).metadata.num_rows
        logger.info("[%s] cached at %s (%d rows) - skipping (use --force to rebuild)", stage, out, rows)
        return {"rows": rows, "cached": True}

    with StageTimer(stage, logger):
        gt = read_parquet(raw_parquet_path("train", "ground_truth"))
        gt = gt[gt["matched_entity_ids"] != ""]
        pairs = (gt.assign(other_id=gt["matched_entity_ids"].str.split(","))
                   .explode("other_id", ignore_index=True)
                   [["source1_entity_id", "other_id"]]
                   .rename(columns={"source1_entity_id": "s1_id"}))
        pairs["other_id"] = pairs["other_id"].astype("str")
        pairs["other_source"] = pairs["other_id"].str[:2]
        del gt
        gc.collect()

        if (pairs["other_id"] == "").any():
            raise AssertionError(f"{stage}: empty id inside a matched_entity_ids list")
        bad_src = sorted(set(pairs["other_source"].unique()) - {"S2", "S3"})
        if bad_src:
            raise AssertionError(f"{stage}: unexpected id prefixes {bad_src}")
        n_dup = int(pairs["other_id"].duplicated().sum())
        if n_dup:
            raise AssertionError(f"{stage}: {n_dup} other_id values linked to more than one S1")
        missing = {}
        for k in (2, 3):
            ids = read_parquet(raw_parquet_path("train", f"source{k}"), columns=["entity_id"])["entity_id"]
            sub = pairs.loc[pairs["other_source"] == f"S{k}", "other_id"]
            missing[f"S{k}"] = int((~sub.isin(ids)).sum())
            del ids, sub
        if any(missing.values()):
            raise AssertionError(f"{stage}: ground-truth ids missing from source files: {missing}")

        write_parquet(pairs, out)
        info = {"rows": len(pairs), "s1_with_matches": int(pairs["s1_id"].nunique()),
                "by_source": {k: int(v) for k, v in pairs["other_source"].value_counts().items()},
                "duplicate_other_id": n_dup, "missing_in_source": missing,
                "mem_deep_mb": round(pairs.memory_usage(deep=True).sum() / MB, 1),
                "parquet_mb": round(out.stat().st_size / MB, 1)}
        logger.info("[%s] %s", stage, info)
        del pairs
        gc.collect()
    return info


def main() -> None:
    """Parse CLI flags, log machine info, convert every table and write logs/data_summary.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild Parquet even if cached")
    args = ap.parse_args()

    set_seeds()
    logger = get_logger("prepare_data")
    machine = machine_info()
    logger.info("machine: %s", machine)

    summary_path = LOG_DIR / SUMMARY_FILE
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["machine_at_last_run"] = machine
    tables = summary.setdefault("tables", {})

    with StageTimer("prepare_all", logger):
        for split, name, tsv_path, loader in table_jobs():
            info = convert_table(split, name, tsv_path, loader, args.force, logger)
            key = f"{split}_{name}"
            if info.get("cached") and key in tables:
                continue  # keep the full description from the run that built it
            tables[key] = info
        info = build_train_pairs(args.force, logger)
        if not (info.get("cached") and "train_pairs" in tables):
            tables["train_pairs"] = info

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("summary written to %s", summary_path)


if __name__ == "__main__":
    main()
