"""Stage 2: normalise names and addresses (rules in src/text_norm.py).

Stages (from code/business_entity_resolution/):
    python -m src.normalize --stage normalize [--force] [--workers N]
        -> cache/norm/{train,test}_s{1,2,3}.parquet, logs/normalize_summary.json
    python -m src.normalize --stage examples
        -> logs/normalize_examples.md (random + targeted before/after examples)
    python -m src.normalize --stage nonlatin_tokens
        -> most frequent name_clean tokens among names written in non-Latin script

Each file is processed one country at a time, in batches written straight to
Parquet row groups, so memory stays bounded. Batches are spread over a spawn-safe
process pool because the rules are pure-Python string code (no library threading
applies). Original business_name / business_address are kept for debugging.
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import CACHE_DIR, LOG_DIR, SEED, SOURCES, SPLITS, add_path_args, cpu_count, env_int, set_seeds
from .io_utils import list_countries, raw_parquet_path, read_parquet
from .logging_utils import StageTimer, get_logger
from .text_norm import ADDR_FIELDS, NAME_FIELDS, normalize_batch

NORM_DIR = CACHE_DIR / "norm"
BATCH_ROWS = 50_000
BOOL_FIELDS = {"is_domain", "name_nonlatin", "name_core_fallback", "addr_empty"}
BASE_FIELDS = ("entity_id", "country", "business_name", "business_address")
SCHEMA = pa.schema([(f, pa.bool_() if f in BOOL_FIELDS else pa.string())
                    for f in BASE_FIELDS + NAME_FIELDS + ADDR_FIELDS])
SUMMARY_FILE = "normalize_summary.json"
EXAMPLES_FILE = "normalize_examples.md"


def norm_path(split: str, source: int):
    """Return the cached normalised Parquet path for one split and source."""
    return NORM_DIR / f"{split}_s{source}.parquet"


def default_workers() -> int:
    """Return the default worker count: env N_WORKERS, else usable CPUs minus one (main process writes), at least one."""
    return env_int("N_WORKERS", max(1, cpu_count() - 1))


# ------------------------------------------------------------------ normalise stage
def _new_stats() -> Counter:
    """Return an empty counter for per-country statistics."""
    return Counter()


def _update_stats(stats: Counter, cols: dict) -> None:
    """Add one normalised batch's counts to ``stats``."""
    stats["rows"] += len(cols["name_core"])
    stats["empty_name_core"] += sum(1 for v in cols["name_core"] if not v)
    stats["empty_numbers"] += sum(1 for v in cols["numbers"] if not v)
    stats["filler_numbers"] += sum(1 for v in cols["filler_numbers"] if v)
    stats["name_nonlatin"] += sum(cols["name_nonlatin"])
    stats["is_domain"] += sum(cols["is_domain"])
    stats["has_legal_form"] += sum(1 for v in cols["legal_form"] if v)
    stats["name_core_fallback"] += sum(cols["name_core_fallback"])
    stats["addr_empty"] += sum(cols["addr_empty"])


def _pct(stats: Counter) -> dict:
    """Convert raw counts to percentages of rows (rows kept as a count)."""
    rows = stats["rows"] or 1
    return {"rows": stats["rows"], **{f"pct_{k}": round(100 * v / rows, 3) for k, v in stats.items() if k != "rows"}}


def _batches(df):
    """Yield (names, addresses) lists of at most BATCH_ROWS rows from ``df``."""
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    for i in range(0, len(names), BATCH_ROWS):
        yield names[i:i + BATCH_ROWS], addrs[i:i + BATCH_ROWS]


def normalize_file(split: str, source: int, pool, force: bool, logger) -> dict:
    """Normalise one raw source table country by country; return per-country stats."""
    out = norm_path(split, source)
    stage = f"normalize_{split}_s{source}"
    if out.exists() and not force:
        logger.info("[%s] cached at %s - skipping (use --force to rebuild)", stage, out)
        return {"cached": True}

    raw = raw_parquet_path(split, f"source{source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    per_country = {}
    with StageTimer(stage, logger) as timer:
        t0 = time.perf_counter()
        with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
            for country in list_countries(raw):
                df = read_parquet(raw, country=country)
                stats = _new_stats()
                mapper = pool.imap(normalize_batch, _batches(df)) if pool else map(normalize_batch, _batches(df))
                offset = 0
                for cols in mapper:
                    n = len(cols["name_core"])
                    part = df.iloc[offset:offset + n]
                    offset += n
                    arrays = {f: part[f].tolist() for f in BASE_FIELDS}
                    arrays.update(cols)
                    writer.write_table(pa.table({f: pa.array(arrays[f], type=SCHEMA.field(f).type)
                                                 for f in SCHEMA.names}, schema=SCHEMA))
                    _update_stats(stats, cols)
                per_country[country] = _pct(stats)
                logger.info("[%s] %s: %s", stage, country, per_country[country])
                del df
                gc.collect()
        os.replace(tmp, out)
        rows = sum(v["rows"] for v in per_country.values())
        rate = rows / (time.perf_counter() - t0)
        logger.info("[%s] %d rows at %.0f rows/s", stage, rows, rate)
        timer.extra.update(rows=rows, rows_per_s=round(rate))
    return {"by_country": per_country, "rows": rows, "rows_per_s": round(rate)}


def run_normalize(force: bool, workers: int, logger) -> None:
    """Normalise every train/test source file and write logs/normalize_summary.json."""
    summary_path = LOG_DIR / SUMMARY_FILE
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    files = summary.setdefault("files", {})
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers) if workers > 1 else None
    try:
        with StageTimer("normalize_all", logger, workers=workers) as timer:
            t0, total = time.perf_counter(), 0
            for split in SPLITS:
                for source in SOURCES:
                    info = normalize_file(split, source, pool, force, logger)
                    if not info.get("cached"):
                        files[f"{split}_s{source}"] = info
                        total += info["rows"]
            if total:
                rate = total / (time.perf_counter() - t0)
                summary["last_run"] = {"rows": total, "workers": workers, "rows_per_s": round(rate)}
                timer.extra.update(rows=total, rows_per_s=round(rate))
                logger.info("normalised %d rows at %.0f rows/s with %d workers", total, rate, workers)
    finally:
        if pool:
            pool.close()
            pool.join()
    summary["by_country"] = aggregate_by_country(files)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("per-country summary:\n%s", format_country_table(summary["by_country"]))


def aggregate_by_country(files: dict) -> dict:
    """Combine per-file, per-country percentages into per-(split, country) percentages."""
    acc = {}
    for key, info in files.items():
        split = key.split("_")[0]
        for country, st in info.get("by_country", {}).items():
            a = acc.setdefault(f"{split}|{country}", Counter())
            a["rows"] += st["rows"]
            for k, v in st.items():
                if k.startswith("pct_"):
                    a[k[4:]] += v * st["rows"] / 100
    return {k: _pct(v) for k, v in sorted(acc.items())}


def format_country_table(by_country: dict) -> str:
    """Render the per-(split, country) percentages as a text table."""
    cols = ["empty_name_core", "empty_numbers", "name_nonlatin", "is_domain", "filler_numbers",
            "addr_empty", "name_core_fallback"]
    lines = [f"{'split|country':<16}{'rows':>11}" + "".join(f"{c:>20}" for c in cols)]
    for key, st in by_country.items():
        lines.append(f"{key:<16}{st['rows']:>11,}" + "".join(f"{st.get('pct_' + c, 0):>19.2f}%" for c in cols))
    return "\n".join(lines)


# ------------------------------------------------------------------ examples stage
EXAMPLE_COLS = ["entity_id", "country", "business_name", "business_address", "name_clean", "legal_form",
                "name_core", "name_compact", "initials", "is_domain", "name_nonlatin", "addr_clean",
                "numbers", "filler_numbers", "num_keys", "addr_tokens"]
LEET_TOKEN = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[01345])(?![0-9]+(?:st|nd|rd|th)\b)[a-z0-9]+\b")
TARGETS = [  # (label, column, regex on the original text, country filter or None, how many)
    ("Devanagari name", "business_name", r"[ऀ-ॿ]", None, 1),
    ("Telugu name", "business_name", r"[ఀ-౿]", None, 1),
    ("Bengali name", "business_name", r"[ঀ-৿]", None, 1),
    ("domain name", "is_domain", None, None, 3),
    ("leetspeak typo", "name_clean", LEET_TOKEN.pattern, None, 3),
    ("French address with R / AV / bis", "business_address", r"(?i)(?:\b(?:r|av)\.?\s)|\bbis\b", "France", 3),
    ("US address with St", "business_address", r"\b(?:St|ST)\b", "US", 2),
    ("PMB / PO Box", "business_address", r"(?i)\b(?:pmb|p\.?\s?o\.?\s?box)\b", None, 2),
    ("dotted initials", "business_name", r"\b[A-Za-z]\.[A-Za-z]\.", None, 2),
    ("junk prefix *** / <<", "business_name", r"^\s*(?:\*\*\*|<<)", None, 2),
    ("'#' house number (India)", "business_address", r"#\s*\d", "India", 2),
]


def _sample_pool(split: str, source: int, n_groups: int, rng: random.Random) -> pa.Table:
    """Return a random subset of row groups (all countries represented) from one normalised file."""
    pf = pq.ParquetFile(norm_path(split, source))
    groups = sorted(rng.sample(range(pf.num_row_groups), min(n_groups, pf.num_row_groups)))
    return pf.read_row_groups(groups, columns=EXAMPLE_COLS)


def _fmt_example(r: dict) -> str:
    """Render one record's before/after fields as markdown."""
    return (f"- `{r['entity_id']}` ({r['country']})\n"
            f"  - name: `{r['business_name']}` -> clean `{r['name_clean']}` | core `{r['name_core']}` | "
            f"legal `{r['legal_form']}` | compact `{r['name_compact']}` | initials `{r['initials']}`"
            f"{' | DOMAIN' if r['is_domain'] else ''}{' | NONLATIN' if r['name_nonlatin'] else ''}\n"
            f"  - addr: `{r['business_address']}` -> `{r['addr_clean']}` | numbers `{r['numbers']}` | "
            f"filler `{r['filler_numbers']}` | keys `{r['num_keys']}`\n")


def run_examples(logger, per_country: int = 30, seed: int = SEED) -> None:
    """Write random and targeted before/after examples to logs/normalize_examples.md."""
    rng = random.Random(seed)
    parts = ["# Normalisation examples\n", f"Seed {seed}. Random rows are drawn from a random subset of row groups of each file.\n"]
    targeted_pool = []
    for split in SPLITS:
        tables = [_sample_pool(split, k, 12, rng) for k in SOURCES]
        pool = pa.concat_tables(tables).to_pylist()
        targeted_pool.extend(pool)
        for country in sorted({r["country"] for r in pool}):
            rows = [r for r in pool if r["country"] == country]
            parts.append(f"\n## {split} / {country}: {per_country} random examples\n")
            parts.extend(_fmt_example(r) for r in rng.sample(rows, min(per_country, len(rows))))
    parts.append("\n## Targeted examples\n")
    for label, col, pattern, country, k in TARGETS:
        rows = [r for r in targeted_pool if (country is None or r["country"] == country)
                and (r[col] if pattern is None else re.search(pattern, r[col]))]
        parts.append(f"\n### {label} ({len(rows)} candidates in the sample)\n")
        parts.extend(_fmt_example(r) for r in rng.sample(rows, min(k, len(rows))))
    path = LOG_DIR / EXAMPLES_FILE
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(parts))
    logger.info("examples written to %s", path)


# ------------------------------------------------------------------ non-Latin token stage
def run_nonlatin_tokens(logger, top: int = 60) -> Counter:
    """Log the most frequent name_clean tokens among names written in a non-Latin script."""
    counts, n_names = Counter(), 0
    for split in SPLITS:
        for source in SOURCES:
            pf = pq.ParquetFile(norm_path(split, source))
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i, columns=["name_clean", "name_nonlatin"])
                names = t.filter(pc.equal(t["name_nonlatin"], True))["name_clean"].to_pylist()
                n_names += len(names)
                for name in names:
                    counts.update(name.split())
    logger.info("top %d name tokens among %d non-Latin names:\n%s", top, n_names,
                "\n".join(f"{i + 1:>3}. {tok:<20}{c:>10,}" for i, (tok, c) in enumerate(counts.most_common(top))))
    return counts


def main() -> None:
    """Parse CLI flags and run the requested stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["normalize", "examples", "nonlatin_tokens"], default="normalize")
    ap.add_argument("--force", action="store_true", help="rebuild cached outputs")
    ap.add_argument("--workers", type=int, default=default_workers(), help="worker processes (1 = in-process)")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("normalize")
    if args.stage == "normalize":
        run_normalize(args.force, args.workers, logger)
    elif args.stage == "examples":
        run_examples(logger)
    else:
        run_nonlatin_tokens(logger)


if __name__ == "__main__":
    main()
