"""Geo-dense validation benchmark: realistic decoy density, so its F0.5 should track the leaderboard.

Units are (country, region). S1 regions come from the S1 address (region keys found by
geo_units.Gazetteer, data-driven). S2/S3 regions come from a mapping learned on train
true pairs: every text component / token of an S2/S3 address votes for the S1 region it
co-occurs with (kept if seen >= MAP_MIN_COUNT times with purity >= MAP_MIN_PURITY).
Regions are taken in stable-hash order until SAMPLE_SHARE of the country's S1 records are
covered, and dealt round-robin (hash order) into N_FOLDS folds; fold 0 is the benchmark.
The benchmark keeps ALL S1 of sampled regions, ALL S2/S3 mapped to a sampled region, plus
ALL S2/S3 without any region (no geographic evidence, e.g. empty address), which keeps
their true pairs and acts as extra distractors (slightly harder than test).

    python -m src.benchmark --stage build     # cache/bench/{s1,queries}.parquet + logs/bench_summary.json
    python -m src.benchmark --stage block     # pass A + pass C restricted to the benchmark -> cache/bench/union.parquet

IDs are the global blocking ids (s1_id = row in norm train_s1; query_id = source * 10**8 + row).
"""
import argparse
import gc
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .blocking import (A_COLS, C_COLS, QUERY_ID_MULT, QUERY_SOURCES, build_s1_keys, c_texts, chunk_slices,
                       fit_c_index, pass_a_chunk, pass_c_chunk, token_df, union_frame)
from .config import CACHE_DIR, LOG_DIR, add_path_args, set_seeds
from .geo_units import Gazetteer, N_FOLDS, split_components, unit_hash
from .io_utils import raw_parquet_path, read_parquet, write_parquet
from .logging_utils import StageTimer, get_logger
from .normalize import norm_path

BENCH_DIR = CACHE_DIR / "bench"
SAMPLE_SHARE = 0.20
MAP_MIN_COUNT = 5
MAP_MIN_PURITY = 0.9


def address_keys(addr_clean: str) -> set:
    """Return the text components and their tokens of a normalised address (region-mapping evidence)."""
    words = split_components(addr_clean)[0]
    return set(words) | {t for w in words for t in w.split()}


def s1_region(gaz: Gazetteer, addr_clean: str) -> str:
    """Return the first region key of an S1 address ('' if none)."""
    for w in split_components(addr_clean)[0]:
        k = gaz._key(w)
        if k in gaz.regions:
            return k
    return ""


def learn_region_map(q_addr: list, regions: np.ndarray) -> dict:
    """Return {address key: (region, votes)} learned from S2/S3 addresses of true pairs and their S1 region."""
    co = defaultdict(Counter)
    for a, r in zip(q_addr, regions):
        if r:
            for k in address_keys(a):
                co[k][r] += 1
    out = {}
    for k, cnt in co.items():
        r, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if tot >= MAP_MIN_COUNT and n / tot >= MAP_MIN_PURITY:
            out[k] = (r, n)
    return out


def query_region(mapping: dict, addr_clean: str) -> str:
    """Return the region voted by the address keys ('' if no key is mapped)."""
    votes = Counter()
    for k in address_keys(addr_clean):
        if k in mapping:
            votes[mapping[k][0]] += mapping[k][1]
    return votes.most_common(1)[0][0] if votes else ""


def sample_regions(s1: pd.DataFrame) -> dict:
    """Return {(country, region): fold} for the sampled regions (hash order until SAMPLE_SHARE of S1 rows)."""
    chosen = {}
    for country, g in s1.groupby("country"):
        sizes = g.groupby("region").size()
        order = sorted(sizes.index, key=lambda r: unit_hash(country, r))
        target, acc = SAMPLE_SHARE * len(g), 0
        for i, r in enumerate(order):
            if acc >= target:
                break
            chosen[(country, r)] = len([k for k in chosen if k[0] == country]) % N_FOLDS
            acc += sizes[r]
    return chosen


def build(logger) -> dict:
    """Assign regions, sample the benchmark and write cache/bench/{s1,queries}.parquet + logs/bench_summary.json."""
    s1 = pq.read_table(norm_path("train", 1), columns=["entity_id", "country", "addr_clean"]).to_pandas()
    s1.insert(0, "s1_id", np.arange(len(s1), dtype=np.int64))
    gaz = {c: Gazetteer(g["addr_clean"].tolist()) for c, g in s1.groupby("country")}
    s1["region"] = [s1_region(gaz[c], a) for c, a in zip(s1["country"], s1["addr_clean"])]
    s1 = s1.drop(columns="addr_clean")
    pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
    owner = dict(zip(pairs["other_id"], pairs["s1_id"]))
    s1_row = pd.Series(s1["s1_id"].to_numpy(), index=s1["entity_id"])
    s1_reg = s1["region"].to_numpy()
    qs = []
    for source in QUERY_SOURCES:
        t = pq.read_table(norm_path("train", source), columns=["entity_id", "country", "addr_clean"]).to_pandas()
        t.insert(0, "query_id", source * QUERY_ID_MULT + np.arange(len(t), dtype=np.int64))
        own = t["entity_id"].map(owner)
        t["true_s1"] = s1_row.reindex(own).fillna(-1).to_numpy().astype(np.int64)
        qs.append(t)
    q = pd.concat(qs, ignore_index=True)
    del qs, owner
    gc.collect()
    matched = q["true_s1"].to_numpy() >= 0
    mapping = {}
    for country in q["country"].unique():
        m = (q["country"] == country).to_numpy() & matched
        mapping[country] = learn_region_map(q.loc[m, "addr_clean"].tolist(), s1_reg[q.loc[m, "true_s1"].to_numpy()])
    q["region"] = [query_region(mapping.get(c, {}), a) for c, a in zip(q["country"], q["addr_clean"])]
    q = q.drop(columns="addr_clean")
    folds = sample_regions(s1)
    s1["fold"] = [folds.get((c, r), -1) for c, r in zip(s1["country"], s1["region"])]
    q["fold"] = [folds.get((c, r), -1) for c, r in zip(q["country"], q["region"])]
    no_region = (q["region"] == "").to_numpy()
    in_bench_q = (q["fold"].to_numpy() >= 0) | no_region
    bs1 = s1[s1["fold"] >= 0].reset_index(drop=True)
    bq = q[in_bench_q].reset_index(drop=True)
    bq["true_s1_in_bench"] = np.isin(bq["true_s1"].to_numpy(), bs1["s1_id"].to_numpy())
    write_parquet(bs1, BENCH_DIR / "s1.parquet")
    write_parquet(bq, BENCH_DIR / "queries.parquet")

    # realism report
    bench_ids = set(bs1["s1_id"])
    true_of_bench = q[np.isin(q["true_s1"].to_numpy(), list(bench_ids))]
    summary = {"regions_sampled": {c: sorted(r for (cc, r) in folds if cc == c) for c in s1["country"].unique()},
               "fold_of_region": {f"{c}|{r}": f for (c, r), f in folds.items()},
               "s1_total": len(s1), "s1_bench": len(bs1), "queries_bench": len(bq),
               "share_s1_sampled": {c: round(float((g["fold"] >= 0).mean()), 4) for c, g in s1.groupby("country")},
               "queries_without_region_pct": round(100 * float(no_region.mean()), 2),
               "true_pairs_of_bench_s1": len(true_of_bench),
               "true_pairs_split_pct": round(100 * float(1 - np.isin(true_of_bench["query_id"], bq["query_id"]).mean()), 3),
               "true_pairs_wrong_region_pct": round(100 * float(((true_of_bench["region"] != "") & (true_of_bench["fold"] < 0)).mean()), 3),
               "by_country": {}, "by_fold": {}}
    for c in bs1["country"].unique():
        qc, sc = bq[bq["country"] == c], bs1[bs1["country"] == c]
        summary["by_country"][c] = {
            "s1": len(sc), "queries": len(qc), "queries_per_s1": round(len(qc) / len(sc), 3),
            "pct_queries_unmatched_in_bench": round(100 * float((~qc["true_s1_in_bench"]).mean()), 2),
            "pct_queries_unmatched_globally": round(100 * float((qc["true_s1"] < 0).mean()), 2)}
    for f in range(N_FOLDS):
        summary["by_fold"][f] = {"s1": int((bs1["fold"] == f).sum()),
                                 "regions": sorted(f"{c}|{r}" for (c, r), ff in folds.items() if ff == f)}
    summary["full_train_reference"] = {"queries_per_s1": round(len(q) / len(s1), 3),
                                       "pct_s2s3_unmatched": round(100 * float((~matched).mean()), 2),
                                       "test_queries_per_s1": 5.754}
    with open(LOG_DIR / "bench_summary.json", "w", encoding="utf-8", newline="") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("benchmark: %s", json.dumps({k: v for k, v in summary.items()
                                            if k not in ("fold_of_region", "by_fold")}, indent=1))
    return summary


def block(logger) -> None:
    """Run pass A + pass C with the benchmark S1 as the index and benchmark queries; write cache/bench/union.parquet."""
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet", columns=["s1_id", "country"])
    bq = pd.read_parquet(BENCH_DIR / "queries.parquet", columns=["query_id", "country"])
    parts = []
    for country, sc in bs1.groupby("country", sort=True):
        s1_pos = np.sort(sc["s1_id"].to_numpy())
        s1 = pq.read_table(norm_path("train", 1), columns=A_COLS + C_COLS).take(pa.array(s1_pos))
        df = token_df(s1["addr_tokens"].to_pylist())
        s1_keys = build_s1_keys(s1["num_keys"].to_pylist(), s1["addr_tokens"].to_pylist(), s1_pos.astype(np.int32), df)
        vec, b = fit_c_index(c_texts(s1))
        del s1
        qc = bq.loc[bq["country"] == country, "query_id"].to_numpy()
        logger.info("[bench %s] index %d S1, %d queries", country, len(s1_pos), len(qc))
        with StageTimer(f"bench_block_{country}", logger, s1=len(s1_pos), queries=len(qc)):
            for source in QUERY_SOURCES:
                qs = np.sort(qc[qc // QUERY_ID_MULT == source])
                t = pq.read_table(norm_path("train", source), columns=A_COLS + C_COLS)
                for i, sl in chunk_slices(len(qs)):
                    out = BENCH_DIR / "cand" / f"union_{country}_s{source}_part_{i:04d}.parquet"
                    if not out.exists():
                        sub = t.take(pa.array(qs[sl] % QUERY_ID_MULT))
                        a = pass_a_chunk(qs[sl], sub["num_keys"].to_pylist(), sub["addr_tokens"].to_pylist(), s1_keys, df)
                        c = pass_c_chunk(vec, b, qs[sl], c_texts(sub), s1_pos)
                        write_parquet(union_frame(a, c), out)
                    parts.append(out)
                del t
                gc.collect()
        del vec, b, s1_keys, df
        gc.collect()
    union = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    write_parquet(union, BENCH_DIR / "union.parquet")
    logger.info("bench union: %d pairs, %d queries", len(union), union["query_id"].nunique())


def main() -> None:
    """Parse flags and run the requested stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["build", "block"], required=True)
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("benchmark")
    with StageTimer(f"benchmark_{args.stage}", logger):
        build(logger) if args.stage == "build" else block(logger)


if __name__ == "__main__":
    main()
