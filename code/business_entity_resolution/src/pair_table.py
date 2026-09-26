"""Labelled / unlabelled pair-feature tables: stage-1 top-K candidates + baseline, decoy-aware and context features.

    python -m src.pair_table --split bench     # cache/bench/union.parquet -> cache/bench/pairs.parquet (with label, fold)
    python -m src.pair_table --split test      # cache/cand/test_union.parquet -> <output>/features/test_pairs.parquet

Feature computation is chunked and spread over a process pool (N_WORKERS); the per-pair Python loops
(numbers, extra name tokens) dominate. Stage 1 = finalize.top_k (cosine_C + W_A x shared_keys_A).
"""
import argparse
import functools
import gc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .blocking import CAND_DIR, QUERY_ID_MULT, QUERY_SOURCES
from .config import OUTPUT_DIR, add_path_args, cpu_count, env_int, set_seeds
from .decoy_features import context_features, main_number, pair_features
from .finalize import TOP_K, load_union, num_match, top_k
from .io_utils import write_parquet
from .logging_utils import StageTimer, get_logger
from .normalize import norm_path

W_A = 0.2                      # stage-1 weight tuned on validation recall@5 (logs/stage1_reduction.json)
COLS = ["name_core", "name_compact", "addr_clean", "numbers", "num_keys"]
CHUNK = 250_000


@functools.lru_cache(maxsize=3)
def _table(split: str, source: int) -> pa.Table:
    """Return (cached) the needed normalised columns of one file."""
    return pq.read_table(norm_path(split, source), columns=COLS)


def texts(split: str, source: int, rows: np.ndarray) -> dict:
    """Return {col: object array} for row indices of a normalised file."""
    uniq, inv = np.unique(rows, return_inverse=True)
    t = _table(split, source).take(pa.array(uniq))
    return {c: np.asarray(t[c].to_pylist(), dtype=object)[inv] for c in COLS}


def chunk_features(args: tuple) -> dict:
    """Worker: compute baseline + decoy features for one chunk of aligned query / S1 texts."""
    qt, st = args
    from rapidfuzz import fuzz, process
    f = pair_features(qt, st)
    name = np.maximum(process.cpdist(qt["name_core"], st["name_core"], scorer=fuzz.token_set_ratio),
                      process.cpdist(qt["name_compact"], st["name_compact"], scorer=fuzz.ratio))
    addr = process.cpdist(qt["addr_clean"], st["addr_clean"], scorer=fuzz.token_set_ratio)
    addr = np.where((qt["addr_clean"] == "") | (st["addr_clean"] == ""), 0.0, addr)
    f["b_name_sim"] = (name / 100).astype(np.float32)
    f["b_addr_sim"] = (addr / 100).astype(np.float32)
    f["b_num_match"] = num_match(qt["num_keys"], st["num_keys"])
    return f


def build_table(split: str, union: pd.DataFrame, logger, workers: int) -> pd.DataFrame:
    """Return the stage-1 top-K pair table with all features (and main query numbers for context)."""
    red = top_k(union, W_A)
    del union
    gc.collect()
    red = red.sort_values(["query_id", "s1_rank"], kind="stable").reset_index(drop=True)
    qid = red["query_id"].to_numpy()
    src = qid // QUERY_ID_MULT
    feats, main_q = {}, np.empty(len(red), dtype=object)
    text_split = "train" if split == "bench" else split
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for source in QUERY_SOURCES:
            idx = np.flatnonzero(src == source)
            jobs, slices = [], []
            for start in range(0, len(idx), CHUNK):
                sl = idx[start:start + CHUNK]
                qt = texts(text_split, source, qid[sl] % QUERY_ID_MULT)
                st = texts(text_split, 1, red["s1_id"].to_numpy()[sl])
                main_q[sl] = [main_number(x) for x in qt["numbers"]]
                jobs.append(pool.submit(chunk_features, (qt, st)))
                slices.append(sl)
                if len(jobs) >= 2 * workers:          # bound memory: drain before submitting more
                    _collect(jobs, slices, feats, len(red))
                    jobs, slices = [], []
            _collect(jobs, slices, feats, len(red))
            logger.info("[%s] features: source %d done (%d pairs)", split, source, len(idx))
    for k, v in feats.items():
        red[k] = v
    return context_features(red, main_q)


def _collect(jobs: list, slices: list, feats: dict, n: int) -> None:
    """Gather finished worker results into the full-length feature arrays."""
    for job, sl in zip(jobs, slices):
        for k, v in job.result().items():
            if k not in feats:
                feats[k] = np.full(n, np.nan, dtype=np.float32)
            feats[k][sl] = v


def label_bench(red: pd.DataFrame) -> pd.DataFrame:
    """Add label (query's true S1 == s1_id), S1 fold, query truth status to a benchmark pair table."""
    from .benchmark import BENCH_DIR
    bq = pd.read_parquet(BENCH_DIR / "queries.parquet", columns=["query_id", "true_s1"])
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet", columns=["s1_id", "fold"])
    true_s1 = red["query_id"].map(pd.Series(bq["true_s1"].to_numpy(), index=bq["query_id"])).to_numpy()
    red["label"] = (true_s1 == red["s1_id"].to_numpy()).astype(np.int8)
    red["q_true_s1"] = true_s1.astype(np.int64)
    red["fold"] = red["s1_id"].map(pd.Series(bs1["fold"].to_numpy(), index=bs1["s1_id"])).astype(np.int8).to_numpy()
    return red


def main() -> None:
    """Build the pair table for the benchmark or test candidates."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["bench", "test"], required=True)
    ap.add_argument("--workers", type=int, default=env_int("N_WORKERS", cpu_count()))
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("pair_table")
    with StageTimer(f"pair_table_{args.split}", logger, workers=args.workers):
        if args.split == "bench":
            from .benchmark import BENCH_DIR
            union = pd.read_parquet(BENCH_DIR / "union.parquet", columns=["query_id", "s1_id", "scoreA", "scoreC"])
            union["scoreA"] = union["scoreA"].fillna(0).astype(np.float32)
            union["scoreC"] = union["scoreC"].fillna(0).astype(np.float32)
            red = label_bench(build_table("bench", union, logger, args.workers))
            out = BENCH_DIR / "pairs.parquet"
        else:
            red = build_table("test", load_union("test"), logger, args.workers)
            out = OUTPUT_DIR / "features" / "test_pairs.parquet"
        write_parquet(red, out)
        logger.info("wrote %s: %d pairs x %d columns (top-%d, W_A=%s)", out, len(red), red.shape[1], TOP_K, W_A)


if __name__ == "__main__":
    main()
