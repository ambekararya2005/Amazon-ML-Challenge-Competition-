"""v3 pair features on top of an existing pair table: tagged address numbers, extra-word lists (for the target
encoding done at model time), group / odd-one-out features within each S1's claimants and each query's S1s.

    python -m src.features_v3 --split bench   # cache/bench/pairs.parquet -> <output>/features_v3/bench/pairs_<country>.parquet
    python -m src.features_v3 --split test    # <output>/features/test_pairs.parquet -> <output>/features_v3/test/pairs_<country>.parquet

Work is done one country at a time (queries and S1 of different countries are never paired, so every group
statistic is exact). Country is only used to partition. Per-pair Python loops run in a process pool.

Group features (all pairs of an S1 = its candidate claimants; "claimants" = pairs with base_score >= CLAIM_MIN):
    g_rank_v2 / g_rank_addr   rank of the pair within its S1 by the v2 rule score / by addr_tsort (1 = best)
    g_gap_v2, g_other_max_v2  S1 group best v2 minus this pair's v2; best v2 among the OTHER queries of the S1
    g_size, g_n_noconf, g_n_noextra   claimants; claimants with num_conflict = 0; claimants with no extra words
    g_same_src, g_rank_src    other claimants from the query's source; rank by v2 within the same source
    q_n_cand, q_n_compat      the query's S1 candidates; those with a compatible street number
    q_v2_best, q_v2_gap, q_v2_margin   the query's best v2, best minus this pair, best minus second best
Twin features (pairs with base_score >= FOCUS_MIN, against the other top-TWIN_TOPK claimants q' of the same S1):
    tw_name_max, tw_addr_max, tw_combo_max   max name / address / mean similarity q <-> q' (a near-twin?)
    for the most similar q': tw_num_better (q' agrees with the S1 main street number, q conflicts), tw_num_worse,
    tw_extra_better (q' has no extra words, q has), tw_extra_worse, tw_v2_diff (v2 of q' minus v2 of q), tw_same_src
"""
import argparse
import gc
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .blocking import CAND_DIR, QUERY_ID_MULT, country_slug
from .config import FEATURE_VARIANT, LOG_DIR, OUTPUT_DIR, add_path_args, cpu_count, env_int, set_seeds
from .decoy_features import extra_token_lists, learn_postal_lengths, tag_numbers, tagged_number_features
from .io_utils import write_parquet
from .logging_utils import StageTimer, get_logger
from .normalize import norm_path
from .scorer_v2 import CONFIG_FILE, find_file, score_v2

V3_DIR = OUTPUT_DIR / ("features_v3" if FEATURE_VARIANT == "v1" else f"features_v3_{FEATURE_VARIANT}")   # kernel output
CLAIM_MIN = 0.6
FOCUS_MIN = 0.45
TWIN_TOPK = 8               # twins are searched among the S1's top-8 claimants by base_score (hub S1 have 600+)
CHUNK = 200_000
TWIN_CHUNK = 3_000_000
TEXT_COLS = ["name_core", "addr_clean"]


# ------------------------------------------------------------------ texts
def unique_texts(split: str, ids: np.ndarray, is_query: bool) -> dict:
    """Return {col: object array} aligned with the sorted unique ``ids`` (query ids or S1 rows)."""
    out = {c: np.empty(len(ids), dtype=object) for c in TEXT_COLS}
    if not is_query:
        t = pq.read_table(norm_path(split, 1), columns=TEXT_COLS)
        for c in TEXT_COLS:
            out[c][:] = t[c].take(ids).to_numpy(zero_copy_only=False)
        return out
    src = ids // QUERY_ID_MULT
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        t = pq.read_table(norm_path(split, int(s)), columns=TEXT_COLS)
        for c in TEXT_COLS:
            out[c][idx] = t[c].take(ids[idx] % QUERY_ID_MULT).to_numpy(zero_copy_only=False)
        del t
    return out


# ------------------------------------------------------------------ workers
def _tag_worker(args: tuple) -> list:
    """Worker: tag_numbers for a list of addresses."""
    addrs, postal = args
    return [tag_numbers(a, postal) for a in addrs]


def _pair_worker(args: tuple) -> dict:
    """Worker: tagged-number features and extra-word lists for one chunk of aligned pairs."""
    q_tags, s_tags, q_core, s_core = args
    f = tagged_number_features(q_tags, s_tags)
    f["xq"], f["xs"] = extra_token_lists(q_core, s_core)
    return f


def pooled_map(pool, fn, jobs):
    """Yield results of ``fn`` over an iterable of jobs, in order, with at most 8 jobs in flight."""
    pending = []
    for j in jobs:
        pending.append(pool.submit(fn, j))
        if len(pending) >= 8:
            yield from (p.result() for p in pending)
            pending = []
    yield from (p.result() for p in pending)


# ------------------------------------------------------------------ group helpers
def group_rank(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Return the 1-based rank of each row within its ``key`` group by descending ``score`` (ties: row order)."""
    order = np.lexsort((np.arange(len(key)), -score, key))
    k = key[order]
    start = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    rank = np.empty(len(key), dtype=np.float32)
    rank[order] = np.arange(len(key)) - np.repeat(start, np.diff(np.r_[start, len(key)])) + 1
    return rank


def group_top2(key: np.ndarray, score: np.ndarray) -> tuple:
    """Return (group max, group second max; -inf if none) broadcast to every row."""
    s = pd.Series(score)
    mx = s.groupby(key).transform("max").to_numpy()
    r = group_rank(key, score)
    second = pd.Series(np.where(r == 2, score, -np.inf)).groupby(key).transform("max").to_numpy()
    return mx, second


def group_features(df: pd.DataFrame, v2: np.ndarray) -> None:
    """Add the vectorised group / query context columns to ``df`` in place."""
    s1, qid = df["s1_id"].to_numpy(), df["query_id"].to_numpy()
    src = qid // QUERY_ID_MULT
    addr = np.nan_to_num(df["addr_tsort"].to_numpy(), nan=-1.0)
    df["v2_score"] = v2.astype(np.float32)
    df["g_rank_v2"] = group_rank(s1, v2)
    df["g_rank_addr"] = group_rank(s1, addr)
    mx, second = group_top2(s1, v2)
    df["g_gap_v2"] = (mx - v2).astype(np.float32)
    other = np.where(df["g_rank_v2"].to_numpy() == 1, second, mx)
    df["g_other_max_v2"] = np.where(np.isfinite(other), other, -1.0).astype(np.float32)
    claim = df["base_score"].to_numpy() >= CLAIM_MIN
    df["is_claimant"] = claim.astype(np.float32)
    extra0 = (df["extra_tokens_q"].to_numpy() + df["extra_tokens_s1"].to_numpy()) == 0

    def cnt(mask):
        """Return, per row, the number of rows of its S1 group satisfying ``mask``."""
        return pd.Series(mask.astype(np.int32)).groupby(s1).transform("sum").to_numpy().astype(np.float32)
    df["g_size"] = cnt(claim)
    df["g_n_noconf"] = cnt(claim & (df["num_conflict"].to_numpy() == 0))
    df["g_n_noextra"] = cnt(claim & extra0)
    key_src = s1.astype(np.int64) * 4 + src
    same = pd.Series(claim.astype(np.int32)).groupby(key_src).transform("sum").to_numpy()
    df["g_same_src"] = (same - claim).astype(np.float32)
    df["g_rank_src"] = group_rank(key_src, v2)
    df["q_n_cand"] = pd.Series(np.ones(len(df), np.int32)).groupby(qid).transform("sum").to_numpy().astype(np.float32)
    comp = (df["st_compat"].to_numpy() > 0).astype(np.int32)
    df["q_n_compat"] = pd.Series(comp).groupby(qid).transform("sum").to_numpy().astype(np.float32)
    qmx, qsec = group_top2(qid, v2)
    df["q_v2_best"] = qmx.astype(np.float32)
    df["q_v2_gap"] = (qmx - v2).astype(np.float32)
    df["q_v2_margin"] = np.where(np.isfinite(qsec), qmx - qsec, 2.0).astype(np.float32)


def twin_features(df: pd.DataFrame, v2: np.ndarray, qtext: dict, uq: np.ndarray, logger) -> None:
    """Add tw_* columns: for focus pairs, compare the query with the other claimants of the same S1."""
    n = len(df)
    names = ("tw_name_max", "tw_addr_max", "tw_combo_max", "tw_num_better", "tw_num_worse", "tw_extra_better",
             "tw_extra_worse", "tw_v2_diff", "tw_same_src")
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in names}
    s1, qid = df["s1_id"].to_numpy(), df["query_id"].to_numpy()
    base = df["base_score"].to_numpy()
    claim = np.flatnonzero((base >= CLAIM_MIN) & (group_rank(s1, base) <= TWIN_TOPK))
    claim = claim[np.argsort(s1[claim], kind="stable")]
    c_s1 = s1[claim]
    focus = np.flatnonzero(base >= FOCUS_MIN)
    lo = np.searchsorted(c_s1, s1[focus], "left")
    hi = np.searchsorted(c_s1, s1[focus], "right")
    has = hi - lo > 0
    focus, lo, hi = focus[has], lo[has], hi[has]
    qpos = np.searchsorted(uq, qid)                       # row -> index into the unique-query texts
    compat = df["st_main_compat"].to_numpy()
    conflict = df["st_main_conflict"].to_numpy()
    extra = df["extra_tokens_q"].to_numpy() + df["extra_tokens_s1"].to_numpy()
    src = qid // QUERY_ID_MULT
    sizes = hi - lo
    ends = np.cumsum(sizes)
    total = int(ends[-1]) if len(ends) else 0
    logger.info("twin: %d focus pairs, %d claimants, %d combos", len(focus), len(claim), total)
    start_f = 0
    while start_f < len(focus):
        done = ends[start_f - 1] if start_f else 0
        stop_f = int(np.searchsorted(ends, done + TWIN_CHUNK, "right"))
        stop_f = max(stop_f, start_f + 1)
        f_idx = np.arange(start_f, stop_f)
        rep = sizes[f_idx]
        a = np.repeat(focus[f_idx], rep)                                  # focus pair row
        off = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep)
        b = claim[np.repeat(lo[f_idx], rep) + off]                        # claimant pair row (same S1)
        keep = qid[a] != qid[b]
        a, b = a[keep], b[keep]
        if len(a):
            qa, qb = qpos[a], qpos[b]
            ns = process.cpdist(qtext["name_core"][qa], qtext["name_core"][qb], scorer=fuzz.token_sort_ratio,
                                workers=-1).astype(np.float32) / 100
            as_ = process.cpdist(qtext["addr_clean"][qa], qtext["addr_clean"][qb], scorer=fuzz.token_sort_ratio,
                                 workers=-1).astype(np.float32) / 100
            combo = (ns + as_) / 2
            g = pd.DataFrame({"a": a, "ns": ns, "as": as_, "combo": combo})
            mx = g.groupby("a")[["ns", "as", "combo"]].max()
            idx = mx.index.to_numpy()
            out["tw_name_max"][idx] = mx["ns"].to_numpy()
            out["tw_addr_max"][idx] = mx["as"].to_numpy()
            out["tw_combo_max"][idx] = mx["combo"].to_numpy()
            order = np.lexsort((-combo, a))
            first = order[np.r_[True, a[order][1:] != a[order][:-1]]]
            ta, tb = a[first], b[first]
            out["tw_num_better"][ta] = (compat[tb] == 1) & (conflict[ta] == 1)
            out["tw_num_worse"][ta] = (compat[ta] == 1) & (conflict[tb] == 1)
            out["tw_extra_better"][ta] = (extra[tb] == 0) & (extra[ta] > 0)
            out["tw_extra_worse"][ta] = (extra[ta] == 0) & (extra[tb] > 0)
            out["tw_v2_diff"][ta] = v2[tb] - v2[ta]
            out["tw_same_src"][ta] = src[tb] == src[ta]
        start_f = stop_f
    for k, v in out.items():
        df[k] = v


# ------------------------------------------------------------------ one partition
def build_partition(df: pd.DataFrame, split: str, postal: frozenset, cfg: dict, pool, logger) -> pd.DataFrame:
    """Return ``df`` (one country's pairs) with all v3 columns added."""
    df = df.reset_index(drop=True)
    uq, qinv = np.unique(df["query_id"].to_numpy(), return_inverse=True)
    us, sinv = np.unique(df["s1_id"].to_numpy(), return_inverse=True)
    qtext, stext = unique_texts(split, uq, True), unique_texts(split, us, False)
    logger.info("partition: %d pairs, %d queries, %d S1", len(df), len(uq), len(us))

    def tags(addrs: np.ndarray) -> np.ndarray:
        """Return tag_numbers tuples for an address array (process pool)."""
        jobs = ((list(addrs[i:i + CHUNK]), postal) for i in range(0, len(addrs), CHUNK))
        res = np.empty(len(addrs), dtype=object)
        res[:] = [t for part in pooled_map(pool, _tag_worker, jobs) for t in part]
        return res
    qtags, stags = tags(qtext["addr_clean"]), tags(stext["addr_clean"])
    logger.info("tagged numbers: queries with street / floor / postal = %.1f / %.1f / %.2f %%",
                *(100 * np.mean([bool(t[i]) for t in qtags]) for i in range(3)))
    bounds = [slice(i, i + CHUNK) for i in range(0, len(df), CHUNK)]
    jobs = ((list(qtags[qinv[sl]]), list(stags[sinv[sl]]), list(qtext["name_core"][qinv[sl]]),
             list(stext["name_core"][sinv[sl]])) for sl in bounds)
    feats, strings = {}, {}
    for sl, res in zip(bounds, pooled_map(pool, _pair_worker, jobs)):
        for k, v in res.items():
            if v.dtype == object:                                   # extra-word lists -> Arrow strings per chunk
                strings.setdefault(k, []).append(pa.array(list(v), type=pa.string()))
            else:
                feats.setdefault(k, np.zeros(len(df), np.float32))[sl] = v
    del qtags, stags, sinv
    for k, v in feats.items():
        df[k] = v
    for k, chunks in strings.items():
        df[k] = pd.Series(pd.arrays.ArrowExtensionArray(pa.chunked_array(chunks, type=pa.string())))
    del feats, strings
    gc.collect()
    v2 = score_v2(df, cfg).astype(np.float32)
    group_features(df, v2)
    twin_features(df, v2, qtext, uq, logger)
    return df


# ------------------------------------------------------------------ inputs
def bench_partitions():
    """Yield (country, pair frame) for the benchmark pair table."""
    from .benchmark import BENCH_DIR
    from .blocking_v4 import V4_BENCH_DIR
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet", columns=["s1_id", "country"])
    pairs = pd.read_parquet((V4_BENCH_DIR if FEATURE_VARIANT == "v4" else BENCH_DIR) / "pairs.parquet")
    country = pairs["s1_id"].map(pd.Series(bs1["country"].to_numpy(), index=bs1["s1_id"])).to_numpy()
    for c in sorted(bs1["country"].unique()):
        yield c, pairs[country == c].reset_index(drop=True)


def test_partitions(logger):
    """Yield (country, pair frame) for the test pair table, reading it row group by row group (v4: one file per
    country from pair_table)."""
    if FEATURE_VARIANT == "v4":
        from pathlib import Path
        d = OUTPUT_DIR / "features_v4" / "test"
        files = sorted(d.glob("pairs_*.parquet")) or sorted(
            p for p in Path("/kaggle/input").rglob("pairs_*.parquet") if p.parent.parent.name == "features_v4")
        s1c = pq.read_table(norm_path("test", 1), columns=["country"])["country"].to_numpy(zero_copy_only=False)
        for f in files:
            df = pd.read_parquet(f)
            yield s1c[int(df["s1_id"].iat[0])], df
        return
    lk = pd.read_parquet(find_file("lookup_s1.parquet", CAND_DIR / "test", "/kaggle/input"),
                         columns=["country"])
    countries = sorted(lk["country"].unique())
    code = pd.Categorical(lk["country"], categories=countries).codes
    path = find_file("test_pairs.parquet", OUTPUT_DIR / "features", "/kaggle/input")
    logger.info("test pairs from %s", path)
    f = pq.ParquetFile(path)
    for ci, c in enumerate(countries):
        parts = []
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg).to_pandas()
            parts.append(t[code[t["s1_id"].to_numpy()] == ci])
        yield c, pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()


def main() -> None:
    """Build the v3 pair tables for the benchmark or test, one country at a time."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["bench", "test"], required=True)
    ap.add_argument("--workers", type=int, default=env_int("N_WORKERS", cpu_count()))
    ap.add_argument("--force", action="store_true")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("features_v3")
    split_text = "train" if args.split == "bench" else "test"
    cfg = json.loads(find_file(CONFIG_FILE, LOG_DIR, "/kaggle/input").read_text(encoding="utf-8"))
    out_dir = V3_DIR / args.split
    with StageTimer(f"features_v3_{args.split}", logger, workers=args.workers):
        s1_addr = pq.read_table(norm_path(split_text, 1), columns=["addr_clean"])["addr_clean"].to_pylist()
        postal = learn_postal_lengths(s1_addr)
        del s1_addr
        logger.info("postal-like digit lengths learned from S1 addresses: %s", sorted(postal) or "none")
        summary = {"postal_lengths": sorted(postal), "partitions": {}}
        parts = bench_partitions() if args.split == "bench" else test_partitions(logger)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for country, df in parts:
                out = out_dir / f"pairs_{country_slug(country)}.parquet"
                if out.exists() and not args.force:
                    logger.info("[%s] cached: %s", country, out)
                    continue
                with StageTimer(f"features_v3_{args.split}_{country}", logger, pairs=len(df)):
                    df = build_partition(df, split_text, postal, cfg, pool, logger)
                    write_parquet(df, out)
                summary["partitions"][country] = {
                    "pairs": int(len(df)), "columns": int(df.shape[1]),
                    "pct_twin_defined": round(100 * float(df["tw_combo_max"].notna().mean()), 2),
                    "pct_floor_conflict": round(100 * float(df["floor_conflict"].mean()), 3),
                    "pct_postal_conflict": round(100 * float(df["postal_conflict"].mean()), 3),
                    "pct_st_conflict_gt0": round(100 * float((df["st_conflict"] > 0).mean()), 2)}
                logger.info("[%s] wrote %s: %s", country, out, summary["partitions"][country])
                del df
                gc.collect()
        with open(LOG_DIR / f"{V3_DIR.name}_{args.split}.json", "w", encoding="utf-8", newline="") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
