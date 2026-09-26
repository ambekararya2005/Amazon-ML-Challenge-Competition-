"""Pass-C speed/recall tuning on a validation sample (levers L1-L3; L4 = SVD+HNSW only if needed).

Per country: the index is ALL train S1 of that country; the queries are N_VAL S2/S3 records matched
to validation S1 entities plus N_RAND random other S2/S3 records (seed 42). Every configuration is
scored on the same sample:
    base      : current pass C (name_compact + name_core + addr_clean, char_wb (3,4), no max_df)
    L1        : name-only text (name_compact + name_core)
    L1+L2     : name-only, max_df in MAX_DFS (fraction of the country's S1), ngram (3,4) or (4,4)
    L1+L2+L3  : as L1+L2, but pass C only for residual queries (see gate_mask)
Each (text, ngram) TF-IDF is fitted once with norm=None; a max_df variant drops the columns whose
S1 document frequency exceeds the threshold and re-applies the L2 norm (identical to refitting with
max_df because a kept term's idf depends only on its own df).

Local timings are converted to Kaggle time with a per-country factor = (Kaggle v1 base pass-C
seconds/query) / (local base pass-C seconds/query), then projected to the full test run using
the test S1 and query counts per country (France at India's rates).

    python -m src.tune_pass_c --kaggle-bench ../../kaggle/runs/20260926-v1/results/blocking_bench.json
"""
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .blocking import (QUERY_SOURCES, build_s1_keys, country_positions, pass_a_chunk, read_country,
                       token_df)
from .config import LOG_DIR, SEED, add_path_args, set_seeds
from .io_utils import raw_parquet_path, read_parquet
from .logging_utils import StageTimer, get_logger, rss_mb
from .normalize import norm_path
from .split import load_split

N_VAL, N_RAND = 20_000, 20_000
TOP_N = 10
MAX_DFS = (0.002, 0.005, 0.01)
NGRAMS = ((3, 4), (4, 4))
GATE_MIN_A, GATE_MIN_RATIO = 3, 60
S1_COLS = ["entity_id", "name_compact", "name_core", "addr_clean", "num_keys", "addr_tokens"]
Q_COLS = S1_COLS + ["numbers", "addr_empty"]
BASE_PARAMS = dict(analyzer="char_wb", min_df=2, sublinear_tf=True, dtype=np.float32, max_features=600_000)
TARGET_TEST_MIN = 180


def texts(t: pd.DataFrame, kind: str) -> list:
    """Return pass-C texts: 'full' = compact + core + address (current), 'name' = compact + core."""
    if kind == "full":
        return (t["name_compact"] + " " + t["name_core"] + " " + t["addr_clean"]).tolist()
    return (t["name_compact"] + " " + t["name_core"]).tolist()


def sample_queries(country: str, s1_row_of: dict, val_owner: dict, rng) -> pd.DataFrame:
    """Return N_VAL validation-matched + N_RAND other S2/S3 queries of ``country`` with their true S1 row."""
    frames = []
    for s in QUERY_SOURCES:
        t = read_country(norm_path("train", s), country, Q_COLS).to_pandas()
        frames.append(t)
    q = pd.concat(frames, ignore_index=True)
    owner = q["entity_id"].map(val_owner)
    is_val = owner.notna().to_numpy()
    val_idx = rng.choice(np.flatnonzero(is_val), size=min(N_VAL, int(is_val.sum())), replace=False)
    oth_idx = rng.choice(np.flatnonzero(~is_val), size=min(N_RAND, int((~is_val).sum())), replace=False)
    q = q.iloc[np.sort(np.concatenate([val_idx, oth_idx]))].reset_index(drop=True)
    q["true_s1"] = q["entity_id"].map(val_owner).map(s1_row_of).fillna(-1).astype(np.int64)
    return q


def recall(cand: pd.DataFrame, q: pd.DataFrame, rows: np.ndarray = None) -> float:
    """Return the fraction of validation queries (optionally only ``rows``) whose true S1 row is in ``cand``."""
    truth = q["true_s1"].to_numpy()
    hit = truth[cand["query_id"].to_numpy()] == cand["s1_id"].to_numpy()
    n_val = int((truth >= 0).sum()) if rows is None else int((truth[rows] >= 0).sum())
    return len(np.unique(cand["query_id"].to_numpy()[hit])) / n_val


def gate_mask(q: pd.DataFrame, a: pd.DataFrame, s1_compact: np.ndarray) -> np.ndarray:
    """Return True for residual queries needing pass C.

    Residual = fewer than GATE_MIN_A pass-A candidates, OR no address numbers, OR empty address,
    OR no pass-A candidate with rapidfuzz ratio(name_compact) >= GATE_MIN_RATIO.
    """
    n_a = np.bincount(a["query_id"].to_numpy(), minlength=len(q))
    qn = q["name_compact"].to_numpy()
    scores = process.cpdist(qn[a["query_id"].to_numpy()], s1_compact[a["s1_local"].to_numpy()],
                            scorer=fuzz.ratio, workers=-1)
    best = np.zeros(len(q))
    np.maximum.at(best, a["query_id"].to_numpy(), scores)
    return ((n_a < GATE_MIN_A) | (q["numbers"].to_numpy() == "") | q["addr_empty"].to_numpy()
            | (best < GATE_MIN_RATIO))


def run_c(qm, b, rows: np.ndarray, s1_pos: np.ndarray, threads: int) -> tuple:
    """Run top-N cosine for query rows ``rows``; return (candidate frame, seconds)."""
    t0 = time.perf_counter()
    c = sp_matmul_topn(qm[rows], b, top_n=TOP_N, sort=True, n_threads=threads)
    secs = time.perf_counter() - t0
    counts = np.diff(c.indptr)
    return pd.DataFrame({"query_id": np.repeat(rows, counts), "s1_id": s1_pos[c.indices]}), secs


def union_stats(a: pd.DataFrame, c: pd.DataFrame, q: pd.DataFrame, rows: np.ndarray = None) -> tuple:
    """Return (union recall, union candidates per query), optionally restricted to query ``rows``."""
    if rows is not None:
        a = a[np.isin(a["query_id"].to_numpy(), rows)]
    u = pd.concat([a[["query_id", "s1_id"]], c[["query_id", "s1_id"]]]).drop_duplicates()
    n = len(q) if rows is None else len(rows)
    return recall(u, q, rows), len(u) / n


def tune_country(country: str, split_df, pairs, threads: int, logger, base_sample: int = 0,
                 groups_mode: str = "base-name", max_dfs=MAX_DFS, ngrams=NGRAMS) -> dict:
    """Run every configuration for one country; return per-config metrics and fixed costs."""
    rng = np.random.default_rng(SEED)
    s1_pos = country_positions(norm_path("train", 1))[country]
    s1 = read_country(norm_path("train", 1), country, S1_COLS).to_pandas()
    s1_row_of = dict(zip(s1["entity_id"], s1_pos))
    val_s1 = set(split_df.loc[split_df["fold"] == "val", "s1_id"])
    vp = pairs[pairs["s1_id"].isin(val_s1)]
    val_owner = dict(zip(vp["other_id"], vp["s1_id"]))
    q = sample_queries(country, s1_row_of, val_owner, rng)
    del val_owner, vp
    gc.collect()
    logger.info("[%s] S1 %d, queries %d (val %d), rss %.0f MB", country, len(s1), len(q),
                int((q["true_s1"] >= 0).sum()), rss_mb())

    # pass A (unchanged; its cost is part of every configuration)
    t0 = time.perf_counter()
    df = token_df(s1["addr_tokens"].tolist())
    s1_keys = build_s1_keys(s1["num_keys"].tolist(), s1["addr_tokens"].tolist(), s1_pos.astype(np.int32), df)
    a_index_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    a = pass_a_chunk(np.arange(len(q)), q["num_keys"].tolist(), q["addr_tokens"].tolist(), s1_keys, df)
    a_s = time.perf_counter() - t0
    del s1_keys, df
    a["s1_local"] = np.searchsorted(s1_pos, a["s1_id"].to_numpy())
    rec_a = recall(a, q)
    t0 = time.perf_counter()
    gate = gate_mask(q, a, s1["name_compact"].to_numpy())
    gate_s = time.perf_counter() - t0
    gate_rows, all_rows = np.flatnonzero(gate), np.arange(len(q))
    logger.info("[%s] pass A: recall %.4f, %.2f cands/q, %.2fs; gate: %.1f%% of queries in %.2fs",
                country, rec_a, len(a) / len(q), a_s, 100 * gate.mean(), gate_s)

    rows = []
    if groups_mode == "base-name":
        groups = [("full", (3, 4), [None])] + [("name", ng, ([None] if ng == (3, 4) else []) + list(MAX_DFS))
                                              for ng in NGRAMS]
    else:  # "full-l2": L2/L3 on the current full text (name + address)
        groups = [("full", ng, list(max_dfs)) for ng in ngrams]
    for kind, ngram, max_dfs in groups:
        t0 = time.perf_counter()
        vec = TfidfVectorizer(ngram_range=ngram, norm=None, **BASE_PARAMS)
        x = vec.fit_transform(texts(s1, kind))
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        qraw = vec.transform(texts(q, kind))
        transform_s = time.perf_counter() - t0
        doc_freq = np.bincount(x.indices, minlength=x.shape[1])
        del vec
        gc.collect()
        for max_df in max_dfs:
            t0 = time.perf_counter()
            keep = np.flatnonzero(doc_freq <= max_df * len(s1)) if max_df else None
            xk = x if keep is None else x[:, keep]
            b = normalize(xk).T.tocsr()
            qm = normalize(qraw if keep is None else qraw[:, keep]).tocsr()
            prep_s = time.perf_counter() - t0
            del xk
            if kind == "full":
                level = "base" if max_df is None else "L2"
            else:
                level = "L1" if max_df is None else "L1+L2"
            name = f"{level} {kind} ng{ngram[0]}{ngram[1]}" + (f" max_df={max_df:.1%}" if max_df else "")
            for gated in ((False, True) if max_df else (False,)):
                sub = None   # base config may be scored on a random subset (speed + recall on that subset)
                if kind == "full" and max_df is None and 0 < base_sample < len(q):
                    sub = np.sort(np.random.default_rng(SEED).choice(len(q), size=base_sample, replace=False))
                sel = gate_rows if gated else (all_rows if sub is None else sub)
                c, c_s = run_c(qm, b, sel, s1_pos, threads)
                c_s *= len(q) / len(sel) if sub is not None else 1.0
                c_s_total = c_s + transform_s * (len(sel) / len(q) if gated else 1.0) + (gate_s if gated else 0.0)
                rec_u, cpq = union_stats(a, c, q, sub)
                row = {"config": (name.replace("L2", "L2+L3", 1) if gated else name),
                       "country": country, "kind": kind, "ngram": list(ngram), "max_df": max_df, "gated": gated,
                       "pct_gated": round(100 * len(sel) / len(q), 1),
                       "c_fit_s": round(fit_s + prep_s, 1), "c_query_s": round(c_s_total, 2),
                       "c_qps_local": round(len(q) / c_s_total, 1),
                       "vocab": int(b.shape[0]), "s1_nnz": int(b.nnz),
                       "base_sample": 0 if sub is None else len(sub),
                       "recall_C": round(recall(c, q, sub), 4), "recall_union": round(rec_u, 4),
                       "cands_per_query": round(cpq, 2), "rss_mb": round(rss_mb())}
                rows.append(row)
                logger.info("[%s] %s", country, row)
                del c
            del b, qm
            gc.collect()
        del x, qraw
        gc.collect()
    return {"s1_rows": len(s1), "queries": len(q), "recall_A": round(rec_a, 4), "a_index_s": round(a_index_s, 1),
            "a_query_s": round(a_s, 2), "pct_gated": round(100 * gate.mean(), 1), "configs": rows}


def project(results: dict, kaggle_bench: dict) -> list:
    """Return one row per configuration with Kaggle-calibrated pass-C speed and projected full-test minutes."""
    test = kaggle_bench["projection"]["test"]["by_country"]
    table = []
    names = [r["config"] for r in results["US"]["configs"]]
    for name in names:
        per = {}
        for c in ("US", "India"):
            res, kb = results[c], kaggle_bench["bench"][c]
            base = next(r for r in res["configs"] if r["config"].startswith("base"))
            r = next(r for r in res["configs"] if r["config"] == name)
            factor = (kb["c_per_100k_s"] / 1e5) / (base["c_query_s"] / res["queries"])   # kaggle s / local s
            fit_factor = kb["c_fit_s"] / base["c_fit_s"]
            per[c] = {"c_s_per_q": factor * r["c_query_s"] / res["queries"],
                      "fit_s_per_s1": fit_factor * r["c_fit_s"] / res["s1_rows"],
                      "a_s_per_q": kb["a_per_100k_s"] / 1e5, "a_fit_per_s1": kb["a_index_s"] / kb["s1_rows"],
                      "r": r, "factor": factor}
        minutes = {}
        for c, d in test.items():
            p = per.get(c, per["India"])      # France at India-like speed
            secs = d["s1"] * (p["fit_s_per_s1"] + p["a_fit_per_s1"]) + d["queries"] * (p["c_s_per_q"] + p["a_s_per_q"])
            minutes[c] = round(secs / 60, 1)
        row = {"config": name, "test_minutes": round(sum(minutes.values()), 1), "test_by_country": minutes}
        for c in ("US", "India"):
            r = per[c]["r"]
            row[c] = {"c_qps_kaggle": round(1 / per[c]["c_s_per_q"], 1), "c_qps_local": r["c_qps_local"],
                      "pct_gated": r["pct_gated"], "recall_C": r["recall_C"], "recall_union": r["recall_union"],
                      "cands_per_query": r["cands_per_query"], "calibration": round(per[c]["factor"], 2)}
        table.append(row)
    return table


def pick(table: list) -> dict:
    """Return the fastest config with union recall (val-pair weighted US+India mean) within 1 pp of the best and <= 3 h."""
    for row in table:
        row["recall_union_mean"] = round((row["US"]["recall_union"] + row["India"]["recall_union"]) / 2, 4)
    best = max(r["recall_union_mean"] for r in table)
    ok = [r for r in table if r["recall_union_mean"] >= best - 0.01 and r["test_minutes"] <= TARGET_TEST_MIN]
    return min(ok, key=lambda r: r["test_minutes"]) if ok else None


def main() -> None:
    """Run the tuning for the requested countries and write logs/pass_c_tuning.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", nargs="+", default=["India", "US"])
    ap.add_argument("--threads", type=int, default=4, help="sp_matmul_topn threads (4 = Kaggle's CPU count)")
    ap.add_argument("--base-sample", type=int, default=0,
                    help="score the (slow) base config on this many random sample queries only (0 = all)")
    ap.add_argument("--groups", choices=["base-name", "full-l2"], default="base-name",
                    help="base-name: base + L1 (+L2/L3 on names); full-l2: L2/L3 on the full text, appended")
    ap.add_argument("--max-dfs", type=float, nargs="+", default=list(MAX_DFS), help="max_df grid for full-l2")
    ap.add_argument("--ngrams", nargs="+", default=["34", "44"], help="ngram ranges for full-l2, e.g. 34 44")
    ap.add_argument("--kaggle-bench", required=True, help="blocking_bench.json from Kaggle run v1 (calibration)")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("tune_pass_c")
    out_path = LOG_DIR / "pass_c_tuning.json"
    results = json.loads(out_path.read_text(encoding="utf-8")).get("results", {}) if out_path.exists() else {}
    split_df = load_split()
    pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
    for country in args.countries:
        with StageTimer(f"tune_pass_c_{country}", logger, country=country, threads=args.threads):
            res = tune_country(country, split_df, pairs, args.threads, logger, args.base_sample, args.groups,
                               tuple(args.max_dfs), tuple((int(n[0]), int(n[1])) for n in args.ngrams))
            if args.groups == "full-l2" and country in results:
                done = {r["config"] for r in res["configs"]}
                res["configs"] = [r for r in results[country]["configs"] if r["config"] not in done] + res["configs"]
            results[country] = res
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            json.dump({"results": results}, f, indent=2)
            f.write("\n")
    kb = json.loads(open(args.kaggle_bench, encoding="utf-8").read())
    out = {"results": results}
    if all(c in results for c in ("US", "India")):
        out["table"] = project(results, kb)
        out["pick"] = pick(out["table"])
        logger.info("pick: %s", out["pick"] and out["pick"]["config"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
