"""Stage 4: combine check, stage-1 reduction, rule-based baseline scorer and the two submission files.

Inputs (cache/): cand/test/<country>/*_part_*.parquet + cand/test_union.parquet (after
`blocking --stage combine --split test`), cand/train_union.parquet (train/validation candidates),
cand/<split>/{queries,lookup_s1}.parquet, norm/, raw/ (train_pairs, test_source1), split.parquet.

Steps
1. Combine check: every test S2/S3 query was processed exactly once (every expected part file of
   every country / source / shard exists; no query id in two parts; parts only hold their own
   chunk's queries; no duplicate (query, S1) pair in test_union). -> logs/combine_check.json
2. Stage-1 reduction: per query keep the top TOP_K S1 by cheap = cosine_C + w_A * shared_keys_A;
   w_A tuned on the validation queries for recall@TOP_K. -> logs/stage1_reduction.json
3. Baseline scorer (rule score, no learning): per pair
       name_sim  = max(token_set_ratio(name_core), ratio(name_compact)) / 100
       addr_sim  = token_set_ratio(addr_clean) / 100            (0 if either address is empty)
       num_match = |shared num_keys| / min(|num_keys|)           (0 if either has none)
       score     = w_name * name_sim + w_addr * addr_sim + w_num * num_match
   Weights on a simplex grid (step WEIGHT_STEP) x thresholds, chosen to maximise validation macro
   F0.5 over ALL validation S1 entities (singletons included), with one-to-one assignment (each
   S2/S3 record goes only to its highest-scoring S1) and without it.
4. Applies the best configuration to test and writes output/candidate_pairs.tsv (the stage-1 set,
   i.e. exactly what the scorer saw) and output/matching_results.tsv, runs the organiser validator
   (must PASS) and re-checks there is no b"\\r".
5. Report -> logs/baseline_report.{json,md}.

    python -m src.finalize [--force] [--skip-combine-check]
"""
import argparse
import functools
import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .blocking import (CAND_DIR, CHUNK_ROWS, QUERY_ID_MULT, QUERY_SOURCES, country_slug, pair_keys,
                       shard_of)
from .config import DATA_DIR, LOG_DIR, N_THREADS, OUTPUT_DIR, add_path_args, set_seeds
from .io_utils import (assert_no_cr_file, raw_parquet_path, read_parquet, read_source_tsv,
                       write_candidate_pairs, write_matching_results, write_parquet)
from .logging_utils import StageTimer, get_logger
from .metric import f05_breakdown
from .normalize import norm_path
from .split import load_split

FINAL_DIR = CAND_DIR.parent / "final"
TOP_K = 5
W_A_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
WEIGHT_STEP = 0.1
COARSE_T = np.round(np.arange(0.20, 0.96, 0.025), 4)
FINE_STEP = 0.005
FEATURE_CHUNK = 1_000_000
TEXT_COLS = ["name_core", "name_compact", "addr_clean", "num_keys"]
PART_RE = re.compile(r"^(passA|passC|union)_s(\d)(?:_sh(\d+)of(\d+))?_part_(\d+)\.parquet$")


# ------------------------------------------------------------------ 1. combine check
def check_combine(logger) -> dict:
    """Verify every test S2/S3 query was blocked exactly once; raise AssertionError otherwise."""
    q = pd.read_parquet(CAND_DIR / "test" / "queries.parquet", columns=["query_id", "source", "country"])
    parts = {}
    for p in (CAND_DIR / "test").glob("*/*_part_*.parquet"):
        m = PART_RE.match(p.name)
        if m:
            kind, src, k, n, i = m.groups()
            parts[(p.parent.name, kind, int(src), int(k or 1), int(n or 1), int(i))] = p
    report, problems, seen = {"countries": {}}, [], {}
    for country, qc in q.groupby("country", sort=True):
        slug = country_slug(country)
        layouts = {(k, n) for (c, kind, _, k, n, _) in parts if c == slug and kind == "union"} or {(1, 1)}
        ns = {n for _, n in layouts}
        if len(ns) != 1 or {k for k, _ in layouts} != set(range(1, next(iter(ns)) + 1)):
            problems.append(f"{country}: inconsistent shard layout {sorted(layouts)}")
            continue
        n = next(iter(ns))
        info = {"queries": len(qc), "shards": n, "parts": 0, "queries_with_candidates": 0}
        for source in QUERY_SOURCES:
            qs = qc.loc[qc["source"] == source, "query_id"].to_numpy()
            for k in range(1, n + 1):
                sub = qs if n == 1 else qs[shard_of(qs, n) == k - 1]
                n_chunks = -(-len(sub) // CHUNK_ROWS)
                for i in range(n_chunks):
                    chunk = sub[i * CHUNK_ROWS:(i + 1) * CHUNK_ROWS]
                    for kind in ("passA", "passC", "union"):
                        if (slug, kind, source, k, n, i) not in parts:
                            problems.append(f"{country}: missing {kind} s{source} shard {k}/{n} part {i}")
                    p = parts.get((slug, "union", source, k, n, i))
                    if p is None:
                        continue
                    ids = np.unique(pq.read_table(p, columns=["query_id"])["query_id"].to_numpy())
                    if not np.isin(ids, chunk).all():
                        problems.append(f"{p.name} ({country}) holds queries outside its chunk")
                    seen[p] = ids
                    info["parts"] += 1
                    info["queries_with_candidates"] += len(ids)
        info["pct_zero_candidates"] = round(100 * (1 - info["queries_with_candidates"] / max(1, len(qc))), 3)
        report["countries"][country] = info
    all_ids = np.concatenate(list(seen.values())) if seen else np.empty(0, np.int64)
    dup = len(all_ids) - len(np.unique(all_ids))
    if dup:
        problems.append(f"{dup} query ids appear in more than one union part")
    extra = sorted(set(slug for (slug, *_rest) in parts) - {country_slug(c) for c in q["country"].unique()})
    if extra:
        problems.append(f"part files for countries not in the query set: {extra}")
    comb = CAND_DIR / "test_union.parquet"
    if comb.exists():
        t = pq.read_table(comb, columns=["query_id", "s1_id"])
        keys = pair_keys(t["query_id"].to_numpy(), t["s1_id"].to_numpy())
        report["combined_rows"] = t.num_rows
        report["combined_rows_match_parts"] = t.num_rows == sum(pq.ParquetFile(p).metadata.num_rows for p in seen)
        report["combined_duplicate_pairs"] = int(len(keys) - len(np.unique(keys)))
        if not report["combined_rows_match_parts"] or report["combined_duplicate_pairs"]:
            problems.append("combined test_union does not equal the union of the parts")
        del t, keys
    else:
        problems.append("cand/test_union.parquet missing (run blocking --stage combine --split test)")
    report.update(total_queries=len(q), queries_with_candidates=int(len(all_ids)),
                  duplicate_query_ids_across_parts=int(dup), problems=problems, ok=not problems)
    save_json(report, "combine_check.json")
    logger.info("combine check: %s", {k: v for k, v in report.items() if k != "countries"})
    if problems:
        raise AssertionError("combine check failed: " + "; ".join(problems[:10]))
    return report


# ------------------------------------------------------------------ 2. stage-1 reduction
def load_union(split: str) -> pd.DataFrame:
    """Return the combined candidate table (query_id, s1_id, scoreA, scoreC) with NaN scores as 0."""
    df = pd.read_parquet(CAND_DIR / f"{split}_union.parquet", columns=["query_id", "s1_id", "scoreA", "scoreC"])
    df["scoreA"] = df["scoreA"].fillna(0).astype(np.float32)
    df["scoreC"] = df["scoreC"].fillna(0).astype(np.float32)
    return df


def top_k(df: pd.DataFrame, w_a: float, k: int = TOP_K) -> pd.DataFrame:
    """Return, per query, the k candidates with the highest cosine_C + w_a * shared_keys_A (ties: lower s1_id)."""
    cheap = df["scoreC"].to_numpy() + np.float32(w_a) * df["scoreA"].to_numpy()
    order = np.lexsort((df["s1_id"].to_numpy(), -cheap, df["query_id"].to_numpy()))
    qid = df["query_id"].to_numpy()[order]
    start = np.r_[0, np.flatnonzero(np.diff(qid)) + 1]
    rank = np.arange(len(qid)) - np.repeat(start, np.diff(np.r_[start, len(qid)]))
    keep = order[rank < k]
    out = df.iloc[keep].reset_index(drop=True)
    out["cheap"] = cheap[keep]
    out["s1_rank"] = (rank[rank < k] + 1).astype(np.int8)
    return out


def val_truth(logger) -> dict:
    """Return validation ground truth: pair keys, val S1 ids, per-val-S1 true counts and id maps (train split)."""
    queries = pd.read_parquet(CAND_DIR / "train" / "queries.parquet",
                              columns=["query_id", "entity_id", "country", "role"])
    lookup = pd.read_parquet(CAND_DIR / "train" / "lookup_s1.parquet")
    split_df = load_split()
    val_ent = split_df.loc[split_df["fold"] == "val", "s1_id"]
    s1_of = pd.Series(lookup["s1_id"].to_numpy(), index=lookup["entity_id"])
    val_s1 = np.sort(s1_of.reindex(val_ent).to_numpy().astype(np.int64))
    pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
    pairs = pairs[pairs["s1_id"].isin(set(val_ent))]
    q_of = pd.Series(queries["query_id"].to_numpy(), index=queries["entity_id"])
    qid = q_of.reindex(pairs["other_id"]).to_numpy()
    if np.isnan(qid.astype(float)).any():
        raise AssertionError("some validation pairs are missing from the train query set")
    sid = s1_of.reindex(pairs["s1_id"]).to_numpy().astype(np.int64)
    keys = np.sort(pair_keys(qid.astype(np.int64), sid))
    n_true = np.bincount(np.searchsorted(val_s1, sid), minlength=len(val_s1))
    logger.info("validation: %d S1 entities, %d true pairs, singleton rate %.4f",
                len(val_s1), len(keys), float((n_true == 0).mean()))
    return {"keys": keys, "val_s1": val_s1, "n_true": n_true, "lookup": lookup, "queries": queries,
            "pair_country": lookup["country"].to_numpy()[sid]}


def is_true_pair(df: pd.DataFrame, keys: np.ndarray) -> np.ndarray:
    """Return a bool array: (query_id, s1_id) of each row is a validation ground-truth pair."""
    k = pair_keys(df["query_id"].to_numpy(), df["s1_id"].to_numpy())
    idx = np.searchsorted(keys, k)
    idx[idx == len(keys)] = 0
    return keys[idx] == k


def tune_stage1(union: pd.DataFrame, truth: dict, logger) -> tuple:
    """Pick w_A maximising validation recall@TOP_K; return (w_A, report)."""
    n_pairs = len(truth["keys"])
    hit_all = is_true_pair(union, truth["keys"])
    report = {"val_pairs": n_pairs, "recall_before_reduction": round(float(hit_all.sum()) / n_pairs, 4),
              "cands_per_query_before": round(len(union) / union["query_id"].nunique(), 2), "grid": {}}
    best = (-1.0, None)
    for w in W_A_GRID:
        red = top_k(union, w)
        r = float(is_true_pair(red, truth["keys"]).sum()) / n_pairs
        report["grid"][str(w)] = round(r, 5)
        logger.info("stage-1 w_A=%s: recall@%d %.5f", w, TOP_K, r)
        if r > best[0]:
            best = (r, w)
        del red
        gc.collect()
    w = best[1]
    red = top_k(union, w)
    hit = is_true_pair(red, truth["keys"])
    tc = truth["pair_country"]
    report.update(w_A=w, recall_at_k=round(best[0], 5), k=TOP_K,
                  cands_per_query_after=round(len(red) / red["query_id"].nunique(), 2))
    report["recall_at_rank"] = {str(r): round(float(hit[red["s1_rank"].to_numpy() <= r].sum()) / n_pairs, 5)
                                for r in range(1, TOP_K + 1)}
    s1_country = truth["lookup"]["country"].to_numpy()[red["s1_id"].to_numpy()]
    report["by_country"] = {}
    for c in sorted(set(tc)):
        n_c = int((tc == c).sum())
        report["by_country"][c] = {
            "val_pairs": n_c,
            "recall_before_reduction": round(float(hit_all[truth["lookup"]["country"].to_numpy()[
                union["s1_id"].to_numpy()] == c].sum()) / n_c, 4),
            "recall_at_k": round(float(hit[s1_country == c].sum()) / n_c, 4)}
    return w, report, red


# ------------------------------------------------------------------ 3. features
@functools.lru_cache(maxsize=3)
def text_table(split: str, source: int) -> pa.Table:
    """Return (cached) the TEXT_COLS of one normalised file as an Arrow table."""
    return pq.read_table(norm_path(split, source), columns=TEXT_COLS)


def take_texts(split: str, source: int, rows: np.ndarray) -> dict:
    """Return {col: object array} of TEXT_COLS for the given row indices of a normalised file."""
    uniq, inv = np.unique(rows, return_inverse=True)
    t = text_table(split, source).take(pa.array(uniq))
    return {c: np.asarray(t[c].to_pylist(), dtype=object)[inv] for c in TEXT_COLS}


def num_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return |shared keys| / min(|keys|) for space-separated num_keys strings (0 if either is empty)."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            out[i] = len(sx & sy) / min(len(sx), len(sy))
    return out


def pair_features(qt: dict, st: dict) -> dict:
    """Return name_sim, addr_sim, num_match (float32) for aligned query / S1 text dicts."""
    name = np.maximum(process.cpdist(qt["name_core"], st["name_core"], scorer=fuzz.token_set_ratio, workers=-1),
                      process.cpdist(qt["name_compact"], st["name_compact"], scorer=fuzz.ratio, workers=-1))
    addr = process.cpdist(qt["addr_clean"], st["addr_clean"], scorer=fuzz.token_set_ratio, workers=-1)
    empty = (qt["addr_clean"] == "") | (st["addr_clean"] == "")
    addr = np.where(empty, 0.0, addr)
    return {"name_sim": (name / 100).astype(np.float32), "addr_sim": (addr / 100).astype(np.float32),
            "num_match": num_match(qt["num_keys"], st["num_keys"])}


def add_features(split: str, red: pd.DataFrame, logger) -> pd.DataFrame:
    """Add the three baseline features to the stage-1 pairs (chunked, per query source)."""
    feats = {k: np.zeros(len(red), dtype=np.float32) for k in ("name_sim", "addr_sim", "num_match")}
    qid = red["query_id"].to_numpy()
    src = qid // QUERY_ID_MULT
    for source in QUERY_SOURCES:
        idx = np.flatnonzero(src == source)
        for start in range(0, len(idx), FEATURE_CHUNK):
            sl = idx[start:start + FEATURE_CHUNK]
            qt = take_texts(split, source, qid[sl] % QUERY_ID_MULT)
            st = take_texts(split, 1, red["s1_id"].to_numpy()[sl])
            f = pair_features(qt, st)
            for k in feats:
                feats[k][sl] = f[k]
            del qt, st, f
            gc.collect()
        logger.info("[%s] features done for source %d (%d pairs)", split, source, len(idx))
    for k, v in feats.items():
        red[k] = v
    return red


def stage1_features(split: str, union: pd.DataFrame, w_a: float, force: bool, logger) -> pd.DataFrame:
    """Return (and cache) the stage-1 top-K pairs with features for ``split``."""
    path = FINAL_DIR / f"{split}_stage1_k{TOP_K}_wA{w_a:g}.parquet"
    if path.exists() and not force:
        logger.info("cached %s", path)
        return pd.read_parquet(path)
    red = add_features(split, top_k(union, w_a), logger)
    write_parquet(red, path)
    return red


# ------------------------------------------------------------------ 4. scoring + grid search
def weight_grid(step: float = WEIGHT_STEP) -> list:
    """Return all (w_name, w_addr, w_num) on the simplex with the given step, w_name > 0."""
    n = round(1 / step)
    return [(i / n, j / n, (n - i - j) / n) for i in range(1, n + 1) for j in range(0, n - i + 1)]


def best_per_query(qid: np.ndarray, score: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Return a bool mask of the single highest-scoring pair per query (ties: lower s1_id). qid must be sorted."""
    order = np.lexsort((s1, -score, qid))
    first = np.r_[True, qid[order][1:] != qid[order][:-1]]
    mask = np.zeros(len(qid), dtype=bool)
    mask[order[first]] = True
    return mask


def macro_f05_fast(ent: np.ndarray, is_true: np.ndarray, n_true: np.ndarray) -> float:
    """Return macro F0.5 over all entities given predicted pairs (entity index, is_true) and true counts."""
    n = len(n_true)
    pred = np.bincount(ent, minlength=n)
    tp = np.bincount(ent, weights=is_true, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n_true == 0, (pred == 0).astype(float),
                     np.where(pred == 0, 0.0, 1.25 * tp / (pred + 0.25 * n_true)))
    return float(f.mean())


def grid_search(val: dict, logger) -> dict:
    """Search weights x thresholds for max validation macro F0.5, with and without one-to-one."""
    qid, s1, ent, true = val["qid"], val["s1"], val["ent"], val["true"]
    f = np.stack([val["name_sim"], val["addr_sim"], val["num_match"]], axis=1)
    rel = ent >= 0
    results = {"o2o": (-1, None), "all": (-1, None)}
    for w in weight_grid():
        score = f @ np.asarray(w, dtype=np.float32)
        best = best_per_query(qid, score, s1)
        for mode, mask in (("o2o", rel & best), ("all", rel)):
            sc, e, t = score[mask], ent[mask], true[mask]
            for thr in COARSE_T:
                m = sc >= thr
                v = macro_f05_fast(e[m], t[m], val["n_true"])
                if v > results[mode][0]:
                    results[mode] = (v, {"weights": w, "threshold": float(thr)})
    for mode in results:   # refine the threshold around the best coarse value
        w = results[mode][1]["weights"]
        score = f @ np.asarray(w, dtype=np.float32)
        mask = rel & best_per_query(qid, score, s1) if mode == "o2o" else rel
        sc, e, t = score[mask], ent[mask], true[mask]
        t0 = results[mode][1]["threshold"]
        for thr in np.round(np.arange(t0 - 0.025, t0 + 0.0251, FINE_STEP), 4):
            m = sc >= thr
            v = macro_f05_fast(e[m], t[m], val["n_true"])
            if v > results[mode][0]:
                results[mode] = (v, {"weights": w, "threshold": float(thr)})
        logger.info("best %s: F0.5 %.5f at %s", mode, *results[mode])
    return {m: {"macro_f05": round(v, 5), **cfg} for m, (v, cfg) in results.items()}


def predict(red: pd.DataFrame, weights, threshold: float, o2o: bool) -> np.ndarray:
    """Return a bool mask of predicted pairs for the stage-1 table (query_id-sorted)."""
    score = (weights[0] * red["name_sim"].to_numpy() + weights[1] * red["addr_sim"].to_numpy()
             + weights[2] * red["num_match"].to_numpy())
    keep = score >= threshold
    if o2o:
        keep &= best_per_query(red["query_id"].to_numpy(), score, red["s1_id"].to_numpy())
    return keep


# ------------------------------------------------------------------ 5. outputs
def test_s1_order() -> list:
    """Return test S1 entity ids in the organiser file order."""
    p = raw_parquet_path("test", "source1")
    if p.exists():
        return read_parquet(p, columns=["entity_id"])["entity_id"].tolist()
    return read_source_tsv("test", 1)["entity_id"].tolist()


def id_lists(red: pd.DataFrame, mask: np.ndarray, s1_ent: np.ndarray, q_ent: pd.Series) -> dict:
    """Return {S1 entity id: [S2/S3 entity ids]} for the selected pairs (ordered by stage-1 rank)."""
    sub = red.loc[mask, ["query_id", "s1_id", "cheap"]].sort_values(["s1_id", "cheap"], ascending=[True, False])
    ents = q_ent.reindex(sub["query_id"]).to_numpy()
    s1s = sub["s1_id"].to_numpy()
    out = {}
    if len(s1s):
        cut = np.flatnonzero(np.diff(s1s)) + 1
        for s, grp in zip(s1s[np.r_[0, cut]], np.split(ents, cut)):
            out[s1_ent[s]] = grp.tolist()
    return out


def run_validator(matching: Path, candidate: Path, logger) -> str:
    """Run the organiser validator (stdlib only); raise unless it prints PASS and exits 0."""
    validator = DATA_DIR.parent / "utils" / "validate_submission.py"
    if not validator.exists():                  # packaged copy of the organiser validator (unchanged)
        validator = Path(__file__).resolve().parent / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate),
                        "--test-dir", str(DATA_DIR / "test"), "--check-ids"], capture_output=True,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    text = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    with open(LOG_DIR / "validate_submission.txt", "w", encoding="utf-8", newline="") as f:
        f.write(text)
    logger.info("validator (rc %d):\n%s", r.returncode, text[-2000:])
    if r.returncode != 0 or "PASS" not in text:
        raise AssertionError("validate_submission.py did not PASS")
    return text


def save_json(obj: dict, name: str) -> None:
    """Write ``obj`` as JSON to logs/<name>."""
    with open(LOG_DIR / name, "w", encoding="utf-8", newline="") as f:
        json.dump(obj, f, indent=2, default=float)
        f.write("\n")


def report_md(rep: dict) -> str:
    """Render the baseline report as Markdown."""
    b = rep["best"]
    lines = ["# Baseline report", "",
             f"Stage 1: top-{rep['stage1']['k']} per query by cosine_C + {rep['stage1']['w_A']} x shared_keys_A; "
             f"validation recall {rep['stage1']['recall_before_reduction']} -> {rep['stage1']['recall_at_k']} "
             f"(@{rep['stage1']['k']}).", "",
             f"Best config: mode={b['mode']}, weights (name, addr, num) = {b['weights']}, threshold = {b['threshold']}", "",
             "| validation | macro F0.5 | mean precision | mean recall | singleton acc | mean pred size |",
             "|---|---|---|---|---|---|"]
    for label, m in rep["validation_rows"].items():
        lines.append(f"| {label} | {m['macro_f05']:.4f} | {m['mean_precision']:.4f} | {m['mean_recall']:.4f} | "
                     f"{m['empty_pred_rate_singletons']:.4f} | {m['mean_pred_size']:.3f} |")
    lines += ["", "| test country | S1 | % empty predictions | mean predicted per S1 | mean candidates per S1 |",
              "|---|---|---|---|---|"]
    for c, s in rep["test_by_country"].items():
        lines.append(f"| {c} | {s['s1']:,} | {s['pct_empty']:.2f} | {s['mean_pred']:.3f} | {s['mean_cands']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run steps 1-5 (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="recompute cached stage-1 feature tables")
    ap.add_argument("--skip-combine-check", action="store_true", help="for local smoke tests on partial data")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("finalize")
    logger.info("threads %d, cache %s, output %s", N_THREADS, CAND_DIR.parent, OUTPUT_DIR)
    rep = {}

    if not args.skip_combine_check:
        with StageTimer("finalize_combine_check", logger):
            rep["combine_check"] = {k: v for k, v in check_combine(logger).items() if k != "countries"}

    with StageTimer("finalize_stage1", logger):
        truth = val_truth(logger)
        union = load_union("train")
        w_a, rep["stage1"], _ = tune_stage1(union, truth, logger)
        save_json(rep["stage1"], "stage1_reduction.json")
        tr = stage1_features("train", union, w_a, args.force, logger)
        del union
        gc.collect()

    with StageTimer("finalize_grid", logger):
        val_pos = np.full(len(truth["lookup"]), -1, dtype=np.int64)
        val_pos[truth["val_s1"]] = np.arange(len(truth["val_s1"]))
        val = {"qid": tr["query_id"].to_numpy(), "s1": tr["s1_id"].to_numpy(),
               "ent": val_pos[tr["s1_id"].to_numpy()], "true": is_true_pair(tr, truth["keys"]).astype(np.float64),
               "n_true": truth["n_true"], "name_sim": tr["name_sim"].to_numpy(),
               "addr_sim": tr["addr_sim"].to_numpy(), "num_match": tr["num_match"].to_numpy()}
        rep["grid"] = grid_search(val, logger)
        mode = "o2o" if rep["grid"]["o2o"]["macro_f05"] >= rep["grid"]["all"]["macro_f05"] else "all"
        best = rep["grid"][mode]
        rep["best"] = {"mode": mode, **best, "w_A": w_a, "top_k": TOP_K}

        # exact metric (src.metric) on the validation predictions, with and without one-to-one
        lk = truth["lookup"]
        s1_ent = lk["entity_id"].to_numpy()
        q_ent = pd.Series(truth["queries"]["entity_id"].to_numpy(), index=truth["queries"]["query_id"])
        pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
        val_ents = s1_ent[truth["val_s1"]]
        true_map = {e: [] for e in val_ents}
        for s, o in zip(pairs["s1_id"], pairs["other_id"]):
            if s in true_map:
                true_map[s].append(o)
        country_map = dict(zip(val_ents, lk["country"].to_numpy()[truth["val_s1"]]))
        rows = {}
        for label, o2o, cfg in ((f"best ({mode}) with one-to-one", True, best),
                                (f"best ({mode}) without one-to-one", False, best),
                                ("best no-one-to-one config", False, rep["grid"]["all"])):
            m = predict(tr, cfg["weights"], cfg["threshold"], o2o) & (val["ent"] >= 0)
            b = f05_breakdown(id_lists(tr, m, s1_ent, q_ent), true_map, country_map)
            rows[label] = b["overall"]
            for c, v in b["by_country"].items():
                rows[f"{label} | {c}"] = v
        rep["validation_rows"] = rows
        for label, m in rows.items():
            logger.info("%s: %s", label, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
        del tr, val, pairs, true_map
        gc.collect()

    with StageTimer("finalize_test", logger):
        te = stage1_features("test", load_union("test"), w_a, args.force, logger)
        keep = predict(te, best["weights"], best["threshold"], mode == "o2o")
        lk = pd.read_parquet(CAND_DIR / "test" / "lookup_s1.parquet")
        tq = pd.read_parquet(CAND_DIR / "test" / "queries.parquet", columns=["query_id", "entity_id"])
        q_ent = pd.Series(tq["entity_id"].to_numpy(), index=tq["query_id"])
        s1_ent = lk["entity_id"].to_numpy()
        cands = id_lists(te, np.ones(len(te), dtype=bool), s1_ent, q_ent)
        matches = id_lists(te, keep, s1_ent, q_ent)
        order = test_s1_order()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cand_path, match_path = OUTPUT_DIR / "candidate_pairs.tsv", OUTPUT_DIR / "matching_results.tsv"
        n_c = write_candidate_pairs(cand_path, order, cands)
        n_m = write_matching_results(match_path, order, matches)
        for p in (cand_path, match_path):
            assert_no_cr_file(p)
        if not all(set(v) <= set(cands.get(k, ())) for k, v in matches.items()):
            raise AssertionError("a matched id is not among that S1's candidates")
        rep["validator"] = run_validator(match_path, cand_path, logger).strip().splitlines()[-3:]
        by_c = {}
        country = lk.set_index("entity_id")["country"]
        for c, ents in country.groupby(country).groups.items():
            ents = list(ents)
            n_pred = np.array([len(matches.get(e, ())) for e in ents])
            n_cand = np.array([len(cands.get(e, ())) for e in ents])
            by_c[c] = {"s1": len(ents), "pct_empty": 100 * float((n_pred == 0).mean()),
                       "mean_pred": float(n_pred.mean()), "mean_cands": float(n_cand.mean())}
        rep["test_by_country"] = by_c
        rep["test_rows"] = {"matching": n_m, "candidate": n_c, "pairs_scored": int(len(te)),
                            "pairs_matched": int(keep.sum())}
    save_json(rep, "baseline_report.json")
    with open(LOG_DIR / "baseline_report.md", "w", encoding="utf-8", newline="") as f:
        f.write(report_md(rep))
    logger.info("report:\n%s", report_md(rep))


if __name__ == "__main__":
    main()
