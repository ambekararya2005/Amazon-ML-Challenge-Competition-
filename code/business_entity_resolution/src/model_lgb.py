"""Two-stage LightGBM pair model + S1-level has-match model + expected-F0.5 decoder (benchmark train / test submit).

    python -m src.model_lgb --stage train    # <output>/features_v3/bench/*.parquet -> <output>/models_v3/,
                                             #   logs/model_v3_report.{json,md}, logs/model_v3_errors.txt
    python -m src.model_lgb --stage submit   # models + <output>/features_v3/test/*.parquet -> output/*.tsv
    python -m src.model_lgb --stage train_full   # MODEL_FULL=1: one model per stage on ALL 5 bench folds
    python -m src.model_lgb --stage assemble     # per-country parts (SUBMIT_PARTS=1) -> output/final/*.tsv

Env: MODEL_FULL=1 uses <output>/models_<tag>full (rounds = 1.1 x mean CV best iteration, calibration / decoder
settings copied from the CV artefacts); SUBMIT_COUNTRY=<c> scores one country only; SUBMIT_PARTS=1 writes
<output>/submit_parts_<model>/<country>.parquet (one row per S1: entity id, candidate list, match list) instead
of the TSVs; SAVE_PAIRS=1 also writes <output>/test_pairs_<model>/pairs_<country>.parquet (query_id, s1_id, p2,
kept = query argmax, keep = selected, xq for unselected argmax pairs) + h_<country>.parquet; assemble reads every such part (local output or PARTS_DIR) and writes + validates the TSVs.

Training uses benchmark folds 1-4 only (fold 0 = the gate): leave-one-fold-out CV (the folds are groups of regions)
gives out-of-fold (OOF) predictions for folds 1-4; fold 0 and test use the mean of the 4 fold models.
  target encoding  per extra word, smoothed P(not a true pair | the word is extra) over claimant pairs
                   (base_score >= CLAIM_MIN) of the fitting folds (OOF for training rows); max / mean / sum and the
                   unseen-word count per pair, for words extra in the query and words extra in the S1
  stage 1          LightGBM binary on all pair features -> p1
  stage 2          + group aggregates of p1 within the S1 and within the query -> p2, isotonic-calibrated on OOF
  has-match        one row per S1 (label = the S1 has >= 1 true match): claimant / p2 statistics -> h (isotonic)
  decoders         one-to-one (each query keeps its argmax-p2 S1), then (a) threshold + empty rule or
                   (b) exact expected F0.5 with temperature / miss prior / optional h; tuned on OOF folds 1-4.
No country feature: country only partitions the work.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking import QUERY_ID_MULT
from .config import FEATURE_VARIANT, LOG_DIR, N_THREADS, OUTPUT_DIR, SEED, add_path_args, env_float, set_seeds
from .decoder import error_budget, expected_f05_decode, fold_metrics, threshold_decode
from .features_v3 import CLAIM_MIN, V3_DIR, group_rank, group_top2
from .finalize import best_per_query
from .io_utils import raw_parquet_path, read_parquet, write_parquet
from .logging_utils import StageTimer, get_logger
from .normalize import norm_path
from .scorer_v2 import BASELINE, CONFIG_FILE, find_file, score_baseline

TAG = "v3" if FEATURE_VARIANT == "v1" else FEATURE_VARIANT       # v1 candidates + v3 features = "v3"
FULL = os.environ.get("MODEL_FULL", "0") == "1"
MODEL_NAME = TAG + ("full" if FULL else "")
CV_MODEL_DIR = OUTPUT_DIR / f"models_{TAG}"      # leave-one-fold-out models (folds 1-4) + OOF artefacts
MODEL_DIR = OUTPUT_DIR / f"models_{MODEL_NAME}"
SUBMIT_COUNTRY = os.environ.get("SUBMIT_COUNTRY", "")
SUBMIT_PARTS = os.environ.get("SUBMIT_PARTS", "0") == "1"
PARTS_DIR = OUTPUT_DIR / f"submit_parts_{MODEL_NAME}"
SAVE_PAIRS = os.environ.get("SAVE_PAIRS", "0") == "1"          # also write per-pair test p2 / masks
PAIRS_DIR = OUTPUT_DIR / f"test_pairs_{MODEL_NAME}"
FULL_ROUND_MULT = 1.1
TRAIN_FOLDS = (1, 2, 3, 4)
NON_FEATURES = {"query_id", "s1_id", "label", "q_true_s1", "fold", "xq", "xs", "n_xq", "n_xs", "country"}
TE_M = 20.0                       # smoothing strength (pseudo-count) of the extra-word target encoding
TE_MIN_COUNT = 1                  # a word needs this many fitting occurrences to count as "seen"
PRED_CHUNK = 2_000_000
NEG_KEEP = env_float("NEG_KEEP", 1.0)     # share of negative pairs kept for training (1 = all; weights restore)
DEBUG_FRAC = env_float("DEBUG_FRAC", 1.0) # < 1: local dry run on a slice of the benchmark S1
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, feature_fraction=0.8,
                  min_data_in_leaf=200, lambda_l2=1.0, max_bin=255, metric="binary_logloss", verbose=-1,
                  seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED,
                  num_threads=N_THREADS)
HM_PARAMS = dict(LGB_PARAMS, num_leaves=31, min_data_in_leaf=100)
ROUNDS, EARLY_STOP = 2000, 50
T_GRID = np.round(np.arange(0.20, 0.951, 0.025), 3)
TEMP_GRID = (0.7, 0.85, 1.0, 1.2, 1.5)
MISS_GRID = (0.0, 0.015, 0.03, 0.05)
AGG1 = ["p1", "p1_s1_max", "p1_s1_second", "p1_s1_sum", "p1_s1_cnt05", "p1_s1_rank", "p1_minus_s1max",
        "p1_s1_other_max", "p1_q_other_max", "p1_q_margin", "p1_q_rank"]
HM_FEATURES = ["hm_max", "hm_second", "hm_sum", "hm_cnt", "hm_cnt05", "hm_all_max", "hm_all_sum", "hm_n_pairs",
               "hm_claimants", "hm_noconf", "hm_noextra", "hm_best_name", "hm_best_addr", "hm_best_v2",
               "hm_best_twin", "s1_name_freq", "s1_addr_missing"]


# ------------------------------------------------------------------ extra-word target encoding
def explode_tokens(strings) -> tuple:
    """Return (flat token array, parent row index) for a column of space-joined words."""
    arr = pa.array(strings) if not isinstance(strings, (pa.Array, pa.ChunkedArray)) else strings
    lst = pc.utf8_split_whitespace(arr)
    tok, parent = pc.list_flatten(lst), pc.list_parent_indices(lst).to_numpy()
    nonempty = pc.greater(pc.utf8_length(tok), 0)             # '' splits into one empty token
    return tok.filter(nonempty), parent[nonempty.to_numpy(zero_copy_only=False)]


def te_fit(strings, label: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Return the vocabulary table (token, n, n_neg, score) fitted on the given row indices."""
    tok, parent = explode_tokens(pa.array(pd.Series(strings).iloc[rows].array))
    df = pd.DataFrame({"token": tok.to_numpy(zero_copy_only=False), "neg": 1 - label[rows][parent]})
    g = df.groupby("token")["neg"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "n_neg"})
    prior = float(df["neg"].mean()) if len(df) else 0.5
    g["score"] = (g["n_neg"] + TE_M * prior) / (g["n"] + TE_M)
    g.attrs["prior"] = prior
    return g.reset_index()


def te_apply(strings, vocab: pd.DataFrame, prior: float, prefix: str, n: int) -> dict:
    """Return te_<prefix>_{max,mean,sum,unseen} for every row given a fitted vocabulary."""
    tok, parent = explode_tokens(strings)
    v = vocab[vocab["n"] >= TE_MIN_COUNT]
    idx = pc.index_in(tok, value_set=pa.array(v["token"].to_numpy(), type=pa.string()))
    idx = idx.to_numpy(zero_copy_only=False)
    seen = ~pd.isna(idx)
    score = np.full(len(idx), prior, dtype=np.float64)
    score[seen] = v["score"].to_numpy()[idx[seen].astype(np.int64)]
    cnt = np.bincount(parent, minlength=n)
    s = np.bincount(parent, weights=score, minlength=n)
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, parent, score)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {f"te_{prefix}_max": np.where(cnt > 0, mx, np.nan), f"te_{prefix}_mean": np.where(cnt > 0, s / cnt, np.nan),
               f"te_{prefix}_sum": s, f"te_{prefix}_unseen": np.bincount(parent, weights=~seen, minlength=n)}
    return {k: v.astype(np.float32) for k, v in out.items()}


def add_te_bench(df: pd.DataFrame, logger) -> dict:
    """Add OOF target-encoding columns to the benchmark table; return the vocabularies fitted on all train folds."""
    label, fold = df["label"].to_numpy(), df["fold"].to_numpy()
    fit_mask = (df["base_score"].to_numpy() >= CLAIM_MIN)
    vocabs = {}
    for side in ("xq", "xs"):
        cols = {}
        for k in TRAIN_FOLDS:
            rows = np.flatnonzero(fit_mask & np.isin(fold, [f for f in TRAIN_FOLDS if f != k]))
            v = te_fit(df[side], label, rows)
            target = np.flatnonzero(fold == k)
            part = te_apply(pa.array(df[side].iloc[target].array), v, v.attrs["prior"], side[1], len(target))
            for c, a in part.items():
                cols.setdefault(c, np.full(len(df), np.nan, np.float32))[target] = a
        v = te_fit(df[side], label, np.flatnonzero(fit_mask & np.isin(fold, TRAIN_FOLDS)))
        target = np.flatnonzero(~np.isin(fold, TRAIN_FOLDS))
        part = te_apply(pa.array(df[side].iloc[target].array), v, v.attrs["prior"], side[1], len(target))
        for c, a in part.items():
            cols[c][target] = a
        for c, a in cols.items():
            df[c] = a
        vocabs[side] = v
        top = v[v["n"] >= 200].sort_values("score", ascending=False)
        logger.info("TE %s: %d words (prior %.3f); most decoy-like: %s", side, len(v), v.attrs["prior"],
                    top.head(15)[["token", "n", "score"]].round(3).values.tolist())
    return vocabs


# ------------------------------------------------------------------ LightGBM helpers
def feature_list(df: pd.DataFrame) -> list:
    """Return the model feature columns (numeric, not ids / labels / strings)."""
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def matrix(df: pd.DataFrame, feats: list) -> np.ndarray:
    """Return a float32 C-contiguous feature matrix."""
    X = np.empty((len(df), len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = df[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return X


def predict_chunked(models: list, X: np.ndarray, rows: np.ndarray = None) -> np.ndarray:
    """Return the mean prediction of ``models`` over rows of X, in chunks."""
    rows = np.arange(len(X)) if rows is None else rows
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), PRED_CHUNK):
        r = rows[s:s + PRED_CHUNK]
        out[s:s + PRED_CHUNK] = np.mean([m.predict(X[r], num_iteration=m.best_iteration) for m in models], axis=0)
    return out


def cv_train(X: np.ndarray, y: np.ndarray, fold: np.ndarray, feats: list, params: dict, tag: str, logger) -> tuple:
    """Leave-one-fold-out over TRAIN_FOLDS; return (prediction for every row: OOF on folds 1-4, mean of the fold
    models elsewhere; models; per-fold info)."""
    rng = np.random.default_rng(SEED)
    w = np.ones(len(y), dtype=np.float32)
    use = np.ones(len(y), dtype=bool)
    if NEG_KEEP < 1.0:
        neg = y == 0
        use = ~neg | (rng.random(len(y)) < NEG_KEEP)
        w[neg] = 1.0 / NEG_KEEP
    ds = lgb.Dataset(X, label=y, weight=w, feature_name=feats, free_raw_data=False,
                     params={"max_bin": params["max_bin"], "verbose": -1}).construct()
    pred = np.zeros(len(y), dtype=np.float32)
    models, info = [], []
    for k in TRAIN_FOLDS:
        tr = np.flatnonzero(np.isin(fold, [f for f in TRAIN_FOLDS if f != k]) & use)
        va = np.flatnonzero(fold == k)
        m = lgb.train(params, ds.subset(tr), num_boost_round=ROUNDS, valid_sets=[ds.subset(va)],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
        pred[va] = predict_chunked([m], X, va)
        models.append(m)
        info.append({"fold": k, "best_iteration": int(m.best_iteration), "train_rows": int(len(tr)),
                     "valid_logloss": round(float(m.best_score["valid_0"]["binary_logloss"]), 5)})
        logger.info("[%s] fold %d: %s", tag, k, info[-1])
    other = np.flatnonzero(~np.isin(fold, TRAIN_FOLDS))
    if len(other):
        pred[other] = predict_chunked(models, X, other)
    return pred, models, info


def importance(models: list, feats: list, top: int = 25) -> list:
    """Return the top features by mean gain over the fold models."""
    g = np.mean([m.feature_importance("gain") for m in models], axis=0)
    order = np.argsort(-g)[:top]
    tot = g.sum()
    return [[feats[i], round(float(g[i] / tot * 100), 2)] for i in order]


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the weighted isotonic (non-decreasing) regression of y (pool-adjacent-violators)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi), wts.append(wi), cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v, ww, c = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] = (vals[-1] * wts[-1] + v * ww) / (wts[-1] + ww)
            wts[-1] += ww
            cnt[-1] += c
    return np.repeat(vals, cnt)


def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 2000) -> dict:
    """Return an isotonic calibration map fitted on OOF predictions: PAV over equal-count bins of p (x = bin mean p)."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(n_bins, len(order)))
    x = np.array([p[b].mean() for b in bins])
    yy = np.array([y[b].mean() for b in bins])
    w = np.array([len(b) for b in bins], dtype=float)
    return {"x": x.tolist(), "y": pav(yy, w).tolist()}


def apply_isotonic(cal: dict, p: np.ndarray) -> np.ndarray:
    """Return calibrated probabilities."""
    return np.interp(p, cal["x"], cal["y"]).astype(np.float32)


# ------------------------------------------------------------------ group aggregates of p1
def p1_aggregates(df: pd.DataFrame, p1: np.ndarray) -> None:
    """Add the stage-2 group aggregates of p1 (within the S1 and within the query) to ``df``."""
    s1, qid = df["s1_id"].to_numpy(), df["query_id"].to_numpy()
    df["p1"] = p1
    mx, sec = group_top2(s1, p1)
    r = group_rank(s1, p1)
    sec = np.where(np.isfinite(sec), sec, 0.0)
    df["p1_s1_max"], df["p1_s1_second"] = mx.astype(np.float32), sec.astype(np.float32)
    df["p1_s1_sum"] = pd.Series(p1).groupby(s1).transform("sum").to_numpy().astype(np.float32)
    df["p1_s1_cnt05"] = pd.Series((p1 > 0.5).astype(np.float32)).groupby(s1).transform("sum").to_numpy()
    df["p1_s1_rank"] = r
    df["p1_minus_s1max"] = (p1 - mx).astype(np.float32)
    df["p1_s1_other_max"] = np.where(r == 1, sec, mx).astype(np.float32)
    qmx, qsec = group_top2(qid, p1)
    qr = group_rank(qid, p1)
    qo = np.where(qr == 1, np.where(np.isfinite(qsec), qsec, 0.0), qmx)
    df["p1_q_other_max"] = qo.astype(np.float32)
    df["p1_q_margin"] = (p1 - qo).astype(np.float32)
    df["p1_q_rank"] = qr


# ------------------------------------------------------------------ S1-level has-match table
def s1_text_stats(split: str) -> pd.DataFrame:
    """Return per S1 row: s1_name_freq (log1p of how often its name_core repeats in S1), s1_addr_missing."""
    t = pq.read_table(norm_path(split, 1), columns=["name_core", "addr_clean"]).to_pandas()
    freq = t["name_core"].map(t["name_core"].value_counts()).to_numpy()
    return pd.DataFrame({"s1_name_freq": np.log1p(freq).astype(np.float32),
                         "s1_addr_missing": (t["addr_clean"] == "").to_numpy().astype(np.float32)})


def hasmatch_table(df: pd.DataFrame, p2: np.ndarray, kept: np.ndarray, s1_ids: np.ndarray,
                   stats: pd.DataFrame) -> pd.DataFrame:
    """Return one row per S1 in ``s1_ids`` with the has-match features (S1 without pairs: NaN / 0)."""
    s1 = df["s1_id"].to_numpy()
    g = pd.DataFrame({"s1_id": s1, "p": p2, "kept": kept, "pk": np.where(kept, p2, np.nan),
                      "claim": df["is_claimant"].to_numpy(), "noconf": df["is_claimant"].to_numpy()
                      * (df["num_conflict"].to_numpy() == 0),
                      "noextra": df["is_claimant"].to_numpy() * ((df["extra_tokens_q"].to_numpy()
                                                                  + df["extra_tokens_s1"].to_numpy()) == 0)})
    agg = g.groupby("s1_id").agg(hm_all_max=("p", "max"), hm_all_sum=("p", "sum"), hm_n_pairs=("p", "size"),
                                 hm_cnt=("kept", "sum"), hm_sum=("pk", "sum"), hm_claimants=("claim", "sum"),
                                 hm_noconf=("noconf", "sum"), hm_noextra=("noextra", "sum"))
    k = g[g["kept"]]
    agg["hm_cnt05"] = (k["p"] > 0.5).groupby(k["s1_id"]).sum()
    r = group_rank(k["s1_id"].to_numpy(), k["p"].to_numpy())
    top = k.assign(r=r)
    agg["hm_max"] = top[top["r"] == 1].set_index("s1_id")["p"]
    agg["hm_second"] = top[top["r"] == 2].set_index("s1_id")["p"]
    best_rows = np.flatnonzero(kept)[r == 1]
    b = pd.DataFrame({"s1_id": s1[best_rows], "hm_best_name": df["name_tsort"].to_numpy()[best_rows],
                      "hm_best_addr": df["addr_tsort"].to_numpy()[best_rows],
                      "hm_best_v2": df["v2_score"].to_numpy()[best_rows],
                      "hm_best_twin": df["tw_combo_max"].to_numpy()[best_rows]}).set_index("s1_id")
    agg = agg.join(b)
    out = agg.reindex(s1_ids)
    for c in ("hm_cnt", "hm_sum", "hm_cnt05", "hm_n_pairs", "hm_claimants", "hm_noconf", "hm_noextra", "hm_all_sum"):
        out[c] = out[c].fillna(0)
    out["hm_second"] = out["hm_second"].fillna(0)
    out[["s1_name_freq", "s1_addr_missing"]] = stats.iloc[s1_ids].to_numpy()
    return out[HM_FEATURES].astype(np.float32).reset_index()


# ------------------------------------------------------------------ evaluation views
class View:
    """Evaluation of a pair-level prediction for a set of S1 entities (pairs outside the set are ignored)."""

    def __init__(self, df: pd.DataFrame, s1_ids: np.ndarray, n_true: np.ndarray):
        """Store the entity index of each pair (-1 outside), labels and true counts."""
        pos = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
        self.ent = pos.reindex(df["s1_id"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()
        self.label = df["label"].to_numpy()
        self.q_true = df["q_true_s1"].to_numpy()
        self.n_true = n_true
        self.n = len(s1_ids)

    def metrics(self, pred: np.ndarray, budget: bool = False) -> dict:
        """Return fold metrics (and the error budget) for a predicted-pair mask."""
        m = fold_metrics(self.ent, pred, self.label, self.n_true)
        if budget:
            m["error_budget"] = error_budget(self.ent, pred, self.label, self.q_true, self.n_true)
        return m


def tune_threshold(view: View, p: np.ndarray, kept: np.ndarray) -> dict:
    """Return the best (t_accept, t_keep) on a view for one-to-one kept pairs scored by p."""
    ent = np.where(kept, view.ent, -1)
    best = (-1.0, None)
    for ta in T_GRID:
        for dk in (0.0, 0.05, 0.1, 0.15, 0.2):
            tk = round(float(ta + dk), 3)
            f = view.metrics(threshold_decode(ent, p, ta, tk))["macro_f05"]
            if f > best[0]:
                best = (f, {"t_accept": float(ta), "t_keep": tk})
    return {"oof_f05": round(best[0], 5), **best[1]}


def tune_decoder(view: View, p: np.ndarray, kept: np.ndarray, h: np.ndarray = None) -> dict:
    """Return the best (temperature, miss) of the expected-F0.5 decoder on a view (optionally with h)."""
    ent = np.where(kept, view.ent, -1)
    best = (-1.0, None)
    for t in TEMP_GRID:
        for miss in MISS_GRID:
            f = view.metrics(expected_f05_decode(ent, p, view.n, h, t, miss))["macro_f05"]
            if f > best[0]:
                best = (f, {"temperature": t, "miss": miss})
    return {"oof_f05": round(best[0], 5), **best[1]}


def feature_dir(split: str) -> Path:
    """Return the folder holding the v3 pair tables of a split (local output, else an attached Kaggle input)."""
    local = V3_DIR / split
    if local.exists() and any(local.glob("pairs_*.parquet")):
        return local
    for p in sorted(Path("/kaggle/input").rglob(V3_DIR.name)):
        if (p / split).is_dir() and any((p / split).glob("pairs_*.parquet")):
            return p / split
    raise FileNotFoundError(f"no features_v3/{split}/pairs_*.parquet under {V3_DIR} or /kaggle/input")


def load_bench_tables(logger) -> tuple:
    """Return (v3 benchmark pairs, bench S1 table with n_true / country / fold)."""
    from .benchmark import BENCH_DIR
    parts = sorted(feature_dir("bench").glob("pairs_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if DEBUG_FRAC < 1:                               # local dry run: a hash slice of the S1 (groups stay whole)
        df = df[(df["s1_id"].to_numpy() % 1000) < DEBUG_FRAC * 1000].reset_index(drop=True)
    df = df.sort_values(["query_id", "s1_id"], kind="stable").reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet")
    gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id"])
    bs1["n_true"] = bs1["entity_id"].map(gt["s1_id"].value_counts()).fillna(0).astype(np.int64)
    if DEBUG_FRAC < 1:
        bs1 = bs1[(bs1["s1_id"].to_numpy() % 1000) < DEBUG_FRAC * 1000].reset_index(drop=True)
    logger.info("bench v3 pairs %d x %d from %d files; S1 %d", len(df), df.shape[1], len(parts), len(bs1))
    return df, bs1


def views_for(df: pd.DataFrame, bs1: pd.DataFrame, folds) -> dict:
    """Return {'all': View, <country>: View} for the S1 of the given folds."""
    out = {}
    m = bs1["fold"].isin(folds)
    out["all"] = View(df, bs1.loc[m, "s1_id"].to_numpy(), bs1.loc[m, "n_true"].to_numpy())
    for c in sorted(bs1["country"].unique()):
        mc = m & (bs1["country"] == c)
        out[c] = View(df, bs1.loc[mc, "s1_id"].to_numpy(), bs1.loc[mc, "n_true"].to_numpy())
    return out


# ------------------------------------------------------------------ train stage
def run_train(logger) -> dict:
    """Train / evaluate every model on the benchmark and write the artefacts and the fold-0 report."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df, bs1 = load_bench_tables(logger)
    y, fold = df["label"].to_numpy().astype(np.float32), df["fold"].to_numpy()
    qid, s1 = df["query_id"].to_numpy(), df["s1_id"].to_numpy()
    rep = {"rows": int(len(df)), "neg_keep": NEG_KEEP}
    with StageTimer("model_te", logger):
        vocabs = add_te_bench(df, logger)
        for side, v in vocabs.items():
            write_parquet(v.assign(prior=v.attrs["prior"]), MODEL_DIR / f"te_vocab_{side}.parquet")
    f1 = feature_list(df.drop(columns=AGG1, errors="ignore"))
    rep["features_stage1"] = f1
    X = matrix(df, f1)
    with StageTimer("model_stage1", logger, rows=len(df), features=len(f1)):
        p1, m1, info1 = cv_train(X, y, fold, f1, LGB_PARAMS, "stage1", logger)
    rep["stage1_folds"], rep["importance_stage1"] = info1, importance(m1, f1)
    for i, m in enumerate(m1):
        m.save_model(str(MODEL_DIR / f"stage1_f{TRAIN_FOLDS[i]}.txt"))
    del X
    gc.collect()
    p1_aggregates(df, p1)
    f2 = f1 + AGG1
    X = matrix(df, f2)
    with StageTimer("model_stage2", logger, rows=len(df), features=len(f2)):
        p2raw, m2, info2 = cv_train(X, y, fold, f2, LGB_PARAMS, "stage2", logger)
    rep["stage2_folds"], rep["importance_stage2"] = info2, importance(m2, f2)
    for i, m in enumerate(m2):
        m.save_model(str(MODEL_DIR / f"stage2_f{TRAIN_FOLDS[i]}.txt"))
    tr = np.isin(fold, TRAIN_FOLDS)
    cal2 = fit_isotonic(p2raw[tr], y[tr])
    cal1 = fit_isotonic(p1[tr], y[tr])
    p2 = apply_isotonic(cal2, p2raw)
    p1c = apply_isotonic(cal1, p1)
    kept1 = best_per_query(qid, p1, s1)
    kept2 = best_per_query(qid, p2raw, s1)

    # ---------------- has-match model (one row per bench S1)
    stats = s1_text_stats("train")
    hm = hasmatch_table(df, p2, kept2, bs1["s1_id"].to_numpy(), stats)
    hm_y = (bs1["n_true"].to_numpy() > 0).astype(np.float32)
    hm_fold = bs1["fold"].to_numpy()
    Xh = hm[HM_FEATURES].to_numpy(np.float32)
    with StageTimer("model_hasmatch", logger, rows=len(hm)):
        hraw, mh, infoh = cv_train(Xh, hm_y, hm_fold, HM_FEATURES, HM_PARAMS, "hasmatch", logger)
    htr = np.isin(hm_fold, TRAIN_FOLDS)
    calh = fit_isotonic(hraw[htr], hm_y[htr])
    h_all = apply_isotonic(calh, hraw)
    rep["hasmatch_folds"], rep["importance_hasmatch"] = infoh, importance(mh, HM_FEATURES, 17)
    for i, m in enumerate(mh):
        m.save_model(str(MODEL_DIR / f"hasmatch_f{TRAIN_FOLDS[i]}.txt"))
    h_of = pd.Series(h_all, index=bs1["s1_id"].to_numpy())

    # ---------------- decoders: tune on OOF folds 1-4, report fold 0
    v14, v0 = views_for(df, bs1, TRAIN_FOLDS), views_for(df, bs1, [0])
    cfg = json.loads(find_file(CONFIG_FILE, LOG_DIR, "/kaggle/input").read_text(encoding="utf-8"))
    from .scorer_v2 import score_v2, select
    base = score_baseline(df)
    v2s = score_v2(df, cfg)
    preds = {"baseline": select(qid, s1, base, best_per_query(qid, base, s1), BASELINE["threshold"],
                                BASELINE["threshold"]),
             "v2": select(qid, s1, v2s, best_per_query(qid, v2s, s1), cfg["t_accept"], cfg["t_keep"])}
    thr1 = tune_threshold(v14["all"], p1, kept1)
    preds["stage1+threshold"] = _thr_pred(s1, p1, kept1, thr1)
    thr2 = tune_threshold(v14["all"], p2, kept2)
    preds["stage2+threshold"] = _thr_pred(s1, p2, kept2, thr2)
    dec1 = tune_decoder(v14["all"], p1c, kept1)
    preds["stage1+decoder"] = _dec_pred(s1, p1c, kept1, dec1)
    dec2 = tune_decoder(v14["all"], p2, kept2)
    preds["stage2+decoder"] = _dec_pred(s1, p2, kept2, dec2)
    h_pair = h_of.reindex(s1).to_numpy()
    dec2h = tune_decoder(v14["all"], p2, kept2, h=_h_view(v14["all"], s1, h_of))
    preds["stage2+decoder+hasmatch"] = _dec_pred(s1, p2, kept2, dec2h, h_of)
    rep["tuned"] = {"stage1_threshold": thr1, "stage2_threshold": thr2, "stage1_decoder": dec1,
                    "stage2_decoder": dec2, "stage2_decoder_hasmatch": dec2h}
    oof = {"stage1+threshold": thr1["oof_f05"], "stage2+threshold": thr2["oof_f05"],
           "stage1+decoder": dec1["oof_f05"], "stage2+decoder": dec2["oof_f05"],
           "stage2+decoder+hasmatch": dec2h["oof_f05"]}
    chosen = max(oof, key=oof.get)
    rep["oof_folds1_4"], rep["chosen"] = oof, chosen
    logger.info("OOF folds 1-4: %s -> chosen %s", oof, chosen)
    rep["fold0"] = {name: {c: v.metrics(pm, budget=(name in ("v2", chosen))) for c, v in v0.items()}
                    for name, pm in preds.items()}
    rep["fold0_singleton_part_before_after_hasmatch"] = {
        "stage2+decoder": rep["fold0"]["stage2+decoder"]["all"]["f05_singletons"],
        "stage2+decoder+hasmatch": rep["fold0"]["stage2+decoder+hasmatch"]["all"]["f05_singletons"]}
    f0 = fold == 0
    rep["p2_hist_fold0"] = np.histogram(p2[f0], bins=np.linspace(0, 1, 11))[0].tolist()
    rep["p2_hist_fold0_kept"] = np.histogram(p2[f0 & kept2], bins=np.linspace(0, 1, 11))[0].tolist()
    art = {"stage1_features": f1, "stage2_features": f2, "hasmatch_features": HM_FEATURES, "cal_p1": cal1,
           "cal_p2": cal2, "cal_h": calh, "tuned": rep["tuned"], "chosen": chosen, "folds": list(TRAIN_FOLDS),
           "p2_hist_fold0": rep["p2_hist_fold0"], "p2_hist_fold0_kept": rep["p2_hist_fold0_kept"],
           "fold0_chosen": rep["fold0"][chosen]["all"]}
    with open(MODEL_DIR / "artifacts.json", "w", encoding="utf-8", newline="") as f:
        json.dump(art, f)
    write_parquet(pd.DataFrame({"query_id": qid, "s1_id": s1, "fold": fold, "label": y.astype(np.int8),
                                "p1": p1, "p2": p2, "kept2": kept2, "h": h_pair.astype(np.float32)}),
                  MODEL_DIR / "bench_preds.parquet")
    with open(LOG_DIR / f"model_{TAG}_errors.txt", "w", encoding="utf-8", newline="") as f:
        f.write(error_examples(df, preds[chosen], p2, m2, f2, v0["all"], logger))
    with open(LOG_DIR / f"model_{TAG}_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1, default=float)
        f.write("\n")
    with open(LOG_DIR / f"model_{TAG}_report.md", "w", encoding="utf-8", newline="") as f:
        f.write(report_md(rep))
    logger.info("\n%s", report_md(rep))
    return rep


# ------------------------------------------------------------------ full retrain (all 5 bench folds)
def _fit_full(X: np.ndarray, y: np.ndarray, feats: list, params: dict, rounds: int, tag: str, logger):
    """Train one LightGBM model on every row for a fixed number of rounds and save it as <tag>_fall.txt."""
    with StageTimer(f"full_{tag}", logger, rows=len(y), features=len(feats), rounds=rounds):
        ds = lgb.Dataset(X, label=y, feature_name=feats, params={"max_bin": params["max_bin"], "verbose": -1})
        m = lgb.train(params, ds, num_boost_round=rounds)
    m.save_model(str(MODEL_DIR / f"{tag}_fall.txt"))
    return m


def run_train_full(logger) -> dict:
    """Retrain stage 1, stage 2 and has-match once on ALL 5 bench folds (rounds = 1.1 x mean CV best iteration).

    Stage-2 and has-match inputs are the CV run's out-of-fold p1 / p2 (bench_preds.parquet: OOF on folds 1-4, mean
    of the fold models on fold 0), i.e. the same kind of inputs the CV models saw. Calibration maps and decoder
    settings are copied from the CV artefacts (fitted on OOF predictions). The extra-word vocabulary for test is
    refitted on all 5 folds. Ends with an in-sample fold-0 pass through the submit scorer (a bug check only: the
    full models have seen fold 0, so this is not a gate)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    art = json.loads(find_file("artifacts.json", CV_MODEL_DIR, "/kaggle/input").read_text(encoding="utf-8"))
    rounds = {}
    for prefix in ("stage1", "stage2", "hasmatch"):
        its = [load_models(prefix, [k], CV_MODEL_DIR)[0].current_iteration() for k in art["folds"]]
        rounds[prefix] = int(round(FULL_ROUND_MULT * float(np.mean(its))))
        logger.info("%s CV best iterations %s -> full rounds %d", prefix, its, rounds[prefix])
    df, bs1 = load_bench_tables(logger)
    y, fold = df["label"].to_numpy().astype(np.float32), df["fold"].to_numpy()
    qid, s1 = df["query_id"].to_numpy(), df["s1_id"].to_numpy()
    with StageTimer("full_te", logger):
        add_te_bench(df, logger)                  # out-of-fold encodings for every training row (fold 0: fit on 1-4)
        fit_rows = np.flatnonzero(df["base_score"].to_numpy() >= CLAIM_MIN)
        for side in ("xq", "xs"):
            v = te_fit(df[side], df["label"].to_numpy(), fit_rows)
            write_parquet(v.assign(prior=v.attrs["prior"]), MODEL_DIR / f"te_vocab_{side}.parquet")
    X = matrix(df, art["stage1_features"])
    m1 = _fit_full(X, y, art["stage1_features"], LGB_PARAMS, rounds["stage1"], "stage1", logger)
    del X
    gc.collect()
    bp = pd.read_parquet(find_file("bench_preds.parquet", CV_MODEL_DIR, "/kaggle/input"),
                         columns=["query_id", "s1_id", "p1", "p2", "kept2"])
    if not (len(bp) == len(df) and np.array_equal(bp["query_id"].to_numpy(), qid)
            and np.array_equal(bp["s1_id"].to_numpy(), s1)):
        bp = df[["query_id", "s1_id"]].merge(bp, on=["query_id", "s1_id"], how="left", validate="one_to_one")
        if bp["p1"].isna().any():
            raise ValueError(f"bench_preds misses {int(bp['p1'].isna().sum())} bench pairs")
    p1_aggregates(df, bp["p1"].to_numpy(np.float32))
    X = matrix(df, art["stage2_features"])
    m2 = _fit_full(X, y, art["stage2_features"], LGB_PARAMS, rounds["stage2"], "stage2", logger)
    del X
    gc.collect()
    stats = s1_text_stats("train")
    hm = hasmatch_table(df, bp["p2"].to_numpy(np.float32), bp["kept2"].to_numpy(bool), bs1["s1_id"].to_numpy(), stats)
    hm_y = (bs1["n_true"].to_numpy() > 0).astype(np.float32)
    mh = _fit_full(hm[HM_FEATURES].to_numpy(np.float32), hm_y, HM_FEATURES, HM_PARAMS, rounds["hasmatch"],
                   "hasmatch", logger)
    art_full = dict(art, folds=["all"], full_rounds=rounds, cv_fold0_chosen=art["fold0_chosen"])
    with open(MODEL_DIR / "artifacts.json", "w", encoding="utf-8", newline="") as f:
        json.dump(art_full, f)
    rep = {"rounds": rounds, "cv_fold0_chosen": art["fold0_chosen"],
           "importance_stage1": importance([m1], art["stage1_features"]),
           "importance_stage2": importance([m2], art["stage2_features"])}
    # in-sample fold-0 check through the submit scorer (catches wiring bugs; expect >= the CV fold-0 score)
    keep_cols = [c for c in df.columns if c not in AGG1 and not c.startswith("te_")]
    d0 = df.loc[fold == 0, keep_cols].reset_index(drop=True)
    del df
    gc.collect()
    vocab = {s: pd.read_parquet(MODEL_DIR / f"te_vocab_{s}.parquet") for s in ("xq", "xs")}
    s1_0 = np.sort(bs1.loc[bs1["fold"] == 0, "s1_id"].to_numpy())
    r = score_partition(d0, art_full, ([m1], [m2], [mh]), vocab, stats, s1_0)
    rep["fold0_in_sample"] = {c: v.metrics(r["keep"]) for c, v in views_for(d0, bs1, [0]).items()}
    logger.info("full models: rounds %s; fold-0 IN-SAMPLE (bug check, not a gate): %s; CV fold 0: %s", rounds,
                {c: round(m["macro_f05"], 4) for c, m in rep["fold0_in_sample"].items()},
                round(art["fold0_chosen"]["macro_f05"], 4))
    with open(LOG_DIR / f"model_{MODEL_NAME}_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1, default=float)
        f.write("\n")
    return rep


def _thr_pred(s1, p, kept, cfg) -> np.ndarray:
    """Return the threshold-decoder mask over all pairs (entity = S1 id)."""
    ent = np.where(kept, pd.factorize(s1)[0], -1)
    return threshold_decode(ent, p, cfg["t_accept"], cfg["t_keep"])


def _h_view(view: View, s1: np.ndarray, h_of: pd.Series) -> np.ndarray:
    """Return h aligned with a view's entities (the entity order of the view)."""
    ids = np.empty(view.n, dtype=np.int64)
    m = view.ent >= 0
    ids[view.ent[m]] = s1[m]
    have = np.zeros(view.n, dtype=bool)
    have[view.ent[m]] = True
    out = np.full(view.n, np.nan)
    out[have] = h_of.reindex(ids[have]).to_numpy()
    return np.nan_to_num(out, nan=0.05)        # an S1 without any candidate pair: its h is irrelevant (no pairs)


def _dec_pred(s1, p, kept, cfg, h_of: pd.Series = None) -> np.ndarray:
    """Return the expected-F0.5 decoder mask over all pairs (entities = distinct S1 ids of the pairs)."""
    codes, uniq = pd.factorize(s1)
    ent = np.where(kept, codes, -1)
    h = None if h_of is None else h_of.reindex(uniq).fillna(0.05).to_numpy()
    return expected_f05_decode(ent, p, len(uniq), h, cfg["temperature"], cfg["miss"])


# ------------------------------------------------------------------ report
def error_examples(df, pred, p2, models, feats, view: View, logger, n: int = 15) -> str:
    """Return raw text of n fold-0 false positives and n false negatives with their top feature contributions."""
    rng = np.random.default_rng(SEED)
    in0 = view.ent >= 0
    fp = np.flatnonzero(in0 & pred & (df["label"].to_numpy() == 0))
    fn = np.flatnonzero(in0 & ~pred & (df["label"].to_numpy() == 1))
    rows = np.r_[rng.choice(fp, min(n, len(fp)), replace=False), rng.choice(fn, min(n, len(fn)), replace=False)]
    kinds = ["FP"] * min(n, len(fp)) + ["FN"] * min(n, len(fn))
    X = matrix(df.iloc[rows], feats)
    contrib = np.mean([m.predict(X, num_iteration=m.best_iteration, pred_contrib=True) for m in models], axis=0)
    cols = ["business_name", "business_address"]
    tabs = {s: pq.read_table(norm_path("train", s), columns=cols) for s in (1, 2, 3)}
    out = []
    for kind, r, c in zip(kinds, rows, contrib):
        q, s = int(df["query_id"].iat[r]), int(df["s1_id"].iat[r])
        src, row = divmod(q, QUERY_ID_MULT)
        a, b = tabs[1].slice(s, 1).to_pylist()[0], tabs[src].slice(row, 1).to_pylist()[0]
        top = np.argsort(-np.abs(c[:-1]))[:6]
        why = ", ".join(f"{feats[i]}={X[len(out), i]:.3g} ({c[i]:+.2f})" for i in top)
        tag = "decoy" if df["q_true_s1"].iat[r] < 0 else ("other S1" if kind == "FP" else "true")
        out.append(f"{kind} [{tag}] p2={p2[r]:.3f}\n  S1: {a['business_name']} | {a['business_address']}\n"
                   f"  S{src}: {b['business_name']} | {b['business_address']}\n  drivers: {why}\n")
    return "\n".join(out)


def report_md(rep: dict) -> str:
    """Render the fold-0 comparison table, error budget and feature importance as Markdown."""
    lines = [f"# Model {TAG} - fold 0 (gate)", "",
             "| model | scope | F0.5 | precision | recall | pred/S1 | % empty | singleton F0.5 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, scopes in rep["fold0"].items():
        for c, m in scopes.items():
            lines.append(f"| {name} | {c} | {m['macro_f05']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['pred_per_s1']:.2f} | {m['pct_empty']:.2f} | {m['f05_singletons']:.3f} |")
    lines += ["", f"OOF folds 1-4 (tuning): {rep['oof_folds1_4']}; chosen: **{rep['chosen']}**", "",
              "## Error budget (points of F0.5 lost, fold 0)", "",
              "| model | scope | singleton non-empty | FP decoy | FP other | FN blocking | FN scoring | total |",
              "|---|---|---|---|---|---|---|---|"]
    for name in ("v2", rep["chosen"]):
        for c, m in rep["fold0"][name].items():
            b = m["error_budget"]
            lines.append(f"| {name} | {c} | {b['singleton_nonempty']:.4f} | {b['fp_decoy']:.4f} | "
                         f"{b['fp_other']:.4f} | {b['fn_blocking']:.4f} | {b['fn_scoring']:.4f} | {b['total_lost']:.4f} |")
    lines += ["", "## Top-25 features (gain %)", "", "| stage 1 | stage 2 |", "|---|---|"]
    for a, b in zip(rep["importance_stage1"], rep["importance_stage2"]):
        lines.append(f"| {a[0]} {a[1]} | {b[0]} {b[1]} |")
    lines += ["", f"Has-match importance: {rep['importance_hasmatch']}", "",
              f"Tuned: {json.dumps(rep['tuned'])}", ""]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ submit stage
def load_models(prefix: str, folds, model_dir: Path = None) -> list:
    """Return the saved LightGBM fold models (fold "all" = the full-data model)."""
    model_dir = MODEL_DIR if model_dir is None else model_dir
    return [lgb.Booster(model_file=str(find_file(f"{prefix}_f{k}.txt", model_dir, "/kaggle/input"))) for k in folds]


CHOSEN_KEY = {"stage1+threshold": "stage1_threshold", "stage2+threshold": "stage2_threshold",
              "stage1+decoder": "stage1_decoder", "stage2+decoder": "stage2_decoder",
              "stage2+decoder+hasmatch": "stage2_decoder_hasmatch"}


def score_partition(df: pd.DataFrame, art: dict, models: tuple, vocab: dict, stats: pd.DataFrame,
                    s1_all: np.ndarray) -> dict:
    """Score one country's v3 pair table with the saved artefacts; return p1, p2, one-to-one mask, decoded mask, h.

    ``s1_all`` = every S1 row of the partition's country (S1 without pairs still get a has-match row).
    """
    m1, m2, mh = models
    chosen = art["chosen"]
    for side in ("xq", "xs"):
        v = vocab[side]
        for c, a in te_apply(pa.array(df[side].array), v, float(v["prior"].iat[0]), side[1], len(df)).items():
            df[c] = a
    X = matrix(df, art["stage1_features"])
    p1 = predict_chunked(m1, X)
    del X
    p1_aggregates(df, p1)
    X = matrix(df, art["stage2_features"])
    p2raw = predict_chunked(m2, X)
    del X
    gc.collect()
    p2 = apply_isotonic(art["cal_p2"], p2raw)
    qid, s1 = df["query_id"].to_numpy(), df["s1_id"].to_numpy()
    if chosen.startswith("stage1"):
        p, kept = (apply_isotonic(art["cal_p1"], p1) if "decoder" in chosen else p1), best_per_query(qid, p1, s1)
    else:
        p, kept = p2, best_per_query(qid, p2raw, s1)
    h_of = None
    if chosen.endswith("hasmatch"):
        hm = hasmatch_table(df, p2, best_per_query(qid, p2raw, s1), s1_all, stats)
        h = apply_isotonic(art["cal_h"], predict_chunked(mh, hm[HM_FEATURES].to_numpy(np.float32)))
        h_of = pd.Series(h, index=s1_all)
    cfg = art["tuned"][CHOSEN_KEY[chosen]]
    keep = _thr_pred(s1, p, kept, cfg) if "threshold" in chosen else _dec_pred(s1, p, kept, cfg, h_of)
    return {"p1": p1, "p2": p2, "kept": kept, "keep": keep, "h_of": h_of}


def run_submit(logger) -> dict:
    """Score the test v3 tables with the trained models and write / validate the submission files."""
    from .blocking import CAND_DIR
    from .finalize import id_lists, run_validator, test_s1_order
    from .io_utils import assert_no_cr_file, write_candidate_pairs, write_matching_results
    art = json.loads(find_file("artifacts.json", MODEL_DIR, "/kaggle/input").read_text(encoding="utf-8"))
    folds = art["folds"]
    m1, m2, mh = load_models("stage1", folds), load_models("stage2", folds), load_models("hasmatch", folds)
    vocab = {s: pd.read_parquet(find_file(f"te_vocab_{s}.parquet", MODEL_DIR, "/kaggle/input")) for s in ("xq", "xs")}
    lk = pd.read_parquet(find_file("lookup_s1.parquet", CAND_DIR / "test", "/kaggle/input"))
    tq = pd.read_parquet(find_file("queries.parquet", CAND_DIR / "test", "/kaggle/input"),
                         columns=["query_id", "entity_id", "country"])
    files = sorted(feature_dir("test").glob("pairs_*.parquet"))
    if SUBMIT_COUNTRY:
        files = [p for p in files if p.stem == f"pairs_{SUBMIT_COUNTRY}"]
        if not files:
            raise FileNotFoundError(f"no test pair table for SUBMIT_COUNTRY={SUBMIT_COUNTRY!r}")
    stats = s1_text_stats("test")
    chosen = art["chosen"]
    logger.info("model %s (folds %s); test partitions %s; chosen decoder %s; parts %s", MODEL_NAME, folds,
                [p.name for p in files], chosen, SUBMIT_PARTS)
    q_ent = pd.Series(tq["entity_id"].to_numpy(), index=tq["query_id"])
    s1_ent = lk["entity_id"].to_numpy()
    cands, matches = {}, {}
    rep = {"model": MODEL_NAME, "chosen": chosen, "by_country": {}, "p2_hist_fold0": art["p2_hist_fold0"],
           "p2_hist_fold0_kept": art["p2_hist_fold0_kept"]}
    for path in files:
        df = pd.read_parquet(path)
        country = lk["country"].iat[int(df["s1_id"].iat[0])]
        with StageTimer(f"submit_{MODEL_NAME}_{country}", logger, pairs=len(df)):
            s1_all = np.flatnonzero((lk["country"] == country).to_numpy())
            r = score_partition(df, art, (m1, m2, mh), vocab, stats, s1_all)
            p2, kept, keep, h_of = r["p2"], r["kept"], r["keep"], r["h_of"]
            qid, s1 = df["query_id"].to_numpy(), df["s1_id"].to_numpy()
            n_pred = np.bincount(np.searchsorted(s1_all, s1[keep]), minlength=len(s1_all))
            qc = tq.loc[tq["country"] == country, "query_id"].to_numpy()
            rep["by_country"][country] = {
                "s1": int(len(s1_all)), "pairs": int(len(df)), "pred_per_s1": round(float(n_pred.mean()), 3),
                "pct_empty": round(100 * float((n_pred == 0).mean()), 2),
                "pct_s2s3_assigned": round(100 * float(np.isin(qc, qid[keep]).mean()), 2),
                "p2_hist": np.histogram(p2, bins=np.linspace(0, 1, 11))[0].tolist(),
                "p2_hist_kept": np.histogram(p2[kept], bins=np.linspace(0, 1, 11))[0].tolist()}
            if h_of is not None:
                rep["by_country"][country]["mean_h"] = round(float(h_of.mean()), 4)
            logger.info("[%s] %s", country, rep["by_country"][country])
            if SAVE_PAIRS:          # per-pair test probabilities for post-processing / diagnostics
                xq = np.where(kept & ~keep, df["xq"].to_numpy(), "")
                write_parquet(pd.DataFrame({"query_id": qid, "s1_id": s1, "p2": p2, "kept": kept, "keep": keep,
                                            "xq": xq}), PAIRS_DIR / f"pairs_{country}.parquet")
                if h_of is not None:
                    write_parquet(pd.DataFrame({"s1_id": h_of.index.to_numpy(), "h": h_of.to_numpy(np.float32)}),
                                  PAIRS_DIR / f"h_{country}.parquet")
            te = pd.DataFrame({"query_id": qid, "s1_id": s1, "cheap": df["cheap"].to_numpy()})
            del df, r
            gc.collect()
            c_cands = id_lists(te, np.ones(len(te), dtype=bool), s1_ent, q_ent)
            c_matches = id_lists(te, keep, s1_ent, q_ent)
            del te
            if SUBMIT_PARTS:
                ents = s1_ent[s1_all]
                part = pd.DataFrame({"entity_id": ents, "country": country,
                                     "candidates": [",".join(c_cands.get(e, ())) for e in ents],
                                     "matches": [",".join(c_matches.get(e, ())) for e in ents]})
                write_parquet(part, PARTS_DIR / f"{country}.parquet")
                with open(PARTS_DIR / f"{country}_report.json", "w", encoding="utf-8", newline="") as f:
                    json.dump(rep["by_country"][country], f, indent=1, default=float)
                    f.write("\n")
                del part
            else:
                cands.update(c_cands)
                matches.update(c_matches)
            del c_cands, c_matches
        gc.collect()
    if SUBMIT_PARTS:
        with open(LOG_DIR / f"submit_{MODEL_NAME}_{SUBMIT_COUNTRY or 'all'}_report.json", "w", encoding="utf-8",
                  newline="") as f:
            json.dump(rep, f, indent=1, default=float)
            f.write("\n")
        logger.info("parts written to %s: %s", PARTS_DIR, sorted(p.name for p in PARTS_DIR.glob("*")))
        return rep
    order = test_s1_order()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_path, match_path = OUTPUT_DIR / "candidate_pairs.tsv", OUTPUT_DIR / "matching_results.tsv"
    rep["rows_candidate"] = write_candidate_pairs(cand_path, order, cands)
    rep["rows_matching"] = write_matching_results(match_path, order, matches)
    for p in (cand_path, match_path):
        assert_no_cr_file(p)
    rep["validator"] = run_validator(match_path, cand_path, logger).strip().splitlines()[-3:]
    with open(LOG_DIR / f"submit_{TAG}_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1, default=float)
        f.write("\n")
    logger.info("submit report: %s", json.dumps(rep, indent=1, default=float))
    return rep


def run_assemble(logger) -> dict:
    """Merge the per-country parts into <output>/<FINAL_SUBDIR or final>/*.tsv, check them and run the validator."""
    from .finalize import run_validator, test_s1_order
    from .io_utils import assert_no_cr_file, write_candidate_pairs, write_matching_results
    root = Path(os.environ["PARTS_DIR"]) if os.environ.get("PARTS_DIR") else PARTS_DIR
    files = sorted(root.rglob("*.parquet"))
    parts = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    logger.info("assemble: %d parts from %s (%s): %d S1 rows", len(files), root, [p.name for p in files], len(parts))
    if not parts["entity_id"].is_unique:
        raise ValueError("an S1 entity appears in more than one part")
    order = test_s1_order()
    missing = set(order) - set(parts["entity_id"])
    if missing:
        raise ValueError(f"{len(missing)} test S1 missing from the parts, e.g. {sorted(missing)[:5]}")
    split = lambda s: s.split(",") if s else []                                   # noqa: E731
    cands = dict(zip(parts["entity_id"], map(split, parts["candidates"])))
    matches = dict(zip(parts["entity_id"], map(split, parts["matches"])))
    not_sub = sum(1 for e, m in matches.items() if not set(m) <= set(cands[e]))
    if not_sub:
        raise ValueError(f"{not_sub} S1 have matches outside their candidates")
    n_match = parts["matches"].map(lambda s: len(split(s)))
    rep = {"parts": [p.name for p in files], "by_country": {}}
    for c, g in n_match.groupby(parts["country"]):
        rep["by_country"][c] = {"s1": int(len(g)), "pred_per_s1": round(float(g.mean()), 3),
                                "pct_empty": round(100 * float((g == 0).mean()), 2)}
    for p in root.rglob("*_report.json"):
        rep["by_country"].setdefault(p.stem.replace("_report", ""), {})["kernel"] = json.loads(p.read_text("utf-8"))
    out = OUTPUT_DIR / os.environ.get("FINAL_SUBDIR", "final")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty (never overwrite a finished submission; set FINAL_SUBDIR)")
    cand_path, match_path = out / "candidate_pairs.tsv", out / "matching_results.tsv"
    rep["rows_candidate"] = write_candidate_pairs(cand_path, order, cands)
    rep["rows_matching"] = write_matching_results(match_path, order, matches)
    for p in (cand_path, match_path):
        assert_no_cr_file(p)
    rep["validator"] = run_validator(match_path, cand_path, logger).strip().splitlines()[-3:]
    with open(out / "assemble_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1, default=float)
        f.write("\n")
    logger.info("assembled %s: %s", out, json.dumps({k: v for k, v in rep.items() if k != "by_country"}))
    return rep


def main() -> None:
    """Parse flags and run the requested stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "submit", "train_full", "assemble"], required=True)
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("model_lgb")
    run = {"train": run_train, "submit": run_submit, "train_full": run_train_full, "assemble": run_assemble}
    with StageTimer(f"model_lgb_{args.stage}", logger, threads=N_THREADS):
        run[args.stage](logger)


if __name__ == "__main__":
    main()
