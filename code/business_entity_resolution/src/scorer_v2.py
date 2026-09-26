"""Rule scorer v2 (decoy-aware): benchmark evaluation / grid search, and the test submission.

    python -m src.scorer_v2 --stage eval     # needs cache/bench/pairs.parquet -> logs/scorer_v2_report.{json,md},
                                             #   logs/scorer_v2_config.json, logs/decoy_examples.txt
    python -m src.scorer_v2 --stage submit   # test pair table + config -> output/{matching_results,candidate_pairs}.tsv

Score = w_name * name_tsort + w_addr * addr_tsort (name_tsort if an address is missing) + w_num * num_compatible
        - w_extra * min(extra_tokens_q + extra_tokens_s1, 3) / 3
        [- major_pen if the query's main number disagrees with the S1's other claimants]
num_conflict -> veto (score = -1) or - conflict_pen. One-to-one (each query -> its best S1), accept if
score >= t_accept; an S1 keeps its predictions only if its best accepted score >= t_keep (else empty).
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOG_DIR, OUTPUT_DIR, add_path_args, set_seeds
from .decoy_features import unmatched_tokens
from .finalize import best_per_query
from .io_utils import raw_parquet_path, read_parquet
from .logging_utils import StageTimer, get_logger

CONFIG_FILE = "scorer_v2_config.json"
BASELINE = {"weights": (0.3, 0.6, 0.1), "threshold": 0.64}
SCORE_COLS = ["query_id", "s1_id", "name_tsort", "addr_tsort", "num_compatible", "num_conflict",
              "extra_tokens_q", "extra_tokens_s1", "num_agree_major", "b_name_sim", "b_addr_sim", "b_num_match"]


# ------------------------------------------------------------------ scoring
def score_v2(df: pd.DataFrame, p: dict) -> np.ndarray:
    """Return the v2 rule score for every pair."""
    extra = np.minimum(df["extra_tokens_q"].to_numpy() + df["extra_tokens_s1"].to_numpy(), 3) / 3
    name = df["name_tsort"].to_numpy()
    addr = df["addr_tsort"].to_numpy()
    addr = np.where(np.isnan(addr), name, addr)          # missing address = neutral (name stands in)
    s = (p["w_name"] * name + p["w_addr"] * addr
         + p["w_num"] * df["num_compatible"].to_numpy() - p["w_extra"] * extra)
    s = s - p.get("major_pen", 0.0) * (df["num_agree_major"].to_numpy() == 0)
    conflict = df["num_conflict"].to_numpy() == 1
    return np.where(conflict, -1.0, s) if p.get("veto", True) else s - p.get("conflict_pen", 0.0) * conflict


def score_baseline(df: pd.DataFrame) -> np.ndarray:
    """Return the submission #1 rule score."""
    w = BASELINE["weights"]
    return w[0] * df["b_name_sim"].to_numpy() + w[1] * df["b_addr_sim"].to_numpy() + w[2] * df["b_num_match"].to_numpy()


def select(qid: np.ndarray, s1: np.ndarray, score: np.ndarray, best: np.ndarray, t_accept: float,
           t_keep: float) -> np.ndarray:
    """Return the predicted-pair mask: one-to-one best, score >= t_accept, S1 best accepted score >= t_keep."""
    keep = best & (score >= t_accept)
    if t_keep > t_accept:
        s1_max = pd.Series(np.where(keep, score, -np.inf)).groupby(s1).transform("max").to_numpy()
        keep &= s1_max >= t_keep
    return keep


# ------------------------------------------------------------------ metric
def metrics(ent: np.ndarray, is_true: np.ndarray, n_true: np.ndarray) -> dict:
    """Return macro F0.5 and diagnostics over entities 0..len(n_true)-1 given predicted pairs (entity, is_true)."""
    n = len(n_true)
    pred = np.bincount(ent, minlength=n)
    tp = np.bincount(ent, weights=is_true, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n_true == 0, (pred == 0).astype(float), np.where(pred == 0, 0.0, 1.25 * tp / (pred + 0.25 * n_true)))
        prec = np.where(pred == 0, 1.0, tp / np.maximum(pred, 1))
        rec = np.where(n_true == 0, 1.0, tp / np.maximum(n_true, 1))
    single = n_true == 0
    return {"macro_f05": float(f.mean()), "mean_precision": float(prec.mean()), "mean_recall": float(rec.mean()),
            "pred_per_s1": float(pred.mean()), "pct_empty": 100 * float((pred == 0).mean()),
            "singleton_share": float(single.mean()), "f05_singletons": float(f[single].mean()) if single.any() else 1.0,
            "f05_non_singletons": float(f[~single].mean()), "n_entities": int(n)}


class Fold:
    """Evaluation view of the pair table for a set of S1 entities (one-to-one uses ALL pairs of a query)."""

    def __init__(self, pairs: pd.DataFrame, ents: np.ndarray, n_true: np.ndarray):
        """Store the pairs (sorted by query), the entity index of each pair (-1 if outside) and true counts."""
        self.p = pairs
        pos = pd.Series(np.arange(len(ents)), index=ents)
        self.ent = pos.reindex(pairs["s1_id"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()
        self.n_true = n_true
        self.qid, self.s1 = pairs["query_id"].to_numpy(), pairs["s1_id"].to_numpy()
        self.true = pairs["label"].to_numpy().astype(float)

    def evaluate(self, score: np.ndarray, t_accept: float, t_keep: float, best=None) -> dict:
        """Return metrics for a score vector and thresholds (``best`` = precomputed one-to-one mask)."""
        best = best_per_query(self.qid, score, self.s1) if best is None else best
        keep = select(self.qid, self.s1, score, best, t_accept, t_keep) & (self.ent >= 0)
        return metrics(self.ent[keep], self.true[keep], self.n_true)


# ------------------------------------------------------------------ eval stage
def load_bench(logger):
    """Return (pairs, bench S1 table with n_true, bench queries) for the benchmark."""
    from .benchmark import BENCH_DIR
    pairs = pd.read_parquet(BENCH_DIR / "pairs.parquet")
    pairs = pairs.sort_values(["query_id", "s1_id"], kind="stable").reset_index(drop=True)
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet")
    gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id"])
    cnt = gt["s1_id"].value_counts()
    bs1["n_true"] = bs1["entity_id"].map(cnt).fillna(0).astype(np.int64)
    bq = pd.read_parquet(BENCH_DIR / "queries.parquet")
    logger.info("bench pairs %d, S1 %d, queries %d", len(pairs), len(bs1), len(bq))
    return pairs, bs1, bq


def fold_view(pairs, bs1, folds, country=None) -> Fold:
    """Return a Fold view for the S1 entities in ``folds`` (optionally one country)."""
    m = bs1["fold"].isin(folds)
    if country is not None:
        m &= bs1["country"] == country
    sub = bs1[m]
    return Fold(pairs, sub["s1_id"].to_numpy(), sub["n_true"].to_numpy())


def grid(view: Fold, logger) -> dict:
    """Staged grid search of v2 parameters and thresholds on a Fold view; return the best config."""
    best = (-1.0, None)
    t_grid = np.round(np.arange(0.40, 0.96, 0.025), 3)
    for w_name, w_num, w_extra in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2, 0.3)):
        w_addr = round(1 - w_name - w_num, 3)
        if w_addr < 0.1:
            continue
        p = {"w_name": w_name, "w_addr": w_addr, "w_num": w_num, "w_extra": w_extra, "veto": True}
        s = score_v2(view.p, p)
        b = best_per_query(view.qid, s, view.s1)
        for t in t_grid:
            m = view.evaluate(s, t, t, b)["macro_f05"]
            if m > best[0]:
                best = (m, {**p, "t_accept": float(t), "t_keep": float(t)})
    logger.info("grid stage A: %.5f %s", *best)
    p = dict(best[1])
    variants = [dict(p, veto=True, major_pen=mp) for mp in (0.0, 0.1, 0.2, 1.0)]
    variants += [dict(p, veto=False, conflict_pen=cp, major_pen=0.0) for cp in (0.1, 0.2, 0.3, 0.5)]
    for q in variants:
        s = score_v2(view.p, q)
        b = best_per_query(view.qid, s, view.s1)
        t0 = q["t_accept"]
        for ta in np.round(np.arange(t0 - 0.05, t0 + 0.0501, 0.01), 3):
            for dk in (0.0, 0.025, 0.05, 0.1, 0.15):
                m = view.evaluate(s, ta, ta + dk, b)["macro_f05"]
                if m > best[0]:
                    best = (m, {**q, "t_accept": float(ta), "t_keep": float(round(ta + dk, 3))})
    logger.info("grid final: %.5f %s", *best)
    return best[1]


def decoy_table(pairs: pd.DataFrame, folds_mask: np.ndarray) -> tuple:
    """Return (rates table, decoy pairs) on folds 1-4: true pairs vs decoys accepted by the baseline."""
    base = score_baseline(pairs)
    in_f = folds_mask
    true = pairs[(pairs["label"] == 1).to_numpy() & in_f]
    unm = pairs[(pairs["q_true_s1"] < 0).to_numpy()].copy()
    unm["bscore"] = base[(pairs["q_true_s1"] < 0).to_numpy()]
    unm = unm.sort_values(["query_id", "bscore"], ascending=[True, False]).drop_duplicates("query_id")
    decoy = unm[unm["bscore"] >= BASELINE["threshold"]]

    def rates(d):
        """Return the feature rates for a block of pairs."""
        extra = (d["extra_tokens_q"] + d["extra_tokens_s1"]) > 0
        return {"pairs": int(len(d)), "num_equal": round(100 * float(d["num_equal"].mean()), 2),
                "num_compatible": round(100 * float(d["num_compatible"].mean()), 2),
                "num_conflict": round(100 * float(d["num_conflict"].mean()), 2),
                "extra_tokens_gt0": round(100 * float(extra.mean()), 2),
                "both_conflict_or_extra": round(100 * float(((d["num_conflict"] == 1) | extra).mean()), 2)}
    return {"true_pairs": rates(true), "decoys_accepted_by_baseline": rates(decoy)}, decoy


def extra_words(pairs_sample: pd.DataFrame, n: int = 30) -> list:
    """Return the most frequent extra name words (both sides) for a sample of pairs."""
    from .pair_table import texts
    c = Counter()
    qid = pairs_sample["query_id"].to_numpy()
    for source in (2, 3):
        m = qid // 10 ** 8 == source
        if not m.any():
            continue
        qt = texts("train", source, qid[m] % 10 ** 8)
        st = texts("train", 1, pairs_sample["s1_id"].to_numpy()[m])
        for a, b in zip(qt["name_core"], st["name_core"]):
            at, bt = a.split(), b.split()
            c.update(unmatched_tokens(at, bt))
            c.update(unmatched_tokens(bt, at))
    return c.most_common(n)


def examples(decoy: pd.DataFrame, n: int = 20) -> str:
    """Return raw text of n decoy pairs accepted by the baseline (for eyeballing)."""
    import pyarrow.parquet as pq
    from .normalize import norm_path
    rng = np.random.default_rng(42)
    d = decoy.iloc[rng.choice(len(decoy), size=min(n, len(decoy)), replace=False)]
    cols = ["entity_id", "business_name", "business_address"]
    tabs = {s: pq.read_table(norm_path("train", s), columns=cols) for s in (1, 2, 3)}
    lines = []
    for q, s1, bs in zip(d["query_id"], d["s1_id"], d["bscore"]):
        src, row = divmod(int(q), 10 ** 8)
        a = tabs[1].slice(int(s1), 1).to_pylist()[0]
        b = tabs[src].slice(row, 1).to_pylist()[0]
        lines.append(f"baseline score {bs:.3f}\n  S1 {a['entity_id']}: {a['business_name']} | {a['business_address']}\n"
                     f"  S{src} {b['entity_id']} (matches nobody): {b['business_name']} | {b['business_address']}\n")
    return "\n".join(lines)


def run_eval(logger) -> dict:
    """Benchmark realism checks, decoy table, baseline on fold 0, v2 grid on folds 1-4, fold-0 report."""
    pairs, bs1, bq = load_bench(logger)
    rep = {}
    # blocking recall on the benchmark (true pairs of bench S1)
    tot_true = bs1.groupby("country")["n_true"].sum()
    found = pairs[pairs["label"] == 1].merge(bs1[["s1_id", "country"]], on="s1_id").groupby("country").size()
    rep["recall_at5_bench"] = {c: round(float(found.get(c, 0)) / int(tot_true[c]), 4) for c in tot_true.index}
    rep["recall_at5_bench"]["ALL"] = round(float(found.sum()) / int(tot_true.sum()), 4)

    s1_fold = pairs["fold"].to_numpy()
    f14 = np.isin(s1_fold, [1, 2, 3, 4])
    rep["decoy_table_folds1_4"], decoy = decoy_table(pairs, f14)
    rng = np.random.default_rng(42)
    tp = pairs[(pairs["label"] == 1).to_numpy() & f14]
    rep["extra_words_true"] = extra_words(tp.iloc[rng.choice(len(tp), size=min(30_000, len(tp)), replace=False)])
    rep["extra_words_decoy"] = extra_words(decoy.iloc[rng.choice(len(decoy), size=min(30_000, len(decoy)), replace=False)])
    with open(LOG_DIR / "decoy_examples.txt", "w", encoding="utf-8", newline="") as f:
        f.write(examples(decoy))

    views = {"fold0": fold_view(pairs, bs1, [0]), "folds1_4": fold_view(pairs, bs1, [1, 2, 3, 4])}
    for c in sorted(bs1["country"].unique()):
        views[f"fold0|{c}"] = fold_view(pairs, bs1, [0], c)
    base = score_baseline(pairs)
    rep["baseline"] = {k: v.evaluate(base, BASELINE["threshold"], BASELINE["threshold"]) for k, v in views.items()}
    logger.info("baseline fold0: %s", rep["baseline"]["fold0"])
    rep["stop_benchmark_lacks_decoys"] = rep["baseline"]["fold0"]["macro_f05"] >= 0.9
    if rep["stop_benchmark_lacks_decoys"]:
        logger.warning("BASELINE >= 0.9 ON FOLD 0 - benchmark still lacks decoys; stopping before tuning")
    else:
        cfg = grid(views["folds1_4"], logger)
        s = score_v2(pairs, cfg)
        rep["v2_config"] = cfg
        rep["v2"] = {k: v.evaluate(s, cfg["t_accept"], cfg["t_keep"]) for k, v in views.items()}
        rep["v2_minus_baseline_fold0"] = rep["v2"]["fold0"]["macro_f05"] - rep["baseline"]["fold0"]["macro_f05"]
        with open(LOG_DIR / CONFIG_FILE, "w", encoding="utf-8", newline="") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    with open(LOG_DIR / "scorer_v2_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=2, default=float)
        f.write("\n")
    logger.info("report: %s", json.dumps({k: v for k, v in rep.items() if not k.startswith("extra_words")},
                                         indent=1, default=float))
    return rep


# ------------------------------------------------------------------ submit stage
def find_file(name: str, *roots) -> Path:
    """Return the first existing ``name`` under the given roots (searched recursively)."""
    for root in roots:
        root = Path(root)
        if (root / name).exists():
            return root / name
        if root.exists():
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


def run_submit(logger) -> dict:
    """Apply the v2 config to the test pair table and write/validate the two submission files."""
    from .blocking import CAND_DIR
    from .finalize import id_lists, run_validator, test_s1_order
    from .io_utils import assert_no_cr_file, write_candidate_pairs, write_matching_results
    cfg = json.loads(find_file(CONFIG_FILE, LOG_DIR, "/kaggle/input").read_text(encoding="utf-8"))
    path = find_file("test_pairs.parquet", OUTPUT_DIR / "features", "/kaggle/input")
    te = pd.read_parquet(path, columns=SCORE_COLS + ["cheap"])
    logger.info("test pairs %d from %s; config %s", len(te), path, cfg)
    s = score_v2(te, cfg)
    qid, s1 = te["query_id"].to_numpy(), te["s1_id"].to_numpy()
    keep = select(qid, s1, s, best_per_query(qid, s, s1), cfg["t_accept"], cfg["t_keep"])
    lk = pd.read_parquet(CAND_DIR / "test" / "lookup_s1.parquet")
    tq = pd.read_parquet(CAND_DIR / "test" / "queries.parquet", columns=["query_id", "entity_id", "country"])
    q_ent = pd.Series(tq["entity_id"].to_numpy(), index=tq["query_id"])
    s1_ent = lk["entity_id"].to_numpy()
    cands = id_lists(te, np.ones(len(te), dtype=bool), s1_ent, q_ent)
    matches = id_lists(te, keep, s1_ent, q_ent)
    order = test_s1_order()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_path, match_path = OUTPUT_DIR / "candidate_pairs.tsv", OUTPUT_DIR / "matching_results.tsv"
    rep = {"config": cfg, "rows_candidate": write_candidate_pairs(cand_path, order, cands),
           "rows_matching": write_matching_results(match_path, order, matches)}
    for p in (cand_path, match_path):
        assert_no_cr_file(p)
    rep["validator"] = run_validator(match_path, cand_path, logger).strip().splitlines()[-3:]
    assigned = set(qid[keep])
    rep["by_country"] = {}
    country = lk.set_index("entity_id")["country"]
    for c, ents in country.groupby(country).groups.items():
        n_pred = np.array([len(matches.get(e, ())) for e in ents])
        qc = tq.loc[tq["country"] == c, "query_id"].to_numpy()
        rep["by_country"][c] = {"s1": int(len(ents)), "pred_per_s1": round(float(n_pred.mean()), 3),
                                "pct_empty": round(100 * float((n_pred == 0).mean()), 2),
                                "pct_s2s3_assigned": round(100 * float(np.isin(qc, list(assigned)).mean()), 2),
                                "cands_per_s1": round(float(np.mean([len(cands.get(e, ())) for e in ents])), 3)}
    with open(LOG_DIR / "submit_v2_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=2, default=float)
        f.write("\n")
    logger.info("submit report: %s", json.dumps(rep, indent=1, default=float))
    return rep


def main() -> None:
    """Parse flags and run the requested stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["eval", "submit"], required=True)
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("scorer_v2")
    with StageTimer(f"scorer_v2_{args.stage}", logger):
        run_eval(logger) if args.stage == "eval" else run_submit(logger)


if __name__ == "__main__":
    main()
