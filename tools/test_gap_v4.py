"""Test-gap diagnostic (report only): fold-0 US / India vs test countries on v4 probability statistics.

Fold 0: bench_preds (OOF-style fold-0 predictions of the CV models). Test: the per-country kernel reports
(submit_parts_v4/<country>_report.json: p2 histogram over all pairs and over each query's argmax pair, predictions
per S1, % empty, % S2/S3 assigned) and, where saved, per-pair test probabilities (test_pairs_v4/pairs_<c>.parquet)
for the S1 best-p distribution and S1 examples.
    python tools/test_gap_v4.py <parts dir> [<test pairs dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pp_common import ROOT, load_bench

QMULT = 10 ** 8
DEC = np.linspace(0, 1, 11)


def unc_share(hist) -> float:
    """Return the share of mass in the 0.3-0.7 deciles of a 10-bin histogram."""
    h = np.asarray(hist, float)
    return float(h[3:7].sum() / h.sum())


def s1_best_stats(best: np.ndarray) -> dict:
    """Return quantiles / shares of the S1 best p (S1 without any candidate count as 0)."""
    return {"q10": round(float(np.quantile(best, 0.1)), 3), "median": round(float(np.median(best)), 3),
            "share_lt_0.3": round(float((best < 0.3).mean()), 4),
            "share_0.3_0.7": round(float(((best >= 0.3) & (best < 0.7)).mean()), 4),
            "share_ge_0.9": round(float((best >= 0.9).mean()), 4)}


parts_dir = Path(sys.argv[1])
pairs_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
rep = {"fold0": {}, "test": {}}

bp, bs1 = load_bench()
f0 = bs1["fold"].to_numpy() == 0
for c in sorted(bs1["country"].unique()):
    ids = np.flatnonzero(f0 & (bs1["country"] == c).to_numpy())
    m = np.isin(bp["ent"].to_numpy(), ids)
    d = bp[m]
    best = d.groupby("ent")["p2"].max().reindex(ids).fillna(0).to_numpy()
    n_pred = d[d["sel"]].groupby("ent").size().reindex(ids).fillna(0).to_numpy()
    rep["fold0"][c] = {"s1": int(len(ids)), "pred_per_s1": round(float(n_pred.mean()), 3),
                       "pct_empty": round(100 * float((n_pred == 0).mean()), 2),
                       "pct_queries_assigned": round(100 * float(d.loc[d["sel"], "query_id"].nunique()
                                                                 / d.loc[d["kept2"], "query_id"].nunique()), 2),
                       "query_best_p_0.3_0.7": round(unc_share(np.histogram(d.loc[d["kept2"], "p2"], DEC)[0]), 4),
                       "p2_hist_kept": np.histogram(d.loc[d["kept2"], "p2"], DEC)[0].tolist(),
                       "s1_best_p": s1_best_stats(best), "mean_h": round(float(d.groupby("ent")["h"].first().mean()), 4)}

for p in sorted(parts_dir.glob("*_report.json")):
    c = p.stem.replace("_report", "")
    r = json.loads(p.read_text("utf-8"))
    rep["test"][c] = {"s1": r["s1"], "pred_per_s1": r["pred_per_s1"], "pct_empty": r["pct_empty"],
                      "pct_s2s3_assigned": r["pct_s2s3_assigned"],
                      "query_best_p_0.3_0.7": round(unc_share(r["p2_hist_kept"]), 4),
                      "p2_hist_kept": r["p2_hist_kept"], "mean_h": r.get("mean_h")}

examples = []
if pairs_dir is not None:
    for p in sorted(pairs_dir.glob("pairs_*.parquet")):
        c = p.stem.replace("pairs_", "")
        d = pd.read_parquet(p)
        s1_all = pd.read_parquet(pairs_dir / f"h_{c}.parquet")["s1_id"].to_numpy()
        best = d.groupby("s1_id")["p2"].max().reindex(s1_all).fillna(0).to_numpy()
        rep["test"].setdefault(c, {})["s1_best_p"] = s1_best_stats(best)
        if c == "France":
            rng = np.random.default_rng(42)
            pick = rng.choice(np.unique(d["s1_id"]), 15, replace=False)
            s1t = pq.read_table(ROOT / "cache" / "norm" / "test_s1.parquet",
                                columns=["entity_id", "business_name", "business_address"])
            qt = {s: pq.read_table(ROOT / "cache" / "norm" / f"test_s{s}.parquet",
                                   columns=["entity_id", "business_name", "business_address"]) for s in (2, 3)}
            for s in pick:
                a = s1t.slice(int(s), 1).to_pylist()[0]
                g = d[(d["s1_id"] == s) & d["kept"]].sort_values("p2", ascending=False).head(8)
                lines = [f"S1 {a['entity_id']}: {a['business_name']} | {a['business_address']}"]
                for q, p2, k in zip(g["query_id"], g["p2"], g["keep"]):
                    src, row = divmod(int(q), QMULT)
                    b = qt[src].slice(row, 1).to_pylist()[0]
                    lines.append(f"   {'SEL' if k else 'rej'} p={p2:.3f} {b['entity_id']}: {b['business_name']} | "
                                 f"{b['business_address']}")
                examples.append("\n".join(lines))

print(json.dumps(rep, indent=1))
with open(ROOT / "logs" / "test_gap_v4.json", "w", encoding="utf-8", newline="") as f:
    json.dump(rep, f, indent=1)
    f.write("\n")
if examples:
    with open(ROOT / "logs" / "test_gap_france_examples.txt", "w", encoding="utf-8", newline="") as f:
        f.write("\n\n".join(examples) + "\n")
    print("\n\n".join(examples))
