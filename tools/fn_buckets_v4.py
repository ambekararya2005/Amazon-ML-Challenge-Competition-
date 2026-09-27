"""Split the v4 fold-0 "FN scoring/decoder" loss into buckets (saved OOF / fold-0 predictions only).

A missed true pair = true pair present in the candidates but not predicted:
  (a) argmax_decoder : the query's argmax S1 is the true S1, the decoder did not select it
  (b) argmax_elsewhere : the query's argmax is another S1 (one-to-one sent it elsewhere);
        b1 = that other S1 selected it (it became an FP there), b2 = nobody selected it
  (c) other : anything else (should be ~0)
Points follow decoder.error_budget: an S1's FN loss 1 - F_noFP is split blocking / scoring by count, and the scoring
part is split over the buckets by count of that S1's missed pairs.
"""
import json

import numpy as np
import pandas as pd

from pp_common import BETA2, ROOT, load_bench, scores

bp, bs1 = load_bench()
n = len(bs1)
ent, lab, sel, kept = (bp[c].to_numpy() for c in ("ent", "label", "sel", "kept2"))
n_true = bs1["n_true"].to_numpy()
qsel = pd.Series(sel).groupby(bp["query_id"].to_numpy()).transform("max").to_numpy()   # query selected anywhere

miss = (ent >= 0) & (lab == 1) & ~sel
bucket = np.full(len(bp), "", dtype=object)
bucket[miss & kept] = "a_argmax_decoder"
bucket[miss & ~kept & qsel] = "b1_argmax_elsewhere_selected"
bucket[miss & ~kept & ~qsel] = "b2_argmax_elsewhere_unselected"
bucket[miss & (bucket == "")] = "c_other"

m = sel & (ent >= 0)
k = np.bincount(ent[m], minlength=n).astype(float)
tp = np.bincount(ent[m], weights=lab[m].astype(float), minlength=n)
found = np.bincount(ent[ent >= 0], weights=lab[ent >= 0].astype(float), minlength=n)
single = n_true == 0
with np.errstate(divide="ignore", invalid="ignore"):
    f_nofp = np.where(tp > 0, (1 + BETA2) * tp / (tp + BETA2 * n_true), 0.0)
    loss_fn = np.where(single, 0.0, 1 - f_nofp)
    miss_block = np.maximum(n_true - found, 0)
    miss_score = np.maximum(found - tp, 0)
    score_loss = loss_fn * np.where(miss_block + miss_score > 0, miss_score / (miss_block + miss_score), 0.0)
f0 = bs1["fold"].to_numpy() == 0
rep = {"fold0": scores(bp, bs1, sel, [0]), "fn_scoring_total": {}, "buckets": {}}
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
for scope in ["all"] + sorted(bs1["country"].unique()):
    ms = f0 if scope == "all" else f0 & (bs1["country"] == scope).to_numpy()
    rep["fn_scoring_total"][scope] = round(float(score_loss[ms].sum() / ms.sum()), 5)
    for b in sorted(set(bucket[miss])):
        cnt = np.bincount(ent[bucket == b], minlength=n).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(miss_score > 0, cnt / miss_score, 0.0)
        pts = float((score_loss * share)[ms].sum() / ms.sum())
        rows = (bucket == b) & ms[np.maximum(ent, 0)] & (ent >= 0)
        d = rep["buckets"].setdefault(b, {})
        d[scope] = {"points": round(pts, 5), "pairs": int(rows.sum())}
        if scope == "all":
            d["p2_hist"] = dict(zip([f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(bins[:-1], bins[1:])],
                                    np.histogram(bp["p2"].to_numpy()[rows], bins=bins)[0].tolist()))
            d["p2_median"] = round(float(np.median(bp["p2"].to_numpy()[rows])), 3) if rows.any() else None
print(json.dumps(rep, indent=1))
with open(ROOT / "logs" / "fn_buckets_v4.json", "w", encoding="utf-8", newline="") as f:
    json.dump(rep, f, indent=1)
    f.write("\n")
