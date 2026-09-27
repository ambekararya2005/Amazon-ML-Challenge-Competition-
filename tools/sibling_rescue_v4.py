"""Sibling rescue post-processing for v4 (saved OOF / fold-0 predictions only; no retraining).

For an S1 with >= 1 selected match, add an unselected claimant q (q's argmax S1 is this S1) if
    p2(q) >= a  AND  sim(q, s) >= b for some SELECTED match s of this S1 with a compatible street number
    AND q has no decoy-type extra words (vs the S1).
sim = min(token_sort(name_core), token_sort(addr_clean)) ("min") or their mean ("mean"); number compatible = either
side has no number, or the first (main) numbers are equal; decoy-type extra word = extra query word whose target
encoding P(not a true pair | word extra) >= d (vocabulary of the v4 CV run, fitted on folds 1-4).
Tuned on OOF folds 1-4, gated on fold 0 (keep only if >= +0.002 and the singleton part does not drop).
"""
import itertools
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from pp_common import DL, ROOT, load_bench, scores

QMULT = 10 ** 8
A_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)
B_GRID = (85, 90, 95)
D_GRID = (0.5, 0.7, 1.01)          # 1.01 = ignore the extra-word test
SIM_MODES = ("min", "mean")

bp, bs1 = load_bench()
feats = pd.concat([pd.read_parquet(p, columns=["query_id", "s1_id", "xq"])
                   for p in sorted((DL / "features_v3_v4" / "bench").glob("pairs_*.parquet"))], ignore_index=True)

sel, kept, ent = bp["sel"].to_numpy(), bp["kept2"].to_numpy(), bp["ent"].to_numpy()
has_sel = np.bincount(ent[sel & (ent >= 0)], minlength=len(bs1)) > 0
cand = kept & ~sel & (ent >= 0) & has_sel[np.maximum(ent, 0)] & (bp["p2"].to_numpy() >= min(A_GRID))
C = bp.loc[cand, ["query_id", "s1_id", "p2", "label"]].reset_index()
S = bp.loc[sel & np.isin(bp["s1_id"].to_numpy(), C["s1_id"].unique()), ["query_id", "s1_id"]]
print(f"rescue candidates {len(C)} (true {int(C['label'].sum())}); selected siblings {len(S)}", flush=True)

# query texts (norm train tables; query_id = source x 1e8 + row)
qids = np.unique(np.r_[C["query_id"].to_numpy(), S["query_id"].to_numpy()])
txt = {}
for src in (2, 3):
    rows = qids[qids // QMULT == src] % QMULT
    t = pq.read_table(ROOT / "cache" / "norm" / f"train_s{src}.parquet",
                      columns=["name_core", "addr_clean", "numbers"]).take(rows).to_pandas()
    t.index = rows + src * QMULT
    txt[src] = t
T = pd.concat(txt.values())

# extra-word decoy score of each candidate: max TE score over its extra query words
vocab = pd.read_parquet(DL / "models_v4" / "te_vocab_xq.parquet")
score_of = dict(zip(vocab["token"], vocab["score"]))
prior = float(vocab["prior"].iat[0])
C = C.merge(feats, on=["query_id", "s1_id"], how="left")
C["xq_max"] = C["xq"].fillna("").map(lambda s: max((score_of.get(w, prior) for w in s.split()), default=0.0))

# best compatible sibling similarity per candidate
sib = S.groupby("s1_id")["query_id"].apply(list).to_dict()


def num_compat(a: str, b: str) -> bool:
    """Return True if either side has no number or the main (first) numbers agree."""
    a, b = a.split(), b.split()
    return not a or not b or a[0] == b[0]


best = {"min": np.zeros(len(C)), "mean": np.zeros(len(C))}
for i, (q, s1) in enumerate(zip(C["query_id"].to_numpy(), C["s1_id"].to_numpy())):
    tq = T.loc[q]
    for s in sib.get(s1, ()):
        ts = T.loc[s]
        if not num_compat(tq["numbers"], ts["numbers"]):
            continue
        ns = fuzz.token_sort_ratio(tq["name_core"], ts["name_core"])
        ad = fuzz.token_sort_ratio(tq["addr_clean"], ts["addr_clean"])
        best["min"][i] = max(best["min"][i], min(ns, ad))
        best["mean"][i] = max(best["mean"][i], (ns + ad) / 2)
print("similarities done", flush=True)

base_tr, base0 = scores(bp, bs1, sel, [1, 2, 3, 4]), scores(bp, bs1, sel, [0])
C_idx, C_p, C_lab, C_x = C["index"].to_numpy(), C["p2"].to_numpy(), C["label"].to_numpy(), C["xq_max"].to_numpy()
res = []
for mode, a, b, d in itertools.product(SIM_MODES, A_GRID, B_GRID, D_GRID):
    add = (C_p >= a) & (best[mode] >= b) & (C_x < d)
    pred = sel.copy()
    pred[C_idx[add]] = True
    f = scores(bp, bs1, pred, [1, 2, 3, 4])["all"]
    res.append({"mode": mode, "a": a, "b": b, "d": d, "oof": f, "added": int(add.sum()), "tp_added": int(C_lab[add].sum())})
res.sort(key=lambda r: -r["oof"])
top = res[0]
add = (C_p >= top["a"]) & (best[top["mode"]] >= top["b"]) & (C_x < top["d"])
pred = sel.copy()
pred[C_idx[add]] = True
after0 = scores(bp, bs1, pred, [0])
f0m = bs1["fold"].to_numpy()[np.maximum(ent[C_idx], 0)] == 0
rep = {"oof_before": base_tr["all"], "best": top, "top10": res[:10], "fold0_before": base0, "fold0_after": after0,
       "fold0_delta": after0["all"] - base0["all"],
       "fold0_added": int((add & f0m).sum()), "fold0_tp_added": int((add & f0m & (C_lab == 1)).sum()),
       "fold0_fp_added": int((add & f0m & (C_lab == 0)).sum())}
rep["keep"] = bool(rep["fold0_delta"] >= 0.002 and after0["singleton_part"] >= base0["singleton_part"])
print(json.dumps(rep, indent=1, default=float))
with open(ROOT / "logs" / "sibling_rescue_v4.json", "w", encoding="utf-8", newline="") as f:
    json.dump(rep, f, indent=1, default=float)
    f.write("\n")
