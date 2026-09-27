"""Cross-check two sets of per-country submission parts (e.g. v4 full retrain vs v4-safe) on test.

    python tools/compare_parts.py <parts dir A (reference)> <parts dir B> [out json]
Per country: predicted-pair agreement (both / A, both / B, both / union), predictions per S1, % empty,
% S2/S3 assigned and the p2 histogram of each query's argmax pair (decile shares), relative differences B vs A.
Sanity rule used for the final: agreement (both / union) >= 97% and every statistic within ~2% relative.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, B = Path(sys.argv[1]), Path(sys.argv[2])
out = {}
for pa in sorted(A.glob("*.parquet")):
    c = pa.stem
    a, b = pd.read_parquet(pa), pd.read_parquet(B / pa.name)
    m = a[["entity_id", "matches"]].merge(b[["entity_id", "matches"]], on="entity_id", suffixes=("_a", "_b"))
    both = na = nb = 0
    for x, y in zip(m["matches_a"], m["matches_b"]):
        sa, sb = set(x.split(",")) - {""}, set(y.split(",")) - {""}
        both += len(sa & sb)
        na += len(sa)
        nb += len(sb)
    ra = json.loads((A / f"{c}_report.json").read_text("utf-8"))
    rb = json.loads((B / f"{c}_report.json").read_text("utf-8"))
    ha = np.asarray(ra["p2_hist_kept"], float) / sum(ra["p2_hist_kept"])
    hb = np.asarray(rb["p2_hist_kept"], float) / sum(rb["p2_hist_kept"])
    rel = {k: round(100 * (rb[k] - ra[k]) / ra[k], 2) for k in ("pred_per_s1", "pct_empty", "pct_s2s3_assigned")}
    out[c] = {"s1_rows": int(len(m)), "pairs_a": na, "pairs_b": nb,
              "agree_of_a": round(100 * both / na, 2), "agree_of_b": round(100 * both / nb, 2),
              "agree_of_union": round(100 * both / (na + nb - both), 2),
              "a": {k: ra[k] for k in rel}, "b": {k: rb[k] for k in rel}, "rel_diff_pct": rel,
              "p2_kept_decile_share_a": ha.round(4).tolist(), "p2_kept_decile_share_b": hb.round(4).tolist(),
              "p2_kept_abs_diff_max_pp": round(100 * float(np.abs(ha - hb).max()), 3)}
    out[c]["sane"] = bool(out[c]["agree_of_union"] >= 97 and all(abs(v) <= 2 for v in rel.values())
                          and out[c]["p2_kept_abs_diff_max_pp"] <= 2)
print(json.dumps(out, indent=1))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w", encoding="utf-8", newline="") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
