"""Independent audit of a submission, using only entity_id strings and the raw Parquet data.

Checks: cross-country pairs, raw-text similarity of predicted pairs vs 200k true train pairs,
coverage of S2/S3, predictions-per-S1 distribution vs the train ground truth, empty predictions,
and 15 random S1 examples per country. Writes logs/audit_sub1.json and logs/audit_sub1_examples.txt.
Used for the submission #1 audit (2026-09-26).

    python tools/audit_submission.py        # from the repo root; needs cache/raw (stage 0)
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "cache" / "raw"
OUT = REPO / "logs"
SUBMISSION = REPO / "output" / "matching_results.tsv"
rng = np.random.default_rng(42)
TSV = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
rep = {}


def log(*a):
    """Print and flush."""
    print(*a, flush=True)


def raw(split, name, cols=None):
    """Read a raw cached table (cache/raw/<split>_<name>.parquet)."""
    return pd.read_parquet(RAW / f"{split}_{name}.parquet", columns=cols)


def sims(a_name, b_name, a_addr, b_addr):
    """Return token_set_ratio of names and of addresses, and whether the addresses share a number."""
    n = process.cpdist(a_name, b_name, scorer=fuzz.token_set_ratio, processor=utils.default_process, workers=-1)
    a = process.cpdist(a_addr, b_addr, scorer=fuzz.token_set_ratio, processor=utils.default_process, workers=-1)
    num = np.array([bool({x.lstrip("0") or "0" for x in re.findall(r"\d+", p)} &
                         {x.lstrip("0") or "0" for x in re.findall(r"\d+", q)})
                    for p, q in zip(a_addr, b_addr)])
    return n, a, num


def summarize(df):
    """Return median similarities, shared-number rate and the clearly-wrong share for a block of pairs."""
    wrong = (df["name_sim"] < 40) & (df["addr_sim"] < 40)
    return {"pairs": len(df), "median_name_sim": float(df["name_sim"].median()),
            "median_addr_sim": float(df["addr_sim"].median()), "pct_share_number": round(100 * df["num"].mean(), 2),
            "pct_clearly_wrong": round(100 * wrong.mean(), 2),
            "pct_name_lt_60": round(100 * (df["name_sim"] < 60).mean(), 2)}


# ---- load submission + raw test
m = pd.read_csv(SUBMISSION, **TSV)
s1 = raw("test", "source1")
assert len(m) == len(s1) and set(m["source1_entity_id"]) == set(s1["entity_id"]), "S1 rows mismatch"
m = m.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id", how="left")
m["n_pred"] = np.where(m["matched_entity_ids"] == "", 0, m["matched_entity_ids"].str.count(",") + 1)
pairs = (m.loc[m["n_pred"] > 0, ["source1_entity_id", "matched_entity_ids", "country"]]
         .assign(o=lambda d: d["matched_entity_ids"].str.split(",")).explode("o")
         .drop(columns="matched_entity_ids").rename(columns={"country": "c1"}))
log("pairs", len(pairs))

oc = pd.concat([raw("test", "source2", ["entity_id", "country"]), raw("test", "source3", ["entity_id", "country"])],
               ignore_index=True)
oc["country"] = oc["country"].astype("category")
pairs = pairs.merge(oc.rename(columns={"entity_id": "o", "country": "c2"}), on="o", how="left")
rep["pairs_total"] = len(pairs)
rep["pairs_other_id_not_in_test_S2S3"] = int(pairs["c2"].isna().sum())
rep["pct_country_differs"] = round(100 * float((pairs["c1"] != pairs["c2"].astype(str)).mean()), 4)
rep["pct_country_differs_by_s1_country"] = {c: round(100 * float((g["c1"] != g["c2"].astype(str)).mean()), 4)
                                            for c, g in pairs.groupby("c1")}
rep["duplicate_other_ids_across_s1"] = int(pairs["o"].duplicated().sum())
log("country check", rep["pct_country_differs"], rep["pct_country_differs_by_s1_country"])

# ---- 5 coverage + 7 singletons
tot = oc.groupby("country", observed=True).size()
assigned = pairs.drop_duplicates("o").groupby("c2", observed=True).size()
rep["coverage_pct_s2s3_assigned"] = {c: round(100 * float(assigned.get(c, 0)) / int(tot[c]), 2) for c in tot.index}
rep["coverage_all"] = round(100 * pairs["o"].nunique() / len(oc), 2)
rep["empty_pred_by_country"] = {c: {"s1": len(g), "empty": int((g["n_pred"] == 0).sum()),
                                    "pct_empty": round(100 * float((g["n_pred"] == 0).mean()), 2)}
                                for c, g in m.groupby("country")}
gt = raw("train", "ground_truth")
gt_n = np.where(gt["matched_entity_ids"] == "", 0, gt["matched_entity_ids"].str.count(",") + 1)
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1000]
lab = ["0", "1", "2", "3", "4", "5", "6", "7", "8-9", "10-11", "12+"]
rep["pred_per_s1_dist_pct"] = {c: (pd.cut(g["n_pred"], bins, right=False, labels=lab).value_counts(normalize=True)
                                   .reindex(lab) * 100).round(2).to_dict() for c, g in m.groupby("country")}
rep["train_gt_per_s1_dist_pct"] = (pd.cut(pd.Series(gt_n), bins, right=False, labels=lab)
                                   .value_counts(normalize=True).reindex(lab) * 100).round(2).to_dict()
rep["mean_pred_per_s1"] = {c: round(float(g["n_pred"].mean()), 3) for c, g in m.groupby("country")}
rep["train_gt_mean_per_s1"] = round(float(gt_n.mean()), 3)
log("coverage", rep["coverage_pct_s2s3_assigned"], "empty", rep["empty_pred_by_country"])
del oc, gt
# ---- 1+3 sample 100k pairs per country, raw text similarity
idx = np.concatenate([rng.choice(ix, size=min(100_000, len(ix)), replace=False)
                      for ix in pairs.groupby("c1").indices.values()])
samp = pairs.iloc[np.sort(idx)].reset_index(drop=True)
need = set(samp["o"])
txt = []
for s in ("source2", "source3"):
    t = raw("test", s, ["entity_id", "business_name", "business_address"])
    txt.append(t[t["entity_id"].isin(need)])
    del t
txt = pd.concat(txt).set_index("entity_id")
s1t = s1.set_index("entity_id")
a = s1t.loc[samp["source1_entity_id"]]
b = txt.loc[samp["o"]]
samp["name_sim"], samp["addr_sim"], samp["num"] = sims(a["business_name"].tolist(), b["business_name"].tolist(),
                                                        a["business_address"].tolist(), b["business_address"].tolist())
rep["test_pred_pairs_similarity"] = {c: summarize(g) for c, g in samp.groupby("c1")}

# true train pairs
tp = raw("train", "pairs", ["s1_id", "other_id"])
tp = tp.iloc[rng.choice(len(tp), size=200_000, replace=False)]
t1 = raw("train", "source1").set_index("entity_id")
to = []
for s in ("source2", "source3"):
    t = raw("train", s, ["entity_id", "business_name", "business_address"])
    to.append(t[t["entity_id"].isin(set(tp["other_id"]))])
    del t
to = pd.concat(to).set_index("entity_id")
a, b = t1.loc[tp["s1_id"]], to.loc[tp["other_id"]]
tp["c1"] = a["country"].to_numpy()
tp["name_sim"], tp["addr_sim"], tp["num"] = sims(a["business_name"].tolist(), b["business_name"].tolist(),
                                                  a["business_address"].tolist(), b["business_address"].tolist())
rep["train_true_pairs_similarity"] = {c: summarize(g) for c, g in tp.groupby("c1")}
log(json.dumps({k: rep[k] for k in ("test_pred_pairs_similarity", "train_true_pairs_similarity")}, indent=1))

# ---- 4 examples
lines = []
for c in ("France", "India", "US"):
    ids = m.loc[m["country"] == c, "source1_entity_id"].to_numpy()
    lines.append(f"\n========== {c}: 15 random test S1 ==========\n")
    for e in rng.choice(ids, size=15, replace=False):
        r = s1t.loc[e]
        preds = m.loc[m["source1_entity_id"] == e, "matched_entity_ids"].iloc[0]
        preds = preds.split(",") if preds else []
        lines.append(f"\nS1 {e}: {r['business_name']} | {r['business_address']}\n")
        if not preds:
            lines.append("    (no prediction)\n")
        missing = [p for p in preds if p not in txt.index]
        if missing:
            extra = []
            for s in ("source2", "source3"):
                t = raw("test", s, ["entity_id", "business_name", "business_address"])
                extra.append(t[t["entity_id"].isin(set(missing))])
            txt = pd.concat([txt, pd.concat(extra).set_index("entity_id")])
        for p in preds:
            q = txt.loc[p]
            ns = fuzz.token_set_ratio(r["business_name"], q["business_name"], processor=utils.default_process)
            ad = fuzz.token_set_ratio(r["business_address"], q["business_address"], processor=utils.default_process)
            lines.append(f"    -> {p} [name {ns:.0f} / addr {ad:.0f}]: {q['business_name']} | {q['business_address']}\n")
with open(OUT / "audit_sub1_examples.txt", "w", encoding="utf-8", newline="") as f:
    f.writelines(lines)
with open(OUT / "audit_sub1.json", "w", encoding="utf-8", newline="") as f:
    json.dump(rep, f, indent=2, default=float)
    f.write("\n")
log("done")
