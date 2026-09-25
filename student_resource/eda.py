"""
Amazon ML Challenge 2026 - Business Entity Resolution
EDA script: run from inside the student_resource/ folder.

    python eda.py                # full dataset
    python eda.py --sample 5     # also print 5 example groups per country (default 8)

Writes everything to eda_report.txt (and prints it). Paste that file back.
Needs only pandas + numpy.
"""
import argparse, csv, gc, os, re, random, sys, time, unicodedata
from collections import Counter
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset")
ap.add_argument("--sample", type=int, default=8)
args = ap.parse_args()
random.seed(42)

OUT = open("eda_report.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

def read(path):
    # dtype=str + no NA conversion so empty fields stay "" ; QUOTE_NONE so stray quotes don't merge rows
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, on_bad_lines="warn")

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()

def pct(a, b): return f"{100*a/max(b,1):.2f}%"

t0 = time.time()
p("="*70); p("EDA REPORT"); p("="*70)

# ---------------- 1. Files ----------------
# Test frames are summarised on load and then freed to keep peak memory low.
src, test_sizes, test_preview = {}, {}, {}
for split in ["train", "test"]:
    for k in [1, 2, 3]:
        path = f"{args.root}/{split}/{split}_source{k}.tsv"
        df = read(path)
        p(f"\n[{split} S{k}] {path}  size={os.path.getsize(path)/1e6:.1f}MB  rows={len(df):,}  cols={list(df.columns)}")
        p("  country:", dict(Counter(df["country"]).most_common(10)))
        for c in ["business_name", "business_address"]:
            L = df[c].str.len()
            p(f"  {c}: empty={pct((df[c].str.strip()=='').sum(), len(df))}  "
              f"len mean={L.mean():.1f} p50={L.median():.0f} p95={L.quantile(.95):.0f} max={L.max()}")
        p("  duplicate entity_id:", df["entity_id"].duplicated().sum(),
          "| exact duplicate (name,address) rows:", df.duplicated(["business_name","business_address"]).sum())
        p("  sample rows:")
        for _, r in df.sample(min(3, len(df)), random_state=1).iterrows():
            p(f"    {r.entity_id} | {r.business_name} | {r.business_address} | {r.country}")
        if split == "train":
            src[(split, k)] = df
        else:
            test_sizes[k] = len(df)
            fr = df[~df.country.isin(["US","India"])]
            test_preview[k] = (len(fr), fr.head(4))
            del df, fr; gc.collect()

# ---------------- 2. Ground truth ----------------
gt = read(f"{args.root}/train/train_ground_truth.tsv")
p(f"\n[GROUND TRUTH] rows={len(gt):,} cols={list(gt.columns)}")
s1ids = set(src[("train",1)]["entity_id"])
p("  GT rows covering train S1:", pct(len(set(gt.source1_entity_id) & s1ids), len(s1ids)),
  "| duplicate S1 rows in GT:", gt.source1_entity_id.duplicated().sum())

gt["lst"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i] if x else [])
gt["n"] = gt["lst"].str.len()
gt["n2"] = gt["lst"].apply(lambda l: sum(i.startswith("S2-") for i in l))
gt["n3"] = gt["n"] - gt["n2"]
p("  SINGLETON RATE (no matches):", pct((gt.n==0).sum(), len(gt)))
p("  matches per S1 distribution:", dict(sorted(Counter(gt.n.clip(upper=10)).items())), "(10 = 10+)")
p("  mean matches (non-singletons):", round(gt.loc[gt.n>0,"n"].mean(), 3), "| max:", gt.n.max())
p("  S2 matches per S1:", dict(sorted(Counter(gt.n2.clip(upper=6)).items())))
p("  S3 matches per S1:", dict(sorted(Counter(gt.n3.clip(upper=6)).items())))

pairs = gt[["source1_entity_id","lst"]].explode("lst").dropna()
pairs.columns = ["s1","other"]
p(f"  total positive pairs: {len(pairs):,}")

# KEY CHECK: does any S2/S3 record belong to more than one S1?
multi = pairs["other"].value_counts()
p("  S2/S3 IDs linked to >1 S1 entity:", (multi>1).sum(), "(0 => one-to-one constraint holds)")

for k in [2,3]:
    ids = set(src[("train",k)]["entity_id"])
    used = set(pairs.other[pairs.other.str.startswith(f"S{k}-")])
    p(f"  S{k}: {pct(len(used), len(ids))} of records are matched to some S1 "
      f"(rest are unmatched noise/distractors); GT ids missing from file: {len(used-ids)}")
    del ids, used
del s1ids, multi; gc.collect()

# ---------------- 3. Pair-level properties ----------------
allrec = pd.concat([src[("train",k)] for k in [1,2,3]]).set_index("entity_id")
samp = pairs.sample(min(200000, len(pairs)), random_state=0)
a = allrec.loc[samp.s1.values]; b = allrec.loc[samp.other.values]
p("\n[POSITIVE PAIRS, sample of", len(samp), "]")
p("  same country label:", pct((a.country.values==b.country.values).sum(), len(samp)))

def jacc(x, y):
    x, y = set(norm(x)), set(norm(y))
    return len(x&y)/max(len(x|y),1)
def nums(s): return set(re.findall(r"\d+", s))
nj = np.array([jacc(x,y) for x,y in zip(a.business_name.values, b.business_name.values)])
aj = np.array([jacc(x,y) for x,y in zip(a.business_address.values, b.business_address.values)])
exact = np.array([" ".join(norm(x))==" ".join(norm(y)) for x,y in zip(a.business_name.values, b.business_name.values)])
p("  exact normalized name equal:", pct(exact.sum(), len(samp)))
p("  name token-Jaccard quantiles p10/p25/p50/p75:", np.round(np.quantile(nj,[.1,.25,.5,.75]),3))
p("  addr token-Jaccard quantiles p10/p25/p50/p75:", np.round(np.quantile(aj,[.1,.25,.5,.75]),3))
p("  name Jaccard == 0 (no shared name token):", pct((nj==0).sum(), len(samp)))
na = [nums(x) for x in a.business_address.values]; nb = [nums(y) for y in b.business_address.values]
both = sum(1 for x,y in zip(na,nb) if x and y)
share = sum(1 for x,y in zip(na,nb) if x & y)
conflict = sum(1 for x,y in zip(na,nb) if x and y and not (x & y))
p(f"  addr both have numbers: {pct(both,len(samp))} | share >=1 number: {pct(share,len(samp))} | "
  f"both have numbers but share none: {pct(conflict,len(samp))}")
for ctry in sorted(set(a.country)):
    m = a.country.values==ctry
    p(f"  [{ctry}] n={m.sum()} name-J p50={np.median(nj[m]):.3f} addr-J p50={np.median(aj[m]):.3f}")

# ---------------- 4. Hard-negative signal: repeated names in S1 (chains/branches) ----------------
s1 = src[("train",1)].copy()
s1["nn"] = s1.business_name.apply(lambda x: " ".join(norm(x)))
vc = s1.nn.value_counts()
p("\n[S1 NAME REPETITION] S1 records whose normalized name appears >1 times in S1:",
  pct(vc[vc>1].sum(), len(s1)), "| top:", list(vc.head(8).items()))

# ---------------- 5. Token vocab (abbreviation hints) ----------------
tok = Counter()
for s in src[("train",2)].business_address.sample(min(200000,len(src[("train",2)])), random_state=0):
    tok.update(norm(s))
p("\n[TOP ADDRESS TOKENS S2]", tok.most_common(40))
tokn = Counter()
for s in src[("train",1)].business_name.sample(min(200000,len(s1)), random_state=0):
    tokn.update(norm(s))
p("[TOP NAME TOKENS S1]", tokn.most_common(40))

# ---------------- 6. Example groups ----------------
p("\n[EXAMPLE MATCH GROUPS]")
g2 = gt[gt.n>0].merge(s1[["entity_id","country"]], left_on="source1_entity_id", right_on="entity_id")
for ctry in sorted(g2.country.unique()):
    p(f"\n--- {ctry} ---")
    for _, r in g2[g2.country==ctry].sample(min(args.sample, (g2.country==ctry).sum()), random_state=3).iterrows():
        x = allrec.loc[r.source1_entity_id]
        p(f"  {r.source1_entity_id}: {x.business_name} | {x.business_address}")
        for o in r.lst[:5]:
            y = allrec.loc[o]; p(f"     -> {o}: {y.business_name} | {y.business_address}")
p("\n[EXAMPLE SINGLETONS]")
for sid in gt[gt.n==0].source1_entity_id.sample(min(5,(gt.n==0).sum()), random_state=4):
    x = allrec.loc[sid]; p(f"  {sid}: {x.business_name} | {x.business_address} | {x.country}")

# ---------------- 7. Test set ----------------
p("\n[TEST] S1/S2/S3 sizes:", [test_sizes[k] for k in [1,2,3]],
  "| train sizes:", [len(src[("train",k)]) for k in [1,2,3]])
for k in [1,2,3]:
    n, head = test_preview[k]
    p(f"  test S{k} non-US/India rows: {n:,}")
    for _, r in head.iterrows():
        p(f"    {r.entity_id} | {r.business_name} | {r.business_address} | {r.country}")

p(f"\nDone in {time.time()-t0:.0f}s")
OUT.close()
