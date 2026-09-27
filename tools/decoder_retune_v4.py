"""Step 3 check: re-tune the expected-F0.5 decoder on v4 OOF predictions (folds 1-4), gate on fold 0."""
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Arya Ambekar\Downloads\Amazon Ml Challenge Competition\code\business_entity_resolution")
from src.decoder import expected_f05_decode, fold_metrics  # noqa: E402
from src.io_utils import raw_parquet_path, read_parquet  # noqa: E402

ROOT = r"C:\Users\Arya Ambekar\Downloads\Amazon Ml Challenge Competition"
bp = pd.read_parquet(ROOT + r"\kaggle\runs\k12d-bench-v4\dl\output\models_v4\bench_preds.parquet")
bs1 = pd.read_parquet(ROOT + r"\cache\bench\s1.parquet")
gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id"])
bs1["n_true"] = bs1["entity_id"].map(gt["s1_id"].value_counts()).fillna(0).astype(np.int64)
h_of = bp.groupby("s1_id")["h"].first()


def view(folds, country=None):
    """Return (entity index per pair restricted to kept pairs, n_true, h per entity) for S1 of the folds."""
    m = bs1["fold"].isin(folds) & ((bs1["country"] == country) if country else True)
    ids = bs1.loc[m, "s1_id"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    ent = pos.reindex(bp["s1_id"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()
    ent = np.where(bp["kept2"].to_numpy(), ent, -1)
    h = h_of.reindex(ids).fillna(0.05).to_numpy()
    return ent, bs1.loc[m, "n_true"].to_numpy(), h


def h_temp(h, t):
    """Return sigmoid(logit(h) / t)."""
    q = np.clip(h, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(q / (1 - q)) / t))


p, label = bp["p2"].to_numpy(), bp["label"].to_numpy()
V14, V0 = view([1, 2, 3, 4]), view([0])
V0c = {c: view([0], c) for c in sorted(bs1["country"].unique())}


def score(v, t, miss, ht):
    """Return fold metrics of one decoder config on a view."""
    ent, n_true, h = v
    return fold_metrics(ent, expected_f05_decode(ent, p, len(n_true), h_temp(h, ht), t, miss), label, n_true)


t0 = time.time()
base = score(V0, 1.0, 0.05, 1.0)
print("reproduce fold0 (T=1, miss=0.05):", round(base["macro_f05"], 4), f"{time.time() - t0:.0f}s", flush=True)
res = []
for t, miss, ht in itertools.product((0.85, 1.0, 1.15), (0.0, 0.05, 0.1), (0.7, 1.0, 1.4)):
    f = score(V14, t, miss, ht)["macro_f05"]
    res.append((f, t, miss, ht))
    print(f"T={t} miss={miss} hT={ht}: OOF {f:.5f}", flush=True)
best = max(res)
print("best OOF", best)
f0 = {c: round(score(v, *best[1:])["macro_f05"], 4) for c, v in {"all": V0, **V0c}.items()}
b0 = {c: round(score(v, 1.0, 0.05, 1.0)["macro_f05"], 4) for c, v in {"all": V0, **V0c}.items()}
print("fold0 chosen", f0, "vs current", b0)
json.dump({"grid": res, "best": best, "fold0_best": f0, "fold0_current": b0},
          open(ROOT + r"\logs\decoder_retune_v4.json", "w", encoding="utf-8", newline=""), indent=1)
