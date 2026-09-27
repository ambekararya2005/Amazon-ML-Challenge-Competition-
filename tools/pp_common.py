"""Shared loaders for the v4 post-processing analyses (saved OOF / fold-0 predictions only; no retraining).

bench_preds.parquet (K12d, models_v4): query_id, s1_id, fold, label, p1, p2 (isotonic), kept2 (query argmax), h.
The decoded selection is recomputed with the chosen v4 decoder (stage2+decoder+hasmatch, T 1.0, miss 0.05).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.decoder import BETA2, expected_f05_decode, f05_per_entity  # noqa: E402
from src.io_utils import raw_parquet_path, read_parquet  # noqa: E402

DL = ROOT / "kaggle" / "runs" / "k12d-bench-v4" / "dl" / "output"
DEC = {"temperature": 1.0, "miss": 0.05}


def load_bench() -> tuple:
    """Return (pairs with p2 / kept2 / h / label / fold / ent index / sel = decoded mask, bench S1 table)."""
    bp = pd.read_parquet(DL / "models_v4" / "bench_preds.parquet")
    bs1 = pd.read_parquet(ROOT / "cache" / "bench" / "s1.parquet")
    gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id"])
    bs1["n_true"] = bs1["entity_id"].map(gt["s1_id"].value_counts()).fillna(0).astype(np.int64)
    bs1 = bs1.sort_values("s1_id").reset_index(drop=True)
    pos = pd.Series(np.arange(len(bs1)), index=bs1["s1_id"].to_numpy())
    bp["ent"] = pos.reindex(bp["s1_id"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()
    h = bp.groupby("ent")["h"].first().reindex(np.arange(len(bs1))).fillna(0.05).to_numpy()
    ent_k = np.where(bp["kept2"].to_numpy(), bp["ent"].to_numpy(), -1)
    bp["sel"] = expected_f05_decode(ent_k, bp["p2"].to_numpy(), len(bs1), h, DEC["temperature"], DEC["miss"])
    return bp, bs1


def per_entity_f(ent: np.ndarray, pred: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Return F0.5 per entity for a predicted-pair mask."""
    n = len(n_true)
    m = pred & (ent >= 0)
    k = np.bincount(ent[m], minlength=n).astype(float)
    tp = np.bincount(ent[m], weights=label[m].astype(float), minlength=n)
    return f05_per_entity(k, tp, n_true)


def scores(bp: pd.DataFrame, bs1: pd.DataFrame, pred: np.ndarray, folds) -> dict:
    """Return macro F0.5 (all / per country) and the singleton part for the S1 of ``folds``."""
    f = per_entity_f(bp["ent"].to_numpy(), pred, bp["label"].to_numpy(), bs1["n_true"].to_numpy())
    m = bs1["fold"].isin(folds).to_numpy()
    out = {"all": float(f[m].mean())}
    for c in sorted(bs1["country"].unique()):
        mc = m & (bs1["country"] == c).to_numpy()
        out[c] = float(f[mc].mean())
    single = m & (bs1["n_true"].to_numpy() == 0)
    out["singleton_part"] = float(f[single].mean())
    return out


__all__ = ["BETA2", "ROOT", "DL", "load_bench", "per_entity_f", "scores"]
