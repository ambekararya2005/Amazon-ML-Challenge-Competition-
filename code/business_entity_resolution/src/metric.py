"""Official challenge metric: macro-averaged F0.5 per Source 1 entity.

f05_entity and macro_f05 follow docs plan Section 8 exactly. ``true_map`` defines
the evaluated S1 entities; an entity missing from ``pred_map`` counts as an
empty prediction. Singletons are included (empty vs empty = 1.0).
"""
from typing import Iterable, Mapping, Optional

import pandas as pd


def f05_entity(pred: Iterable[str], true: Iterable[str]) -> float:
    """Return F0.5 for one S1 entity given its predicted and true match ids."""
    pred, true = set(pred), set(true)
    if not true:                      # singleton
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    t = len(pred & true)
    return 1.25 * t / (len(pred) + 0.25 * len(true))


def macro_f05(pred_map: Mapping[str, Iterable[str]], true_map: Mapping[str, Iterable[str]]) -> float:
    """Return the macro F0.5 over every S1 entity in ``true_map`` (missing predictions = empty)."""
    return sum(f05_entity(pred_map.get(k, []), v) for k, v in true_map.items()) / len(true_map)


def entity_scores(pred_map: Mapping[str, Iterable[str]], true_map: Mapping[str, Iterable[str]],
                  country_map: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Return one row per evaluated S1 entity with F0.5, precision, recall and set sizes.

    Conventions for the diagnostic precision/recall (F0.5 itself is unaffected):
    an empty prediction has precision 1.0 (no false merges); an entity with no
    true matches has recall 1.0 (nothing to find).
    """
    rows = []
    for s1, true in true_map.items():
        pred, true = set(pred_map.get(s1, [])), set(true)
        t, k, n = len(pred & true), len(pred), len(true)
        rows.append((s1, f05_entity(pred, true), t / k if k else 1.0, t / n if n else 1.0, k, n))
    df = pd.DataFrame(rows, columns=["s1_id", "f05", "precision", "recall", "n_pred", "n_true"])
    if country_map is not None:
        df["country"] = df["s1_id"].map(country_map)
        if df["country"].isna().any():
            raise ValueError("country_map is missing some evaluated S1 ids")
    return df


def _summarise(df: pd.DataFrame) -> dict:
    """Return aggregate metrics for a block of per-entity scores."""
    single = df["n_true"] == 0
    return {
        "n_entities": int(len(df)),
        "macro_f05": float(df["f05"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "singleton_rate": float(single.mean()),
        "empty_pred_rate_singletons": float((df.loc[single, "n_pred"] == 0).mean()) if single.any() else float("nan"),
        "empty_pred_rate_non_singletons": float((df.loc[~single, "n_pred"] == 0).mean()) if (~single).any() else float("nan"),
        "mean_pred_size": float(df["n_pred"].mean()),
    }


def f05_breakdown(pred_map: Mapping[str, Iterable[str]], true_map: Mapping[str, Iterable[str]],
                  country_map: Mapping[str, str]) -> dict:
    """Return metrics overall and per country: macro F0.5, mean precision/recall and empty-prediction rates.

    Countries are whatever labels ``country_map`` contains (open set).
    Output: {"overall": {...}, "by_country": {label: {...}}}.
    """
    df = entity_scores(pred_map, true_map, country_map)
    return {
        "overall": _summarise(df),
        "by_country": {c: _summarise(g) for c, g in df.groupby("country", sort=True)},
    }


def format_breakdown(breakdown: dict) -> str:
    """Render a breakdown dict as a compact fixed-width text table."""
    cols = ["n_entities", "macro_f05", "mean_precision", "mean_recall",
            "empty_pred_rate_singletons", "empty_pred_rate_non_singletons"]
    labels = ["entities", "F0.5", "precision", "recall", "empty|single", "empty|non-single"]
    head = f"{'scope':<10}" + "".join(f"{lab:>18}" for lab in labels)
    lines = [head]
    for name, m in [("overall", breakdown["overall"]), *breakdown["by_country"].items()]:
        lines.append(f"{name:<10}" + "".join(
            f"{m[c]:>18d}" if c == "n_entities" else f"{m[c]:>18.4f}" for c in cols))
    return "\n".join(lines)
