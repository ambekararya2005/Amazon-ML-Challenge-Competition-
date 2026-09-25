"""Validation split: hold out 15% of train S1 entities, stratified by country x match-count bucket.

Writes cache/split.parquet (s1_id, fold in {"train", "val"}). Match-count
buckets are 0, 1, 2, 3, 4, 5, 6+. Country is used only as a stratification
label (open set of strings).

Usage (from code/business_entity_resolution/):
    python -m src.split            # skip if cache/split.parquet exists
    python -m src.split --force    # rebuild
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CACHE_DIR, SEED, set_seeds
from .io_utils import raw_parquet_path, read_parquet, write_parquet
from .logging_utils import StageTimer, get_logger

SPLIT_PATH = CACHE_DIR / "split.parquet"
VAL_FRACTION = 0.15
MAX_BUCKET = 6          # bucket 6 means "6 or more matches"
# stratification should make these gaps tiny; larger gaps signal a bug
MAX_SINGLETON_RATE_GAP = 0.002
MAX_MEAN_MATCHES_GAP = 0.01


def load_split(path=SPLIT_PATH) -> pd.DataFrame:
    """Return the cached split table (s1_id, fold)."""
    return read_parquet(path)


def entity_table() -> pd.DataFrame:
    """Return one row per train S1 entity: s1_id, country, n_matches, bucket."""
    s1 = read_parquet(raw_parquet_path("train", "source1"), columns=["entity_id", "country"])
    gt = read_parquet(raw_parquet_path("train", "ground_truth"))
    ids = gt["matched_entity_ids"]
    n = np.where(ids == "", 0, ids.str.count(",") + 1).astype(np.int16)
    gt = pd.DataFrame({"s1_id": gt["source1_entity_id"], "n_matches": n})
    df = s1.rename(columns={"entity_id": "s1_id"}).merge(gt, on="s1_id", how="left", validate="1:1")
    if df["n_matches"].isna().any():
        raise AssertionError("some train S1 entities have no ground-truth row")
    df["n_matches"] = df["n_matches"].astype(np.int16)
    df["bucket"] = df["n_matches"].clip(upper=MAX_BUCKET)
    return df


def make_split(df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> pd.Series:
    """Return a Series of fold labels ("train"/"val") aligned to ``df``, stratified by country x bucket."""
    strata = df["country"].astype(str) + "|" + df["bucket"].astype(str)
    _, val_idx = train_test_split(np.arange(len(df)), test_size=val_fraction,
                                  stratify=strata, random_state=seed, shuffle=True)
    fold = np.full(len(df), "train", dtype=object)
    fold[val_idx] = "val"
    return pd.Series(fold, index=df.index, name="fold").astype("str")


def fold_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return entity counts, singleton rate and mean matches per (country, fold) and overall per fold."""
    def agg(g: pd.DataFrame) -> pd.Series:
        """Summarise one group of entities."""
        nz = g.loc[g["n_matches"] > 0, "n_matches"]
        return pd.Series({"entities": len(g), "singleton_rate": (g["n_matches"] == 0).mean(),
                          "mean_matches": g["n_matches"].mean(), "mean_matches_nonsingleton": nz.mean()})

    per_country = df.groupby(["country", "fold"]).apply(agg, include_groups=False)
    overall = df.groupby("fold").apply(agg, include_groups=False)
    overall.index = pd.MultiIndex.from_product([["ALL"], overall.index], names=["country", "fold"])
    return pd.concat([per_country, overall])


def check_balance(stats: pd.DataFrame) -> None:
    """Raise if train and val differ by more than the tolerances in singleton rate or mean matches."""
    for country in stats.index.get_level_values("country").unique():
        tr, va = stats.loc[(country, "train")], stats.loc[(country, "val")]
        gap_s = abs(tr["singleton_rate"] - va["singleton_rate"])
        gap_m = abs(tr["mean_matches"] - va["mean_matches"])
        if gap_s > MAX_SINGLETON_RATE_GAP or gap_m > MAX_MEAN_MATCHES_GAP:
            raise AssertionError(f"{country}: train/val imbalance (singleton gap {gap_s:.4f}, mean gap {gap_m:.4f})")


def main() -> None:
    """Build (or load) the split, print the sanity table and check train/val balance."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if cache/split.parquet exists")
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("split")

    with StageTimer("split", logger):
        df = entity_table()
        if SPLIT_PATH.exists() and not args.force:
            logger.info("cached split at %s - loading (use --force to rebuild)", SPLIT_PATH)
            split = load_split()
            df = df.merge(split, on="s1_id", how="left", validate="1:1")
            if df["fold"].isna().any():
                raise AssertionError("cached split does not cover every train S1 entity; rerun with --force")
        else:
            df["fold"] = make_split(df)
            write_parquet(df[["s1_id", "fold"]], SPLIT_PATH)
            logger.info("wrote %s", SPLIT_PATH)

        stats = fold_stats(df)
        with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
            logger.info("split sanity (per country x fold):\n%s", stats.to_string())
        check_balance(stats)
        logger.info("balance check passed (singleton gap <= %.3f, mean-matches gap <= %.3f)",
                    MAX_SINGLETON_RATE_GAP, MAX_MEAN_MATCHES_GAP)


if __name__ == "__main__":
    main()
