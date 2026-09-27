"""Step-0 data checks on the geo-dense benchmark (before modelling).

    python -m src.data_checks        # needs cache/bench/{pairs,s1,queries}.parquet -> logs/data_checks.json

a) matches per S1 per source (S2 count, S3 count) on the benchmark true pairs (and full train);
b) decoys accepted by rule scorer v2 (one-to-one, t_accept / t_keep): the most similar TRUE match of the same S1
   (the "sibling", by mean of name and address token_sort) vs the S1 itself - is a decoy a mutated copy of a true
   record or of the S1?
c) does a decoy come from the same source as its sibling (vs the share expected if the source were random)?
"""
import argparse
import json

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .blocking import QUERY_ID_MULT
from .config import LOG_DIR, add_path_args, set_seeds
from .decoy_features import main_number, num_relation
from .finalize import best_per_query
from .io_utils import raw_parquet_path, read_parquet
from .logging_utils import StageTimer, get_logger
from .pair_table import texts
from .scorer_v2 import CONFIG_FILE, find_file, score_v2, select

DECOY_SAMPLE = 30_000


def per_source_counts(bq: pd.DataFrame, s1_ids: np.ndarray) -> dict:
    """Return the distribution of true matches per S1 for each query source (S1 in ``s1_ids``)."""
    t = bq[np.isin(bq["true_s1"].to_numpy(), s1_ids)]
    src = t["query_id"].to_numpy() // QUERY_ID_MULT
    out = {}
    for s in (2, 3):
        c = pd.Series(t["true_s1"].to_numpy()[src == s]).value_counts().reindex(s1_ids, fill_value=0)
        out[f"S{s}"] = {int(k): int(v) for k, v in c.value_counts().sort_index().items()}
        out[f"S{s}_max"] = int(c.max())
    return out


def _sim(a, b) -> np.ndarray:
    """Return token_sort similarity (0-1) of aligned string arrays."""
    return process.cpdist(list(a), list(b), scorer=fuzz.token_sort_ratio, workers=-1) / 100


def decoy_siblings(pairs: pd.DataFrame, bq: pd.DataFrame, keep: np.ndarray, logger) -> dict:
    """Return the decoy-vs-sibling report for accepted pairs whose query matches nobody."""
    acc = pairs.loc[keep, ["query_id", "s1_id", "q_true_s1", "label"]]
    rep = {"accepted": int(len(acc)), "accepted_true": int(acc["label"].sum()),
           "accepted_false_matches_nobody": int(((acc["label"] == 0) & (acc["q_true_s1"] < 0)).sum()),
           "accepted_false_other_s1": int(((acc["label"] == 0) & (acc["q_true_s1"] >= 0)).sum())}
    dec = acc[(acc["label"] == 0).to_numpy() & (acc["q_true_s1"] < 0).to_numpy()]
    dec = dec.sample(n=min(DECOY_SAMPLE, len(dec)), random_state=42)
    sib = bq.loc[np.isin(bq["true_s1"].to_numpy(), dec["s1_id"].to_numpy()), ["query_id", "true_s1"]]
    m = dec.merge(sib.rename(columns={"query_id": "sib_id", "true_s1": "s1_id"}), on="s1_id")
    logger.info("decoys %d with %d (decoy, sibling) combos; %d decoys have no sibling",
                len(dec), len(m), len(dec) - m["query_id"].nunique())

    def tx(ids: np.ndarray, side: str) -> dict:
        """Return aligned name_core / addr_clean / numbers / source for query ids or S1 ids."""
        out = {k: np.empty(len(ids), dtype=object) for k in ("name_core", "addr_clean", "numbers")}
        if side == "s1":
            t = texts("train", 1, ids)
            return {**{k: t[k] for k in out}, "source": np.ones(len(ids), dtype=np.int64)}
        src = ids // QUERY_ID_MULT
        for s in (2, 3):
            idx = np.flatnonzero(src == s)
            if len(idx):
                t = texts("train", s, ids[idx] % QUERY_ID_MULT)
                for k in out:
                    out[k][idx] = t[k]
        return {**out, "source": src}

    d, sb, s1 = tx(m["query_id"].to_numpy(), "q"), tx(m["sib_id"].to_numpy(), "q"), tx(m["s1_id"].to_numpy(), "s1")
    m["n_ds"], m["a_ds"] = _sim(d["name_core"], sb["name_core"]), _sim(d["addr_clean"], sb["addr_clean"])
    m["n_d1"], m["a_d1"] = _sim(d["name_core"], s1["name_core"]), _sim(d["addr_clean"], s1["addr_clean"])
    m["n_s1"], m["a_s1"] = _sim(sb["name_core"], s1["name_core"]), _sim(sb["addr_clean"], s1["addr_clean"])
    m["num_eq_ds"] = [num_relation(main_number(a), main_number(b))[0] for a, b in zip(d["numbers"], sb["numbers"])]
    m["num_eq_d1"] = [num_relation(main_number(a), main_number(b))[0] for a, b in zip(d["numbers"], s1["numbers"])]
    m["exact_ds"] = (d["name_core"] == sb["name_core"]) & (d["addr_clean"] == sb["addr_clean"])
    m["src_d"], m["src_s"] = d["source"], sb["source"]
    m["combo"] = (m["n_ds"] + m["a_ds"]) / 2
    best = m.sort_values(["query_id", "combo"], ascending=[True, False]).drop_duplicates("query_id")
    med = lambda c: round(float(best[c].median()), 4)        # noqa: E731
    mean = lambda c: round(float(best[c].mean()), 4)         # noqa: E731
    rep["decoys_sampled"] = int(len(dec))
    rep["decoys_with_sibling"] = int(len(best))
    rep["median_name_decoy_sibling"], rep["median_addr_decoy_sibling"] = med("n_ds"), med("a_ds")
    rep["median_name_decoy_s1"], rep["median_addr_decoy_s1"] = med("n_d1"), med("a_d1")
    rep["median_name_sibling_s1"], rep["median_addr_sibling_s1"] = med("n_s1"), med("a_s1")
    rep["mean_name_decoy_sibling"], rep["mean_name_decoy_s1"] = mean("n_ds"), mean("n_d1")
    rep["mean_addr_decoy_sibling"], rep["mean_addr_decoy_s1"] = mean("a_ds"), mean("a_d1")
    rep["pct_closer_to_sibling_than_s1"] = round(100 * float(((best["n_ds"] + best["a_ds"])
                                                              > (best["n_d1"] + best["a_d1"])).mean()), 2)
    rep["pct_main_number_equal_sibling"] = round(100 * float(best["num_eq_ds"].mean()), 2)
    rep["pct_main_number_equal_s1"] = round(100 * float(best["num_eq_d1"].mean()), 2)
    rep["pct_identical_text_to_sibling"] = round(100 * float(best["exact_ds"].mean()), 2)
    # c) same source as the most similar sibling vs expected under a random source
    same = best["src_d"] == best["src_s"]
    exp = m.groupby("query_id").apply(lambda g: float((g["src_s"] == g["src_d"]).mean()), include_groups=False)
    rep["same_source_as_sibling"] = {}
    for s in (2, 3):
        b = best["src_d"] == s
        rep["same_source_as_sibling"][f"decoy_S{s}"] = {
            "decoys": int(b.sum()), "pct_same_source": round(100 * float(same[b].mean()), 2),
            "pct_expected_if_random": round(100 * float(exp.reindex(best.loc[b, "query_id"]).mean()), 2)}
    rep["same_source_as_sibling"]["all"] = round(100 * float(same.mean()), 2)
    return rep


def run(logger) -> dict:
    """Run checks a-c and write logs/data_checks.json."""
    from .benchmark import BENCH_DIR
    pairs = pd.read_parquet(BENCH_DIR / "pairs.parquet").sort_values(["query_id", "s1_id"], kind="stable")
    pairs = pairs.reset_index(drop=True)
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet", columns=["s1_id", "country", "fold"])
    bq = pd.read_parquet(BENCH_DIR / "queries.parquet", columns=["query_id", "true_s1"])
    rep = {"a_matches_per_s1_per_source": {"bench": per_source_counts(bq, bs1["s1_id"].to_numpy())}}
    gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_source"])
    c = gt.groupby(["s1_id", "other_source"]).size().unstack(fill_value=0)
    rep["a_matches_per_s1_per_source"]["train_s1_with_matches"] = {
        s: {int(k): int(v) for k, v in c[s].value_counts().sort_index().items()} for s in c.columns}
    cfg = json.loads(find_file(CONFIG_FILE, LOG_DIR).read_text(encoding="utf-8"))
    s = score_v2(pairs, cfg)
    qid, s1 = pairs["query_id"].to_numpy(), pairs["s1_id"].to_numpy()
    keep = select(qid, s1, s, best_per_query(qid, s, s1), cfg["t_accept"], cfg["t_keep"])
    rep["b_c_decoys_v2"] = decoy_siblings(pairs, bq, keep, logger)
    with open(LOG_DIR / "data_checks.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=2)
        f.write("\n")
    logger.info("data checks: %s", json.dumps(rep, indent=1))
    return rep


def main() -> None:
    """Parse flags and run the checks."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.parse_args()
    set_seeds()
    logger = get_logger("data_checks")
    with StageTimer("data_checks", logger):
        run(logger)


if __name__ == "__main__":
    main()
