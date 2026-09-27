"""Blocking v4 (generic, no country logic): cross-script token dictionary before char TF-IDF, pass D (rarest name
token), adaptive top-k. Benchmark evaluation writes to a NEW cache path; the v1 tables are never overwritten.

    python -m src.blocking_v4 --split bench   # cache/bench_v4/{union,dictionary}.parquet + logs/blocking_v4_report.json
    python -m src.blocking_v4 --split test --country India   # <output>/cand_v4/test/topk_India.parquet

Cross-script dictionary: mined from train true pairs (for the benchmark: only pairs whose S1 is NOT in the benchmark)
where one record has non-Latin characters in its raw name / address (normalisation already applied anyascii).
Name tokens are aligned by position after removing tokens shared by both sides (only when the leftovers have equal
length); address components are aligned to the most similar unmatched component with the same token count
(rapidfuzz ratio >= ADDR_ALIGN_MIN). A token pair a -> b is kept if support >= DICT_MIN_SUPPORT and
support / aligned occurrences of a >= DICT_MIN_PRECISION. The dictionary maps tokens of every record (S1 and queries).

Pass D: for a query whose (mapped) name has a token with S1 document frequency <= D_MAX_DF, the S1 holding its rarest
such token are candidates; the top D_KEEP by name token_sort ratio (>= D_MIN_RATIO) are kept. (The region part of the
key is dropped: test queries of an unlabelled country have no learned region, and the benchmark must behave like test.)

Top-k: cheap = cosine_C + 0.2 x shared_keys_A; keep the top-5, or the top-8 when cheap(1st) - cheap(5th) < margin;
pass-D candidates are always kept.
"""
import argparse
import gc
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .blocking import QUERY_ID_MULT, QUERY_SOURCES, chunk_slices, fit_c_index, pass_c_chunk
from .config import CACHE_DIR, LOG_DIR, N_THREADS, OUTPUT_DIR, add_path_args, set_seeds
from .io_utils import raw_parquet_path, read_parquet, write_parquet
from .logging_utils import StageTimer, get_logger
from .normalize import norm_path

V4_BENCH_DIR = CACHE_DIR / "bench_v4"
TEST_V4_DIR = OUTPUT_DIR / "cand_v4" / "test"      # kept as Kaggle kernel output
MARGIN = 0.1               # chosen on the benchmark: India recall@k 0.9468 -> 0.9691 for +17% pairs
DICT_MIN_SUPPORT = 3
DICT_MIN_PRECISION = 0.6
ADDR_ALIGN_MIN = 50
D_MAX_DF = 50
D_KEEP = 3
D_MIN_RATIO = 50
W_A = 0.2
K_BASE, K_WIDE = 5, 8
MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2)       # reported; the chosen margin is the smallest reaching most of the gain
TEXT_COLS = ["business_name", "business_address", "name_core", "name_compact", "addr_clean"]


# ------------------------------------------------------------------ dictionary
def has_nonlatin(s: str) -> bool:
    """Return True if a raw string holds characters beyond Latin Extended-B (e.g. Devanagari)."""
    return any(ord(c) > 0x24F for c in s)


def _unmatched(a: list, b: list) -> tuple:
    """Return the tokens of a not in b and of b not in a (order kept)."""
    sa, sb = set(a), set(b)
    return [t for t in a if t not in sb], [t for t in b if t not in sa]


def align_names(a: str, b: str) -> list:
    """Return aligned (a_token, b_token) pairs from two name_core strings (leftovers of equal length only)."""
    u, v = _unmatched(a.split(), b.split())
    return list(zip(u, v)) if u and len(u) == len(v) else []


def align_addresses(a: str, b: str) -> list:
    """Return aligned token pairs from unmatched address components (best-ratio partner, same token count)."""
    ca = [c.strip() for c in a.split(",") if c.strip()]
    cb = [c.strip() for c in b.split(",") if c.strip()]
    ua, ub = _unmatched(ca, cb)
    out = []
    for c in ua:
        n = len(c.split())
        same = [d for d in ub if len(d.split()) == n]
        if not same:
            continue
        best = process.extractOne(c, same, scorer=fuzz.ratio, score_cutoff=ADDR_ALIGN_MIN)
        if best is not None:
            out.extend((x, y) for x, y in zip(c.split(), best[0].split()) if x != y)
    return out


def mine_dictionary(q_texts: dict, s_texts: dict, logger) -> dict:
    """Return {token: latin token} mined from aligned true pairs where exactly one side is non-Latin.

    ``q_texts`` / ``s_texts``: aligned lists (business_name, business_address, name_core, addr_clean) per true pair.
    """
    support, occ = Counter(), Counter()
    n_used = 0
    for qn, qa, qc, qac, sn, sa, sc, sac in zip(q_texts["business_name"], q_texts["business_address"],
                                                q_texts["name_core"], q_texts["addr_clean"],
                                                s_texts["business_name"], s_texts["business_address"],
                                                s_texts["name_core"], s_texts["addr_clean"]):
        q_nl, s_nl = has_nonlatin(qn) or has_nonlatin(qa), has_nonlatin(sn) or has_nonlatin(sa)
        if q_nl == s_nl:
            continue
        n_used += 1
        (nc, lc, na, la) = (qc, sc, qac, sac) if q_nl else (sc, qc, sac, qac)
        pairs = align_names(nc, lc) + align_addresses(na, la)
        for x, y in pairs:
            if x.isdigit() or len(x) < 2:
                continue
            support[(x, y)] += 1
            occ[x] += 1
    best = defaultdict(lambda: (0, ""))
    for (x, y), s in support.items():
        if s > best[x][0]:
            best[x] = (s, y)
    d = {x: y for x, (s, y) in best.items() if s >= DICT_MIN_SUPPORT and s / occ[x] >= DICT_MIN_PRECISION and x != y}
    logger.info("dictionary: %d cross-script pairs used, %d candidate tokens, %d kept; examples %s", n_used,
                len(occ), len(d), sorted(d.items(), key=lambda kv: -occ[kv[0]])[:25])
    return d


def map_text(s: str, d: dict) -> str:
    """Return s with every dictionary token replaced (commas kept)."""
    if not d or not s:
        return s
    return ", ".join(" ".join(d.get(t, t) for t in c.split()) for c in s.split(","))


def mapped_c_texts(t: dict, d: dict) -> list:
    """Return pass-C texts (name_compact + name_core + addr_clean) after the dictionary mapping."""
    out = []
    for comp, core, addr in zip(t["name_compact"], t["name_core"], t["addr_clean"]):
        core2 = map_text(core, d)
        comp2 = core2.replace(" ", "") if core2 != core else comp
        out.append(f"{comp2} {core2} {map_text(addr, d)}")
    return out


# ------------------------------------------------------------------ pass D
def name_df(cores: list, d: dict) -> Counter:
    """Return the document frequency of (mapped) name_core tokens over a list of S1 names."""
    df = Counter()
    for c in cores:
        df.update(set(map_text(c, d).split()))
    return df


def pass_d(q_ids: np.ndarray, q_cores: list, s1_ids: np.ndarray, s1_cores: list, d: dict) -> pd.DataFrame:
    """Return pass-D candidates (query_id, s1_id, scoreD) for one country."""
    mapped_s1 = [map_text(c, d) for c in s1_cores]
    post = defaultdict(list)
    for i, c in enumerate(mapped_s1):
        for t in set(c.split()):
            post[t].append(i)
    rows_q, rows_s, texts_q, texts_s = [], [], [], []
    for qi, c in zip(q_ids, q_cores):
        mc = map_text(c, d)
        toks = [t for t in set(mc.split()) if not t.isdigit() and 0 < len(post.get(t, ())) <= D_MAX_DF]
        if not toks:
            continue
        rare = min(toks, key=lambda t: (len(post[t]), t))
        for j in post[rare]:
            rows_q.append(qi), rows_s.append(j), texts_q.append(mc), texts_s.append(mapped_s1[j])
    if not rows_q:
        return pd.DataFrame({"query_id": np.empty(0, np.int64), "s1_id": np.empty(0, np.int32),
                             "scoreD": np.empty(0, np.float32)})
    sc = process.cpdist(texts_q, texts_s, scorer=fuzz.token_sort_ratio, workers=-1)
    t = pd.DataFrame({"query_id": np.asarray(rows_q, np.int64), "s1_id": s1_ids[np.asarray(rows_s)].astype(np.int32),
                      "scoreD": (sc / 100).astype(np.float32)})
    t = t[t["scoreD"] >= D_MIN_RATIO / 100]
    t = t.sort_values(["query_id", "scoreD", "s1_id"], ascending=[True, False, True], kind="stable")
    return t[t.groupby("query_id").cumcount() < D_KEEP].reset_index(drop=True)


# ------------------------------------------------------------------ top-k
def adaptive_top_k(u: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Return per query the top-5 by cheap (top-8 when cheap(1st) - cheap(5th) < margin) plus every pass-D pair."""
    cheap = u["scoreC"].to_numpy() + np.float32(W_A) * u["scoreA"].to_numpy()
    qid = u["query_id"].to_numpy()
    order = np.lexsort((u["s1_id"].to_numpy(), -cheap, qid))
    qs, cs = qid[order], cheap[order]
    start = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    size = np.diff(np.r_[start, len(qs)])
    rank = np.arange(len(qs)) - np.repeat(start, size)
    first = np.repeat(cs[start], size)
    fifth_pos = np.minimum(start + K_BASE - 1, np.r_[start[1:], len(qs)] - 1)
    fifth = np.repeat(cs[fifth_pos], size)
    k = np.where((np.repeat(size, size) > K_BASE) & (first - fifth < margin), K_WIDE, K_BASE)
    keep = (rank < k) | (u["scoreD"].to_numpy()[order] > 0)
    out = u.iloc[order[keep]].reset_index(drop=True)
    out["cheap"] = cheap[order[keep]]
    out["s1_rank"] = (rank[keep] + 1).astype(np.int16)
    return out


def recall_table(cands: pd.DataFrame, bq: pd.DataFrame, bs1: pd.DataFrame) -> dict:
    """Return recall of bench true pairs (queries whose true S1 is a bench S1) and candidates per query, by country."""
    t = bq[bq["true_s1_in_bench"]][["query_id", "true_s1"]]
    hit = t.merge(cands[["query_id", "s1_id"]], left_on=["query_id", "true_s1"], right_on=["query_id", "s1_id"],
                  how="left")["s1_id"].notna().to_numpy()
    country = t["true_s1"].map(bs1.set_index("s1_id")["country"]).to_numpy()
    out = {}
    for c in sorted(set(country)):
        m = country == c
        out[c] = round(float(hit[m].mean()), 4)
    out["ALL"] = round(float(hit.mean()), 4)
    out["cands_per_query"] = round(len(cands) / max(1, cands["query_id"].nunique()), 3)
    out["pairs"] = int(len(cands))
    return out


# ------------------------------------------------------------------ bench stage
def texts_of(split: str, source: int, rows: np.ndarray) -> dict:
    """Return {col: list} of TEXT_COLS for rows of a normalised file."""
    t = pq.read_table(norm_path(split, source), columns=TEXT_COLS).take(pa.array(rows))
    return {c: t[c].to_pylist() for c in TEXT_COLS}


def pair_texts(split: str, s1_rows: np.ndarray, q_ids: np.ndarray) -> tuple:
    """Return aligned text dicts for true pairs (S1 rows, query ids)."""
    q = {c: [None] * len(q_ids) for c in TEXT_COLS}
    src = q_ids // QUERY_ID_MULT
    for s in QUERY_SOURCES:
        idx = np.flatnonzero(src == s)
        t = texts_of(split, s, q_ids[idx] % QUERY_ID_MULT)
        for c in TEXT_COLS:
            for i, v in zip(idx, t[c]):
                q[c][i] = v
    return q, texts_of(split, 1, s1_rows)


def build_dictionary(exclude_s1_rows: np.ndarray, logger) -> dict:
    """Return the cross-script dictionary mined from train true pairs whose S1 row is not in ``exclude_s1_rows``."""
    gt = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
    s1_ent = pq.read_table(norm_path("train", 1), columns=["entity_id"])["entity_id"].to_pandas()
    s1_row = pd.Series(np.arange(len(s1_ent)), index=s1_ent.to_numpy())
    qrow = {}
    for s in QUERY_SOURCES:
        ents = pq.read_table(norm_path("train", s), columns=["entity_id"])["entity_id"].to_pandas()
        qrow.update(dict(zip(ents.to_numpy(), s * QUERY_ID_MULT + np.arange(len(ents), dtype=np.int64))))
    gt["s1_row"] = s1_row.reindex(gt["s1_id"].to_numpy()).to_numpy()
    gt["query_id"] = gt["other_id"].map(qrow).to_numpy()
    del qrow
    gc.collect()
    gt = gt[~np.isin(gt["s1_row"].to_numpy(), exclude_s1_rows)]
    with StageTimer("v4_dictionary", logger, pairs=len(gt)):
        qt, st = pair_texts("train", gt["s1_row"].to_numpy().astype(np.int64), gt["query_id"].to_numpy())
        d = mine_dictionary(qt, st, logger)
    return d


def run_bench(logger) -> dict:
    """Mine the dictionary (non-bench pairs), rerun pass C with it + pass D + adaptive top-k on the benchmark."""
    from .benchmark import BENCH_DIR
    bs1 = pd.read_parquet(BENCH_DIR / "s1.parquet", columns=["s1_id", "country", "fold"])
    bq = pd.read_parquet(BENCH_DIR / "queries.parquet", columns=["query_id", "country", "true_s1", "true_s1_in_bench"])
    rep = {}
    d = build_dictionary(bs1["s1_id"].to_numpy(), logger)
    V4_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.DataFrame({"token": list(d), "latin": list(d.values())}), V4_BENCH_DIR / "dictionary.parquet")
    rep["dictionary_size"] = len(d)

    old = pd.read_parquet(BENCH_DIR / "union.parquet", columns=["query_id", "s1_id", "scoreA", "scoreC"])
    parts = []
    for country, sc in bs1.groupby("country", sort=True):
        s1_pos = np.sort(sc["s1_id"].to_numpy())
        s1t = texts_of("train", 1, s1_pos)
        with StageTimer(f"v4_passC_{country}", logger, s1=len(s1_pos)):
            vec, b = fit_c_index(mapped_c_texts(s1t, d))
            qc = np.sort(bq.loc[bq["country"] == country, "query_id"].to_numpy())
            cs, ds = [], []
            for s in QUERY_SOURCES:
                qs = qc[qc // QUERY_ID_MULT == s]
                for _, sl in chunk_slices(len(qs)):
                    t = texts_of("train", s, qs[sl] % QUERY_ID_MULT)
                    c = pass_c_chunk(vec, b, qs[sl], mapped_c_texts(t, d), s1_pos)
                    cs.append(c[["query_id", "s1_id", "score"]].rename(columns={"score": "scoreC"}))
                    ds.append(pass_d(qs[sl], t["name_core"], s1_pos, s1t["name_core"], d))
            del vec, b
            gc.collect()
        c = pd.concat(cs, ignore_index=True)
        dd = pd.concat(ds, ignore_index=True)
        a = old[np.isin(old["s1_id"].to_numpy(), s1_pos) & (old["scoreA"].fillna(0).to_numpy() > 0)][
            ["query_id", "s1_id", "scoreA"]]
        u = a.merge(c, on=["query_id", "s1_id"], how="outer").merge(dd, on=["query_id", "s1_id"], how="outer")
        parts.append(u)
        logger.info("[%s] pass C %d, pass D %d (%d queries), union %d", country, len(c), len(dd),
                    dd["query_id"].nunique(), len(u))
    union = pd.concat(parts, ignore_index=True)
    for col in ("scoreA", "scoreC", "scoreD"):
        union[col] = union[col].fillna(0).astype(np.float32)
    union["s1_id"] = union["s1_id"].astype(np.int32)
    write_parquet(union, V4_BENCH_DIR / "union.parquet")

    old = old.fillna(0)
    old["scoreD"] = np.float32(0)
    rep["recall"] = {"v1_union": recall_table(old, bq, bs1), "v1_top5": recall_table(adaptive_top_k(old, 0.0), bq, bs1),
                     "v4_union": recall_table(union, bq, bs1)}
    no_d = union.assign(scoreD=np.float32(0))
    rep["recall"]["v4_top5_no_passD"] = recall_table(adaptive_top_k(no_d, 0.0), bq, bs1)
    for m in MARGINS:
        rep["recall"][f"v4_adaptive_m{m}"] = recall_table(adaptive_top_k(union, m), bq, bs1)
        rep["recall"][f"v1_adaptive_m{m}"] = recall_table(adaptive_top_k(old, m), bq, bs1)
    for k, v in rep["recall"].items():
        logger.info("recall %-22s %s", k, v)
    with open(LOG_DIR / "blocking_v4_report.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    return rep


def run_test(country: str, logger) -> dict:
    """Blocking v4 for one test country: pass A (v1 rules) + pass C on dictionary-mapped texts + pass D, adaptive
    top-k (MARGIN) -> <output>/cand_v4/test/topk_<country>.parquet (dictionary mined on ALL train true pairs)."""
    from .blocking import (A_COLS, build_s1_keys, country_positions, country_slug, pass_a_chunk, read_country,
                           token_df)
    out = TEST_V4_DIR / f"topk_{country_slug(country)}.parquet"
    if out.exists():
        logger.info("[%s] cached: %s", country, out)
        return {}
    d = build_dictionary(np.empty(0, dtype=np.int64), logger)
    s1_pos = country_positions(norm_path("test", 1))[country]
    s1t = texts_of("test", 1, s1_pos)
    s1a = read_country(norm_path("test", 1), country, A_COLS)
    df = token_df(s1a["addr_tokens"].to_pylist())
    s1_keys = build_s1_keys(s1a["num_keys"].to_pylist(), s1a["addr_tokens"].to_pylist(), s1_pos.astype(np.int32), df)
    del s1a
    vec, b = fit_c_index(mapped_c_texts(s1t, d))
    parts = []
    with StageTimer(f"v4_test_{country}", logger, s1=len(s1_pos)):
        for s in QUERY_SOURCES:
            pos = country_positions(norm_path("test", s)).get(country, np.empty(0, np.int64))
            qids = s * QUERY_ID_MULT + pos
            full = pq.read_table(norm_path("test", s), columns=TEXT_COLS + A_COLS).take(pa.array(pos))
            for _, sl in chunk_slices(len(qids)):
                ta = full.slice(sl.start, sl.stop - sl.start)
                t = {c: ta[c].to_pylist() for c in TEXT_COLS}
                a = pass_a_chunk(qids[sl], ta["num_keys"].to_pylist(), ta["addr_tokens"].to_pylist(), s1_keys, df)
                c = pass_c_chunk(vec, b, qids[sl], mapped_c_texts(t, d), s1_pos)
                dd = pass_d(qids[sl], t["name_core"], s1_pos, s1t["name_core"], d)
                u = a[["query_id", "s1_id", "score"]].rename(columns={"score": "scoreA"}).merge(
                    c[["query_id", "s1_id", "score"]].rename(columns={"score": "scoreC"}),
                    on=["query_id", "s1_id"], how="outer").merge(dd, on=["query_id", "s1_id"], how="outer")
                for col in ("scoreA", "scoreC", "scoreD"):
                    u[col] = u[col].fillna(0).astype(np.float32)
                u["s1_id"] = u["s1_id"].astype(np.int32)
                parts.append(adaptive_top_k(u, MARGIN))
            logger.info("[%s] source %d: %d queries", country, s, len(qids))
    top = pd.concat(parts, ignore_index=True)
    write_parquet(top, out)
    rep = {"country": country, "dictionary_size": len(d), "pairs": int(len(top)),
           "queries": int(top["query_id"].nunique()),
           "cands_per_query": round(len(top) / max(1, top["query_id"].nunique()), 3),
           "pct_pairs_from_passD_only": round(100 * float(((top["scoreC"] == 0) & (top["scoreA"] == 0)).mean()), 2)}
    with open(LOG_DIR / f"blocking_v4_test_{country_slug(country)}.json", "w", encoding="utf-8", newline="") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    logger.info("[%s] %s", country, rep)
    return rep


def main() -> None:
    """Parse flags and run the requested split."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["bench", "test"], required=True)
    ap.add_argument("--country", help="test: the country to block (one per Kaggle kernel)")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("blocking_v4")
    with StageTimer(f"blocking_v4_{args.split}", logger, threads=N_THREADS):
        if args.split == "bench":
            run_bench(logger)
        else:
            run_test(args.country, logger)


if __name__ == "__main__":
    main()
