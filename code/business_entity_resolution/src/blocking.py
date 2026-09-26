"""Stage 3: blocking v1 in the REVERSE direction (plan Section 6.2, passes A and C).

Each S2/S3 record (the query) looks for candidate owners among the S1 records of
the same country label (the index). Country is only used to partition work.

    python -m src.blocking --stage bench  --split train   # 100k-query benchmark + runtime/RAM projections
    python -m src.blocking --stage block  --split train   # recall query set (see build_query_set)
    python -m src.blocking --stage block  --split test    # every S2/S3 record
    python -m src.blocking --stage recall --split train   # recall report + logs/blocking_misses.txt
    python -m src.blocking --stage block --split test --country France   # one country only (no combine)
    python -m src.blocking --stage block --split test --country India --shard 1/2   # half of India's queries
    python -m src.blocking --stage combine --split test   # combine all part files (e.g. after per-country runs)

IDs are integers, never repeated strings:
    s1_id    = row index in cache/norm/<split>_s1.parquet                     (int32)
    query_id = source * 10**8 + row index in cache/norm/<split>_s<source>.parquet (int64)
Lookups: cache/cand/<split>/lookup_s1.parquet and cache/cand/<split>/queries.parquet.

Outputs (per country, per source, per 100k-query chunk; existing parts are skipped,
so a crashed run resumes):
    cache/cand/<split>/<country>/{passA,passC,union}_s<source>_part_XXXX.parquet
and the combined tables cache/cand/<split>_passA.parquet, <split>_passC.parquet
(query_id, s1_id, pass, score, rank) and <split>_union.parquet
(query_id, s1_id, scoreA, scoreC, in_A, in_C).
"""
import argparse
import gc
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sparse_dot_topn import sp_matmul_topn

from .config import CACHE_DIR, LOG_DIR, N_THREADS, SEED, add_path_args, env_float, set_seeds
from .io_utils import raw_parquet_path, read_parquet, write_parquet
from .logging_utils import StageTimer, get_logger, machine_info, rss_mb
from .normalize import norm_path
from .split import load_split

CAND_DIR = CACHE_DIR / "cand"
QUERY_SOURCES = (2, 3)
QUERY_ID_MULT = 10 ** 8
PAIR_KEY_SHIFT = 22                  # pair key = query_id << 22 | s1_id  (s1_id < 4.19M)
CHUNK_ROWS = 100_000
MIN_FREE_GB = env_float("MIN_FREE_GB", 7.0)   # RAM gate; override with env MIN_FREE_GB or --min-free-gb

A_RARE_TOKENS = 2
A_MAX_BLOCK = 50
A_MAX_OWNERS = 20

C_TOP_N = 10
# Tuned 2026-09-26 (logs/pass_c_tuning_table.md): char_wb 4-grams, drop fragments in > 3% of the
# country's S1 (max_df). Union recall US 0.9881 / India 0.9516 vs 0.9890 / 0.9663 for the old
# (3,4)-gram unpruned setting, at ~10x less pass-C time.
C_PARAMS = dict(analyzer="char_wb", ngram_range=(4, 4), min_df=2, max_df=0.03, sublinear_tf=True,
                dtype=np.float32, max_features=600_000)

RECALL_RANDOM_QUERIES = 300_000      # train query set: distractors (S2/S3 not matched to a sampled S1)
TRAIN_MATCH_ENTITIES = 200_000       # train query set: train-fold S1 whose matched S2/S3 are added
BENCH_QUERIES = 100_000
TEST_PROJECTION_LIMIT_MIN = 90
MISSES_PER_COUNTRY = 30


# ------------------------------------------------------------------ helpers
def check_free_ram(logger, allow_low: bool, min_gb: float = MIN_FREE_GB) -> dict:
    """Log machine RAM; exit with a warning if free RAM is under ``min_gb`` unless ``allow_low``."""
    info = machine_info()
    logger.info("machine: %s", info)
    if info["ram_available_gb"] < min_gb:
        logger.warning("ONLY %.2f GB RAM FREE (< %.0f GB). Close other programs, or re-run with "
                       "--allow-low-ram to proceed anyway.", info["ram_available_gb"], min_gb)
        if not allow_low:
            sys.exit(2)
    return info


def country_slug(country: str) -> str:
    """Return a filesystem-safe folder name for an arbitrary country label."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", country) or "_"


def country_positions(path: Path) -> dict:
    """Return {country: sorted global row indices} for a normalised Parquet file."""
    col = pq.read_table(path, columns=["country"])["country"]
    return {c: np.flatnonzero(pc.equal(col, c).to_numpy(zero_copy_only=False))
            for c in sorted(pc.unique(col).to_pylist())}


def read_country(path: Path, country: str, columns: list) -> pa.Table:
    """Read one country's rows (file order) of a normalised Parquet file as an Arrow table."""
    return pq.read_table(path, columns=columns, filters=[("country", "==", country)])


def chunk_slices(n: int, size: int = CHUNK_ROWS):
    """Yield (chunk index, slice) covering range(n) in blocks of ``size``."""
    for i, start in enumerate(range(0, n, size)):
        yield i, slice(start, min(start + size, n))


def part_path(split: str, country: str, kind: str, source: int, i: int, tag: str = "") -> Path:
    """Return the part-file path for one pass/union chunk (``tag`` marks a query shard, e.g. '_sh1of2')."""
    return CAND_DIR / split / country_slug(country) / f"{kind}_s{source}{tag}_part_{i:04d}.parquet"


def shard_of(query_id: np.ndarray, n: int) -> np.ndarray:
    """Return the 0-based shard (of ``n``) of each query id via a fixed multiplicative hash (deterministic)."""
    h = (np.asarray(query_id, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
    return (h % np.uint64(n)).astype(np.int64)


def parse_shard(spec: str) -> tuple:
    """Parse 'k/n' (1-based k) into (k, n); None -> (1, 1)."""
    if not spec:
        return 1, 1
    k, n = (int(x) for x in spec.split("/"))
    if not 1 <= k <= n:
        raise ValueError(f"bad shard {spec!r}")
    return k, n


def pair_keys(query_id: np.ndarray, s1_id: np.ndarray) -> np.ndarray:
    """Return int64 keys identifying (query_id, s1_id) pairs."""
    return (query_id.astype(np.int64) << PAIR_KEY_SHIFT) | s1_id.astype(np.int64)


# ------------------------------------------------------------------ lookups and query sets
def build_lookups(split: str, logger) -> None:
    """Write cache/cand/<split>/lookup_s1.parquet (s1_id, entity_id, country) if missing."""
    out = CAND_DIR / split / "lookup_s1.parquet"
    if out.exists():
        return
    t = pq.read_table(norm_path(split, 1), columns=["entity_id", "country"])
    if t.num_rows >= 1 << PAIR_KEY_SHIFT:
        raise ValueError("too many S1 rows for the pair-key encoding")
    df = t.to_pandas()
    df.insert(0, "s1_id", np.arange(len(df), dtype=np.int32))
    write_parquet(df, out)
    logger.info("wrote %s (%d rows)", out, len(df))


def build_query_set(split: str, logger) -> pd.DataFrame:
    """Build (or load) the query set: query_id, source, row, country, entity_id, is_val_match.

    test: every S2/S3 record. train: every S2/S3 record matched to a validation S1 entity
    (role "val"), plus every S2/S3 record matched to TRAIN_MATCH_ENTITIES train-fold S1
    entities sampled stratified by country (role "train", for training the scorer), plus
    RECALL_RANDOM_QUERIES other random S2/S3 records (role "distractor"); seed 42. The index
    is always all S1 of the country.
    """
    out = CAND_DIR / split / "queries.parquet"
    if out.exists():
        return pd.read_parquet(out)
    frames = []
    for source in QUERY_SOURCES:
        t = pq.read_table(norm_path(split, source), columns=["entity_id", "country"]).to_pandas()
        t.insert(0, "row", np.arange(len(t), dtype=np.int64))
        t.insert(0, "source", np.int8(source))
        frames.append(t)
    allq = pd.concat(frames, ignore_index=True)
    del frames
    if split == "train":
        split_df = load_split()
        val_s1 = set(split_df.loc[split_df["fold"] == "val", "s1_id"])
        tr = split_df[split_df["fold"] == "train"].merge(
            pq.read_table(norm_path("train", 1), columns=["entity_id", "country"]).to_pandas()
            .rename(columns={"entity_id": "s1_id"}), on="s1_id", how="left", validate="1:1")
        n_tr = min(TRAIN_MATCH_ENTITIES, len(tr))
        tr_s1 = set(tr["s1_id"]) if n_tr == len(tr) else set(train_test_split(
            tr["s1_id"].to_numpy(), train_size=n_tr, stratify=tr["country"].to_numpy(), random_state=SEED)[0])
        pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
        val_other = pairs.loc[pairs["s1_id"].isin(val_s1), "other_id"]
        tr_other = pairs.loc[pairs["s1_id"].isin(tr_s1), "other_id"]
        allq["is_val_match"] = allq["entity_id"].isin(val_other).to_numpy()
        is_tr = allq["entity_id"].isin(tr_other).to_numpy()
        others = np.flatnonzero(~allq["is_val_match"].to_numpy() & ~is_tr)
        rng = np.random.default_rng(SEED)
        extra = rng.choice(others, size=min(RECALL_RANDOM_QUERIES, len(others)), replace=False)
        role = np.full(len(allq), "", dtype=object)
        role[extra] = "distractor"
        role[is_tr] = "train"
        role[allq["is_val_match"].to_numpy()] = "val"
        allq["role"] = role
        allq = allq[role != ""]
        logger.info("query set: %d val-matched + %d train-matched (%d train S1) + %d distractors = %d",
                    int((allq["role"] == "val").sum()), int((allq["role"] == "train").sum()), len(tr_s1),
                    len(extra), len(allq))
        del pairs, val_other, tr_other, split_df, tr
    else:
        allq["is_val_match"] = False
        allq["role"] = "test"
    allq.insert(0, "query_id", allq["source"].astype(np.int64) * QUERY_ID_MULT + allq["row"])
    allq = allq.sort_values(["country", "source", "row"], kind="stable").reset_index(drop=True)
    write_parquet(allq, out)
    return allq


# ------------------------------------------------------------------ pass A
def token_df(addr_tokens: list) -> Counter:
    """Return document frequency of address tokens (addr_tokens are unique per record)."""
    df = Counter()
    for toks in addr_tokens:
        df.update(toks.split())
    return df


def record_keys(num_keys: list, addr_tokens: list, df: Counter, query: bool) -> tuple:
    """Return (key strings, record positions) for pass A: every num_key x each of the 2 rarest tokens.

    Rarity is S1 document frequency. For queries, tokens unseen in S1 are skipped
    (they cannot match anything and would waste a slot).
    """
    keys, owners = [], []
    for i, (nks, toks) in enumerate(zip(num_keys, addr_tokens)):
        if not nks:
            continue
        cand = toks.split()
        if query:
            cand = [t for t in cand if t in df]
        rare = sorted(cand, key=lambda t: (df.get(t, 0), t))[:A_RARE_TOKENS]
        for nk in nks.split():
            for t in rare:
                keys.append(f"{nk}|{t}")
                owners.append(i)
    return keys, np.asarray(owners, dtype=np.int64)


def hash_keys(keys: list) -> np.ndarray:
    """Hash key strings to deterministic int64 values."""
    if not keys:
        return np.empty(0, dtype=np.int64)
    return pd.util.hash_array(np.asarray(keys, dtype=object), categorize=False).view(np.int64)


def build_s1_keys(num_keys: list, addr_tokens: list, s1_ids: np.ndarray, df: Counter) -> pd.DataFrame:
    """Return the S1 key table (key, s1_id) without keys whose block exceeds A_MAX_BLOCK."""
    keys, owners = record_keys(num_keys, addr_tokens, df, query=False)
    t = pd.DataFrame({"key": hash_keys(keys), "s1_id": s1_ids[owners].astype(np.int32)}).drop_duplicates()
    size = t.groupby("key")["s1_id"].transform("size")
    return t[size <= A_MAX_BLOCK].reset_index(drop=True)


def pass_a_chunk(query_ids: np.ndarray, num_keys: list, addr_tokens: list,
                 s1_keys: pd.DataFrame, df: Counter) -> pd.DataFrame:
    """Return pass-A candidates for one query chunk, ranked by number of shared keys (max A_MAX_OWNERS)."""
    keys, owners = record_keys(num_keys, addr_tokens, df, query=True)
    q = pd.DataFrame({"key": hash_keys(keys), "query_id": query_ids[owners]}).drop_duplicates()
    m = q.merge(s1_keys, on="key", how="inner")
    g = m.groupby(["query_id", "s1_id"], sort=False).size().reset_index(name="score")
    g = g.sort_values(["query_id", "score", "s1_id"], ascending=[True, False, True], kind="stable")
    g["rank"] = (g.groupby("query_id").cumcount() + 1).astype(np.int16)
    g = g[g["rank"] <= A_MAX_OWNERS]
    return _cand_frame(g["query_id"].to_numpy(), g["s1_id"].to_numpy(), g["score"].to_numpy(),
                       g["rank"].to_numpy(), "A")


# ------------------------------------------------------------------ pass C
def c_texts(t: pa.Table) -> list:
    """Return the pass-C text per record: name_compact + ' ' + name_core + ' ' + addr_clean."""
    return [f"{a} {b} {c}" for a, b, c in zip(t["name_compact"].to_pylist(), t["name_core"].to_pylist(),
                                               t["addr_clean"].to_pylist())]


def fit_c_index(s1_texts: list) -> tuple:
    """Fit the char TF-IDF on one country's S1; return (vectorizer, S1 matrix transposed as CSR)."""
    params = dict(C_PARAMS)
    if isinstance(params.get("max_df"), float) and params["max_df"] < 1.0:
        # same cut as the fraction (keep df <= floor(max_df * n)), but never below min_df (tiny indexes)
        params["max_df"] = max(params["min_df"], int(params["max_df"] * len(s1_texts)))
    vec = TfidfVectorizer(**params)
    x = vec.fit_transform(s1_texts)
    b = x.T.tocsr()
    del x
    gc.collect()
    return vec, b


def pass_c_chunk(vec: TfidfVectorizer, b, query_ids: np.ndarray, texts: list,
                 s1_ids: np.ndarray) -> pd.DataFrame:
    """Return the pass-C top-C_TOP_N S1 candidates (cosine) for one query chunk."""
    a = vec.transform(texts)
    c = sp_matmul_topn(a, b, top_n=C_TOP_N, sort=True, n_threads=N_THREADS)
    counts = np.diff(c.indptr)
    rows = np.repeat(np.arange(len(texts)), counts)
    rank = (np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts) + 1).astype(np.int16)
    return _cand_frame(query_ids[rows], s1_ids[c.indices], c.data, rank, "C")


def _cand_frame(query_id, s1_id, score, rank, pass_name: str) -> pd.DataFrame:
    """Return a candidate table with the standard columns and compact dtypes."""
    n = len(query_id)
    return pd.DataFrame({
        "query_id": np.asarray(query_id, dtype=np.int64),
        "s1_id": np.asarray(s1_id, dtype=np.int32),
        "pass": pd.Categorical([pass_name] * n, categories=["A", "C"]),
        "score": np.asarray(score, dtype=np.float32),
        "rank": np.asarray(rank, dtype=np.int16),
    })


def union_frame(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Outer-join pass A and C candidates into (query_id, s1_id, scoreA, scoreC, in_A, in_C)."""
    u = a[["query_id", "s1_id", "score"]].rename(columns={"score": "scoreA"}).merge(
        c[["query_id", "s1_id", "score"]].rename(columns={"score": "scoreC"}),
        on=["query_id", "s1_id"], how="outer")
    u["in_A"] = u["scoreA"].notna()
    u["in_C"] = u["scoreC"].notna()
    return u.astype({"scoreA": np.float32, "scoreC": np.float32}).sort_values(
        ["query_id", "s1_id"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ block stage
A_COLS = ["num_keys", "addr_tokens"]
C_COLS = ["name_compact", "name_core", "addr_clean"]


def block_country(split: str, country: str, queries: pd.DataFrame, s1_pos: np.ndarray,
                  src_pos: dict, logger, tag: str = "") -> None:
    """Run passes A and C and the union for one country's queries, writing part files."""
    plan = []  # (source, chunk index, local row indices into the country slice, query ids)
    for source in QUERY_SOURCES:
        q = queries[queries["source"] == source]
        local = np.searchsorted(src_pos[source], q["row"].to_numpy())
        for i, sl in chunk_slices(len(q)):
            plan.append((source, i, local[sl], q["query_id"].to_numpy()[sl]))
    n_queries = len(queries)
    if not plan:
        return

    # ---- pass A
    todo = [p for p in plan if not part_path(split, country, "passA", p[0], p[1], tag).exists()]
    if todo:
        with StageTimer(f"blockA_{split}_{country}", logger, split=split, country=country,
                        passname="A", queries=n_queries) as timer:
            s1 = read_country(norm_path(split, 1), country, A_COLS)
            nk, tk = s1["num_keys"].to_pylist(), s1["addr_tokens"].to_pylist()
            df = token_df(tk)
            s1_keys = build_s1_keys(nk, tk, s1_pos.astype(np.int32), df)
            del s1, nk, tk
            logger.info("[A %s] S1 key table %d rows", country, len(s1_keys))
            for source in QUERY_SOURCES:
                src_todo = [p for p in todo if p[0] == source]
                if not src_todo:
                    continue
                t = read_country(norm_path(split, source), country, A_COLS)
                for _, i, local, qids in src_todo:
                    sub = t.take(pa.array(local))
                    res = pass_a_chunk(qids, sub["num_keys"].to_pylist(), sub["addr_tokens"].to_pylist(),
                                       s1_keys, df)
                    write_parquet(res, part_path(split, country, "passA", source, i, tag))
                del t
                gc.collect()
            timer.extra["n_chunks"] = len(todo)
            del s1_keys, df
            gc.collect()

    # ---- pass C
    todo = [p for p in plan if not part_path(split, country, "passC", p[0], p[1], tag).exists()]
    if todo:
        with StageTimer(f"blockC_{split}_{country}", logger, split=split, country=country,
                        passname="C", queries=n_queries) as timer:
            t0 = time.perf_counter()
            s1 = read_country(norm_path(split, 1), country, C_COLS)
            vec, b = fit_c_index(c_texts(s1))
            del s1
            gc.collect()
            fit_s = time.perf_counter() - t0
            logger.info("[C %s] fit on %d S1 in %.1fs: vocab %d, nnz %d, rss %.0f MB", country, b.shape[1],
                        fit_s, len(vec.vocabulary_), b.nnz, rss_mb())
            for source in QUERY_SOURCES:
                src_todo = [p for p in todo if p[0] == source]
                if not src_todo:
                    continue
                t = read_country(norm_path(split, source), country, C_COLS)
                for _, i, local, qids in src_todo:
                    res = pass_c_chunk(vec, b, qids, c_texts(t.take(pa.array(local))), s1_pos)
                    write_parquet(res, part_path(split, country, "passC", source, i, tag))
                del t
                gc.collect()
            timer.extra.update(n_chunks=len(todo), fit_s=round(fit_s, 1))
            del vec, b
            gc.collect()

    # ---- union
    for source, i, _, _ in plan:
        out = part_path(split, country, "union", source, i, tag)
        if not out.exists():
            a = pd.read_parquet(part_path(split, country, "passA", source, i, tag))
            c = pd.read_parquet(part_path(split, country, "passC", source, i, tag))
            write_parquet(union_frame(a, c), out)


def combine_parts(split: str, kind: str, out_name: str, logger) -> None:
    """Stream every part file of one kind into a single Parquet file."""
    parts = sorted((CAND_DIR / split).glob(f"*/{kind}_s*_part_*.parquet"))
    out = CAND_DIR / out_name
    tmp = out.with_suffix(".parquet.tmp")
    writer, rows = None, 0
    for p in parts:
        t = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        rows += t.num_rows
    if writer is not None:
        writer.close()
        tmp.replace(out)
    logger.info("combined %d %s parts -> %s (%d rows)", len(parts), kind, out, rows)


def run_block(split: str, logger, allow_low_ram: bool, force: bool, min_free_gb: float = MIN_FREE_GB,
              countries: list = None, shard: str = None) -> None:
    """Run blocking for every country of ``split`` (or only ``countries``) and write the candidate tables.

    With ``countries`` given, only those countries' part files are written and the combine step is
    skipped (run --stage combine once all countries are done).
    """
    check_free_ram(logger, allow_low_ram, min_free_gb)
    if force and countries:
        for c in countries:
            shutil.rmtree(CAND_DIR / split / country_slug(c), ignore_errors=True)
    elif force and (CAND_DIR / split).exists():
        shutil.rmtree(CAND_DIR / split)
    (CAND_DIR / split).mkdir(parents=True, exist_ok=True)
    build_lookups(split, logger)
    queries = build_query_set(split, logger)
    s1_pos_all = country_positions(norm_path(split, 1))
    src_pos_all = {s: country_positions(norm_path(split, s)) for s in QUERY_SOURCES}
    k, n = parse_shard(shard)
    tag = "" if n == 1 else f"_sh{k}of{n}"
    if n > 1:
        queries = queries[shard_of(queries["query_id"].to_numpy(), n) == k - 1]
        logger.info("shard %d/%d: %d queries", k, n, len(queries))
    with StageTimer(f"block_{split}{tag}", logger, split=split, queries=len(queries)):
        missing = sorted(set(countries or []) - set(queries["country"]))
        if missing:
            raise SystemExit(f"--country {missing} not in the {split} query set")
        for country, q in queries.groupby("country", sort=True):
            if countries and country not in countries:
                continue
            if country not in s1_pos_all:
                logger.warning("country %r has queries but no S1 records - no candidates", country)
                continue
            logger.info("country %s: %d queries, %d S1", country, len(q), len(s1_pos_all[country]))
            block_country(split, country, q, s1_pos_all[country],
                          {s: src_pos_all[s].get(country, np.empty(0, np.int64)) for s in QUERY_SOURCES}, logger, tag)
            gc.collect()
        if countries or n > 1:
            logger.info("partial run (countries %s, shard %d/%d): skipping combine; run --stage combine "
                        "when all parts are done", countries, k, n)
        else:
            run_combine(split, logger)


def run_combine(split: str, logger) -> None:
    """Combine every pass-A, pass-C and union part file of ``split`` into the three candidate tables."""
    for kind, name in (("passA", f"{split}_passA.parquet"), ("passC", f"{split}_passC.parquet"),
                       ("union", f"{split}_union.parquet")):
        combine_parts(split, kind, name, logger)


# ------------------------------------------------------------------ benchmark
def run_bench(logger, allow_low_ram: bool, min_free_gb: float = MIN_FREE_GB) -> dict:
    """Time passes A and C on BENCH_QUERIES train queries and project full train-recall and test runs."""
    info = check_free_ram(logger, allow_low_ram, min_free_gb)
    base_mb = rss_mb()
    rng = np.random.default_rng(SEED)
    s1_pos = country_positions(norm_path("train", 1))
    src_pos = {s: country_positions(norm_path("train", s)) for s in QUERY_SOURCES}
    n_by_country = {c: sum(len(src_pos[s].get(c, [])) for s in QUERY_SOURCES) for c in s1_pos}
    total = sum(n_by_country.values())
    results = {}
    for country in s1_pos:
        n_q = max(1, round(BENCH_QUERIES * n_by_country[country] / total))
        with StageTimer(f"bench_{country}", logger, country=country, queries=n_q) as timer:
            t0 = time.perf_counter()
            s1 = read_country(norm_path("train", 1), country, A_COLS + C_COLS)
            df = token_df(s1["addr_tokens"].to_pylist())
            s1_keys = build_s1_keys(s1["num_keys"].to_pylist(), s1["addr_tokens"].to_pylist(),
                                    s1_pos[country].astype(np.int32), df)
            a_index_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            vec, b = fit_c_index(c_texts(s1))
            del s1
            gc.collect()
            c_fit_s = time.perf_counter() - t0
            fit_peak_mb = timer.peak_mb
            source = 2
            local = np.sort(rng.choice(len(src_pos[source][country]), size=n_q, replace=False))
            t = read_country(norm_path("train", source), country, A_COLS + C_COLS).take(pa.array(local))
            qids = source * QUERY_ID_MULT + src_pos[source][country][local]
            t0 = time.perf_counter()
            ra = pass_a_chunk(qids, t["num_keys"].to_pylist(), t["addr_tokens"].to_pylist(), s1_keys, df)
            a_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            rc = pass_c_chunk(vec, b, qids, c_texts(t), s1_pos[country])
            c_s = time.perf_counter() - t0
            results[country] = {
                "s1_rows": len(s1_pos[country]), "bench_queries": n_q,
                "a_index_s": round(a_index_s, 1), "c_fit_s": round(c_fit_s, 1),
                "a_per_100k_s": round(a_s * 1e5 / n_q, 1), "c_per_100k_s": round(c_s * 1e5 / n_q, 1),
                "c_vocab": len(vec.vocabulary_), "c_nnz": int(b.nnz),
                "a_cands_per_query": round(len(ra) / n_q, 2), "c_cands_per_query": round(len(rc) / n_q, 2),
                "peak_rss_after_fit_mb": round(fit_peak_mb),
            }
            del vec, b, s1_keys, df, t, ra, rc
            gc.collect()
        results[country]["peak_rss_mb"] = round(timer.peak_mb)
        logger.info("bench %s: %s", country, results[country])

    proj = project_runtimes(results, base_mb)
    out = {"machine": info, "n_threads": N_THREADS, "bench": results, "projection": proj}
    with open(LOG_DIR / "blocking_bench.json", "w", encoding="utf-8", newline="") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info("projection: %s", json.dumps(proj, indent=2))
    if proj["test"]["minutes"] > TEST_PROJECTION_LIMIT_MIN:
        logger.warning("TEST PROJECTION %.0f min EXCEEDS %d min - STOP AND ASK before the test run.",
                       proj["test"]["minutes"], TEST_PROJECTION_LIMIT_MIN)
    return out


def project_runtimes(bench: dict, base_mb: float) -> dict:
    """Project runtime and peak RAM of the full train-recall and test runs from benchmark rates.

    Fit/index time scales with S1 rows; per-query time is taken from the train
    country benchmark (an upper bound for test, whose S1 indexes are smaller).
    A country without a benchmark (France) uses the slowest measured rates.
    """
    per_s1_row = max((r["a_index_s"] + r["c_fit_s"]) / r["s1_rows"] for r in bench.values())
    per_q_worst = max((r["a_per_100k_s"] + r["c_per_100k_s"]) / 1e5 for r in bench.values())
    peak_per_s1_row = max((r["peak_rss_mb"] - base_mb) / r["s1_rows"] for r in bench.values())
    out = {}
    for split in ("train", "test"):
        s1_pos = country_positions(norm_path(split, 1))
        if split == "train":
            q = build_query_set("train", get_logger("blocking"))
            n_q = q.groupby("country").size().to_dict()
        else:
            n_q = Counter()
            for s in QUERY_SOURCES:
                for c, pos in country_positions(norm_path(split, s)).items():
                    n_q[c] += len(pos)
        secs, peak = 0.0, 0.0
        detail = {}
        for c, pos in s1_pos.items():
            r = bench.get(c)
            fit = (r["a_index_s"] + r["c_fit_s"]) * len(pos) / r["s1_rows"] if r else per_s1_row * len(pos)
            per_q = (r["a_per_100k_s"] + r["c_per_100k_s"]) / 1e5 if r else per_q_worst
            s = fit + n_q.get(c, 0) * per_q
            detail[c] = {"s1": len(pos), "queries": int(n_q.get(c, 0)), "minutes": round(s / 60, 1)}
            secs += s
            peak = max(peak, base_mb + peak_per_s1_row * len(pos))
        out[split] = {"minutes": round(secs / 60, 1), "peak_rss_gb": round(peak / 1024, 2), "by_country": detail}
    return out


# ------------------------------------------------------------------ recall report
def run_recall(logger) -> dict:
    """Measure pair recall of passes A, C and the union on validation ground-truth pairs (train)."""
    split = "train"
    queries = build_query_set(split, logger)
    lookup = pd.read_parquet(CAND_DIR / split / "lookup_s1.parquet")
    split_df = load_split()
    val_s1 = split_df.loc[split_df["fold"] == "val", "s1_id"]
    pairs = read_parquet(raw_parquet_path("train", "pairs"), columns=["s1_id", "other_id"])
    pairs = pairs[pairs["s1_id"].isin(set(val_s1))]
    s1_map = pd.Series(lookup["s1_id"].to_numpy(), index=lookup["entity_id"])
    q_map = pd.Series(queries["query_id"].to_numpy(), index=queries["entity_id"])
    vp = pd.DataFrame({"query_id": q_map.reindex(pairs["other_id"]).to_numpy(),
                       "s1_id": s1_map.reindex(pairs["s1_id"]).to_numpy()})
    if vp.isna().any().any():
        raise AssertionError("validation pairs missing from the query set or S1 lookup")
    vp = vp.astype({"query_id": np.int64, "s1_id": np.int64})
    vp["country"] = lookup["country"].to_numpy()[vp["s1_id"].to_numpy()]
    vp["key"] = pair_keys(vp["query_id"].to_numpy(), vp["s1_id"].to_numpy())
    del pairs, s1_map, q_map
    gc.collect()

    report, misses = {}, {}
    for country, vpc in vp.groupby("country"):
        val_keys = np.sort(vpc["key"].to_numpy())
        found = {k: np.zeros(len(val_keys), dtype=bool) for k in ("A", "C", "union", "C@1", "C@3", "C@5", "C@10")}
        n_cand = Counter()
        cdir = CAND_DIR / split / country_slug(country)

        def mark(frame, name):
            """Flag validation pairs present in ``frame``."""
            k = pair_keys(frame["query_id"].to_numpy(), frame["s1_id"].to_numpy())
            idx = np.searchsorted(val_keys, k)
            idx[idx == len(val_keys)] = 0
            hit = val_keys[idx] == k
            found[name][idx[hit]] = True

        for p in sorted(cdir.glob("union_s*_part_*.parquet")):
            u = pd.read_parquet(p)
            mark(u, "union")
            mark(u[u["in_A"]], "A")
            n_cand["union"] += len(u)
            n_cand["A"] += int(u["in_A"].sum())
            n_cand["C"] += int(u["in_C"].sum())
            n_cand["q_zero_union"] -= u["query_id"].nunique()
            n_cand["q_zero_A"] -= u.loc[u["in_A"], "query_id"].nunique()
            n_cand["q_zero_C"] -= u.loc[u["in_C"], "query_id"].nunique()
        for p in sorted(cdir.glob("passC_s*_part_*.parquet")):
            c = pd.read_parquet(p, columns=["query_id", "s1_id", "rank"])
            for k in (1, 3, 5, 10):
                mark(c[c["rank"] <= k], f"C@{k}")
        found["C"] = found["C@10"]
        nq = int((queries["country"] == country).sum())
        for k in ("union", "A", "C"):
            n_cand[f"q_zero_{k}"] += nq
        report[country] = {
            "val_pairs": len(val_keys), "queries": nq,
            **{f"recall_{k}": round(float(v.mean()), 4) for k, v in found.items()},
            **{f"cands_per_query_{k}": round(n_cand[k] / nq, 2) for k in ("A", "C", "union")},
            **{f"pct_zero_cands_{k}": round(100 * n_cand[f"q_zero_{k}"] / nq, 2) for k in ("A", "C", "union")},
        }
        missed = val_keys[~found["union"]]
        rng = np.random.default_rng(SEED)
        misses[country] = rng.choice(missed, size=min(MISSES_PER_COUNTRY, len(missed)), replace=False)
        logger.info("recall %s: %s", country, report[country])

    tot = {"val_pairs": sum(r["val_pairs"] for r in report.values()),
           "queries": sum(r["queries"] for r in report.values())}
    for k in [k for k in next(iter(report.values())) if k.startswith("recall_")]:
        tot[k] = round(sum(r[k] * r["val_pairs"] for r in report.values()) / tot["val_pairs"], 4)
    for k in [k for k in next(iter(report.values())) if k.startswith(("cands_", "pct_zero"))]:
        tot[k] = round(sum(r[k] * r["queries"] for r in report.values()) / tot["queries"], 2)
    report["ALL"] = tot
    with open(LOG_DIR / "blocking_recall.json", "w", encoding="utf-8", newline="") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    write_misses(split, misses, logger)
    return report


def write_misses(split: str, misses: dict, logger) -> None:
    """Write sampled missed validation pairs (raw and normalised fields) to logs/blocking_misses.txt."""
    cols = ["entity_id", "business_name", "business_address", "name_core", "addr_clean", "num_keys"]
    mask = (1 << PAIR_KEY_SHIFT) - 1
    lines = []
    for country, keys in misses.items():
        lines.append(f"===== {country}: {len(keys)} sampled missed validation pairs =====\n")
        for key in keys:
            qid, sid = int(key >> PAIR_KEY_SHIFT), int(key & mask)
            src, row = divmod(qid, QUERY_ID_MULT)
            s1 = _read_row(norm_path(split, 1), sid, cols)
            q = _read_row(norm_path(split, src), row, cols)
            for label, r in (("S1", s1), (f"S{src}", q)):
                lines.append(f"  {label} {r['entity_id']}: {r['business_name']} | {r['business_address']}\n"
                             f"       core='{r['name_core']}' addr='{r['addr_clean']}' keys='{r['num_keys']}'\n")
            lines.append("\n")
    with open(LOG_DIR / "blocking_misses.txt", "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    logger.info("missed-pair examples written to %s", LOG_DIR / "blocking_misses.txt")


def _read_row(path: Path, row: int, cols: list) -> dict:
    """Read one row of a Parquet file by global row index (via its row group)."""
    pf = pq.ParquetFile(path)
    start = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row < start + n:
            return pf.read_row_group(g, columns=cols).slice(row - start, 1).to_pylist()[0]
        start += n
    raise IndexError(row)


def main() -> None:
    """Parse CLI flags and run the requested stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["bench", "block", "recall", "combine"], required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--force", action="store_true", help="delete this split's candidate parts and rebuild")
    ap.add_argument("--country", action="append", help="block only this country label (repeatable)")
    ap.add_argument("--shard", help="k/n: block only query shard k of n (hash of query_id)")
    ap.add_argument("--allow-low-ram", action="store_true", help="run even with less free RAM than --min-free-gb")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                    help=f"free-RAM gate in GB (env MIN_FREE_GB; default {MIN_FREE_GB:g})")
    add_path_args(ap)
    args = ap.parse_args()
    set_seeds()
    logger = get_logger("blocking")
    logger.info("threads: %d | cache root: %s", N_THREADS, CACHE_DIR)
    if args.stage == "bench":
        run_bench(logger, args.allow_low_ram, args.min_free_gb)
    elif args.stage == "block":
        run_block(args.split, logger, args.allow_low_ram, args.force, args.min_free_gb, args.country, args.shard)
    elif args.stage == "combine":
        run_combine(args.split, logger)
    else:
        if args.split != "train":
            raise SystemExit("recall needs ground truth: use --split train")
        run_recall(logger)


if __name__ == "__main__":
    main()
