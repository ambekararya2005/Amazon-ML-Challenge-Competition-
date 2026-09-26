"""TSV/Parquet I/O and the writers for the two submission files.

Reading: every TSV is read with sep="\\t", dtype=str, keep_default_na=False,
quoting=csv.QUOTE_NONE, and checked so that no text value is NA (empty fields
stay "") and no value contains a carriage return.

Writing: submission files are written with "\\n" line endings only, then read
back in binary and checked for b"\\r".
"""
import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from .config import (CACHE_DIR, CANDIDATE_HEADER, DATA_DIR, GT_COLS, MATCHING_HEADER,
                     SOURCE_COLS)

TSV_READ_KWARGS = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
VALID_MATCH_PREFIXES = ("S2-", "S3-")
FORBIDDEN_ID_CHARS = ("\t", ",", "\r", "\n")


# ---------------------------------------------------------------- paths
def data_file(split: str, name: str) -> Path:
    """Return DATA_DIR/<split>/<name> (organiser layout), or DATA_DIR/<name> if only the flat layout exists."""
    nested = DATA_DIR / split / name
    flat = DATA_DIR / name
    return flat if not nested.exists() and flat.exists() else nested


def source_tsv_path(split: str, source: int) -> Path:
    """Return the raw TSV path for one split ("train"/"test") and source (1/2/3)."""
    return data_file(split, f"{split}_source{source}.tsv")


def ground_truth_tsv_path(split: str = "train") -> Path:
    """Return the raw ground-truth TSV path (only exists for train)."""
    return data_file(split, f"{split}_ground_truth.tsv")


def raw_parquet_path(split: str, name: str) -> Path:
    """Return the cached Parquet path for a raw table, e.g. ("train", "source1")."""
    return CACHE_DIR / "raw" / f"{split}_{name}.parquet"


# ---------------------------------------------------------------- checks
def assert_no_na(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    """Raise if any of ``cols`` holds an NA value (empty fields must be "")."""
    bad = {c: int(n) for c, n in df[list(cols)].isna().sum().items() if n}
    if bad:
        raise ValueError(f"{label}: NA values found (empty fields must stay ''): {bad}")


def assert_no_cr_values(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    """Raise if any value in ``cols`` contains a carriage return."""
    bad = {c: int(n) for c in cols if (n := df[c].str.contains("\r", regex=False).sum())}
    if bad:
        raise ValueError(f"{label}: values containing '\\r' found: {bad}")


def assert_no_cr_file(path: Path, chunk_size: int = 1 << 24) -> None:
    """Read ``path`` in binary and raise if it contains any b"\\r" byte."""
    with open(path, "rb") as f:
        offset = 0
        while chunk := f.read(chunk_size):
            pos = chunk.find(b"\r")
            if pos != -1:
                raise ValueError(f"{path}: found b'\\r' at byte {offset + pos}; "
                                 "output must use '\\n' line endings only")
            offset += len(chunk)


# ---------------------------------------------------------------- reading
def read_tsv(path: Path, expected_cols: Sequence[str]) -> pd.DataFrame:
    """Read a challenge TSV with the mandatory settings and validate it.

    Checks the header matches ``expected_cols`` exactly, that no value is NA and
    that no value contains a carriage return.
    """
    df = pd.read_csv(path, **TSV_READ_KWARGS)
    if list(df.columns) != list(expected_cols):
        raise ValueError(f"{path}: columns {list(df.columns)} != expected {list(expected_cols)}")
    assert_no_na(df, df.columns, str(path))
    assert_no_cr_values(df, df.columns, str(path))
    return df


def read_source_tsv(split: str, source: int) -> pd.DataFrame:
    """Read one raw source TSV (entity_id, business_name, business_address, country)."""
    return read_tsv(source_tsv_path(split, source), SOURCE_COLS)


def read_ground_truth_tsv(split: str = "train") -> pd.DataFrame:
    """Read the raw ground-truth TSV (source1_entity_id, matched_entity_ids)."""
    return read_tsv(ground_truth_tsv_path(split), GT_COLS)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to Parquet atomically (tmp file then rename) so a crash never leaves a partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def read_parquet(path: Path, columns: Optional[Sequence[str]] = None,
                 country: Optional[str] = None) -> pd.DataFrame:
    """Read a cached Parquet table, optionally only some columns and one country.

    ``country`` is an arbitrary label (open set); filtering happens inside
    pyarrow so only that country's rows are materialised. Text columns are
    checked to be NA-free.
    """
    cols = None if columns is None else list(columns)
    filters = None if country is None else [("country", "==", country)]
    df = pd.read_parquet(path, columns=cols, filters=filters)
    assert_no_na(df, df.columns, str(path))
    return df


def list_countries(path: Path) -> list:
    """Return the sorted distinct country labels in a cached source table."""
    return sorted(pd.read_parquet(path, columns=["country"])["country"].unique().tolist())


# ---------------------------------------------------------------- writing
def write_id_list_tsv(path: Path, header: Sequence[str], s1_ids: Iterable[str],
                      id_lists: Mapping[str, Sequence[str]]) -> int:
    """Write one row per S1 id with its comma-joined S2/S3 id list; return rows written.

    ``s1_ids`` fixes the required rows and their order; S1 ids absent from
    ``id_lists`` get an empty list. Enforces the submission rules (no duplicate
    S1 rows, S2-/S3- ids only, no duplicates within a list, no stray keys),
    writes with "\\n" line endings via a tmp file, then asserts the file holds
    no b"\\r" before moving it into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    seen = set()
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n",
                                quoting=csv.QUOTE_NONE, escapechar=None)
            writer.writerow(header)
            for s1 in s1_ids:
                if s1 in seen:
                    raise ValueError(f"duplicate source1_entity_id row: {s1}")
                seen.add(s1)
                ids = list(id_lists.get(s1, ()))
                if len(ids) != len(set(ids)):
                    raise ValueError(f"{s1}: duplicate ids in list")
                bad = [i for i in ids if not i.startswith(VALID_MATCH_PREFIXES)]
                if bad:
                    raise ValueError(f"{s1}: ids without S2-/S3- prefix: {bad[:5]}")
                bad = [i for i in [s1, *ids] if any(ch in i for ch in FORBIDDEN_ID_CHARS)]
                if bad:
                    raise ValueError(f"{s1}: ids containing tab/comma/CR/LF: {bad[:5]!r}")
                writer.writerow([s1, ",".join(ids)])
        extra = set(id_lists) - seen
        if extra:
            raise ValueError(f"{len(extra)} S1 ids in id_lists are not in s1_ids, e.g. {sorted(extra)[:5]}")
        assert_no_cr_file(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    assert_no_cr_file(path)
    return len(seen)


def write_matching_results(path: Path, s1_ids: Iterable[str],
                           matches: Mapping[str, Sequence[str]]) -> int:
    """Write output/matching_results.tsv (final matches); see write_id_list_tsv."""
    return write_id_list_tsv(path, MATCHING_HEADER, s1_ids, matches)


def write_candidate_pairs(path: Path, s1_ids: Iterable[str],
                          candidates: Mapping[str, Sequence[str]]) -> int:
    """Write output/candidate_pairs.tsv (the shortlist the model scores); see write_id_list_tsv."""
    return write_id_list_tsv(path, CANDIDATE_HEADER, s1_ids, candidates)
