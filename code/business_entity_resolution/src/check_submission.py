"""Explicit compliance checks on a submission (asserted and printed one by one; exit 1 on the first failure).

    python -m src.check_submission --matching <matching_results.tsv> --candidate <candidate_pairs.tsv>
        --test-dir <organiser test folder> [--reference <matching_results.tsv whose md5 must be equal>]

Checks: exact header and column names; tab-separated with exactly 2 fields per row; '\\n' line endings only (no
b'\\r') and a final newline; every test S1 id exactly once (no missing, no duplicate, no unknown rows); matched ids
only S2-/S3- ids that exist in the test S2/S3 files (no S1 ids, so no self-matches); no duplicate id within a list;
every S1's matches are a subset of its candidates; optional md5 equality with a reference file. Stdlib only.
"""
import argparse
import hashlib
import sys
from pathlib import Path

HEADERS = {"matching": "source1_entity_id\tmatched_entity_ids", "candidate": "source1_entity_id\tcandidate_entity_ids"}


def ok(msg: str) -> None:
    """Print a passed check."""
    print(f"  [PASS] {msg}", flush=True)


def check(cond: bool, msg: str) -> None:
    """Assert a check; print it, or print the failure and exit 1."""
    if not cond:
        print(f"  [FAIL] {msg}", flush=True)
        sys.exit(1)
    ok(msg)


def md5(path: Path) -> str:
    """Return the md5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def first_column(path: Path) -> list:
    """Return the entity ids (first column) of an organiser TSV."""
    with open(path, encoding="utf-8", newline="") as f:
        f.readline()
        return [line.split("\t", 1)[0] for line in f if line.strip("\r\n")]


def read_lists(path: Path, kind: str) -> dict:
    """Check the byte-level format of a submission file and return {S1 id: [ids]} (duplicate rows fail)."""
    raw = path.read_bytes()
    check(b"\r" not in raw, f"{kind}: '\\n' line endings only (no b'\\r')")
    check(raw.endswith(b"\n"), f"{kind}: final newline present")
    lines = raw.decode("utf-8").split("\n")[:-1]
    check(lines[0] == HEADERS[kind], f"{kind}: header is exactly {HEADERS[kind]!r}")
    rows = [line.split("\t") for line in lines[1:]]
    check(all(len(r) == 2 for r in rows), f"{kind}: tab-separated, exactly 2 fields in all {len(rows):,} rows")
    out = {}
    for s1, ids in rows:
        if s1 in out:
            check(False, f"{kind}: duplicate S1 row {s1}")
        out[s1] = ids.split(",") if ids else []
    ok(f"{kind}: no duplicate S1 rows")
    return out


def main() -> None:
    """Parse flags and run every check."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--reference", type=Path, help="file whose md5 must equal the matching file's")
    a = ap.parse_args()
    print(f"matching:  {a.matching}\ncandidate: {a.candidate}\ntest dir:  {a.test_dir}")
    s1_ids = first_column(a.test_dir / "test_source1.tsv")
    q_ids = set(first_column(a.test_dir / "test_source2.tsv")) | set(first_column(a.test_dir / "test_source3.tsv"))
    m = read_lists(a.matching, "matching")
    c = read_lists(a.candidate, "candidate")
    need = set(s1_ids)
    check(len(need) == len(s1_ids), f"test S1 ids are unique ({len(s1_ids):,})")
    for kind, d in (("matching", m), ("candidate", c)):
        check(set(d) == need, f"{kind}: every test S1 id appears exactly once (missing {len(need - set(d))}, "
                              f"unknown {len(set(d) - need)})")
    all_m = [i for ids in m.values() for i in ids]
    check(all(i.startswith(("S2-", "S3-")) for i in all_m), f"matching: all {len(all_m):,} matched ids are S2-/S3- ids")
    check(not (set(all_m) & need), "matching: no S1 id among the matches (no self-matches)")
    check(set(all_m) <= q_ids, "matching: every matched id exists in test_source2/3")
    check(all(len(ids) == len(set(ids)) for ids in m.values()), "matching: no duplicate id within any list")
    check(all(len(ids) == len(set(ids)) for ids in c.values()), "candidate: no duplicate id within any list")
    all_c = {i for ids in c.values() for i in ids}
    check(all_c <= q_ids, "candidate: every candidate id exists in test_source2/3")
    bad = sum(1 for s, ids in m.items() if not set(ids) <= set(c[s]))
    check(bad == 0, f"every S1's matches are a subset of its candidate_pairs row ({bad} violations)")
    ok(f"summary: {len(m):,} S1 rows, {len(all_m):,} matches, {sum(1 for v in m.values() if not v):,} empty")
    dm = md5(a.matching)
    print(f"  md5 matching  = {dm}\n  md5 candidate = {md5(a.candidate)}")
    if a.reference:
        dr = md5(a.reference)
        check(dm == dr, f"md5 equals the reference {a.reference} ({dr})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
