"""Write a small, self-consistent sample of the organiser data (for the smoke test in README.md).

    python -m src.make_sample --src <data root> --dst <sample root> [--train-share 0.03] [--test-share 0.01]

Train: S1 rows with hash(entity_id) < train-share, their ground-truth rows, the S2/S3 records they match, plus the same
share of the other S2/S3 records as distractors. Test: the given share of each test source (all countries kept).
The hash is a stable CRC32 of the entity id, so the sample is deterministic. Files keep the organiser layout
(<dst>/train/train_source1.tsv, ...) and are read / written as plain TSV ('\\n' line endings, QUOTE_NONE semantics:
rows are copied verbatim).
"""
import argparse
import zlib
from pathlib import Path


def keep(entity_id: str, share: float) -> bool:
    """Return True for a stable pseudo-random ``share`` of entity ids."""
    return zlib.crc32(entity_id.encode("utf-8")) % 1_000_000 < share * 1_000_000


def copy_rows(src: Path, dst: Path, pred) -> int:
    """Copy the header and every data row whose first field satisfies ``pred``; return rows written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, encoding="utf-8", newline="") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline().rstrip("\r\n") + "\n")
        for line in fi:
            line = line.rstrip("\r\n")
            if pred(line.split("\t", 1)[0]):
                fo.write(line + "\n")
                n += 1
    return n


def main() -> None:
    """Parse flags and write the sample."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="organiser data root (holds train/ and test/)")
    ap.add_argument("--dst", required=True, type=Path, help="sample data root to create")
    ap.add_argument("--train-share", type=float, default=0.03)
    ap.add_argument("--test-share", type=float, default=0.01)
    a = ap.parse_args()
    s1_keep = lambda e: keep(e, a.train_share)                                            # noqa: E731
    n = copy_rows(a.src / "train" / "train_source1.tsv", a.dst / "train" / "train_source1.tsv", s1_keep)
    matched = set()
    gt_out = a.dst / "train" / "train_ground_truth.tsv"
    with open(a.src / "train" / "train_ground_truth.tsv", encoding="utf-8", newline="") as fi, \
            open(gt_out, "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline().rstrip("\r\n") + "\n")
        for line in fi:
            line = line.rstrip("\r\n")
            s1, ids = line.split("\t", 1)
            if s1_keep(s1):
                fo.write(line + "\n")
                matched.update(i for i in ids.split(",") if i)
    print(f"train S1 {n}, true matches {len(matched)}")
    for s in (2, 3):
        n = copy_rows(a.src / "train" / f"train_source{s}.tsv", a.dst / "train" / f"train_source{s}.tsv",
                      lambda e: e in matched or keep(e, a.train_share))
        print(f"train S{s} {n}")
    for s in (1, 2, 3):
        n = copy_rows(a.src / "test" / f"test_source{s}.tsv", a.dst / "test" / f"test_source{s}.tsv",
                      lambda e: keep(e, a.test_share))
        print(f"test S{s} {n}")


if __name__ == "__main__":
    main()
