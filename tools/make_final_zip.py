"""Build the final submission zip in the structure required by student_resource/README.md.

    python tools/make_final_zip.py --outputs output/best/<run> --team <team_name>
    python tools/make_final_zip.py --outputs output/best/<run> --out <dir>/test.zip --validate

Zip layout (paths relative to the zip root):
    output/matching_results.tsv            <- <outputs>/matching_results.tsv
    output/candidate_pairs.tsv             <- <outputs>/candidate_pairs.tsv
    code/business_entity_resolution/src/   <- code/business_entity_resolution/src/ (*.py only)
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md              <- repo root (the filled copy)

Checks (any failure aborts before the zip is written):
    * every required file exists and is non-empty; src/ holds Python files only (caches, data, __pycache__ excluded)
    * both TSVs: the exact header, '\\n' line endings only (no b'\\r'), a final newline, one row per S1, no duplicate
      ids within a list, S2-/S3- ids only, the same set of S1 rows in both files
    * matches are a subset of the candidates, per S1
    * optional (--validate): the organiser validator student_resource/utils/validate_submission.py
After writing, the zip is re-opened, its CRCs are tested and its name list is compared with the plan.
Standard library only.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code" / "business_entity_resolution"
DOC_FILE = ROOT / "Documentation_template.md"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
TEST_DIR = ROOT / "student_resource" / "dataset" / "test"

TSV_HEADERS = {
    "matching_results.tsv": b"source1_entity_id\tmatched_entity_ids",
    "candidate_pairs.tsv": b"source1_entity_id\tcandidate_entity_ids",
}
SRC_SUFFIXES = {".py"}                                  # anything else under src/ is skipped (and reported)
EXCLUDED_DIRS = {"__pycache__", ".ipynb_checkpoints", "cache", "data", "dataset", ".pytest_cache"}
CHUNK = 64 * 1024 * 1024


def fail(msg: str) -> None:
    """Print an error and exit with status 1."""
    print(f"ERROR: {msg}")
    sys.exit(1)


def human(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def check_line_endings(path: Path) -> None:
    """Assert the file has no b'\\r' anywhere and ends with b'\\n' (streamed in chunks)."""
    last = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            if b"\r" in block:
                fail(f"{path.name}: contains '\\r' (CRLF line endings); only '\\n' is allowed")
            last = block
    if not last.endswith(b"\n"):
        fail(f"{path.name}: does not end with '\\n'")


def read_id_lists(path: Path, header: bytes, keep: bool) -> dict:
    """Parse an id-list TSV and check every row; return {s1_id: raw id-list bytes} (or {s1_id: b''} if not keep)."""
    rows = {}
    with open(path, "rb") as f:
        first = f.readline().rstrip(b"\n")
        if first != header:
            fail(f"{path.name}: header {first!r} != {header!r}")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) != 2:
                fail(f"{path.name}:{lineno}: expected 2 tab-separated columns, got {len(parts)}")
            s1, ids = parts
            if not s1.startswith(b"S1-"):
                fail(f"{path.name}:{lineno}: bad source1_entity_id {s1[:40]!r}")
            if s1 in rows:
                fail(f"{path.name}:{lineno}: duplicate row for {s1.decode()}")
            if ids:
                lst = ids.split(b",")
                if len(set(lst)) != len(lst):
                    fail(f"{path.name}:{lineno}: duplicate id in the list of {s1.decode()}")
                bad = [x for x in lst if not (x.startswith(b"S2-") or x.startswith(b"S3-"))]
                if bad:
                    fail(f"{path.name}:{lineno}: non S2/S3 id {bad[0][:40]!r} for {s1.decode()}")
            rows[s1] = ids if keep else b""
    return rows


def check_tsvs(match_path: Path, cand_path: Path) -> dict:
    """Check both submission TSVs and the matches-subset-of-candidates rule; return summary statistics."""
    for p in (match_path, cand_path):
        check_line_endings(p)
    print("  line endings: '\\n' only, final newline present (both files)")
    matches = read_id_lists(match_path, TSV_HEADERS["matching_results.tsv"], keep=True)
    n_match_ids = sum(v.count(b",") + 1 for v in matches.values() if v)
    n_empty = sum(1 for v in matches.values() if not v)
    print(f"  matching_results.tsv: {len(matches):,} S1 rows, {n_match_ids:,} matched ids, "
          f"{n_empty:,} empty ({100 * n_empty / max(len(matches), 1):.2f}%)")
    seen, n_cand_ids, n_not_subset, example = set(), 0, 0, None
    with open(cand_path, "rb") as f:
        if f.readline().rstrip(b"\n") != TSV_HEADERS["candidate_pairs.tsv"]:
            fail("candidate_pairs.tsv: wrong header")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) != 2:
                fail(f"candidate_pairs.tsv:{lineno}: expected 2 columns, got {len(parts)}")
            s1, ids = parts
            if s1 in seen:
                fail(f"candidate_pairs.tsv:{lineno}: duplicate row for {s1.decode()}")
            seen.add(s1)
            cand = ids.split(b",") if ids else []
            if len(set(cand)) != len(cand):
                fail(f"candidate_pairs.tsv:{lineno}: duplicate id in the list of {s1.decode()}")
            if any(not (x.startswith(b"S2-") or x.startswith(b"S3-")) for x in cand):
                fail(f"candidate_pairs.tsv:{lineno}: non S2/S3 id for {s1.decode()}")
            n_cand_ids += len(cand)
            m = matches.get(s1)
            if m is None:
                fail(f"candidate_pairs.tsv:{lineno}: {s1.decode()} has no row in matching_results.tsv")
            if m:
                missing = set(m.split(b",")) - set(cand)
                if missing:
                    n_not_subset += 1
                    example = example or (s1.decode(), sorted(x.decode() for x in missing)[:3])
    if len(seen) != len(matches):
        fail(f"S1 rows differ: matching_results {len(matches):,} vs candidate_pairs {len(seen):,}")
    if n_not_subset:
        fail(f"{n_not_subset:,} S1 rows have matches outside their candidates, e.g. {example}")
    print(f"  candidate_pairs.tsv: {len(seen):,} S1 rows, {n_cand_ids:,} candidate ids "
          f"({n_cand_ids / max(len(seen), 1):.2f} per S1)")
    print("  matches subset of candidates: OK (every S1 row)")
    return {"s1_rows": len(matches), "matched_ids": n_match_ids, "candidate_ids": n_cand_ids}


def run_validator(match_path: Path, cand_path: Path) -> None:
    """Run the organiser validator (stdlib script) with --check-ids when the test data is present."""
    if not VALIDATOR.exists() or not TEST_DIR.exists():
        print("  organiser validator skipped (validator or test data not found)")
        return
    cmd = [sys.executable, str(VALIDATOR), "--matching", str(match_path), "--candidate", str(cand_path),
           "--test-dir", str(TEST_DIR)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    tail = (res.stdout + res.stderr).strip().splitlines()[-5:]
    for line in tail:
        print(f"  validator> {line}")
    if res.returncode != 0:
        fail("organiser validator did not PASS")


def plan_files(outputs: Path) -> list:
    """Return [(archive name, source path)] for every file that goes into the zip; report skipped files."""
    plan = [("output/matching_results.tsv", outputs / "matching_results.tsv"),
            ("output/candidate_pairs.tsv", outputs / "candidate_pairs.tsv")]
    src = CODE_DIR / "src"
    if not src.is_dir():
        fail(f"missing {src}")
    skipped = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if p.is_file():
            if p.suffix in SRC_SUFFIXES:
                plan.append((f"code/business_entity_resolution/src/{rel.as_posix()}", p))
            else:
                skipped.append(rel.as_posix())
    if skipped:
        print(f"  skipped non-source files under src/: {skipped}")
    plan += [("code/business_entity_resolution/README.md", CODE_DIR / "README.md"),
             ("code/business_entity_resolution/requirements.txt", CODE_DIR / "requirements.txt"),
             ("Documentation_template.md", DOC_FILE)]
    return plan


def check_structure(plan: list) -> None:
    """Assert every planned file exists and is non-empty, and that the required entries are present."""
    for arc, path in plan:
        if not path.is_file():
            fail(f"missing {path} (for {arc})")
        if path.stat().st_size == 0:
            fail(f"empty file {path} (for {arc})")
    names = {arc for arc, _ in plan}
    if not any(n.startswith("code/business_entity_resolution/src/") and n.endswith(".py") for n in names):
        fail("no Python source under code/business_entity_resolution/src/")
    todo = DOC_FILE.read_text(encoding="utf-8").count("TODO")
    if todo:
        print(f"  WARNING: Documentation_template.md still contains {todo} 'TODO' marker(s)")


def print_tree(names: list, sizes: dict) -> None:
    """Print the zip contents as an indented tree with file sizes (uncompressed / compressed)."""
    printed = set()
    for name in sorted(names, key=lambda n: (n.count("/") == 0, n)):
        parts = name.split("/")
        for depth in range(len(parts) - 1):
            d = "/".join(parts[:depth + 1])
            if d not in printed:
                print(f"  {'    ' * depth}{parts[depth]}/")
                printed.add(d)
        size, csize = sizes[name]
        print(f"  {'    ' * (len(parts) - 1)}{parts[-1]:<32} {human(size):>10}  ({human(csize)} zipped)")


def build_zip(plan: list, out: Path) -> None:
    """Write the zip (deflate, zip64), then re-open it, test CRCs and compare its name list with the plan."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for arc, path in plan:
            zf.write(path, arc)
    tmp.replace(out)
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad:
            fail(f"CRC error in {bad}")
        names = zf.namelist()
        sizes = {i.filename: (i.file_size, i.compress_size) for i in zf.infolist()}
    if sorted(names) != sorted(arc for arc, _ in plan):
        fail("zip name list differs from the plan")
    for n in names:
        if "__pycache__" in n or n.endswith((".pyc", ".parquet")) or n.startswith(("cache/", "data/")):
            fail(f"excluded content in zip: {n}")
    print(f"\nZip: {out}")
    print_tree(names, sizes)
    total = sum(s for s, _ in sizes.values())
    print(f"\n  {len(names)} files, {human(total)} uncompressed -> zip size {human(out.stat().st_size)} "
          f"({out.stat().st_size:,} bytes)")


def main() -> None:
    """Parse arguments, run all checks, build and verify the zip."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows console defaults to cp1252
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, type=Path,
                    help="folder holding matching_results.tsv and candidate_pairs.tsv (e.g. output/best/<run>)")
    ap.add_argument("--team", default="team", help="team name; the zip is <team>_submission.zip")
    ap.add_argument("--out", type=Path, help="zip path (default: dist/<team>_submission.zip in the repo)")
    ap.add_argument("--validate", action="store_true", help="also run the organiser validator (--test-dir)")
    ap.add_argument("--skip-tsv-checks", action="store_true", help="skip the TSV content checks (not recommended)")
    args = ap.parse_args()

    outputs = args.outputs if args.outputs.is_absolute() else (Path.cwd() / args.outputs)
    out = args.out or (ROOT / "dist" / f"{args.team}_submission.zip")
    print(f"Outputs: {outputs}")
    plan = plan_files(outputs)
    check_structure(plan)
    print("  structure: OK")
    if not args.skip_tsv_checks:
        check_tsvs(outputs / "matching_results.tsv", outputs / "candidate_pairs.tsv")
    if args.validate:
        run_validator(outputs / "matching_results.tsv", outputs / "candidate_pairs.tsv")
    build_zip(plan, out)


if __name__ == "__main__":
    main()
