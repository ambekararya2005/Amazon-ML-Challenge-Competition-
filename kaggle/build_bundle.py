"""Zip the pipeline code into kaggle/code_bundle/ for the private Kaggle dataset <USERNAME>/amlc2026-code.

Packs code/business_entity_resolution/{src, tests, requirements.txt} (no __pycache__)
into kaggle/code_bundle/amlc2026_code.zip, under the folder business_entity_resolution/,
plus BUNDLE_INFO.json (build time, git commit, dirty flag, file list with SHA-256) so the
kernel can tell which code it ran. Also (re)writes dataset-metadata.json.

    python kaggle/build_bundle.py
"""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = KAGGLE_DIR.parent
PROJECT_DIR = REPO_ROOT / "code" / "business_entity_resolution"
BUNDLE_DIR = KAGGLE_DIR / "code_bundle"
ZIP_NAME = "amlc2026_code.zip"
ARC_ROOT = "business_entity_resolution"
INCLUDE = ("src", "tests", "requirements.txt")
USERNAME = "aryaambekar"
DATASET_SLUG = "amlc2026-code"


def git(*args: str) -> str:
    """Return the stripped stdout of a git command in the repo ('' if git fails)."""
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def bundle_files() -> list:
    """Return the sorted list of files to pack (paths relative to PROJECT_DIR)."""
    files = []
    for name in INCLUDE:
        p = PROJECT_DIR / name
        if p.is_file():
            files.append(p.relative_to(PROJECT_DIR))
        else:
            files += [f.relative_to(PROJECT_DIR) for f in p.rglob("*")
                      if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    return sorted(files, key=lambda f: f.as_posix())


def write_metadata() -> Path:
    """Write kaggle/code_bundle/dataset-metadata.json for the private code dataset."""
    meta = {"title": DATASET_SLUG, "id": f"{USERNAME}/{DATASET_SLUG}",
            "subtitle": "Amazon ML Challenge 2026 pipeline code (private)",
            "licenses": [{"name": "other"}]}
    path = BUNDLE_DIR / "dataset-metadata.json"
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return path


def build() -> Path:
    """Build the zip and metadata; return the zip path."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    files = bundle_files()
    info = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", str(PROJECT_DIR))),
            "files": {f.as_posix(): hashlib.sha256((PROJECT_DIR / f).read_bytes()).hexdigest()[:16]
                      for f in files}}
    out = BUNDLE_DIR / ZIP_NAME
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(PROJECT_DIR / f, f"{ARC_ROOT}/{f.as_posix()}")
        z.writestr(f"{ARC_ROOT}/BUNDLE_INFO.json", json.dumps(info, indent=2) + "\n")
    tmp.replace(out)
    write_metadata()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(files)} files, "
          f"commit {info['git_commit'][:8] or '?'}{' +dirty' if info['git_dirty'] else ''})")
    return out


if __name__ == "__main__":
    build()
