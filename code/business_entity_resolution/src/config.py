"""Paths, seeds, CPU counts and shared constants for the entity-resolution pipeline.

Every directory is resolved once, at import, in this order:
    1. CLI flag   --data-root / --cache-root / --output-root / --log-root  (any stage)
    2. env var    DATA_ROOT / CACHE_ROOT / OUTPUT_ROOT / LOG_ROOT
                  (legacy aliases BER_DATA_DIR / BER_CACHE_DIR / BER_OUTPUT_DIR / BER_LOG_DIR)
    3. default    the repo layout (student_resource/dataset, cache/, output/, logs/)
If the default data folder has no train_source1.tsv and /kaggle/input exists, the
data folder is auto-detected by searching /kaggle/input for train_source1.tsv.
Resolved roots are written back to os.environ so spawned worker processes see them.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42

# code/business_entity_resolution/
PROJECT_DIR = Path(__file__).resolve().parents[1]
# repo root (== zip root): contains code/, output/, cache/, logs/
REPO_ROOT = PROJECT_DIR.parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
DATA_MARKER = "train_source1.tsv"

# name -> (CLI flag, env var, legacy env var, local default)
PATH_SETTINGS = {
    "data": ("--data-root", "DATA_ROOT", "BER_DATA_DIR", REPO_ROOT / "student_resource" / "dataset"),
    "cache": ("--cache-root", "CACHE_ROOT", "BER_CACHE_DIR", REPO_ROOT / "cache"),
    "output": ("--output-root", "OUTPUT_ROOT", "BER_OUTPUT_DIR", REPO_ROOT / "output"),
    "log": ("--log-root", "LOG_ROOT", "BER_LOG_DIR", REPO_ROOT / "logs"),
}


def add_path_args(ap: argparse.ArgumentParser) -> None:
    """Declare the path flags on a stage's parser (values are already applied by this module at import)."""
    for name, (flag, env, _, default) in PATH_SETTINGS.items():
        ap.add_argument(flag, metavar="DIR", help=f"{name} root (env {env}; default {default})")


def _cli_paths(argv: list) -> dict:
    """Return {name: value} for path flags present in ``argv``; everything else is ignored."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name, (flag, *_rest) in PATH_SETTINGS.items():
        ap.add_argument(flag, dest=name)
    known, _ = ap.parse_known_args(argv)
    return {k: v for k, v in vars(known).items() if v}


def find_data_root(search_root: Path = KAGGLE_INPUT) -> "Path | None":
    """Return the dataset folder under ``search_root`` holding train_source1.tsv, or None.

    For the organiser layout (<root>/train/train_source1.tsv) this is <root>; for a
    flat upload (<root>/train_source1.tsv) it is the folder itself.
    """
    if not search_root.is_dir():
        return None
    for dirpath, _, files in sorted(os.walk(search_root, followlinks=True)):
        if DATA_MARKER in files:
            p = Path(dirpath)
            return p.parent if p.name == "train" else p
    return None


def resolve_paths(argv: list = None, environ: dict = None) -> dict:
    """Resolve the four roots (CLI flag > env var > legacy env var > default, plus Kaggle data auto-detect)."""
    argv = sys.argv[1:] if argv is None else argv
    environ = os.environ if environ is None else environ
    cli = _cli_paths(argv)
    out = {}
    for name, (_, env, legacy, default) in PATH_SETTINGS.items():
        value = cli.get(name) or environ.get(env) or environ.get(legacy)
        out[name] = Path(value) if value else Path(default)
    explicit_data = bool(cli.get("data") or environ.get("DATA_ROOT") or environ.get("BER_DATA_DIR"))
    if not explicit_data and not any((out["data"] / sub / DATA_MARKER).exists() for sub in ("train", ".")):
        found = find_data_root()
        if found is not None:
            out["data"] = found
    return out


_PATHS = resolve_paths()
DATA_DIR = _PATHS["data"]
CACHE_DIR = _PATHS["cache"]
OUTPUT_DIR = _PATHS["output"]
LOG_DIR = _PATHS["log"]
for _name, (_, _env, _, _) in PATH_SETTINGS.items():
    os.environ[_env] = str(_PATHS[_name])


def cpu_count() -> int:
    """Return the CPUs this process may use (affinity-aware on Linux), at least 1."""
    if hasattr(os, "process_cpu_count"):
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)


def env_int(name: str, default: int) -> int:
    """Return int(os.environ[name]) if set, else ``default``."""
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    """Return float(os.environ[name]) if set, else ``default``."""
    value = os.environ.get(name)
    return float(value) if value else default


# Candidate / feature variant: "v1" (blocking v1, top-5) or "v4" (blocking v4: cross-script dictionary, pass D,
# adaptive top-k). v4 tables live in their own folders (cache/bench_v4, output/features_v4, output/features_v3_v4,
# output/models_v4); v1 tables are never overwritten.
FEATURE_VARIANT = os.environ.get("FEATURE_VARIANT", "v1")

# Thread count for libraries with their own threading (sparse_dot_topn, rapidfuzz, lightgbm).
N_THREADS = env_int("N_THREADS", cpu_count())

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

TEXT_COLS = ("business_name", "business_address", "country")
SOURCE_COLS = ("entity_id",) + TEXT_COLS
GT_COLS = ("source1_entity_id", "matched_entity_ids")

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def set_seeds(seed: int = SEED) -> None:
    """Seed Python's and NumPy's global random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
