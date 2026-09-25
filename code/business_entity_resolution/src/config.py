"""Paths, seeds and shared constants for the entity-resolution pipeline.

Every directory can be overridden with an environment variable so the code runs
unchanged from the repo checkout or from the unpacked submission zip.
"""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# code/business_entity_resolution/
PROJECT_DIR = Path(__file__).resolve().parents[1]
# repo root (== zip root): contains code/, output/, cache/, logs/
REPO_ROOT = PROJECT_DIR.parents[1]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "student_resource" / "dataset"))
CACHE_DIR = Path(os.environ.get("BER_CACHE_DIR", REPO_ROOT / "cache"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_ROOT / "output"))
LOG_DIR = Path(os.environ.get("BER_LOG_DIR", REPO_ROOT / "logs"))

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
