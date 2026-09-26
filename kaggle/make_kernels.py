"""Generate the Kaggle kernel folders for a run plan from the template kernel (kaggle/kernel/).

    python kaggle/make_kernels.py blocking     # K1-K5: test blocking (France, India x2 shards, US) + train candidates
    python kaggle/make_kernels.py finalize     # K6: combine + stage-1 reduction + baseline + submission files

Writes kaggle/kernels/<slug>/{run_pipeline_kaggle.py, kernel-metadata.json}. Each kernel gets its own
STAGES and reuse settings and kernel_sources. Push each with:  kaggle kernels push -p kaggle/kernels/<slug>
Kaggle allows 5 concurrent batch CPU sessions per account (at the time of writing).
"""
import argparse
import json
import re
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = KAGGLE_DIR / "kernel"
OUT_DIR = KAGGLE_DIR / "kernels"
USERNAME = "aryaambekar"
V1 = f"{USERNAME}/amlc2026-pipeline"          # run v1: raw + norm + split cache
TEST_ONLY = ["norm"]
TRAIN_NEEDS = ["norm", "raw", "split.parquet"]

# slug -> settings substituted into the template (Python literals) + kernel_sources
BLOCKING = {
    "amlc2026-test-france": dict(STAGES=["block_test@France"], REUSE_CACHE_SUBDIRS=TEST_ONLY, sources=[V1]),
    "amlc2026-test-india-1": dict(STAGES=["block_test@India@1/2"], REUSE_CACHE_SUBDIRS=TEST_ONLY, sources=[V1]),
    "amlc2026-test-india-2": dict(STAGES=["block_test@India@2/2"], REUSE_CACHE_SUBDIRS=TEST_ONLY, sources=[V1]),
    "amlc2026-test-us": dict(STAGES=["block_test@US"], REUSE_CACHE_SUBDIRS=TEST_ONLY, sources=[V1]),
    "amlc2026-train-cands": dict(STAGES=["block_train", "recall"], REUSE_CACHE_SUBDIRS=TRAIN_NEEDS, sources=[V1]),
}
FINALIZE = {
    "amlc2026-finalize": dict(STAGES=["combine_test", "finalize"], REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink",
                              CACHE_IN_TMP=True, sources=[f"{USERNAME}/{k}" for k in BLOCKING]),
}
PLANS = {"blocking": BLOCKING, "finalize": FINALIZE}


def substitute(script: str, name: str, value) -> str:
    """Replace the top-level assignment ``name = ...`` in the template with ``value`` (Python literal)."""
    out, n = re.subn(rf"^{name} = .*$", f"{name} = {value!r}", script, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"template script has no top-level '{name} = ...' line")
    return out


def make(slug: str, settings: dict) -> Path:
    """Write one kernel folder; return its path."""
    out = OUT_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    script = (TEMPLATE_DIR / "run_pipeline_kaggle.py").read_text(encoding="utf-8")
    for key, value in settings.items():
        if key != "sources":
            script = substitute(script, key, value)
    with open(out / "run_pipeline_kaggle.py", "w", encoding="utf-8", newline="") as f:
        f.write(script)
    meta = json.loads((TEMPLATE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"{USERNAME}/{slug}", title=slug, kernel_sources=settings["sources"])
    with open(out / "kernel-metadata.json", "w", encoding="utf-8", newline="") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return out


def main() -> None:
    """Parse the plan name and write its kernel folders."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", choices=sorted(PLANS))
    args = ap.parse_args()
    for slug, settings in PLANS[args.plan].items():
        print("wrote", make(slug, settings), settings["STAGES"])


if __name__ == "__main__":
    main()
