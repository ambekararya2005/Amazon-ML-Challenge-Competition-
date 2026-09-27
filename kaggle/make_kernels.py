"""Generate the Kaggle kernel folders for a run plan from the template kernel (kaggle/kernel/).

    python kaggle/make_kernels.py blocking     # K1-K5: test blocking (France, India x2 shards, US) + train candidates
    python kaggle/make_kernels.py finalize     # K6: combine + stage-1 reduction + baseline + submission files
    python kaggle/make_kernels.py test_features  # K7a: test pair-feature table (stage-1 top-5 + all features)
    python kaggle/make_kernels.py submit_v2      # K7b: apply rule scorer v2 to the K7a table -> submission files
    python kaggle/make_kernels.py bench          # K8: geo-dense benchmark + pair table + scorer v2 eval
    python kaggle/make_kernels.py test_features_v3   # K9: v3 features on the K7a test pair table (per country)
    python kaggle/make_kernels.py model_v3       # K10: step-0 checks + v3 bench features + LightGBM train / eval
    python kaggle/make_kernels.py submit_v3      # K11: K10 models + K9 test features -> submission files
    python kaggle/make_kernels.py blocking_v4    # K12a-c: blocking v4 test candidates, one kernel per country
    python kaggle/make_kernels.py bench_v4       # K12d: blocking v4 bench + pair table + v3 features + model (v4)
    python kaggle/make_kernels.py test_features_v4   # K13: v4 test pair tables + v3 features
    python kaggle/make_kernels.py submit_v4      # K14: v4 models + v4 test features -> submission files

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
TEST_KERNELS = [f"{USERNAME}/{k}" for k in BLOCKING if k.startswith("amlc2026-test-")]
TEST_FEATURES = {
    "amlc2026-test-features": dict(STAGES=["combine_test", "pair_table_test"], REUSE_CACHE_SUBDIRS=None,
                                   REUSE_MODE="symlink", CACHE_IN_TMP=True, sources=TEST_KERNELS),
}
SUBMIT_V2 = {
    "amlc2026-submit-v2": dict(STAGES=["submit_v2"], REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink",
                               CACHE_IN_TMP=True,
                               sources=[f"{USERNAME}/amlc2026-test-features", f"{USERNAME}/amlc2026-test-france",
                                        f"{USERNAME}/amlc2026-bench"]),
}
BENCH = {
    "amlc2026-bench": dict(STAGES=["bench_build", "bench_block", "pair_table_bench", "scorer_eval"],
                           REUSE_CACHE_SUBDIRS=TRAIN_NEEDS, sources=[V1]),
}
K7A, K8 = f"{USERNAME}/amlc2026-test-features", f"{USERNAME}/amlc2026-bench"
K_FRANCE = f"{USERNAME}/amlc2026-test-france"          # holds cache/cand/test lookups + a copy of cache/norm
TEST_FEATURES_V3 = {
    "amlc2026-test-features-v3": dict(STAGES=["features_v3_test"], REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink",
                                      CACHE_IN_TMP=True, sources=[K7A, K_FRANCE, K8]),
}
MODEL_V3 = {
    "amlc2026-model-v3": dict(STAGES=["data_checks", "features_v3_bench", "model_train"], REUSE_CACHE_SUBDIRS=None,
                              REUSE_MODE="symlink", CACHE_IN_TMP=True, sources=[K8]),
}
SUBMIT_V3 = {
    "amlc2026-submit-v3": dict(STAGES=["model_submit"], REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink",
                               CACHE_IN_TMP=True,
                               sources=[f"{USERNAME}/amlc2026-model-v3", f"{USERNAME}/amlc2026-test-features-v3",
                                        K_FRANCE, K8]),
}
V4 = {"FEATURE_VARIANT": "v4"}
BLOCKING_V4 = {f"amlc2026-blocking-v4-{c.lower()}": dict(STAGES=[f"blocking_v4_test@{c}"], REUSE_CACHE_SUBDIRS=None,
                                                          REUSE_MODE="symlink", CACHE_IN_TMP=True, sources=[K8])
               for c in ("France", "India", "US")}
BENCH_V4 = {
    "amlc2026-bench-v4": dict(STAGES=["blocking_v4_bench", "pair_table_bench", "features_v3_bench", "model_train"],
                              REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink", CACHE_IN_TMP=True, EXTRA_ENV=V4,
                              sources=[K8]),
}
TEST_FEATURES_V4 = {
    "amlc2026-test-features-v4": dict(STAGES=["pair_table_test", "features_v3_test"], REUSE_CACHE_SUBDIRS=None,
                                      REUSE_MODE="symlink", CACHE_IN_TMP=True, EXTRA_ENV={**V4, "N_WORKERS": 3},
                                      sources=[f"{USERNAME}/{k}" for k in BLOCKING_V4] + [K8]),
}
SUBMIT_V4 = {
    "amlc2026-submit-v4": dict(STAGES=["model_submit"], REUSE_CACHE_SUBDIRS=None, REUSE_MODE="symlink",
                               CACHE_IN_TMP=True, EXTRA_ENV=V4,
                               sources=[f"{USERNAME}/amlc2026-bench-v4", f"{USERNAME}/amlc2026-test-features-v4",
                                        K_FRANCE, K8]),
}
PLANS = {"blocking": BLOCKING, "finalize": FINALIZE, "test_features": TEST_FEATURES, "submit_v2": SUBMIT_V2,
         "bench": BENCH, "test_features_v3": TEST_FEATURES_V3, "model_v3": MODEL_V3, "submit_v3": SUBMIT_V3,
         "blocking_v4": BLOCKING_V4, "bench_v4": BENCH_V4, "test_features_v4": TEST_FEATURES_V4,
         "submit_v4": SUBMIT_V4}


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
