"""Run the whole v4 pipeline end to end, in order, as separate stage processes (the submitted "v4-safe" path).

    python -m src.run_pipeline --data-root <organiser data> --work <work dir>          # full reproduction
    python -m src.run_pipeline --smoke --data-root <organiser data> --work <work dir>  # small-sample smoke test

Stages (each is also a stand-alone CLI, see README.md):
   1 prepare_data              TSV -> Parquet (+ train_pairs)
   2 split                     random hold-out table (used by the early experiments / v1 recall)
   3 normalize                 per-country text normalisation
   4 benchmark build / block   geo-dense benchmark (5 region folds) + v1 candidate union
   5 pair_table bench (v1) + scorer_v2 eval     rule score v2 (config feeds the v2_score feature)
   6 blocking_v4 bench         cross-script dictionary, passes A / C / D, adaptive top-k on the benchmark
   7 pair_table + features_v3 bench (v4)        labelled pair features
   8 model_lgb train (v4)      stage 1 / stage 2 / has-match, leave-one-fold-out over folds 1-4; fold 0 = report
   9 blocking test (France)    test lookup tables (lookup_s1 / queries)
  10 blocking_v4 test          one run per test country
  11 pair_table + features_v3 test (v4)
  12 model_lgb submit          one run per country (mean of the 4 CV fold models, one-to-one, has-match, decoder)
  13 model_lgb assemble        -> <work>/output/final/{matching_results,candidate_pairs}.tsv + validator
Roots: <work>/{cache,output,logs}. --smoke first writes a hash sample of the data (src.make_sample) to <work>/data
and uses it. Every stage skips work already cached, so a failed run can be resumed with the same command.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def run(args: list, env: dict, log) -> None:
    """Run one stage module (``python -m ...``) from the project folder; abort on failure."""
    t = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", *args], cwd=PROJECT_DIR, env=env)
    msg = f"{' '.join(args)}: rc {r.returncode}, {time.time() - t:.0f}s"
    log.write(msg + "\n")
    log.flush()
    print(msg, flush=True)
    if r.returncode != 0:
        sys.exit(f"stage failed: {' '.join(args)}")


def test_countries(cache: Path) -> list:
    """Return the country labels of test Source 1 (an open set; read from the cached raw table)."""
    import pyarrow.parquet as pq
    t = pq.read_table(cache / "raw" / "test_source1.parquet", columns=["country"]).to_pandas()
    return sorted(t["country"].unique().tolist())


def main() -> None:
    """Parse flags, prepare the roots and run every stage in order."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path, help="organiser data (train/ and test/ folders)")
    ap.add_argument("--work", required=True, type=Path, help="work folder for cache / output / logs")
    ap.add_argument("--smoke", action="store_true", help="run on a small hash sample of the data")
    ap.add_argument("--train-share", type=float, default=0.03, help="--smoke: share of train S1 kept")
    ap.add_argument("--test-share", type=float, default=0.01, help="--smoke: share of each test source kept")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = a.data_root.resolve()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    log = open(work / "run_pipeline.log", "a", encoding="utf-8", newline="")
    if a.smoke:
        sample = work / "data"
        if not (sample / "test" / "test_source1.tsv").exists():
            run(["src.make_sample", "--src", str(data), "--dst", str(sample), "--train-share", str(a.train_share),
                 "--test-share", str(a.test_share)], env, log)
        data = sample
    env.update(DATA_ROOT=str(data), CACHE_ROOT=str(work / "cache"), OUTPUT_ROOT=str(work / "output"),
               LOG_ROOT=str(work / "logs"), FEATURE_VARIANT="v1")
    v4 = dict(env, FEATURE_VARIANT="v4")
    run(["src.prepare_data"], env, log)
    run(["src.split"], env, log)
    run(["src.normalize", "--stage", "normalize"], env, log)
    run(["src.benchmark", "--stage", "build"], env, log)
    run(["src.benchmark", "--stage", "block"], env, log)
    run(["src.pair_table", "--split", "bench"], env, log)
    run(["src.scorer_v2", "--stage", "eval"], env, log)
    run(["src.blocking_v4", "--split", "bench"], v4, log)
    run(["src.pair_table", "--split", "bench"], v4, log)
    run(["src.features_v3", "--split", "bench"], v4, log)
    run(["src.model_lgb", "--stage", "train"], v4, log)
    countries = test_countries(work / "cache")
    run(["src.blocking", "--stage", "block", "--split", "test", "--country", countries[0], "--allow-low-ram"], env, log)
    for c in countries:
        run(["src.blocking_v4", "--split", "test", "--country", c], v4, log)
    run(["src.pair_table", "--split", "test"], v4, log)
    run(["src.features_v3", "--split", "test"], v4, log)
    for c in countries:
        run(["src.model_lgb", "--stage", "submit"], dict(v4, SUBMIT_COUNTRY=c, SUBMIT_PARTS="1"), log)
    run(["src.model_lgb", "--stage", "assemble"], dict(v4, FINAL_SUBDIR="final"), log)
    print(f"\nDONE: {work / 'output' / 'final'}", flush=True)


if __name__ == "__main__":
    main()
