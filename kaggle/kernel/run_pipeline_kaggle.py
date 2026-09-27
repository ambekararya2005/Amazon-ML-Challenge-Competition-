"""Kaggle runner for the Amazon ML Challenge 2026 pipeline (heavy stages only; code is developed locally).

Inputs (attached via kernel-metadata.json):
    <USERNAME>/amlc2026-data   the organiser TSVs (found by searching /kaggle/input for train_source1.tsv)
    <USERNAME>/amlc2026-code   kaggle/code_bundle/amlc2026_code.zip (zip, or the folder Kaggle unpacked it to)
    optional: previous runs (kernel_sources) -> their cache/ and logs/ are merged in (all of them)

What it does:
    a) prints RAM, CPU, disk and Python versions;
    b) copies the code to /kaggle/working/code and pip installs only the pinned packages that are
       missing or at a different version on the Kaggle image;
    c) runs the STAGES below in order (stops at the first failure);
    d) cache -> /kaggle/working/cache, logs -> /kaggle/working/logs;
    e) copies the small result files to /kaggle/working/results (+ kernel_stages.json, experiments_entry.md).
The code copy is deleted at the end so the kernel output holds only cache/, logs/, results/.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ============================== edit per run ==============================
STAGES = ["load", "normalize", "split", "block_benchmark"]
# "<stage>@<country>" runs a blocking stage for one country only, e.g. "block_test@France";
# "<stage>@<country>@k/n" also restricts it to query shard k of n, e.g. "block_test@India@1/2"
REUSE_PREVIOUS_CACHE = True     # merge every attached previous run's cache/ + logs/ (existing files kept)
REUSE_CACHE_SUBDIRS = None      # None = the whole previous cache; e.g. ["norm"] = only cache/norm
REUSE_MODE = "copy"             # "copy" (output is self-contained) or "symlink" (fast, no disk; not reusable)
CACHE_IN_TMP = False            # True: cache in /tmp (not saved as kernel output; avoids the 20 GB limit)
MIN_FREE_GB = 7.0               # blocking RAM gate (free GB required before blocking starts)
EXTRA_ENV = {}                  # extra environment for every stage, e.g. {"FEATURE_VARIANT": "v4"}
# ==========================================================================

STAGE_COMMANDS = {
    "load": ["-m", "src.prepare_data"],
    "normalize": ["-m", "src.normalize", "--stage", "normalize"],
    "split": ["-m", "src.split"],
    "block_benchmark": ["-m", "src.blocking", "--stage", "bench"],
    "block_train": ["-m", "src.blocking", "--stage", "block", "--split", "train"],
    "recall": ["-m", "src.blocking", "--stage", "recall", "--split", "train"],
    "block_test": ["-m", "src.blocking", "--stage", "block", "--split", "test"],
    "combine_train": ["-m", "src.blocking", "--stage", "combine", "--split", "train"],
    "combine_test": ["-m", "src.blocking", "--stage", "combine", "--split", "test"],
    "finalize": ["-m", "src.finalize"],
    "pair_table_test": ["-m", "src.pair_table", "--split", "test"],
    "bench_build": ["-m", "src.benchmark", "--stage", "build"],
    "bench_block": ["-m", "src.benchmark", "--stage", "block"],
    "pair_table_bench": ["-m", "src.pair_table", "--split", "bench"],
    "scorer_eval": ["-m", "src.scorer_v2", "--stage", "eval"],
    "submit_v2": ["-m", "src.scorer_v2", "--stage", "submit"],
    "data_checks": ["-m", "src.data_checks"],
    "features_v3_bench": ["-m", "src.features_v3", "--split", "bench"],
    "features_v3_test": ["-m", "src.features_v3", "--split", "test"],
    "model_train": ["-m", "src.model_lgb", "--stage", "train"],
    "model_submit": ["-m", "src.model_lgb", "--stage", "submit"],
    "blocking_v4_bench": ["-m", "src.blocking_v4", "--split", "bench"],
    "blocking_v4_test": ["-m", "src.blocking_v4", "--split", "test"],
    "tests": ["-m", "unittest"],
}
COUNTRY_STAGES = ("block_train", "block_test", "blocking_v4_test")

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE_DIR = WORK / "code"
CACHE_DIR = Path("/tmp/amlc_cache") if CACHE_IN_TMP else WORK / "cache"
LOG_DIR = WORK / "logs"
OUTPUT_DIR = WORK / "output"
RESULTS_DIR = WORK / "results"
BUNDLE_ZIP = "amlc2026_code.zip"
BUNDLE_INFO = "BUNDLE_INFO.json"
PROJECT_NAME = "business_entity_resolution"
RUN_MARKER = "RUN_INFO.json"
DATA_MARKER = "train_source1.tsv"
RESULT_FILES = ["blocking_bench.json", "blocking_recall.json", "blocking_misses.txt", "stage_metrics.jsonl",
                "data_summary.json", "normalize_summary.json", "prepare_data.log", "normalize.log",
                "split.log", "blocking.log", "combine_check.json", "stage1_reduction.json",
                "baseline_report.json", "baseline_report.md", "validate_submission.txt", "finalize.log",
                "pair_table.log", "scorer_v2.log", "submit_v2_report.json",
                "benchmark.log", "bench_summary.json", "scorer_v2_report.json", "scorer_v2_config.json",
                "decoy_examples.txt"]
SMALL_LOG_BYTES = 2 * 1024 ** 2   # every other log file up to this size is copied to results/ as well
GB = 1024 ** 3


# ------------------------------------------------------------------ environment
def mem_gb() -> dict:
    """Return total / available / used system RAM in GB (from /proc/meminfo, no dependencies)."""
    info = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) * 1024
    total, avail = info["MemTotal"], info.get("MemAvailable", info["MemFree"])
    return {"total": round(total / GB, 2), "available": round(avail / GB, 2), "used": round((total - avail) / GB, 2)}


def usable_cpus() -> int:
    """Return the CPUs this process may use."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def print_environment() -> dict:
    """Print and return RAM, CPU, disk and Python/platform versions."""
    disk = shutil.disk_usage(WORK)
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "cpu_usable": usable_cpus(), "ram_gb": mem_gb(),
           "disk_working_free_gb": round(disk.free / GB, 1), "started_utc": now()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            env["cpu_model"] = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "?")
    except OSError:
        pass
    print("=" * 70, "\nENVIRONMENT\n" + json.dumps(env, indent=2), "\n" + "=" * 70, flush=True)
    return env


def now() -> str:
    """Return the current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ inputs
def find_data_root() -> Path:
    """Return the dataset folder under /kaggle/input holding train_source1.tsv (organiser or flat layout)."""
    for dirpath, _, files in sorted(os.walk(INPUT, followlinks=True)):
        if DATA_MARKER in files:
            p = Path(dirpath)
            return p.parent if p.name == "train" else p
    raise FileNotFoundError(f"no {DATA_MARKER} under {INPUT}; attach the data dataset")


def install_code() -> dict:
    """Copy the newest code bundle found under /kaggle/input to CODE_DIR; return its BUNDLE_INFO."""
    candidates = []  # (built_utc, kind, path)
    for dirpath, _, files in os.walk(INPUT, followlinks=True):
        if BUNDLE_ZIP in files:
            z = Path(dirpath) / BUNDLE_ZIP
            with zipfile.ZipFile(z) as zf:
                info = json.loads(zf.read(f"{PROJECT_NAME}/{BUNDLE_INFO}"))
            candidates.append((info["built_utc"], "zip", z))
        if BUNDLE_INFO in files and (Path(dirpath) / "src").is_dir():
            info = json.loads((Path(dirpath) / BUNDLE_INFO).read_text(encoding="utf-8"))
            candidates.append((info["built_utc"], "dir", Path(dirpath)))
    if not candidates:
        raise FileNotFoundError(f"no {BUNDLE_ZIP} or unpacked bundle under {INPUT}; attach the code dataset")
    _, kind, src = max(candidates)
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            zf.extractall(CODE_DIR)
    else:
        shutil.copytree(src, CODE_DIR / PROJECT_NAME)
    info = json.loads((CODE_DIR / PROJECT_NAME / BUNDLE_INFO).read_text(encoding="utf-8"))
    print(f"code bundle: {src} ({kind}) built {info['built_utc']} commit {info['git_commit'][:8]}"
          f"{' +dirty' if info['git_dirty'] else ''}", flush=True)
    return info


def install_requirements(req: Path) -> dict:
    """pip install only the pinned requirements that are missing or at another version; return what changed."""
    from importlib.metadata import PackageNotFoundError, version
    todo, report = [], {}
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, want = (s.strip() for s in line.split("==", 1))
        try:
            have = version(name)
        except PackageNotFoundError:
            have = None
        report[name] = {"kaggle": have, "pinned": want}
        if have != want:
            todo.append(f"{name}=={want}")
    print("requirements (kaggle image vs pinned):", flush=True)
    for name, r in report.items():
        print(f"  {name:<20} {str(r['kaggle']):<14} {r['pinned']:<14} {'INSTALL' if r['kaggle'] != r['pinned'] else 'ok'}")
    if todo:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-input", *todo], check=True)
    return {"installed": todo, "versions": report}


def find_previous_runs() -> list:
    """Return every attached previous-run output folder (holds RUN_INFO.json and cache/), newest first."""
    runs = []
    for dirpath, _, files in os.walk(INPUT, followlinks=True):
        if RUN_MARKER in files and (Path(dirpath) / "cache").is_dir():
            info = json.loads((Path(dirpath) / RUN_MARKER).read_text(encoding="utf-8"))
            runs.append((info.get("finished_utc", ""), Path(dirpath)))
    return [p for _, p in sorted(runs, reverse=True)]


def reuse_previous(prev: Path) -> dict:
    """Copy a previous run's cache/ and logs/ into /kaggle/working (existing files are kept)."""
    copied = {}
    for name, dst in (("cache", CACHE_DIR), ("logs", LOG_DIR)):
        src, n, size = prev / name, 0, 0
        if not src.is_dir():
            continue
        for f in src.rglob("*"):
            rel = f.relative_to(src)
            if name == "cache" and REUSE_CACHE_SUBDIRS is not None and rel.parts[0] not in REUSE_CACHE_SUBDIRS:
                continue
            target = dst / rel
            if f.is_file() and not target.exists() and not f.name.endswith(".tmp"):
                target.parent.mkdir(parents=True, exist_ok=True)
                if REUSE_MODE == "symlink" and name == "cache":
                    target.symlink_to(f)
                else:
                    shutil.copy2(f, target)
                n, size = n + 1, size + f.stat().st_size
        copied[name] = {"files": n, "gb": round(size / GB, 2)}
    print(f"reused previous run {prev}: {copied}", flush=True)
    return copied


# ------------------------------------------------------------------ stages
class MemSampler:
    """Background sampler of system RAM used and the stage process-tree RSS (GB peaks)."""

    def __init__(self, pid: int, interval: float = 0.5):
        """Store the root pid to follow and the sampling interval."""
        self.pid, self.interval = pid, interval
        self.peak_system_used, self.peak_tree_rss = 0.0, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tree_rss(self) -> float:
        """Return the RSS (GB) of the process and all its children (0 if psutil is unavailable)."""
        try:
            import psutil
            p = psutil.Process(self.pid)
            return sum(q.memory_info().rss for q in [p, *p.children(recursive=True)]) / GB
        except Exception:
            return 0.0

    def _run(self) -> None:
        """Sample until stopped."""
        while not self._stop.is_set():
            self.peak_system_used = max(self.peak_system_used, mem_gb()["used"])
            self.peak_tree_rss = max(self.peak_tree_rss, self._tree_rss())
            self._stop.wait(self.interval)

    def start(self) -> "MemSampler":
        """Start sampling."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling."""
        self._stop.set()
        self._thread.join()


def stage_env(data_root: Path) -> dict:
    """Return the environment for stage subprocesses (all roots and CPU counts explicit)."""
    env = dict(os.environ)
    env.update(DATA_ROOT=str(data_root), CACHE_ROOT=str(CACHE_DIR), OUTPUT_ROOT=str(OUTPUT_DIR),
               LOG_ROOT=str(LOG_DIR), N_THREADS=str(usable_cpus()), MIN_FREE_GB=str(MIN_FREE_GB),
               PYTHONUNBUFFERED="1", PYTHONHASHSEED="42")
    env.update({k: str(v) for k, v in EXTRA_ENV.items()})
    return env


def stage_command(name: str) -> list:
    """Return the module arguments for a stage; "<stage>@<country>[@k/n]" adds --country [and --shard]."""
    base, *rest = name.split("@")
    if rest and base not in COUNTRY_STAGES:
        raise SystemExit(f"'@country' is only valid for {COUNTRY_STAGES}, got {name!r}")
    extra = ["--country", rest[0]] if rest else []
    if len(rest) > 1:
        extra += ["--shard", rest[1]]
    return STAGE_COMMANDS[base] + extra


def run_stage(name: str, env: dict) -> dict:
    """Run one stage as a subprocess; return its status, seconds and peak RAM."""
    cmd = [sys.executable, *stage_command(name)]
    print(f"\n{'=' * 70}\nSTAGE {name}: {' '.join(cmd[1:])}   (ram {mem_gb()})\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=CODE_DIR / PROJECT_NAME, env=env)
    sampler = MemSampler(proc.pid).start()
    rc = proc.wait()
    sampler.stop()
    rec = {"stage": name, "returncode": rc, "status": "ok" if rc == 0 else "FAILED",
           "seconds": round(time.perf_counter() - t0, 1),
           "peak_tree_rss_gb": round(sampler.peak_tree_rss, 2),
           "peak_system_used_gb": round(sampler.peak_system_used, 2)}
    print(f"STAGE {name} -> {rec}", flush=True)
    return rec


# ------------------------------------------------------------------ results
def experiments_entry(env: dict, bundle: dict, records: list, reused: dict) -> str:
    """Return a Markdown entry for logs/experiments.md summarising this run."""
    lines = [f"## {env['started_utc'][:10]} — Kaggle run: {', '.join(STAGES)}",
             f"- Machine: {env['cpu_usable']} CPUs ({env.get('cpu_model', '?')}), RAM {env['ram_gb']['total']} GB, "
             f"Python {env['python']}. Code commit {bundle['git_commit'][:8]}{' +dirty' if bundle['git_dirty'] else ''}.",
             f"- Reused previous cache: {reused or 'no'}."]
    for r in records:
        lines.append(f"- `{r['stage']}`: {r['status']} in {r['seconds'] / 60:.1f} min, peak process RSS "
                     f"{r['peak_tree_rss_gb']} GB, peak system RAM used {r['peak_system_used_gb']} GB.")
    bench = LOG_DIR / "blocking_bench.json"
    if bench.exists() and "block_benchmark" in STAGES:
        b = json.loads(bench.read_text(encoding="utf-8"))
        for split, p in b["projection"].items():
            lines.append(f"- Projection {split}: {p['minutes']} min, peak RSS {p['peak_rss_gb']} GB "
                         f"(by country: " + ", ".join(f"{c} {d['minutes']} min" for c, d in p["by_country"].items()) + ").")
    return "\n".join(lines) + "\n"


def collect_results(env: dict, bundle: dict, records: list, reused: dict, reqs: dict) -> None:
    """Copy small result files to RESULTS_DIR and write kernel_stages.json + experiments_entry.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        if (LOG_DIR / name).exists():
            shutil.copy2(LOG_DIR / name, RESULTS_DIR / name)
    for f in LOG_DIR.glob("*"):
        if f.is_file() and f.stat().st_size <= SMALL_LOG_BYTES and not (RESULTS_DIR / f.name).exists():
            shutil.copy2(f, RESULTS_DIR / f.name)
    summary = {"environment": env, "bundle": {k: v for k, v in bundle.items() if k != "files"},
               "requirements": reqs, "reused_previous": reused, "stages_requested": STAGES,
               "stages": records, "finished_utc": now()}
    for path, text in ((RESULTS_DIR / "kernel_stages.json", json.dumps(summary, indent=2) + "\n"),
                       (RESULTS_DIR / "experiments_entry.md", experiments_entry(env, bundle, records, reused)),
                       (WORK / RUN_MARKER, json.dumps({"finished_utc": summary["finished_utc"],
                                                       "stages": records}, indent=2) + "\n")):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(f"\nresults in {RESULTS_DIR}: {sorted(p.name for p in RESULTS_DIR.iterdir())}", flush=True)


def main() -> None:
    """Set up code and inputs, run STAGES, and always collect results."""
    unknown = [s for s in STAGES if s.split("@")[0] not in STAGE_COMMANDS]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {list(STAGE_COMMANDS)}")
    for d in (CACHE_DIR, LOG_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    env = print_environment()
    bundle, records, reused, reqs = {"git_commit": "?", "git_dirty": False}, [], {}, {}
    try:
        data_root = find_data_root()
        print(f"data root: {data_root}", flush=True)
        bundle = install_code()
        reqs = install_requirements(CODE_DIR / PROJECT_NAME / "requirements.txt")
        for prev in (find_previous_runs() if REUSE_PREVIOUS_CACHE else []):
            reused[str(prev)] = reuse_previous(prev)
        senv = stage_env(data_root)
        for name in STAGES:
            rec = run_stage(name, senv)
            records.append(rec)
            if rec["returncode"] != 0:
                print(f"STOPPING: stage {name} failed (rc {rec['returncode']})", flush=True)
                break
    finally:
        collect_results(env, bundle, records, reused, reqs)
        shutil.rmtree(CODE_DIR, ignore_errors=True)
        if OUTPUT_DIR.exists() and not any(OUTPUT_DIR.iterdir()):
            OUTPUT_DIR.rmdir()
    failed = [r["stage"] for r in records if r["returncode"] != 0]
    print(f"\nDONE. stages: {[(r['stage'], r['status'], r['seconds']) for r in records]}"
          f"{'  FAILED: ' + str(failed) if failed else ''}", flush=True)


if __name__ == "__main__":
    main()
