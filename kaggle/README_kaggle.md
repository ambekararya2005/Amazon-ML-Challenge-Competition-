# Running the heavy stages on Kaggle

Code is written and tested locally; Kaggle (Linux, ~30 GB RAM, 4 CPUs, 12 h/run) only runs it.

| Piece | Where |
|---|---|
| Organiser data (private dataset) | `aryaambekar/amlc2026-data` |
| Code (private dataset) | `aryaambekar/amlc2026-code` ← `kaggle/code_bundle/` (`amlc2026_code.zip` + `dataset-metadata.json`) |
| Kernel (private, CPU, internet on) | `aryaambekar/amlc2026-pipeline` ← `kaggle/kernel/` (`run_pipeline_kaggle.py` + `kernel-metadata.json`) |
| Kernel output | `/kaggle/working/{cache,logs,results}/`, `RUN_INFO.json` |

The pipeline itself is path-agnostic: each root comes from a CLI flag, else an env var, else the local default.

| Root | CLI flag | Env var | Local default | On Kaggle |
|---|---|---|---|---|
| data | `--data-root` | `DATA_ROOT` | `student_resource/dataset` | auto-detected: folder under `/kaggle/input` holding `train_source1.tsv` |
| cache | `--cache-root` | `CACHE_ROOT` | `cache/` | `/kaggle/working/cache` |
| output | `--output-root` | `OUTPUT_ROOT` | `output/` | `/kaggle/working/output` |
| logs | `--log-root` | `LOG_ROOT` | `logs/` | `/kaggle/working/logs` |

The older `BER_*_DIR` env vars still work. Threads: `N_THREADS` (default `os.cpu_count()`). Normalise workers:
`--workers` or `N_WORKERS` (default CPUs − 1). Blocking RAM gate: `--min-free-gb` or `MIN_FREE_GB` (default 7).

## 0. One-time setup (PowerShell, repo root)

```powershell
.\.venv\Scripts\python.exe -m pip install "kaggle>=1.8.0"
```

Put the Kaggle access token in the **user** environment variable `KAGGLE_API_TOKEN` (Windows Settings →
"Edit environment variables for your account"; never paste it into a command or a file in the repo). Restart
VS Code so new shells inherit it, or load it into the current shell only:

```powershell
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')
.\.venv\Scripts\kaggle.exe datasets list --mine          # auth check
```

All commands below assume `$K = ".\.venv\Scripts\kaggle.exe"`.

## 1. Code dataset: build, create, update

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s code\business_entity_resolution -t code\business_entity_resolution   # tests first
.\.venv\Scripts\python.exe kaggle\build_bundle.py                                   # -> kaggle\code_bundle\amlc2026_code.zip

& $K datasets create  -p kaggle\code_bundle                                        # first time only (private by default)
& $K datasets version -p kaggle\code_bundle -m "code <git short hash>: <what changed>"   # every later update
& $K datasets status aryaambekar/amlc2026-code                                      # wait for "ready" before pushing the kernel
```

## 2. Push and run the kernel

Edit the block at the top of `kaggle/kernel/run_pipeline_kaggle.py`:

```python
STAGES = ["load", "normalize", "split", "block_benchmark"]
# available: load, normalize, split, block_benchmark, block_train, recall, block_test, tests
```

then push (a push creates a new version and starts the run immediately):

```powershell
& $K kernels push -p kaggle\kernel
```

### Reusing a previous run's cache

Kernel outputs can be attached as inputs. Add the kernel whose output you want to start from to
`kernel_sources` in `kaggle/kernel/kernel-metadata.json`, e.g.

```json
"kernel_sources": ["aryaambekar/amlc2026-pipeline"]
```

The script finds any attached folder containing `RUN_INFO.json` + `cache/` (the newest one if several) and copies its
`cache/` and `logs/` into `/kaggle/working` before running. Stages then skip everything already cached, so e.g.
`STAGES = ["block_train", "recall"]` starts straight from the normalised Parquet. Set `REUSE_PREVIOUS_CACHE = False`
to ignore attached runs. (If Kaggle refuses a kernel attaching its own output, push the follow-up under a second
slug, e.g. change `id`/`title` to `aryaambekar/amlc2026-pipeline-2`, and keep `kernel_sources` pointing at the first.)

## 3. Status

```powershell
& $K kernels status aryaambekar/amlc2026-pipeline       # queued / running / complete / error
```

## 4. Download outputs

Small results only (result files, run marker and the kernel log):

```powershell
$run = "kaggle\runs\$(Get-Date -Format yyyyMMdd-HHmm)"; New-Item -ItemType Directory -Force $run | Out-Null
& $K kernels output aryaambekar/amlc2026-pipeline -p $run --file-pattern "^(results/|RUN_INFO\.json)"
```

Everything, including the multi-GB cache (only when you need it locally):

```powershell
& $K kernels output aryaambekar/amlc2026-pipeline -p $run
```

`kaggle/runs/` is git-ignored. After a run, append `results/experiments_entry.md` to `logs/experiments.md`.

## What the kernel script does

1. Prints RAM, CPU (count and model), free disk and Python/platform versions.
2. Finds the newest code bundle under `/kaggle/input`, copies it to `/kaggle/working/code`, compares every
   pinned requirement against the Kaggle image and `pip install`s only those that are missing or at a different version.
3. Optionally reuses an attached previous run (see above).
4. Runs each stage in `STAGES` as `python -m src.<module> ...` with `DATA_ROOT`, `CACHE_ROOT`, `LOG_ROOT`,
   `OUTPUT_ROOT`, `N_THREADS`, `MIN_FREE_GB` set; stops at the first failed stage. Per stage it records wall time,
   peak RSS of the stage's process tree and peak system RAM used.
5. Always (even after a failure) copies the small files (`blocking_bench.json`, `blocking_recall.json`,
   `blocking_misses.txt`, `stage_metrics.jsonl`, summaries, stage logs) to `/kaggle/working/results/`, writes
   `results/kernel_stages.json` and `results/experiments_entry.md`, writes `RUN_INFO.json`, and deletes the code copy.
