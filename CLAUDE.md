# CLAUDE.md — Amazon ML Challenge 2026 (Business Entity Resolution)

Deadline: **Sunday 27 Sep 2026, 23:59 IST**. Design doc: `docs/Amazon_ML_Challenge_2026_Implementation_Plan.pdf` (follow it).

## Layout
- `student_resource/` — organiser files: `dataset/` (train/test TSVs), `utils/validate_submission.py`,
  `README.md`, `Documentation_template.md`, EDA (`eda.py`, `eda_report.txt`). **Read-only.**
- `code/business_entity_resolution/src/` — all pipeline source code (ships in the final zip).
- `code/business_entity_resolution/README.md`, `requirements.txt` — reproduction instructions + pinned deps.
- `output/` — `matching_results.tsv`, `candidate_pairs.tsv`. `cache/` — per-stage Parquet. `logs/` — runtimes, memory, experiment + submission logs.
- venv: `.venv/` (Python 3.x, see requirements.txt).

## Standing rules (apply to every task)

### Task & metric
- For each Source 1 entity, find all matching Source 2/3 records.
- Metric: **macro F0.5 per S1 entity**, singletons included (empty prediction on a true singleton = 1.0; any prediction on it = 0.0). Precision is weighted 2x over recall.

### Key EDA facts
- Train S1/S2/S3 = 2.21M / 5.03M / 5.29M rows. Test = 1.73M / 4.89M / 5.08M.
- Test adds **France** (259k S1), which has **no training labels**.
- Singleton rate 5.58%. Non-singletons average 3.67 matches (max 11).
- **Zero** S2/S3 records are linked to more than one S1 — strict one-to-one.
- ~26% of S2/S3 records match nothing.
- True matches always have the same country label.

### Country
- Country is an **open set of strings**. Never hard-code, filter, or one-hot `{US, India}`.
- Never use country as a model feature. Only use it to partition work.

### Data & licences
- No external data, APIs, geocoders or downloaded dictionaries. Only the provided files.
- Rule-based libraries with MIT/Apache/BSD/ISC licences are fine; avoid GPL (e.g. use `anyascii`, not `unidecode`).
- Any model must be MIT/Apache 2.0 and at most 8B parameters.

### I/O
- Read all TSVs with `sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE`.
- Write submissions tab-separated, UTF-8; validate with `student_resource/utils/validate_submission.py`.
- **Line endings: `\n` only, never `\r\n`** (we are on Windows). Use `open(path, "w", encoding="utf-8", newline="")`
  with `csv.writer(..., lineterminator="\n")`, or `df.to_csv(..., lineterminator="\n")`. After writing any output
  file, read it back in binary and assert `b"\r"` is absent. The writers in `src/io_utils.py` do this permanently — use them.

### Parallelism (Windows)
- Windows has no `fork`: any multiprocessing code must sit under `if __name__ == "__main__":` and must not rely on fork semantics.
- Prefer libraries' own threading over hand-written multiprocessing: `sparse_dot_topn` `n_threads`,
  `rapidfuzz` `workers=-1`, `lightgbm` `num_threads`.

### Memory & caching
- Process **one country at a time** to keep memory low (machine: 16 GB RAM, 8 cores, RTX 4050 laptop GPU, Windows).
- Cache every stage's output as Parquet in `cache/`; skip recomputation if the cache exists. Every stage CLI has a `--force` flag to recompute.

### Code hygiene
- Every function gets a docstring.
- Fix random seeds (42).
- Log runtimes and peak memory per stage to `logs/`.
- Never modify files in `student_resource/dataset/` or `student_resource/utils/`.

### Logging & git
- After each task, append a short entry to `logs/experiments.md`: date, what changed, key numbers.
- Record each leaderboard submission in `logs/submissions.md`.
- Never `git commit` / `git push` unless explicitly asked; list changed files and propose a commit message instead.
