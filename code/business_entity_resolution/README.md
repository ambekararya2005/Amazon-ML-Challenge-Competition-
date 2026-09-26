# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline that, for every Source 1 entity, finds all matching Source 2 / Source 3 records and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

> Status: project scaffold only. Pipeline stages and exact run commands are added as they are built.

## Environment

- Python 3.11.9
- Hardware used: Windows 11, 16 GB RAM, 8 CPU cores, NVIDIA RTX 4050 Laptop GPU

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r code/business_entity_resolution/requirements.txt
```

## Data

Place the organiser data under `student_resource/dataset/{train,test}/` (unchanged). No external data,
APIs or downloaded dictionaries are used.

## Reproduce (end to end)

Run all commands from `code/business_entity_resolution/`. Every stage caches its output in `cache/` and
skips work if the cache exists; add `--force` to recompute.

| # | Command | Output | Runtime / peak RSS (reference machine) |
|---|---------|--------|----------------------------------------|
| 0 | `python -m src.prepare_data` | `cache/raw/*.parquet` (incl. `train_pairs.parquet`), `logs/data_summary.json` | 135 s + 67 s (pairs) / 3.1 GB |
| 1 | `python -m src.split` | `cache/split.parquet` (s1_id, fold: 85% train / 15% val) | 14 s / 1.1 GB |
| 2 | `python -m src.normalize --stage normalize` | `cache/norm/{train,test}_s{1,2,3}.parquet` | 225–370 s (6 workers) / 2.2 GB |
| 3a | `python -m src.blocking --stage block --split test` (or per country: `--country France`, `--country India --shard 1/2`, ...; then `--stage combine --split test`) | `cache/cand/test/<country>/*_part_*.parquet`, `cache/cand/test_{passA,passC,union}.parquet` | Kaggle 4 CPU: France 5 min, India 2 x 113-142 min, US 106 min / 2.3 GB |
| 3b | `python -m src.blocking --stage block --split train` then `--stage recall` | `cache/cand/train_*.parquet`, `logs/blocking_recall.json`, `logs/blocking_misses.txt` | Kaggle 113 min / 2.3 GB |
| 4 | `python -m src.finalize` | `output/candidate_pairs.tsv`, `output/matching_results.tsv`, `logs/baseline_report.{md,json}`, `logs/stage1_reduction.json`, `logs/combine_check.json` | Kaggle 38 min / 20 GB |

Optional reports: `python -m src.normalize --stage examples` (→ `logs/normalize_examples.md`),
`python -m src.normalize --stage nonlatin_tokens`.

Tests: `python -m unittest -v`

Heavy stages (3a, 3b, 4) run on Kaggle: see `kaggle/README_kaggle.md` (`python kaggle/make_kernels.py blocking|finalize`).

Directory overrides (every stage): CLI `--data-root/--cache-root/--output-root/--log-root`, else env
`DATA_ROOT/CACHE_ROOT/OUTPUT_ROOT/LOG_ROOT` (legacy `BER_*_DIR` still accepted), else the local defaults above.
On Kaggle the data folder is auto-detected under `/kaggle/input`. Threads: `N_THREADS` (default `os.cpu_count()`);
normalise workers: `--workers` / `N_WORKERS` (default CPUs − 1); blocking RAM gate: `--min-free-gb` / `MIN_FREE_GB` (7).
Heavy stages run on Kaggle: see `kaggle/README_kaggle.md`.

## Layout

```
src/
  config.py         paths, seeds, shared constants
  io_utils.py       TSV/Parquet I/O; writers for matching_results.tsv / candidate_pairs.tsv
                    (enforce submission rules, '\n' line endings, assert no b'\r')
  logging_utils.py  loggers, machine info, per-stage runtime + peak memory (logs/stage_metrics.jsonl)
  prepare_data.py   stage 0: TSV -> Parquet with NA / round-trip checks; train_pairs (one row per positive pair)
  split.py          stage 1: 15% validation hold-out, stratified by country x match-count bucket
  metric.py         official macro F0.5 + per-country breakdown
  text_norm.py      pure-text normalisation rules + rule tables (legal forms, abbreviations, transliterations)
  normalize.py      stage 2: per-country normalisation -> cache/norm/*.parquet; examples / non-Latin token reports
tests/              unit tests (stdlib unittest)
README.md           this file
requirements.txt    pinned dependencies
```

Stage outputs are cached as Parquet in `cache/` (pass `--force` to recompute); runtimes and peak memory
are logged to `logs/`.
