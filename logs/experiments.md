# Experiment log

## 2026-09-25 — Project setup
- Created layout: `code/business_entity_resolution/{src,README.md,requirements.txt}`, `output/`, `cache/`, `logs/`.
- Added root `CLAUDE.md` with standing rules; `logs/submissions.md` template; `.gitignore` excludes dataset, cache, output, venv.
- Venv `.venv/` on Python 3.11.9: pandas 3.0.6, pyarrow 25.0.1, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1,
  rapidfuzz 3.14.6, sparse_dot_topn 1.2.0, lightgbm 4.7.0, anyascii 0.3.3, tqdm 4.70.1, psutil 7.2.2. All imports + smoke tests pass.
- No modelling code yet. Key numbers: n/a.

## 2026-09-25 — I/O layer + Parquet cache (stage 0)
- Added `src/{config,io_utils,logging_utils,prepare_data}.py`. CLAUDE.md: `\n`-only output rule and Windows parallelism rule.
- Writers enforce submission rules and assert no `b"\r"`; tested with the official validator (`--check-ids`, 1.73M rows): PASS.
- Machine: 15.25 GB RAM total, only 2.06 GB available at run start (86.5% used); 8 physical / 16 logical cores.
- All 7 tables: 0 NA values; Parquet round-trip identical (`assert_frame_equal`), empty-string counts preserved. pandas 3 dtype = `str`.
- In-memory (deep) MB: train S1/S2/S3 = 259/601/622, test S1/S2/S3 = 213/616/618, GT = 151.
  Parquet MB: 90/219/227, 72/220/223, 51. Empty addresses: train S2 168,967, S3 175,916; test S2 129,408, S3 136,098.
- Full conversion 135 s, peak RSS 3.06 GB.

## 2026-09-25 — Train pairs, validation split, official metric
- Load stage now also writes `cache/raw/train_pairs.parquet` (s1_id, other_id, other_source): 7,638,365 rows
  (S2 3,693,619 / S3 3,944,746), 0 duplicate other_id, 0 ids missing from S2/S3 files; 2,083,574 S1 with matches. 67 s, peak 2.65 GB.
- `src/metric.py` (f05_entity, macro_f05 exactly as plan §8; f05_breakdown per country) + 10 unit tests, all pass (README example = 0.714).
- `src/split.py`: 15% val, stratified country x bucket(0..6+), seed 42 -> `cache/split.parquet`.
  train 1,875,797 (India 750,710 / US 1,125,087); val 331,024 (India 132,478 / US 198,546).
  Singleton rate 0.0558 both folds; mean matches 3.4612 train vs 3.4614 val.
- Reference points on val: perfect prediction F0.5 = 1.0000; all-empty F0.5 = 0.0558 (= singleton rate).
