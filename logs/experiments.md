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
