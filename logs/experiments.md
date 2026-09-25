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

## 2026-09-26 — Normalisation v1
- `src/text_norm.py` (rules + config tables) and `src/normalize.py` (stage runner, per country, 50k-row batches, spawn pool of 6).
  Output `cache/norm/{train,test}_s{1,2,3}.parquet`, originals kept. 26 unit tests pass.
- Speed: single process 28k rows/s; 6 workers 65k–108k rows/s depending on machine load (24.2M rows in 225–370 s), peak RSS 2.2 GB.
- Transliterated legal words mined from the top non-Latin name tokens: praivet/praibhet/piraivet/praivrr -> private,
  limitet/limirrd/limtid -> limited, elelpi -> llp, 'pra li' -> private limited, 8 enterprises variants. No company/trust variants in data.
  After: 91.2% of non-Latin train_s2 names carry a legal form.
- Deviation from spec, data-driven: '#<digits>' kept as a real number (in S1 address for 96% of US / 72% of India true pairs);
  only PMB/PO Box/box are filler (0% in S1). Empty `numbers` fell 11.3->7.7% (train US), 12.7->9.2% (train India), 8.5->5.8% (test France).
- Final per (split|country) %: empty name_core 0.00 everywhere; empty numbers France 5.76, test India 7.73, test US 6.47,
  train India 9.23, train US 7.70; name_nonlatin India 15.0-15.7, US/France 0.
