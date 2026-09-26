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

## 2026-09-26 — Kaggle runner setup (path-agnostic pipeline)
- `src/config.py`: every root resolved as CLI flag (`--data-root/--cache-root/--output-root/--log-root`, accepted by every stage)
  > env (`DATA_ROOT/CACHE_ROOT/OUTPUT_ROOT/LOG_ROOT`, legacy `BER_*`) > local default (unchanged). Data folder auto-detected
  under `/kaggle/input` (search for `train_source1.tsv`; nested or flat layout). `N_THREADS` = `os.cpu_count()`-based (affinity-aware).
- `normalize.py` workers default = CPUs − 1 (env `N_WORKERS`); `blocking.py` uses `config.N_THREADS`; RAM gate configurable
  (`--min-free-gb` / env `MIN_FREE_GB`, default 7). New `tests/test_config.py`; 43 tests pass.
- `kaggle/`: `build_bundle.py` (zip 43 KB, 16 files + BUNDLE_INFO.json), `code_bundle/dataset-metadata.json`,
  `kernel/run_pipeline_kaggle.py` + `kernel-metadata.json`, `README_kaggle.md`. Kernel script smoke-tested locally on a fake
  /kaggle tree (bundle install, pin check, stage run, results collection, previous-cache reuse).
- Kaggle run not started yet: `KAGGLE_API_TOKEN` not found in the User/Machine environment.

## 2026-09-26 — Kaggle run v1: load, normalize, split, block_benchmark (kernel aryaambekar/amlc2026-pipeline v1)
- Machine: 4 vCPU (2 physical, Xeon 2.20 GHz), 31.35 GB RAM, Python 3.12.13. Code commit d75b6e83 +dirty. Pinned
  packages installed over the image (pandas 3.0.6, numpy 2.4.6, sklearn 1.9.1, ...); pip conflicts only in unused preinstalled packages.
- Reproducibility: row counts, train_pairs 7,638,365, split (train 1,875,797 / val 331,024, same singleton rates) and
  every per-country normalisation statistic identical to the local run.
- Stages: load 4.4 min (peak RSS 3.34 GB), normalize 14.5 min (3 workers, 2.48 GB), split 0.3 min (1.14 GB),
  block_benchmark 34.5 min (3.58 GB). Whole kernel ~55 min; peak system RAM used 4.4 GB of 31 GB.
- Benchmark (100k train S2 queries, 4 threads): pass A index 10 s, 2.1-3.5 s / 100k queries, 6.5-8.4 cands/query;
  pass C fit 100-108 s (vocab ~190-200k, nnz ~123M), **1661 s (India) / 1922 s (US) per 100k queries** = 99.8% of query time.
- Projection: train recall run 442 min (India 163, US 280), peak 4.9 GB; test 2996 min (~50 h: France 461, India 1310,
  US 1225), peak 3.1 GB. Test exceeds the 90-min gate and the 12 h Kaggle limit -> pass C must be sped up before blocking.

## 2026-09-26 — Pass-C speed/recall tuning (local, validation sample)
- `src/tune_pass_c.py` + `src/tune_pass_c_report.py`. Per country: index = all train S1; queries = 20k S2/S3 matched to val
  entities + 20k random others (seed 42); 4 threads. Local->Kaggle factor from the India base (query x0.575, fit x1.67).
  US base scored on an 8k subset. Some runs overlapped (India full-text vs US base) -> timings +-30% noisy; recalls exact.
- Pass A alone: union-free recall US 0.790 / India 0.702; gate (L3) sends 59% of queries to pass C.
- Base (current pass C): union recall US 0.9890 / India 0.9663; projected test 4,066 min (Kaggle v1 own projection 2,996).
- L1 (name-only): union US 0.940-0.951 / India 0.856-0.873 -> ruled out (addresses matter for pass C).
- L2 full text, ng44: max_df 0.5% -> US 0.9819 / IN 0.9303, 46 min; 1% -> 0.9862 / 0.9386, 118 min;
  2% -> 0.9877 / 0.9481, 310 min; 3% -> 0.9881 / 0.9516, 310 min sum (max country 145 min). ng44 >= ng34 at equal cost.
- L3 gating: -0.05 pp US, -0.7 to -0.8 pp India, ~40% less pass-C time.
- Rule (mean union within 1 pp of best 0.9777 AND <= 180 min sum): no configuration qualifies. L4 not run (time box).
  Candidate: L2 full ng44 max_df=3%, run as 3 parallel country kernels (max 145 min wall clock). Awaiting decision.
- Also: `blocking.py --country` + `--stage combine`; kernel runner `stage@Country` + REUSE_CACHE_SUBDIRS;
  `kaggle/make_country_kernels.py`, `kaggle/download_results.ps1` (sets PYTHONUTF8=1); README_kaggle.md updated.

## 2026-09-26 — Blocking v1 on Kaggle (5 parallel kernels) + rule baseline (submission #1)
- pass C set to char_wb (4,4), max_df 3% (sklearn absolute cap floored at min_df). Test blocking as 4 kernels:
  France 5.0 min, India shard 1/2 113 min, shard 2/2 142 min, US 106 min (peak RSS <= 2.3 GB each). All 9,969,589 test
  queries processed exactly once (105 union parts, 0 duplicates); 286 queries (0.003%) have no candidate; 152,961,279 pairs.
- Train/validation candidates (K5, 113 min): 1,145,795 val-matched + 692,421 train-matched (200k train-fold S1, stratified)
  + 300,000 distractors. Validation union recall: US 0.9879 / India 0.9543 / all 0.9744 (A 0.7549, C 0.9700, C@5 0.9627);
  15.9 cands/query. 30 missed pairs per country in logs/blocking_misses.txt (non-Latin names, corrupted addresses).
- Stage 1: top-5 per query by cosine_C + 0.2 x shared_keys_A -> validation recall@5 0.9656 (US 0.9820 / India 0.9409).
- Rule baseline (src/finalize.py): name_sim = max(token_set_ratio(core), ratio(compact)); addr_sim = token_set_ratio(addr);
  num_match = shared/min keys. Best: weights (0.3, 0.6, 0.1), threshold 0.64, one-to-one.
  Validation macro F0.5 0.9641 (US 0.9716 / India 0.9529); without one-to-one at that config 0.9119; best no-o2o 0.9488.
  Mean precision 0.9862, mean recall 0.9329, singleton accuracy 0.9341.
- CAVEAT: the validation query set holds ~1M non-val records vs all 10M on test, so FPs are under-represented. On test the
  scorer predicts 8,995,975 pairs (5.2 per S1) vs ~7.38M true pairs expected (74% of 9.97M records), and only 0.04-0.52%
  empty predictions vs a 5.6% singleton rate -> expect test F0.5 well below 0.964.
- Test: France 0.52% empty / 4.98 predicted / 27.7 candidates per S1; India 0.31% / 5.18 / 29.1; US 0.04% / 5.29 / 28.8.
- Files: output/matching_results.tsv (131 MB), output/candidate_pairs.tsv (633 MB); validator PASS (Kaggle with --check-ids, local).

## 2026-09-26 — Audit of submission #1 (public LB 0.636 vs local 0.964)
- Independent audit (entity_id strings + raw data only; script in the session scratchpad, results in logs/audit_sub1*.{json,txt}).
- No mapping bug: 0% cross-country pairs; 16,000 sampled candidate rows (France, India shard 1, India shard 2, US; s2 + s3)
  round-trip query_id / s1_id -> entity_id -> raw text 100%; all 4 kernels used identical TEST lookup/query tables
  (md5 equal, 0 train ids); no duplicate or unknown ids in the submission.
- Predicted pairs look like true pairs on name/address (median name 91-100, addr 90-95, 0% clearly wrong), but share a
  number less often (61-75% vs 82-84% for true train pairs).
- Real cause: hard-negative siblings in test (same name + extra word such as Holding/International/Exports, or same
  street with a different house number). The scorer accepts them: token_set_ratio ignores extra words, num weight 0.1.
  Coverage: 90.2% of test S2/S3 assigned (France 90.0 / India 89.0 / US 91.9) vs ~74% expected; 5.2 predictions per S1
  vs 3.46 in train GT; only 0.04-0.52% empty predictions vs 5.6% singletons.
- Why validation missed it: the validation query set held only 300k random distractors of ~2.6M unmatched train records,
  so most siblings were absent.
