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

## 2026-09-26 — Geo-dense benchmark + decoy-aware rule scorer v2
- City-level units split 21.8% of true pairs (S2/S3 drop / change / misspell the city) -> coarser unit = (country, region):
  S1 regions found data-driven (src/geo_units.py), S2/S3 region learned from true-pair co-occurrence (src/benchmark.py);
  records with no region (4.8%, mostly empty addresses) are added to the benchmark as extra distractors.
  Sample: 4 India + 13 US regions = 535,608 S1 (21.5% / 26.1%), 2,883,225 queries; 5 folds by region (fold 0 = 126,414 S1).
  True pairs split: 0.18%. Queries per S1: India 5.20 / US 5.48 (test 5.75). Unmatched queries in bench 33.7% / 37.0%
  (26.0% full train; higher because no-region records whose S1 lies outside the sample count as unmatched).
- Blocking + stage-1 top-5 on the benchmark (Kaggle K8): recall@5 India 0.9438 / US 0.9841 / all 0.9698.
- Baseline (submission #1 config) on fold 0: F0.5 0.7259 (India 0.7191 / US 0.7313), precision 0.705, recall 0.925,
  4.28 predictions/S1, 0.34% empty, singleton part 0.033. LB was 0.636 -> the benchmark now sees the decoy problem.
- Folds 1-4, true pairs vs decoys accepted by the baseline: num_conflict 9.3% vs 83.0%; num_compatible 79.7% vs 15.3%;
  extra name tokens 39.7% vs 73.9%; conflict or extra 45.2% vs 97.2%. Decoy extra words: group, holdings, partners,
  center, industries, services, enterprises, exports, ventures, overseas, infratech, downtown, west, south, metro ...
  ('holdings' is not a legal form here, so it already counts as an extra word.) True-pair extras: center, services, dba,
  sri / smt / shri, formerly, ...
- Rule scorer v2 (src/scorer_v2.py, features src/decoy_features.py): 0.3 name_tsort + 0.5 addr_tsort + 0.2 num_compatible
  - 0.1 extra-token penalty, num_conflict veto (beat penalties 0.1-0.5 in the grid), one-to-one, t_accept = t_keep = 0.64.
  Fold 0: F0.5 0.8230 (+0.0970 vs baseline); US 0.9100 (+0.179), India 0.7113 (-0.008); precision 0.880, recall 0.821,
  3.25 predictions/S1, 6.96% empty, singleton part 0.679. Folds 1-4: 0.8451.
- Pair-feature tables: benchmark (5 folds, labelled) = K8 output cache/bench/pairs.parquet; test = K7a output
  output/features/test_pairs.parquet (49,846,318 pairs x 38 columns). Both on Kaggle (not downloaded).
- Submission #2 files (Kaggle K7b, 6.7 min): validator PASS (Kaggle --check-ids, local). Test per country: France 3.48 pred/S1,
  5.14% empty, 62.9% of S2/S3 assigned; India 3.12 / 9.47% / 53.5%; US 3.23 / 6.64% / 56.0% (target ~74% assigned: v2 now
  under-assigns, consistent with benchmark recall 0.82). France sits between US and India on every statistic.

## 2026-09-26 — Step 0 checks, v3 features, blocking v4 (overnight)
- LB calibration: submission #2 scored 0.829 vs fold-0 0.823 -> fold 0 of the geo-dense benchmark is now the gate.
- Step 0 (src/data_checks.py, logs/data_checks.json): (a) no per-source cap - S2 matches per S1 0-5, S3 0-6, mode 1,
  identical US / India distributions. (b) v2 accepted 1.72M pairs: 1.51M true, 92.8k decoys (query matches nobody),
  114.9k wrong-entity (query belongs to another S1). Decoys are mutated copies of the entity, not of one sibling:
  median name/addr sim decoy<->closest sibling 0.78/0.88 vs decoy<->S1 0.79/0.87; closer to the sibling only 48%;
  main number equals the S1's 65.9% vs the sibling's 56.3%; 0.14% identical copies. (c) same source as the closest
  sibling 70% (S2, 49% expected) / 61% (S3, 52% expected) - mild.
- v3 features (src/features_v3.py, src/decoy_features.py): tagged address numbers (street / floor+ordinal / postal -
  the learned postal rule finds no postal-like length in any country, so postal features are 0), street-number
  compatible / conflicting counts + Jaccard, floor conflict, extra-word lists (target-encoded at model time), group
  ranks within the S1's claimants, near-twin features vs the S1's top-8 claimants. Bench 14.4M pairs (local 6.6 min),
  test 49.8M pairs (Kaggle K9, 18 min, peak tree RSS 30.4 GB).
- Blocking v4 (src/blocking_v4.py): cross-script dictionary mined from train true pairs (bench eval: 3,047 tokens from
  877k cross-script pairs, excluding benchmark S1; test: 3,739 tokens) e.g. mharastr->maharashtra, dilli->delhi,
  eksports->exports, tredimg->trading; pass C on mapped texts; pass D (rarest name token, S1 df <= 50, top-3 by
  token_sort); adaptive top-k (8 when cheap(1st) - cheap(5th) < 0.1). Benchmark recall (bench true pairs):
  v1 top-5 India 0.9468 / US 0.9851 / all 0.9715 (5.00 cands/query); v4 union India 0.9763 (v1 union 0.9625);
  v4 top-5 without pass D India 0.9631; v4 adaptive m=0.1 India **0.9691** / US 0.9870 / all 0.9806 (5.87 cands/query,
  +17% pairs). Test: France smoke run locally 5.91 cands/query, 6.35% pairs from pass D only.
  Overnight Kaggle: blocking-v4-{france,india,us} (test candidates -> output/cand_v4/test) and bench-v4 (v4 bench
  pairs + features + model, FEATURE_VARIANT=v4; v1 tables untouched).

## 2026-09-27 — LightGBM pair model v3 / v4 (Kaggle K10, K12d), fold 0 = gate
- Two-stage LightGBM (leave-one-fold-out over bench folds 1-4, num_leaves 127, lr 0.05, ff 0.8, early stop 50),
  extra-word target encoding (OOF), isotonic (own PAV; sklearn's DLL is blocked locally), S1 has-match model, exact
  expected-F0.5 decoder. K10 model_train 170 min, K12d 216 min (4 CPUs).
- Fold 0 F0.5 all / India / US (precision, recall, pred/S1, % empty, singleton F0.5 on "all"):
  baseline 0.7259 / 0.7191 / 0.7313; v2 0.8230 / 0.7113 / 0.9100 (0.880, 0.821, 3.25, 6.96, 0.679);
  v3 stage1+thr 0.9617; stage2+thr 0.9638; stage1+decoder 0.9618; stage2+decoder 0.9649;
  **v3 stage2+decoder+hasmatch 0.9652 / 0.9486 / 0.9781** (0.990, 0.930, 3.22, 6.15, 0.957);
  **v4 (blocking v4 candidates) stage2+decoder+hasmatch 0.9719 / 0.9638 / 0.9782** (0.990, 0.946, 3.29, 5.89, 0.957).
  Chosen on OOF folds 1-4 (v3 0.96738, v4 0.96895): stage2+decoder+hasmatch, T = 1.0, miss = 0.0.
- Has-match: singleton F0.5 0.943 -> 0.957 (v3 and v4). Decoder vs best threshold: +0.0011 (v3) / +0.0013 (v4).
- Error budget v4 fold 0 (points): singleton non-empty 0.0024, FP decoy 0.0037, FP other 0.0028, FN blocking 0.0073,
  FN scoring 0.0119 (total 0.0281); v3: blocking 0.0145 (India 0.0260 -> v4 0.0103). v2 total was 0.1770.
- Importance: stage 1 s1_rank 48%, b_num_match 20% (v4) / cheap 13% (v3), then q_gap_to_best, tw_v2_diff, te_q_max,
  v2_score, st_jaccard; stage 2 p1 + p1_q_margin ~92%.
- Remaining errors (logs/model_v4_errors.txt): mostly name-only queries (empty address); decoys differing only by a
  legal form ("... Public Limited"; legal forms are stripped from name_core -> add a legal-form mismatch feature);
  sibling sub-numbers (12-1-331/C/8 vs /C/1).
- Test v4 candidates: France 8.48M / India 27.78M / US 21.35M pairs (5.59-5.91 per query, 5.9-7.7% from pass D only).

## 2026-09-27 (afternoon) — final-day plan: state, error budget, v4 final run
- Submission #3 (v3, v1 candidates): public LB **0.941** vs fold-0 0.9652 (-0.024). Files kept in output/best/sub3_v3_fold0.965/.
- State in #3: stage-2 group aggregates YES (p1 max / second / sum / cnt>0.5 / rank / p1-max within S1; other-S1 max,
  margin, rank within the query); S1 has-match model YES; expected-F0.5 decoder with EMPTY YES; blocking v4 (pass D,
  cross-script dictionary, adaptive top-k) NO -> exists as v4 (fold 0 0.9719, +0.0067), test features READY (K13).
- Error budget fold 0 (points lost; all / US / India): #3 singleton 0.0024/0.0016/0.0034, decoy FP 0.0036/0.0020/0.0055,
  other FP 0.0025/0.0025/0.0024, FN blocking 0.0145/0.0055/0.0260, FN scoring 0.0119/0.0102/0.0141 (total 0.0348);
  v4: 0.0024, 0.0037, 0.0028, 0.0073, 0.0119 (total 0.0281).
- Step 1 (stage 2) and Step 3 (has-match + decoder) already exist. Step 3 re-tune on v4 OOF (T x miss x h-temperature,
  27 configs, logs/decoder_retune_v4.json): OOF surface flat (0.96881-0.96897); best (T 1.0, miss 0.1, hT 0.7) fold 0
  0.9718 vs 0.9719 -> DROP. Singleton part before/after has-match (v4 fold 0): 0.943 -> 0.957.
- Step 2 (cross-source sibling agreement): the text part already exists as the twin features (top-8 claimants: max
  name/addr sim, number / extra-word better/worse, v2 diff, same source); the p1-of-sibling part would need new
  bench + test features and a full test rescoring before 19:00 -> DROPPED (time).
- Step 4: v4 kept (+0.0067 on fold 0).
- Step 5 (lr 0.03, num_leaves 255, 3 seeds): each needs a stage-1 retrain (CV 162 min on Kaggle at lr 0.05) -> DROPPED (time).
- Final run: src/model_lgb.py gained --stage train_full (MODEL_FULL=1: one model per stage on all 5 bench folds,
  rounds = 1.1 x mean CV best iteration; stage-2 / has-match inputs = CV OOF p1 / p2; calibration + decoder from CV),
  per-country scoring (SUBMIT_COUNTRY, SUBMIT_PARTS=1 -> submit_parts_<model>/<country>.parquet; avoids the end-of-run
  OOM that killed K11) and --stage assemble (parts -> TSVs, matches within candidates, validator). Assemble round-trip
  of #3 reproduces its files byte for byte (md5). Kaggle 14:29: K14a-c amlc2026-submit-v4-{france,india,us} (CV
  models, safe v4) and K15 amlc2026-full-v4 (full retrain + test parts).

## 2026-09-27 (evening) — FN split, sibling rescue, test-gap diagnostic, v4-safe files
Saved OOF / fold-0 / test probabilities only (no retraining, no re-blocking). Scripts: tools/pp_common.py,
tools/fn_buckets_v4.py, tools/sibling_rescue_v4.py, tools/test_gap_v4.py; results logs/fn_buckets_v4.json,
logs/sibling_rescue_v4.json, logs/test_gap_v4.json, logs/test_gap_france_examples.txt.
- FN "scoring/decoder" 0.0119 on fold 0 (v4; reproduced F0.5 0.9719), split by missed true pairs present in the
  candidates: (a) query argmax = true S1, decoder did not select it: **0.0096** (13,717 pairs; India 0.0114 / US 0.0081;
  p2 median 0.53, 71% of them in 0.3-0.8, none >= 0.8); (b) argmax = another S1: 0.0023 (b1 that S1 selected the query:
  0.0001, 59 pairs; b2 nobody selected it: 0.0023, 3,083 pairs, p2 median 0.06); (c) other: 0.
  -> second-choice rescue not tried (bucket b < 0.003).
- Sibling rescue (for bucket a): add an unselected argmax claimant if p2 >= a, best compatible-number similarity to a
  selected match of the S1 >= b (min or mean of name/address token_sort), max extra-word TE score < d. Grid a 0.2-0.6,
  b 85/90/95, d 0.5/0.7/off, 2 sim modes (90 configs) on OOF folds 1-4: every config <= baseline (best 0.96891 vs
  0.96895; a 0.6, b 95, min, d 0.7). Fold 0: 0.97184 vs 0.97189 (-0.00005; US -0.00002, India -0.00009), singleton
  part unchanged 0.957; adds 256 pairs = 162 TP + 94 FP. Decoys are near-copies of the true siblings, so text
  similarity to a selected sibling does not separate them. **DROPPED** -> no post-processed submission.
- Decoder re-tune (T x miss x h-temperature) earlier: flat, dropped.
- Test gap (report only). Share of queries whose best p is in 0.3-0.7: fold 0 US 2.5% / India 3.3%;
  test US 3.9% / India 3.3% / France 5.0%. S1 best p (France test vs fold 0 US / India): median 1.0 everywhere,
  share < 0.3: 4.9% vs 5.6 / 5.2, share 0.3-0.7: 0.8% vs 0.6 / 1.0. Pred/S1, % empty, mean h: fold 0 US 3.30 / 5.93 /
  0.944, India 3.27 / 5.84 / 0.948; test France 3.40 / 5.39 / 0.950, India 3.33 / 5.95 / 0.946, US 3.49 / 5.66 / 0.946.
  France is the most uncertain at query level (2x fold-0 US) and test US is also above fold-0 US, so part of the
  0.024 LB gap is likely test-wide (harder or denser decoys than the benchmark regions), with France adding to it.
  France examples (15 random S1): French legal forms (SARL, SASU, EI, SCI, S.A.R.L.) handled; decoys with a different
  house number or an extra word (International, Distribution, Developpement, Ecole) rejected; one likely FP ("Maison
  de Agriculteurs & Fils" selected at p 1.0); a few 0.4-0.6 rejects that share the full address but have another name.
- v4-safe (CV fold 1-4 models) test files: output/final_v4safe/ (validator PASS). K14a-c runtimes France 36 min,
  India 115 min, US 80 min. K16 (France, SAVE_PAIRS=1) reproduces the France part byte for byte.
- #4 (v4-safe) public LB **0.945** (fold 0 0.9719, gap -0.027). Copied to output/final_chosen/ as the default.
- Full retrain K15 (amlc2026-full-v4): train_full 87 min (stage 1 1323 rounds 63 min, stage 2 294 rounds 15 min,
  peak 18.7 GB) + submit 90 min (one model per stage). In-sample fold 0 0.978 (India 0.974 / US 0.981; bug check).
  Cross-check vs v4-safe (tools/compare_parts.py, logs/compare_full_vs_safe_v4.json): agreement both/union France 97.75,
  India 98.27, US 98.53%; pred/S1 +0.74 / +0.21 / +0.40% rel, % empty -0.56 / 0.00 / +0.53%, % assigned +0.75 / +0.21 /
  +0.38%; argmax-p2 decile shares max |diff| 0.34 / 0.43 / 0.20 pp -> SANE. Files output/final_full/ (validator PASS).
- Documentation_template.md filled (submission table, v4 error budget, FN bucket split, test-gap table, next steps);
  README updated (per-country submit parts, assemble, train_full, kernel plan).
