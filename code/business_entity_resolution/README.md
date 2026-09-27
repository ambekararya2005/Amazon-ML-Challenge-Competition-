# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 (S1) entity, find all matching Source 2 / Source 3 records. The pipeline writes
`matching_results.tsv` (final matches) and `candidate_pairs.tsv` (the exact candidate set the model scores; every match
is one of these candidates). Everything needed to regenerate both files from the organiser data is in this folder.

**Submitted output = "v4-safe"**: blocking v4 → v3 pair features → LightGBM stage 1 + stage 2 + S1 has-match model,
each the **average of the 4 leave-one-fold-out CV models** (benchmark folds 1–4) → one-to-one → exact expected-F0.5
decoder. Fold 0 of the geo-dense benchmark: **0.9719** (India 0.9638 / US 0.9782); **public LB 0.945**.
Optional variant: the same models refitted once on all 5 folds (`train_full`, below) scored LB 0.944 and was not chosen.

## Pipeline overview

```
load (TSV -> Parquet) -> normalize (per country) -> blocking passes A / C / D -> adaptive top-k shortlist
   -> pair features (similarities, number compatibility, extra words, group / sibling features)
   -> LightGBM stage 1 (p1) -> LightGBM stage 2 (p1 + group aggregates, isotonic) -> p2
   -> S1 has-match model (h) -> one-to-one (each S2/S3 record keeps its best S1)
   -> exact expected-F0.5 decoder per S1 -> matching_results.tsv
```

| Step | Module | What it does |
|---|---|---|
| Load | `prepare_data.py` | Reads the organiser TSVs (`sep="\t"`, `dtype=str`, `keep_default_na=False`, `QUOTE_NONE`) into Parquet; builds `train_pairs` (one row per true pair). |
| Normalize | `text_norm.py`, `normalize.py` | anyascii transliteration, lower-casing, legal-form / abbreviation canonicalisation (`pvt`→`private`, `st`→`street`, transliterated legal words mined from the data), `name_core` (legal forms removed), `name_compact`, `addr_clean`, ordered `numbers`. Rules only, no country branches. |
| Benchmark | `geo_units.py`, `benchmark.py` | Geo-dense validation set: whole (country, region) units of train (region keys found from the data), 5 folds by region; fold 0 is the gate. |
| Blocking pass A | `blocking.py` | Exact keys: each address number × each of the 2 rarest address tokens. Blocking runs in reverse: each S2/S3 record queries the S1 index of its own country. |
| Blocking pass C | `blocking.py`, `blocking_v4.py` | char_wb 4-gram TF-IDF (max_df 3%) on name + address, sparse top-k cosine (`sparse_dot_topn`). v4 first maps tokens through a **cross-script dictionary mined from train true pairs** (e.g. `mharastr`→`maharashtra`, `eksports`→`exports`). |
| Blocking pass D | `blocking_v4.py` | Rarest name token (S1 document frequency ≤ 50) → top-3 S1 by token_sort ratio; rescues typo'd / transliterated names. |
| Top-k | `blocking_v4.py` | cheap = cosine_C + 0.2 × shared_keys_A; keep the top-5, or the top-8 when the gap between the 1st and 5th candidates is < 0.1; pass-D candidates are always kept. This shortlist is `candidate_pairs.tsv`. |
| Pair features | `pair_table.py`, `decoy_features.py`, `features_v3.py`, `scorer_v2.py` | Name / address similarities; tagged address numbers (street, floor) with compatible / conflicting counts; extra name words; group ranks within each S1's claimants and each query's S1s; near-twin (sibling / odd-one-out) features; the v2 rule score. |
| Model | `model_lgb.py` | Extra-word target encoding (out-of-fold); LightGBM stage 1 → p1; stage 2 on p1 + group aggregates → p2 (isotonic); S1-level has-match model → h. |
| Decode | `decoder.py`, `model_lgb.py` | One-to-one, then per S1 the prefix of candidates sorted by p2 (including the empty set) that maximises the exact expected F0.5, with P(singleton) = 1 − h. |
| Write / check | `io_utils.py`, `model_lgb.py --stage assemble`, `check_submission.py`, `validate_submission.py` | Both TSVs (one row per test S1, `\n` line endings, asserts no `\r`), organiser validator (unchanged copy), explicit rule checks. |

The metric in `metric.py` is the official macro F0.5 with singletons included. Earlier stages are kept because later
ones use them: `finalize.py` (rule baseline, submission #1, id-list helpers), `scorer_v2.py` (submission #2; its tuned
config feeds the `v2_score` feature), `split.py`, `data_checks.py`, `tune_pass_c*.py`.

## Environment

- **Python 3.11.9** (Kaggle runs: Python 3.12.13 with the same pins). All dependencies are pinned in `requirements.txt`
  (licences: MIT / BSD / Apache-2.0 / ISC; no GPL).
- Reference hardware: Windows 11, 16 GB RAM, 8 cores (no GPU used). The full-size heavy stages need ~30 GB RAM and were
  run on Kaggle CPU kernels (4 vCPU, ~31 GB RAM); see "Running on Kaggle".

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows  (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Data paths

Put the organiser files, unchanged, in one folder with `train/` and `test/` sub-folders
(`train/train_source{1,2,3}.tsv`, `train/train_ground_truth.tsv`, `test/test_source{1,2,3}.tsv`). No other input is used.
Every stage takes the roots from CLI flags `--data-root/--cache-root/--output-root/--log-root`, else from the env vars
`DATA_ROOT/CACHE_ROOT/OUTPUT_ROOT/LOG_ROOT`, else from the defaults `<repo>/student_resource/dataset`, `<repo>/cache`,
`<repo>/output`, `<repo>/logs` (`<repo>` = two levels above this folder). Every stage caches its output as Parquet,
skips cached work, and accepts `--force` where it has one. Threads: `N_THREADS` (default `os.cpu_count()`).

## Reproduce end to end

Run from this folder (`code/business_entity_resolution/`). One command runs every stage below in order:

```bash
python -m src.run_pipeline --data-root <organiser data> --work <work dir>
# -> <work dir>/output/final/matching_results.tsv, candidate_pairs.tsv (+ organiser validator run)
```

The same stages as individual commands (set `DATA_ROOT`, `CACHE_ROOT`, `OUTPUT_ROOT`, `LOG_ROOT` first; `FEATURE_VARIANT`
is `v1` by default and must be `v4` where stated; PowerShell: `$env:FEATURE_VARIANT = "v4"`):

| # | Command | Output | Kaggle wall time / peak RAM |
|---|---------|--------|-----------------------------|
| 1 | `python -m src.prepare_data` | `cache/raw/*.parquet` incl. `train_pairs` | 3–4 min / 3.3 GB |
| 2 | `python -m src.split` | `cache/split.parquet` | < 1 min / 1.1 GB |
| 3 | `python -m src.normalize --stage normalize` | `cache/norm/{train,test}_s{1,2,3}.parquet` | 4–15 min / 2.5 GB |
| 4 | `python -m src.benchmark --stage build`, then `--stage block` | `cache/bench/{s1,queries,union}.parquet` (5 region folds) | 6 + 24 min / 5.4 GB |
| 5 | `python -m src.pair_table --split bench`, then `python -m src.scorer_v2 --stage eval` | v1 bench pair table, `logs/scorer_v2_config.json` (v2 rule score feature) | 6 + 22 min |
| 6 | `FEATURE_VARIANT=v4 python -m src.blocking_v4 --split bench` | cross-script dictionary, v4 bench candidates `cache/bench_v4/` | 39 min / 8.4 GB |
| 7 | `FEATURE_VARIANT=v4`: `python -m src.pair_table --split bench` → `python -m src.features_v3 --split bench` | `output/features_v3_v4/bench/pairs_<C>.parquet` | 7 + 8 min / 32 GB |
| 8 | `FEATURE_VARIANT=v4 python -m src.model_lgb --stage train` | stage 1 / stage 2 / has-match, leave-one-fold-out over folds 1–4, isotonic maps, decoder tuned on OOF → `output/models_v4/`, fold-0 report `logs/model_v4_report.md` | 216 min / 24 GB |
| 9 | `python -m src.blocking --stage block --split test --country France` | test lookup tables `cache/cand/test/{lookup_s1,queries}.parquet` | 5 min |
| 10 | `FEATURE_VARIANT=v4 python -m src.blocking_v4 --split test --country <C>`, for every test country | `output/cand_v4/test/topk_<C>.parquet` | France 12, India 244, US 110 min / 10 GB |
| 11 | `FEATURE_VARIANT=v4`: `python -m src.pair_table --split test` → `python -m src.features_v3 --split test` | `output/features_v3_v4/test/pairs_<C>.parquet` | 23 + 34 min / 22 GB |
| 12 | `FEATURE_VARIANT=v4 SUBMIT_COUNTRY=<C> SUBMIT_PARTS=1 python -m src.model_lgb --stage submit`, for every test country | scores with the mean of the 4 CV fold models, one-to-one, has-match, decoder → `output/submit_parts_v4/<C>.parquet` | France 36, India 115, US 80 min / 9–25 GB |
| 13 | `FEATURE_VARIANT=v4 python -m src.model_lgb --stage assemble` | `output/final/{matching_results,candidate_pairs}.tsv` + organiser validator | 5 min / 6 GB |

Test countries are read from the data (an open set); country is only used to partition the work.
Optional variant (#5, not submitted): `FEATURE_VARIANT=v4 MODEL_FULL=1 python -m src.model_lgb --stage train_full`
(one model per stage on all 5 folds, rounds = 1.1 × mean CV best iteration; 87 min / 19 GB), then steps 12–13 with
`MODEL_FULL=1` (parts in `output/submit_parts_v4full/`).

## Validate

```bash
python src/validate_submission.py --matching <dir>/matching_results.tsv --candidate <dir>/candidate_pairs.tsv \
    --test-dir <organiser data>/test --check-ids          # organiser validator (unchanged copy)
python -m src.check_submission --matching <dir>/matching_results.tsv --candidate <dir>/candidate_pairs.tsv \
    --test-dir <organiser data>/test [--reference <file whose md5 must match>]
```

`check_submission` asserts and prints each rule: exact header, tab-separated two columns, `\n` only, every test S1 id
exactly once, matched ids only existing S2/S3 ids (no S1 / self matches), no duplicates within a list, matches ⊆
candidates per S1, optional md5 equality.

## Smoke test (small sample, ~10 min on a laptop)

```bash
python -m src.run_pipeline --smoke --data-root <organiser data> --work smoke_run
```

It writes a deterministic hash sample of the data to `smoke_run/data` (3% of train S1 with their true matches plus 3%
distractors; 1% of every test source, all countries kept) and runs all 13 stages on it, ending with the organiser
validator on `smoke_run/output/final/`. Then check the result, for example:

```bash
python -m src.check_submission --matching smoke_run/output/final/matching_results.tsv \
    --candidate smoke_run/output/final/candidate_pairs.tsv --test-dir smoke_run/data/test
```

The sample is too small for meaningful scores (the fold-0 report is written to `smoke_run/logs/model_v4_report.md`);
it proves the code path end to end.

## Running on Kaggle

The full-size heavy stages ran as private Kaggle CPU kernels (4 vCPU, ~31 GB RAM, 12 h limit, up to 5 in parallel).
The tooling is in `src/kaggle_runner/`:

- `build_bundle.py` zips `src/` + `requirements.txt` into `code_bundle/amlc2026_code.zip`, uploaded as a private
  Kaggle dataset (`kaggle datasets version -p code_bundle`). The organiser data is a second private dataset.
- `make_kernels.py <plan>` writes `kernels/<slug>/{run_pipeline_kaggle.py, kernel-metadata.json}`; push each with
  `kaggle kernels push -p kernels/<slug>`. Authentication uses the standard Kaggle CLI credentials of the user
  (nothing is stored in the code).
- `run_pipeline_kaggle.py` (the kernel script) prints the environment, installs the pinned requirements, merges the
  cache of attached earlier kernels (`kernel_sources`), runs its `STAGES` as `python -m src.<module>` with per-stage
  wall time and peak RSS, and copies the small results to `/kaggle/working/results/`.

Kernel plan of the submitted version (outputs of each kernel are attached to the next ones):

| Kernel (plan) | Stages | Wall time / peak RAM |
|---|---|---|
| v1 run | prepare_data, normalize, split | ~25 min |
| K8 `bench` | benchmark build + block, pair_table bench, scorer_v2 eval | ~60 min |
| K12a–c `blocking_v4` (France / India / US, parallel) | blocking_v4 test, one country each | 12 / 244 / 110 min / 10 GB |
| K12d `bench_v4` | blocking_v4 bench, pair_table bench, features_v3 bench, model_lgb train | 272 min / 32 GB |
| test lookups (France kernel of the v1 test blocking) | blocking block test France | 5 min |
| K13 `test_features_v4` | pair_table test, features_v3 test | 58 min / 24 GB |
| K14a–c `submit_v4_parts` (France / India / US, parallel) | model_lgb submit, one country each → parts | 36 / 115 / 80 min / 9–25 GB |
| local | model_lgb assemble → both TSVs + validator | 5 min / 6 GB |
| K15 `full_v4` (optional #5) | model_lgb train_full + submit → parts | 87 + 90 min / 19 GB |

## Constraints respected

- **No external data, APIs, geocoders or downloaded dictionaries.** Every rule table, the gazetteer and the
  cross-script dictionary are mined from the provided training files.
- **Models:** LightGBM (MIT) gradient-boosted trees only, far below 8B parameters. No pretrained or neural models.
  All libraries are BSD / MIT / Apache-2.0 / ISC (anyascii instead of the GPL unidecode).
- **Country is never a model feature and never hard-coded.** It is an open set of strings used only to partition the
  work. France, which has no training labels, goes through exactly the same code path.
- **Matches ⊆ candidates:** `matching_results.tsv` is decoded only from the pairs in `candidate_pairs.tsv`
  (asserted by `assemble` and by `check_submission`).
- Fixed seeds (42); `\n`-only line endings enforced by the writers; runtimes and peak memory per stage are logged to
  `<log root>/stage_metrics.jsonl`.

## Layout

```
src/
  run_pipeline.py      end-to-end runner (all stages in order; --smoke)
  make_sample.py       deterministic small sample of the organiser data (smoke test)
  config.py            paths (CLI > env > default), seeds, threads, FEATURE_VARIANT
  io_utils.py          TSV/Parquet I/O; submission writers ('\n' only, assert no b'\r')
  logging_utils.py     loggers, per-stage runtime + peak memory
  prepare_data.py      TSV -> Parquet, train_pairs
  split.py             random hold-out table (early experiments)
  metric.py            official macro F0.5 (+ per-country breakdown)
  text_norm.py         normalisation rules and rule tables
  normalize.py         normalisation runner (per country, process pool)
  geo_units.py         data-driven region / city units
  benchmark.py         geo-dense benchmark (region folds)
  blocking.py          v1 blocking: pass A keys + pass C char TF-IDF, test lookups
  blocking_v4.py       cross-script dictionary, pass D, adaptive top-k
  finalize.py          rule baseline (submission #1), id-list helpers, validator call
  decoy_features.py    decoy-aware pair features (numbers, extra words, similarities)
  pair_table.py        pair-feature tables (bench labelled / test)
  scorer_v2.py         decoy-aware rule scorer (submission #2; v2_score feature)
  data_checks.py       data checks
  features_v3.py       tagged numbers, extra-word lists, group / twin features
  model_lgb.py         two-stage LightGBM + has-match; train / train_full / submit / assemble
  decoder.py           threshold / exact expected-F0.5 decoders, error budget
  check_submission.py  explicit submission rule checks
  validate_submission.py  organiser validator (unchanged copy)
  tune_pass_c*.py      pass-C tuning (experiments)
  kaggle_runner/       Kaggle kernel script, kernel generator, code bundler
README.md              this file
requirements.txt       pinned dependencies (Python 3.11.9)
```
