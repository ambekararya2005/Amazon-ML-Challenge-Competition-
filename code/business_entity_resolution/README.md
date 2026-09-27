# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 (S1) entity, find all matching Source 2 / Source 3 records. The pipeline writes
`output/matching_results.tsv` (final matches) and `output/candidate_pairs.tsv` (the exact candidate set the
model scores; every match is one of these candidates).

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
| Normalize | `text_norm.py`, `normalize.py` | anyascii transliteration, lower-casing, legal-form / abbreviation canonicalisation (`pvt`→`private`, `st`→`street`, transliterated legal words mined from the data), `name_core` (legal forms removed), `name_compact`, `addr_clean`, ordered `numbers` (PMB / PO Box removed). Rules only, no country branches. |
| Blocking pass A | `blocking.py` | Exact keys: each address number × each of the 2 rarest address tokens (rarity = S1 document frequency). Blocking runs in the reverse direction: each S2/S3 record queries the S1 index of its own country. |
| Blocking pass C | `blocking.py`, `blocking_v4.py` | char_wb 4-gram TF-IDF (max_df 3%) on name + address, sparse top-k cosine (`sparse_dot_topn`), per country. v4 first maps tokens through a cross-script dictionary mined from train true pairs (e.g. `mharastr`→`maharashtra`, `eksports`→`exports`). |
| Blocking pass D | `blocking_v4.py` | Rarest name token (S1 document frequency ≤ 50) → top-3 S1 by token_sort ratio; rescues typo'd / transliterated names. |
| Top-k | `blocking_v4.py` | cheap = cosine_C + 0.2 × shared_keys_A; keep top-5, or top-8 when the score gap between the 1st and 5th candidates is < 0.1; pass-D candidates are always kept. This shortlist is `candidate_pairs.tsv`. |
| Pair features | `pair_table.py`, `decoy_features.py`, `features_v3.py` | Name / address token-set / token-sort / Jaro-Winkler similarities; tagged address numbers (street, floor) with compatible / conflicting counts; extra name words; group ranks within each S1's claimants and each query's S1s; near-twin ("sibling") features. |
| Model | `model_lgb.py` | Extra-word target encoding (out-of-fold); LightGBM stage 1 → p1; stage 2 on p1 + group aggregates → p2 (isotonic); S1-level has-match model → h. |
| Decode | `decoder.py`, `model_lgb.py` | One-to-one, then per S1 the prefix of candidates sorted by p2 (including the empty set) that maximises the exact expected F0.5, with P(singleton) = 1 − h. |
| Write | `io_utils.py` | Writes both TSVs (one row per test S1, `\n` line endings, asserts no `\r`). |

Validation: `benchmark.py` + `geo_units.py` build a **geo-dense benchmark**. Whole (country, region) units of train are
sampled so that near-duplicate decoys stay together with their true matches, as they do in the test set. The benchmark
has 5 folds by region: folds 1–4 are used for training / tuning (leave-one-fold-out) and fold 0 is the gate, which tracks
the public leaderboard (see `Documentation_template.md`). The metric in `metric.py` is the official macro F0.5 with
singletons included.

Earlier stages kept for reproducibility of the logged experiments: `finalize.py` (rule baseline, submission #1),
`scorer_v2.py` (decoy-aware rule scorer, submission #2; its tuned config also feeds the `v2_score` feature),
`data_checks.py`, `tune_pass_c*.py`, `split.py`.

## Environment

- Python 3.11.9 locally (Kaggle: 3.12.13). All dependencies are pinned in `requirements.txt`.
- Reference hardware: Windows 11, 16 GB RAM, 8 cores, RTX 4050 laptop GPU (the GPU is not used). The heavy stages ran
  on Kaggle CPU kernels (4 vCPU, ~31 GB RAM, 12 h limit).

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows  (Linux: source .venv/bin/activate)
pip install -r code/business_entity_resolution/requirements.txt
```

## Data

Place the organiser files unchanged under `student_resource/dataset/{train,test}/` (or pass `--data-root`). No other
input is used.

## Reproduce end to end

Run from `code/business_entity_resolution/`. Every stage caches its output as Parquet and skips work that is already
cached; add `--force` to recompute. The final model uses the v4 candidates, so set `FEATURE_VARIANT=v4`
(PowerShell: `$env:FEATURE_VARIANT = "v4"`) for steps 5–8.

| # | Command | Output | Runtime / peak RSS |
|---|---------|--------|--------------------|
| 0 | `python -m src.prepare_data` | `cache/raw/*.parquet` incl. `train_pairs.parquet` | 3–4 min / 3.3 GB |
| 1 | `python -m src.split` | `cache/split.parquet` | < 1 min / 1.1 GB |
| 2 | `python -m src.normalize --stage normalize` | `cache/norm/{train,test}_s{1,2,3}.parquet` | 4–15 min / 2.5 GB |
| 3 | `python -m src.benchmark --stage build` then `--stage block` | `cache/bench/{s1,queries,union}.parquet` | 6 + 24 min / 5.4 GB |
| 4 | `python -m src.pair_table --split bench` then `python -m src.scorer_v2 --stage eval` | v1 bench pair table, `logs/scorer_v2_config.json` (rule score used as a feature) | 6 + 22 min |
| 5 | `python -m src.blocking_v4 --split bench` | `cache/bench_v4/{union,dictionary}.parquet`, `logs/blocking_v4_report.json` | 39 min / 8.4 GB |
| 6 | `python -m src.pair_table --split bench` → `python -m src.features_v3 --split bench` → `python -m src.model_lgb --stage train` | `cache/bench_v4/pairs.parquet`, `<output>/features_v3_v4/bench/`, models in `<output>/models_v4/`, `logs/model_v4_report.{json,md}` | 7 + 8 + 216 min / 32 GB |
| 7 | `python -m src.blocking --stage block --split test --country France` (writes the test lookups `cache/cand/test/{lookup_s1,queries}.parquet`), then `python -m src.blocking_v4 --split test --country <C>` for every test country | `<output>/cand_v4/test/topk_<C>.parquet` | France 12 min, India 244 min, US 110 min / 10 GB |
| 8 | `python -m src.pair_table --split test` → `python -m src.features_v3 --split test` | `<output>/features_v3_v4/test/pairs_<C>.parquet` | 23 + 34 min / 22 GB |
| 9 | for every test country: `SUBMIT_COUNTRY=<C> SUBMIT_PARTS=1 python -m src.model_lgb --stage submit` | `<output>/submit_parts_v4/<C>.parquet` + `<C>_report.json` | France 36, India 115, US 80 min / 9–25 GB |
| 10 | `python -m src.model_lgb --stage assemble` (`FINAL_SUBDIR=<dir>`, default `final`; `PARTS_DIR` if the parts are elsewhere) | `output/<dir>/{matching_results,candidate_pairs}.tsv`, validator run | 5 min / 6 GB |

Step 9 without `SUBMIT_PARTS` scores every country in one process and writes the two TSVs directly (needs ~30 GB).
Optional (submission #5): `MODEL_FULL=1 python -m src.model_lgb --stage train_full` refits one model per stage on all
5 benchmark folds (rounds = 1.1 × mean CV best iteration; 87 min / 19 GB), then steps 9–10 with `MODEL_FULL=1`
(parts in `submit_parts_v4full/`).

Runtimes are Kaggle 4-vCPU wall clock. Steps 6 and 8 need roughly 30 GB of RAM, so run them on Kaggle or on a
machine of that size. Then validate:

```bash
cd student_resource
python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

Unit tests live in the development repository (`code/business_entity_resolution/tests/`, run with
`python -m unittest -v`); they are not needed to reproduce the outputs and are not shipped in the submission zip.

Directory overrides for every stage: CLI `--data-root/--cache-root/--output-root/--log-root`, else env
`DATA_ROOT/CACHE_ROOT/OUTPUT_ROOT/LOG_ROOT`, else the local defaults above. Threads: `N_THREADS` (default
`os.cpu_count()`); process-pool workers: `--workers` / `N_WORKERS`; blocking RAM gate: `--min-free-gb` / `MIN_FREE_GB`.

### On Kaggle

The heavy stages were run as Kaggle CPU kernels. Each kernel reuses the cache of earlier kernels through
`kernel_sources`, and several kernels run in parallel, one per country. The kernel tooling lives in the repository's
`kaggle/` folder (not part of this package): `build_bundle.py` zips `src/` + `requirements.txt` into a private code
dataset, and `make_kernels.py <plan>` writes the kernel folders. Kernel plan used for the final model:

| Kernel | Stages (module) |
|---|---|
| v1 run | load, normalize, split |
| K8 bench | benchmark build + block, pair_table bench, scorer_v2 eval |
| K12a–c blocking-v4-{france,india,us} | blocking_v4 test, one country each (in parallel) |
| K12d bench-v4 | blocking_v4 bench, pair_table bench, features_v3 bench, model_lgb train |
| K13 test-features-v4 | pair_table test, features_v3 test |
| K14a–c submit-v4-{france,india,us} | model_lgb submit, one country each → parts (submission #4) |
| K15 full-v4 | model_lgb train_full + submit → parts (submission #5) |
| local | model_lgb assemble → both TSVs + validator |

## Constraints respected

- **No external data, APIs, geocoders or downloaded dictionaries.** Every rule table, gazetteer and the cross-script
  dictionary is mined from the provided training files.
- **Models:** LightGBM (MIT) gradient-boosted trees only, with far fewer than 8B parameters. No pretrained or neural
  models. All libraries are BSD / MIT / Apache-2.0 / ISC (anyascii instead of the GPL unidecode).
- **Country is never a model feature and never hard-coded.** It is treated as an open set of strings and used only to
  partition work (one country at a time), because true matches always share the country label. France, which has no
  training labels, goes through exactly the same code path.
- **Matches ⊆ candidates:** `matching_results.tsv` is decoded only from the pairs in `candidate_pairs.tsv`. This is
  checked when the submission zip is built.
- Fixed seeds (42); `\n`-only line endings enforced by the writers; runtimes and peak memory per stage are logged to
  `logs/stage_metrics.jsonl`.

## Layout

```
src/
  config.py          paths (CLI > env > default), seeds, threads, FEATURE_VARIANT
  io_utils.py        TSV/Parquet I/O; submission writers ('\n' only, assert no b'\r')
  logging_utils.py   loggers, per-stage runtime + peak memory
  prepare_data.py    stage 0: TSV -> Parquet, train_pairs
  split.py           15% validation hold-out (early experiments)
  metric.py          official macro F0.5 (+ per-country breakdown)
  text_norm.py       normalisation rules and rule tables
  normalize.py       stage 2 runner (per country, process pool)
  blocking.py        v1 blocking: pass A keys + pass C char TF-IDF, lookups, combine, recall
  blocking_v4.py     cross-script dictionary, pass D, adaptive top-k
  geo_units.py       data-driven region / city units
  benchmark.py       geo-dense benchmark (region folds)
  finalize.py        stage-1 reduction + rule baseline (submission #1)
  decoy_features.py  decoy-aware pair features (numbers, extra words, similarities)
  pair_table.py      pair-feature tables (bench labelled / test)
  scorer_v2.py       decoy-aware rule scorer (submission #2)
  data_checks.py     step-0 data checks
  features_v3.py     tagged numbers, extra-word lists, group / twin features
  model_lgb.py       two-stage LightGBM + has-match model; train / train_full / submit / assemble
  decoder.py         threshold / exact expected-F0.5 decoders, error budget
  tune_pass_c*.py    pass-C speed / recall tuning
README.md            this file
requirements.txt     pinned dependencies
```
