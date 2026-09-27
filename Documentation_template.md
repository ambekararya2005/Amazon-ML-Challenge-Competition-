# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ENIGMA
**Team Members:** Arya Ambekar, Ishan Ambekar, Arnav Sirse, Mervin Jude
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Each Source 2 / Source 3 record queries the Source 1 records of its own country through three blocking passes: exact
number × rare-token keys, char 4-gram TF-IDF over a cross-script dictionary mined from the training data, and a rarest-name-token pass. An adaptive top-k keeps a short
candidate list. A two-stage LightGBM pair model, an S1-level "has any match" model, a one-to-one constraint and an
exact expected-F0.5 set decoder then choose the matches. The key finding came from an audit of our first submission:
the test set contains many **near-duplicate decoys**, such as sibling businesses with an extra word or a different
house number. A random validation split hides them. We therefore built a **geo-dense benchmark** of whole regions, whose
fold 0 tracks the public leaderboard, and designed features aimed at the decoys: number compatibility, target-encoded
extra words, and group / sibling features.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA of the provided files:

- Train S1 / S2 / S3 = 2.21M / 5.03M / 5.29M records (US, India). Test = 1.73M / 4.89M / 5.08M and adds **France**
  (259k S1), which has **no training labels**.
- 5.58% of S1 entities are singletons. Non-singletons have 3.67 matches on average (max 11). S2 matches per S1 range
  from 0 to 5 and S3 matches from 0 to 6 (no per-source cap).
- **Strict one-to-one:** no S2/S3 record is linked to more than one S1, and about 26% of S2/S3 records match nothing.
- True matches always share the country label, so country is a safe partition key.
- Noise: abbreviations and legal forms (`Pvt Ltd` / `Private Limited`), transliterated legal words in Indian names
  (`praivet`, `limitet`, `elelpi`), non-Latin scripts (15% of Indian names), empty addresses (3.4% of S2/S3), and dropped
  or changed city names (the city differs in 21.8% of true pairs). House numbers such as `#12` are real signal: they
  appear in the S1 address of 96% (US) / 72% (India) of true pairs.

**Decoy discovery (audit of submission #1).** Our first rule baseline scored 0.964 on a random 15% hold-out but
**0.636** on the public leaderboard. We audited it with an independent script that uses only the raw strings:

- There was no ID-mapping bug: 16,000 sampled rows round-tripped 100%, there were no cross-country pairs, and no id was
  duplicated or unknown.
- Predicted pairs looked plausible (median name similarity 91–100, address 90–95), but they shared a house number less
  often than true pairs (61–75% vs 82–84%).
- The baseline assigned 90% of test S2/S3 records (about 74% expected), made 5.2 predictions per S1 (3.46 in train), and
  left only 0.04–0.52% of S1 empty (5.6% singletons expected).
- Cause: **hard-negative siblings**, which fall into two kinds:
  - the same name plus one extra word (Holdings, International, Exports, Group, Partners, Ventures, …);
  - the same street with a different house number.

  `token_set_ratio` ignores extra words, and the number weight was only 0.1. The random hold-out held just 300k random
  distractors, so these siblings were almost never present in validation.
- Later checks (`data_checks.py`) showed that decoys are mutated copies of the entity itself, not of one particular
  sibling. Their main number equals the S1's in 65.9% of cases. Only 0.14% are identical copies.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + set-level decoder (with one-to-one constraint)
**Core Innovation:** A decoy-aware setup: a geo-dense benchmark whose fold 0 predicts the leaderboard, features aimed at
decoys (number compatibility, target-encoded extra words, group and sibling ranks), and an exact expected-F0.5 decoder
combined with an S1 has-match probability.

**Geo-dense benchmark (validation design).** A realistic density of decoys is kept by sampling **whole geographic
units** instead of random entities:

- Units are (country, region). Region keys are found from the S1 addresses without any hand-written rules
  (`geo_units.py`). Each S2/S3 region is learned from how address tokens co-occur with S1 regions in train true pairs.
  City-level units would have split 21.8% of true pairs; regions split only 0.18%.
- The sample is 4 Indian + 13 US regions: 535,608 S1 and 2.88M queries. Records without a region (4.8%, mostly with
  empty addresses) are added as extra distractors.
- There are **5 folds by region**. Folds 1–4 are used for leave-one-fold-out training and tuning; **fold 0 is the gate**
  and is never used for fitting. A new version is submitted only if fold 0 improves by at least 0.02.

Calibration of fold 0 against the public leaderboard:

| # | Version | Fold-0 F0.5 (all / India / US) | Public LB |
|---|---|---|---|
| 1 | Rule baseline (blocking v1, weights 0.3 name / 0.6 addr / 0.1 number, threshold 0.64, one-to-one) | 0.7259 / 0.7191 / 0.7313 | **0.636** |
| 2 | Decoy-aware rule scorer v2 | 0.8230 / 0.7113 / 0.9100 | **0.829** |
| 3 | LightGBM two-stage + has-match + expected-F0.5 decoder, v1 candidates, v3 features | 0.9652 / 0.9486 / 0.9781 | **0.941** |
| 4 | Same model on blocking-v4 candidates (mean of the 4 CV fold models) | 0.9719 / 0.9638 / 0.9782 | **0.945** |
| 5 | #4 retrained on all 5 folds (one model per stage, rounds = 1.1 × mean CV best iteration) | 0.9719 (CV estimate; the retrain sees fold 0) | 0.944 |
| FINAL | #4, v4 (blocking v4, stage 1 + stage 2 + has-match + expected-F0.5 decoder, 4 CV fold models) | 0.9719 / 0.9638 / 0.9782 | **0.945** |

The benchmark was built after submission #1 and still over-rated that decoy-blind model (0.726 vs 0.636). From
submission #2 on, fold 0 ranks every version in the same order as the leaderboard. For the LightGBM versions it is
optimistic by 0.024–0.027 (#3 0.9652 vs 0.941, #4 0.9719 vs 0.945); see the test-gap diagnostic in Section 5.

---

## 3. Candidate Generation (Blocking)

Blocking runs in the **reverse direction**: each S2/S3 record (query) searches the S1 records of the same country. This
suits the one-to-one structure and gives every query its own ranked list of candidate S1 entities.

- **Blocking keys used:**
  - *Pass A* (exact keys): each address number × each of the two rarest address tokens (rarity = S1 document
    frequency).
  - *Pass C* (fuzzy): char_wb 4-gram TF-IDF on normalised name + address, `max_df` 3%, sparse top-k cosine
    (`sparse_dot_topn`). In v4 the tokens are first mapped through a **cross-script dictionary** mined from train true
    pairs where one side is non-Latin: 3,739 token pairs such as `mharastr`→`maharashtra`, `dilli`→`delhi`,
    `eksports`→`exports`, `tredimg`→`trading`.
  - *Pass D* (v4): the rarest name token with S1 document frequency ≤ 50, keeping the top-3 S1 by token_sort ratio.
    This rescues typo'd and transliterated names.
  - *Adaptive top-k*: cheap score = cosine_C + 0.2 × shared_keys_A. Keep the top-5, or the top-8 when the cheap score
    of the 1st candidate minus that of the 5th is < 0.1 (ambiguous queries). Pass-D candidates are always kept.
- **Candidate pairs generated (test, v4):** 57.6M query–S1 pairs = France 8.48M / India 27.78M / US 21.35M, i.e.
  5.6–5.9 candidates per query; 5.9–7.7% of the pairs come from pass D only. The v1 union before top-k had 153M
  pairs. Written to `candidate_pairs.tsv` grouped by S1. This is exactly the set the model scores.
- **How you ensured true matches were not lost:** We measured recall on the geo-dense benchmark after every change:

  | Blocking | India | US | All | Candidates / query |
  |---|---|---|---|---|
  | v1 union (before top-k) | 0.9625 | — | — | — |
  | v1 top-5 | 0.9468 | 0.9851 | 0.9715 | 5.00 |
  | v4 top-5, no pass D | 0.9631 | — | — | — |
  | **v4 adaptive top-k** | **0.9691** | **0.9870** | **0.9806** | 5.87 |

  Pass C settings were chosen on a validation sample. A name-only TF-IDF was rejected because it lost 4–10 recall
  points. The error budget below shows that blocking misses cost only 0.0073 of F0.5 on fold 0.

---

## 4. Matching Model

**Features used** (84 pair features in stage 1, 95 in stage 2, 17 S1-level has-match features; all computed per pair or within the candidate set; **country is never a
feature**):

- Name features: token-set / token-sort / ratio / Jaro-Winkler on `name_core` (legal forms removed) and on the
  compact name.
  **Extra words** are name tokens present on one side only. They are listed per pair and **target-encoded
  out-of-fold** as a smoothed P(not a true pair | word is extra). The encoding is aggregated per pair as max / mean /
  sum plus the number of unseen words, separately for the query side and the S1 side. It learns that `holdings`,
  `exports`, `ventures`, `overseas` signal decoys, while `dba`, `sri` / `smt` / `shri`, `formerly`, `center`
  are harmless.
- Address features: token-set / token-sort similarities on the cleaned address. **Number compatibility**: numbers are
  tagged as street, floor / ordinal or postal. The features count compatible and conflicting street numbers, take their
  Jaccard, flag a floor conflict, and compare the main number and its magnitude (log difference). These tolerate
  corrupted digits and sub-numbers (`12-1-331/C/8`).
  - On benchmark folds 1–4, a number conflict appeared in 9.3% of true pairs vs 83.0% of the decoys accepted by the
    baseline.
  - A conflict or an extra word appeared in 45.2% of true pairs vs 97.2% of those decoys.
- Group / sibling features:
  - within each S1's claimants: rank by rule score and by address similarity, gap to the best, number of claimants
    without a conflict or without extra words, same-source claimants;
  - within each query's S1 candidates: rank, margin to the best, number of compatible S1s;
  - near-twin features against the S1's top-8 claimants: max name / address similarity to a sibling, and whether the
    closest sibling agrees better on the street number or has fewer extra words (`tw_num_better`, `tw_extra_better`,
    `tw_v2_diff`).
- Other: the blocking scores (cosine_C, shared keys A, pass flags), the v2 rule score, and missing-field flags.

**Model type:** LightGBM (MIT licence; gradient-boosted trees, far below 8B parameters).
- *Stage 1*: binary pair classifier on all features (num_leaves 127, learning rate 0.05, feature fraction 0.8, early
  stopping 50), trained leave-one-fold-out over benchmark folds 1–4 → out-of-fold p1. Fold 0 and test use the mean of
  the 4 fold models. Top gains: s1_rank 48%, number match 20%, q_rank 12%, tw_v2_diff, q_gap_to_best, te_q_max.
- *Stage 2*: p1 plus its aggregates within the S1 and within the query (max, second, margin, rank, sum) → p2, then
  isotonic calibration (our own PAV implementation) on the out-of-fold predictions.
- *Has-match model*: one row per S1, target = "has ≥ 1 true match", with features built from claimant / p2 statistics
  → h (isotonic). It raised singleton F0.5 on fold 0 from 0.943 to 0.957.

**Threshold selection method:** No fixed threshold is used.

1. **One-to-one**: each query keeps only its argmax-p2 S1.
2. **Exact expected-F0.5 decoder**: for each S1, candidates are sorted by p2, and the prefix (including the empty set)
   that maximises the exact expected F0.5 is chosen.
   - Pair labels are modelled as independent Bernoulli(min(p2 / h, 1)) variables, conditional on the S1 having any
     match.
   - The S1 is a singleton with probability 1 − h:
     E[F | set] = (1 − h)·[set empty] + h·E_Bernoulli[F | set].
   - Matches lost in blocking are modelled as a Poisson term.
   - The temperature (1.0) and miss prior were tuned on out-of-fold folds 1–4.

On fold 0 the decoder beat the best global threshold by +0.0011 to +0.0013.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** fold 0 of the geo-dense benchmark (model v4) **0.9719**: India 0.9638, US 0.9782.
  Precision 0.990, recall 0.946, 3.29 predictions per S1, 5.89% empty, singleton F0.5 0.957. Out-of-fold folds 1–4:
  0.9690.
  FINAL MODEL (#4, v4 with the 4 CV fold models): fold 0 = 0.9719, public LB = 0.945.

| Version (fold 0) | All | India | US | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (#1) | 0.7259 | 0.7191 | 0.7313 | 0.705 | 0.925 |
| Rule scorer v2 (#2) | 0.8230 | 0.7113 | 0.9100 | 0.880 | 0.821 |
| v3 stage 1 + threshold | 0.9617 | — | — | — | — |
| v3 stage 2 + decoder + has-match (#3) | 0.9652 | 0.9486 | 0.9781 | 0.990 | 0.930 |
| v4 stage 2 + decoder + has-match | 0.9719 | 0.9638 | 0.9782 | 0.990 | 0.946 |
| **FINAL MODEL** (#4, v4 (blocking v4, stage 1 + stage 2 + has-match + expected-F0.5 decoder, 4 CV fold models)) | **0.9719** | **0.9638** | **0.9782** | 0.990 | 0.946 |

Error budget, v4 fold 0 (F0.5 points lost; total 0.0281, vs 0.1770 for the v2 rule scorer):

| Source | All | India | US |
|---|---|---|---|
| Non-empty prediction on a singleton | 0.0024 | 0.0034 | 0.0016 |
| False positive, decoy (query matches nobody) | 0.0037 | 0.0060 | 0.0019 |
| False positive, other entity | 0.0028 | 0.0029 | 0.0028 |
| False negative, missed by blocking | 0.0073 | 0.0103 | 0.0049 |
| False negative, missed by scoring | 0.0119 | 0.0136 | 0.0105 |

Test-set statistics per country (final model; France has no labels, so these are the only checks available):

| Country | Predictions / S1 | % S1 empty | % S2/S3 assigned |
|---|---|---|---|
| France | 3.40 | 5.39 | 61.5 |
| India | 3.33 | 5.95 | 57.2 |
| US | 3.49 | 5.66 | 60.7 |
| FINAL MODEL (all) | 3.40 | 5.76 | 59.1 |

The full retrain (#5) agrees with #4 on 97.8% (France), 98.3% (India) and 98.5% (US) of the predicted pairs (both /
union), and every statistic above differs by less than 1% relative.

These are in line with train (3.46 matches per S1, 5.6% singletons; at the test density of 5.76 queries per S1,
3.46 matches per S1 means about 60% of S2/S3 assigned). France sits between US and India on every
statistic, which suggests the country-agnostic pipeline transfers.

**Split of the 0.0119 "missed by scoring"** (v4 fold 0; true pair present in the candidates but not predicted):

| Bucket | F0.5 points (all / India / US) | Missed pairs | p2 of the missed pairs |
|---|---|---|---|
| (a) the query's argmax is the true S1, the decoder did not select it | 0.0096 / 0.0114 / 0.0081 | 13,717 | median 0.53; 71% in 0.3–0.8 |
| (b1) argmax is another S1, which selected the query | 0.0001 | 59 | median 0.12 |
| (b2) argmax is another S1, which did not select it | 0.0023 | 3,083 | median 0.06 |
| (c) other | 0 | 0 | – |

Bucket (a) consists of genuinely uncertain pairs. A "sibling rescue" rule tried to recover them: add an unselected
argmax claimant when p2 ≥ a and it is near-identical (similarity ≥ b, compatible number, no decoy-type extra word) to
a selected match of the same S1. It was tuned over 90 settings on folds 1–4 and never beat the baseline. On fold 0 it
added 162 true pairs and 94 false ones (−0.00005), because decoys are near-copies of the true siblings. The decoder
settings (temperature, miss prior, has-match sharpness) are equally flat (±0.0001).

**Test-gap diagnostic** (the LB is 0.024–0.027 below fold 0):

| | Fold 0 US | Fold 0 India | Test US | Test India | Test France |
|---|---|---|---|---|---|
| Uncertain queries (best p2 in 0.3–0.7) | 2.5% | 3.3% | 3.9% | 3.3% | 5.0% |
| Predictions / S1 | 3.30 | 3.27 | 3.49 | 3.33 | 3.40 |
| % S1 empty | 5.93 | 5.84 | 5.66 | 5.95 | 5.39 |
| Mean has-match h | 0.944 | 0.948 | 0.946 | 0.946 | 0.950 |

France is the most uncertain country (about 2× fold-0 US), and test US is also above fold-0 US. The gap therefore looks
partly test-wide (harder decoys than in the benchmark regions), with the unlabelled France adding to it. A manual
review of 15 random French S1 shows that French legal forms (SARL, SASU, EI, SCI) are handled and that decoys with
another house number or an extra word (International, Distribution, Développement) are rejected.

- **Common false positives (wrong merges):**
  - decoys that differ from the entity only in the legal form (`… Public Limited` vs `… Limited`; legal forms are
    stripped from `name_core`);
  - sibling sub-numbers (`12-1-331/C/8` vs `/C/1`);
  - name-only queries with an empty address, where the number and address features carry no information.
- **Common false negatives (missed matches):**
  - name-only records;
  - records with heavily corrupted or landmark-only addresses;
  - non-Latin (Indian-script) names whose transliteration shares few character n-grams with the Latin S1 name. These
    are the main source of blocking misses in India, even after the cross-script dictionary and pass D.

---

## 6. Conclusion

The largest gain did not come from a stronger model but from **validating on the right distribution**. A random
hold-out rated a decoy-blind baseline at 0.964, while the leaderboard gave it 0.636. The region-level geo-dense
benchmark tracks the leaderboard, and features aimed at decoys (number compatibility, target-encoded extra words, sibling
features) combined with an S1 has-match model and an exact expected-F0.5 decoder raised fold 0 from 0.726 to 0.972.
The pipeline uses only the provided data, small MIT-licensed tree models, and treats country only as a partition key.

**What did not work.**
- A random 15% hold-out (0.964 locally vs 0.636 on the leaderboard): it contained almost no decoys.
- `token_set_ratio`-style name similarity, which ignores extra words, and a low weight on house numbers.
- Name-only TF-IDF blocking (−4 to −10 recall points).
- Post-processing on top of the decoder: a sibling-rescue rule (−0.00005 on fold 0) and re-tuning the decoder (±0.0001).
- Refitting on all 5 folds (#5, one model per stage): LB 0.944 vs 0.945 for the average of the 4 CV fold models (#4).

**Limitations.**
- *India, non-Latin scripts*: anyascii transliteration and the mined dictionary cover the common tokens, but rare
  words in Indic scripts still lose recall. India remains about 1.4 points below the US on fold 0.
- *France is unlabelled*: no French pair was ever seen in training, and the fold-0 gate covers only US and India.
  French quality is inferred from test statistics (prediction rate, empty rate, coverage), not measured. The
  dictionary, gazetteer and extra-word encodings are learned from US / India data; unseen French extra words fall back
  to the "unseen word" count.
- The heavy stages need about 30 GB of RAM (feature tables, 4-fold model training) and were run on Kaggle CPU kernels.

**What we would do next.**
- *Sibling-p1 stage-2 features*: for each claimant, the p1 of its most similar claimant from the other source and
  whether it agrees with the highest-p1 sibling on the street number and extra words. The text-only version already
  exists (twin features); adding the siblings' model scores targets bucket (a) above (0.0096 points).
- *Higher blocking recall for non-Latin names* (0.0103 points lost in India): a larger or character-level
  transliteration dictionary, and a phonetic key for Indic-script names.
- *3-seed averaging* of stage 1 and stage 2, together with a lower learning rate (0.03) and more rounds.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds all source code in `src/`, the `README.md` with exact commands, and the pinned
`requirements.txt`. Each stage is a CLI module (`python -m src.<module>`) that caches its output as Parquet and accepts
`--force`:

1. `prepare_data` — organiser TSVs → Parquet
2. `normalize` — per-country normalisation
3. `benchmark` (`--stage build`, `--stage block`) — geo-dense benchmark
4. `blocking` (`--split test`, test lookups) and `blocking_v4` (`--split bench|test`) — candidate generation
5. `pair_table` — pair-feature tables
6. `features_v3` — v3 features
7. `model_lgb --stage train` — fits the models on the benchmark (leave-one-fold-out over folds 1–4, fold-0 report);
   `--stage train_full` with `MODEL_FULL=1` refits one model per stage on all 5 folds (submission #5)
8. `model_lgb --stage submit` — scores test (`SUBMIT_COUNTRY=<c>`, `SUBMIT_PARTS=1`: one part per country), then
   `model_lgb --stage assemble` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` through
   `io_utils` (`\n` line endings, validator-compatible)

Set `FEATURE_VARIANT=v4` for steps 5–8. `python -m src.run_pipeline --data-root <data> --work <dir>` runs every
stage in this order (`--smoke` on a small hash sample made by `src.make_sample`); `src.check_submission` asserts the
submission rules one by one; `src/kaggle_runner/` holds the Kaggle kernel runner, the kernel generator and the code
bundler used for the heavy stages. Supporting modules: `text_norm`, `geo_units`, `decoy_features`, `decoder`,
`metric` (official macro F0.5), `config`, `logging_utils`. Unit tests: `python -m unittest`.

### B. Additional Results

Blocking pass C tuning (validation sample, union recall US / India):

- max_df 0.5%: 0.9819 / 0.9303
- max_df 1%: 0.9862 / 0.9386
- max_df 2%: 0.9877 / 0.9481
- max_df 3%: 0.9881 / 0.9516 (chosen)
- name-only TF-IDF: 0.940–0.951 / 0.856–0.873 (rejected)

Test blocking v1 took 5 min (France), 2 × 113–142 min (India) and 106 min (US) on 4 vCPU kernels. Blocking v4 took
12 / 244 / 110 min.

Model ladder on fold 0 (all countries):

| Step | Fold-0 F0.5 |
|---|---|
| v3 stage 1 + threshold | 0.9617 |
| stage 2 + threshold | 0.9638 |
| stage 1 + decoder | 0.9618 |
| stage 2 + decoder | 0.9649 |
| + has-match | 0.9652 |
| v4 candidates | 0.9719 |

Stage-2 gain is dominated by p1 (67%) and p1_q_margin (26%). The has-match model relies mainly on the maximum claimant
p2.

### C. Licences, data and fair play

- **Models:** LightGBM gradient-boosted trees (MIT). No pretrained models, no neural networks, no LLMs; every model is
  trained from scratch on the provided training files and is far below the 8B-parameter limit.
- **Data:** only the organiser files. No external data, APIs, geocoders or downloaded dictionaries. The cross-script
  dictionary, gazetteer (region keys), legal-form tables and extra-word encodings are all mined from the training data.
- **Country** is never a model feature and is never hard-coded; it is an open set of strings used only to partition the
  work. France (no training labels) runs through exactly the same code path.
- **Dependencies** (pinned in `requirements.txt`, Python 3.11.9; Kaggle runs used Python 3.12.13 with the same pins):

  | Package | Version | Licence | Use |
  |---|---|---|---|
  | lightgbm | 4.7.0 | MIT | stage 1 / stage 2 / has-match models |
  | pandas | 3.0.6 | BSD-3-Clause | tables |
  | numpy | 2.4.6 | BSD-3-Clause | arrays |
  | pyarrow | 25.0.1 | Apache-2.0 | Parquet I/O |
  | scipy | 1.17.1 | BSD-3-Clause | sparse matrices |
  | scikit-learn | 1.9.1 | BSD-3-Clause | TF-IDF vectoriser (blocking pass C) |
  | sparse_dot_topn | 1.2.0 | Apache-2.0 | sparse top-k cosine (blocking pass C) |
  | rapidfuzz | 3.14.6 | MIT | string similarities |
  | anyascii | 0.3.3 | ISC | transliteration (instead of the GPL unidecode) |
  | psutil | 7.2.2 | BSD-3-Clause | memory logging |
  | joblib 1.6.0, threadpoolctl 3.7.0 | | BSD-3-Clause | scikit-learn runtime |
  | narwhals 2.26.0, six 1.17.0 | | MIT | pandas / dateutil runtime |
  | python-dateutil 2.9.0.post0 | | Apache-2.0 / BSD-3-Clause | pandas runtime |
  | tzdata 2026.4 | | Apache-2.0 | pandas runtime |

  No GPL / copyleft dependency. `src/validate_submission.py` is an unchanged copy of the organiser validator.
