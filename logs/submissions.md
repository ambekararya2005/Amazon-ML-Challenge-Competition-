# Leaderboard submissions

| # | datetime (IST) | commit | change | local F0.5 (all / US / India) | public LB score |
|---|----------------|--------|--------|-------------------------------|-----------------|
| 1 | 2026-09-26 17:40 (uploaded after) | 6a7798c | Baseline: blocking v1 (pass A + pass C char 4-gram max_df 3%) -> stage-1 top-5 (cos_C + 0.2 x keys_A) -> rule score 0.3 name + 0.6 addr + 0.1 num, thr 0.64, one-to-one. Validator PASS. Local F0.5 is optimistic (sparse validation distractors; test predicts 5.2/S1 vs ~4.3 true). | 0.9641 / 0.9716 / 0.9529 | **0.636** (vs local 0.964 — under audit) |
| 2 | 2026-09-26 21:14 | 5978f75 | Rule scorer v2: 0.3 name_tsort + 0.5 addr_tsort + 0.2 num_compatible - 0.1 extra-token penalty, num_conflict veto, one-to-one, t=0.64 (tuned on geo-dense benchmark folds 1-4). Test: 3.1-3.5 pred/S1, 5-9% empty, 54-63% S2/S3 assigned. | fold 0: 0.8230 / US 0.9100 / India 0.7113 (baseline on fold 0: 0.7259 vs LB 0.636) | **0.829** |
| 3 | 2026-09-27 13:45 (files ready) | uncommitted (on top of 5978f75) | Two-stage LightGBM pair model (v3 features: tagged numbers, extra-word target encoding, group / twin features) + S1 has-match model + exact expected-F0.5 decoder, one-to-one. v1 candidates (top-5). Kaggle K11 (4.3 h; stage marked FAILED after both files were written - validated locally: PASS, 1,732,544 rows, 5.9% empty). Test: France 3.37 pred/S1, 5.45% empty, 61.0% S2/S3 assigned; India 3.26 / 6.26 / 56.0; US 3.49 / 5.63 / 60.6. | fold 0: 0.9652 / US 0.9781 / India 0.9486 | **0.941** (fold 0 0.9652, gap -0.024) |
| 4 | 2026-09-27 17:24 (files ready) | uncommitted (on top of b4ab7d4) | v4-safe: blocking v4 candidates (cross-script dictionary, pass D, adaptive top-k) + v3 features + two-stage LightGBM (CV fold 1-4 models, mean) + has-match + expected-F0.5 decoder (T 1.0, miss 0.05). Per-country kernels K14a-c -> parts -> local assemble (output/final_v4safe/, validator PASS, md5 matching 41b9f157..., candidate 9e8bfb9a...). Test: France 3.40 pred/S1, 5.39% empty, 61.5% S2/S3 assigned; India 3.33 / 5.95 / 57.2; US 3.49 / 5.66 / 60.7. | fold 0: 0.9719 / US 0.9782 / India 0.9638 (+0.0067 vs #3) | _pending upload_ |

## Calibration (after submission #3)
- #3: public LB 0.941 vs fold-0 0.9652 (-0.024); #2 was +0.006, #1 -0.090. Fold 0 still ranks the versions correctly
  (0.726 < 0.823 < 0.965 on fold 0; 0.636 < 0.829 < 0.941 on LB), but it is now optimistic by ~0.02.
  Test S2/S3 assigned 56-61% is consistent with train density (3.46 matches/S1 x 1.73M S1 / 9.97M S2+S3 = 60%).
  Unmeasured parts of the gap: France (no labels), test regions outside the benchmark sample.
- #3 files: output/best/sub3_v3_fold0.965/ (md5 identical to output/*.tsv at upload; matching f9141e7c...,
  candidate 4db5ae3f...).

## Calibration (after submission #2)
- #2: public LB 0.829 vs fold-0 0.823 (+0.006). #1: LB 0.636 vs fold-0 0.726 (the benchmark was built after #1 and
  still over-rated the decoy-blind baseline). From now on **fold 0 of the geo-dense benchmark is the gate** for every
  submission: prepare a test submission only when fold 0 beats the last uploaded version by >= 0.02.
- Best-so-far submission files are kept in `output/best/` (never overwritten; new bests go in a new sub-folder).
