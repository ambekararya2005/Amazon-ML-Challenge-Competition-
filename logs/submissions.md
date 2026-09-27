# Leaderboard submissions

| # | datetime (IST) | commit | change | local F0.5 (all / US / India) | public LB score |
|---|----------------|--------|--------|-------------------------------|-----------------|
| 1 | 2026-09-26 17:40 (uploaded after) | 6a7798c | Baseline: blocking v1 (pass A + pass C char 4-gram max_df 3%) -> stage-1 top-5 (cos_C + 0.2 x keys_A) -> rule score 0.3 name + 0.6 addr + 0.1 num, thr 0.64, one-to-one. Validator PASS. Local F0.5 is optimistic (sparse validation distractors; test predicts 5.2/S1 vs ~4.3 true). | 0.9641 / 0.9716 / 0.9529 | **0.636** (vs local 0.964 — under audit) |
| 2 | 2026-09-26 21:14 | 5978f75 | Rule scorer v2: 0.3 name_tsort + 0.5 addr_tsort + 0.2 num_compatible - 0.1 extra-token penalty, num_conflict veto, one-to-one, t=0.64 (tuned on geo-dense benchmark folds 1-4). Test: 3.1-3.5 pred/S1, 5-9% empty, 54-63% S2/S3 assigned. | fold 0: 0.8230 / US 0.9100 / India 0.7113 (baseline on fold 0: 0.7259 vs LB 0.636) | **0.829** |
| 3 | 2026-09-27 13:45 (files ready) | uncommitted (on top of 5978f75) | Two-stage LightGBM pair model (v3 features: tagged numbers, extra-word target encoding, group / twin features) + S1 has-match model + exact expected-F0.5 decoder, one-to-one. v1 candidates (top-5). Kaggle K11 (4.3 h; stage marked FAILED after both files were written - validated locally: PASS, 1,732,544 rows, 5.9% empty). Test: France 3.37 pred/S1, 5.45% empty, 61.0% S2/S3 assigned; India 3.26 / 6.26 / 56.0; US 3.49 / 5.63 / 60.6. | fold 0: 0.9652 / US 0.9781 / India 0.9486 | _pending upload_ |

## Calibration (after submission #2)
- #2: public LB 0.829 vs fold-0 0.823 (+0.006). #1: LB 0.636 vs fold-0 0.726 (the benchmark was built after #1 and
  still over-rated the decoy-blind baseline). From now on **fold 0 of the geo-dense benchmark is the gate** for every
  submission: prepare a test submission only when fold 0 beats the last uploaded version by >= 0.02.
- Best-so-far submission files are kept in `output/best/` (never overwritten; new bests go in a new sub-folder).
