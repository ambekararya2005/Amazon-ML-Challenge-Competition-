# Leaderboard submissions

| # | datetime (IST) | commit | change | local F0.5 (all / US / India) | public LB score |
|---|----------------|--------|--------|-------------------------------|-----------------|
| 1 | 2026-09-26 17:40 (uploaded after) | 6a7798c | Baseline: blocking v1 (pass A + pass C char 4-gram max_df 3%) -> stage-1 top-5 (cos_C + 0.2 x keys_A) -> rule score 0.3 name + 0.6 addr + 0.1 num, thr 0.64, one-to-one. Validator PASS. Local F0.5 is optimistic (sparse validation distractors; test predicts 5.2/S1 vs ~4.3 true). | 0.9641 / 0.9716 / 0.9529 | **0.636** (vs local 0.964 — under audit) |
| 2 | 2026-09-26 20:50 (prepared, not yet uploaded) | (uncommitted; code as of e0414ce + scorer v2) | Rule scorer v2: 0.3 name_tsort + 0.5 addr_tsort + 0.2 num_compatible - 0.1 extra-token penalty, num_conflict veto, one-to-one, t=0.64 (tuned on geo-dense benchmark folds 1-4). Test: 3.1-3.5 pred/S1, 5-9% empty, 54-63% S2/S3 assigned. | fold 0: 0.8230 / US 0.9100 / India 0.7113 (baseline on fold 0: 0.7259 vs LB 0.636) | _pending upload_ |
