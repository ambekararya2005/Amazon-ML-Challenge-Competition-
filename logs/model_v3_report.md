# Model v3 - fold 0 (gate)

| model | scope | F0.5 | precision | recall | pred/S1 | % empty | singleton F0.5 |
|---|---|---|---|---|---|---|---|
| baseline | all | 0.7259 | 0.705 | 0.925 | 4.28 | 0.34 | 0.033 |
| baseline | India | 0.7191 | 0.704 | 0.901 | 4.24 | 0.59 | 0.058 |
| baseline | US | 0.7313 | 0.705 | 0.944 | 4.30 | 0.15 | 0.014 |
| v2 | all | 0.8230 | 0.880 | 0.821 | 3.25 | 6.96 | 0.679 |
| v2 | India | 0.7113 | 0.795 | 0.728 | 3.22 | 7.93 | 0.479 |
| v2 | US | 0.9100 | 0.946 | 0.893 | 3.28 | 6.20 | 0.831 |
| stage1+threshold | all | 0.9617 | 0.987 | 0.931 | 3.25 | 6.28 | 0.956 |
| stage1+threshold | India | 0.9437 | 0.983 | 0.897 | 3.16 | 6.68 | 0.942 |
| stage1+threshold | US | 0.9757 | 0.990 | 0.957 | 3.33 | 5.97 | 0.966 |
| stage2+threshold | all | 0.9638 | 0.989 | 0.934 | 3.25 | 6.43 | 0.970 |
| stage2+threshold | India | 0.9468 | 0.985 | 0.902 | 3.16 | 6.89 | 0.960 |
| stage2+threshold | US | 0.9770 | 0.991 | 0.959 | 3.33 | 6.08 | 0.977 |
| stage1+decoder | all | 0.9618 | 0.986 | 0.929 | 3.23 | 5.83 | 0.916 |
| stage1+decoder | India | 0.9446 | 0.982 | 0.896 | 3.14 | 6.12 | 0.900 |
| stage1+decoder | US | 0.9752 | 0.989 | 0.954 | 3.31 | 5.60 | 0.928 |
| stage2+decoder | all | 0.9649 | 0.989 | 0.930 | 3.23 | 6.00 | 0.943 |
| stage2+decoder | India | 0.9485 | 0.985 | 0.899 | 3.13 | 6.28 | 0.924 |
| stage2+decoder | US | 0.9777 | 0.992 | 0.955 | 3.30 | 5.79 | 0.957 |
| stage2+decoder+hasmatch | all | 0.9652 | 0.990 | 0.930 | 3.22 | 6.15 | 0.957 |
| stage2+decoder+hasmatch | India | 0.9486 | 0.986 | 0.898 | 3.13 | 6.45 | 0.939 |
| stage2+decoder+hasmatch | US | 0.9781 | 0.993 | 0.954 | 3.30 | 5.93 | 0.971 |

OOF folds 1-4 (tuning): {'stage1+threshold': 0.9654, 'stage2+threshold': 0.96607, 'stage1+decoder': 0.96545, 'stage2+decoder': 0.967, 'stage2+decoder+hasmatch': 0.96738}; chosen: **stage2+decoder+hasmatch**

## Error budget (points of F0.5 lost, fold 0)

| model | scope | singleton non-empty | FP decoy | FP other | FN blocking | FN scoring | total |
|---|---|---|---|---|---|---|---|
| v2 | all | 0.0180 | 0.0387 | 0.0277 | 0.0174 | 0.0752 | 0.1770 |
| v2 | India | 0.0288 | 0.0740 | 0.0332 | 0.0322 | 0.1206 | 0.2887 |
| v2 | US | 0.0096 | 0.0112 | 0.0234 | 0.0059 | 0.0398 | 0.0900 |
| stage2+decoder+hasmatch | all | 0.0024 | 0.0036 | 0.0025 | 0.0145 | 0.0119 | 0.0348 |
| stage2+decoder+hasmatch | India | 0.0034 | 0.0055 | 0.0024 | 0.0260 | 0.0141 | 0.0514 |
| stage2+decoder+hasmatch | US | 0.0016 | 0.0020 | 0.0025 | 0.0055 | 0.0102 | 0.0219 |

## Top-25 features (gain %)

| stage 1 | stage 2 |
|---|---|
| s1_rank 49.11 | p1 49.42 |
| cheap 13.31 | p1_q_margin 40.94 |
| b_num_match 10.04 | p1_minus_s1max 7.27 |
| q_rank 5.14 | p1_s1_max 1.36 |
| q_gap_to_best 2.99 | p1_s1_cnt05 0.2 |
| tw_v2_diff 2.69 | s1_claims 0.06 |
| te_q_max 2.53 | p1_q_other_max 0.06 |
| v2_score 1.44 | p1_s1_sum 0.06 |
| q_v2_margin 1.02 | te_q_max 0.05 |
| st_jaccard 0.88 | p1_s1_rank 0.03 |
| num_logdiff 0.82 | tw_addr_max 0.03 |
| b_addr_sim 0.75 | te_q_mean 0.03 |
| te_q_sum 0.73 | p1_s1_second 0.03 |
| tw_num_better 0.72 | te_q_sum 0.03 |
| q_margin 0.57 | q_margin 0.02 |
| addr_tset 0.55 | g_size 0.02 |
| te_q_mean 0.51 | p1_s1_other_max 0.02 |
| st_conflict 0.41 | name_tset 0.01 |
| name_jw 0.39 | te_s_max 0.01 |
| tw_addr_max 0.37 | tw_combo_max 0.01 |
| num_agree_major 0.37 | st_jaccard 0.01 |
| b_name_sim 0.29 | base_score 0.01 |
| s1_claims 0.28 | q_gap_to_best 0.01 |
| g_rank_v2 0.24 | name_jw 0.01 |
| q_v2_gap 0.24 | g_rank_addr 0.01 |

Has-match importance: [['hm_max', 57.12], ['hm_all_max', 23.54], ['hm_all_sum', 7.71], ['hm_cnt05', 5.66], ['hm_sum', 2.75], ['s1_name_freq', 0.83], ['hm_best_twin', 0.55], ['hm_best_addr', 0.42], ['hm_second', 0.39], ['hm_claimants', 0.25], ['hm_best_v2', 0.21], ['hm_best_name', 0.21], ['hm_n_pairs', 0.15], ['hm_cnt', 0.08], ['hm_noextra', 0.07], ['hm_noconf', 0.07], ['s1_addr_missing', 0.0]]

Tuned: {"stage1_threshold": {"oof_f05": 0.9654, "t_accept": 0.675, "t_keep": 0.675}, "stage2_threshold": {"oof_f05": 0.96607, "t_accept": 0.675, "t_keep": 0.675}, "stage1_decoder": {"oof_f05": 0.96545, "temperature": 0.85, "miss": 0.0}, "stage2_decoder": {"oof_f05": 0.967, "temperature": 1.0, "miss": 0.0}, "stage2_decoder_hasmatch": {"oof_f05": 0.96738, "temperature": 1.0, "miss": 0.0}}

