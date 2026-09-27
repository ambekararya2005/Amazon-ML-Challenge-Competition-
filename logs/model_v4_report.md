# Model v4 - fold 0 (gate)

| model | scope | F0.5 | precision | recall | pred/S1 | % empty | singleton F0.5 |
|---|---|---|---|---|---|---|---|
| baseline | all | 0.7309 | 0.706 | 0.939 | 4.32 | 0.25 | 0.029 |
| baseline | India | 0.7317 | 0.710 | 0.932 | 4.33 | 0.39 | 0.051 |
| baseline | US | 0.7302 | 0.704 | 0.945 | 4.31 | 0.14 | 0.013 |
| v2 | all | 0.8227 | 0.878 | 0.824 | 3.27 | 6.83 | 0.673 |
| v2 | India | 0.7125 | 0.793 | 0.734 | 3.25 | 7.70 | 0.474 |
| v2 | US | 0.9086 | 0.944 | 0.893 | 3.29 | 6.16 | 0.825 |
| stage1+threshold | all | 0.9681 | 0.987 | 0.945 | 3.30 | 6.01 | 0.954 |
| stage1+threshold | India | 0.9582 | 0.983 | 0.930 | 3.28 | 6.04 | 0.937 |
| stage1+threshold | US | 0.9758 | 0.991 | 0.956 | 3.32 | 5.99 | 0.966 |
| stage2+threshold | all | 0.9706 | 0.988 | 0.949 | 3.31 | 6.11 | 0.967 |
| stage2+threshold | India | 0.9622 | 0.984 | 0.938 | 3.30 | 6.13 | 0.954 |
| stage2+threshold | US | 0.9772 | 0.991 | 0.958 | 3.32 | 6.09 | 0.977 |
| stage1+decoder | all | 0.9684 | 0.986 | 0.943 | 3.28 | 5.56 | 0.916 |
| stage1+decoder | India | 0.9594 | 0.982 | 0.928 | 3.26 | 5.53 | 0.901 |
| stage1+decoder | US | 0.9754 | 0.990 | 0.954 | 3.31 | 5.59 | 0.927 |
| stage2+decoder | all | 0.9716 | 0.989 | 0.946 | 3.28 | 5.75 | 0.943 |
| stage2+decoder | India | 0.9635 | 0.985 | 0.934 | 3.26 | 5.68 | 0.924 |
| stage2+decoder | US | 0.9778 | 0.992 | 0.955 | 3.30 | 5.80 | 0.957 |
| stage2+decoder+hasmatch | all | 0.9719 | 0.990 | 0.946 | 3.29 | 5.89 | 0.957 |
| stage2+decoder+hasmatch | India | 0.9638 | 0.985 | 0.935 | 3.27 | 5.84 | 0.938 |
| stage2+decoder+hasmatch | US | 0.9782 | 0.993 | 0.956 | 3.30 | 5.93 | 0.971 |

OOF folds 1-4 (tuning): {'stage1+threshold': 0.96701, 'stage2+threshold': 0.96767, 'stage1+decoder': 0.96709, 'stage2+decoder': 0.96851, 'stage2+decoder+hasmatch': 0.96895}; chosen: **stage2+decoder+hasmatch**

## Error budget (points of F0.5 lost, fold 0)

| model | scope | singleton non-empty | FP decoy | FP other | FN blocking | FN scoring | total |
|---|---|---|---|---|---|---|---|
| v2 | all | 0.0183 | 0.0394 | 0.0286 | 0.0089 | 0.0821 | 0.1773 |
| v2 | India | 0.0291 | 0.0753 | 0.0340 | 0.0134 | 0.1356 | 0.2875 |
| v2 | US | 0.0099 | 0.0114 | 0.0244 | 0.0053 | 0.0403 | 0.0914 |
| stage2+decoder+hasmatch | all | 0.0024 | 0.0037 | 0.0028 | 0.0073 | 0.0119 | 0.0281 |
| stage2+decoder+hasmatch | India | 0.0034 | 0.0060 | 0.0029 | 0.0103 | 0.0136 | 0.0362 |
| stage2+decoder+hasmatch | US | 0.0016 | 0.0019 | 0.0028 | 0.0049 | 0.0105 | 0.0218 |

## Top-25 features (gain %)

| stage 1 | stage 2 |
|---|---|
| s1_rank 48.27 | p1 66.84 |
| b_num_match 19.8 | p1_q_margin 25.85 |
| q_rank 12.1 | p1_minus_s1max 4.63 |
| tw_v2_diff 3.15 | p1_s1_max 0.94 |
| q_gap_to_best 2.52 | p1_q_rank 0.74 |
| te_q_max 2.21 | p1_s1_rank 0.07 |
| st_conflict 0.97 | p1_q_other_max 0.07 |
| q_v2_margin 0.72 | te_q_max 0.06 |
| st_jaccard 0.68 | s1_claims 0.06 |
| base_score 0.66 | p1_s1_sum 0.05 |
| b_addr_sim 0.62 | p1_s1_other_max 0.05 |
| addr_tset 0.6 | tw_addr_max 0.04 |
| q_margin 0.58 | p1_s1_second 0.03 |
| num_logdiff 0.57 | te_q_sum 0.03 |
| num_agree_major 0.43 | te_q_mean 0.03 |
| tw_addr_max 0.37 | st_jaccard 0.02 |
| te_q_sum 0.32 | q_margin 0.02 |
| name_jw 0.3 | g_size 0.02 |
| te_q_mean 0.25 | name_tset 0.02 |
| s1_claims 0.25 | tw_combo_max 0.01 |
| g_rank_v2 0.24 | te_s_max 0.01 |
| b_name_sim 0.23 | name_jw 0.01 |
| q_v2_gap 0.23 | addr_tsort 0.01 |
| tw_name_max 0.22 | g_rank_addr 0.01 |
| v2_score 0.21 | v2_score 0.01 |

Has-match importance: [['hm_max', 59.37], ['hm_all_max', 27.19], ['hm_sum', 8.48], ['hm_cnt05', 1.5], ['s1_name_freq', 0.69], ['hm_best_twin', 0.51], ['hm_all_sum', 0.47], ['hm_second', 0.41], ['hm_best_addr', 0.38], ['hm_best_name', 0.23], ['hm_claimants', 0.21], ['hm_best_v2', 0.18], ['hm_n_pairs', 0.14], ['hm_cnt', 0.11], ['hm_noextra', 0.07], ['hm_noconf', 0.06], ['s1_addr_missing', 0.0]]

Tuned: {"stage1_threshold": {"oof_f05": 0.96701, "t_accept": 0.675, "t_keep": 0.675}, "stage2_threshold": {"oof_f05": 0.96767, "t_accept": 0.675, "t_keep": 0.675}, "stage1_decoder": {"oof_f05": 0.96709, "temperature": 0.85, "miss": 0.0}, "stage2_decoder": {"oof_f05": 0.96851, "temperature": 1.0, "miss": 0.0}, "stage2_decoder_hasmatch": {"oof_f05": 0.96895, "temperature": 1.0, "miss": 0.05}}

