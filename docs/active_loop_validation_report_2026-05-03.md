# Active-Loop Validation Report

Date: 2026-05-03

## Verdict

The artifact-gate selector is the best conservative primary artifact right now,
but not because it dominates the plain blend on coverage.

The evidence is narrower:

- It eliminates likely-artifact and spike selections in the simulated top-K
  prefixes.
- It keeps most of the coverage of the plain TS2Vec/window blend at K=10/K=25.
- It clearly beats the broader trace-gate variant, which was too aggressive
  because it demoted clean low-motion clips.
- It remains below the oracle and trails the plain blend at K=50/K=100, so the
  honest framing is "cleaner conservative selector", not "strictly better
  selector".

The exact-window/plain blend should remain the coverage-forward fallback. The
artifact-gate version should be the primary submission candidate if we are
optimizing for top-ranked trust and artifact avoidance.

Mode: `full`
Representation: `balanced`
Primary policy: `artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05`

## Inputs

- Coverage rows: `/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale/coverage_gain_by_episode_full.csv`
- Selection audit rows: `/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale/topk_selection_audit_full.csv`
- Bootstrap samples: `2000`

## Balanced Relative Gain

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.1157 [0.0877, 0.1464] | 0.1604 [0.1307, 0.1917] | 0.2237 [0.1949, 0.2559] | 0.2666 [0.2391, 0.2966] | 0.3109 [0.2833, 0.3394] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.1111 [0.0832, 0.1420] | 0.1605 [0.1329, 0.1909] | 0.2242 [0.1956, 0.2544] | 0.2722 [0.2453, 0.3004] | 0.3249 [0.2989, 0.3531] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.1150 [0.0878, 0.1455] | 0.1539 [0.1247, 0.1844] | 0.2137 [0.1847, 0.2454] | 0.2551 [0.2272, 0.2858] | 0.2983 [0.2729, 0.3269] |
| kcenter_greedy_quality_gated | 0.0914 [0.0762, 0.1079] | 0.1176 [0.1006, 0.1348] | 0.1844 [0.1657, 0.2029] | 0.2379 [0.2179, 0.2577] | 0.2878 [0.2679, 0.3074] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.0680 [0.0427, 0.0974] | 0.1115 [0.0828, 0.1425] | 0.2002 [0.1730, 0.2314] | 0.2568 [0.2293, 0.2876] | 0.3190 [0.2932, 0.3467] |
| old_novelty_only | 0.0520 [0.0422, 0.0625] | 0.0834 [0.0718, 0.0968] | 0.1649 [0.1444, 0.1840] | 0.2381 [0.2141, 0.2650] | 0.3101 [0.2850, 0.3352] |
| oracle_greedy_eval_only | 0.2532 [0.2285, 0.2815] | 0.3027 [0.2779, 0.3294] | 0.3520 [0.3281, 0.3774] | 0.3676 [0.3429, 0.3941] | 0.3680 [0.3422, 0.3948] |

## Oracle Fraction

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.4907 [0.4230, 0.5603] | 0.5669 [0.5118, 0.6154] | 0.6913 [0.6514, 0.7270] | 0.7862 [0.7597, 0.8119] | 0.8805 [0.8577, 0.9000] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.4726 [0.4063, 0.5405] | 0.5714 [0.5245, 0.6166] | 0.7006 [0.6702, 0.7305] | 0.8120 [0.7922, 0.8316] | 0.9220 [0.9094, 0.9329] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 0.4773 [0.4090, 0.5438] | 0.5426 [0.4877, 0.5948] | 0.6592 [0.6154, 0.6999] | 0.7570 [0.7255, 0.7877] | 0.8483 [0.8185, 0.8751] |
| kcenter_greedy_quality_gated | 0.5472 [0.4913, 0.5993] | 0.5547 [0.5101, 0.6013] | 0.6816 [0.6427, 0.7158] | 0.7841 [0.7566, 0.8088] | 0.8910 [0.8679, 0.9106] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.1894 [0.1425, 0.2375] | 0.2893 [0.2409, 0.3410] | 0.4979 [0.4501, 0.5445] | 0.6623 [0.6206, 0.7008] | 0.8425 [0.8138, 0.8667] |
| old_novelty_only | 0.3295 [0.2687, 0.3875] | 0.4042 [0.3525, 0.4503] | 0.5710 [0.5288, 0.6105] | 0.7387 [0.7047, 0.7710] | 0.8970 [0.8762, 0.9144] |
| oracle_greedy_eval_only | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |

## Hygiene

| Policy | K | Likely Artifact | Spike Fail | Broad Trace Fail | Duplicate |
| --- | ---: | ---: | ---: | ---: | ---: |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0469 [0.0219, 0.0719] | 0.0000 [0.0000, 0.0000] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 10 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0437 [0.0281, 0.0609] | 0.0094 [0.0016, 0.0203] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 25 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0381 [0.0275, 0.0494] | 0.0281 [0.0156, 0.0419] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 50 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0297 [0.0228, 0.0372] | 0.0250 [0.0172, 0.0334] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 100 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0181 [0.0144, 0.0222] | 0.0189 [0.0138, 0.0247] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 5 | 0.0656 [0.0375, 0.1000] | 0.0469 [0.0250, 0.0719] | 0.1062 [0.0719, 0.1469] | 0.0000 [0.0000, 0.0000] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 10 | 0.0922 [0.0687, 0.1157] | 0.0703 [0.0500, 0.0922] | 0.1344 [0.1046, 0.1672] | 0.0094 [0.0016, 0.0219] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 25 | 0.1419 [0.1212, 0.1638] | 0.1144 [0.0969, 0.1331] | 0.1763 [0.1562, 0.1981] | 0.0231 [0.0119, 0.0369] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 50 | 0.1284 [0.1169, 0.1406] | 0.1044 [0.0931, 0.1153] | 0.1559 [0.1431, 0.1684] | 0.0234 [0.0159, 0.0316] |
| blend_kcenter_ts2vec_window_mean_std_pool_a05 | 100 | 0.0898 [0.0822, 0.0973] | 0.0698 [0.0628, 0.0769] | 0.1070 [0.0991, 0.1148] | 0.0184 [0.0136, 0.0238] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 5 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 10 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0109 [0.0031, 0.0234] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 25 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0269 [0.0150, 0.0400] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 50 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0228 [0.0147, 0.0313] |
| trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 100 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0169 [0.0120, 0.0225] |
| kcenter_greedy_quality_gated | 5 | 0.1469 [0.1094, 0.1875] | 0.1063 [0.0750, 0.1406] | 0.1719 [0.1313, 0.2156] | 0.0000 [0.0000, 0.0000] |
| kcenter_greedy_quality_gated | 10 | 0.1813 [0.1484, 0.2141] | 0.1469 [0.1203, 0.1750] | 0.2063 [0.1719, 0.2406] | 0.0000 [0.0000, 0.0000] |
| kcenter_greedy_quality_gated | 25 | 0.2037 [0.1850, 0.2238] | 0.1644 [0.1462, 0.1825] | 0.2194 [0.2000, 0.2400] | 0.0000 [0.0000, 0.0000] |
| kcenter_greedy_quality_gated | 50 | 0.1459 [0.1328, 0.1584] | 0.1187 [0.1072, 0.1319] | 0.1566 [0.1434, 0.1706] | 0.0000 [0.0000, 0.0000] |
| kcenter_greedy_quality_gated | 100 | 0.0938 [0.0864, 0.1006] | 0.0716 [0.0644, 0.0788] | 0.1013 [0.0942, 0.1087] | 0.0000 [0.0000, 0.0000] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 5 | 0.0531 [0.0281, 0.0812] | 0.0375 [0.0188, 0.0563] | 0.0938 [0.0594, 0.1313] | 0.0187 [0.0062, 0.0344] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 10 | 0.0344 [0.0219, 0.0500] | 0.0250 [0.0156, 0.0359] | 0.0969 [0.0734, 0.1219] | 0.0609 [0.0453, 0.0781] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 25 | 0.0375 [0.0275, 0.0481] | 0.0275 [0.0188, 0.0369] | 0.0981 [0.0800, 0.1163] | 0.0906 [0.0781, 0.1031] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 50 | 0.0444 [0.0359, 0.0534] | 0.0350 [0.0269, 0.0434] | 0.0859 [0.0731, 0.0988] | 0.0709 [0.0634, 0.0787] |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 100 | 0.0544 [0.0477, 0.0613] | 0.0445 [0.0377, 0.0517] | 0.0786 [0.0702, 0.0872] | 0.0498 [0.0450, 0.0548] |
| old_novelty_only | 5 | 0.2406 [0.2094, 0.2781] | 0.0219 [0.0063, 0.0406] | 0.2812 [0.2375, 0.3313] | 0.1625 [0.1219, 0.2062] |
| old_novelty_only | 10 | 0.1688 [0.1469, 0.1937] | 0.0406 [0.0250, 0.0578] | 0.2109 [0.1781, 0.2438] | 0.2109 [0.1796, 0.2438] |
| old_novelty_only | 25 | 0.1813 [0.1619, 0.2012] | 0.1119 [0.0950, 0.1300] | 0.2231 [0.2019, 0.2456] | 0.1831 [0.1588, 0.2062] |
| old_novelty_only | 50 | 0.2191 [0.2037, 0.2331] | 0.1694 [0.1547, 0.1837] | 0.2566 [0.2409, 0.2725] | 0.1391 [0.1231, 0.1547] |
| old_novelty_only | 100 | 0.2122 [0.2042, 0.2200] | 0.1697 [0.1623, 0.1773] | 0.2370 [0.2273, 0.2469] | 0.0930 [0.0834, 0.1020] |
| oracle_greedy_eval_only | 5 | 0.1219 [0.0813, 0.1656] | 0.0969 [0.0625, 0.1313] | 0.1750 [0.1281, 0.2219] | 0.0000 [0.0000, 0.0000] |
| oracle_greedy_eval_only | 10 | 0.1047 [0.0781, 0.1328] | 0.0766 [0.0546, 0.1016] | 0.1422 [0.1141, 0.1719] | 0.0078 [0.0016, 0.0156] |
| oracle_greedy_eval_only | 25 | 0.1006 [0.0837, 0.1188] | 0.0762 [0.0619, 0.0919] | 0.1238 [0.1044, 0.1444] | 0.0169 [0.0106, 0.0238] |
| oracle_greedy_eval_only | 50 | 0.1078 [0.0959, 0.1200] | 0.0841 [0.0738, 0.0947] | 0.1253 [0.1131, 0.1391] | 0.0400 [0.0313, 0.0491] |
| oracle_greedy_eval_only | 100 | 0.1887 [0.1831, 0.1950] | 0.1492 [0.1437, 0.1552] | 0.2078 [0.2003, 0.2155] | 0.0627 [0.0558, 0.0700] |

## Episode-Level Comparisons

| Comparison | K | Primary Wins | Win Rate | Relative Gain Delta |
| --- | ---: | ---: | ---: | ---: |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs blend_kcenter_ts2vec_window_mean_std_pool_a05 | 5 | 60/64 | 0.9375 | 0.0046 [0.0001, 0.0119] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs blend_kcenter_ts2vec_window_mean_std_pool_a05 | 10 | 52/64 | 0.8125 | -0.0001 [-0.0040, 0.0034] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs blend_kcenter_ts2vec_window_mean_std_pool_a05 | 25 | 36/64 | 0.5625 | -0.0005 [-0.0066, 0.0052] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs blend_kcenter_ts2vec_window_mean_std_pool_a05 | 50 | 27/64 | 0.4219 | -0.0055 [-0.0121, 0.0012] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs blend_kcenter_ts2vec_window_mean_std_pool_a05 | 100 | 25/64 | 0.3906 | -0.0140 [-0.0227, -0.0070] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 5 | 60/64 | 0.9375 | 0.0007 [-0.0030, 0.0046] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 10 | 54/64 | 0.8438 | 0.0065 [0.0004, 0.0143] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 25 | 52/64 | 0.8125 | 0.0100 [0.0025, 0.0188] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 50 | 50/64 | 0.7812 | 0.0115 [0.0045, 0.0198] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 | 100 | 45/64 | 0.7031 | 0.0126 [0.0054, 0.0212] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs kcenter_greedy_quality_gated | 5 | 34/64 | 0.5312 | 0.0243 [-0.0033, 0.0533] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs kcenter_greedy_quality_gated | 10 | 36/64 | 0.5625 | 0.0428 [0.0179, 0.0709] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs kcenter_greedy_quality_gated | 25 | 39/64 | 0.6094 | 0.0393 [0.0179, 0.0646] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs kcenter_greedy_quality_gated | 50 | 39/64 | 0.6094 | 0.0288 [0.0080, 0.0529] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs kcenter_greedy_quality_gated | 100 | 42/64 | 0.6562 | 0.0231 [0.0042, 0.0448] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs window_shape_stats_q85_stat90_abs60_clustercap2 | 5 | 55/64 | 0.8594 | 0.0477 [0.0338, 0.0633] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs window_shape_stats_q85_stat90_abs60_clustercap2 | 10 | 54/64 | 0.8438 | 0.0489 [0.0338, 0.0656] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs window_shape_stats_q85_stat90_abs60_clustercap2 | 25 | 48/64 | 0.7500 | 0.0235 [0.0099, 0.0367] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs window_shape_stats_q85_stat90_abs60_clustercap2 | 50 | 43/64 | 0.6719 | 0.0098 [-0.0021, 0.0229] |
| artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05 vs window_shape_stats_q85_stat90_abs60_clustercap2 | 100 | 30/64 | 0.4688 | -0.0081 [-0.0176, 0.0010] |
