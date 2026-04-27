# Quality-Gated Old-Novelty Scientific Diagnostic

## Run

- mode: `full`
- rows: `20000`
- source groups: `5437`
- folds: `4`
- primary representations: `temporal_order, raw_shape_stats`

## Mean Coverage

| policy | K | window | temporal | raw | primary avg | min quality | max stationary | max abs | stationary >0.90 | max abs >60 | largest source frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_high_quality | 50 | 0.0083 | 0.0009 | 0.0270 | 0.0139 | 0.651 | 0.956 | 58.3 | 0.031 | 0.016 | 0.040 |
| quality_only | 50 | 0.0377 | 0.0040 | 0.0562 | 0.0301 | 1.000 | 0.908 | 20.7 | 0.025 | 0.000 | 0.050 |
| old_novelty_only | 50 | 0.0716 | 0.0155 | 0.0453 | 0.0304 | 0.524 | 0.993 | 79.1 | 0.070 | 0.010 | 0.055 |
| old_novelty_raw_shape_stats | 50 | 0.0202 | 0.0017 | 0.1664 | 0.0840 | 0.379 | 0.803 | 116.7 | 0.010 | 0.095 | 0.085 |
| quality_gated_old_novelty_raw_shape_stats_q85 | 50 | 0.0202 | 0.0018 | 0.1632 | 0.0825 | 0.874 | 0.831 | 106.3 | 0.015 | 0.070 | 0.085 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | 50 | 0.0190 | 0.0018 | 0.1632 | 0.0825 | 0.879 | 0.616 | 106.3 | 0.000 | 0.070 | 0.085 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | 50 | 0.0189 | 0.0018 | 0.1624 | 0.0821 | 0.879 | 0.616 | 50.4 | 0.000 | 0.000 | 0.085 |
| random_high_quality | 100 | 0.0123 | 0.0034 | 0.0618 | 0.0326 | 0.545 | 0.985 | 79.6 | 0.034 | 0.015 | 0.028 |
| quality_only | 100 | 0.0525 | 0.0059 | 0.0775 | 0.0417 | 1.000 | 0.972 | 23.0 | 0.035 | 0.000 | 0.037 |
| old_novelty_only | 100 | 0.0767 | 0.0173 | 0.0572 | 0.0372 | 0.499 | 0.999 | 79.1 | 0.105 | 0.005 | 0.033 |
| old_novelty_raw_shape_stats | 100 | 0.0203 | 0.0029 | 0.1893 | 0.0961 | 0.260 | 0.910 | 116.7 | 0.010 | 0.055 | 0.045 |
| quality_gated_old_novelty_raw_shape_stats_q85 | 100 | 0.0203 | 0.0030 | 0.1688 | 0.0859 | 0.865 | 0.984 | 106.3 | 0.015 | 0.043 | 0.043 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | 100 | 0.0190 | 0.0030 | 0.1688 | 0.0859 | 0.868 | 0.618 | 106.3 | 0.000 | 0.043 | 0.043 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | 100 | 0.0190 | 0.0030 | 0.1658 | 0.0844 | 0.864 | 0.618 | 52.4 | 0.000 | 0.000 | 0.045 |
| random_high_quality | 200 | 0.0224 | 0.0063 | 0.0955 | 0.0509 | 0.507 | 0.999 | 89.7 | 0.031 | 0.014 | 0.018 |
| quality_only | 200 | 0.0700 | 0.0098 | 0.1035 | 0.0566 | 0.999 | 0.972 | 29.4 | 0.024 | 0.000 | 0.022 |
| old_novelty_only | 200 | 0.0793 | 0.0199 | 0.0703 | 0.0451 | 0.436 | 0.999 | 82.0 | 0.061 | 0.003 | 0.020 |
| old_novelty_raw_shape_stats | 200 | 0.0272 | 0.0098 | 0.2020 | 0.1059 | 0.135 | 0.988 | 122.8 | 0.011 | 0.034 | 0.024 |
| quality_gated_old_novelty_raw_shape_stats_q85 | 200 | 0.0272 | 0.0099 | 0.1824 | 0.0962 | 0.856 | 0.984 | 106.8 | 0.010 | 0.025 | 0.022 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | 200 | 0.0259 | 0.0099 | 0.1824 | 0.0962 | 0.860 | 0.734 | 106.8 | 0.000 | 0.025 | 0.022 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | 200 | 0.0259 | 0.0099 | 0.1788 | 0.0944 | 0.860 | 0.770 | 53.6 | 0.000 | 0.000 | 0.022 |
| random_high_quality | 400 | 0.0471 | 0.0137 | 0.1513 | 0.0825 | 0.491 | 0.999 | 104.1 | 0.030 | 0.013 | 0.011 |
| quality_only | 400 | 0.0710 | 0.0112 | 0.1341 | 0.0726 | 0.997 | 0.978 | 37.4 | 0.021 | 0.000 | 0.012 |
| old_novelty_only | 400 | 0.0795 | 0.0217 | 0.1223 | 0.0720 | 0.214 | 0.999 | 101.3 | 0.033 | 0.003 | 0.011 |
| old_novelty_raw_shape_stats | 400 | 0.0446 | 0.0178 | 0.2133 | 0.1155 | 0.135 | 0.988 | 122.8 | 0.006 | 0.021 | 0.012 |
| quality_gated_old_novelty_raw_shape_stats_q85 | 400 | 0.0446 | 0.0183 | 0.1913 | 0.1048 | 0.856 | 0.984 | 106.8 | 0.006 | 0.014 | 0.011 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | 400 | 0.0433 | 0.0183 | 0.1913 | 0.1048 | 0.859 | 0.770 | 106.8 | 0.000 | 0.014 | 0.011 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | 400 | 0.0433 | 0.0183 | 0.1880 | 0.1031 | 0.859 | 0.770 | 53.6 | 0.000 | 0.000 | 0.011 |

## Paired Primary Deltas

| challenger | baseline | K | mean delta | fold wins | fold ties | folds |
|---|---|---:|---:|---:|---:|---:|
| quality_gated_old_novelty_raw_shape_stats_q85 | random_high_quality | 50 | 0.0685 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | quality_only | 50 | 0.0524 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | old_novelty_only | 50 | 0.0520 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | random_high_quality | 50 | 0.0685 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | quality_only | 50 | 0.0524 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | old_novelty_only | 50 | 0.0520 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | random_high_quality | 50 | 0.0682 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | quality_only | 50 | 0.0520 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | old_novelty_only | 50 | 0.0517 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | random_high_quality | 100 | 0.0533 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | quality_only | 100 | 0.0442 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | old_novelty_only | 100 | 0.0486 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | random_high_quality | 100 | 0.0533 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | quality_only | 100 | 0.0442 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | old_novelty_only | 100 | 0.0486 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | random_high_quality | 100 | 0.0518 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | quality_only | 100 | 0.0427 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | old_novelty_only | 100 | 0.0472 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | random_high_quality | 200 | 0.0453 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | quality_only | 200 | 0.0395 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | old_novelty_only | 200 | 0.0511 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | random_high_quality | 200 | 0.0453 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | quality_only | 200 | 0.0395 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | old_novelty_only | 200 | 0.0511 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | random_high_quality | 200 | 0.0434 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | quality_only | 200 | 0.0377 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | old_novelty_only | 200 | 0.0493 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | random_high_quality | 400 | 0.0223 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | quality_only | 400 | 0.0322 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85 | old_novelty_only | 400 | 0.0328 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | random_high_quality | 400 | 0.0223 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | quality_only | 400 | 0.0322 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90 | old_novelty_only | 400 | 0.0328 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | random_high_quality | 400 | 0.0207 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | quality_only | 400 | 0.0305 | 4 | 0 | 4 |
| quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 | old_novelty_only | 400 | 0.0311 | 4 | 0 | 4 |

## Decision Notes

- Treat a method as stronger only when it wins the primary aggregate and wins in most folds.
- Use `window_mean_std_pool` as a sanity check, not as the sole decision metric, because it is also the current ranking space.
- If a method improves minimum quality but not coverage, it is a safety control rather than a marginal-value improvement.

## Current Read

- `quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60` is the current best science/media compromise.
- It keeps nearly all of the q85/stat90 marginal-coverage gain while eliminating measured top-K selections with `stationary_fraction > 0.90` and `max_abs_value > 60`.
- At K100, primary average is `0.0844`, versus `0.0417` for `quality_only` and `0.0859` for q85/stat90 without the spike gate.
- At K200, primary average is `0.0944`, versus `0.0566` for `quality_only` and `0.0962` for q85/stat90 without the spike gate.
- The win is still mostly raw-shape coverage. Temporal-order coverage is weak at K50/K100, so this is not yet evidence of broad temporal behavior discovery.
- Full visual audit on the new split shows top100 mean quality `0.9762`, min quality `0.8565`, max stationary `0.8072`, max abs `56.49`, and no physical-validity failures under the stat90/abs60 gates.
- Parent concentration is still a watch item: the largest parent cluster is `0.29` of top200 in the full visual audit. The next small scientific/media test should be one source/parent cap on top of q85_stat90_abs60.

## Follow-up: Cluster-Cap Media Diagnostic

- `q85_stat90_abs60_clustercap2` was added as a small top-K diversity diagnostic, not as a new scientific winner.
- Full new-split ranking/audit runs:
  - ranking: `https://modal.com/apps/vishkrish200/main/ap-YTqKlnk7sFhrL5jUvBFsrL`
  - visual audit: `https://modal.com/apps/vishkrish200/main/ap-xCPC8gheFyvLFafjUBqSNH`
- At K100, it improves parent concentration from `0.07` to `0.02` and unique parent clusters from `91` to `96`, while keeping min quality `0.8565`, max stationary `0.7831`, and max abs `56.49`.
- At K200, it improves parent concentration only from `0.29` to `0.235`; this is better but not solved.
- The reason is structural: after q85/stat90/abs60 there are only `146` cap-selected passing rows before fallback across `135` passing clusters, so a filled top200 must admit fallback rows from the large cluster unless we change the quality/validity tradeoff or stop filling capped fallbacks.
- Full marginal-coverage run for the capped raw-shape policy: `https://modal.com/apps/vishkrish200/main/ap-mBiDhqGfyRSwaVjMRCn5pm`
- `q85_stat90_abs60_clustercap2` is slightly positive versus uncapped at K100/K200/K400, but the effect is tiny:
  - K100 primary average: `0.0856` capped versus `0.0844` uncapped.
  - K200 primary average: `0.0946` capped versus `0.0944` uncapped.
  - K400 primary average: `0.1032` capped versus `0.1031` uncapped.
  - K50 primary average: `0.0808` capped versus `0.0821` uncapped.
- Paired deltas versus uncapped: K100 `+0.0012` with 4/4 fold wins; K200 `+0.0002` with 2 wins and 1 tie; K400 `+0.0001` with 1 win and 2 ties; K50 `-0.0013`.
- Treat `q85_stat90_abs60_clustercap2` as the current science/media candidate for top100-style use because it improves media diversity and does not materially hurt marginal coverage.
- Keep `q85_stat90_abs60` as the simpler uncapped control.
- Source-cap implementation was hardened so missing explicit cap keys cannot silently become per-sample no-ops; `new_cluster_parent_id` falls back to `new_cluster_id` when no parent annotation exists.
