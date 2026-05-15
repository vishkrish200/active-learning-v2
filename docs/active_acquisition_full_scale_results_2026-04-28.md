# Active Acquisition Full-Scale Results

Date: 2026-04-28

This note records the first full 64-episode active-acquisition evaluation and
label-generation run after the full pretrain registry became available.

## Embedding Precompute

Command:

```text
.venv/bin/python -m modal run modal_active_embedding_precompute.py \
  --config-path configs/active_embedding_precompute_scale_pretrain.json \
  --run-full \
  --skip-smoke
```

Modal run:

```text
https://modal.com/apps/vishkrish200/main/ap-MBERvgPWlcHVuWrCIP4EFD
```

Result:

```text
episodes: 64
unique episode clips: 26,725
shards: 53
workers: 16
manifest: /artifacts/active/embedding_cache/scale_pretrain/embeddings_bb9f75ba46165a9322d7f7a6.shards.json
report: /artifacts/active/embedding_cache/scale_pretrain/active_embedding_precompute_report_full.json
```

The eval and label jobs both loaded this cache with `status = shard_hit`.

## Registry Coverage

```text
old manifest URLs: 200,000
old cached URLs: 200,000
old registry clips: 200,000
old workers/source groups: 10,000
new manifest URLs: 2,000
new cached URLs: 2,000
new registry clips: 2,000
new workers: 2,000
skipped uncached/raw/features: 0
```

## Episode Diagnostics

```text
episodes: 64
support clips per episode: 256
candidates per episode: 176
hidden target clips per episode: 32
heldout groups per episode: 16
known groups per episode: 64
support-target same-group violations: 0
```

Candidate role totals:

```text
heldout_novel: 4096
known_like: 4096
near_duplicate: 2048
low_quality: 1024
```

The candidate pool is intentionally mixed. `candidate_support_same_group_rate`
is nonzero by design because known-like distractors are included.

## Full Active-Loop Eval

Command:

```text
.venv/bin/python -m modal run modal_active_loop_eval.py \
  --config-path configs/active_loop_eval_scale_pretrain.json \
  --run-full \
  --skip-smoke
```

Modal run:

```text
https://modal.com/apps/vishkrish200/main/ap-uJWp7Pajpo4070c8xGujuG
```

Artifacts:

```text
/artifacts/active/eval/scale_pretrain/coverage_gain_report_full.json
/artifacts/active/eval/scale_pretrain/coverage_gain_by_episode_full.csv
/artifacts/active/eval/scale_pretrain/topk_selection_audit_full.csv
```

Balanced relative gain and oracle fraction:

| Policy | K | Balanced Relative Gain | Oracle Fraction |
| --- | ---: | ---: | ---: |
| oracle_greedy_eval_only | 5 | 0.2451 | 1.000 |
| kcenter_greedy_quality_gated | 5 | 0.0841 | 0.250 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 5 | 0.0737 | 0.212 |
| old_novelty_only | 5 | 0.0695 | 0.195 |
| random_valid | 5 | 0.0206 | 0.081 |
| quality_only | 5 | 0.0144 | 0.057 |
| oracle_greedy_eval_only | 10 | 0.2827 | 1.000 |
| kcenter_greedy_quality_gated | 10 | 0.1185 | 0.305 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 10 | 0.1138 | 0.319 |
| old_novelty_only | 10 | 0.1007 | 0.264 |
| random_valid | 10 | 0.0392 | 0.131 |
| quality_only | 10 | 0.0344 | 0.133 |
| oracle_greedy_eval_only | 25 | 0.3197 | 1.000 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 25 | 0.1844 | 0.480 |
| kcenter_greedy_quality_gated | 25 | 0.1819 | 0.465 |
| old_novelty_only | 25 | 0.1699 | 0.422 |
| random_valid | 25 | 0.0850 | 0.266 |
| quality_only | 25 | 0.0782 | 0.237 |
| oracle_greedy_eval_only | 50 | 0.3276 | 1.000 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 50 | 0.2347 | 0.635 |
| kcenter_greedy_quality_gated | 50 | 0.2210 | 0.581 |
| old_novelty_only | 50 | 0.2161 | 0.555 |
| random_valid | 50 | 0.1543 | 0.461 |
| quality_only | 50 | 0.1372 | 0.440 |
| oracle_greedy_eval_only | 100 | 0.3276 | 1.000 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 100 | 0.2883 | 0.845 |
| kcenter_greedy_quality_gated | 100 | 0.2870 | 0.826 |
| old_novelty_only | 100 | 0.2838 | 0.815 |
| random_valid | 100 | 0.2451 | 0.744 |
| quality_only | 100 | 0.2244 | 0.716 |

Interpretation:

- The episodes are not degenerate: oracle is clearly above deployable policies
  at K = 5, 10, 25, and 50.
- Deployable geometric policies beat random and quality-only controls at small
  and medium K.
- K = 100 is close to saturation for 176-candidate episodes; use it as a
  secondary metric, not the main training decision.
- The frozen window-shape baseline and k-center are close. A learned policy
  must beat both, or the honest result is a simple/hybrid geometric policy.

## Selection Diagnostics

At K = 10:

| Policy | Artifact Rate | Low-Quality Rate | Duplicate Rate | Mean Unique Groups |
| --- | ---: | ---: | ---: | ---: |
| kcenter_greedy_quality_gated | 0.000 | 0.000 | 0.041 | 9.09 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.000 | 0.000 | 0.073 | 9.16 |
| old_novelty_only | 0.100 | 0.100 | 0.070 | 9.12 |
| random_valid | 0.000 | 0.000 | 0.756 | 9.61 |
| quality_only | 0.000 | 0.000 | 0.764 | 9.45 |
| oracle_greedy_eval_only | 0.038 | 0.038 | 0.670 | 9.03 |

The deployable diversity policies control early duplicate rate much better than
random, quality-only, and oracle. Oracle optimizes hidden-target coverage only;
it is an upper bound, not a deployable diversity policy.

## Full Label Generation

Command:

```text
.venv/bin/python -m modal run modal_active_label_gain.py \
  --config-path configs/active_label_gain_scale_pretrain.json \
  --run-full \
  --skip-smoke
```

Modal run:

```text
https://modal.com/apps/vishkrish200/main/ap-CWOd5IgDn3SG7y6UNNljxy
```

Artifacts:

```text
/artifacts/active/labels/scale_pretrain/active_label_gain_report_full.json
/artifacts/active/labels/scale_pretrain/active_label_gain_full.jsonl
/artifacts/active/labels/scale_pretrain/active_label_gain_full.csv
```

Label summary:

```text
labels: 11,264
positive balanced-gain labels: 5,154
positive rate: 45.8%
mean balanced_gain: 0.0000455
max balanced_gain: 0.0160545
mean balanced_relative_gain: 0.00395
max balanced_relative_gain: 0.74679
mean gain_after_greedy_prefix: 0.0000197
```

Role-wise solo balanced gain:

| Role | Count | Positive Rate | Mean Balanced Gain | Mean Relative Gain |
| --- | ---: | ---: | ---: | ---: |
| heldout_novel | 4096 | 0.626 | 0.0000920 | 0.00749 |
| known_like | 4096 | 0.355 | 0.0000156 | 0.00172 |
| near_duplicate | 2048 | 0.410 | 0.0000187 | 0.00224 |
| low_quality | 1024 | 0.289 | 0.0000329 | 0.00212 |

Interpretation:

- Labels are not all zero and not trivially dense.
- Heldout-novel candidates have higher average gain, as expected.
- Low-quality candidates can still have nonzero simulated gain because role
  assignment can overlap with useful source novelty. This reinforces that
  candidate role must stay diagnostic-only; deployable quality features should
  be modeled explicitly instead of using role labels.

## Next Gate

The next implementation step is not final submission ranking. It is:

```text
episode-level train/validation/test split
deployable feature matrix extraction
simple supervised ranker on balanced_gain
active-loop evaluation of model-ranked top-K
```

Promotion requires beating both:

```text
kcenter_greedy_quality_gated
window_shape_stats_q85_stat90_abs60_clustercap2
```

on held-out active-loop coverage gain without worsening artifact or duplicate
rates.

## Supervised Ranker Gate Result

Command:

```text
.venv/bin/python -m modal run modal_active_ranker_train.py \
  --config-path configs/active_ranker_train_scale_pretrain.json \
  --run-full \
  --skip-smoke
```

Modal runs:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-vhDZU2Ih5t6rKtFYht8IHB
full:  https://modal.com/apps/vishkrish200/main/ap-X6FLOTvOPoPs04t9flSHMR
```

Artifacts:

```text
/artifacts/active/ranker/scale_pretrain/active_ranker_report_full.json
/artifacts/active/ranker/scale_pretrain/active_ranker_model_full.json
/artifacts/active/ranker/scale_pretrain/active_ranker_scores_full.csv
/artifacts/active/ranker/scale_pretrain/active_ranker_coverage_by_episode_full.csv
/artifacts/active/ranker/scale_pretrain/active_ranker_topk_selection_audit_full.csv
```

Run summary:

```text
episodes: 64
candidate rows: 11,264
feature count: 48
target: balanced_gain
embedding cache: shard_hit, 26,725 clips
split: 48 train / 8 validation / 8 test episodes
forbidden feature count: 0
```

Held-out test balanced relative gain:

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle_greedy_eval_only | 0.2810 | 0.3127 | 0.3433 | 0.3500 | 0.3501 |
| learned_ridge_balanced_gain | 0.1368 | 0.1705 | 0.2129 | 0.2421 | 0.2756 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.1475 | 0.1615 | 0.2328 | 0.2731 | 0.3069 |
| kcenter_greedy_quality_gated | 0.1512 | 0.1753 | 0.2243 | 0.2605 | 0.3043 |
| old_novelty_only | 0.1313 | 0.1521 | 0.2178 | 0.2638 | 0.3083 |
| random_valid | 0.0154 | 0.0551 | 0.0968 | 0.1892 | 0.2897 |
| quality_only | 0.0246 | 0.0594 | 0.0830 | 0.1186 | 0.2222 |

Held-out test oracle fraction:

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_ridge_balanced_gain | 0.379 | 0.491 | 0.556 | 0.669 | 0.771 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.342 | 0.391 | 0.568 | 0.700 | 0.828 |
| kcenter_greedy_quality_gated | 0.396 | 0.436 | 0.587 | 0.647 | 0.808 |
| old_novelty_only | 0.281 | 0.352 | 0.569 | 0.667 | 0.811 |

Held-out test selection hygiene at K = 10:

| Policy | Artifact Rate | Low-Quality Rate | Duplicate Rate | Unique Source Groups | Unique New Clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_ridge_balanced_gain | 0.138 | 0.138 | 0.637 | 6.875 | 3.625 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.000 | 0.000 | 0.037 | 9.375 | 9.625 |
| kcenter_greedy_quality_gated | 0.000 | 0.000 | 0.037 | 9.000 | 9.625 |
| old_novelty_only | 0.100 | 0.100 | 0.062 | 9.000 | 9.375 |

Interpretation:

- The first learned ridge ranker is useful as a diagnostic, but it is not
  promotable as the acquisition policy.
- It learns real signal on train/validation and captures part of the oracle
  gap, but it does not beat the strongest deployable baselines on held-out
  test episodes.
- Its selection hygiene is worse than the k-center and frozen window-shape
  controls: artifact/low-quality and duplicate rates are too high at early K.
- This result argues for a quality-gated diversity policy, or a hybrid learned
  score plus k-center/diversity reranking, rather than a pure score sort.

Next scientific move:

```text
learned score -> hard quality gate -> candidate pool -> diversity/k-center rerank
```

Evaluate that hybrid in active-loop mode against the same held-out test
episodes before any final challenge ranking.

## Hybrid Quality-Gated K-Center Result

Command:

```text
.venv/bin/python -m modal run modal_active_ranker_train.py \
  --config-path configs/active_ranker_train_scale_pretrain.json \
  --run-full \
  --skip-smoke
```

Modal runs:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-jhoud1gWtNKbJFeEpVjTfj
full:  https://modal.com/apps/vishkrish200/main/ap-yjkvFLv1wSPke0KHq6lUBg
```

Hybrid policy:

```text
learned_ridge_quality_gated_kcenter
quality_threshold: 0.85
max_stationary_fraction: 0.90
max_abs_value: 60.0
pool_multiplier: 1.5
```

Run summary:

```text
episodes: 64
candidate rows: 11,264
feature count: 48
embedding cache: shard_hit, 26,725 clips
split: 48 train / 8 validation / 8 test episodes
forbidden feature count: 0
```

Held-out test balanced relative gain:

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle_greedy_eval_only | 0.2810 | 0.3127 | 0.3433 | 0.3500 | 0.3501 |
| learned_ridge_quality_gated_kcenter | 0.1258 | 0.1483 | 0.1856 | 0.2212 | 0.2656 |
| learned_ridge_balanced_gain | 0.1368 | 0.1705 | 0.2129 | 0.2421 | 0.2756 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.1475 | 0.1615 | 0.2328 | 0.2731 | 0.3069 |
| kcenter_greedy_quality_gated | 0.1512 | 0.1753 | 0.2243 | 0.2605 | 0.3043 |
| old_novelty_only | 0.1313 | 0.1521 | 0.2178 | 0.2638 | 0.3083 |

Held-out test selection hygiene at K = 10:

| Policy | Artifact Rate | Low-Quality Rate | Duplicate Rate | Unique Source Groups | Unique New Clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_ridge_quality_gated_kcenter | 0.000 | 0.000 | 0.087 | 9.000 | 9.125 |
| learned_ridge_balanced_gain | 0.138 | 0.138 | 0.637 | 6.875 | 3.625 |
| window_shape_stats_q85_stat90_abs60_clustercap2 | 0.000 | 0.000 | 0.037 | 9.375 | 9.625 |
| kcenter_greedy_quality_gated | 0.000 | 0.000 | 0.037 | 9.000 | 9.625 |
| old_novelty_only | 0.100 | 0.100 | 0.062 | 9.000 | 9.375 |

Interpretation:

- The hybrid fixed the learned-only model's hygiene problem: early top-K
  artifact and low-quality rates dropped to zero, and duplicate rate dropped
  sharply.
- The hybrid is still not promotable. Its held-out coverage gain is below the
  deployable k-center and frozen window-shape controls across the main K
  values.
- This suggests the first learned score is not yet useful enough as a candidate
  pool filter. Restricting k-center to the learned-high pool removes useful
  geometric coverage before it improves target gain.
- The honest current policy remains a simple deployable geometric control:
  quality-gated k-center or the frozen window-shape cluster-cap baseline.
