# Ranker Hygiene Fix Results

Date: 2026-04-30

## Summary

The hygiene-gated label experiment was run as an ablation, not as the promoted
selector path.

Result: hard-gating labels helped the pure ridge ranker, but did not make pure
score sorting hygienic enough to promote. The current best deployable selector
remains:

```text
blend_kcenter_ts2vec_window_mean_std_pool_a05
```

## Implementation Notes

- Added `gated_balanced_gain` and `gated_balanced_relative_gain` labels.
- Hard-zeroed labels for clips failing:
  - `quality_score < 0.85`
  - `stationary_fraction > 0.90`
  - `max_abs_value > 60.0`
- Duplicate label-gating was tested and rejected for this corpus. Using
  candidate nearest similarity with threshold `0.985` zeroed 10,808 / 11,264
  rows, or 96.5% of the label table. That is too aggressive for training.
- Duplicate control is left to the selector layer, where k-center already
  controls it reliably.
- Added TS2Vec novelty features to the ranker table when TS2Vec embeddings are
  available.

Final label gate calibration:

```text
labels: 11,264
zeroed labels: 1,040 (9.23%)
quality failures: 1,030
stationary failures: 75
max-abs failures: 1
duplicate label gate: disabled
```

## Modal Runs

Gated label generation:

```text
.venv/bin/modal run --detach modal_active_label_gain.py \
  --config-path configs/active_label_gain_ranker_hygiene_scale.json \
  --run-full \
  --skip-smoke
```

Artifacts:

```text
/artifacts/active/labels/ranker_hygiene_fix/active_label_gain_full.csv
/artifacts/active/labels/ranker_hygiene_fix/active_label_gain_report_full.json
```

Config A, raw balanced-gain target:

```text
.venv/bin/modal run --detach modal_active_ranker_train.py \
  --config-path configs/active_ranker_train_hygiene_ablation_balanced_gain.json \
  --run-full \
  --skip-smoke
```

Config B, hygiene-gated target:

```text
.venv/bin/modal run --detach modal_active_ranker_train.py \
  --config-path configs/active_ranker_train_hygiene_ablation_gated_gain.json \
  --run-full \
  --skip-smoke
```

Both ranker runs used:

```text
episodes: 64
train / validation / test: 48 / 8 / 8
candidate rows: 11,264
feature count: 60
embedding cache: /artifacts/active/embedding_cache/ts2vec_candidate_scale
```

## Held-Out Test Coverage

Balanced relative gain:

| Policy | Target | K=10 | K=50 |
| --- | --- | ---: | ---: |
| learned ridge | balanced_gain | 0.1616 | 0.2515 |
| learned ridge | gated_balanced_gain | 0.1688 | 0.2533 |
| learned + quality-gated k-center | balanced_gain | 0.1269 | 0.2141 |
| learned + quality-gated k-center | gated_balanced_gain | 0.1269 | 0.2142 |
| blend k-center TS2Vec/window a05 | n/a | 0.2149 | 0.3042 |
| k-center quality-gated | n/a | 0.1281 | 0.2167 |
| window-shape cluster-cap | n/a | 0.1398 | 0.2872 |
| oracle greedy eval-only | n/a | 0.3212 | 0.3805 |

## Held-Out Test Hygiene

At K=10:

| Policy | Target | Artifact | Low-quality | Duplicate |
| --- | --- | ---: | ---: | ---: |
| learned ridge | balanced_gain | 0.175 | 0.175 | 0.150 |
| learned ridge | gated_balanced_gain | 0.088 | 0.088 | 0.162 |
| learned + quality-gated k-center | balanced_gain | 0.000 | 0.000 | 0.000 |
| learned + quality-gated k-center | gated_balanced_gain | 0.000 | 0.000 | 0.000 |
| blend k-center TS2Vec/window a05 | n/a | 0.000 | 0.000 | 0.012 |
| k-center quality-gated | n/a | 0.000 | 0.000 | 0.000 |
| window-shape cluster-cap | n/a | 0.000 | 0.000 | 0.037 |

At K=50:

| Policy | Target | Artifact | Low-quality | Duplicate |
| --- | --- | ---: | ---: | ---: |
| learned ridge | balanced_gain | 0.115 | 0.115 | 0.160 |
| learned ridge | gated_balanced_gain | 0.092 | 0.092 | 0.160 |
| learned + quality-gated k-center | balanced_gain | 0.000 | 0.000 | 0.000 |
| learned + quality-gated k-center | gated_balanced_gain | 0.000 | 0.000 | 0.000 |
| blend k-center TS2Vec/window a05 | n/a | 0.000 | 0.000 | 0.030 |
| k-center quality-gated | n/a | 0.000 | 0.000 | 0.000 |
| window-shape cluster-cap | n/a | 0.000 | 0.000 | 0.055 |

## Conclusion

The label hygiene fix is useful diagnostically:

- It improved pure ridge balanced gain at K=10 from `0.1616` to `0.1688`.
- It reduced pure ridge artifact rate at K=10 from `0.175` to `0.088`.

But it did not meet the promotion gate:

```text
artifact rate target: < 0.05
observed pure gated ridge at K=10: 0.088

duplicate rate target: < 0.10
observed pure gated ridge at K=10: 0.162
```

The learned score still needs a diversity layer to be hygienic, and once that
layer is added, it does not beat the simple geometric blend. The promoted
full-scale selector should therefore remain:

```text
blend_kcenter_ts2vec_window_mean_std_pool_a05
```

The learned ranker should stay as a diagnostic or future feature source, not as
the final acquisition policy for this run.
