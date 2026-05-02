# TS2Vec Fixed-Crop Revalidation

Date: 2026-05-02

## Verdict

The crop-provenance bug is fixed in code, and a bounded fixed-crop checkpoint
was trained and evaluated. The fixed checkpoint is not promoted over the
current TS2Vec checkpoint yet.

Reason: on the same 8-episode active-loop slice, the fixed checkpoint is mixed.
It improves some alpha/policy cells, but the current checkpoint still wins for
the previously promoted `blend_kcenter_ts2vec_window_mean_std_pool_a05` policy
at K=10 and K=50.

## Code Fix

The public crop helper no longer returns two copies of the same sampled crop.
The training helper now delegates to the same canonical overlapping-crop
primitive.

Relevant files:

```text
marginal_value/models/ts2vec_loss.py
marginal_value/training/train_ts2vec.py
tests/test_ts2vec_pipeline.py
```

## Fixed-Crop Training Run

Config was passed through the Modal training entrypoint.

```text
checkpoint_dir: /artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval
checkpoint: /artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval/ts2vec_best.pt
epochs: 3
max_steps_per_epoch: 150
collapse_sample_size: 128
alpha: 0.1
max_temporal_positions: 64
```

Training log:

| Epoch | Effective Rank | Mean Pairwise Cosine | Train Loss | Instance Loss | Temporal Loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.4116 | 0.8416 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 9.1496 | 0.5197 | 0.582785 | 0.276691 | 3.337637 |
| 2 | 11.3723 | 0.4955 | 0.482826 | 0.170895 | 3.290204 |
| 3 | 12.8765 | 0.4930 | 0.466812 | 0.151247 | 3.306902 |

The fixed model did not collapse, but the temporal loss remains much larger
than the instance loss. This is a sign that the representation is still heavily
driven by within-clip temporal consistency.

## Medium Active-Loop Comparison

Both checkpoints were evaluated on the same capped 8-episode slice.

Configs:

```text
configs/active_loop_eval_ts2vec_window_blend_medium.json
configs/active_loop_eval_ts2vec_fixed_window_blend_medium.json
```

Artifacts:

```text
/artifacts/active/eval/ts2vec_window_blend_medium/coverage_gain_report_full.json
/artifacts/active/eval/ts2vec_fixed_crop_window_blend_medium/coverage_gain_report_full.json
```

Balanced relative gain:

| Policy | Current K=10 | Current K=50 | Current K=100 | Fixed K=10 | Fixed K=50 | Fixed K=100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blend k-center a03 | 0.0560 | 0.1628 | 0.2315 | 0.0608 | 0.1719 | 0.2336 |
| blend k-center a05 | 0.0862 | 0.1712 | 0.2220 | 0.0784 | 0.1599 | 0.2230 |
| blend k-center a07 | 0.0581 | 0.1703 | 0.2176 | 0.0779 | 0.1433 | 0.2046 |
| window-shape cluster-cap | 0.0563 | 0.1480 | 0.2215 | 0.0581 | 0.1560 | 0.2253 |
| k-center quality-gated | 0.0568 | 0.1688 | 0.2171 | 0.0735 | 0.1428 | 0.2050 |
| old novelty only | 0.0449 | 0.1675 | 0.2218 | 0.0655 | 0.1385 | 0.2190 |

K=10 hygiene for the blend/k-center policies was clean for both checkpoints:
0.000 artifact rate, 0.000 low-quality rate, and 0.000 duplicate rate for
alpha 0.5 and alpha 0.7.

## Decision

Do not replace the current final artifact with the fixed-crop checkpoint.

The code fix should stay. The bounded checkpoint proves the fixed path runs, but
the fixed representation needs either longer training or further loss balancing
before it can be promoted scientifically.

For current challenge ranking, keep the existing checkpoint-backed budgeted
blend unless a longer fixed-crop training run beats it on active-loop eval.
