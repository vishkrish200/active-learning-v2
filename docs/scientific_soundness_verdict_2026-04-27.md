# Scientific Soundness Verdict

## Summary

- candidate: `quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60_clustercap2`
- overall status: `conditional`
- primary representations: `temporal_order, raw_shape_stats`
- independent representations: `temporal_order`

## Claim Status

| claim | status |
|---|---|
| artifact_safe_raw_shape_supported | pass |
| broad_behavior_discovery_supported | fail |
| hidden_test_ready | needs_blind_external_test |

## Gates

| gate | status | key result |
|---|---|---|
| primary_vs_simple_controls | pass | weakest mean delta +0.0379 vs quality_only at K=200 (4/4 fold wins) |
| physical_validity | pass | max stationary>0.90=0.000, max abs>60=0.000, min quality=0.850 |
| uncapped_regression | pass | weakest mean delta +0.0002 vs quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60 at K=200 (2/4 fold wins) |
| independent_temporal_vs_controls | fail | weakest mean delta -0.0143 vs old_novelty_only at K=100 (0/4 fold wins) |

## Interpretation

The candidate is defensible as an artifact-safe raw-shape/media selector, but it is not yet scientifically supported as broad behavior discovery because independent temporal coverage does not pass.
