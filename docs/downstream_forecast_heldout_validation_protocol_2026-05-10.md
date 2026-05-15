# Downstream Forecast Held-Out Validation Protocol - 2026-05-10

## Decision

Run one held-out source validation for the frozen survivor policies.

This is not another 10-seed estimate. It is a three-seed validation on source groups that were not in the locked canary manifest used to choose `window_kcenter_v1`.

## Frozen Inputs

- config: `configs/downstream_forecast_task_gcp_heldout_source_validation_3seed.json`
- manifest: `artifacts/offline_active_benchmark/gcp_inputs/heldout_pretrain_urls_workers_0501_0900_c20.txt`
- held-out source groups: `worker00501` through `worker00900`
- excluded prior canary manifest: `artifacts/offline_active_benchmark/gcp_inputs/pretrain_urls_gcp_10000_g500_c20.txt`
- excluded prior canary groups: `worker00001` through `worker00500`
- seeds: `109, 127, 149`
- final budget: `K=4`

## Frozen Policies

- `quality_stratified_random_v1`
- `quality_only_v1`
- `window_kcenter_v1`
- `submitted_full_replay_v1`
- `ts2vec_kcenter_v1`

No `support_gap_window_probcover_v1`. No new selector. No tuning after results.

## Metric

Primary metric is mean held-out target forecast MSE after acquisition at `K=4`.

The downstream task remains the raw IMU autoregressive ridge forecast task: selected candidate clips are appended to support, the forecaster is retrained, and target forecast MSE is measured.

## Decision Rule

If `window_kcenter_v1` remains best on mean K=4 held-out target forecast MSE and beats `quality_stratified_random_v1`, keep the frozen window-geometry interpretation as externally validated on a source-disjoint slice.

If it does not, report the reversal plainly. Do not tune a new policy on this validation result.

## Launch Command

```bash
python3 scripts/launch_coverage_benchmark_gcp.py \
  --config configs/downstream_forecast_task_gcp_heldout_source_validation_3seed.json \
  --run-id heldout-forecast-sourceval-20260510T0000Z
```

## Non-Claims

- This is not a learned query policy.
- This is not large neural downstream training.
- This is not TS2Vec retraining.
- This is not proof that window geometry is universally optimal.
- This is not a new policy search.
