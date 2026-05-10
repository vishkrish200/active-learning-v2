# Held-Out Downstream Forecast Validation Readout

Date: 2026-05-10
Run ID: `heldout-forecast-sourceval-20260510T1107Z`
Config: `configs/downstream_forecast_task_gcp_heldout_source_validation_3seed.json`

## What This Was

This was the pre-registered source-disjoint validation read for the frozen survivor policies. It was not another 10-seed repeat, not a ProbCover rescue, and not a new selector search.

- Held-out manifest: `artifacts/offline_active_benchmark/gcp_inputs/heldout_pretrain_urls_workers_0501_0900_c20.txt`
- Held-out rows: 8,000 clips from 400 source groups, `worker00501` through `worker00900`
- Locked prior canary exclusion: `worker00001` through `worker00500`
- Source-group overlap with the locked canary manifest: 0
- Seeds: `109`, `127`, `149`
- Episodes: 12 total validation units
- Downstream task: raw IMU autoregressive ridge forecast, retrained after each acquisition
- No GPU, no TS2Vec retraining, no large neural downstream training

## Primary Result

The held-out validation reversed the frozen window-champion interpretation. At final budget `K=4`, `window_kcenter_v1` still beat the random baseline, but it did not beat the fixed survivor comparators.

| rank | policy | mean after MSE | mean relative MSE reduction | best/tie wins |
|---:|---|---:|---:|---:|
| 1 | `quality_only_v1` | 0.297020109 | 0.002004935 | 3 |
| 2 | `ts2vec_kcenter_v1` | 0.297188491 | 0.001427044 | 6 |
| 3 | `submitted_full_replay_v1` | 0.297425503 | 0.000888791 | 0 |
| 4 | `window_kcenter_v1` | 0.297481472 | 0.000634493 | 1 |
| 5 | `quality_stratified_random_v1` | 0.298367534 | -0.000512730 | 2 |

Key pairwise checks:

- `window_kcenter_v1` beat `quality_stratified_random_v1` by mean after-MSE advantage `0.000886062` and won 9 of 12 paired units.
- `submitted_full_replay_v1` beat `window_kcenter_v1` by mean after-MSE advantage `0.000055969` and won 6 of 12 paired units, with 2 ties.
- `ts2vec_kcenter_v1` beat `window_kcenter_v1` by mean after-MSE advantage `0.000292981` and won 8 of 12 paired units.

## Hygiene

The run passed the basic scientific hygiene checks.

- Leakage checks all OK: true
- Selected invalid rate max: 0.0
- Out-of-pool selections: 0
- Target leaks: 0
- Duplicate selected clips: 0
- VM result state: success
- VM cleanup: deleted after artifact copy

## Interpretation

This should be treated as mixed-to-weak evidence for the selector story, not as a new reason to policy-shop.

`window_kcenter_v1` remains better than random on this held-out validation, so the geometry signal is not nonsense. But the stronger claim, that window geometry is the current robust downstream champion, did not survive this source-disjoint read.

`quality_only_v1` winning the mean is a warning sign about this particular raw forecast canary, not a reason to promote quality-only as the scientific acquisition strategy. It collapses hard onto repeated high-quality source regions: duplicate-source batch rate `1.0`, largest source fraction `0.625`.

`ts2vec_kcenter_v1` beating window here also means TS2Vec should not be dismissed. The safer read is that TS2Vec is unstable as a lead story but remains a competitive feature source / ablation under this downstream task.

## Decision

Do not tune a new selector on this validation target. Do not revive `support_gap_window_probcover_v1` as a rescue. Do not start big neural downstream training from this result alone.

The honest next scientific move is either:

1. write this up as mixed evidence / negative validation for the current downstream canary, or
2. define a genuinely separate pre-registered downstream objective before spending on more training.

For the current canary, stop selector iteration here.
