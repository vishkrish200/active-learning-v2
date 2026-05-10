# Locked Survivor Confirmation Summary

Run id: `survivors-locked-20260509T1515Z`

Downloaded artifacts:
`artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/gcp_downloads/survivors-locked-20260509T1515Z`

## Scope

- Survivor-only confirmation, not policy search.
- Seeds: `17`, `23`, `37`, `41`, `53`, `67`, `79`, `83`, `97`, `101`
- Episodes per seed: `4`
- Final budget: `K=4`
- Policies:
  - `quality_stratified_random_v1`
  - `quality_only_v1`
  - `window_kcenter_v1`
  - `submitted_full_replay_v1`
  - `ts2vec_kcenter_v1`
- Excluded by design: `support_gap_window_probcover_v1`
- Downstream task: raw IMU autoregressive ridge forecast
- No GPU, no TS2Vec retraining, no large neural downstream training

## Hygiene

All ten seeds completed with clean proof summaries:

- `leakage_ok`: `true`
- `selected_invalid_rate_max`: `0.0`
- `selected_out_of_pool_count_total`: `0`
- `selected_target_leak_count_total`: `0`
- `selected_duplicate_clip_count_total`: `0`

## Primary Downstream Result

Lower `mean_after_mse` is better. This is the decision endpoint.

| rank | policy | rows | mean after MSE | mean relative MSE reduction |
|---:|---|---:|---:|---:|
| 1 | `window_kcenter_v1` | 40 | 0.118593614 | 0.086395388 |
| 2 | `ts2vec_kcenter_v1` | 40 | 0.121089793 | 0.078833143 |
| 3 | `submitted_full_replay_v1` | 40 | 0.121103872 | 0.078781829 |
| 4 | `quality_stratified_random_v1` | 40 | 0.122818619 | 0.061063806 |
| 5 | `quality_only_v1` | 40 | 0.125936982 | 0.049101767 |

Pairwise final-K checks against `window_kcenter_v1`:

| comparison | mean after-MSE delta vs window | window lower/tied units |
|---|---:|---:|
| `quality_stratified_random_v1` - `window_kcenter_v1` | 0.004225004 | 28 / 40 |
| `submitted_full_replay_v1` - `window_kcenter_v1` | 0.002510258 | 24 / 40 |
| `ts2vec_kcenter_v1` - `window_kcenter_v1` | 0.002496179 | 16 / 40 |

Note: `ts2vec_kcenter_v1` wins more individual units than `window_kcenter_v1` (`24 / 40`), but `window_kcenter_v1` has better mean MSE because its favorable units have larger magnitude. The decision rule is mean held-out downstream MSE at `K=4`, so `window_kcenter_v1` remains the current downstream-confirmed selector.

## Proxy Coverage Diagnostic

The same run's proxy coverage report disagrees with the downstream forecast endpoint. Higher gain is better:

| rank | policy | mean final proxy gain | CI95 low | CI95 high |
|---:|---|---:|---:|---:|
| 1 | `ts2vec_kcenter_v1` | 0.1777 | 0.1360 | 0.2251 |
| 2 | `submitted_full_replay_v1` | 0.1467 | 0.1122 | 0.1868 |
| 3 | `window_kcenter_v1` | 0.1353 | 0.0987 | 0.1836 |
| 4 | `quality_stratified_random_v1` | 0.1096 | 0.0648 | 0.1581 |
| 5 | `quality_only_v1` | 0.0839 | 0.0524 | 0.1187 |

Diagnostic read: `ts2vec_kcenter_v1` wins proxy coverage but loses the real downstream forecast endpoint by mean MSE. Treat proxy coverage as a screening/diagnostic signal, not as the selector objective.

## Decision

`window_kcenter_v1` is the current preferred selector for the non-neural downstream forecast canary.

`ts2vec_kcenter_v1` remains useful and non-nonsense: it is the strongest proxy-coverage method and remains close on downstream MSE. But it is not the current downstream-confirmed active-learning selector.

`submitted_full_replay_v1` remains defensible as the submitted-system comparator.

`support_gap_window_probcover_v1` stays killed. It was excluded from this run because it failed the pre-registered downstream forecast rule, and no rescue analysis is used for selector choice.

## Stop Rule

Do not add more selector policies against this target. Do not tune ProbCover. Do not run big neural downstream training yet. The clean next artifact is this locked confirmation report plus the decision cards:

- `downstream_forecast_survivor_decision_card.md`
- `downstream_forecast_survivor_decision_card.json`
