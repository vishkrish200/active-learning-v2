# Selected Motion Regime Audit

## Scope

- run input: `artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/gcp_downloads/survivors-locked-20260509T1515Z/results_survivors-locked-20260509T1515Z.tgz`
- seeds: `10`
- budget: `K=4`
- selected unique clips: `229`
- loaded raw clips: `229`
- max samples per clip: `900`
- focal policy: `window_kcenter_v1`

## Read

- This audit joins selected clip IDs back to raw IMU motion summaries; it does not change policy rankings.
- Regime labels are relative to the selected-clip pool, so they explain this run rather than defining universal activity classes.
- Motion energy is centered acceleration energy plus gyro energy; raw gravity magnitude is reported separately through acceleration percentiles.
- `window_kcenter_v1` selected clips have mean gyro p95 `0.868211` and mean motion energy `8.654065`.
- `window_kcenter_v1`-only clips vs `ts2vec_kcenter_v1`-only clips: motion energy `10.904860` vs `2.521718`, gyro p95 `1.017625` vs `0.474142`.
- `window_kcenter_v1`-only clips vs `submitted_full_replay_v1`-only clips: motion energy `7.201962` vs `3.032908`, gyro p95 `0.622577` vs `0.414941`.
- `window_kcenter_v1`-only clips vs `quality_stratified_random_v1`-only clips: motion energy `9.359926` vs `3.289022`, gyro p95 `0.900511` vs `0.532124`.

## Policy Motion Profiles

Dynamic energy is centered acceleration energy plus gyro energy, so gravity magnitude does not dominate this column.

| policy | selected rows | unique clips | dynamic energy | acc p95 | gyro p95 | acc delta p95 | gyro delta p95 | stationary | top regimes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `quality_only_v1` | 160 | 77 | 4.202589 | 10.669052 | 0.600506 | 0.956308 | 0.185240 | 0.050368 | mixed_motion:65, rotation_dominant:43, low_motion:36, high_dynamic:16 |
| `quality_stratified_random_v1` | 160 | 106 | 3.493797 | 10.602148 | 0.555082 | 0.825353 | 0.171575 | 0.050222 | mixed_motion:60, low_motion:47, rotation_dominant:47, high_dynamic:6 |
| `submitted_full_replay_v1` | 160 | 83 | 7.377292 | 10.827027 | 0.804623 | 1.421306 | 0.265156 | 0.084118 | rotation_dominant:67, mixed_motion:46, low_motion:34, high_dynamic:13 |
| `ts2vec_kcenter_v1` | 160 | 87 | 3.100233 | 10.507512 | 0.508153 | 0.928126 | 0.153910 | 0.105896 | mixed_motion:50, rotation_dominant:49, low_motion:41, high_dynamic:20 |
| `window_kcenter_v1` | 160 | 84 | 8.654065 | 10.960703 | 0.868211 | 1.729717 | 0.307218 | 0.080507 | rotation_dominant:68, mixed_motion:39, low_motion:37, high_dynamic:16 |

## Focal-Only Motion Contrasts

| comparison | units | focal-only clips | comparison-only clips | focal dynamic energy | comparison dynamic energy | focal gyro p95 | comparison gyro p95 | focal regimes | comparison regimes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `ts2vec_kcenter_v1` | 40 | 106 | 106 | 10.904860 | 2.521718 | 1.017625 | 0.474142 | rotation_dominant:48, low_motion:23, mixed_motion:23, high_dynamic:12 | mixed_motion:34, rotation_dominant:29, low_motion:27, high_dynamic:16 |
| `submitted_full_replay_v1` | 40 | 49 | 49 | 7.201962 | 3.032908 | 0.622577 | 0.414941 | rotation_dominant:14, mixed_motion:13, high_dynamic:11, low_motion:11 | mixed_motion:20, rotation_dominant:13, high_dynamic:8, low_motion:8 |
| `quality_stratified_random_v1` | 40 | 136 | 136 | 9.359926 | 3.289022 | 0.900511 | 0.532124 | rotation_dominant:58, mixed_motion:33, low_motion:30, high_dynamic:15 | mixed_motion:54, low_motion:40, rotation_dominant:37, high_dynamic:5 |

## Top Focal-Only Clip Examples

| comparison | sample | regime | motion energy | gyro p95 | acc delta p95 |
|---|---|---|---:|---:|---:|
| `ts2vec_kcenter_v1` | `worker00030_clip004` | `rotation_dominant` | 89.770973 | 4.762421 | 2.902842 |
| `ts2vec_kcenter_v1` | `worker00122_clip002` | `rotation_dominant` | 77.628007 | 2.837599 | 5.971571 |
| `ts2vec_kcenter_v1` | `worker00156_clip007` | `rotation_dominant` | 63.628515 | 2.959848 | 3.871684 |
| `ts2vec_kcenter_v1` | `worker00401_clip003` | `rotation_dominant` | 42.969044 | 1.835935 | 4.998253 |
| `ts2vec_kcenter_v1` | `worker00067_clip014` | `rotation_dominant` | 39.793853 | 2.874905 | 3.709024 |
| `ts2vec_kcenter_v1` | `worker00212_clip012` | `rotation_dominant` | 34.971122 | 2.862880 | 11.414129 |
| `ts2vec_kcenter_v1` | `worker00037_clip003` | `rotation_dominant` | 23.684760 | 3.905675 | 4.705445 |
| `ts2vec_kcenter_v1` | `worker00311_clip017` | `rotation_dominant` | 23.025707 | 2.138581 | 4.483462 |
| `submitted_full_replay_v1` | `worker00122_clip002` | `rotation_dominant` | 77.628007 | 2.837599 | 5.971571 |
| `submitted_full_replay_v1` | `worker00311_clip017` | `rotation_dominant` | 23.025707 | 2.138581 | 4.483462 |
| `submitted_full_replay_v1` | `worker00479_clip018` | `rotation_dominant` | 19.965304 | 2.517814 | 6.269591 |
| `submitted_full_replay_v1` | `worker00311_clip010` | `rotation_dominant` | 19.505819 | 2.297360 | 3.172598 |
| `submitted_full_replay_v1` | `worker00073_clip002` | `high_dynamic` | 13.205313 | 0.851637 | 2.747448 |
| `submitted_full_replay_v1` | `worker00335_clip018` | `rotation_dominant` | 9.966201 | 2.086612 | 1.610946 |
| `submitted_full_replay_v1` | `worker00364_clip002` | `high_dynamic` | 8.183618 | 0.468630 | 9.481917 |
| `submitted_full_replay_v1` | `worker00064_clip018` | `rotation_dominant` | 7.431548 | 1.311174 | 1.471883 |
| `quality_stratified_random_v1` | `worker00030_clip004` | `rotation_dominant` | 89.770973 | 4.762421 | 2.902842 |
| `quality_stratified_random_v1` | `worker00122_clip002` | `rotation_dominant` | 77.628007 | 2.837599 | 5.971571 |
| `quality_stratified_random_v1` | `worker00156_clip007` | `rotation_dominant` | 63.628515 | 2.959848 | 3.871684 |
| `quality_stratified_random_v1` | `worker00401_clip003` | `rotation_dominant` | 42.969044 | 1.835935 | 4.998253 |
| `quality_stratified_random_v1` | `worker00067_clip014` | `rotation_dominant` | 39.793853 | 2.874905 | 3.709024 |
| `quality_stratified_random_v1` | `worker00212_clip012` | `rotation_dominant` | 34.971122 | 2.862880 | 11.414129 |
| `quality_stratified_random_v1` | `worker00037_clip003` | `rotation_dominant` | 23.684760 | 3.905675 | 4.705445 |
| `quality_stratified_random_v1` | `worker00311_clip017` | `rotation_dominant` | 23.025707 | 2.138581 | 4.483462 |
