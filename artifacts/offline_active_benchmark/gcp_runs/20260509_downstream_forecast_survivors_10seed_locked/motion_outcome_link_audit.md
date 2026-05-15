# Motion Outcome Link Audit

## Scope

- run input: `artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/gcp_downloads/survivors-locked-20260509T1515Z/results_survivors-locked-20260509T1515Z.tgz`
- motion audit: `artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/selected_motion_regime_audit.json`
- seeds: `10`
- budget: `K=4`
- policy-unit rows: `200`
- focal policy: `window_kcenter_v1`

## Read

- This is a descriptive mechanism-to-outcome audit on the locked downstream run, not causal proof and not a new policy-selection loop.
- It asks whether selected raw-motion regimes line up with lower held-out forecast MSE after acquisition.
- The result supports a regime-level explanation, not a scalar motion-energy rule.
- `window_kcenter_v1` has mean after MSE `0.118594` with selected dynamic energy `8.654065` and rotation-dominant rate `0.425000`.
- `window_kcenter_v1` vs `ts2vec_kcenter_v1`: MSE advantage `0.002496`, motion-energy delta `5.553832`, focal-more-motion-and-lower-MSE units `12/40`.
- `window_kcenter_v1` vs `submitted_full_replay_v1`: MSE advantage `0.002510`, motion-energy delta `1.276773`, focal-more-motion-and-lower-MSE units `12/40`.
- `window_kcenter_v1` vs `quality_stratified_random_v1`: MSE advantage `0.004225`, motion-energy delta `5.160268`, focal-more-motion-and-lower-MSE units `23/40`.

## Policy Motion-Outcome Profiles

| policy | rows | after MSE | rel MSE reduction | selected dynamic energy | gyro p95 | rotation rate | high-dynamic rate | missing selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `window_kcenter_v1` | 40 | 0.118594 | 0.086395 | 8.654065 | 0.868211 | 0.425000 | 0.100000 | 0.000000 |
| `ts2vec_kcenter_v1` | 40 | 0.121090 | 0.078833 | 3.100233 | 0.508153 | 0.306250 | 0.125000 | 0.000000 |
| `submitted_full_replay_v1` | 40 | 0.121104 | 0.078782 | 7.377292 | 0.804623 | 0.418750 | 0.081250 | 0.000000 |
| `quality_stratified_random_v1` | 40 | 0.122819 | 0.061064 | 3.493797 | 0.555082 | 0.293750 | 0.037500 | 0.000000 |
| `quality_only_v1` | 40 | 0.125937 | 0.049102 | 4.202589 | 0.600506 | 0.268750 | 0.100000 | 0.000000 |

## Focal Pairwise Motion-Outcome Contrasts

Positive MSE advantage means the focal policy has lower downstream MSE.

| comparison | units | MSE advantage | median advantage | focal lower MSE | comparison lower MSE | motion delta | gyro delta | rotation-rate delta | focal more motion and lower MSE | motion-delta corr with advantage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ts2vec_kcenter_v1` | 40 | 0.002496 | -0.000083 | 16 | 24 | 5.553832 | 0.360058 | 0.118750 | 12 | -0.121813 |
| `submitted_full_replay_v1` | 40 | 0.002510 | 0.000001 | 20 | 16 | 1.276773 | 0.063589 | 0.006250 | 12 | 0.128389 |
| `quality_stratified_random_v1` | 40 | 0.004225 | 0.000344 | 28 | 12 | 5.160268 | 0.313129 | 0.131250 | 23 | -0.060590 |

## Descriptive Associations

These correlations are descriptive diagnostics from the locked run, not prospective selector evidence.

| feature | outcome | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| `mean_selected_motion_energy` | `after_mse` | 200 | 0.100080 | 0.122754 |
| `mean_selected_motion_energy` | `relative_mse_reduction` | 200 | 0.037419 | 0.259867 |
| `mean_selected_gyro_norm_p95` | `after_mse` | 200 | 0.325069 | 0.242524 |
| `mean_selected_gyro_norm_p95` | `relative_mse_reduction` | 200 | 0.031590 | 0.199410 |
| `mean_selected_acc_delta_norm_p95` | `after_mse` | 200 | 0.211001 | 0.191661 |
| `mean_selected_acc_delta_norm_p95` | `relative_mse_reduction` | 200 | 0.110108 | 0.289023 |
| `mean_selected_gyro_delta_norm_p95` | `after_mse` | 200 | 0.310830 | 0.228844 |
| `mean_selected_gyro_delta_norm_p95` | `relative_mse_reduction` | 200 | 0.057263 | 0.257102 |
| `mean_selected_gyro_energy_fraction` | `after_mse` | 200 | -0.216890 | -0.287742 |
| `mean_selected_gyro_energy_fraction` | `relative_mse_reduction` | 200 | -0.045147 | -0.090489 |
| `mean_selected_stationary_fraction` | `after_mse` | 200 | 0.082148 | 0.248095 |
| `mean_selected_stationary_fraction` | `relative_mse_reduction` | 200 | -0.135187 | -0.152732 |
| `mean_selected_rotation_dominant_rate` | `after_mse` | 200 | 0.343601 | 0.237169 |
| `mean_selected_rotation_dominant_rate` | `relative_mse_reduction` | 200 | 0.092043 | 0.253355 |
| `mean_selected_high_dynamic_rate` | `after_mse` | 200 | -0.168197 | -0.294946 |
| `mean_selected_high_dynamic_rate` | `relative_mse_reduction` | 200 | -0.003794 | 0.127456 |
