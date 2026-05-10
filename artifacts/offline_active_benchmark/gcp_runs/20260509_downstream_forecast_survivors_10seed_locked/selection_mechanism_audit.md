# Selection Mechanism Audit

## Scope

- run input: `artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/gcp_downloads/survivors-locked-20260509T1515Z/results_survivors-locked-20260509T1515Z.tgz`
- source kind: `tar_archive`
- seeds: `10`
- units: `40`
- budget: `K=4`
- focal policy: `window_kcenter_v1`

## Read

- This is a selection-mechanism audit: it explains selected-set behavior, not a new policy decision.
- All summaries use already-selected rows and downstream forecast rows; no target features are used for selection.
- `window_kcenter_v1` vs `ts2vec_kcenter_v1` has largely different selected clips at K with mean clip Jaccard `0.225476`.
- `window_kcenter_v1` vs `submitted_full_replay_v1` has partially overlapping selected clips at K with mean clip Jaccard `0.557143`.
- `window_kcenter_v1` vs `quality_stratified_random_v1` has largely different selected clips at K with mean clip Jaccard `0.091667`.
- `window_kcenter_v1` duplicate-source batch rate is `0.225000` and mean unique-source fraction is `0.943750`.

## Policy Selection Profiles

| policy | selected rows | unique clips | unique sources | mean quality | mean artifact | duplicate-source batch rate | unique-source fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `quality_only_v1` | 160 | 77 | 37 | 1.000000000 | 0.000000000 | 1.000000000 | 0.562500000 |
| `quality_stratified_random_v1` | 160 | 106 | 54 | 1.000000000 | 0.000000000 | 0.875000000 | 0.718750000 |
| `submitted_full_replay_v1` | 160 | 83 | 66 | 0.992800025 | 0.001026604 | 0.250000000 | 0.937500000 |
| `ts2vec_kcenter_v1` | 160 | 87 | 69 | 0.995014486 | 0.000840054 | 0.200000000 | 0.950000000 |
| `window_kcenter_v1` | 160 | 84 | 61 | 0.994026928 | 0.001194614 | 0.225000000 | 0.943750000 |

## Policy Outcome Profiles

| policy | rows | mean after MSE | median after MSE | best/tie wins |
|---|---:|---:|---:|---:|
| `window_kcenter_v1` | 40 | 0.118593614 | 0.084865957 | 11 |
| `ts2vec_kcenter_v1` | 40 | 0.121089793 | 0.081280270 | 12 |
| `submitted_full_replay_v1` | 40 | 0.121103872 | 0.085900974 | 9 |
| `quality_stratified_random_v1` | 40 | 0.122818619 | 0.088552207 | 4 |
| `quality_only_v1` | 40 | 0.125936982 | 0.088684448 | 8 |

## Focal Policy Contrasts

| focal policy | comparison | units | clip Jaccard | source Jaccard | focal lower MSE | comparison lower MSE | focal advantage | Jaccard when focal wins | Jaccard when comparison wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `window_kcenter_v1` | `ts2vec_kcenter_v1` | 40 | 0.225476190 | 0.388214286 | 16 | 24 | 0.002496179 | 0.181547619 | 0.254761905 |
| `window_kcenter_v1` | `submitted_full_replay_v1` | 40 | 0.557142857 | 0.710833333 | 20 | 16 | 0.002510258 | 0.493333333 | 0.526190476 |
| `window_kcenter_v1` | `quality_stratified_random_v1` | 40 | 0.091666667 | 0.235654762 | 28 | 12 | 0.004225004 | 0.088435374 | 0.099206349 |

## Pairwise Selection Contrasts

| policy A | policy B | units | clip Jaccard | source Jaccard | A lower MSE | B lower MSE | A advantage | Jaccard when A wins | Jaccard when B wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `quality_stratified_random_v1` | `window_kcenter_v1` | 40 | 0.091666667 | 0.235654762 | 12 | 28 | -0.004225004 | 0.099206349 | 0.088435374 |
| `submitted_full_replay_v1` | `ts2vec_kcenter_v1` | 40 | 0.376190476 | 0.512083333 | 19 | 19 | -0.000014079 | 0.253132832 | 0.433583960 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 40 | 0.557142857 | 0.710833333 | 16 | 20 | -0.002510258 | 0.526190476 | 0.493333333 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 40 | 0.225476190 | 0.388214286 | 24 | 16 | -0.002496179 | 0.254761905 | 0.181547619 |

## Largest Focal Episode Contrasts

| comparison | unit | direction | MSE advantage | shared clips | focal-only clips | comparison-only clips |
|---|---|---|---:|---:|---:|---:|
| `ts2vec_kcenter_v1` | `seed_37:episode_001` | `focal_lower_mse` | 0.058160430 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_37:episode_003` | `focal_lower_mse` | 0.058160430 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_101:episode_002` | `comparison_lower_mse` | -0.039068364 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_101:episode_003` | `focal_lower_mse` | 0.023148115 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_41:episode_001` | `focal_lower_mse` | 0.014471795 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_41:episode_003` | `focal_lower_mse` | 0.014471795 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_23:episode_001` | `comparison_lower_mse` | -0.013299835 | 0 | 4 | 4 |
| `ts2vec_kcenter_v1` | `seed_23:episode_003` | `comparison_lower_mse` | -0.013299835 | 0 | 4 | 4 |
| `ts2vec_kcenter_v1` | `seed_53:episode_001` | `comparison_lower_mse` | -0.011283830 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_53:episode_003` | `comparison_lower_mse` | -0.011283830 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_37:episode_000` | `focal_lower_mse` | 0.006555128 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_37:episode_002` | `focal_lower_mse` | 0.006555128 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_97:episode_000` | `focal_lower_mse` | 0.005337930 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_97:episode_002` | `focal_lower_mse` | 0.005337930 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_97:episode_001` | `focal_lower_mse` | 0.001427191 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_97:episode_003` | `focal_lower_mse` | 0.001427191 | 1 | 3 | 3 |
| `ts2vec_kcenter_v1` | `seed_53:episode_000` | `comparison_lower_mse` | -0.001028492 | 0 | 4 | 4 |
| `ts2vec_kcenter_v1` | `seed_53:episode_002` | `comparison_lower_mse` | -0.001028492 | 0 | 4 | 4 |
| `ts2vec_kcenter_v1` | `seed_23:episode_000` | `comparison_lower_mse` | -0.000766471 | 2 | 2 | 2 |
| `ts2vec_kcenter_v1` | `seed_23:episode_002` | `comparison_lower_mse` | -0.000766471 | 2 | 2 | 2 |
