# Downstream Forecast Policy Decision Card

## Decision

- decision: `survivor_confirmation_window_kcenter`
- killed policy: `support_gap_window_probcover_v1`
- current champion: `window_kcenter_v1`
- final budget: `K=4`
- seed reports: `10`
- downstream task: `raw_imu_autoregressive_ridge_forecast`

- support_gap_window_probcover_v1 is intentionally excluded; this run is not a ProbCover appeal.
- window_kcenter_v1 is the locked candidate champion; compare it only to fixed survivors and baselines.
- submitted_full_replay_v1 remains a submitted-system comparator.
- ts2vec_kcenter_v1 remains a representation ablation / feature source.
- Do not add or tune policies based on this confirmation run.

## Final K Policy Ranking

| rank | policy | rows | mean after MSE | mean relative MSE reduction | best/tie wins | decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | `window_kcenter_v1` | 40 | 0.118593614 | 0.086395388 | 11 | current champion |
| 2 | `ts2vec_kcenter_v1` | 40 | 0.121089793 | 0.078833143 | 12 | feature source / ablation |
| 3 | `submitted_full_replay_v1` | 40 | 0.121103872 | 0.078781829 | 9 | defensible submitted comparator |
| 4 | `quality_stratified_random_v1` | 40 | 0.122818619 | 0.061063806 | 4 | required baseline |
| 5 | `quality_only_v1` | 40 | 0.125936982 | 0.049101767 | 8 | negative control floor |

## Pairwise Final K Deltas

Positive advantage means policy A has lower MSE than policy B.

| policy A | policy B | units | mean A-B MSE | A advantage | A wins |
|---|---|---:|---:|---:|---:|
| `quality_stratified_random_v1` | `window_kcenter_v1` | 40 | 0.004225004 | -0.004225004 | 12 / 40 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 40 | 0.002510258 | -0.002510258 | 16 / 40 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 40 | 0.002496179 | -0.002496179 | 24 / 40 |

## Acquisition Curves

| policy | K | rows | mean after MSE | mean relative MSE reduction |
|---|---:|---:|---:|---:|
| `quality_only_v1` | 0 | 40 | 0.152067497 | 0.000000000 |
| `quality_only_v1` | 1 | 40 | 0.127934672 | 0.033807411 |
| `quality_only_v1` | 2 | 40 | 0.127625043 | 0.038241688 |
| `quality_only_v1` | 4 | 40 | 0.125936982 | 0.049101767 |
| `quality_stratified_random_v1` | 0 | 40 | 0.152067497 | 0.000000000 |
| `quality_stratified_random_v1` | 1 | 40 | 0.127378731 | 0.031412245 |
| `quality_stratified_random_v1` | 2 | 40 | 0.124337661 | 0.044384483 |
| `quality_stratified_random_v1` | 4 | 40 | 0.122818619 | 0.061063806 |
| `submitted_full_replay_v1` | 0 | 40 | 0.152067497 | 0.000000000 |
| `submitted_full_replay_v1` | 1 | 40 | 0.127054095 | 0.043084732 |
| `submitted_full_replay_v1` | 2 | 40 | 0.124485721 | 0.052200159 |
| `submitted_full_replay_v1` | 4 | 40 | 0.121103872 | 0.078781829 |
| `ts2vec_kcenter_v1` | 0 | 40 | 0.152067497 | 0.000000000 |
| `ts2vec_kcenter_v1` | 1 | 40 | 0.128157077 | 0.038870201 |
| `ts2vec_kcenter_v1` | 2 | 40 | 0.126274573 | 0.054387793 |
| `ts2vec_kcenter_v1` | 4 | 40 | 0.121089793 | 0.078833143 |
| `window_kcenter_v1` | 0 | 40 | 0.152067497 | 0.000000000 |
| `window_kcenter_v1` | 1 | 40 | 0.125529648 | 0.041937897 |
| `window_kcenter_v1` | 2 | 40 | 0.124921054 | 0.056523506 |
| `window_kcenter_v1` | 4 | 40 | 0.118593614 | 0.086395388 |

## Archived ProbCover

- `support_gap_window_probcover_v1` is not present in this run. That is intentional for survivor-only confirmation.

## Selected Set Overlap

| policy A | policy B | units | mean Jaccard | exact match rate |
|---|---|---:|---:|---:|
| `quality_stratified_random_v1` | `window_kcenter_v1` | 40 | 0.091666667 | 0.000000000 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 40 | 0.557142857 | 0.100000000 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 40 | 0.225476190 | 0.000000000 |

## Selection Diagnostics

| policy | selected rows | mean quality | mean artifact | valid rate | duplicate-source batch rate | largest source frac |
|---|---:|---:|---:|---:|---:|---:|
| `quality_only_v1` | 160 | 1.000000000 | 0.000000000 | 1.000000000 | 1.000000000 | 0.575000000 |
| `quality_stratified_random_v1` | 160 | 1.000000000 | 0.000000000 | 1.000000000 | 0.875000000 | 0.487500000 |
| `submitted_full_replay_v1` | 160 | 0.992800025 | 0.001026604 | 1.000000000 | 0.250000000 | 0.312500000 |
| `ts2vec_kcenter_v1` | 160 | 0.995014486 | 0.000840054 | 1.000000000 | 0.200000000 | 0.300000000 |
| `window_kcenter_v1` | 160 | 0.994026928 | 0.001194614 | 1.000000000 | 0.225000000 | 0.306250000 |

## Hygiene

- leakage all ok: `True`
- max selected invalid rate: `0.000000000`
- total out-of-pool selections: `0`
- total target leaks: `0`
- total duplicate selected clips: `0`

## Frozen Next Step

- Write/keep this result card as the frozen ProbCover decision.
- Run only diagnostics that explain the loss; do not alter ProbCover thresholds or graph construction.
- If running more compute, use a locked survivor-only confirmation with no new policies and no ProbCover rescue.
- Hold big neural downstream training until the survivor-only selector conclusion is stable.
