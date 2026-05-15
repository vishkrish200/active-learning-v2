# Downstream Forecast Policy Decision Card

## Decision

- decision: `survivor_confirmation_reversal`
- killed policy: `support_gap_window_probcover_v1`
- current champion: `window_kcenter_v1`
- final budget: `K=4`
- seed reports: `3`
- downstream task: `raw_imu_autoregressive_ridge_forecast`

- support_gap_window_probcover_v1 is intentionally excluded; this run is not a ProbCover appeal.
- window_kcenter_v1 is the locked candidate champion; compare it only to fixed survivors and baselines.
- submitted_full_replay_v1 remains a submitted-system comparator.
- ts2vec_kcenter_v1 remains a representation ablation / feature source.
- Do not add or tune policies based on this confirmation run.

## Final K Policy Ranking

| rank | policy | rows | mean after MSE | mean relative MSE reduction | best/tie wins | decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | `quality_only_v1` | 12 | 0.297020109 | 0.002004935 | 3 | negative control floor |
| 2 | `ts2vec_kcenter_v1` | 12 | 0.297188491 | 0.001427044 | 6 | feature source / ablation |
| 3 | `submitted_full_replay_v1` | 12 | 0.297425503 | 0.000888791 | 0 | defensible submitted comparator |
| 4 | `window_kcenter_v1` | 12 | 0.297481472 | 0.000634493 | 1 | survivor, not current best |
| 5 | `quality_stratified_random_v1` | 12 | 0.298367534 | -0.000512730 | 2 | required baseline |

## Pairwise Final K Deltas

Positive advantage means policy A has lower MSE than policy B.

| policy A | policy B | units | mean A-B MSE | A advantage | A wins |
|---|---|---:|---:|---:|---:|
| `quality_stratified_random_v1` | `window_kcenter_v1` | 12 | 0.000886062 | -0.000886062 | 3 / 12 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 12 | -0.000055969 | 0.000055969 | 6 / 12 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 12 | -0.000292981 | 0.000292981 | 8 / 12 |

## Acquisition Curves

| policy | K | rows | mean after MSE | mean relative MSE reduction |
|---|---:|---:|---:|---:|
| `quality_only_v1` | 0 | 12 | 0.298198772 | 0.000000000 |
| `quality_only_v1` | 1 | 12 | 0.296867454 | 0.002402029 |
| `quality_only_v1` | 2 | 12 | 0.296908600 | 0.002326196 |
| `quality_only_v1` | 4 | 12 | 0.297020109 | 0.002004935 |
| `quality_stratified_random_v1` | 0 | 12 | 0.298198772 | 0.000000000 |
| `quality_stratified_random_v1` | 1 | 12 | 0.298167576 | 0.000317609 |
| `quality_stratified_random_v1` | 2 | 12 | 0.298212255 | 0.000222237 |
| `quality_stratified_random_v1` | 4 | 12 | 0.298367534 | -0.000512730 |
| `submitted_full_replay_v1` | 0 | 12 | 0.298198772 | 0.000000000 |
| `submitted_full_replay_v1` | 1 | 12 | 0.297614032 | -0.000228200 |
| `submitted_full_replay_v1` | 2 | 12 | 0.297605328 | 0.000419731 |
| `submitted_full_replay_v1` | 4 | 12 | 0.297425503 | 0.000888791 |
| `ts2vec_kcenter_v1` | 0 | 12 | 0.298198772 | 0.000000000 |
| `ts2vec_kcenter_v1` | 1 | 12 | 0.298171283 | -0.000028433 |
| `ts2vec_kcenter_v1` | 2 | 12 | 0.298174549 | 0.000308105 |
| `ts2vec_kcenter_v1` | 4 | 12 | 0.297188491 | 0.001427044 |
| `window_kcenter_v1` | 0 | 12 | 0.298198772 | 0.000000000 |
| `window_kcenter_v1` | 1 | 12 | 0.297307540 | 0.001425767 |
| `window_kcenter_v1` | 2 | 12 | 0.297575974 | 0.000514082 |
| `window_kcenter_v1` | 4 | 12 | 0.297481472 | 0.000634493 |

## Archived ProbCover

- `support_gap_window_probcover_v1` is not present in this run. That is intentional for survivor-only confirmation.

## Selected Set Overlap

| policy A | policy B | units | mean Jaccard | exact match rate |
|---|---|---:|---:|---:|
| `quality_stratified_random_v1` | `window_kcenter_v1` | 12 | 0.087301587 | 0.000000000 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 12 | 0.546031746 | 0.166666667 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 12 | 0.206349206 | 0.000000000 |

## Selection Diagnostics

| policy | selected rows | mean quality | mean artifact | valid rate | duplicate-source batch rate | largest source frac |
|---|---:|---:|---:|---:|---:|---:|
| `quality_only_v1` | 48 | 1.000000000 | 0.000000000 | 1.000000000 | 1.000000000 | 0.625000000 |
| `quality_stratified_random_v1` | 48 | 1.000000000 | 0.000000000 | 1.000000000 | 0.500000000 | 0.437500000 |
| `submitted_full_replay_v1` | 48 | 0.995133482 | 0.000973304 | 1.000000000 | 0.333333333 | 0.333333333 |
| `ts2vec_kcenter_v1` | 48 | 0.993434063 | 0.001313187 | 1.000000000 | 0.333333333 | 0.333333333 |
| `window_kcenter_v1` | 48 | 0.992128004 | 0.001143246 | 1.000000000 | 0.333333333 | 0.333333333 |

## Hygiene

- leakage all ok: `True`
- max selected invalid rate: `0.000000000`
- total out-of-pool selections: `0`
- total target leaks: `0`
- total duplicate selected clips: `0`

## Frozen Next Step

- Treat this as a validation-only read of the frozen survivor policy set.
- Report whether the locked window-geometry interpretation survives or reverses on this run.
- If it reverses, do not add or tune policies on this validation target.
- Hold big neural downstream training unless a separate pre-registered downstream objective is approved.
