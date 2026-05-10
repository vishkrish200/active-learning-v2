# Downstream Forecast Policy Decision Card

## Decision

- decision: `kill_probcover_promote_window_kcenter`
- killed policy: `support_gap_window_probcover_v1`
- current champion: `window_kcenter_v1`
- final budget: `K=4`
- seed reports: `3`
- downstream task: `raw_imu_autoregressive_ridge_forecast`

- support_gap_window_probcover_v1 failed the pre-registered K=4 downstream MSE rule; do not tune it further.
- window_kcenter_v1 is the current low-budget champion on this downstream forecast canary.
- submitted_full_replay_v1 remains a defensible submitted-system comparator, not proof of downstream retraining optimality.
- ts2vec_kcenter_v1 remains useful as a representation ablation / feature source, not the lead active-learning strategy.
- The hidden downstream target has now been used for decisions; stop designing new selectors against it.

## Final K Policy Ranking

| rank | policy | rows | mean after MSE | mean relative MSE reduction | best/tie wins | decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | `window_kcenter_v1` | 12 | 0.145679406 | 0.097243361 | 6 | current champion |
| 2 | `submitted_full_replay_v1` | 12 | 0.147856591 | 0.082992028 | 2 | defensible submitted comparator |
| 3 | `ts2vec_kcenter_v1` | 12 | 0.154147268 | 0.061952405 | 3 | feature source / ablation |
| 4 | `quality_stratified_random_v1` | 12 | 0.157406922 | 0.036060050 | 1 | required baseline |
| 5 | `support_gap_window_probcover_v1` | 12 | 0.163094270 | 0.032208134 | 2 | failed; archive |
| 6 | `quality_only_v1` | 12 | 0.167782254 | 0.000600577 | 0 | negative control floor |

## Pairwise Final K Deltas

Positive advantage means policy A has lower MSE than policy B.

| policy A | policy B | units | mean A-B MSE | A advantage | A wins |
|---|---|---:|---:|---:|---:|
| `quality_only_v1` | `support_gap_window_probcover_v1` | 12 | 0.004687984 | -0.004687984 | 2 / 12 |
| `quality_stratified_random_v1` | `support_gap_window_probcover_v1` | 12 | -0.005687348 | 0.005687348 | 6 / 12 |
| `quality_stratified_random_v1` | `window_kcenter_v1` | 12 | 0.011727516 | -0.011727516 | 4 / 12 |
| `submitted_full_replay_v1` | `support_gap_window_probcover_v1` | 12 | -0.015237679 | 0.015237679 | 8 / 12 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 12 | 0.002177185 | -0.002177185 | 4 / 12 |
| `support_gap_window_probcover_v1` | `ts2vec_kcenter_v1` | 12 | 0.008947002 | -0.008947002 | 4 / 12 |
| `support_gap_window_probcover_v1` | `window_kcenter_v1` | 12 | 0.017414865 | -0.017414865 | 6 / 12 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 12 | 0.008467863 | -0.008467863 | 6 / 12 |

## Acquisition Curves

| policy | K | rows | mean after MSE | mean relative MSE reduction |
|---|---:|---:|---:|---:|
| `quality_only_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `quality_only_v1` | 1 | 12 | 0.166932614 | -0.001209577 |
| `quality_only_v1` | 2 | 12 | 0.168427781 | -0.005684527 |
| `quality_only_v1` | 4 | 12 | 0.167782254 | 0.000600577 |
| `quality_stratified_random_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `quality_stratified_random_v1` | 1 | 12 | 0.162091126 | 0.015882008 |
| `quality_stratified_random_v1` | 2 | 12 | 0.157958253 | 0.029855762 |
| `quality_stratified_random_v1` | 4 | 12 | 0.157406922 | 0.036060050 |
| `submitted_full_replay_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `submitted_full_replay_v1` | 1 | 12 | 0.164334035 | 0.022089066 |
| `submitted_full_replay_v1` | 2 | 12 | 0.158008085 | 0.045123756 |
| `submitted_full_replay_v1` | 4 | 12 | 0.147856591 | 0.082992028 |
| `support_gap_window_probcover_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `support_gap_window_probcover_v1` | 1 | 12 | 0.166132064 | 0.004858728 |
| `support_gap_window_probcover_v1` | 2 | 12 | 0.165407784 | 0.012520225 |
| `support_gap_window_probcover_v1` | 4 | 12 | 0.163094270 | 0.032208134 |
| `ts2vec_kcenter_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `ts2vec_kcenter_v1` | 1 | 12 | 0.164319797 | 0.022207481 |
| `ts2vec_kcenter_v1` | 2 | 12 | 0.163842093 | 0.027067555 |
| `ts2vec_kcenter_v1` | 4 | 12 | 0.154147268 | 0.061952405 |
| `window_kcenter_v1` | 0 | 12 | 0.166447390 | 0.000000000 |
| `window_kcenter_v1` | 1 | 12 | 0.157778382 | 0.033225422 |
| `window_kcenter_v1` | 2 | 12 | 0.153002899 | 0.063761079 |
| `window_kcenter_v1` | 4 | 12 | 0.145679406 | 0.097243361 |

## ProbCover Episode Deltas

Positive advantage means ProbCover has lower MSE than the comparison.

| comparison | mean advantage | median advantage | wins | worst unit | worst advantage |
|---|---:|---:|---:|---|---:|
| vs `window_kcenter_v1` | -0.017414865 | 0.000030493 | 6 / 12 | `seed_37:episode_001:1` | -0.101088476 |
| vs `quality_stratified_random_v1` | -0.005687348 | -0.000245025 | 6 / 12 | `seed_37:episode_003:3` | -0.044127579 |
| vs `submitted_full_replay_v1` | -0.015237679 | -0.000058358 | 4 / 12 | `seed_37:episode_001:1` | -0.092159465 |
| vs `ts2vec_kcenter_v1` | -0.008947002 | -0.000625339 | 4 / 12 | `seed_37:episode_001:1` | -0.042928046 |

## Selected Set Overlap

| policy A | policy B | units | mean Jaccard | exact match rate |
|---|---|---:|---:|---:|
| `quality_only_v1` | `support_gap_window_probcover_v1` | 12 | 0.158730159 | 0.000000000 |
| `quality_stratified_random_v1` | `support_gap_window_probcover_v1` | 12 | 0.137301587 | 0.000000000 |
| `quality_stratified_random_v1` | `window_kcenter_v1` | 12 | 0.063492063 | 0.000000000 |
| `submitted_full_replay_v1` | `support_gap_window_probcover_v1` | 12 | 0.103174603 | 0.000000000 |
| `submitted_full_replay_v1` | `window_kcenter_v1` | 12 | 0.666666667 | 0.166666667 |
| `support_gap_window_probcover_v1` | `ts2vec_kcenter_v1` | 12 | 0.126984127 | 0.000000000 |
| `support_gap_window_probcover_v1` | `window_kcenter_v1` | 12 | 0.119047619 | 0.000000000 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | 12 | 0.258730159 | 0.000000000 |

## Selection Diagnostics

| policy | selected rows | mean quality | mean artifact | valid rate | duplicate-source batch rate | largest source frac |
|---|---:|---:|---:|---:|---:|---:|
| `quality_only_v1` | 48 | 1.000000000 | 0.000000000 | 1.000000000 | 1.000000000 | 0.541666667 |
| `quality_stratified_random_v1` | 48 | 1.000000000 | 0.000000000 | 1.000000000 | 0.833333333 | 0.479166667 |
| `submitted_full_replay_v1` | 48 | 0.996523915 | 0.000695217 | 1.000000000 | 0.500000000 | 0.375000000 |
| `support_gap_window_probcover_v1` | 48 | 0.995133482 | 0.000973304 | 1.000000000 | 0.000000000 | 0.250000000 |
| `ts2vec_kcenter_v1` | 48 | 0.997489495 | 0.000502101 | 1.000000000 | 0.166666667 | 0.291666667 |
| `window_kcenter_v1` | 48 | 0.996408046 | 0.000718391 | 1.000000000 | 0.333333333 | 0.333333333 |

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
