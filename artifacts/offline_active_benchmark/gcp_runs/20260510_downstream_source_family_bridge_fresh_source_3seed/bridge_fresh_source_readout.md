# Fresh-Source Bridge Downstream Readout

Date: 2026-05-10
Run ID: `bridge-fresh-source-20260510T1125Z`
Config: `configs/downstream_source_family_bridge_fresh_source_3seed.json`

## What This Was

This was the pre-registered fresh-source source-family bridge validation. It was not a forecast retry, not another 10-seed run, not TS2Vec retraining, and not large downstream training.

- Fresh manifest: `artifacts/offline_active_benchmark/gcp_inputs/heldout_pretrain_urls_workers_0901_1200_c20.txt`
- Fresh rows: 6,000 clips from 300 source groups, `worker00901` through `worker01200`
- Overlap with locked canary workers `worker00001-worker00500`: 0
- Overlap with forecast-validation workers `worker00501-worker00900`: 0
- Seeds: `211`, `223`, `251`
- Final validation units: 18 seed-episodes
- VM state: success; VM deleted after artifact copy

## Primary Read

This run is not selector-promotive under the pre-registered downstream primary metric.

The pre-registered primary metric was final `K=4` ridge-classifier balanced-accuracy gain on the source-family bridge objective. On that metric, `quality_stratified_random_v1` was best among deployable policies.

| rank | policy | ridge balanced-accuracy gain | ridge NLL reduction | target-family discovery |
|---:|---|---:|---:|---:|
| 1 | `quality_stratified_random_v1` | 0.092592593 | 14.580146315 | 0.555555556 |
| 2 | `submitted_full_replay_v1` | 0.078703704 | 20.351156195 | 0.777777778 |
| 3 | `oracle_greedy_target_family_v1` | 0.075617284 | 26.093077083 | 1.000000000 |
| 4 | `quality_only_v1` | 0.066358025 | 11.002027217 | 0.416666667 |
| 5 | `window_kcenter_v1` | 0.058641975 | 18.141548061 | 0.694444444 |
| 6 | `ts2vec_kcenter_v1` | 0.057098765 | 18.173129810 | 0.694444444 |

The oracle having perfect target-family discovery but lower ridge balanced-accuracy gain than the random baseline is a warning that the ridge primary head is not a clean identifiability metric for this objective.

## Secondary Signals

The secondary signals are more favorable to the deterministic selectors, but they do not override the primary rule.

Coverage final gain at `K=4`:

| rank | policy | mean final gain |
|---:|---|---:|
| 1 | `oracle_greedy_target_family_v1` | 0.1299 |
| 2 | `window_kcenter_v1` | 0.1240 |
| 3 | `submitted_full_replay_v1` | 0.1192 |
| 4 | `ts2vec_kcenter_v1` | 0.1119 |
| 5 | `quality_stratified_random_v1` | 0.0628 |
| 6 | `quality_only_v1` | 0.0445 |

Target-family discovery also favored `submitted_full_replay_v1`, `window_kcenter_v1`, and `ts2vec_kcenter_v1` over the random baseline.

## Hygiene

The basic hygiene checks were clean.

- selected invalid rate: 0.0 for all policies
- duplicate clip fraction: 0.0 for all policies
- max artifact score among deployable policies: 0.0178
- VM deleted: yes

## Decision

Treat this as inconclusive / negative for downstream promotion.

The fresh bridge objective says the selectors are good at coverage and target-family discovery, but it does not show a reliable downstream classifier-accuracy advantage over random. That means it should not rescue the selector story after the held-out forecast reversal.

Do not tune policies on this target. Do not start big neural downstream training from this result. The honest scientific state is now mixed evidence across downstream checks, with no robust selector champion.
