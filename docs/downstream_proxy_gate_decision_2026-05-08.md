# Downstream Proxy Gate Decision - 2026-05-08

## Decision

Hold downstream training.

The source-family pseudo-label downstream gate did not promote TS2Vec-based acquisition policies. The completed quality-stratified repeat shows this proxy is more sensitive to quality and source-family effects than to the submitted TS2Vec/window novelty selector.

This does not invalidate the submitted challenge package. The submitted system remains a deterministic marginal-data-value / novelty-ranking selector using frozen TS2Vec plus window features, quality gates, redundancy control, and artifact-aware reranking. It is not yet validated as a complete active-learning policy.

## Evidence

Primary repeat run:

- run id: `downstream_supervised_label_holdout_quality_control_3seed_20260507T205411Z`
- seeds: `17, 23, 37`
- compute: GCP CPU, frozen TS2Vec, no retraining
- GCS reports: `gs://active-learning-v2-802636843791/offline_active_benchmark/downstream_supervised_label_holdout_quality_control_3seed_20260507T205411Z/reports/`
- leakage OK: `True`
- hard gate non-flat all seeds: `False`
- decision: `hold_gate_not_discriminative`
- paired audit next gate: `hold_proxy_not_promotive`

Paired total utility vs `random_valid`:

| policy | after accuracy | paired total gain delta vs random |
|---|---:|---:|
| `quality_stratified_random` | `0.805556` | `+0.041667` |
| `quality_only` | `0.708333` | `-0.055556` |
| `submitted_full_replay` | `0.625000` | `-0.138889` |
| `old_novelty_ts2vec` | `0.583333` | `-0.180556` |

Final-round incremental metrics were misleading:

- `quality_only` had `+0.375000` final-round incremental accuracy delta vs random.
- Paired total accuracy from the common round-0 baseline instead put `quality_stratified_random` above random and TS2Vec policies below random.

## Interpretation

The current proxy is not a scientific promotion gate for the submitted selector.

It proves that the code path can run an acquisition-update-evaluation loop, but the utility target is not aligned enough with the challenge claim. Source-family pseudo-label classification rewards acquiring label-family bridges and high-quality clips. That is not the same as proving marginal value for unseen IMU novelty.

## Stop Rule

Do not launch downstream training, TS2Vec retraining, nonlinear rankers, or larger GCP/GPU runs from this gate.

A TS2Vec-based policy can be reconsidered only after it beats:

- `random_valid`
- `quality_only`
- `quality_stratified_random`

under paired total utility from a common initial baseline, on a utility target that is not just source-family pseudo-label classification.

## Next Scientific Move

Design a better held-out utility target before spending more compute.

Candidate directions:

1. Held-out coverage utility over sensor-distribution shifts, evaluated in TS2Vec/window/raw-shape spaces with paired policy deltas.
2. Human/visual audit of selected clips from `submitted_full_replay`, `quality_stratified_random`, `quality_only`, and `random_valid`.
3. A downstream model only after a real target is chosen and the selector wins the paired gate.
