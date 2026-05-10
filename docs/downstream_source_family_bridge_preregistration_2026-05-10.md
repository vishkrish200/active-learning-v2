# Fresh-Source Bridge Downstream Preregistration - 2026-05-10

## Decision

Run at most one fresh-source source-family bridge validation for the frozen survivor policies.

This is a separate downstream objective after the held-out forecast reversal. It is not a forecast retry, not another 10-seed estimate, and not a new selector search.

## Frozen Inputs

- config: `configs/downstream_source_family_bridge_fresh_source_3seed.json`
- manifest: `artifacts/offline_active_benchmark/gcp_inputs/heldout_pretrain_urls_workers_0901_1200_c20.txt`
- fresh source groups: `worker00901` through `worker01200`
- excluded locked canary groups: `worker00001` through `worker00500`
- excluded forecast-validation groups: `worker00501` through `worker00900`
- seeds: `211, 223, 251`
- final budget: `K=4`

## Frozen Policies

- `quality_stratified_random_v1`
- `quality_only_v1`
- `window_kcenter_v1`
- `submitted_full_replay_v1`
- `ts2vec_kcenter_v1`
- `oracle_greedy_target_family_v1` as diagnostic ceiling only

No ProbCover rescue. No density-weighted variant. No learned query policy.

## Objective

The objective is source-family bridge downstream classification with frozen representations.

Selected candidate clips are appended to the support set, then frozen-feature classifier heads are retrained and evaluated on held-out target-family clips.

Primary metric:

- `K=4` ridge-classifier mean balanced-accuracy gain

Secondary metrics:

- ridge-classifier NLL reduction
- nearest-centroid balanced-accuracy gain
- target-family discovery rate

## Decision Rule

A fixed non-oracle policy must beat both `quality_stratified_random_v1` and `quality_only_v1` on the primary metric, and must not lose target-family discovery, to count as positive evidence for this separate objective.

If `quality_only_v1` wins, the objective is quality-sensitive and not selector-promotive.

If results are flat, the objective is not identifiable enough to justify more work.

If the result conflicts with the forecast canary again, report mixed evidence and stop. Do not tune selectors on this target.

## Non-Claims

- This is not challenge-label downstream proof.
- This is not large neural downstream training.
- This is not TS2Vec retraining.
- This is not a claim that any selector is robustly optimal.
- This is not permission to choose whichever metric likes a policy after the fact.

## Launch Command

```bash
python3 scripts/launch_coverage_benchmark_gcp.py \
  --config configs/downstream_source_family_bridge_fresh_source_3seed.json \
  --run-id bridge-fresh-source-20260510T0000Z
```
