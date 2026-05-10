# Downstream Selector Evidence Synthesis - 2026-05-10

## Bottom Line

The current selector evidence is mixed and should be written up honestly. The submitted system remains defensible as a deterministic marginal-data-value selector, but the later active-learning evidence does not support a strong claim that any one selector is a robust downstream winner.

Do not tune another selector on the raw forecast canary.

## Evidence So Far

### Submitted Package

The submitted package was a deterministic geometric marginal-data-value selector. It used frozen TS2Vec/window novelty, quality gates, k-center redundancy, artifact-aware reranking, full-support comparison against old clips, and ranking over the public new clips.

This was not submitted as a learned query policy, not as TS2Vec retraining, and not as downstream retraining proof.

### Initial Downstream Forecast Canary

The first three-seed raw IMU autoregressive forecast canary favored window/raw geometry at final `K=4`:

1. `window_kcenter_v1`
2. `submitted_full_replay_v1`
3. `ts2vec_kcenter_v1`
4. `quality_stratified_random_v1`
5. `quality_only_v1`

That result was enough to demote TS2Vec from lead story to feature source / ablation, and to make window geometry the frozen candidate champion for one held-out validation.

### Held-Out Source Forecast Validation

The source-disjoint held-out validation reversed the window-champion interpretation. It ran on `worker00501-worker00900`, disjoint from the locked `worker00001-worker00500` canary slice.

Final `K=4` forecast MSE ranking:

1. `quality_only_v1`: `0.297020109`
2. `ts2vec_kcenter_v1`: `0.297188491`
3. `submitted_full_replay_v1`: `0.297425503`
4. `window_kcenter_v1`: `0.297481472`
5. `quality_stratified_random_v1`: `0.298367534`

The useful nuance is:

- `window_kcenter_v1` still beat `quality_stratified_random_v1`.
- `window_kcenter_v1` did not beat `ts2vec_kcenter_v1` or `submitted_full_replay_v1`.
- `quality_only_v1` winning is a warning about this canary's sensitivity to quality and duplicate-source collapse, not a reason to promote quality-only as the acquisition story.
- TS2Vec should remain a competitive feature source / ablation, not be dismissed as nonsense and not be promoted as the main strategy.

## Scientific Decision

For the raw forecast canary, stop selector iteration. The evidence is not strong enough to justify big neural downstream training or another policy search.

The correct write-up is:

- the submitted selector is mechanically and methodologically defensible for the challenge package;
- the active-learning downstream evidence is mixed;
- simple geometric/window selection is useful but not robustly dominant;
- TS2Vec is competitive as a representation but unstable as a lead story;
- quality-only controls expose objective sensitivity and possible noise in the raw forecast canary.

## Next Objective

If continuing scientifically, use a genuinely separate pre-registered objective instead of tuning against the forecast canary. The next objective is frozen in:

`configs/downstream_source_family_bridge_fresh_source_3seed.json`

That objective uses fresh workers `worker00901-worker01200`, disjoint from both prior source slices, and evaluates a source-family bridge classifier with frozen feature heads. It is still a pseudo-label objective, not real challenge-label proof.
