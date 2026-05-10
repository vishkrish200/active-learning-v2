# Frozen Scientific Claim Memo

## Decision

- decision: `freeze_window_kcenter_as_downstream_canary_mean_risk_incumbent`
- status: frozen synthesis; no new selector decision is made here

## What This Claims

- `window_kcenter_v1` is the current downstream forecast canary mean-risk incumbent by mean K=4 after-MSE (`0.118593614` over `40` policy-unit rows).
- `ts2vec_kcenter_v1` remains scientifically useful as a representation ablation or feature source, but it is not the lead acquisition story in this benchmark (`mean after-MSE 0.121089793`).
- `submitted_full_replay_v1` remains a defensible submitted-system comparator (`mean after-MSE 0.121103872`), not the proof target.
- `quality_stratified_random_v1` is still the required baseline (`mean after-MSE 0.122818619`); the incumbent must be described relative to it.
- The best mechanism-level read is that `window_kcenter_v1` covers different raw-motion regimes: selected dynamic energy `8.654065135` and selected rotation-dominant rate `0.425000000`.

## What This Does Not Claim

- This is not a learned query policy.
- This is not proof that neural downstream training would improve with the same selected clips.
- This is not evidence that TS2Vec is useless; TS2Vec remains a useful feature source / ablation.
- This is not a scalar motion-energy ranker: the motion audit is explanatory, and per-unit scalar correlations are weak or mixed.
- This is not a license to tune another selector on the same hidden downstream target.
- This is not a claim that `window_kcenter_v1` wins every paired unit; the TS2Vec comparison is heterogeneous.

## Evidence Anchors

- Against quality-stratified random, `window_kcenter_v1` is cleaner: mean MSE advantage `0.004225004`.
- Against TS2Vec, `window_kcenter_v1` has positive mean MSE advantage `0.002496179`, but paired wins are heterogeneous: `16` vs `24`.
- Selected-set mechanism differs from TS2Vec: clip Jaccard `0.225476190`, source Jaccard `0.388214286`.
- Window-only vs TS2Vec-only clips are much more dynamic: energy `10.904860476` vs `2.521717968`.
- Motion-outcome link is not monotone: median advantage `-0.000082515`, focal lower-MSE count `16` vs comparison `24`.
- Hygiene remains clean: leakage_all_ok `True`, target leaks `0`, out-of-pool selections `0`.

## Self-Deception Risks

- policy shopping: the hidden downstream target has already influenced decisions, so do not introduce or tune new selectors against this same canary.
- Mean-vs-win-count overread: mean MSE favors window, but the TS2Vec pairwise result is heterogeneous and must not be phrased as a universal win.
- Mechanism overread: higher dynamic/rotation-heavy support is a plausible regime-level explanation, not a causal scalar feature rule.
- Submission overclaim: the submitted system can be described as defensible, but not retroactively proven as a full active-learning loop.
- Complexity creep: no TS2Vec retraining, nonlinear ranker, or neural downstream training should be started from this memo alone.

## GPT-5.5 Pro Advisor Read

- source: `GPT-5.5 Pro / Extended Pro web consultation`
- summary: Treat the planned memo as a freeze/memo-review problem: separate defensible claims from statements stronger than the evidence supports.
- Claim `window_kcenter_v1` as the frozen post-submission offline downstream canary mean-MSE incumbent at K=4.
- Quantify the claim narrowly: window beats quality-stratified random by mean MSE and is lower in 28/40 policy-unit rows.
- Against submitted replay, say window improves mean MSE modestly but does not decisively dominate row-wise.
- Against TS2Vec, be especially careful: window wins mean MSE, but TS2Vec is lower in 24/40 rows and the median window advantage is slightly negative.
- Do not say TS2Vec is simply a loser. Frame it as a competitive heterogeneous ablation / complement that should not headline this downstream selector.
- The motion audit supports a hypothesis that window covers more dynamic/rotation-heavy regimes, but it does not prove scalar motion energy is the mechanism.
- Do not claim statistical significance without paired uncertainty tests or sign/permutation/bootstrap intervals.
- The next scientific step, if any, should be a pre-registered held-out validation with frozen policy set, K values, seeds, target splits, metrics, statistical tests, and decision rule.

## Frozen Next Step

- Freeze this memo as the current scientific interpretation of the downstream canary.
- Do not tune or add selectors on this same downstream target.
- Use the memo to decide whether a genuinely different downstream objective/model family is worth a separate pre-registered experiment.
- Keep TS2Vec framed as a feature source / ablation and window geometry as the current low-budget canary mean-risk incumbent.
