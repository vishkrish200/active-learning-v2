# GPT-5.5 Pro Scientific Claim Advisor

Source: ChatGPT web in Comet, visible model surface `Latest • 5.5` with `Pro • Extended` selected.

Consult date: 2026-05-10

## Advisor Read

- Treat the planned memo as a freeze/memo-review problem: separate defensible claims from statements stronger than the evidence supports.
- Claim `window_kcenter_v1` as the frozen post-submission offline downstream canary mean-MSE incumbent at K=4.
- Quantify the claim narrowly: window beats quality-stratified random by mean MSE and is lower in 28/40 policy-unit rows.
- Against submitted replay, say window improves mean MSE modestly but does not decisively dominate row-wise.
- Against TS2Vec, be especially careful: window wins mean MSE, but TS2Vec is lower in 24/40 rows and the median window advantage is slightly negative.
- Do not say TS2Vec is simply a loser. Frame it as a competitive heterogeneous ablation / complement that should not headline this downstream selector.
- The motion audit supports a hypothesis that window covers more dynamic/rotation-heavy regimes, but it does not prove scalar motion energy is the mechanism.
- Do not claim statistical significance without paired uncertainty tests or sign/permutation/bootstrap intervals.
- The next scientific step, if any, should be a pre-registered held-out validation with frozen policy set, K values, seeds, target splits, metrics, statistical tests, and decision rule.

## Final Recommendation

Claim `window_kcenter_v1` as the current frozen offline mean-MSE champion/canary, not as a proven universal selector. Keep submitted replay defensible but not best. Keep TS2Vec as a competitive heterogeneous ablation/complement. Treat the motion audit as a mechanistic hypothesis generator, not proof. The next move should be pre-registered held-out validation; otherwise the biggest threat is policy shopping dressed up as scientific discovery.
