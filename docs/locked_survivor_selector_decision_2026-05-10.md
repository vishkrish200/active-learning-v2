# Locked Survivor Selector Decision

Date: 2026-05-10

Worktree:
`/Users/vishnukrishnan/.config/superpowers/worktrees/active-learning-v2/codex/offline-active-benchmark`

Do not apply this research-worktree decision note to the clean submitted repo:
`/Users/vishnukrishnan/Developer/imu-novelty-submission`

## Scope

This note freezes the active-learning selector decision after the locked survivor-only downstream forecast confirmation.

It is not a new submission claim, not neural downstream training proof, and not evidence that TS2Vec should be the lead query policy. It is a narrow canary result: selected clips are appended to support, a raw IMU autoregressive ridge forecaster is retrained after acquisition, and held-out hidden target forecast MSE is measured.

## Locked Confirmation Run

Run id: `survivors-locked-20260509T1515Z`

Summary artifact:
`artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/locked_survivor_confirmation_summary.md`

Decision card:
`artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/downstream_forecast_survivor_decision_card.md`

Success marker:
`artifacts/offline_active_benchmark/gcp_runs/20260509_downstream_forecast_survivors_10seed_locked/gcp_downloads/survivors-locked-20260509T1515Z/status/success.json`

Scope:

- Seeds: `17`, `23`, `37`, `41`, `53`, `67`, `79`, `83`, `97`, `101`
- Episodes per seed: `4`
- Final budget: `K=4`
- Policies: `quality_stratified_random_v1`, `quality_only_v1`, `window_kcenter_v1`, `submitted_full_replay_v1`, `ts2vec_kcenter_v1`
- Excluded by design: `support_gap_window_probcover_v1`
- No GPU, no TS2Vec retraining, no large neural downstream training

## Primary Result

Lower mean held-out forecast MSE is better.

| rank | policy | final K=4 mean after MSE | mean relative MSE reduction |
|---:|---|---:|---:|
| 1 | `window_kcenter_v1` | 0.118593614 | 0.086395388 |
| 2 | `ts2vec_kcenter_v1` | 0.121089793 | 0.078833143 |
| 3 | `submitted_full_replay_v1` | 0.121103872 | 0.078781829 |
| 4 | `quality_stratified_random_v1` | 0.122818619 | 0.061063806 |
| 5 | `quality_only_v1` | 0.125936982 | 0.049101767 |

The decision endpoint is mean held-out downstream forecast MSE at `K=4`, so `window_kcenter_v1` is the current downstream-confirmed selector.

## Diagnostic Split

Proxy coverage disagrees with the downstream endpoint: `ts2vec_kcenter_v1` wins the proxy coverage report, while `window_kcenter_v1` wins the downstream forecast endpoint by mean MSE.

Use this split conservatively:

- Proxy coverage remains useful for diagnosis and screening.
- It is not the selector objective for this decision.
- TS2Vec is not nonsense, but it should remain a feature source / ablation / proxy signal rather than the lead active-learning policy.

## ProbCover Decision

`support_gap_window_probcover_v1` stays killed.

It was scientifically cleaner than an ad hoc density tweak, and its selections were hygiene-clean, but it did not beat `window_kcenter_v1` on the downstream forecast canary. Do not tune it for days, rescue it with a new metric, or treat medoid-like behavior as sufficient.

## Current Recommendation

Freeze selector status as:

- Lead selector for follow-up work: `window_kcenter_v1`
- Submitted-system comparator: `submitted_full_replay_v1`
- Feature/proxy ablation: `ts2vec_kcenter_v1`
- Lower-bound controls: `quality_stratified_random_v1`, `quality_only_v1`
- Archived failed policy: `support_gap_window_probcover_v1`

Stop adding selector policies against this target. Do not start large neural downstream training yet. Any future run should be framed as confirmation of this frozen decision or as a separately preregistered research question, not as open-ended policy search.
