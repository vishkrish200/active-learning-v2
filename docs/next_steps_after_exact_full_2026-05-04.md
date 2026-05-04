# Next Steps After Exact Full-Support TS2Vec - 2026-05-04

## Current State

The project has crossed the main support-approximation hurdle.

Current primary package:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate
```

Primary CSV:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips.csv
```

Current honest claim:

```text
Exact full-support TS2Vec / exact-window blended k-center selector with
artifact-aware trace rerank.
```

What is now true:

- all 2,000 new candidate clips are ranked;
- TS2Vec old support covers 200,000 / 200,000 old clips;
- `window_mean_std_pool` old support covers 200,000 / 200,000 old clips;
- quality, physical-validity, redundancy, and trace-artifact gates are applied;
- top-50 post-rerank hygiene is clean;
- the exact-full ranking is close to the previous promoted package, so the
  support upgrade did not destabilize the output.

What is still not true:

- this is not a validated clean TS2Vec active-learning paper result;
- this is not a promoted learned acquisition policy;
- this does not prove downstream utility on Eddy's hidden evaluator;
- the promoted TS2Vec checkpoint is still the current checkpoint, not a newly
  retrained fixed-crop checkpoint.

## Inference From The Latest Result

The previous partial TS2Vec support cache was a credibility issue, but it was
not materially driving the ranking. Replacing it with exact full old-support
TS2Vec produced a highly similar final list:

```text
top10_overlap_vs_previous: 0.900
top50_overlap_vs_previous: 0.940
top100_overlap_vs_previous: 0.970
rank_correlation_vs_previous: 0.9572
```

That means the earlier result was not a fragile artifact of the 22k TS2Vec
support subset. The full-support upgrade mostly strengthens the claim rather
than changing the practical ranking.

The artifact gate remains important. The exact-full pre-rerank top 50 contained
one spike-fail clip and three trace-artifact-fail clips. The artifact-gated
package removes them from the top 50 while preserving 94% overlap with the
pre-rerank exact-full top 50.

## Recommended Next Actions

### 1. Submission package polish

This is the highest-return non-training work.

Build a clean `submission/` folder or archive containing:

```text
ranked_new_clips.csv
ranked_new_clips_new_worker_id.csv
ranked_new_clips_worker_id.csv
diagnostics.csv
selector_config.json
feature_schema.json
selector_report.json
exact_selector_report.json
final_package_report.json
README_final.md
```

Also include a one-page `METHODS.md` with the exact claim and limitations.

Acceptance gate:

- the package validates with `run_final --validate-only`;
- the README points only to the exact-full package;
- no stale partial-support package is described as primary.

### 2. Cold-clone reproducibility check

Before sending code to anyone, test from a fresh clone or clean worktree:

```text
install package
run unit tests
validate final package config
run the frozen simple selector on a tiny manifest fixture
prepare hidden-test run directory with --validate-only
```

This checks that a reviewer can understand and run the project without the
current messy local state.

Acceptance gate:

- simple `marginal-value select` path works from clean checkout;
- `run_final --validate-only` works when source artifacts are present;
- `run_hidden_test --validate-only` writes a coherent run package.

### 3. Clarify held-out testing assumption with Eddy

This is not code, but it affects the next engineering choice.

Ask whether the held-out evaluation uses:

```text
A. same public old corpus + hidden new candidate manifest
B. hidden old corpus + hidden new candidate manifest
C. CSV-only evaluation of the provided 2,000 public new clips
```

If A, the exact-full old-support TS2Vec cache remains useful.
If B, exact full-support TS2Vec requires another old-support precompute for the
hidden old manifest.
If C, the current `ranked_new_clips.csv` is the main deliverable.

### 4. Bigger active-loop eval for exact-full package

Run this only after package polish. It is evaluation compute, not training.

Compare:

- exact-full TS2Vec/window artifact-gate package;
- previous partial-TS2Vec artifact-gate package;
- exact-window old-novelty baseline;
- k-center quality-gated baseline;
- window-shape cluster-cap baseline;
- learned ridge as diagnostic only.

Use at least 64 episodes, and report:

- balanced relative gain;
- oracle fraction;
- artifact rate at K;
- low-quality rate at K;
- duplicate rate at K;
- episode-level win counts;
- bootstrap confidence intervals.

Acceptance gate:

- exact-full package does not regress materially against the current promoted
  package;
- top-K hygiene remains clean.

### 5. Hidden-test runner exact-TS2Vec upgrade

Do this only if Eddy expects to run code on a fresh old manifest.

Current hidden-test runner rebuilds exact window support from the supplied old
manifest and uses TS2Vec for the new side. For arbitrary old manifests, exact
full-support TS2Vec requires a support precompute stage.

Upgrade path:

- add an optional old-support TS2Vec precompute stage;
- write old-support TS2Vec shards into the run directory or artifact volume;
- wire exact-full TS2Vec/window ranking against that generated cache;
- keep exact-window old-novelty as fallback.

Cost depends directly on old-manifest size. The public 200k old-support run
took about 3.1 hours of active embedding on one L4 after setup.

### 6. Fixed-crop TS2Vec retrain

This is the real representation-learning cleanup, but it is no longer the first
blocker.

Only do this after the package/reproducibility work is clean.

Promotion rule:

- retrain using fixed overlapping crops;
- evaluate checkpoints in the active-loop, not just by rank/cosine metrics;
- replace the current checkpoint only if fixed-crop TS2Vec matches or beats the
  exact-full package without hygiene regression.

### 7. Learned ranker revisit

Do not promote LightGBM or a learned ranker yet.

Revisit only after:

- the exact-full geometric selector is frozen as a strong baseline;
- active-loop evaluation is stable;
- forbidden feature hygiene remains audited;
- the learned ranker beats exact-full geometric controls at K=10, K=25, and
  K=50 without artifact or duplicate regression.

## Recommended Order

```text
1. Package polish and submission archive
2. Cold-clone reproducibility check
3. Clarify Eddy held-out protocol
4. Bigger exact-full active-loop eval
5. Hidden-test exact-TS2Vec runner only if needed
6. Fixed-crop TS2Vec retrain
7. Learned ranker experiments
```

## Current Decision

The submission is ready as a CSV artifact for the provided public manifests.

It is not fully evaluator-proof as arbitrary-manifest code unless either:

- the evaluator uses the same public old corpus, or
- the hidden-test runner gets an old-support TS2Vec precompute stage, or
- we submit the exact-window fallback as the fully cold-runnable path.
