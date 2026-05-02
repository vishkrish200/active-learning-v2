# Acquisition Ranker Feature Contract

Date: 2026-04-28

This contract defines what a learned active-acquisition ranker may use after
the source-blocked episode evaluator and marginal-gain labeler have run. The
ranker target may come from simulated hidden-target gain labels, but the input
features must be deployable for real challenge inference.

## Scientific Status

The active-acquisition pipeline now has:

- full cached old/new registry coverage
- source-blocked simulated acquisition episodes
- baseline policy evaluation against hidden-target coverage gain
- oracle greedy evaluation as an upper bound
- candidate-level marginal-gain labels

This is sufficient substrate for a first supervised ranker, provided training
uses only deployable features and validation is active-loop coverage gain on
held-out episodes.

## First Training Target

Use this as the first target:

```text
balanced_gain
```

This is the candidate's solo marginal coverage gain averaged across the active
representations. It is simpler and less prefix-dependent than greedy-oracle
imitation.

Secondary targets for later experiments:

```text
balanced_relative_gain
balanced_gain_after_greedy_prefix
balanced_relative_gain_after_greedy_prefix
```

The oracle-prefix labels are valid for an oracle-imitation experiment, but they
should not be the first promoted training target because inference prefixes are
chosen by the model or deployable reranker, not by the oracle.

## Allowed Features

Allowed inputs are features available at real inference time from:

- the old support corpus
- the candidate batch being ranked
- the candidate clip itself

Allowed feature families:

```text
quality_score
stationary_fraction
max_abs_value
missingness / parse-quality features
old_novelty_window_mean_std_pool
old_novelty_temporal_order
old_novelty_raw_shape_stats
old_novelty_window_shape_stats
candidate_batch_density
nearest_candidate_distance
new_cluster_size
distance_to_candidate_cluster_medoid
duplicate_score
within_clip_motion_diversity
representation agreement / disagreement features
quality-gated novelty interactions
```

Identifiers may be used only for grouping, splitting, diagnostics, and
duplicate suppression logic:

```text
episode_id
source_group_id
worker_id
sample_id
```

They must not be direct predictive model inputs.

## Forbidden Features

These must never be used as ranker inputs:

```text
candidate_role
heldout_novel flag
known_like flag
near_duplicate role flag
low_quality role flag
hidden_target membership
target source group membership
candidate-target distance
support-target distance
coverage_before
coverage_after
gain_if_added_alone
balanced_gain
balanced_relative_gain
gain_after_greedy_prefix
oracle prefix rank
oracle selected flag
rank from any evaluation-only oracle
URL text
raw source path text
split label as a predictive feature
```

Candidate roles are diagnostics created by the simulator. They are useful for
auditing whether episodes are mixed and non-degenerate, but they are not
available in the external challenge.

## Split Discipline

Do not randomly split candidate rows.

Minimum split:

```text
train episodes: 48
validation episodes: 8
test episodes: 8
```

Preferred split:

```text
heldout source groups disjoint across train, validation, and test episodes
```

If a model is trained on all 64 current episodes, it is a development model
only. Promotion requires held-out active-loop evaluation.

## Promotion Metric

Do not promote a ranker based on RMSE or AUC alone.

The promoted metric is:

```text
model-ranked top-K -> hidden-target coverage gain
```

Evaluate at:

```text
K = 5, 10, 25, 50, 100
```

Representations:

```text
window_mean_std_pool
temporal_order
raw_shape_stats
window_shape_stats
balanced aggregate
```

Required baselines:

```text
random_valid
quality_only
old_novelty_only
kcenter_greedy_quality_gated
window_shape_stats_q85_stat90_abs60_clustercap2
oracle_greedy_eval_only
```

The oracle is evaluation-only and is not a deployable selector.

## Acceptance Gate

A learned policy is credible only if held-out active-loop evaluation shows:

- balanced gain beats `window_shape_stats_q85_stat90_abs60_clustercap2`
- balanced gain beats or complements `kcenter_greedy_quality_gated`
- artifact and low-quality rates stay controlled
- duplicate rate does not exceed the simple diversity baselines at small K
- gains are not explained only by `raw_shape_stats`
- oracle remains above deployable policies, proving nonzero headroom

If the learned policy does not beat k-center or the frozen window-shape
baseline, the honest final method should be a simple or hybrid geometric
policy rather than a learned ranker.
