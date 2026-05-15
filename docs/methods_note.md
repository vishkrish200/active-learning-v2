# Methods Note: Quality-Gated IMU Coverage Ranking

Date: 2026-04-28

## Problem

The task is to rank newly arrived IMU clips by how much useful information they
add relative to an existing IMU corpus. The practical input is an old corpus of
worker traces and a new candidate batch. The output is a deterministic ranking
of the new clips, with the highest-ranked clips intended to be clean,
nonredundant, and under-covered by the old corpus.

This method should be understood as a novelty-ranking system for an external
challenge, not as a learned active-learning policy or proof of semantic
factory/workflow discovery.

## Method

The current frozen selector is:

```text
window_shape_stats_q85_stat90_abs60_clustercap2
```

Old support traces are segmented into non-overlapping 180-second support clips.
Each new manifest row is treated as one candidate clip. For every support and
candidate clip, the selector computes fixed window-shape statistics from IMU
window features. Candidate novelty is the mean cosine distance to the five
nearest old-support clips in this feature space.

Candidate clips must pass fixed quality and physical-validity gates:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
```

Passing clips are ranked by old-support novelty, with quality and stable
sample id used as tie-breaks. To reduce duplicate-looking early selections, the
top of the ranking applies a simple cap of at most two clips per new-batch
feature cluster. The selector still emits a complete ranking of all candidates.

## Why This Is Scientifically Defensible

The core scientific claim is narrow: high-quality clips that are far from the
old corpus in fixed IMU feature space should improve coverage of held-out IMU
behavior more than random high-quality selection.

To test this, we use source-blocked marginal-coverage validation. Source groups
are withheld from the old support set, split into pseudo-new candidates and
hidden targets, and mixed with source-covered distractors. A policy is scored
by whether adding its top K candidates reduces nearest-neighbor distance to the
hidden target clips across multiple representations.

The current candidate beats `quality_gated_random_clustercap2` at K=100 and
K=200 in the primary temporal-plus-raw coverage aggregate. It is also balanced
across temporal, raw-shape, and window-shape coverage, unlike raw-shape-only
selection, which mainly improves its own representation.

## Important Limitation

The strongest geometric baseline, `kcenter_greedy_quality_gated`, is essentially
tied with the frozen selector at K=200 and K=400. Therefore, the evidence does
not support claiming that old-support novelty is uniquely superior. The honest
claim is that a simple quality-gated geometric coverage strategy is useful, and
that both old-novelty ranking and k-center-style coverage are strong members of
that family.

## Hidden-Test Interface

The evaluator should run the selector with only old support and new candidate
manifests:

```bash
python3 -m marginal_value.select \
  --old-support pretrain_paths_or_urls.txt \
  --candidate-pool new_paths_or_urls.txt \
  --output ranked_candidates.csv
```

The selector must not receive labels, hidden target clips, evaluation
embeddings, or target-source metadata.

## Output

The output CSV contains a full ranking plus audit columns:

```text
sample_id,rank,score,quality_score,old_novelty_score,
quality_gate_pass,physical_validity_pass,
physical_validity_failure_reasons,stationary_fraction,max_abs_value,
new_cluster_id,new_cluster_size,reranker,raw_path
```

This makes the ranking inspectable: low-quality or physically invalid clips can
be identified, and early cluster concentration can be audited.
