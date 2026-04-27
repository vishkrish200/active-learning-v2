# Submission Contract

Date: 2026-04-28

## Claim

The algorithm ranks newly arrived 3-minute IMU clips by quality-gated novelty
relative to old-corpus 3-minute support segments.

## Non-Claim

This is not a learned active-learning policy, not a downstream task model, and
not proof of semantic behavior discovery. It is a deterministic novelty-ranking
function for the external challenge.

## Frozen Candidate

```text
window_shape_stats_q85_stat90_abs60_clustercap2
```

## Selector

```text
old support traces -> non-overlapping 180-second support clips
new manifest rows -> candidate clips
representation -> window_shape_stats
old-support novelty -> mean cosine distance to k=5 nearest old support clips
quality gate -> quality_score >= 0.85
physical gate -> stationary_fraction <= 0.90 and max_abs_value <= 60.0
new-batch diversity -> max 2 early selections per new_cluster_id
tie-breaks -> quality descending, sample_id ascending
```

## Hidden-Test Interface

```bash
python3 -m marginal_value.select \
  --old-support pretrain_paths_or_urls.txt \
  --candidate-pool new_paths_or_urls.txt \
  --output ranked_candidates.csv
```

Inputs can be CSV manifests with `sample_id,raw_path` or one-path/URL-per-line
text manifests. The selector must not receive labels, hidden target clips,
evaluation embeddings, or target-source metadata.

## Primary Controls

```text
quality_only
old_novelty_only
quality_gated_random_clustercap2
raw_shape_specialist
temporal_order_specialist
kcenter_greedy_quality_gated
```

## Failure Condition

If the frozen candidate loses temporal/window-shape coverage to
`old_novelty_only` or quality-gated random across repeated folds, do not claim
balanced marginal coverage. If it only wins in `window_shape_stats`, report it
as representation-specific novelty ranking.

If the frozen candidate ties `kcenter_greedy_quality_gated`, describe the method
as quality-gated geometric coverage ranking rather than a uniquely superior
old-novelty rule.
