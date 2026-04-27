# External Hidden-Test Protocol

Date: 2026-04-27

## Purpose

This protocol gives an external evaluator a clean way to test the current
selector without exposing hidden targets, labels, or evaluation embeddings.

The current defensible claim is narrow:

```text
Select high-quality IMU clips that expand raw-shape/distributional coverage relative to old support.
```

The selector should not be described as proven active learning or general
marginal data value until a hidden/downstream test establishes that claim.

## Inputs

Provide two CSV manifests:

```text
old_support.csv
candidate_pool.csv
```

Required columns:

```text
sample_id,raw_path
```

Optional column:

```text
source_group_id
```

`raw_path` can be absolute or relative to the manifest location. Each raw file
may be either:

- JSONL with `acc` and `gyro` arrays, or six numeric IMU channels.
- CSV with six numeric IMU channels and optional timestamp column.

Do not provide the hidden target set to the selector.

## Command

```bash
python3 -m marginal_value.select \
  --old-support old_support.csv \
  --candidate-pool candidate_pool.csv \
  --output ranked_candidates.csv
```

Equivalent installed command:

```bash
marginal-value select \
  --old-support old_support.csv \
  --candidate-pool candidate_pool.csv \
  --output ranked_candidates.csv
```

## Frozen Default Method

```text
representation: raw_shape_stats
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
old-support novelty: mean cosine distance to k=5 nearest old-support clips
candidate diversity: cap early selections at 2 per new_cluster_id
cluster threshold: cosine similarity >= 0.985
tie-break: quality descending, sample_id ascending
```

## Output

The output CSV includes:

```text
sample_id
rank
score
quality_score
old_novelty_score
quality_gate_pass
physical_validity_pass
physical_validity_failure_reasons
stationary_fraction
max_abs_value
new_cluster_id
new_cluster_size
was_selected_by_source_cap_fallback
reranker
raw_path
```

The evaluator can then measure hidden-target coverage or downstream task
utility using representations and metrics the selector did not control.

## Recommended Hidden Evaluation

The strongest next test is external and representation-independent:

1. The evaluator supplies old support and candidate pool manifests.
2. The selector emits `ranked_candidates.csv`.
3. The evaluator adds top-K selected rows to old support.
4. The evaluator measures held-out target improvement using private
   representations and/or downstream task metrics.

Important:

```text
The selector must not receive hidden targets, labels, evaluation embeddings, or target-source metadata.
```

If the selector only improves the same raw-shape representation it ranks on,
the supported claim remains raw-shape coverage only.
