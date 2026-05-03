# Final IMU Ranking Package

Date: 2026-05-03

## Status

The current evaluator-facing package target is the artifact-gate exact-window
blend:

```text
artifact-gate exact-window blend
```

This is the conservative primary selector. It is chosen for top-ranked trust and
artifact avoidance, not because it strictly dominates the plain exact-window
blend on coverage at every K.

## Package Command

```bash
python -m marginal_value.active.run_final \
  --config-path configs/final_package_artifact_gate.json
```

This command packages existing promoted artifacts. It does not launch Modal,
train a model, or recompute embeddings.

The default output directory is:

```text
artifacts/final_selector/artifact_gate_exact_window
```

## Expected Package Files

```text
ranked_new_clips.csv
ranked_new_clips_new_worker_id.csv
ranked_new_clips_worker_id.csv
diagnostics.csv
selector_config.json
feature_schema.json
selector_report.json
support_audit.json
stability_report.json
validation_report.json
final_package_report.json
README_final.md
```

Use `ranked_new_clips.csv` as the primary submission file. It is equivalent to
the `new_worker_id` variant and is the clearest external-facing name.

Keep `ranked_new_clips_worker_id.csv` as a backup ID-format file.

## Source Artifacts

The package is built from the promoted artifact-gate rerank:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/
```

Primary source CSV:

```text
spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv
```

Backup worker-ID CSV:

```text
spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv
```

Diagnostics:

```text
spike_hygiene_ablation_artifact_gate_diagnostics_full.csv
```

## Method Claim

Use this wording:

> We rank new IMU clips with a partial-TS2Vec / exact-window blended k-center
> selector with artifact-aware trace rerank. The method combines TS2Vec novelty
> against a partial old-support cache with exact full-corpus window-stat novelty,
> applies quality and physical-validity gates, uses k-center-style redundancy
> control, and demotes likely sensor artifacts.

Avoid these claims:

- validated TS2Vec active learning
- full-corpus learned marginal value
- exact full-200k TS2Vec novelty
- semantic workflow discovery

## Evidence Files

Core evidence:

```text
docs/final_submission_2026-05-02.md
docs/artifact_gate_active_loop_eval_2026-05-03.md
docs/active_loop_validation_report_2026-05-03.md
docs/exact_full_window_results_2026-05-02.md
docs/exact_window_topk_audit_2026-05-03.md
```

The most important validation result is that artifact-gate removes likely
artifact and spike selections while staying close to the plain blend at
K=10/K=25. It trails the plain blend at K=50/K=100, so the correct conclusion is
cleaner conservative primary selector, not strict coverage winner.

## Known Limitations

- This is not an exact full-200k TS2Vec search.
- The TS2Vec old-support view is partial.
- The window-stat old-support view is exact full-support.
- The fixed-crop TS2Vec code path exists, but the fixed-crop checkpoint is not
  promoted.
- The learned ranker remains diagnostic only.
- The external held-out evaluator remains the real test.

## Package Validation

After packaging, validate the output directory:

```bash
python -m marginal_value.active.run_final \
  --config-path configs/final_package_artifact_gate.json \
  --validate-only
```

The validation checks:

- the primary CSV has exactly 2,000 rows;
- ranks are contiguous from 1 to 2,000;
- IDs are nonblank and unique;
- the backup worker-ID CSV has the same row count;
- diagnostics and package report are present.
