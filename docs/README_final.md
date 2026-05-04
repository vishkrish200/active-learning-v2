# Final IMU Ranking Package

Date: 2026-05-04

## Status

The current evaluator-facing package target is:

```text
artifact-gate exact-full TS2Vec / exact-window blend
```

This supersedes the earlier partial-TS2Vec / exact-window package. The new
package ranks all 2,000 new clips using both exact old-support views:

```text
TS2Vec old support: 200000 / 200000 clips
window-stat old support: 200000 / 200000 clips
new candidates: 2000 / 2000 clips
```

It is chosen for top-ranked trust, artifact avoidance, and removal of the
partial TS2Vec support caveat. It should still not be described as validated
clean TS2Vec active learning or as a learned active-acquisition policy.

## Package Command

```bash
python -m marginal_value.active.run_final \
  --config-path configs/final_package_exact_full_ts2vec_artifact_gate.json
```

This command packages existing promoted artifacts. It does not launch Modal,
train a model, or recompute embeddings.

The default output directory is:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate
```

## Primary Submission File

Use:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips.csv
```

Clean sendable copy:

```text
submission/ranked_new_clips.csv
```

This file uses `new_worker_id` as the primary ID column. Keep these backup
variants in the package:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips_new_worker_id.csv
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips_worker_id.csv
```

The top-level `submission/` folder is the package to send if the evaluator only
needs the ranked CSV and supporting reports for the provided 2,000 clips.

## Expected Package Files

```text
ranked_new_clips.csv
ranked_new_clips_new_worker_id.csv
ranked_new_clips_worker_id.csv
diagnostics.csv
selector_config.json
feature_schema.json
selector_report.json
exact_selector_report.json
validation_report.json
final_package_report.json
README_final.md
```

`selector_report.json` is the artifact-gate hygiene report.
`exact_selector_report.json` is the exact full-support TS2Vec/window ranking
report.

## Source Artifacts

The package is built from the promoted artifact-gate rerank:

```text
/artifacts/active/final_blend_rank/exact_full_ts2vec_window_a05/artifact_hygiene_ablation/
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

> We rank new IMU clips with an exact full-support TS2Vec / exact-window
> blended k-center selector with artifact-aware trace rerank. The method
> combines TS2Vec novelty against all 200,000 old-support clips with exact
> full-corpus window-stat novelty, applies quality and physical-validity gates,
> uses k-center-style redundancy control, and demotes likely sensor artifacts.

Avoid these claims:

- validated TS2Vec active learning
- full-corpus learned marginal value
- learned active-acquisition policy
- semantic workflow discovery

## Evidence Files

Core evidence:

```text
submission/README.md
submission/METHODS.md
submission/RUN_ON_HELDOUT_NEW.md
docs/exact_full_ts2vec_window_results_2026-05-04.md
docs/gcp_ts2vec_embedding_run_2026-05-04.md
docs/artifact_gate_active_loop_eval_2026-05-03.md
docs/active_loop_validation_report_2026-05-03.md
docs/exact_full_window_results_2026-05-02.md
docs/exact_window_topk_audit_2026-05-03.md
docs/final_submission_2026-05-02.md
```

The most important result is now two-part:

1. Exact full-support TS2Vec/window ranking stays highly consistent with the
   previous artifact-gated package, so removing partial TS2Vec support did not
   destabilize the ranking.
2. Artifact gating removes likely artifact and spike selections from the top
   prefixes while preserving most of the exact-full ranking.

Current exact-full comparison against the previous final package:

```text
top10_overlap: 0.900
top50_overlap: 0.940
top100_overlap: 0.970
rank_correlation: 0.9572
```

Artifact-gated top-50 hygiene:

```text
quality_fail_rate: 0.000
physical_fail_rate: 0.000
spike_fail_rate: 0.000
trace_artifact_fail_rate: 0.000
unique_clusters: 50
```

## Hidden-Test Run Preparation

For a fresh old/new manifest pair, prepare a manifest-bound Modal run package
with:

```bash
python -m marginal_value.active.run_hidden_test \
  --old-manifest /path/to/pretrain_urls.txt \
  --new-manifest /path/to/new_urls.txt \
  --run-dir artifacts/hidden_test_run \
  --run-id external_eval_001
```

Then inspect `artifacts/hidden_test_run/README_hidden_test.md` and run
`artifacts/hidden_test_run/commands.sh` when ready. If the run directory is
outside the git checkout, set `REPO_ROOT=/path/to/active-learning-v2` before
running the script.

Important distinction: the exact-full TS2Vec package above is exact for the
provided public old corpus because all 200,000 old-support TS2Vec embeddings
were precomputed. A truly arbitrary held-out old manifest would need its own
old-support TS2Vec precompute, or should use the exact-window fallback path.

Before launching Modal, run the local self-check:

```bash
python -m marginal_value.active.run_hidden_test \
  --run-dir artifacts/hidden_test_run \
  --validate-only
```

## Known Limitations

- This is exact full-support for both TS2Vec and window-stat old-support views
  on the provided public old corpus.
- The fixed-crop TS2Vec code path exists, but the fixed-crop checkpoint is not
  promoted.
- The learned ranker remains diagnostic only.
- For arbitrary new old-corpus manifests, exact full-support TS2Vec requires a
  fresh old-support TS2Vec precompute.
- The external held-out evaluator remains the real test.

## Package Validation

After packaging, validate the output directory:

```bash
python -m marginal_value.active.run_final \
  --config-path configs/final_package_exact_full_ts2vec_artifact_gate.json \
  --validate-only
```

The validation checks:

- the primary CSV has exactly 2,000 rows;
- ranks are contiguous from 1 to 2,000;
- IDs are nonblank and unique;
- the backup worker-ID CSV has the same row count;
- diagnostics and package report are present.
