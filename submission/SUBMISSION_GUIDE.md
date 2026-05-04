# Submission Guide

## What To Send

Send the `submission/` package or the zip made from it.

Primary file:

```text
ranked_new_clips.csv
```

Support files:

```text
README.md
METHODS.md
RUN_ON_HELDOUT_NEW.md
diagnostics.csv
selector_config.json
feature_schema.json
selector_report.json
exact_selector_report.json
validation_report.json
final_package_report.json
SUBMISSION_MANIFEST.json
```

Backup ID files:

```text
ranked_new_clips_new_worker_id.csv
ranked_new_clips_worker_id.csv
```

## Recommended Email Text

Subject:

```text
Active learning challenge - IMU novelty ranking submission
```

Body:

```text
Hi Eddy,

I'm sending my submission for the active-learning IMU challenge.

The attached package contains the ranked output for the public 2,000 new IMU
clips. The primary file is ranked_new_clips.csv, with columns:

rank,score,new_worker_id

The method is an exact full-support TS2Vec/window geometric acquisition
selector with quality gates, k-center redundancy control, and artifact-aware
trace reranking. It compares each new clip against all 200,000 old-support clips
in both a trained TS2Vec representation and a handcrafted window-stat
representation.

The repo also includes a held-out-new runner for the likely evaluation mode:
same public 200k old corpus plus a different hidden new-candidate manifest. See
RUN_ON_HELDOUT_NEW.md for the exact command flow. If the old corpus itself is
changed, the exact TS2Vec old-support cache needs to be recomputed, and the
runner also emits an exact-window fallback.

I frame this as a novelty-ranking / marginal-data-value acquisition selector,
not as a downstream retraining proof.

Best,
Vishnu
```

## Short Claim

Use:

```text
Exact full-support TS2Vec/window novelty-ranking selector with quality gates,
k-center redundancy control, and artifact-aware reranking.
```

Avoid:

```text
validated downstream active learning
learned query policy
semantic workflow discovery
clean TS2Vec research result
```

## Test Modes

### Public 2,000-Clip Evaluation

Evaluator loads:

```text
ranked_new_clips.csv
```

Then joins by:

```text
new_worker_id
```

and evaluates top-K prefixes.

### Held-Out New Manifest Evaluation

Evaluator or submitter runs:

```text
python -m marginal_value.active.run_hidden_test \
  --old-manifest /path/to/pretrain_urls.txt \
  --new-manifest /path/to/hidden_new_urls.txt \
  --run-dir artifacts/hidden_test_run \
  --run-id eddy_hidden_new
```

Then:

```text
python -m marginal_value.active.run_hidden_test \
  --run-dir artifacts/hidden_test_run \
  --validate-only
```

Then:

```text
cd artifacts/hidden_test_run
REPO_ROOT=/path/to/active-learning-v2 ./commands.sh
```

Primary output:

```text
artifacts/hidden_test_run/final_package/ranked_new_clips.csv
```
