# Hidden-Test Runner Upgrade

Date: 2026-05-03

## What Changed

Added a local hidden-test preparation path:

```text
python -m marginal_value.active.run_hidden_test
```

This creates a self-contained run directory for arbitrary old/new manifest
files. The package contains copied manifests, Modal stage configs, an executable
`commands.sh`, and a `README_hidden_test.md`.

The generated pipeline is:

```text
old manifest + new manifest
  -> exact window-stat shard build for supplied old/new clips
  -> frozen TS2Vec embedding precompute for supplied new clips
  -> partial-TS2Vec / exact-window blended k-center ranking
  -> artifact-aware hygiene rerank
  -> final evaluator-facing CSV package
```

## Why This Is A Scientific Upgrade

Before this change, the promoted method was supported by current artifact paths
and Modal configs, but it was not packaged as a reproducible hidden-test
procedure. That made the system hard for another reviewer to rerun on a fresh
old/new manifest pair.

This upgrade makes the claim sharper:

- the old/new manifests are explicit inputs;
- generated configs are bound to those manifests;
- the exact window-stat support view is rebuilt for the supplied old corpus;
- the expected new-query TS2Vec shard directory is derived deterministically
  from the supplied new manifest and checkpoint metadata;
- the runner documents that it does not use hidden targets, labels, candidate
  roles, or evaluator feedback.

## What It Does Not Claim

This is not heavy training and not a new model.

It still does not make TS2Vec a full-support exact search. The TS2Vec
old-support side remains the frozen partial cache. The window-stat old-support
side is exact for the supplied old manifest.

Correct claim:

```text
Partial-TS2Vec / exact-window blended k-center selector with artifact-aware
trace rerank, packaged as a manifest-bound hidden-test runner.
```

Avoid:

- validated TS2Vec active learning;
- exact full-200k TS2Vec novelty;
- full-corpus learned marginal value.

## Example

```bash
python -m marginal_value.active.run_hidden_test \
  --old-manifest /path/to/pretrain_urls.txt \
  --new-manifest /path/to/new_urls.txt \
  --run-dir artifacts/hidden_test_run \
  --run-id external_eval_001

cd artifacts/hidden_test_run
./commands.sh
```

If the prepared run directory is outside the git checkout, run:

```bash
REPO_ROOT=/path/to/active-learning-v2 ./commands.sh
```

Primary package output:

```text
artifacts/hidden_test_run/final_package/ranked_new_clips.csv
```

Backup ID-format output:

```text
artifacts/hidden_test_run/final_package/ranked_new_clips_worker_id.csv
```

## Validation

The local package preparation is covered by:

```text
tests/test_active_run_hidden_test.py
```

The tests check manifest validation, duplicate rejection, generated config
paths, Modal commands, final package config wiring, and the explicit
no-hidden-targets documentation.
