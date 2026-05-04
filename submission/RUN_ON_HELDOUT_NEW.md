# Running On Held-Out New Clips

This file is for the likely evaluation mode:

```text
same public 200,000-clip old corpus + a different hidden new-candidate manifest
```

In that case, we do not need to retrain TS2Vec and we do not need to recompute
old-support TS2Vec. The existing exact 200k old-support TS2Vec cache can be
reused. The run only needs to:

1. cache the hidden new clips;
2. compute TS2Vec/window embeddings for the hidden new clips;
3. rebuild exact window support for the supplied manifests;
4. rank the hidden new clips;
5. package the CSV.

Prepare a run directory:

```bash
python -m marginal_value.active.run_hidden_test \
  --old-manifest /path/to/pretrain_urls.txt \
  --new-manifest /path/to/hidden_new_urls.txt \
  --run-dir artifacts/hidden_test_run \
  --run-id eddy_hidden_new
```

Validate the generated run package before spending remote compute:

```bash
python -m marginal_value.active.run_hidden_test \
  --run-dir artifacts/hidden_test_run \
  --validate-only
```

Then run:

```bash
cd artifacts/hidden_test_run
REPO_ROOT=/path/to/active-learning-v2 ./commands.sh
```

Primary output:

```text
artifacts/hidden_test_run/final_package/ranked_new_clips.csv
```

Backup exact-window-only baseline:

```text
artifacts/hidden_test_run/final_package_exact_window_old_novelty/ranked_new_clips.csv
```

## Important Assumption

The exact TS2Vec old-support cache is exact for the public 200,000-clip old
corpus. If the evaluator changes the old corpus itself, then exact TS2Vec
support for that new old corpus requires a fresh old-support TS2Vec precompute.

If the old corpus changes and no TS2Vec precompute is allowed, use the
exact-window-only baseline emitted by the same runner.
