# Scientific Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current budgeted TS2Vec/window submission candidate into a cleaner scientific result by fixing crop provenance, revalidating fixed TS2Vec, measuring support-sampling stability, and only then deciding whether full-support/data-shard work is justified.

**Architecture:** Keep the current blend selector as the fallback artifact. Make one scientifically meaningful change at a time, write artifacts under new names, and compare every new result against the current checkpoint/output before promotion.

**Tech Stack:** Python, NumPy, PyTorch-in-Modal, Modal volumes, existing active-loop evaluation and embedding cache.

---

### Task 1: Canonical TS2Vec Crop Sampler

**Files:**
- Modify: `marginal_value/models/ts2vec_loss.py`
- Modify: `marginal_value/training/train_ts2vec.py`
- Modify: `tests/test_ts2vec_pipeline.py`
- Modify: `docs/final_submission_2026-05-02.md`
- Modify: `docs/active_learning_handoff_2026-05-02.md`

- [ ] **Step 1: Write failing tests**

Add tests that call `create_overlapping_crops()` on a monotonic synthetic clip and assert:

```python
left, right, left_idx, right_idx = create_overlapping_crops(
    values,
    min_overlap=0.5,
    crop_min_len=6,
    crop_max_len=6,
    rng=_FixedRng([0, 3]),
)
self.assertFalse(np.array_equal(left, right))
np.testing.assert_array_equal(left[left_idx, 0], right[right_idx, 0])
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv/bin/python -m unittest tests.test_ts2vec_pipeline.TS2VecPipelineTests.test_create_overlapping_crops_returns_distinct_aligned_views
```

Expected: fail because the current helper returns identical crops.

- [ ] **Step 3: Implement canonical sampler**

Update `create_overlapping_crops()` to return `(left_crop, right_crop, left_overlap_indices, right_overlap_indices)` using two shifted starts with guaranteed overlap. Route `_two_overlapping_fixed_crops()` through this helper to avoid two implementations.

- [ ] **Step 4: Verify green**

Run:

```bash
.venv/bin/python -m unittest tests.test_ts2vec_pipeline
.venv/bin/python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit crop fix**

Commit message:

```bash
git commit -m "Fix TS2Vec crop sampler provenance"
```

### Task 2: Bounded Fixed-Crop TS2Vec Training

**Files:**
- Create: `configs/ts2vec_fixed_crop_smoke_h100.json` if a config wrapper is needed
- Modify: `docs/final_submission_2026-05-02.md`

- [ ] **Step 1: Run bounded H100 training**

Run:

```bash
.venv/bin/modal run modal_train_ts2vec.py \
  --checkpoint-dir /artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval \
  --training-log-path /artifacts/docs/ts2vec_fixed_crops_candidate_eval_training_log.md \
  --n-epochs 3 \
  --batch-size 8 \
  --max-steps-per-epoch 150 \
  --collapse-sample-size 1000 \
  --alpha 0.1 \
  --max-temporal-positions 64
```

Expected: checkpoint writes to `/artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval/ts2vec_best.pt`, and the log contains epoch 0-3 rows.

- [ ] **Step 2: Pull and summarize log**

Run:

```bash
.venv/bin/modal volume get activelearning-imu-rebuild-cache /docs/ts2vec_fixed_crops_candidate_eval_training_log.md /tmp/ts2vec_fixed_crops_candidate_eval_training_log.md
```

Expected: record effective rank, mean pairwise cosine, instance loss, and temporal loss.

### Task 3: Fixed-vs-Current Active-Loop Comparison

**Files:**
- Create: `configs/active_loop_eval_ts2vec_fixed_window_blend_scale.json`
- Create: `docs/ts2vec_fixed_crop_eval_results.md`

- [ ] **Step 1: Add fixed-checkpoint eval config**

Copy `configs/active_loop_eval_ts2vec_window_blend_scale.json`, change only:

```json
"ts2vec_checkpoint_path": "/artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval/ts2vec_best.pt"
```

and write output to:

```text
/artifacts/active/eval/ts2vec_fixed_crop_window_blend_scale
```

- [ ] **Step 2: Run smoke eval first**

Run the active-loop Modal entrypoint with the fixed config and no full flag. Expected: two-episode smoke completes.

- [ ] **Step 3: Run full eval if smoke is sane**

Run the full fixed-checkpoint active-loop eval. Compare K=10 and K=50 balanced relative gain against the current blend table.

### Task 4: Support-Sampling Stability

**Files:**
- Create: `marginal_value/active/support_sampling_stability.py`
- Create: `modal_active_support_sampling_stability.py`
- Create: `tests/test_active_support_sampling_stability.py`
- Create: `docs/support_sampling_stability_results.md`

- [ ] **Step 1: Write failing tests**

Create a small synthetic diagnostics table and assert the stability summary reports top-K overlap and Spearman-like rank agreement across seeded support samples.

- [ ] **Step 2: Implement stability runner**

Use the final blend ranker with fixed TS2Vec partial support and varied `right_support_seed` values for the 25k window support. Compute top-10, top-50, top-100 overlap, rank correlation, hygiene, and score drift.

- [ ] **Step 3: Run 3-seed stability first**

Use seeds `[1, 2, 3]`. Promote to five seeds only if the 3-seed result is stable enough to justify extra compute.

### Task 5: Full-Support/Data-Shard Decision

**Files:**
- Create: `docs/full_support_data_shard_plan.md`

- [ ] **Step 1: Decide from evidence**

Proceed to data-shard implementation only if:

```text
fixed TS2Vec eval >= current blend at K=10 or K=50
support stability top-50 overlap is acceptable
```

- [ ] **Step 2: Plan sharded data layer**

Design large shard files with `sample_id`, `worker_id`, `source_group_id`, raw IMU or normalized arrays, and quality metadata so full 200k support can be embedded without small-file JSONL IO dominating H100 time.
