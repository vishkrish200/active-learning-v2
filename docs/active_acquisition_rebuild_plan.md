# Active Acquisition Rebuild Plan

Date: 2026-04-28

## Why This Rebuild Exists

The current selector is a solid deterministic baseline, but it is not yet a
trained active acquisition policy. Its claim is:

```text
quality-gated geometric novelty ranking over accessible old-support clips
```

The Build-style target is stronger:

```text
given an old IMU corpus and a new daily worker batch, rank new 3-minute clips
by predicted marginal value for improving future coverage of real worker,
workflow, factory, or activity patterns
```

The scientific bridge is simulated acquisition. We should use the old corpus to
create pseudo collection episodes, label candidates by actual marginal gain on
hidden targets, and train/evaluate a ranker against strong geometric controls.

## Current Data Reality

Do not describe the current support set as the full theoretical
`10,000 workers x 1 hour` corpus until the source audit proves that raw geometry
is available.

Current verified state from the refreshed Modal audits and archive scan:

```text
original pretrain manifest URLs: 200,000
manifest pretrain workers: 10,000
public GCS pretrain objects: 200,000
public GCS pretrain workers: 10,000
physical/extracted pretrain URLs found on Modal: 68,210
physical pretrain workers found on Modal: 8,483
pretrain archive URLs found on Modal: 100,000
pretrain archive workers found on Modal: 5,000
archive URLs missing from extracted physical source: 39,191
physical + archive source union: 107,401 URLs across 8,916 workers
manifest URLs not found in current Modal source mounts/archive: 92,599
new split URLs available: 2,000
cached physical-source pretrain raw+features from latest audit: 68,208
cached physical+archive pretrain raw+features from latest audit: 107,399 / 107,401
cached new raw+features from latest audit: 2,000
source tar archives visible: 1
```

The old `13,273` cached-support problem has been addressed for the
physical-source path, and the archive audit shows another `39,191` usable
pretrain clips can likely be recovered from `/source/pretrain_100k.tar.zst`.
The immediate cached support target is no longer `68,208`; the source-volume
recovery target is the `physical + archive` union, about `107,401` pretrain
clips. However, the public GCS bucket itself contains the full `200,000`
pretrain objects, so the remaining `92,599` URLs are not bad manifest rows.
They are absent only from the Modal source mirror/archive path we were using.
The full-support path should therefore download/cache directly from the manifest
URLs instead of relying only on `/source`.

Latest source inventory:

```text
command:
.venv/bin/python -m modal run --detach modal_source_inventory.py \
  --run-full \
  --skip-smoke \
  --config-path configs/source_inventory_observe_full.json

modal run:
https://modal.com/apps/vishkrish200/main/ap-VsYGO0TxykyzzpcFq6OPUR

remote report:
/artifacts/source_inventory_observe_full/source_inventory_full.json

result:
pretrain_manifest_url_count = 200,000
pretrain_source_existing_count = 68,210
physical_source_manifest_url_count = 68,210
new_manifest_url_count = 2,000
new_source_existing_count = 2,000
source_tar_count = 1
```

Latest support coverage audit:

```text
command:
.venv/bin/python -m modal run --detach modal_support_coverage_audit.py \
  --run-full \
  --skip-smoke \
  --config-path configs/support_coverage_audit_physical_source.json

modal run:
https://modal.com/apps/vishkrish200/main/ap-DsY6LqFxVHZ0esAlOZlUy0

remote report:
/artifacts/audits/support_coverage_physical_source/support_coverage_audit_full.json

result:
pretrain_manifest_url_count = 68,210
pretrain_cached_both_count = 68,208
new_manifest_url_count = 2,000
new_cached_both_count = 2,000
feature_file_count = 73,513
raw_file_count = 73,515
```

Source archive audit:

```text
command:
.venv/bin/python -m modal run --detach modal_source_archive_audit.py \
  --run-full \
  --skip-smoke \
  --config-path configs/source_archive_audit_pretrain_100k.json

modal run:
https://modal.com/apps/vishkrish200/main/ap-jASQ7nfVbdaY6WIs6UuZrp

remote report:
/artifacts/audits/source_archive_pretrain_100k/source_archive_audit_full.json

local report:
data/audits/source_archive_pretrain_100k/source_archive_audit_full.json

result:
archive_data_member_count = 100,000
archive_unique_url_count = 100,000
archive_manifest_match_count = 100,000
archive_extracted_duplicate_count = 60,809
archive_missing_from_extracted_count = 39,191
archive_not_in_manifest_count = 0
archive_worker_count = 5,000
archive_worker_clip_count = 20 per worker
```

Current action: cache the `39,191` archive-only clips into `/data/cache/raw` and
`/data/cache/features`, then write:

```text
cache/manifests/pretrain_archive_cached_urls.txt
cache/manifests/pretrain_physical_plus_archive_urls.txt
```

Use `pretrain_physical_plus_archive_urls.txt` as the support manifest for the
episode generator and marginal-gain labeler once the cache run and coverage
audit pass.

Source archive cache and union coverage:

```text
cache run:
https://modal.com/apps/vishkrish200/main/ap-sFmYyYe7IBxioOiR3CsChQ

coverage run:
https://modal.com/apps/vishkrish200/main/ap-tF23lOxIhb9Z3CLf6Lzepf

result:
pretrain_physical_plus_archive_urls = 107,401
pretrain_cached_both = 107,399
new_cached_both = 2,000
missing_feature_rows = 2
malformed_archive_rows = 0
```

The two missing feature rows are physical-source URLs that already had raw JSONL
but no feature NPZ. They should be repaired opportunistically, but they are not
a scientific blocker for the episode generator.

Direct GCS source check:

```text
bucket:
buildai-imu-benchmark-v1-preexisting

prefix:
pretrain/

listing result:
object_count = 200,000
worker_count = 10,000
worker_clip_hist = 20 clips for each worker

sampled URL probe:
missing_entire_worker = 80/80 returned HTTP 206
missing_partial_worker = 80/80 returned HTTP 206
archive_only_present = 20/20 returned HTTP 206
physical_present = 20/20 returned HTTP 206
```

Interpretation:

```text
The Eddy/GCS pretrain manifest is consistent with the public bucket. The current
Modal /source mirror is incomplete. The next cache expansion should fetch
missing URLs directly from GCS.
```

## GPU And Modal Policy

Local CPU work should be limited to tests, config validation, and small smoke
runs. Heavy work runs on Modal.

Use GPU where it actually helps:

| Work | Compute policy |
| --- | --- |
| Source inventory and cache coverage audits | Modal CPU, because this is filesystem I/O and JSON/NPZ bookkeeping. |
| kNN novelty, coverage distance, k-center, oracle greedy | Modal GPU with Torch/CUDA backend. |
| Episode marginal-gain labeling | Modal GPU, because it repeats support-target and candidate-target distance matrices. |
| Learned ranker training | Modal GPU if using Torch or GPU LightGBM; Modal CPU is acceptable only for tiny smoke tests. |
| Final ranking over full candidate batch | Modal GPU for kNN/index scoring, CPU threads only for raw loading/parsing. |

Implementation note: cosine kNN and mean nearest-neighbor coverage distance now
route through `marginal_value.indexing.cosine_search`, which uses Torch CUDA
when available and falls back to NumPy locally.

## Frozen Baselines

Keep these as mandatory controls:

```text
random_valid
quality_only
old_novelty_only_window_shape
old_novelty_only_temporal_order
old_novelty_only_raw_shape
kcenter_greedy_quality_gated
window_shape_stats_q85_stat90_abs60_clustercap2
```

The current frozen selector remains useful, but only as:

```text
best deterministic baseline
```

not:

```text
final learned active-learning solution
```

## Target Architecture

```text
accessible old corpus
  -> clip-level feature store
  -> pseudo-active acquisition episodes
  -> marginal-gain labels
  -> learned acquisition ranker
new daily batch
  -> quality features
  -> multi-view old-novelty features
  -> new-batch cluster/coherence features
  -> predicted marginal value
  -> greedy redundancy-aware reranking
  -> ranked 2,000 worker clips
```

Selector code must not import eval code. Training/evaluation modules may import
selector and feature code, but final ranking must not depend on hidden targets,
labels, or evaluation representations.

## Phase 1: Finish Data Truth

Deliverables:

```text
data/source inventory report
support coverage report
short docs update with accessible support count
decision: cache archive-only clips, then build against the physical+archive union
```

Acceptance criteria:

- The source inventory plus archive audit explain how much of the gap between
  `200,000` manifest URLs and `68,210` accessible physical-source URLs can be
  recovered from `/source/pretrain_100k.tar.zst`.
- The support coverage report confirms raw+feature parity for accessible old and
  new clips.
- The project explicitly states that current source mounts/archive expose
  `107,401` pretrain URLs, while `92,599` manifest URLs remain unavailable.

No learned ranker work should happen before this is clear.

## Phase 2: Build Episode Generator

Add:

```text
marginal_value/active/episodes.py
tests/test_active_episodes.py
```

Episode schema:

```text
episode_id
seed
support_clip_ids
candidate_clip_ids
hidden_target_clip_ids
distractor_clip_ids
heldout_source_groups
known_source_groups
```

Candidate batches must mix:

- known-like high-quality clips from support-like groups
- held-out novel-group clips
- near-duplicates
- low-quality/artifact-like clips
- stationary or low-information clips

Acceptance criteria:

- No worker/source group leaks across support and hidden target.
- Candidate pools contain both known-like and held-out-like clips.
- Episode generation is deterministic for a fixed seed.

## Phase 3: Label Marginal Gain

Add:

```text
marginal_value/active/label_gain.py
modal_active_label_gain.py
tests/test_active_label_gain.py
```

Candidate label:

```text
gain(c) =
  coverage_error(hidden_target | support_old)
  -
  coverage_error(hidden_target | support_old union {c})
```

Primary label should be multi-view, with per-view labels retained:

```text
gain_window_shape
gain_temporal_order
gain_raw_shape
gain_window_mean_std
gain_balanced
```

The GPU path matters here because this phase repeatedly computes nearest-neighbor
coverage distances. It should use `marginal_value.indexing.cosine_search`.

Acceptance criteria:

- Labels are zero or near-zero for redundant candidates in simple synthetic
  tests.
- Labels are positive for candidates close to hidden targets but far from
  support in synthetic tests.
- Modal full runs are detached and write JSON/Parquet artifacts under
  `/artifacts/active/labels`.

## Phase 4: Train Acquisition Ranker

Add:

```text
marginal_value/active/features.py
marginal_value/active/train_ranker.py
modal_active_train_ranker.py
```

Start with interpretable features:

```text
quality_score
stationary_fraction
max_abs_value
missing_fraction
flatline_fraction
saturation_fraction
spike_rate
old_novelty_window_shape
old_novelty_temporal_order
old_novelty_raw_shape
old_novelty_window_mean_std
new_cluster_size
new_cluster_compactness
nearest_new_neighbor_distance
distance_to_new_cluster_medoid
within_clip_motion_diversity
temporal_transition_entropy
```

Preferred models:

```text
LightGBM LambdaRank or regression on marginal gain
fallback: sklearn gradient-boosted regressor
```

Acceptance criteria:

- Train/validation split is by episode and source group, never random rows from
  the same worker/source.
- Learned ranker beats quality-only, old-novelty-only, k-center, and the frozen
  window-shape baseline on held-out simulated episodes before promotion.

## Phase 5: Active Loop Evaluation

Add:

```text
marginal_value/active/evaluate_active_loop.py
modal_active_loop_eval.py
```

Policies:

```text
random_valid
quality_only
old_novelty_only_window_shape
old_novelty_only_temporal_order
old_novelty_only_raw_shape
kcenter_greedy_quality_gated
window_shape_stats_q85_stat90_abs60_clustercap2
learned_acquisition_ranker
oracle_greedy
```

Budgets:

```text
K = 10, 25, 50, 100, 200, 400
```

Metrics:

```text
hidden_target_coverage_gain
temporal_order_gain
window_shape_gain
raw_shape_gain
unique_hidden_cluster_recall
artifact_rate@K
duplicate_rate@K
new_cluster_diversity@K
worker/source_diversity@K
distance_to_oracle
```

Promotion rule:

```text
Promote the learned policy only if it beats k-center and the frozen
window-shape baseline on primary held-out acquisition curves without worsening
artifact or duplicate rates.
```

If it fails, report the best validated simple baseline instead of adding more
heuristics.

## Phase 6: Final Daily Batch Ranking

Add:

```text
marginal_value/active/run.py
modal_active_rank_daily_batch.py
```

Expected command:

```bash
python -m marginal_value.active.run \
  --old pretrain_urls.txt \
  --new new_urls.txt \
  --output ranked_new_clips.csv \
  --diagnostics diagnostics.csv \
  --artifact-dir artifacts/build_active_policy
```

Output columns:

```text
worker_id
sample_id
rank
score
predicted_gain
quality_score
artifact_risk
old_novelty_window_shape
old_novelty_temporal_order
old_novelty_raw_shape
new_cluster_id
new_cluster_size
redundancy_penalty
reason_code
```

## Immediate Next Actions

1. Add a direct-GCS cache job for manifest URLs missing from
   `pretrain_physical_plus_archive_urls.txt`.
2. Cache the remaining `92,599` pretrain URLs and run support coverage against
   the full `pretrain_urls.txt` manifest.
3. Implement `marginal_value/active/episodes.py` with tests, using the full
   cached manifest if the direct-GCS cache passes.
4. Implement GPU-backed marginal-gain labeling on Modal.
5. Run active-loop validation against the frozen baselines.

The next modeling code change after the cache/coverage step should be the
episode generator, not more selector tuning.
