# Active Learning IMU Novelty Handoff

Date: 2026-05-02

## Executive Summary

This project ranks 2,000 new egocentric factory-floor IMU clips by novelty
relative to a 200,000-clip old support corpus. The current deployable output is
a budgeted approximate blend selector:

```text
blend_kcenter_ts2vec_window_mean_std_pool_a05
```

It uses TS2Vec only where it is affordable:

- all 2,000 new candidate clips
- 22,327 already-computed old-support TS2Vec embeddings from a stopped partial
  cache
- 25,000 old-support `window_mean_std_pool` embeddings as the cheap geometric
  support side

The final ranked output was produced successfully. It is not an exact
full-200k TS2Vec ranking, by design. Full-support TS2Vec was stopped because the
job was IO-bound and would have cost too much.

## Final Artifacts

Final ranked outputs are in the Modal artifacts volume:

```text
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_submission_full.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_submission_full_worker_id.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_submission_full_new_worker_id.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_diagnostics_full.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_report_full.json
```

Final run summary:

```text
mode: full
ranking_mode: budgeted_candidate_only
selector: blend_kcenter_ts2vec_window_mean_std_pool_a05
n_query: 2000
n_left_support: 22327   # partial old TS2Vec support
n_right_support: 25000  # capped old window support
```

Top-k hygiene from the final report:

| K | Quality Fail | Physical Fail | Duplicate Rate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000 | 0.000 | 0.000 | 10 |
| 50 | 0.000 | 0.000 | 0.000 | 50 |
| 100 | 0.000 | 0.000 | 0.010 | 99 |
| 200 | 0.000 | 0.000 | 0.005 | 199 |

## What Existed Before This Work

The repository already had:

- an active-acquisition pipeline in `marginal_value/active/`
- quality gating in `marginal_value/preprocessing/quality.py`
- an older SSL encoder in `marginal_value/models/ssl_encoder.py` that had
  collapsed and was not promoted
- a handcrafted `window_mean_std_pool` representation that beat the old SSL
  encoder
- Modal training/evaluation infrastructure
- an embedding cache system in `marginal_value/active/embedding_cache.py`

The key architectural fact is that handcrafted geometric representations are
still very strong for this homogeneous IMU corpus.

## TS2Vec Work Added

Main files:

```text
marginal_value/models/ts2vec_encoder.py
marginal_value/models/ts2vec_loss.py
marginal_value/models/ts2vec_inference.py
marginal_value/models/ts2vec_diagnostics.py
marginal_value/training/train_ts2vec.py
modal_train_ts2vec.py
tests/test_ts2vec_pipeline.py
```

Current encoder shape:

- input IMU clip: `(B, T, 6)`
- feature expansion from 6 channels to 14:
  - raw acc/gyro
  - acceleration magnitude
  - gyro magnitude
  - acc x gyro cross product
  - clip-centered acceleration
- nonlinear input projection to 64 dims
- 10 causal dilated temporal convolution blocks
- dilation rates: 1 through 512
- residual connection plus `LayerNorm` per block
- mean/max/std aggregation head projected to a 320-d clip embedding
- inference returns L2-normalized embeddings

Training loop details:

- self-supervised, no labels
- shorter crops: 150 to 600 timesteps
- hierarchical instance plus temporal contrastive loss
- default temporal weight `alpha=0.1`
- temporal positions capped at 64 during training
- collapse checks logged per epoch

Important reviewer note: the TS2Vec training implementation should be audited
before treating the checkpoint as a clean SSL result. In particular,
`create_overlapping_crops()` currently returns two copies of the same sampled
crop rather than two different overlapping crops. The downstream budgeted
selector output is usable as an engineering artifact, but the SSL training loop
is not yet something I would present as fully validated research code.

## TS2Vec Diagnostics And Interpretation

The original rank target of effective rank greater than 50 was probably wrong
for this corpus. Handcrafted `raw_shape_stats`, which is useful in evaluation,
itself had effective rank around 6-8 on full clips. The IMU corpus appears to
have low intrinsic dimensionality: walking, standing work, head orientation
changes, and machinery vibration dominate.

The better question became not "does TS2Vec have rank 50?" but "does TS2Vec
add a useful novelty signal compared with the window baseline?"

The answer from active-loop evaluation was yes: TS2Vec and
`window_mean_std_pool` were complementary. The blend selector was stronger than
the learned ridge ranker in held-out active-loop evaluation.

## Ranker Hygiene Work Added

Main files:

```text
marginal_value/active/label_gain.py
marginal_value/active/ranker.py
configs/active_ranker_train_scale_pretrain.json
configs/active_ranker_train_hygiene_ablation_balanced_gain.json
configs/active_ranker_train_hygiene_ablation_gated_gain.json
docs/ranker_hygiene_fix_results.md
tests/test_active_label_gain.py
tests/test_active_ranker.py
```

What changed:

- added `compute_gated_labels()`
- added `gated_balanced_gain`
- added `gated_balanced_relative_gain`
- made the ridge ranker default target `gated_balanced_gain`
- kept `balanced_gain` available for ablation
- added TS2Vec-derived novelty features when TS2Vec embeddings exist

Hygiene gate:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
```

Duplicate label-gating was tested and rejected because it zeroed 96.5% of the
label table. Duplicate control is left to the selector/reranker layer.

Ranker conclusion:

- hard-gated labels improved the pure ridge ranker
- pure ridge still failed promotion hygiene
- learned + diversity reranking was hygienic but did not beat the simple blend
- the learned ranker should remain diagnostic/future work, not the promoted
  selector

Held-out active-loop test summary from
`docs/ranker_hygiene_fix_results.md`:

| Policy | K=10 Balanced Relative Gain | K=50 Balanced Relative Gain |
| --- | ---: | ---: |
| learned ridge, raw `balanced_gain` | 0.1616 | 0.2515 |
| learned ridge, `gated_balanced_gain` | 0.1688 | 0.2533 |
| blend k-center TS2Vec/window a05 | 0.2149 | 0.3042 |
| k-center quality-gated | 0.1281 | 0.2167 |
| window-shape cluster-cap | 0.1398 | 0.2872 |

## Final Budgeted Selector Work Added

Main files:

```text
marginal_value/active/final_blend_rank.py
modal_active_final_blend_rank.py
configs/active_final_blend_rank_budget_h100.json
configs/active_embedding_precompute_ts2vec_new_only_h100.json
tests/test_active_final_blend_rank.py
tests/test_active_embedding_precompute.py
```

Supporting cache changes:

```text
marginal_value/active/embedding_cache.py
marginal_value/active/embedding_precompute.py
modal_active_embedding_precompute.py
```

What the budgeted selector does:

1. Load partial old-support TS2Vec shards from the stopped full-support attempt.
2. Embed/cache all 2,000 new clips with TS2Vec and `window_mean_std_pool`.
3. Compute/cache `window_mean_std_pool` for a capped 25,000 old-support sample.
4. Compute old-support novelty in both spaces.
5. Blend TS2Vec and window novelty with `alpha=0.5`.
6. Apply quality and physical-validity gates.
7. Use blended k-center greedy selection for diversity.
8. Write ranked submission, diagnostics, finalized ID files, and report.

Budget guardrails:

- new-only TS2Vec precompute uses only the `new` manifest
- `max_clips_per_split = 2500`
- `fail_if_split_exceeds_max = true`
- final ranker uses `right_support_max_clips = 25000`
- Modal H100 wrappers have 1-hour timeout
- all expensive Modal apps were verified stopped after completion

## Modal Runs And Outcomes

Stopped full-support TS2Vec attempt:

- App: `ap-BskSSsKHYaXewvHAtDQirq`
- Stopped after writing partial old-support shards
- Useful residue: 22,327 old-support TS2Vec embeddings
- Reason stopped: full 202k TS2Vec embedding would be roughly an overnight H100
  job and was mostly IO-bound

Successful new-only H100 TS2Vec precompute:

```text
config: configs/active_embedding_precompute_ts2vec_new_only_h100.json
app: ap-lagxI19VYyS5OMo9hSaLTA
call: fc-01KQKRWZ5T8W8XSGCHZPKY278D
n_clips: 2000
n_shards: 2
manifest: /artifacts/active/embedding_cache/ts2vec_window_new_only_h100/embeddings_5fcc8a7d4d3d16bf699786fa.shards.json
```

Stopped first full final-ranker attempt:

- App: `ap-HATGGMezNAsVSIheJ1p7tN`
- Reason stopped: it tried to compute `window_mean_std_pool` for all 200,000
  support clips and was too slow on Modal volume file IO

Successful budgeted final rank:

```text
config: configs/active_final_blend_rank_budget_h100.json
app: ap-NldNI2LjyGFNWPvuoVGiDx
n_query: 2000
n_left_support: 22327
n_right_support: 25000
output_dir: /artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100
```

At the end, `modal app list` showed all relevant apps stopped with 0 active
tasks.

## Why Full H100 Runs Were Slow

The bottleneck was not mainly GPU math. It was:

- reading many small files from Modal volumes
- JSONL parsing for raw IMU
- per-clip normalization
- feature file reads for 200k support clips
- cache shard bookkeeping

H100 helps with the TS2Vec forward pass. It does not solve the small-file IO and
Python parsing bottleneck. This is why the final approach switched to candidate
TS2Vec plus sampled support.

## Verification

Local test command:

```bash
.venv/bin/python -m unittest discover -s tests
```

Latest result:

```text
Ran 314 tests in 2.916s
OK (skipped=6)
```

Targeted tests also passed for:

- budgeted final blend rank
- new-only TS2Vec precompute config
- split caps and fail-closed guardrails
- batched TS2Vec inference

## Current Recommendation

Use the current budgeted output as the best available candidate ranking:

```text
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/active_final_blend_submission_full_new_worker_id.csv
```

Do not describe it as exact full-support TS2Vec. Describe it as:

```text
budgeted TS2Vec/window blended k-center selector
```

This is the honest claim:

- all 2,000 new candidates were embedded with TS2Vec
- old-support TS2Vec was approximated with a 22,327-clip partial cache
- old-support window novelty was approximated with a 25,000-clip support sample
- the result is clean by top-k hygiene checks

## Open Questions For Reviewer

1. Is a 22k/25k old-support sample acceptable for the final challenge ranking,
   or should support sampling be stratified by source group/worker?
2. Should `create_overlapping_crops()` be fixed and TS2Vec retrained before
   making scientific claims about the representation?
3. Should the data layer be redesigned to avoid Modal-volume small-file IO, for
   example by packing raw/feature arrays into large shard files?
4. Should the learned ridge ranker stay diagnostic, or should a non-linear
   hygienic ranker be tried after the selector is stable?
5. Should final submission use `worker_id` or `new_worker_id` format? Both were
   written; confirm the external evaluator's expected column.

