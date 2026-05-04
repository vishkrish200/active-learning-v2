# Exact Full-Support TS2Vec / Window Blend Results - 2026-05-04

## Purpose

Remove the main scientific caveat in the promoted final selector: partial
old-support TS2Vec. The previous final artifact used full old-support
`window_mean_std_pool`, but only a partial old-support TS2Vec cache. This run
uses exact full old-support for both views:

- TS2Vec old support: 200,000 / 200,000 clips
- window-stat old support: 200,000 / 200,000 clips
- new candidate query set: 2,000 / 2,000 clips

This does not retrain TS2Vec. It uses the current promoted checkpoint and only
changes support coverage for ranking.

## Inputs

```text
TS2Vec old-support cache:
  /artifacts/active/embedding_cache/ts2vec_old_support_full_l4_20260504/ts2vec_embedding_shards

TS2Vec new-candidate cache:
  /artifacts/active/embedding_cache/ts2vec_window_new_only_h100/embeddings_5fcc8a7d4d3d16bf699786fa_shards

Exact window old-support shards:
  /artifacts/active/full_support_shards/window_mean_std_v1/full_support_shards_full.json

Ranking config:
  configs/active_exact_full_ts2vec_window_blend_rank.json

Artifact-gate config:
  configs/active_spike_hygiene_ablation_exact_full_ts2vec_window.json

Final package config:
  configs/final_package_exact_full_ts2vec_artifact_gate.json
```

The full TS2Vec old-support cache was produced on GCP and imported into the
Modal artifact volume. Modal verification found 200 TS2Vec shard files under
the imported cache path.

## Modal Runs

### Exact-support ranking smoke

```text
modal_app: ap-zrCYWVupYiIAJ2guZzDW6i
mode: smoke
n_left_support: 512
n_right_support: 128
n_query: 64
status: passed
```

The smoke uses only two TS2Vec support shards through
`smoke_left_support_max_shards`, so it validates wiring without loading all
200,000 support embeddings.

### Exact-support full ranking

```text
modal_app: ap-bNsCYnosC6FhvHzUPK4coD
mode: full
selector: exact_window_blend_kcenter_ts2vec_window_mean_std_pool_a05
ranking_mode: full_left_exact_window_right
n_left_support: 200000
n_right_support: 200000
n_query: 2000
left_support_cache_status: full_support_shard_hit
right_support_cache_status: full_support_shard_hit
```

Output report:

```text
/artifacts/active/final_blend_rank/exact_full_ts2vec_window_a05/active_exact_window_blend_report_full.json
```

Top-K quality before trace artifact rerank:

| K | Quality Fail | Physical Fail | Duplicate Rate | Unique Clusters |
|---:|---:|---:|---:|---:|
| 10 | 0.000 | 0.000 | 0.000 | 10 |
| 50 | 0.000 | 0.000 | 0.000 | 50 |
| 100 | 0.000 | 0.000 | 0.000 | 100 |
| 200 | 0.000 | 0.000 | 0.005 | 199 |

### Artifact-gate full pass

```text
modal_app: ap-ucnUR9senSUXIvqM5lN7z9
mode: full
n_rows: 2000
artifact_gate_top50_trace_artifact_fail_rate: 0.000
hard_gate_top50_spike_fail_rate: 0.000
soft_penalty_top50_spike_fail_rate: 0.000
trace_gate_top50_trace_fail_rate: 0.000
status: passed
```

Artifact-gated output report:

```text
/artifacts/active/final_blend_rank/exact_full_ts2vec_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_report_full.json
```

Artifact-gated top-K hygiene:

| K | Quality Fail | Physical Fail | Spike Fail | Trace Artifact Fail | Unique Clusters | Overlap With Pre-Rerank |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.000 | 0.000 | 0.000 | 0.000 | 10 | 0.800 |
| 50 | 0.000 | 0.000 | 0.000 | 0.000 | 50 | 0.940 |
| 100 | 0.000 | 0.000 | 0.000 | 0.000 | 100 | 0.960 |
| 200 | 0.000 | 0.000 | 0.000 | 0.000 | 199 | 0.975 |

The artifact gate removed the exact-support pre-rerank top-50 trace artifacts:

```text
baseline_top50_spike_fail_count: 1
baseline_top50_spike_fail_removed_count: 1
baseline_top50_trace_fail_count: 3
baseline_top50_trace_fail_removed_count: 3
```

## Final Package

Local final package:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate
```

Primary CSV:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips.csv
```

Backup ID variants:

```text
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips_new_worker_id.csv
artifacts/final_selector/exact_full_ts2vec_artifact_gate/ranked_new_clips_worker_id.csv
```

Package validation:

```text
status: ready
n_rows: 2000
primary_id_column: new_worker_id
unique_primary_ids: 2000
```

The package includes:

- `selector_report.json`: artifact-gate hygiene report
- `exact_selector_report.json`: exact full-support TS2Vec/window ranking report
- `validation_report.json`: prior active-loop validation report
- `selector_config.json`: machine-readable method claim and limitations

## Comparison To Previous Final Package

Previous package:

```text
artifacts/final_selector/artifact_gate_exact_window
```

The previous package was the artifact-gated partial-TS2Vec / exact-window
selector. The new exact-full package is close to it, but removes the partial
old-support TS2Vec caveat.

| Prefix | Top-K Overlap |
|---:|---:|
| 10 | 0.900 |
| 25 | 0.920 |
| 50 | 0.940 |
| 100 | 0.970 |
| 200 | 0.965 |
| 500 | 0.892 |
| 1000 | 0.909 |

Full-rank correlation on ranks:

```text
pearson_on_rank_vectors: 0.9572
```

The exact selector report also compared the exact-full TS2Vec/window ranking
against the earlier exact-window artifact-gated baseline:

```text
rank_spearman: 0.91985
top10_overlap: 0.800
top50_overlap: 0.920
top100_overlap: 0.950
top200_overlap: 0.945
```

## Current Recommendation

Promote `exact_full_ts2vec_artifact_gate` as the best current submission
package.

This is now stronger than the previous final package because:

- both old-support views are exact full-support for the provided old corpus;
- all 2,000 new candidates are ranked;
- top-50 artifact/spike hygiene is clean after trace rerank;
- the output is highly consistent with the previous promoted package, so the
  full-support change did not destabilize the result.

Remaining caveat:

- The TS2Vec checkpoint is still the current promoted checkpoint, not a newly
  retrained fixed-crop checkpoint. The method should be described as an exact
  full-support TS2Vec/window geometric selector, not as validated clean TS2Vec
  active learning.

Recommended follow-up plan:

```text
docs/next_steps_after_exact_full_2026-05-04.md
```
