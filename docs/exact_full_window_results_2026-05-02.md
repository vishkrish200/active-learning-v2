# Exact Full-Window Results

Date: 2026-05-02

## Verdict

The promoted artifact should now be the exact-window blend ranking:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_submission_full_new_worker_id.csv
```

This replaces the 3-seed consensus ranking as the primary candidate output.
The consensus run remains useful provenance and a stability baseline, but the
exact-window run removes the biggest cheap support approximation: sampled
25,000-clip window-stat old support.

The method is still not an exact full-support TS2Vec system because TS2Vec old
support remains partial at 22,327 clips.

## What Changed

The previous best artifact used:

```text
TS2Vec old support: partial 22,327 clips
window_mean_std_pool old support: sampled 25,000 clips per seed
aggregation: 3-seed rank consensus
```

The exact-window artifact uses:

```text
TS2Vec old support: partial 22,327 clips
window_mean_std_pool old support: exact 200,000 clips
query clips: all 2,000 new clips
aggregation: single deterministic blended k-center run
```

This is the highest-value scientific improvement that did not require new
training, five support seeds, or a brute-force full TS2Vec old-support run.

## Runs

Full window-shard build:

```text
Modal app: ap-Qavmxl6rQzJ9yHZd7wUDoM
config: configs/build_full_support_window_shards.json
mode: full
n_clips: 202000
n_shards: 50
manifest: /artifacts/active/full_support_shards/window_mean_std_v1/full_support_shards_full.json
report: /artifacts/active/full_support_shards/window_mean_std_v1/full_support_shard_report_full.json
progress: /artifacts/active/full_support_shards/window_mean_std_v1/full_support_shard_progress_full.jsonl
```

Full exact-window ranking:

```text
Modal app: ap-hivXtqllLALI9UBYyOBJ8e
config: configs/active_exact_window_blend_rank.json
mode: full
n_query: 2000
n_left_support: 22327
n_right_support: 200000
selector: exact_window_blend_kcenter_ts2vec_window_mean_std_pool_a05
```

## Artifacts

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_submission_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_submission_full_worker_id.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_submission_full_new_worker_id.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/active_exact_window_blend_report_full.json
```

Use the `new_worker_id` file as primary unless the evaluator explicitly expects
the internal manifest-hash `worker_id` column.

## Hygiene

Exact-window top-k hygiene:

| K | Quality Fail | Physical Fail | Duplicate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0000 | 0.0000 | 0.0000 | 10 |
| 50 | 0.0000 | 0.0000 | 0.0000 | 50 |
| 100 | 0.0000 | 0.0000 | 0.0000 | 100 |
| 200 | 0.0000 | 0.0000 | 0.0050 | 199 |

This improves the consensus artifact's duplicate rate at K=50 and K=100.

## Comparison To Consensus

Compared against the 3-seed consensus diagnostics:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_diagnostics_full.csv
```

| Metric | Value |
| --- | ---: |
| rank_spearman | 0.9846 |
| top10_overlap | 0.8000 |
| top50_overlap | 0.6800 |
| top100_overlap | 0.8100 |
| top200_overlap | 0.8300 |

Interpretation: exact full-window support changes some top-50 choices, but it
does not radically disagree with the stabilized consensus ranking. The top-10
agreement is much stronger than the weakest seed-to-seed top-10 agreement from
the support-sampling stability run.

## Claim

Use this wording:

> We rank new IMU clips with a multi-view geometric acquisition selector. The
> method combines TS2Vec novelty against a partial old-support cache with exact
> full-corpus window-stat novelty against all 200,000 old-support clips,
> applies hard sensor-quality gates, and uses k-center-style reranking to
> reduce redundancy.

Avoid claiming:

- exact full-support TS2Vec search
- validated TS2Vec active learning
- full-corpus learned marginal value
- semantic workflow discovery

## Remaining Limitations

1. TS2Vec old support remains partial at 22,327 clips.
2. The current promoted TS2Vec checkpoint should still be described as an
   empirical representation signal, not a clean validated TS2Vec result.
3. Old-support shard quality metadata is not complete in this window-only shard
   build. Candidate-side hygiene is still computed and enforced during ranking.
4. External held-out evaluation remains unknown.

## Recommendation

Use exact-window as the new primary artifact. Keep consensus as a fallback and
as evidence that the exact-window result is not an unstable outlier.
