# Consensus Submission Note

Date: 2026-05-02

## Verdict

The recommended artifact is the 3-seed consensus ranking:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
```

This replaces the original single support-sample ranking as the primary
candidate output. The original ranking remains a useful provenance/fallback
artifact.

## Method

Each seed runs the same budgeted blended selector:

```text
blend_kcenter_ts2vec_window_mean_std_pool_a05
```

The selector combines:

- TS2Vec novelty against 22,327 cached old-support TS2Vec embeddings
- `window_mean_std_pool` novelty against a seeded 25,000-clip old-support sample
- hard candidate hygiene gates
- k-center-style diversity reranking within the new batch

The consensus output aggregates seeds `1, 2, 3` by mean rank. Borda score, mean
score, and `worker_id` are used as deterministic tie-breaks.

## Artifacts

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full.csv
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_worker_id.csv
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_diagnostics_full.csv
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_report_full.json
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.json
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.md
```

## Run

```text
Modal app: ap-yhHD37KCA4PsXm2Vpv8osq
config: configs/active_support_sampling_stability_budget_cpu.json
mode: full
seeds: 1, 2, 3
n_query: 2000
n_left_support: 22327
n_right_support_per_seed: 25000
```

The attempted extension to five seeds was stopped during seed 4. It required a
fresh 25,000-clip support-cache build and mostly paid the same Modal-volume
small-file IO cost; the three-seed consensus is the current cost/value point.

## Stability

Pairwise single-seed stability:

| Metric | Value |
| --- | ---: |
| rank_spearman_mean | 0.9589 |
| rank_spearman_min | 0.9477 |
| top10_overlap_mean | 0.5000 |
| top10_overlap_min | 0.3000 |
| top50_overlap_mean | 0.6467 |
| top50_overlap_min | 0.5600 |
| top100_overlap_mean | 0.7000 |
| top100_overlap_min | 0.6500 |

Consensus diagnostics:

| Metric | Value |
| --- | ---: |
| n_runs | 3 |
| n_rows | 2000 |
| rank_std_mean | 51.7567 |
| rank_std_top10_mean | 3.7643 |
| mean_rank_top10_mean | 8.5000 |

## Hygiene

Consensus top-k hygiene:

| K | Quality Fail | Physical Fail | Duplicate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0000 | 0.0000 | 0.0000 | 10 |
| 50 | 0.0000 | 0.0000 | 0.0200 | 49 |
| 100 | 0.0000 | 0.0000 | 0.0100 | 99 |

## Claim

Use this wording:

> We rank new IMU clips with a budgeted multi-view acquisition selector. The
> method combines TS2Vec novelty against a partial old-support cache with cheap
> window-stat novelty against sampled old support, applies hard sensor-quality
> gates, uses k-center-style reranking to reduce redundancy, and aggregates
> three seeded support samples into a consensus order.

Avoid claiming exact full-support TS2Vec search, validated TS2Vec active
learning, or full-corpus learned marginal value.
