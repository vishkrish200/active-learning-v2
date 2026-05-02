# Final Submission Note

Date: 2026-05-02

## Direct Verdict

The current artifact is a credible external-challenge submission candidate. The
recommended output is now the 3-seed consensus ranking over budgeted support
samples, not the original single support-sample ranking.

It should be described as a budgeted multi-view geometric acquisition selector,
not as a clean scientific TS2Vec active-learning result.

Recommended label:

```text
Budgeted TS2Vec/window blended k-center acquisition selector
```

Final selector:

```text
consensus_blend_kcenter_ts2vec_window_mean_std_pool_a05
```

## Submission Files

Primary candidate file:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
```

Backup ID-format file:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_worker_id.csv
```

Diagnostics and report:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_diagnostics_full.csv
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_report_full.json
```

Use the `new_worker_id` file as primary unless the evaluator explicitly expects
the internal manifest-hash `worker_id` column.

## Method Summary

We rank the 2,000 newly arrived IMU clips using a budgeted multi-view
acquisition selector. Each candidate is embedded with a TS2Vec-style temporal
encoder and with a cheap handcrafted window-stat representation. Candidate
novelty is estimated against two old-support views: a partial cached TS2Vec
old-support index and a sampled window-stat old-support index. The two novelty
scores are min-max normalized and blended with `alpha=0.5`.

Before ranking, clips are filtered by hard hygiene gates:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
```

Each seeded ordering uses a quality-gated blended k-center strategy to
prioritize clips that are both under-covered by the old corpus and nonredundant
within the new batch. The final promoted ordering averages ranks across seeds
`1, 2, 3`, with Borda score and mean novelty score as deterministic tie-breaks.

This is not an exact full-200k TS2Vec search. It is a budgeted approximation
designed to fit compute and IO constraints while still using all 2,000 new
candidate clips.

## Exact Config

Final ranking config:

```text
configs/active_final_blend_rank_budget_h100.json
```

Important settings:

```text
budgeted_candidate_only: true
left_representation: ts2vec
right_representation: window_mean_std_pool
alpha: 0.5
left_support_shard_dir: /artifacts/active/embedding_cache/ts2vec_window_full_new/embeddings_7481b57ede264d17002b4316_shards
min_left_support_clips: 20000
right_support_max_clips: 25000
max_query_clips: 2500
quality_threshold: 0.85
max_stationary_fraction: 0.90
max_abs_value: 60.0
```

New-candidate TS2Vec precompute config:

```text
configs/active_embedding_precompute_ts2vec_new_only_h100.json
```

Successful new-candidate cache:

```text
/artifacts/active/embedding_cache/ts2vec_window_new_only_h100/embeddings_5fcc8a7d4d3d16bf699786fa.shards.json
```

## Final Run

Successful single-sample budgeted final ranking:

```text
Modal app: ap-NldNI2LjyGFNWPvuoVGiDx
mode: full
n_query: 2000
n_left_support: 22327
n_right_support: 25000
```

Successful 3-seed consensus run:

```text
Modal app: ap-yhHD37KCA4PsXm2Vpv8osq
mode: full
seeds: 1, 2, 3
n_query: 2000
n_left_support: 22327
n_right_support_per_seed: 25000
```

Consensus top-k hygiene:

| K | Quality Fail | Physical Fail | Duplicate Rate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000 | 0.000 | 0.000 | 10 |
| 50 | 0.000 | 0.000 | 0.020 | 49 |
| 100 | 0.000 | 0.000 | 0.010 | 99 |

## Validation Summary

In source-blocked simulated active-acquisition episodes, the promoted blend
selector outperformed the learned ridge ranker, quality-gated k-center, and the
prior window-shape cluster-cap baseline on balanced hidden-target coverage gain
at K=10 and K=50. Oracle greedy remained higher, indicating residual headroom.

| Policy | K=10 Balanced Relative Gain | K=50 Balanced Relative Gain |
| --- | ---: | ---: |
| blend k-center TS2Vec/window a05 | 0.2149 | 0.3042 |
| k-center quality-gated | 0.1281 | 0.2167 |
| window-shape cluster-cap | 0.1398 | 0.2872 |
| learned ridge, gated target | 0.1688 | 0.2533 |
| oracle greedy eval-only | 0.3212 | 0.3805 |

Detailed ranker hygiene results:

```text
docs/ranker_hygiene_fix_results.md
```

## Known Limitations

1. TS2Vec checkpoint provenance has been fixed in code, but the fixed
   checkpoint is not promoted yet.

   The public `create_overlapping_crops()` helper previously returned two
   copies of one sampled crop. The training loop used a separate shifted-crop
   path, but the split implementation made the checkpoint provenance too muddy
   for a clean scientific claim. The helper has been fixed and a bounded
   fixed-crop checkpoint was trained/evaluated, but it did not clearly beat the
   current checkpoint on the 8-episode medium eval. Do not replace the current
   final artifact with the fixed checkpoint without longer training and another
   active-loop comparison.

2. Old-support TS2Vec is partial.

   The selector uses 22,327 cached old-support TS2Vec embeddings from a stopped
   partial full-support precompute. It does not search all 200,000 old clips in
   TS2Vec space.

3. Window old support is sampled.

   The selector uses a capped 25,000 old-support `window_mean_std_pool` subset,
   not all 200,000 old clips. This was necessary because full support feature IO
   was still slow on Modal volumes. A 3-seed support-sampling stability run
   found high overall rank correlation but weak top-10 overlap for individual
   seeds, so the promoted artifact is a consensus over seeds `1, 2, 3`.

4. External held-out score is unknown.

   Internal active-loop evaluation supports the blend selector, but the
   challenge evaluator remains the real test.

## Support Approximation Audit

Support subset audit code was added in:

```text
marginal_value/active/support_subset_audit.py
modal_active_support_subset_audit.py
tests/test_active_support_subset_audit.py
```

Full audit run:

```text
Modal app: ap-Umv2UGDxZgSrLYSzoFTni0
n_full_support: 200000
n_partial_ts2vec_support: 22327
n_window_support: 25000
```

Audit artifacts:

```text
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_report_full.json
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_source_groups_full.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_workers_full.csv
```

Representativeness summary:

| Support Set | Clips | Unique Workers | Worker Coverage vs Full | Top Worker Fraction | Top 5 Worker Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| full old support | 200,000 | 10,000 | 1.0000 | 0.0001 | 0.0005 |
| partial TS2Vec support | 22,327 | 9,067 | 0.9067 | 0.0004 | 0.0018 |
| capped window support | 25,000 | 9,324 | 0.9324 | 0.0004 | 0.0017 |
| random 25k comparison | 25,000 | 9,294 | 0.9294 | 0.0004 | 0.0017 |

The source-group numbers match the worker numbers in this registry view:
partial TS2Vec covers 9,067/10,000 source groups, and the capped window support
covers 9,324/10,000 source groups. This makes the budgeted support
approximation much less concerning than a first-shard or source-ordered sample:
coverage is broad and close to a deterministic random 25k comparison.

Quality distribution is not included in this audit because the final ranking
config does not attach full-support quality metadata to the registry. The final
candidate ranking still applies hard candidate-side quality and physical-validity
gates.

Re-run command:

```bash
.venv/bin/modal run modal_active_support_subset_audit.py \
  --config-path configs/active_final_blend_rank_budget_h100.json \
  --run-full \
  --skip-smoke
```

Expected full outputs:

```text
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_report_full.json
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_source_groups_full.csv
/artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_workers_full.csv
```

This audit checks worker/source-group coverage for:

- full old support
- partial TS2Vec old support
- capped 25k window old support
- deterministic random 25k old support

Quality distribution is included when quality metadata is attached to the
registry config.

## Support-Sampling Stability

The support sample is broad, but the exact top of any single-seed ranking is
sample-sensitive. The promoted artifact is therefore a 3-seed consensus ranking.

3-seed stability and consensus run:

```text
config: configs/active_support_sampling_stability_budget_cpu.json
Modal app: ap-yhHD37KCA4PsXm2Vpv8osq
report: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.json
consensus: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
```

Summary:

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

All three seeded runs, and the consensus ranking, had 0.000 quality and
physical failure rates at K=10, K=50, and K=100. The single-seed top 10 is not
stable enough to claim an exact support-independent ordering, so the consensus
ranking is the recommended artifact.

Detailed results:

```text
docs/support_sampling_stability_results.md
```

## Recommended Claim

Use this wording:

> We rank new IMU clips with a budgeted multi-view acquisition selector. The
> method combines TS2Vec novelty against a partial old-support cache with cheap
> window-stat novelty against sampled old support, applies hard sensor-quality
> gates, and uses k-center-style reranking to reduce redundancy.

Avoid:

- "validated TS2Vec active learning"
- "full-corpus learned marginal value"
- "exact full-200k TS2Vec novelty"

The original single-sample output remains useful as a provenance/fallback
artifact, but it is no longer the primary recommendation.
