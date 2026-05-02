# Support-Sampling Stability Results

Date: 2026-05-02

## Verdict

The single-seed final ranking is globally stable but top-K sensitive to the
sampled 25,000-clip window-stat old support.

This means the budgeted engineering selector is defensible, but it should not
be described as an exact scientific full-support acquisition result. The
recommended artifact is now the 3-seed consensus ranking, which directly
reduces dependence on any one sampled old-support subset.

The next larger improvement would be to remove the support-sampling
approximation with a real full-support data layer.

## Runs

Support subset audit:

```text
Modal app: ap-Umv2UGDxZgSrLYSzoFTni0
report: /artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_report_full.json
```

Support-sampling stability and consensus:

```text
Modal app: ap-yhHD37KCA4PsXm2Vpv8osq
config: configs/active_support_sampling_stability_budget_cpu.json
report: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.json
markdown: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.md
consensus submission: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
consensus diagnostics: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_diagnostics_full.csv
consensus report: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_report_full.json
seeds: 1, 2, 3
right_support_max_clips: 25000
```

The stability run was deliberately moved from H100 to CPU after observing that
the job was IO/window-stat bound. Candidate TS2Vec embeddings were already
cached, so H100 was not buying useful speed.

## Support Representativeness

| Support Set | Clips | Unique Workers | Worker Coverage vs Full | Top Worker Fraction |
| --- | ---: | ---: | ---: | ---: |
| full old support | 200,000 | 10,000 | 1.0000 | 0.0001 |
| partial TS2Vec support | 22,327 | 9,067 | 0.9067 | 0.0004 |
| capped window support | 25,000 | 9,324 | 0.9324 | 0.0004 |
| random 25k comparison | 25,000 | 9,294 | 0.9294 | 0.0004 |

The support subset is broad by worker/source coverage. The risk is therefore
less "bad shard ordering" and more "top ranks move when the finite support
sample changes."

## Stability Summary

Pairwise across seeds 1, 2, and 3:

| Metric | Value |
| --- | ---: |
| rank_spearman_mean | 0.9589 |
| rank_spearman_min | 0.9477 |
| score_mean_abs_delta_mean | 0.038832 |
| score_mean_abs_delta_max | 0.048690 |
| top10_overlap_mean | 0.5000 |
| top10_overlap_min | 0.3000 |
| top50_overlap_mean | 0.6467 |
| top50_overlap_min | 0.5600 |
| top100_overlap_mean | 0.7000 |
| top100_overlap_min | 0.6500 |

Pair details:

| Pair | Spearman | Top 10 | Top 50 | Top 100 | Mean Abs Score Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed 1 vs seed 2 | 0.9477 | 0.6000 | 0.6400 | 0.7000 | 0.048690 |
| seed 1 vs seed 3 | 0.9529 | 0.6000 | 0.7400 | 0.7500 | 0.045576 |
| seed 2 vs seed 3 | 0.9761 | 0.3000 | 0.5600 | 0.6500 | 0.022232 |

## Hygiene

All three seeded rankings pass top-K hygiene:

| Run | K | Quality Fail | Physical Fail | Duplicate |
| --- | ---: | ---: | ---: | ---: |
| seed 1 | 10 | 0.0000 | 0.0000 | 0.0000 |
| seed 1 | 50 | 0.0000 | 0.0000 | 0.0200 |
| seed 1 | 100 | 0.0000 | 0.0000 | 0.0100 |
| seed 2 | 10 | 0.0000 | 0.0000 | 0.0000 |
| seed 2 | 50 | 0.0000 | 0.0000 | 0.0000 |
| seed 2 | 100 | 0.0000 | 0.0000 | 0.0100 |
| seed 3 | 10 | 0.0000 | 0.0000 | 0.0000 |
| seed 3 | 50 | 0.0000 | 0.0000 | 0.0200 |
| seed 3 | 100 | 0.0000 | 0.0000 | 0.0100 |

## Interpretation

Overall Spearman near 0.96 says the ranking is not random or wildly unstable.
But top-10 overlap as low as 0.30 is too weak for a strong scientific claim
about exact top-ranked clips. The top 50 and top 100 are more stable but still
sample-sensitive.

Conclusion: the method is useful as a budgeted selector, but the precise very
top of any single-seed list depends on the old-support sample. The 3-seed
consensus ranking is therefore the recommended submission artifact before
trying another learned model.

## Consensus Ranking

The consensus ranking aggregates the three seeded full rankings by mean rank,
with Borda score and mean score as deterministic tie-breaks.

Primary consensus file:

```text
/artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/active_final_blend_consensus_submission_full_new_worker_id.csv
```

Consensus top-k hygiene:

| K | Quality Fail | Physical Fail | Duplicate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0000 | 0.0000 | 0.0000 | 10 |
| 50 | 0.0000 | 0.0000 | 0.0200 | 49 |
| 100 | 0.0000 | 0.0000 | 0.0100 | 99 |

Consensus stability diagnostics:

| Metric | Value |
| --- | ---: |
| n_runs | 3 |
| n_rows | 2000 |
| rank_std_mean | 51.7567 |
| rank_std_top10_mean | 3.7643 |
| mean_rank_top10_mean | 8.5000 |

An attempted extension to five seeds was stopped during seed 4 because that
seed required a fresh 25k support-cache build and was mostly paying the same
Modal-volume small-file IO cost. The 3-seed consensus is the right current
cost/value point.
