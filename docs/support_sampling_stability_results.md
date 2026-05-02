# Support-Sampling Stability Results

Date: 2026-05-02

## Verdict

The final ranking is globally stable but top-K sensitive to the sampled
25,000-clip window-stat old support.

This means the current final artifact is still defensible as a budgeted
engineering selector, but it should not be described as an exact scientific
full-support acquisition result. The most honest next improvement is either:

1. build a consensus ranking over several support samples, or
2. remove the support-sampling approximation with a real full-support data
   layer.

## Runs

Support subset audit:

```text
Modal app: ap-Umv2UGDxZgSrLYSzoFTni0
report: /artifacts/active/final_blend_rank/budget_ts2vec_window_a05_h100/support_subset_audit/support_subset_audit_report_full.json
```

Support-sampling stability:

```text
Modal app: ap-OUGAsoOtT1fsWcgXhgXxi8
config: configs/active_support_sampling_stability_budget_cpu.json
report: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.json
markdown: /artifacts/active/final_blend_rank/support_sampling_stability_a05_cpu/support_sampling_stability_report_full.md
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
top of the list depends on the old-support sample. If there is time to improve
the artifact, prefer a consensus ranking over multiple support samples before
trying another learned model.
