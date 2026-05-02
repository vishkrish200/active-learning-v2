# Exact-Window Top-K Visual Audit

Date: 2026-05-03

## Verdict

The exact-window artifact is mostly physically plausible by time-series audit,
but it is not perfectly clean at the very top. The top 50 contains three
spiky/high-jerk clips that pass the current quality and physical-validity gates.

This does not invalidate the artifact, but it does show the next cheap
improvement: add a spike-aware hygiene/rerank check before treating the first
10-50 clips as final.

## Run

Smoke run:

```text
Modal app: ap-dQcGerwJeRTwEAokWp69O0
mode: smoke
n_selected: 16
n_plots_written: 16
```

Full run:

```text
Modal app: ap-7KLS6g32OLs5XK5Zu9xZMz
mode: full
n_selected: 126
n_plots_written: 126
```

The audit selected:

- top 50 exact-window clips;
- top 50 consensus clips;
- 30 largest exact-vs-consensus rank disagreements;
- 30 high-novelty rows rejected by hygiene gates.

## Artifacts

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/topk_audit/topk_audit_report_full.json
/artifacts/active/final_blend_rank/exact_full_window_a05/topk_audit/topk_audit_report_full.md
/artifacts/active/final_blend_rank/exact_full_window_a05/topk_audit/topk_audit_index_full.html
/artifacts/active/final_blend_rank/exact_full_window_a05/topk_audit/topk_audit_plot_index_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/topk_audit/trace_plots_full/
```

## Summary

Overall audited traces:

| Verdict | Count |
| --- | ---: |
| plausible_motion | 92 |
| likely_artifact | 33 |
| mostly_stationary | 1 |

The high artifact count is driven mostly by deliberately selected rejected
high-novelty rows, not by the promoted top 50.

Exact top-K hygiene from diagnostics remains clean:

| K | Exact/Consensus Overlap | Quality Fail | Physical Fail | Unique Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.8000 | 0.0000 | 0.0000 | 10 |
| 50 | 0.6800 | 0.0000 | 0.0000 | 50 |
| 100 | 0.8100 | 0.0000 | 0.0000 | 100 |
| 200 | 0.8300 | 0.0000 | 0.0000 | 199 |

Visual-audit verdict by group:

| Group | Count | Plausible | Likely Artifact | Mostly Stationary |
| --- | ---: | ---: | ---: | ---: |
| exact top 10 | 10 | 8 | 2 | 0 |
| exact top 50 | 50 | 47 | 3 | 0 |
| consensus top 50 | 50 | 47 | 3 | 0 |
| rank disagreements | 30 | 30 | 0 | 0 |
| rejected high-novelty | 30 | 0 | 29 | 1 |

Automated flags:

| Flag | Count |
| --- | ---: |
| clean_motion | 90 |
| spiky_or_extreme | 28 |
| low_quality_score | 23 |
| physical_abs_value_outlier | 7 |
| high_jerk | 6 |
| mostly_stationary | 2 |

## Manual Inspection Notes

I inspected representative plots from the generated audit pack.

`audit 001`, exact rank 1 / consensus rank 2:

- Looks physically plausible.
- There is a strong initial transient, then a long stable motion segment.
- It is not obviously broken or saturated.

`audit 002`, exact rank 2 / consensus rank 1:

- Flagged as `spiky_or_extreme`.
- The clip has several short high-jerk bursts early in the trace.
- The magnitude is not absurd and the clip passes current gates, but it is a
  borderline top-ranked example because the selector appears to reward those
  bursts.

`audit 006`, exact rank 6 / consensus rank 14:

- Flagged as `spiky_or_extreme`.
- The trace contains repeated high acceleration spikes in short windows, then a
  long quiet period.
- This could be a real handling/contact event, but it is exactly the kind of
  clip that can inflate novelty if spike rate is not explicitly controlled.

`audit 035`, exact rank 35 / consensus rank 74:

- Flagged as `spiky_or_extreme`.
- Mostly quiet, with one concentrated motion burst around the middle of the
  clip.
- Again, not necessarily invalid, but spike-driven novelty is visible.

`audit 097`, rejected high-novelty:

- Correctly rejected.
- It has repeated high-amplitude and high-jerk activity with max absolute value
  above the physical threshold.

`audit 099`, rejected high-novelty:

- Correctly rejected.
- Mostly stationary with a couple of isolated events.

## Interpretation

The top-ranked exact-window artifact is not a pile of artifacts. Most top 50
clips are clean-looking, and the exact-vs-consensus disagreements are also
clean-looking. That is good.

The weak spot is narrower: a few very high-ranked clips are spike-heavy but
still pass the current gates because the current final diagnostics only gate on
quality score, stationary fraction, and max absolute value. The quality score
does penalize spike rate, but not enough to prevent some spike-driven clips from
entering the top 10.

The rejected high-novelty audit is reassuring. It shows that the hard gates are
catching the obvious bad cases and pushing them near the bottom of the ranking.

## Recommendation

Do not retrain anything yet.

Follow-up completed:

```text
docs/spike_hygiene_ablation_2026-05-03.md
```

The spike-only ablation removed the single top-50 clip above
`spike_rate=0.025`, but two other likely-artifact top-50 clips were just below
that cutoff. A stricter trace-gate variant removed all three top-50
likely-artifact clips while preserving top-50 cluster diversity and keeping
`0.94` overlap with the current exact-window top 50.

Remaining cheap step before promotion:

1. Generate a small trace-gate audit pack for the three replacement top-50
   clips and the new top 10.
2. Confirm the replacement clips look physically plausible.
3. Promote the trace-gate output only if that targeted visual check passes.

Original proposed ablation:

1. Compute `spike_rate`, `extreme_value_fraction`, and `high_frequency_energy`
   for all 2,000 new clips.
2. Add a spike-aware rerank or hygiene guard as an ablation, not immediately as
   the default.
3. Compare the current exact-window top 50 against the spike-aware top 50.
4. Promote the stricter version only if it removes the three top-50
   spike-heavy clips without reducing diversity or over-selecting bland motion.

Suggested starting ablation:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
spike_rate <= 0.025
```

The threshold should be treated as a candidate, not a final constant. The top-2
clip has `spike_rate=0.0264`, so the choice is sensitive. A slightly softer
alternative is to keep the gate unchanged and use spike rate as a tie-break or
score penalty inside the top 100.
