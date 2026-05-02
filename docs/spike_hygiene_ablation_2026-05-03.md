# Spike Hygiene Ablation

Date: 2026-05-03

## Verdict

The exact-window ranking is cleaner with a trace-aware hygiene pass. The
follow-up targeted visual audit passed, so `trace_gate` is now promoted as the
current primary artifact.

Targeted audit:

```text
docs/trace_gate_targeted_audit_2026-05-03.md
```

The important result is that spike-only gating was too narrow. It removed the
single top-50 clip whose `spike_rate` exceeded `0.025`, but two additional
top-50 clips were still marked `likely_artifact` by the richer trace audit.

The trace-gate variant removed all three likely-artifact clips from the current
top 50 while preserving cluster diversity and most of the original ordering.

## Run

Smoke run:

```text
Modal app: ap-WKpKlJp31OKqJHNLmUT44L
mode: smoke
n_rows: 128
```

Full run:

```text
Modal app: ap-Tbodgi7z3IXDXjp5GSR3p7
mode: full
n_rows: 2000
spike_rate_threshold: 0.025
```

## Artifacts

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_report_full.json
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_report_full.md
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_enriched_current_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_hard_gate_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_soft_penalty_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_trace_gate_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_trace_gate_submission_full_new_worker_id.csv
```

## Variants

`current` is the existing exact-window blend ranking.

`hard_gate` moves rows with `quality__spike_rate > 0.025` below all spike-pass
rows while preserving the original order.

`soft_penalty` applies a score penalty to rows with `quality__spike_rate >
0.025`.

`trace_gate` moves rows below all trace-pass rows if either:

```text
quality__spike_rate > 0.025
trace__verdict in {"likely_artifact", "mostly_stationary"}
```

## Top-K Summary

| Variant | K | Spike Fail | Trace Fail | Current Overlap | Unique Clusters | Mean Original Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 10 | 0.1000 | 0.2000 | 1.0000 | 10 | 5.50 |
| current | 50 | 0.0200 | 0.0600 | 1.0000 | 50 | 25.50 |
| current | 100 | 0.0200 | 0.0400 | 1.0000 | 100 | 50.50 |
| current | 200 | 0.0150 | 0.0250 | 1.0000 | 199 | 100.50 |
| hard_gate | 10 | 0.0000 | 0.1000 | 0.9000 | 10 | 6.40 |
| hard_gate | 50 | 0.0000 | 0.0400 | 0.9800 | 50 | 26.48 |
| hard_gate | 100 | 0.0000 | 0.0200 | 0.9800 | 100 | 51.97 |
| hard_gate | 200 | 0.0000 | 0.0100 | 0.9850 | 199 | 102.36 |
| soft_penalty | 10 | 0.0000 | 0.1000 | 0.9000 | 10 | 6.40 |
| soft_penalty | 50 | 0.0000 | 0.0400 | 0.9800 | 50 | 26.48 |
| soft_penalty | 100 | 0.0000 | 0.0200 | 0.9800 | 100 | 51.97 |
| soft_penalty | 200 | 0.0000 | 0.0100 | 0.9850 | 199 | 102.36 |
| trace_gate | 10 | 0.0000 | 0.0000 | 0.8000 | 10 | 7.00 |
| trace_gate | 50 | 0.0000 | 0.0000 | 0.9400 | 50 | 27.76 |
| trace_gate | 100 | 0.0000 | 0.0000 | 0.9600 | 100 | 53.63 |
| trace_gate | 200 | 0.0000 | 0.0000 | 0.9750 | 199 | 104.20 |

## What Changed In Top 50

The trace-gate variant removes these current top-50 clips:

| Current Rank | Sample | URL | Spike Rate | Trace Verdict |
| ---: | --- | --- | ---: | --- |
| 2 | `2676a46b...` | `sample000984.txt` | 0.0264 | likely_artifact |
| 6 | `df22942d...` | `sample000738.txt` | 0.0244 | likely_artifact |
| 35 | `b1eb0b80...` | `sample001897.txt` | 0.0235 | likely_artifact |

It replaces them with the next clean rows from the original ranking:

| Trace-Gate Rank | Original Rank | Sample | URL | Spike Rate | Trace Verdict |
| ---: | ---: | --- | --- | ---: | --- |
| 48 | 51 | `c683ce3b...` | `sample000275.txt` | 0.0002 | plausible_motion |
| 49 | 52 | `bd81793b...` | `sample000952.txt` | 0.0013 | plausible_motion |
| 50 | 53 | `a2cf4f0e...` | `sample001437.txt` | 0.0150 | plausible_motion |

## Interpretation

This ablation supports adding a trace-aware hygiene stage before treating the
top 10-50 clips as final. The cost in ranking movement is small: top-50 overlap
with the current exact-window artifact is `0.94`, top-100 overlap is `0.96`,
and unique cluster counts stay unchanged at K=50 and K=100.

The stricter gate is scientifically better than spike-only gating because it
matches the actual visual-audit finding: the problematic rows were not all over
the proposed `0.025` spike-rate threshold. Two of the three were just below it.

## Recommendation

Do not retrain.

Promote the trace-gate submission as the current primary artifact:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_trace_gate_submission_full_new_worker_id.csv
```

Keep the original exact-window artifact as fallback/provenance. The trace-gate
change is only a conservative hygiene rerank; it does not change the model or
the underlying exact-window support calculation.
