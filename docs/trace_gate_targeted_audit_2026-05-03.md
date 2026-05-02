# Trace-Gate Targeted Audit

Date: 2026-05-03

## Verdict

Promote the trace-gate rerank as the current primary artifact.

The targeted audit checked the new trace-gate top 10, the three clips promoted
into the trace-gate top 50, and the three clips removed from the exact-window
top 50. The result is clean:

- trace-gate top 10: 10/10 plausible motion;
- trace-gate added top-50 replacements: 3/3 plausible motion;
- current top-50 removals: 3/3 likely artifact / spike-driven.

This is a conservative hygiene improvement, not a new model. It preserves the
exact-window multi-view selector and only demotes rows that fail the trace
hygiene check.

## Run

Smoke run:

```text
Modal app: ap-ryLWOogXnTMMYmp1u4y6nX
mode: smoke
n_selected: 8
n_plots_written: 8
```

Full run:

```text
Modal app: ap-0F0GFZx7SBkP3vawCBV5RA
mode: full
n_selected: 16
n_plots_written: 16
```

## Artifacts

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_audit_report_full.json
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_audit_report_full.md
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_audit_index_full.html
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_audit_plot_index_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_plots_full/
```

Promoted primary submission:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_trace_gate_submission_full_new_worker_id.csv
```

Backup ID-format file:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/spike_hygiene_ablation/spike_hygiene_ablation_trace_gate_submission_full_worker_id.csv
```

## Automated Summary

| Group | Count | Plausible | Likely Artifact |
| --- | ---: | ---: | ---: |
| trace-gate top 10 | 10 | 10 | 0 |
| trace-gate added top 50 | 3 | 3 | 0 |
| current removed top 50 | 3 | 0 | 3 |

Overall verdict counts:

| Verdict | Count |
| --- | ---: |
| plausible_motion | 13 |
| likely_artifact | 3 |

Flag counts:

| Flag | Count |
| --- | ---: |
| clean_motion | 13 |
| spiky_or_extreme | 3 |

## Manual Inspection Notes

I inspected the generated plots directly.

Trace-gate top 10:

- The new top 10 are not simply low-motion or bland.
- Several clips contain meaningful events, orientation changes, or sustained
  motion variation.
- Some top-10 rows still have sharp motion, but the activity is distributed or
  physically structured rather than a few isolated bursts dominating a quiet
  clip.

Added trace-gate top-50 replacements:

- `audit 011`, original rank 51 / trace rank 48: plausible, active throughout,
  no obvious saturation or isolated spike artifact.
- `audit 012`, original rank 52 / trace rank 49: quieter than audit 011 but
  still valid; not stationary, not spiky, and clean by trace metrics.
- `audit 013`, original rank 53 / trace rank 50: contains a large late motion
  event and is close to the max-abs gate, but it remains below the physical
  threshold and looks like a coherent motion/contact event rather than corrupt
  sensor behavior.

Removed current top-50 clips:

- `audit 014`, original rank 2: dominated by early spike bursts and then mostly
  quiet behavior. This is exactly the kind of clip that can inflate novelty.
- `audit 015`, original rank 6: repeated high-jerk acceleration bursts in short
  windows, followed by long quiet periods.
- `audit 016`, original rank 35: one concentrated burst dominates an otherwise
  quiet clip.

## Interpretation

This targeted check resolves the open question from the spike-hygiene ablation.
The trace-gate replacements are not obviously bad, and the removed clips match
the failure mode seen in the visual audit: spike-driven novelty passing the old
quality gates.

The right final claim is:

```text
Partial-TS2Vec / exact-window blended k-center selector with trace-aware hygiene rerank.
```

This should replace the plain exact-window file as the primary output. Keep the
plain exact-window artifact as provenance and fallback.

## Remaining Caveat

This does not fix the remaining TS2Vec limitation: old-support TS2Vec is still
partial. The promoted artifact is cleaner at the top of the ranking, but it is
still not an exact full-support TS2Vec system.
