# Submission Readiness Decision

Date: 2026-04-26

## Decision

Promote this as the current submission candidate:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40.json
```

Do not build another model or run more training before submission-readiness work. The next work should be packaging and final sanity checks around this candidate.

## Why This Changed

The previous conservative default was `stationary_guard` because `tiered_childcap2_5_subcluster40` looked like it might be over-optimizing easy pseudo metrics while worsening parent-level concentration.

The full source-blocked validation changed that. On the harder protocol, `subcluster40` is not just a pseudo-eval winner; it also recovers held-out source/cluster positives better than the safer parent-prefix candidate.

| Candidate | Source-blocked P@100 | Source-blocked nDCG@100 | Source-blocked P@200 | Corruption@100/@200 |
| --- | ---: | ---: | ---: | ---: |
| `stationary_guard` | 0.250 | 0.215 | 0.571 | 0 / 0 |
| `tiered_childcap2_5_subcluster40` | 0.325 | 0.290 | 0.554 | 0 / 0 |

Source-blocked fold P@100:

| Fold | `stationary_guard` | `subcluster40` |
| ---: | ---: | ---: |
| 0 | 0.31 | 0.43 |
| 1 | 0.24 | 0.32 |
| 2 | 0.19 | 0.23 |
| 3 | 0.26 | 0.32 |

The challenger wins every fold at P@100 and nDCG@100. Its P@200 is slightly lower, which means it is more front-loaded rather than uniformly better through the whole top 200. For a top-ranked submission, that is acceptable.

## New-Split Risk Audit

The real new-split ranking audit still shows the known tradeoff: `subcluster40` improves child-cluster diversity and pseudo metrics, while parent concentration is slightly worse.

| Metric | `stationary_guard` | `subcluster40` | Read |
| --- | ---: | ---: | --- |
| pseudo P@100 | 0.94 | 0.96 | challenger better, but not decision-grade |
| pseudo nDCG@100 | 0.946 | 0.962 | challenger better, but not decision-grade |
| top-100 low-quality count | 0 | 0 | tied |
| top-100 corruption negatives | 0 | 0 | tied |
| top-100 mean quality | 0.988 | 0.990 | tied |
| top-100 child clusters | 46 | 49 | challenger better |
| top-100 child max count | 8 | 5 | challenger better |
| top-100 parent clusters | 32 | 28 | safer candidate better |
| top-100 largest parent fraction | 0.44 | 0.50 | safer candidate better |
| top-200 child clusters | 56 | 74 | challenger better |
| top-200 child max count | 8 | 5 | challenger better |
| top-200 parent clusters | 32 | 30 | safer candidate slightly better |
| top-200 largest parent fraction | 0.53 | 0.56 | safer candidate slightly better |

The parent concentration is real, but not disqualifying:

- top-100 largest parent increases from 44 clips to 50 clips
- top-200 largest parent increases from 106 clips to 112 clips
- the dominant parent in `subcluster40` spans 36 split child clusters, not one uncapped duplicate child cluster
- top-100 and top-200 low-quality/corruption counts remain zero
- no duplicate worker IDs are present in the submission file

This looks like a broad high-support parent workflow, not a single broken duplicate mode.

## Artifacts Reviewed

Source-blocked full reports:

```text
/artifacts/eval/source_blocked/window_mean_std_physical_source/source_blocked_eval_full.json
/artifacts/eval/source_blocked/window_mean_std_physical_source_tiered_childcap2_5_subcluster40/source_blocked_eval_full.json
```

New-split ranking audits:

```text
/artifacts/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/baseline_audit_full.json
/artifacts/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40/baseline_audit_full.json
```

Visual audit reports exist for both candidates:

```text
/artifacts/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/visual_audit/top_clip_visual_audit_report_full.json
/artifacts/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40/visual_audit/top_clip_visual_audit_report_full.json
```

Local copies of the compact JSON evidence were pulled into:

```text
data/decision_audit_2026_04_26/raw/
```

## Recommendation

Use `tiered_childcap2_5_subcluster40` as the leading submission candidate.

Do not build a hybrid parent-cap variant now. A hybrid would be worth building only if manual review of the dominant parent plots shows repeated trivial motion or sensor artifacts. The current numeric evidence says the parent concentration is an acceptable price for the source-blocked validation gain.

## Final Checks Before Packaging

Completed packaging checks:

1. Downloaded the existing subcluster40 visual audit HTML, top rows, dominant-parent rows, report, plot index, and trace plots.
2. Built local contact sheets for the top-12 and dominant-parent examples.
3. Inspected the dominant-parent traces; they look varied and do not show one repeated flatline, saturation, or obvious sensor-artifact mode.
4. Finalized public-ID CSVs from the internal hash-ID submission and `new_urls.txt`.
5. Confirmed both finalized CSVs have 2,000 rows, 2,000 unique public IDs, no internal 64-character hash IDs, contiguous ranks, non-increasing scores, and top-100 low-quality count of zero.

Final packaged CSVs:

```text
data/submissions/subcluster40_2026_04_26/submission_subcluster40_worker_id.csv
data/submissions/subcluster40_2026_04_26/submission_subcluster40_new_worker_id.csv
```

Package README:

```text
data/submissions/subcluster40_2026_04_26/README.md
```
