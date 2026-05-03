# Trace-Gate Active-Loop Evaluation

Date: 2026-05-03

## Verdict

Trace-gate is a conservative hygiene trade-off, not a coverage improvement.

In the 64-episode active-loop eval, the trace-gate policy removed trace-failed
clips from the selected prefixes, but it gave up coverage relative to the plain
TS2Vec/window blend at K=10, K=25, K=50, and K=100. The drop is modest at K=10
and K=50, but it is real.

This supports using trace-gate when top-ranked artifact risk matters more than
maximizing simulated coverage. It does not support claiming trace-gate strictly
dominates the plain blend.

## Runs

Smoke:

```text
Modal app: ap-cENF9iv0EQ3n3v4Yh877XC
mode: smoke
n_episodes: 2
```

Full:

```text
Modal app: ap-KX8CV6MSL1F4VFehuc32YR
mode: full
n_episodes: 64
```

Artifacts:

```text
/artifacts/active/eval/trace_gate_ts2vec_window_blend_scale/coverage_gain_report_full.json
/artifacts/active/eval/trace_gate_ts2vec_window_blend_scale/coverage_gain_by_episode_full.csv
/artifacts/active/eval/trace_gate_ts2vec_window_blend_scale/topk_selection_audit_full.csv
```

The full run reused the existing 64-episode embedding cache:

```text
status: hit
n_clips: 26725
path: /artifacts/active/embedding_cache/ts2vec_candidate_scale/embeddings_79e4a1956e9c1c28b4d29c54.npz
```

Trace hygiene was computed for 10,024 unique candidate clips.

## Balanced Relative Gain

| Policy | K=5 | K=10 | K=25 | K=50 | K=100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| trace-gate blend a05 | 0.1150 | 0.1539 | 0.2137 | 0.2551 | 0.2983 |
| plain blend a05 | 0.1111 | 0.1605 | 0.2242 | 0.2722 | 0.3249 |
| k-center quality-gated | 0.0914 | 0.1176 | 0.1844 | 0.2379 | 0.2878 |
| window-shape cluster-cap | 0.0680 | 0.1115 | 0.2002 | 0.2568 | 0.3190 |
| old novelty only | 0.0520 | 0.0834 | 0.1649 | 0.2381 | 0.3101 |
| oracle greedy eval-only | 0.2532 | 0.3027 | 0.3520 | 0.3676 | 0.3680 |

Delta versus plain blend:

| K | Trace-Gate Minus Plain |
| ---: | ---: |
| 5 | +0.0039 |
| 10 | -0.0066 |
| 25 | -0.0105 |
| 50 | -0.0171 |
| 100 | -0.0266 |

## Selection Hygiene

Mean trace-fail and spike-fail rates:

| Policy | K | Trace Fail | Spike Fail | Duplicate Rate |
| --- | ---: | ---: | ---: | ---: |
| trace-gate blend a05 | 10 | 0.0000 | 0.0000 | 0.0109 |
| trace-gate blend a05 | 50 | 0.0000 | 0.0000 | 0.0228 |
| trace-gate blend a05 | 100 | 0.0000 | 0.0000 | 0.0169 |
| plain blend a05 | 10 | 0.1344 | 0.0703 | 0.0094 |
| plain blend a05 | 50 | 0.1559 | 0.1044 | 0.0234 |
| plain blend a05 | 100 | 0.1070 | 0.0698 | 0.0184 |
| window-shape cluster-cap | 10 | 0.0969 | 0.0250 | 0.0609 |
| window-shape cluster-cap | 50 | 0.0859 | 0.0350 | 0.0709 |

Both trace-gate and plain blend had 0.000 artifact and low-quality rates by the
older quality-score audit. The difference is specifically the stricter trace
hygiene signal.

## Interpretation

The visual audit found that some high-ranked exact-window clips were
spike-driven. This active-loop eval confirms that the trace gate removes that
failure mode aggressively. It also shows the gate is probably too broad for
coverage optimization because it demotes all `likely_artifact` and
`mostly_stationary` verdicts.

The likely next no-training improvement is to test a narrower gate:

```text
artifact-only trace gate = demote trace__verdict == "likely_artifact"
do not demote "mostly_stationary" unless it also fails spike/quality/physical gates
```

That may keep the top of the final ranking clean while recovering some of the
coverage lost by the current hard trace gate.

## Recommendation

This was superseded by the narrower artifact-gate follow-up in
`docs/artifact_gate_active_loop_eval_2026-05-03.md`.

Keep broad trace-gate as provenance for the artifact failure diagnosis, but do
not use it as the preferred selector unless the goal is to remove every
`mostly_stationary` clip from selected prefixes.

Keep the plain exact-window artifact as the coverage-forward fallback.

Do not claim trace-gate dominates plain blend. Claim it trades modest simulated
coverage for materially cleaner top-ranked selections.
