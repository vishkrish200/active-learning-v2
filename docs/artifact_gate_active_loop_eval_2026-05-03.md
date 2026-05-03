# Artifact-Gate Active-Loop Evaluation

Date: 2026-05-03

## Verdict

The artifact-only gate is the better no-training hygiene trade-off.

Compared with the broad trace-gate policy, it still removes `likely_artifact`
and spike-driven selections from the selected prefixes, but it does not demote
`mostly_stationary` clips just because they are low-motion. That recovered most
of the coverage lost by the broad gate.

Compared with the plain TS2Vec/window blend, artifact-gate is effectively tied
at K=10 and K=25, better at K=5, and lower at K=50/K=100. It is therefore not a
strict coverage win over the plain blend. It is a cleaner primary artifact if
we care about top-ranked artifact risk.

## Runs

Latest confirmation rerun:

```text
Smoke Modal app: ap-EbPqkaoqTWFajRhzdGhrnq
Full Modal app: ap-A3aimCvvcwJ8Cq2nanY5Fj
mode: full
n_episodes: 64
coverage_rows: 13440
selection_audit_rows: 2240
```

The latest full rerun reused the existing 64-episode embedding cache:

```text
status: hit
n_clips: 26725
path: /artifacts/active/embedding_cache/ts2vec_candidate_scale/embeddings_79e4a1956e9c1c28b4d29c54.npz
trace_hygiene_cache_size: 10024
```

The rerun reproduced the same policy table and decision below. No model
training or embedding recomputation was launched.

Earlier provenance run:

Smoke:

```text
Modal app: ap-McM2pY9FSHzEdIQmLnjrug
mode: smoke
n_episodes: 2
```

Full:

```text
Modal app: ap-KA6jZzD4Jbj8carkMgK9xh
mode: full
n_episodes: 64
```

Artifacts:

```text
/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale/coverage_gain_report_full.json
/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale/coverage_gain_by_episode_full.csv
/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale/topk_selection_audit_full.csv
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
| artifact-gate blend a05 | 0.1157 | 0.1604 | 0.2237 | 0.2666 | 0.3109 |
| broad trace-gate blend a05 | 0.1150 | 0.1539 | 0.2137 | 0.2551 | 0.2983 |
| plain blend a05 | 0.1111 | 0.1605 | 0.2242 | 0.2722 | 0.3249 |
| k-center quality-gated | 0.0914 | 0.1176 | 0.1844 | 0.2379 | 0.2878 |
| window-shape cluster-cap | 0.0680 | 0.1115 | 0.2002 | 0.2568 | 0.3190 |
| old novelty only | 0.0520 | 0.0834 | 0.1649 | 0.2381 | 0.3101 |
| oracle greedy eval-only | 0.2532 | 0.3027 | 0.3520 | 0.3676 | 0.3680 |

Delta versus plain blend:

| K | Artifact-Gate Minus Plain |
| ---: | ---: |
| 5 | +0.0046 |
| 10 | -0.0001 |
| 25 | -0.0005 |
| 50 | -0.0055 |
| 100 | -0.0140 |

Delta versus broad trace-gate:

| K | Artifact-Gate Minus Broad Trace-Gate |
| ---: | ---: |
| 5 | +0.0007 |
| 10 | +0.0065 |
| 25 | +0.0100 |
| 50 | +0.0115 |
| 100 | +0.0126 |

Episode-level win counts:

| K | Artifact >= Plain | Artifact >= Broad Trace |
| ---: | ---: | ---: |
| 5 | 60/64 | 60/64 |
| 10 | 52/64 | 54/64 |
| 25 | 36/64 | 52/64 |
| 50 | 27/64 | 50/64 |
| 100 | 25/64 | 45/64 |

## Selection Hygiene

Mean selected-prefix hygiene:

| Policy | K | Likely Artifact Fail | Broad Trace Fail | Spike Fail | Duplicate Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| artifact-gate blend a05 | 10 | 0.0000 | 0.0438 | 0.0000 | 0.0094 |
| artifact-gate blend a05 | 50 | 0.0000 | 0.0297 | 0.0000 | 0.0250 |
| artifact-gate blend a05 | 100 | 0.0000 | 0.0181 | 0.0000 | 0.0189 |
| broad trace-gate blend a05 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0109 |
| broad trace-gate blend a05 | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0228 |
| broad trace-gate blend a05 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0169 |
| plain blend a05 | 10 | 0.0922 | 0.1344 | 0.0703 | 0.0094 |
| plain blend a05 | 50 | 0.1284 | 0.1559 | 0.1044 | 0.0234 |
| plain blend a05 | 100 | 0.0898 | 0.1070 | 0.0698 | 0.0184 |

Both artifact-gate and broad trace-gate had 0.000 quality and low-quality rates
by the older quality-score audit. The difference is that artifact-gate allows
some `mostly_stationary` rows through when they are not likely artifacts.

## Final 2,000-Clip Artifact

The artifact-only final-rerank artifact was generated from the already enriched
exact-window diagnostics, so it did not recompute raw trace features.

Run:

```text
Modal app: ap-rE7FBFbn2JLHbDrjdSa2zt
mode: full
n_rows: 2000
```

Artifacts:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_report_full.json
```

On the current 2,000 new candidates, artifact-gate and broad trace-gate produce
the same top-K hygiene summary:

| Variant | K | Spike Fail | Broad Trace Fail | Likely Artifact Fail | Current Overlap | Unique Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| artifact-gate | 10 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 10 |
| artifact-gate | 50 | 0.0000 | 0.0000 | 0.0000 | 0.9400 | 50 |
| artifact-gate | 100 | 0.0000 | 0.0000 | 0.0000 | 0.9600 | 100 |
| artifact-gate | 200 | 0.0000 | 0.0000 | 0.0000 | 0.9750 | 199 |

This means the narrower rule is not changing the currently promoted top-K, but
it is better justified by the active-loop eval because it avoids demoting clean
low-motion clips in simulated episodes.

## Recommendation

Promote `artifact_gate` over the broad `trace_gate` as the current conservative
primary. Keep the plain exact-window blend as the coverage-forward fallback.

Do not claim artifact-gate dominates the plain blend on coverage. The accurate
claim is narrower and stronger: it removes likely artifact selections with much
less simulated coverage loss than the broad trace-gate rule.
