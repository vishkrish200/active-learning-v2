# Final Submission Note

Date: 2026-05-02

## Direct Verdict

The current artifact is a credible external-challenge submission candidate. The
recommended output is now the artifact-gate exact-window blend rerank: partial
TS2Vec old-support novelty plus exact full-corpus `window_mean_std_pool`
old-support novelty, with a trace-aware hygiene rerank to demote
`likely_artifact` clips.

It should be described as a multi-view geometric acquisition selector with
partial TS2Vec support, not as a clean scientific TS2Vec active-learning result
or an exact full-support TS2Vec search.

Recommended label:

```text
Partial-TS2Vec / exact-window blended k-center selector with artifact-aware trace rerank
```

Final selector:

```text
artifact_gate_exact_window_blend_kcenter_ts2vec_window_mean_std_pool_a05
```

## Submission Files

Primary candidate file:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv
```

Backup ID-format file:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv
```

Diagnostics and report:

```text
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_diagnostics_full.csv
/artifacts/active/final_blend_rank/exact_full_window_a05/artifact_hygiene_ablation/spike_hygiene_ablation_report_full.json
/artifacts/active/final_blend_rank/exact_full_window_a05/trace_gate_targeted_audit/trace_gate_audit_report_full.json
```

Use the `new_worker_id` file as primary unless the evaluator explicitly expects
the internal manifest-hash `worker_id` column.

## Method Summary

We rank the 2,000 newly arrived IMU clips using a multi-view geometric
acquisition selector. Each candidate is embedded with a TS2Vec-style temporal
encoder and with a cheap handcrafted window-stat representation. Candidate
novelty is estimated against two old-support views: a partial cached TS2Vec
old-support index and an exact full-corpus `window_mean_std_pool` old-support
index over all 200,000 old clips. The two novelty scores are min-max normalized
and blended with `alpha=0.5`.

Before ranking, clips are filtered by hard hygiene gates:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
```

The final ordering uses a quality-gated blended k-center strategy to prioritize
clips that are both under-covered by the old corpus and nonredundant within the
new batch. It then applies an artifact-aware trace rerank that demotes clips
with `trace__verdict == "likely_artifact"` below trace-pass rows. This is a
conservative rerank rather than a new model: it preserves the exact-window
multi-view selector while preventing isolated sensor bursts from occupying the
top of the submission without demoting every low-motion clip.

This is not an exact full-200k TS2Vec search. It is exact for the window-stat
old-support view and partial for the TS2Vec old-support view.

## Exact Config

Final ranking config:

```text
configs/active_exact_window_blend_rank.json
```

Important settings:

```text
left_representation: ts2vec
right_representation: window_mean_std_pool
alpha: 0.5
left_support_shard_dir: /artifacts/active/embedding_cache/ts2vec_window_full_new/embeddings_7481b57ede264d17002b4316_shards
left_query_shard_dir: /artifacts/active/embedding_cache/ts2vec_window_new_only_h100/embeddings_5fcc8a7d4d3d16bf699786fa_shards
min_left_support_clips: 20000
right_support_shard_dir: /artifacts/active/full_support_shards/window_mean_std_v1
max_query_clips: 2500
quality_threshold: 0.85
max_stationary_fraction: 0.90
max_abs_value: 60.0
```

Window full-support shard config:

```text
configs/build_full_support_window_shards.json
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

Successful 3-seed consensus run, now fallback/provenance:

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

Successful exact-window final ranking:

```text
Modal app: ap-hivXtqllLALI9UBYyOBJ8e
mode: full
n_query: 2000
n_left_support: 22327
n_right_support: 200000
```

Full window shard build:

```text
Modal app: ap-Qavmxl6rQzJ9yHZd7wUDoM
n_clips: 202000
n_shards: 50
manifest: /artifacts/active/full_support_shards/window_mean_std_v1/full_support_shards_full.json
```

Exact-window top-k hygiene:

| K | Quality Fail | Physical Fail | Duplicate Rate | Unique New Clusters |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000 | 0.000 | 0.000 | 10 |
| 50 | 0.000 | 0.000 | 0.000 | 50 |
| 100 | 0.000 | 0.000 | 0.000 | 100 |
| 200 | 0.000 | 0.000 | 0.005 | 199 |

Exact-window comparison against 3-seed consensus:

| Metric | Value |
| --- | ---: |
| rank_spearman | 0.9846 |
| top10_overlap | 0.8000 |
| top50_overlap | 0.6800 |
| top100_overlap | 0.8100 |
| top200_overlap | 0.8300 |

Detailed exact-window results:

```text
docs/exact_full_window_results_2026-05-02.md
```

Top-K visual audit:

```text
docs/exact_window_topk_audit_2026-05-03.md
```

Spike-hygiene ablation:

```text
docs/spike_hygiene_ablation_2026-05-03.md
```

Trace-gate targeted audit:

```text
docs/trace_gate_targeted_audit_2026-05-03.md
```

Trace-gate active-loop evaluation:

```text
docs/trace_gate_active_loop_eval_2026-05-03.md
```

Artifact-gate active-loop evaluation:

```text
docs/artifact_gate_active_loop_eval_2026-05-03.md
```

The earlier broad trace-gate ablation removed all three `likely_artifact` clips
from the exact-window top 50 while preserving top-50 cluster diversity and
retaining `0.94` overlap with the exact-window top 50. The targeted replacement
audit passed: the trace-gate top 10 and the three promoted top-50 replacements
were all manually judged as plausible motion, while the three removed top-50
clips matched the spike-driven artifact failure mode. This validated the
artifact failure mode; the narrower artifact-gate rerank is now preferred
because it removes the same likely-artifact top-K issue without using the
over-broad `mostly_stationary` rule.

The active-loop eval shows this is a hygiene trade-off, not a strict coverage
win. On 64 source-blocked episodes, trace-gate blend a05 reduced trace-fail and
spike-fail rates to zero at K=10/K=50/K=100, but balanced relative gain dropped
from `0.1605` to `0.1539` at K=10 and from `0.2722` to `0.2551` at K=50 versus
the plain blend. Keep the plain exact-window artifact as the coverage-forward
fallback.

The narrower artifact-gate follow-up is now preferred over broad trace-gate.
It reduced likely-artifact and spike-fail selections to zero while recovering
most of the broad gate's coverage loss: balanced relative gain was `0.1604` at
K=10 and `0.2666` at K=50. It is effectively tied with the plain blend at
K=10/K=25, below plain at K=50/K=100, and consistently above broad trace-gate.
On the current 2,000 new candidates, artifact-gate and broad trace-gate produce
the same top-K hygiene summary, but artifact-gate is scientifically cleaner
because it does not treat every `mostly_stationary` clip as unacceptable.

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

3. TS2Vec old support is still the remaining support approximation.

   The window-stat view now uses all 200,000 old-support clips. The TS2Vec view
   still uses the partial 22,327-clip old-support cache. A full-support TS2Vec
   run should wait until the large-shard data path is extended to TS2Vec
   inference rather than being launched through small-file IO.

4. Old-support shard quality metadata is incomplete in the window-only shard.

   The final ranking still computes and enforces candidate-side quality and
   physical-validity gates. Do not claim old-support quality-balanced sampling
   from this shard alone.

5. External held-out score is unknown.

   Internal active-loop evaluation supports the blend selector, but the
   challenge evaluator remains the real test.

6. Artifact-gate still trades some coverage for stricter trace hygiene.

   The 64-episode artifact-gate active-loop eval shows zero likely-artifact and
   spike-fail selected prefixes, but lower balanced relative gain than the plain
   blend at K=50 and K=100. This is acceptable only if the final claim is
   hygiene-forward and conservative. Keep the plain exact-window output as the
   coverage-forward fallback.

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
sample-sensitive. The 3-seed consensus ranking remains useful provenance, but
the later exact-window full-support run plus artifact-gate audit is the current
primary recommendation.

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
physical failure rates at K=10, K=50, and K=100. The single-seed budgeted top 10
was not stable enough to claim an exact support-independent ordering. The later
exact-window full-support run replaced the sampled window-stat view, and the
artifact-gate rerank now addresses the remaining spike-driven top-rank hygiene
issue.

Detailed results:

```text
docs/support_sampling_stability_results.md
```

## Active-Loop Validation With Confidence Bands

The current artifact-gate recommendation is supported by the 64-episode
active-loop evaluation summarized in:

```text
docs/active_loop_validation_report_2026-05-03.md
docs/active_loop_validation_report_2026-05-03.json
```

The validation report adds bootstrap confidence intervals, oracle-fraction
summaries, selection-hygiene confidence bands, and episode-level win counts for
artifact-gate versus the plain blend, broad trace-gate, k-center quality-gated,
and window-shape cluster-cap baselines.

The conclusion is deliberately conservative: artifact-gate is promoted for
top-ranked trust and artifact avoidance, not because it is a strict coverage
winner over the plain blend at every K.

## Recommended Claim

Use this wording:

> We rank new IMU clips with a budgeted multi-view acquisition selector. The
> method combines TS2Vec novelty against a partial old-support cache with exact
> full-corpus window-stat novelty, applies hard sensor-quality gates, uses
> k-center-style reranking to reduce redundancy, and applies a conservative
> artifact-aware trace rerank to demote likely sensor artifacts.

Avoid:

- "validated TS2Vec active learning"
- "full-corpus learned marginal value"
- "exact full-200k TS2Vec novelty"

The original single-sample output remains useful as a provenance/fallback
artifact, but it is no longer the primary recommendation.

The plain exact-window output also remains useful as a provenance/fallback
artifact. The primary recommendation is the artifact-gate exact-window rerank.
