# Simple Baseline Reset Plan

Date: 2026-04-27

## Decision Context

The current `subcluster40` ranker is mechanically coherent but failed the new marginal-coverage scientific gate.

The full marginal-coverage run used 20,000 physical-source rows, 5,437 source groups, and 4 folds. At K=100:

| Policy | window mean/std | temporal-order | raw-shape |
| --- | ---: | ---: | ---: |
| `ranker` | 0.0426 | 0.0114 | 0.0521 |
| `random_high_quality` | 0.0123 | 0.0034 | 0.0618 |
| `quality_only` | 0.0525 | 0.0059 | 0.0775 |
| `old_novelty_only` | 0.0766 | 0.0173 | 0.0572 |
| `diverse_source_cluster` | 0.0185 | 0.0023 | 0.0383 |

This means the current stack does not beat simple baselines across non-ranker representations. It should be treated as an overcomplicated heuristic, not a validated marginal-value algorithm.

Claude review artifact:

```text
.omx/artifacts/claude-simple-baseline-redesign-review-20260427.md
```

## Core Problem Restatement

We do not need to infer a hidden grammar of motion yet.

The core selection problem is:

```text
Pick new IMU clips that are clean enough to be usable and cover behavior not already well represented in old support.
```

That reduces the next system to two primary signals:

1. **Quality**: avoid corrupted, flat, saturated, or mostly-useless clips.
2. **Old-support novelty**: prefer clips far from the existing old corpus.

Everything else should be paused until it beats this simpler framing.

## Principles

1. **No new representational machinery.** Use existing cached `window_mean_std_pool` embeddings and raw quality features.
2. **One score, one optional diversity pass.** Avoid multi-stage score promotion, grammar, density, and tiered caps.
3. **Validate with marginal coverage, not candidate P@K.** The eval question is whether selected clips improve held-out target coverage.
4. **Prefer explainability over cleverness.** A reviewer should be able to understand the ranking rule in one sentence.
5. **Only add complexity after a failed simple baseline.** Complexity must earn its place by beating the simple baseline on predeclared gates.

## Options

### Option A: Quality Only

Rank by `quality_score`.

Pros:

- strongest raw-shape coverage in the last eval
- simplest possible policy
- excellent corruption resistance

Cons:

- does not ask whether old support already covers the clip
- likely oversamples clean but redundant motion

Use as a baseline, not the leading candidate.

### Option B: Quality Times Old Novelty

Rank by:

```text
final_score = quality_score * old_novelty_score
```

Implementation mapping:

- set `ranking.novelty_weight = 1.0`
- set `ranking.reranker_method = "mmr"`
- set `ranking.mmr_lambda = 0.0`
- disable grammar features
- disable score guards
- disable large cluster splitting

Why this works with current code:

- `build_scored_rows` computes `final_score = quality_score * ranker_score` in [baseline_ranker.py](/Users/vishnukrishnan/Developer/active-learning-v2/marginal_value/ranking/baseline_ranker.py:441).
- with `novelty_weight = 1.0`, `ranker_score` becomes old-support novelty only.
- `reranker_method = "mmr"` is accepted by config validation in [config.py](/Users/vishnukrishnan/Developer/active-learning-v2/marginal_value/ranking/config.py:85).
- `_rank_rows` falls through to `mmr_rank_rows`; with `mmr_lambda = 0.0`, it sorts by `final_score` without redundancy penalty in [modal_baseline_rank.py](/Users/vishnukrishnan/Developer/active-learning-v2/marginal_value/ranking/modal_baseline_rank.py:1199).

Pros:

- directly matches the core problem
- removes new-batch density, grammar, and subcluster tuning
- cheap to run on Modal
- reproducible and hidden-test friendly

Cons:

- may still concentrate near one source or feature mode
- may lose some raw-shape coverage relative to pure quality

This is the recommended first candidate.

### Option C: Quality Times Old Novelty With One Cap

Same score as Option B, then apply one simple cluster cap:

```text
cluster_cap_top_k = 200
cluster_max_per_cluster = 5
cluster_cap_key = "new_cluster_id"
cluster_bonus_weight = 0.0
mmr_lambda = 0.0
```

Pros:

- protects against repeated near-duplicates in top ranks
- still much simpler than `subcluster40`

Cons:

- cluster diversity is measured in the same representation used by old novelty
- previous `diverse_source_cluster` policy was weak, so cap should be treated as a risk, not an assumed improvement

This is a challenger, not the primary candidate.

## Recommended Design

Build and evaluate exactly two configs:

```text
configs/baseline_ranking_quality_oldnovelty_simple.json
configs/baseline_ranking_quality_oldnovelty_cap5.json
```

Do not add new Python ranking code unless config validation blocks the intended behavior.

The simple config should be submission-capable:

- `run_mode = "submission"`
- `support_split = "pretrain"`
- `query_split = "new"`
- `negative_split = "pretrain"`
- grammar disabled
- learned ranker disabled
- score guards disabled
- large cluster split disabled
- corruption eval can remain enabled for diagnostics, but it must not be used as proof

The coverage eval configs should reuse the same ranking settings and run through:

```text
modal_marginal_coverage_eval.py
```

## Acceptance Gates

Promote a simple baseline only if it passes these gates on the full marginal-coverage eval:

1. Beats `random_high_quality` at K=200 in all three representations: `window_mean_std_pool`, `temporal_order`, and `raw_shape_stats`.
2. Beats `subcluster40` at K=100 in at least two of those three representations.
3. Does not lose badly to `quality_only` on raw-shape at K=100. A loss greater than 10% relative should block promotion.
4. Does not lose badly to `old_novelty_only` on temporal-order at K=100. A loss greater than 10% relative should block promotion.
5. Top-100 low-quality count remains zero or near zero.
6. Largest source/cluster concentration at top 100 is no worse than `subcluster40`.

If neither simple candidate passes, do not return to grammar immediately. First inspect why quality and old novelty conflict.

## What To Avoid

Do not do these next:

- train a new encoder
- tune grammar score weights
- revive new-batch density
- add another source-blocked P@K variant
- tune tiered subcluster caps against the new split
- treat corruption rejection as scientific validation
- build hidden-test packaging around the current complex stack

## Implementation Plan After Approval

1. Add two JSON configs for Option B and Option C.
2. Add or update tests that validate the configs:
   - grammar disabled
   - score guards disabled
   - large cluster split disabled for simple option
   - `novelty_weight = 1.0`
   - `reranker_method = "mmr"` and `mmr_lambda = 0.0` for simple option
   - cap settings for cap5 option
3. Run smoke marginal-coverage evals on Modal for both configs.
4. Run full marginal-coverage evals on Modal for both configs.
5. Pull reports locally and compare against the saved `subcluster40` report.
6. If one passes gates, run `modal_rank.py` for the selected submission config.
7. Finalize public IDs using the existing submission finalization flow.
8. Update project docs with the decision and evidence.

## Expected Outcome

The likely best next candidate is:

```text
quality_oldnovelty_simple
```

The cap5 candidate is worth testing only because duplicate concentration remains a real submission risk. If cap5 reduces coverage, drop it.

## Review Gate

The reset was approved in substance after external review convergence from Gemini/Perplexity and ChatGPT Pro. The implemented primary rule is stricter than the original Claude comparator: quality is a hard gate, and old-support novelty is the score.

Implemented:

```text
quality_gated_old_novelty
quality_gated_old_novelty_sourcecap2
```

Full Modal marginal-coverage result:

```text
config: configs/marginal_coverage_eval_qgate_oldnovelty.json
report: /artifacts/eval/marginal_coverage/qgate_oldnovelty/marginal_coverage_report_full.json
local: data/modal_reports/qgate_oldnovelty/marginal_coverage_report_full.json
```

Important result:

| Policy | K | temporal-order | raw-shape | primary avg |
|---|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 |
| `quality_gated_old_novelty_q45` | 100 | 0.0173 | 0.0572 | 0.0372 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 |
| `quality_gated_old_novelty_q45` | 200 | 0.0199 | 0.0705 | 0.0452 |

Decision:

```text
The quality-gated novelty reset did not beat quality_only on the primary non-ranker aggregate.
```

The gate mostly acts as a safety floor, not as a coverage improver, because old-novelty already selects mostly high-quality clips. Sourcecap2 reduces concentration but slightly hurts coverage. The next move should not be adding back grammar/density/subclusters; it should be either accepting `quality_only` as the currently strongest honest baseline or redesigning the novelty representation.

Submission-capable execution configs:

```text
configs/baseline_ranking_new_quality_only.json
configs/baseline_ranking_new_qgate_oldnovelty_knn5.json
```
