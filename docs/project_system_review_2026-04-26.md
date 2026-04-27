# Project System Review

Date: 2026-04-26

This review asks whether the current IMU marginal-data-value project is actually solving the intended problem, not whether the pipeline merely runs.

## Executive Judgment

The project is solving a defensible proxy for marginal data value, but it has not yet proven that the proxy matches the hidden challenge objective.

The strongest part of the system is not the model complexity. It is the safety and audit harness: split discipline, full physical-source support expansion, raw-signal corruption validation, top-clip visual audit, parent/child diversity reporting, and conservative grammar weighting. Those are real improvements.

The weakest part is validation. The headline candidate-eval metrics are still structurally easy because the eval labels `new` rows as positives and sampled `pretrain` rows as negatives while the support index is also `pretrain`. The physical leave-cluster eval is more meaningful, but it validates recovery of synthetic feature-space clusters from old support. It is not yet a source/factory/workflow-style challenge validation, and it does not fully exercise the production new-split stack.

After the full source-blocked validation pass, the current submission candidate is `tiered_childcap2_5_subcluster40`. The earlier conservative preference for `stationary_guard` was reasonable before harder validation existed, but the challenger now wins blocked P@100 on every fold while keeping top-K corruption and low-quality counts at zero. Its parent concentration remains the main risk, but the concentration is modestly worse rather than disqualifying.

## Intended Objective

The intended target is not anomaly detection and not simply "different from old data." It is:

```text
useful_value ~= quality * old_corpus_novelty * new_batch_support * diversity
```

The project should rank new worker clips that are:

- physically plausible and high quality
- not already covered by the old corpus
- supported by more than one new-batch clip when possible
- diverse enough to avoid returning many near-duplicates from one workflow

The current production family is aligned with this objective in form:

```text
window_mean_std_pool
+ quality_gated_grammar, score_weight = 0.3
+ full physical-source pretrain support
+ raw-signal corruption validation
+ feature-space large-cluster split
+ stationary singleton guard
+ parent/child cluster-aware reranking
```

But formal alignment depends on validation, and that remains the open problem.

## Evidence Summary

The current artifacts support these conclusions:

| Candidate | P@100 | P@200 | corruption@100 | child clusters@200 | parent clusters@200 | largest parent@200 | old novelty@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stationary_guard` | 0.94 | 0.845 | 0.00 | 56 | 32 | 0.53 | 0.153 |
| `tiered_childcap2_5` | 0.90 | 0.830 | 0.00 | 56 | 32 | 0.50 | 0.122 |
| `tiered_childcap2_5_subcluster40` | 0.96 | 0.870 | 0.00 | 74 | 30 | 0.56 | 0.140 |

Physical leave-cluster validation:

| Metric | Value |
| --- | ---: |
| sampled old physical-source clips | 20,000 |
| source clusters | 64 |
| folds | 4 |
| mean precision@100 | 0.695 |
| mean precision@200 | 0.821 |
| mean nDCG@100 | 0.593 |
| mean nDCG@200 | 0.735 |
| mean quality | 0.974 |
| low-quality fraction | 0.0068 |

Interpretation:

- The ranker is not random. It recovers withheld old-corpus feature-space clusters at useful rates.
- Raw-corruption safety is materially better than the earlier hard-coded corruption eval.
- Top-K quality is consistently strong.
- Parent-cluster concentration remains the dominant risk.
- Pseudo-eval precision should not be used as the deciding metric.

## Main Findings

### 1. Candidate Eval Is Still Not Challenge Validation

In `marginal_value/ranking/modal_baseline_rank.py`, `support_rows`, `query_rows`, and `negative_rows` are selected from configured splits. The current submission configs set:

```json
"support_split": "pretrain",
"query_split": "new",
"negative_split": "pretrain"
```

Then candidate eval builds:

```text
positives = query_rows
negatives = negative_rows
support = support_embeddings
```

So the eval is mostly asking whether old-corpus novelty can separate `new` from `pretrain`, while `pretrain` is also the support index. That is useful as a sanity check, but it is not a hard validation of within-new marginal value.

Action:

- Stop treating candidate P@K/NDCG as submission-readiness metrics.
- Keep them only as pipeline sanity checks.
- Add a blocked validation where positives and negatives are both old-corpus rows withheld from the support index under source/worker/cluster rules.

### 2. Physical Leave-Cluster Eval Is Valuable But Incomplete

`marginal_value/eval/physical_leave_cluster_eval.py` clusters old physical-source embeddings, holds out clusters, removes them from support, and checks whether the ranker recovers them.

That is much better than `new` versus `pretrain` candidate eval. However:

- labels are defined by the same feature-space clustering family used by the ranker
- it uses old-support rows, not the real new split
- it validates the base handcrafted novelty/density stack more than the full production grammar/diversity candidate
- it is not source/factory/workflow blocked unless the synthetic clusters happen to match those concepts

Action:

- Evolve this into a source-blocked challenge validation rather than replacing it.
- Reuse its Modal execution pattern and artifact format.

### 3. Diversity Is Still The Hardest Product Decision

`tiered_childcap2_5_subcluster40` satisfies the child/subcluster cap cleanly through the top 200. That looks nice on child-cluster metrics.

But the finer subclusters give the dominant parent more valid slots:

```text
stationary_guard largest parent@100 = 0.44
stationary_guard largest parent@200 = 0.53
subcluster40 largest parent@100 = 0.50
subcluster40 largest parent@200 = 0.56
```

It also reduces top-K old-support novelty:

```text
stationary_guard old novelty@100 = 0.153
subcluster40 old novelty@100 = 0.140
stationary_guard old novelty@200 = 0.080
subcluster40 old novelty@200 = 0.073
```

Action:

- Do not promote `subcluster40` merely because pseudo P@100 is 0.96.
- Promote it because it also wins the full source-blocked validation pass.
- Keep `stationary_guard` as the fallback if visual review rejects the dominant-parent concentration.

### 4. Grammar Helps, But It Is Not A Proven Workflow Model

The current `quality_gated_grammar` blend is safer than the earlier `score_weight = 1.0` grammar-dominated candidate. At `score_weight = 0.3`, phase-A old novelty and new support still carry most of the score.

But grammar diagnostics remain suspicious:

```text
longest_unseen_phrase_len p10/p50/p90 = 5/5/5
rare_phrase_fraction mean = 0.862
token_nll_p95 has a tight central band
old_novelty correlation with token_nll_p95 ~= 0.112
new_density correlation with token_nll_p95 ~= -0.129
```

So grammar is currently a weak temporal-composition prior, not a decisive model of "new workflow composed of familiar atoms."

Action:

- Keep grammar at low weight.
- Do not spend the next iteration refitting a transformer grammar or SSL encoder.
- Refit/tokenize on full physical support only after validation proves grammar is the limiting factor.

### 5. Safety And Quality Work Is A Real Strength

The project made meaningful progress on avoiding artifact ranking:

- raw-signal corruptions recompute features and quality
- sparse impossible spikes were caught after adding `max_abs_value` and `extreme_value_fraction`
- top-100 low-quality count is zero for current candidates
- visual audit caught a stationary singleton failure
- the stationary guard fixed that one failure without broad score churn

This is evidence that the system is not merely anomaly hunting. It actively suppresses obvious artifact modes.

Action:

- Preserve these gates.
- Add decoys to validation that are high-quality but known/redundant, not only low-quality corruptions.

## Recommended Next Move

Do not run more training next. Do not tune more reranker variants against candidate eval.

The next real move should be a source-blocked challenge validation:

1. Parse source group metadata from manifest URLs, at minimum `workerNNNNN`; if no factory/workflow ID exists, use worker group plus feature-space clusters.
2. Build folds where support excludes all rows from selected held-out source groups and selected held-out feature clusters.
3. Create a candidate batch per fold:
   - positives: high-quality clips from held-out source/cluster regions
   - hard negatives: high-quality clips from known/source-covered regions
   - redundancy negatives: near-duplicates from already-covered clusters
   - raw corruptions: recomputed from raw signal, not hard-coded quality
4. Run the same scoring and reranking family used for the new-split candidates.
5. Compare `stationary_guard`, `tiered_childcap2_5`, and `subcluster40` on this protocol.
6. Select by blocked-validation performance first, then by parent diversity, then by pseudo candidate eval.

Suggested acceptance gates:

```text
raw corruption@100 = 0
low quality@100 = 0
blocked precision@100 is materially above random and stable across folds
blocked nDCG@100 does not collapse on any held-out source group
largest parent/source fraction@100 does not increase relative to stationary_guard
```

The exact numeric precision threshold should be set after the first run because the fold difficulty will change. The decision should be comparative: a candidate must beat `stationary_guard` on blocked validation or preserve its safety/diversity profile to replace it.

## Implemented Follow-Up

Added the first version of that harder validation protocol:

```text
marginal_value/eval/source_blocked_eval.py
modal_source_blocked_eval.py
configs/source_blocked_eval.json
configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
tests/test_source_blocked_eval.py
```

This eval differs from the old candidate eval in three important ways:

- positives and negatives both come from old physical-source pretrain rows
- all candidate rows are removed from the support index, so negatives are not exact self-neighbors
- positives come from held-out source groups inside held-out feature clusters, while hard negatives come from source-covered regions still represented in support

It also reuses the production ranking machinery:

- grammar score promotion when configured
- stationary singleton score guard
- feature-space large-cluster splitting
- parent-prefix or tiered cluster-cap reranking
- raw-signal corruption negatives when enabled

The initial configs compare the safer current default against the main child-diversity challenger:

```text
source_blocked_eval.json
source_blocked_eval_tiered_childcap2_5_subcluster40.json
```

Run on Modal:

```bash
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval.json
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
```

Use smoke mode first:

```bash
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval.json --smoke
```

First Modal smoke results:

| Config | Modal run | mean P@100 | mean nDCG@100 | corruption@100/@200 |
| --- | --- | ---: | ---: | ---: |
| `source_blocked_eval.json` | `ap-L9TKuzrnf0HTPFGsbzmVok` | 0.340 | 0.357 | 0 / 0 |
| `source_blocked_eval_tiered_childcap2_5_subcluster40.json` | `ap-3cOARTruuWnxmHYnbwuf8d` | 0.330 | 0.354 | 0 / 0 |

Smoke interpretation: the harder eval is much less comfortable than the old candidate eval, which is good. The child-diversity challenger does not beat the safer current config on the first source-blocked smoke.

Full Modal results:

| Config | Modal run | rows | source groups | source clusters | mean P@100 | mean nDCG@100 | report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source_blocked_eval.json` | `ap-1mLZqeVa7NoGswra05c1rC` | 20,000 | 5,437 | 64 | 0.250 | 0.215 | `/artifacts/eval/source_blocked/window_mean_std_physical_source/source_blocked_eval_full.json` |
| `source_blocked_eval_tiered_childcap2_5_subcluster40.json` | `ap-RyLA2XWehIEQXNfnxAtiWK` | 20,000 | 5,437 | 64 | 0.325 | 0.290 | `/artifacts/eval/source_blocked/window_mean_std_physical_source_tiered_childcap2_5_subcluster40/source_blocked_eval_full.json` |

Full-run interpretation: the source-blocked smoke result did not hold at scale. The `subcluster40` challenger wins the first harder blocked validation pass while keeping raw corruption out of top-100/top-200. That is meaningful evidence in favor of the challenger, but it should be balanced against its worse parent concentration on the real new split before treating it as submission-ready.

## Current Candidate Decision

Update after the full source-blocked run and submission-readiness decision audit:

```text
PROMOTE tiered_childcap2_5_subcluster40 as the current submission candidate.
```

Decision memo:

```text
docs/submission_readiness_decision_2026-04-26.md
```

The earlier conservative preference for `stationary_guard` was correct before blocked validation existed. After the full 20,000-row source-blocked comparison, `subcluster40` wins all four folds at P@100 and improves mean P@100 from `0.250` to `0.325`. Parent concentration is still worse, but the increase is modest relative to the validation gain and the dominant parent spans many child subclusters rather than one uncapped duplicate child cluster.

Use this as the working default:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40.json
```

Keep this as the safer fallback if visual inspection rejects the dominant parent concentration:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json
```

Do not promote:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_parentcap200_p8.json
```

## What This Project Is Now

This is a pragmatic, audited marginal-value ranker:

- handcrafted IMU representation
- old-support novelty
- new-batch support
- conservative grammar prior
- strong quality gate
- diversity reranking
- Modal-first heavy execution
- extensive diagnostics

It is not yet:

- a learned active-learning value model
- a fully validated workflow/factory transfer model
- a reliable proof that top pseudo metrics predict hidden challenge score
- a reason to run more SSL training

The highest-value next work is validation design, not modeling.
