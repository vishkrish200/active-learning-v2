# Skeptical System Audit

Date: 2026-04-25

This audit treats the current system as guilty until proven useful. The purpose is not to restate that the pipeline runs; it is to identify where the ranking could be fooling us.

## Current Candidate Under Audit

The current best candidate is:

```text
representation: window_mean_std_pool
score_variant: quality_gated_grammar
grammar score_weight: 0.3
support_split: pretrain
query_split: new
reranker: cluster_cap
large_cluster_split: enabled
```

Artifacts:

```text
data/submissions/worker_coverage_w03/
```

The previous candidate was the same system with `grammar score_weight = 1.0`:

```text
data/submissions/worker_coverage_final/
```

## Executive Skeptical Read

The `w0.3` candidate is more defensible than `w1.0`, but less proven than the headline metrics imply.

What improved:

- top-200 old-support novelty is much higher than `w1.0`
- top-200 quality remains high
- corruption count in the current diagnostic artifacts remains zero
- top-200 cluster coverage improves under the existing cluster labels

What became suspicious:

- the pseudo-eval metrics remain too easy to trust
- the large-cluster split inflates cluster diversity by creating artificial cluster IDs
- the corruption negatives mostly prove that hard-coded low-quality rows are downweighted
- the old corpus support currently covers 13,273 cached pretrain rows, not the original plan's 1.15M old 3-minute windows
- grammar features show saturation, especially rare phrase and longest unseen phrase metrics

## Finding 1: Perfect Pseudo-Eval Is Not Decision-Grade

Severity: high

The ranking report shows perfect or near-perfect pseudo metrics for the new-split run:

```text
precision@100 = 1.0
nDCG@100 = 1.0
```

This is not strong evidence of hidden challenge performance. In `marginal_value/ranking/modal_baseline_rank.py`, candidate eval is assembled as:

```text
positives = query rows
negatives = sampled pretrain rows
support = pretrain rows
```

For `query_split = new`, this mostly measures whether the ranker separates new samples from support-split samples. Since old-support novelty is computed against pretrain, the eval is structurally favorable.

Impact:

The pseudo-eval is useful as a pipeline sanity check, but not as a readiness proof.

Recommendation:

Report these metrics as "candidate membership sanity metrics," not challenge validation. Use top-K diagnostics, held-out cluster/phrase evals, and manual/top-clip audits for decisions.

## Finding 2: Large-Cluster Split Inflates Diversity Metrics

Severity: high

The implementation now reports parent-cluster diversity in addition to artificial post-split diversity.

For `w0.3`:

| K | post-split clusters | parent clusters | largest parent fraction |
|---:|---:|---:|---:|
| 50 | 30 | 24 | 0.38 |
| 100 | 42 | 28 | 0.47 |
| 200 | 54 | 30 | 0.54 |

For `w1.0`:

| K | post-split clusters | parent clusters | largest parent fraction |
|---:|---:|---:|---:|
| 50 | 23 | 5 | 0.44 |
| 100 | 29 | 5 | 0.59 |
| 200 | 32 | 8 | 0.705 |

This means `w0.3` is genuinely better than `w1.0`, but its diversity is not as strong as the old post-split number suggested.

Root cause:

`split_large_clusters` currently uses a deterministic score round-robin split of mega-clusters. This helps enforce a cap, but it does not prove that the resulting clusters are semantically distinct motion modes.

Impact:

`unique_clusters@K` can look healthier than the true parent-cluster diversity.

Recommendation:

Use parent-cluster metrics as the conservative diversity measure until mega-cluster splitting is replaced with feature-space subclustering. A real split should cluster within the mega-cluster using pooled IMU features plus grammar features, not merely round-robin by score.

Follow-up implemented:

- `cluster_cap_rank_rows` now accepts `cluster_key`.
- A new parent-cap config uses `ranking.cluster_cap_key = "new_cluster_parent_id"`.
- Diversity caps now support `cluster_cap_min_quality`, so the cap cannot promote low-quality singleton artifacts.
- A hybrid reranker now uses a parent-cluster cap for the first 75 ranks, then fills with the child-cluster cap.
- Selection traces now expose `cluster_cap_key`, `cluster_cap_cluster_id`, and `new_cluster_parent_id`.

This does not make the mega-cluster split semantically real; it makes the reranker stop treating those artificial subclusters as independent for top-K caps.

Follow-up result:

Strict parent-cap was not a clean replacement. It improved top-50 parent diversity, but by top-200 the parent cap exhausted and the giant parent cluster came back harder. The safer candidate is the hybrid prefix:

```text
config: configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_hybrid75.json
reranker: parent_prefix_cluster_cap
prefix: first 75 ranks capped by new_cluster_parent_id
fill: remaining ranks capped by new_cluster_id
quality floor for cap eligibility: 0.45
```

Conservative top-K comparison:

| Metric | child-cap `w0.3` | strict parent-cap | hybrid75 |
|---|---:|---:|---:|
| parent largest fraction @50 | 0.38 | 0.16 | 0.16 |
| unique parent clusters @50 | 24 | 28 | 28 |
| parent largest fraction @100 | 0.47 | 0.45 | 0.44 |
| unique parent clusters @100 | 28 | 32 | 32 |
| parent largest fraction @200 | 0.54 | 0.61 | 0.53 |
| unique parent clusters @200 | 30 | 32 | 32 |
| low-quality count @200 | 0 | 0 | 0 |
| candidate corruption count @100 | 0 | 0 | 0 |

The hybrid is not a miracle; the mega-cluster still exists. But it is the best current tradeoff among the tested rerankers because it improves early parent diversity without making top-200 parent concentration worse.

## Finding 3: The Corruption Eval Is Too Easy

Severity: high

The current corruption eval injects embedding-space corruptions and then assigns them:

```text
quality_score = 0.05
label = 0
is_corruption = true
```

This catches rankers that ignore quality entirely, but it does not prove that the quality model itself detects real sensor corruption. It also does not test whether tokenization and grammar features respond correctly to corrupted raw IMU.

Impact:

`corruption_rate@K = 0.0` is reassuring but partly tautological.

Recommendation:

Add raw-signal corruption evaluation:

- corrupt raw IMU JSONL or feature windows
- recompute quality from the corrupted signal
- recompute pooled features
- optionally recompute token/grammar features
- then rank without hard-coding quality to 0.05

This should become the real artifact-safety check.

## Finding 4: Old-Corpus Support Is Smaller Than the Original Plan

Severity: high

The plan expected old 3-minute windows:

```text
10,000 workers x 115 windows/worker = about 1.15M old windows
```

The current full ranking report uses:

```text
n_support = 13,273
n_query = 2,000
```

That may be all that is currently cached, but it is not the same old-corpus support model described in the plan.

Impact:

Old-corpus novelty is measured against the available cached support set, not necessarily the full existing dataset. If the cached support underrepresents workflows, some clips may rank high simply because the old support is sparse.

Recommendation:

Before final confidence, run a Modal coverage audit:

- count available old raw files
- count cached pretrain feature files
- estimate windows represented per worker
- report what fraction of the intended old corpus is in the support index

Do not call the current old-support model "full old corpus" unless that audit passes.

Follow-up completed on Modal:

```text
Run URL: https://modal.com/apps/vishkrish200/main/ap-L3gBiAyZmB1iDGUrzHvr7W
Report: data/audits/support_coverage_worker_coverage/support_coverage_worker_coverage/support_coverage_audit_full.json
```

Coverage results:

| Split | Manifest URLs | Manifest workers | Source exists | Cached raw+features |
|---|---:|---:|---:|---:|
| new | 2,000 | 2,000 | 2,000 | 2,000 |
| pretrain | 200,000 | 10,000 | 68,210 | 13,273 |
| val | 0 | 0 | 0 | 0 |

The target cache has:

```text
feature files: 15,273
raw files: 15,273
orphan files: 0
```

Relative to the plan's 1.15M old 3-minute windows, the current cached pretrain support is:

```text
13,273 / 1,150,000 = 1.154%
```

Relative to the 200,000 pretrain manifest URLs currently available to the pipeline, it is:

```text
13,273 / 200,000 = 6.6365%
```

This means the current ranking is valid only as a ranking against the available cached support slice. It should not be described as a full 10,000-worker-hour old-corpus support model.

## Finding 5: `w0.3` Is Better Than `w1.0`, But Still Dominated By Parent Clusters

Severity: medium-high

The conservative score blend is still the better candidate:

| Metric @200 | `w1.0` | `w0.3` |
|---|---:|---:|
| mean quality | 0.9978 | 0.9934 |
| mean old-support novelty | 0.0283 | 0.1056 |
| mean new-batch support | 0.9624 | 0.8840 |
| post-split unique clusters | 32 | 54 |
| parent unique clusters | 8 | 30 |
| largest parent fraction | 0.705 | 0.54 |

The important correction is that the top-200 still has 108 of 200 clips from one original parent cluster.

Recommendation:

Do not submit on the claim "unique_clusters@200 = 54" alone. If submitting now, describe the candidate as "better but still parent-cluster concentrated." Next ranking run should optimize parent-cluster diversity directly.

## Finding 6: Grammar Features Are Saturated

Severity: medium

Current new-split grammar feature distribution:

```text
rare_phrase_fraction:
  p50 = 0.8947
  p90 = 1.0
  p99 = 1.0

longest_unseen_phrase_len:
  p50 = 5
  p90 = 5
  p99 = 5
  max = 5
```

`longest_unseen_phrase_len` is capped by the n-gram implementation's `order + 2`, so it has little dynamic range. `rare_phrase_fraction` is often near 1.0, which means it may not discriminate well.

Impact:

The grammar score may be driven mostly by token NLL and transition NLL, not by rich phrase novelty.

Recommendation:

Keep grammar useful, but treat current grammar as a lightweight n-gram feature, not the full "motion language model" from the plan. Before building a transformer LM, first improve token/phrase diagnostics and reduce saturation.

## Finding 7: Score Normalization Is Batch-Relative

Severity: medium

`old_novelty_score` and `new_density_score` are min-max normalized over the current candidate set. This is fine for ordering within one batch, but not calibrated across runs.

Impact:

Scores from `w1.0`, `w0.3`, and future runs should not be interpreted as absolute marginal value probabilities.

Recommendation:

Use raw old kNN distances and score quantiles in diagnostics. For future scoring, consider robust percentile scaling fit on pseudo-holdout folds rather than per-run min-max scaling.

## Finding 8: Reason Codes Are Helpful But Not Ground Truth

Severity: medium

Top-200 for `w0.3` has:

```text
HIGH_NOVELTY_SINGLETON: 2
RARE_TEMPORAL_COMPOSITION: 78
RARE_MOTION_PRIMITIVES: 8
REDUNDANT_KNOWN_WORKFLOW: 112
```

The reason code thresholds depend on normalized novelty, density, and grammar promotion deltas. They are useful for triage but should not be treated as semantic labels.

Recommendation:

For final reporting, include reason-code distribution but pair it with feature means and cluster/parent-cluster metrics.

## Finding 9: The Encoder Branch Should Stay Demoted

Severity: medium

The trained SSL encoder passed geometry checks but failed ranking utility:

```text
smoke nDCG@100: 0.2100
smoke precision@100: 0.12
smoke corruption_rate@100: 0.48
```

Impact:

The encoder is not currently useful machinery for final ranking.

Recommendation:

Only revive it with a ranking-aware objective:

- supervised contrastive pseudo-holdout objective
- corruption/artifact head
- pairwise ranking loss against current best feature table

Do not spend more compute on the same reconstruction/VICReg-only objective.

## Code Changes Made During This Audit

The audit code now reports parent-cluster metrics:

- `unique_parent_cluster_count`
- `parent_largest_cluster_fraction`
- `parent_top_cluster_counts`
- `large_cluster_split_count`
- `large_cluster_split_fraction`

Updated files:

```text
marginal_value/eval/ranking_audit.py
marginal_value/ranking/audit_submission.py
tests/test_ranking_audit.py
tests/test_audit_submission.py
```

The support-coverage and parent-cap follow-up added:

```text
marginal_value/data/support_coverage_audit.py
modal_support_coverage_audit.py
configs/support_coverage_audit_worker_coverage.json
configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_parentcap.json
tests/test_support_coverage_audit.py
```

And updated:

```text
marginal_value/ranking/baseline_ranker.py
marginal_value/ranking/modal_baseline_rank.py
marginal_value/ranking/config.py
marginal_value/ranking/audit_submission.py
tests/test_baseline_ranking.py
```

Verification:

```text
python3 -m unittest tests.test_ranking_audit tests.test_audit_submission
python3 -m unittest discover -s tests
```

Result:

```text
184 tests OK
```

## Revised Recommendation

The current best candidate remains `w0.3`, not `w1.0`.

But after this skeptical pass, confidence should be framed carefully:

```text
w0.3 is the best current candidate among our implemented options.
It is not yet strongly proven against the original challenge objective.
```

Recommended next steps, in order:

1. Treat `hybrid75` as the current best reranker candidate, not strict parent-cap.
2. Add raw-signal corruption negatives so artifact safety is not hard-coded.
3. Add real feature-space subclustering for the largest parent cluster if we want another diversity gain.
4. Only then freeze the final submission candidate.
