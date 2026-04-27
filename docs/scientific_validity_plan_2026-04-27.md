# Scientific Validity Plan

Date: 2026-04-27

## Bottom Line

We cannot honestly prove the current ranker is scientifically valid from the existing evidence.

What we can say today:

- the final CSV is mechanically valid
- the system is a coherent deterministic marginal-value heuristic
- source-blocked validation is better than the original new-vs-pretrain pseudo-eval
- but source-blocked P@100 is only about equal to the fold positive base rate for `subcluster40`
- the core validation is still partly self-referential because positives are defined in the same feature family used by the ranker

So the current scientific status is:

```text
plausible but unproven
```

The next work should be a falsification test, not another candidate tweak.

## What Would Count As Evidence?

The project claims that top-ranked clips have marginal data value. That means adding selected clips to an old corpus should improve coverage of held-out behavior more than simple baselines.

A useful scientific test should answer:

```text
If we add the ranker's top-K clips to old support, does a genuinely held-out target set become better covered?
```

Critically, the target-set coverage must be measured in representations that the ranker did not directly optimize.

## Current Evidence Against The Claim

### 1. Candidate Eval Is Not Evidence

The old candidate eval labels new rows as positives and pretrain rows as negatives while pretrain is also the support index.

That mostly tests:

```text
Can old-support novelty separate new from pretrain?
```

It does not test:

```text
Among new clips, which clips are most marginally useful?
```

### 2. Source-Blocked Eval Is Comparative But Weak

Full source-blocked results:

| Candidate | mean P@100 | mean fold positive rate baseline | mean P@100 minus baseline |
| --- | ---: | ---: | ---: |
| `stationary_guard` | 0.250 | about 0.320 | -0.070 |
| `subcluster40` | 0.325 | about 0.320 | +0.005 |

This means `subcluster40` beats `stationary_guard`, but its absolute P@100 is only roughly at random/base positive-rate level.

So source-blocked validation is not strong proof. It is only a weak comparative signal.

### 3. Representation Circularity Remains

The source-blocked folds are built from clusters in `window_mean_std_pool` feature space.

The ranker also uses `window_mean_std_pool` for old-support novelty and new-batch clustering.

That means the eval tests whether the ranker is consistent with its own geometry. This is useful, but it does not prove useful IMU behavior coverage.

### 4. Support Coverage Is A Scientific Risk

The ranking support index currently uses the cached physical-source support slice, not the full theoretical old corpus.

Known coverage:

```text
pretrain manifest URLs: 200,000
physical source exists: 68,210
cached raw/features: 13,273
```

If the support slice is biased, old-support novelty is biased.

## The Falsification Experiment We Should Build

Build a **marginal coverage validation** eval.

### Protocol

For each fold:

1. Start from old physical-source rows.
2. Select held-out source groups without using the ranker's feature clusters as the primary label.
3. Split held-out rows into:
   - candidate pool `C`
   - target set `T`
4. Remove all held-out source rows from old support `S_old`.
5. Rank only candidate pool `C` using the production ranker.
6. For each policy, choose top-K candidates:
   - current `subcluster40`
   - `stationary_guard`
   - random high-quality
   - quality-only
   - old-novelty-only
   - new-density-only
   - simple diverse-by-source/cluster baseline
7. Measure how much adding selected candidates improves nearest-neighbor coverage of target set `T`.

### Coverage Metric

For selected set `A_k`:

```text
coverage_gain@K =
  mean_nn_distance(T, S_old) - mean_nn_distance(T, S_old union A_k)
```

Normalized:

```text
relative_coverage_gain@K =
  coverage_gain@K / mean_nn_distance(T, S_old)
```

Higher is better.

This directly asks whether selected clips make the old corpus better at covering held-out behavior.

### Evaluation Representations

Coverage should be measured in multiple representations:

1. `window_mean_std_pool`
   - expected to favor the current ranker
   - useful as a sanity check

2. Temporal-order embedding
   - first/second/third segment means and stds
   - deltas between segments
   - burst counts and transition summaries
   - tests temporal structure lost by global mean/std pooling

3. Frequency/raw-shape embedding
   - spectral energy bands
   - autocorrelation summaries
   - jerk burst statistics
   - axis-energy ratios
   - tests whether selected clips cover raw signal shape, not only mean/std geometry

4. Grammar/token embedding
   - primitive histogram
   - rare n-gram counts
   - sequence-length and duration summaries
   - tests whether selected clips add temporal-composition coverage

The ranker only passes scientific smell if it improves held-out target coverage beyond baselines in representations it did not directly optimize.

## Acceptance Gates

Treat the current approach as scientifically supported only if:

1. `subcluster40` beats random high-quality by at least 10% relative coverage gain at K=100 in at least two non-ranker representations.
2. `subcluster40` beats quality-only and novelty-only baselines in at least two non-ranker representations.
3. Improvement is stable across folds, not driven by one fold.
4. Top-K low-quality count remains zero or near zero.
5. Dominant parent/source concentration does not erase coverage gains.
6. Results remain directionally similar when support size changes.

Treat the approach as scientifically suspect if:

1. gains only appear in `window_mean_std_pool`
2. gains disappear in temporal/frequency/grammar representations
3. random high-quality or simple diverse sampling matches the ranker
4. parent concentration drives most measured gain
5. results flip wildly by fold

## What Would Prove More?

The strongest possible evidence would be downstream utility:

```text
Train or adapt a downstream IMU model with S_old + selected top-K
Measure improvement on held-out source/workflow target sets
Compare against random/quality/diverse baselines
```

But this likely requires training and a downstream objective. It is valuable, but it is not the fastest next step.

The marginal coverage eval is the best immediate middle ground because it:

- does not require labels
- does not require a new model
- directly tests marginal corpus coverage
- can be run on Modal
- can falsify the current ranker quickly

## Implementation Plan

Add:

```text
marginal_value/eval/marginal_coverage_eval.py
modal_marginal_coverage_eval.py
configs/marginal_coverage_eval_subcluster40.json
tests/test_marginal_coverage_eval.py
```

Outputs:

```text
/artifacts/eval/marginal_coverage/<run_name>/marginal_coverage_report_full.json
/artifacts/eval/marginal_coverage/<run_name>/marginal_coverage_candidates_full.jsonl
```

Report should include:

- fold summaries
- target positive/source group summaries
- coverage gain by policy, K, and representation
- random baseline mean/std over several seeds
- parent/source concentration of selected sets
- quality and corruption summaries

## Decision After This Eval

If `subcluster40` wins marginal coverage in non-ranker representations:

```text
The current deterministic ranker has meaningful scientific support.
Build the hidden-test execution wrapper and submit/ship.
```

If it only wins in ranker space:

```text
The current system is self-consistent but not scientifically validated.
Submit CSV only if deadline requires it, but do not claim algorithmic validity.
```

If random/diverse baselines match or beat it:

```text
The current scoring stack is probably unnecessary.
Replace with a simpler quality + diversity baseline or revisit representation learning.
```

## Do Not Do Yet

- Do not train another SSL encoder before this falsification eval.
- Do not tune cluster caps against the current new split.
- Do not keep adding heuristic score terms.
- Do not call the current source-blocked result proof.

## Implementation Status

Implemented on 2026-04-27:

```text
marginal_value/eval/marginal_coverage_eval.py
modal_marginal_coverage_eval.py
configs/marginal_coverage_eval_subcluster40.json
tests/test_marginal_coverage_eval.py
```

The first Modal smoke run completed successfully:

```text
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_report_smoke.json
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_candidates_smoke.jsonl
```

Smoke result, one fold, 1,024 rows:

| Policy | K | window mean/std rel gain | temporal-order rel gain | raw-shape rel gain |
| --- | ---: | ---: | ---: | ---: |
| `ranker` | 100 | 0.2125 | 0.0180 | 0.0387 |
| `random_high_quality` | 100 | 0.1973 | 0.0709 | 0.0642 |
| `quality_only` | 100 | 0.3773 | 0.0659 | 0.1378 |
| `diverse_source_cluster` | 100 | 0.0360 | 0.0469 | 0.1031 |

This smoke signal is concerning: the ranker does not beat simple baselines in the non-ranker temporal/raw representations. Do not over-interpret a single smoke fold, but treat it as a real warning. The full Modal run is needed before making the scientific decision.

The first uncapped full attempt found `68,208` physical-source cached rows. For the first decision run, the config is capped at `20,000` rows to match the source-blocked eval scale and keep raw-shape validation within the Modal timeout. The first 20k run reached `14,000 / 20,000` raw-shape rows before the Modal local client disconnected, so the rerun keeps the 20k row cap but limits raw-shape extraction to `1,800` samples per clip.

Full 20k run completed on Modal:

```text
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_report_full.json
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_candidates_full.jsonl
```

Local pulled copy:

```text
data/marginal_coverage_full_2026_04_27/marginal_coverage_report_full.json
data/marginal_coverage_full_2026_04_27/marginal_coverage_candidates_full.jsonl
```

Full run shape:

| Field | Value |
| --- | ---: |
| rows | 20,000 |
| source groups | 5,437 |
| folds | 4 |
| fold candidate count range | 756-828 |
| fold target count range | 320-365 |

Mean relative coverage gain:

| Policy | K | window mean/std | temporal-order | raw-shape |
| --- | ---: | ---: | ---: | ---: |
| `ranker` | 100 | 0.0426 | 0.0114 | 0.0521 |
| `random_high_quality` | 100 | 0.0123 | 0.0034 | 0.0618 |
| `quality_only` | 100 | 0.0525 | 0.0059 | 0.0775 |
| `old_novelty_only` | 100 | 0.0766 | 0.0173 | 0.0572 |
| `diverse_source_cluster` | 100 | 0.0185 | 0.0023 | 0.0383 |
| `ranker` | 200 | 0.0789 | 0.0182 | 0.0768 |
| `random_high_quality` | 200 | 0.0224 | 0.0063 | 0.0955 |
| `quality_only` | 200 | 0.0700 | 0.0098 | 0.1035 |
| `old_novelty_only` | 200 | 0.0792 | 0.0199 | 0.0778 |

Decision:

```text
The current subcluster40 ranker is not scientifically validated by this test.
```

Why:

- it beats random high-quality on temporal-order coverage, but not raw-shape coverage
- it loses to `quality_only` on raw-shape coverage at K=100 and K=200
- it loses to `old_novelty_only` on temporal-order, raw-shape, and ranker-space coverage at K=100
- it does not pass the predeclared gate requiring wins in at least two non-ranker representations

The current ranker is therefore a coherent heuristic, not proven marginal-value selection. The next scientific move should be to compare against and possibly promote a simpler baseline, likely quality plus old novelty plus explicit source/parent diversity, before building a hidden-test execution wrapper around the current stack.
