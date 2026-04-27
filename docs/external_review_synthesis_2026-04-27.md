# External Review Synthesis

Date: 2026-04-27

Review artifacts:

```text
.omx/artifacts/gpt55-active-learning-review-20260427.md
.omx/artifacts/claude-active-learning-review-20260427.md
```

## Consensus

Both reviewers agree on the core split:

| Track | Verdict |
| --- | --- |
| Submission CSV | Ready enough if the deliverable is only a ranked CSV. |
| Hidden-test algorithm | Not ready. The current flow is a multi-step Modal recipe with hardcoded private volume/artifact assumptions. |
| Scientific validity | Directionally reasonable but not proven. The validation remains partly self-referential. |

Both reviewers explicitly say that "no trained model" is not by itself a fatal issue. A deterministic batch-ranking algorithm is acceptable for marginal data value ranking if it is reproducible and if batch-transductive ranking is allowed.

## Shared Critique

The current system should not be described as a trained ML model. It is a deterministic ranking pipeline using:

- handcrafted `window_mean_std_pool` representation
- old-support kNN novelty
- new-batch density and clustering
- quality gating
- pretrain-fitted tokenizer/grammar features
- stationary guard
- subcluster splitting
- tiered child-cluster reranking

That is coherent, but an evaluator cannot currently run it cleanly on a hidden held-out set without knowing our Modal volume layout and artifact history.

## Biggest Risks

1. Hidden-test execution is the immediate engineering gap.
2. Source-blocked validation is better than the old candidate eval, but it still defines positives using the ranker's own feature space.
3. The system's temporal representation is weak: mean/std pooling discards ordering.
4. Old-support novelty is measured against the cached support slice, not the full theoretical old corpus.
5. `subcluster40` has real dominant-parent concentration even if the visual audit did not show an obvious artifact mode.
6. If the evaluator expects per-sample prediction rather than batch ranking, this pipeline is structurally mismatched because it uses batch density, clustering, and reranking.

## Post-Review Check: Source-Blocked Base Rate

Claude flagged that source-blocked P@100 is hard to interpret without comparing against each fold's positive rate. That check was run after the reviews.

Result:

| Candidate | mean P@100 | mean fold positive rate baseline | mean P@100 minus baseline |
| --- | ---: | ---: | ---: |
| `stationary_guard` | 0.250 | about 0.320 | -0.070 |
| `subcluster40` | 0.325 | about 0.320 | +0.005 |

Fold-level `subcluster40` P@100 minus base positive rate:

| Fold | P@100 minus base |
| ---: | ---: |
| 0 | +0.0967 |
| 1 | +0.0250 |
| 2 | -0.1010 |
| 3 | -0.0005 |

Interpretation:

The source-blocked eval is still useful comparatively because `subcluster40` beats `stationary_guard`, but the absolute P@100 is not strong evidence of true retrieval skill above random/base positive rate. This weakens the earlier confidence in source-blocked validation as a readiness signal and strengthens the case that the next work should be packaging/reproducibility plus better validation, not more candidate tuning.

## Immediate Decision

If the deliverable is only a CSV:

```text
Submit the current subcluster40 CSV.
```

If Eddy/evaluator wants to run the algorithm on a hidden test set:

```text
Do not claim algorithm readiness yet.
Build the hidden-test execution wrapper and frozen artifact bundle first.
```

## One Thing To Build Next

Build a one-command hidden-test execution package, for example:

```bash
marginal-value hidden-rank \
  --new-manifest hidden_new_urls.txt \
  --pretrain-manifest pretrain_urls.txt \
  --artifact-bundle artifacts.tar.zst \
  --workdir run_hidden \
  --out submission.csv
```

It should:

- avoid hardcoded private Modal volume names where possible
- clearly separate frozen pretrain artifacts from hidden/test data
- transform and score hidden/new data without refitting on it
- run cache -> tokenizer transform -> grammar scoring -> ranking -> ID finalization
- emit a manifest of checksums and output paths
- fail loudly if required support artifacts are missing
- document that the algorithm is batch-transductive

## Do Not Do Next

- Do not train a neural model just to look more ML-like.
- Do not tune more reranker/config variants against the current new split.
- Do not claim source-blocked eval proves true marginal value.
- Do not hand an evaluator a private Modal recipe as if it were a reproducible algorithm.
