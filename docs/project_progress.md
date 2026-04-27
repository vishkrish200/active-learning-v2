# Project Progress Report

Last updated: 2026-04-27

This document records what has been built, tested, learned, and deliberately deferred in the active-learning IMU marginal data value project. It is meant to be the durable project memory: the methods we tried, why we tried them, what passed, what failed, and what the next engineering decisions should be.

## 1. Goal

The project is a ranking system for marginal data value in IMU clips. The target is not anomaly detection. The system should rank new worker clips by how much useful information they add relative to an existing IMU corpus.

Desired output:

```text
worker_id,rank,score,quality_score,reason_code
```

Core scoring idea:

```text
useful_value ~= quality * old_corpus_novelty * new_batch_support * diversity
```

The important distinction throughout the project has been:

| Concept | Intended meaning |
| --- | --- |
| Novelty | Far from old data |
| Usefulness | Novel and worth collecting |
| Artifact | Weird sensor behavior that should not rank high |
| Diversity | Avoid returning many near-duplicates from the same new workflow |

## 2. Current High-Level Status

The repo now contains a runnable Phase A plus a lightweight Phase B skeleton:

| Plan area | Current state |
| --- | --- |
| Data loading and split safety | Implemented with manifest-based pretrain/val/new split handling. |
| Preprocessing and quality | Implemented with robust normalization, derived channels, and artifact-oriented quality features. |
| Phase A representation | Implemented as `window_mean_std_pool`, a handcrafted mean/std clip representation. |
| Old-corpus support | Implemented with exact cosine kNN over embeddings. FAISS is not yet needed for this subset. |
| New-batch support | Implemented via batch density and simple cosine-threshold clusters. |
| Diversity reranking | Implemented with MMR, cluster bonus, cluster cap, and round-robin shadow variants. |
| Tokenizer | Implemented as numpy k-means VQ stand-in plus MotionBPE variable-duration primitive discovery. |
| Grammar model | Implemented as n-gram grammar features, not a transformer LM. |
| Pseudo-holdout evaluation | Implemented for leave-cluster and motion-phrase style validations. |
| Learned ranker | Implemented as a leakage-audited linear ranker, but not promoted because it did not beat the best hand-scored path. |
| SSL encoder | Implemented and trained on Modal H100, including anti-collapse VICReg variant. It passes geometry checks but does not improve ranking yet. |
| Real new ranking | A quality-gated grammar new-split run exists, but final submission should wait until diversity and validation choices are settled. |

Current best practical base remains:

```text
window_mean_std_pool + quality_gated_grammar + cluster-aware diversity
```

The encoder is currently experimental. It is geometrically alive, but not aligned with ranking value.

## 3. Repo Structure Added

Important source areas:

| Path | Purpose |
| --- | --- |
| `marginal_value/data/` | Loading, cached split manifests, new-split caching. |
| `marginal_value/preprocessing/` | IMU feature derivation and quality scoring. |
| `marginal_value/indexing/` | kNN and cluster support features. |
| `marginal_value/models/` | Tokenizer, grammar model, learned linear ranker, SSL encoder modules. |
| `marginal_value/tokenization/` | Modal tokenizer and grammar feature pipelines. |
| `marginal_value/ranking/` | Baseline ranking, grammar promotion, learned-ranker integration, corruption eval, submission writers. |
| `marginal_value/eval/` | Ablations, pseudo-holdouts, audits, encoder eval, shadow ranking, score calibration. |
| `marginal_value/training/` | Modal-only SSL encoder training and normalization. |
| `configs/` | Modal job configs and evaluation/ranking configs. |
| `tests/` | Unit tests for pipeline pieces, config validation, leakage checks, Modal entrypoints, ranking behavior. |
| `modal_*.py` | Thin Modal CLI entrypoints for remote-only jobs. |

The repo has intentionally stayed dependency-light locally:

```text
numpy
pandas
modal optional extra
```

PyTorch is imported inside Modal-only functions so local validation does not accidentally run training on the Mac.

## 4. Data Access and Split Discipline

The project uses Modal volumes:

| Volume | Role |
| --- | --- |
| `imu-novelty-subset-data` | Cached raw/features/manifests. |
| `activelearning-imu-rebuild-cache` | Artifacts, reports, rankings, checkpoints. |

Split manifests are used throughout:

| Split | Purpose |
| --- | --- |
| `pretrain` | Support/training corpus. |
| `val` | Held-out validation query split. |
| `new` | Held-out 2000-worker query split for real ranking. |

Leakage controls implemented:

- Training config explicitly uses `train_split: pretrain`.
- Encoder eval refuses checkpoints not trained with `train_split='pretrain'`.
- Encoder eval also checks `holdout_split='val'` on checkpoint configs.
- Feature scaler fitting is done only over selected pretrain rows.
- Learned ranker has a forbidden-feature list to prevent label, rank, split, reason-code, or holdout-membership leakage.
- Learned ranker can run fold-holdout scoring and exclude eval samples from train.
- New-split ranking config uses `support_split: pretrain` and `query_split: new`.

Key files:

- `marginal_value/data/split_manifest.py`
- `marginal_value/data/support_coverage_audit.py`
- `marginal_value/training/torch_train.py`
- `marginal_value/eval/modal_encoder_eval.py`
- `marginal_value/eval/learned_ranker_eval.py`

Support coverage audit:

```text
config: configs/support_coverage_audit_worker_coverage.json
modal entrypoint: modal_support_coverage_audit.py
full run: https://modal.com/apps/vishkrish200/main/ap-L3gBiAyZmB1iDGUrzHvr7W
local report: data/audits/support_coverage_worker_coverage/support_coverage_worker_coverage/support_coverage_audit_full.json
```

The full audit found:

| Split | Manifest URLs | Manifest workers | Source exists | Cached raw+features |
| --- | ---: | ---: | ---: | ---: |
| `new` | 2,000 | 2,000 | 2,000 | 2,000 |
| `pretrain` | 200,000 | 10,000 | 68,210 | 13,273 |
| `val` | 0 | 0 | 0 | 0 |

This matters: current old-corpus novelty is measured against the available cached support slice, not the full original plan target of about 1.15M old 3-minute windows. The processed cache is clean, though: 15,273 raw files, 15,273 feature files, and no raw/feature orphans.

## 5. Quality Gate

Implemented in:

- `marginal_value/preprocessing/quality.py`
- `marginal_value/preprocessing/features.py`

Quality features include:

- missing rate
- NaN/inf fraction
- flatline fraction
- saturation fraction
- spike rate
- high-frequency energy
- stationary fraction
- axis imbalance
- repeated timestamp fraction
- timestamp jitter fraction

Quality score is a conservative penalty model in `[0, 1]`. It is not a learned artifact classifier yet, but it acts as a multiplicative gate and as a ranking feature.

The quality layer is used in:

- baseline ranking
- grammar score gating
- corruption negative evaluation
- reason-code assignment
- audits of top-K quality

Important behavior:

```text
final_score = quality_score * ranker_score
```

This is a core defense against sensor artifacts being mistaken for valuable novelty.

## 6. Preprocessing and Handcrafted Representation

Implemented preprocessing:

- robust median/IQR normalization
- accel and gyro norms
- jerk and angular jerk norms
- moving-average gravity estimate
- linear acceleration norm
- stationarity flag

The strongest current Phase A representation is:

```text
window_mean_std_pool
```

Defined in `marginal_value/ranking/baseline_ranker.py`.

For each clip/window feature matrix, it concatenates:

```text
mean(window_features) || std(window_features)
```

This representation is simple, cheap, stable, and currently outperforms the trained SSL encoder for ranking.

## 7. Phase A Ranking Baseline

Implemented in:

- `marginal_value/ranking/baseline_ranker.py`
- `marginal_value/ranking/modal_baseline_rank.py`
- `marginal_value/submit/make_submission.py`

The baseline computes:

- old-corpus kNN novelty
- new-batch density
- quality score
- cluster annotations
- reason codes
- MMR or cluster-aware reranking

Default score family:

```text
ranker_score = 0.75 * minmax(old_knn_distance) + 0.25 * minmax(new_batch_density)
final_score = quality_score * ranker_score
```

Important reason codes:

- `LOW_QUALITY`
- `COHESIVE_NEW_WORKFLOW`
- `HIGH_NOVELTY_SINGLETON`
- `RARE_TEMPORAL_COMPOSITION`
- `REDUNDANT_KNOWN_WORKFLOW`
- `RARE_MOTION_PRIMITIVES`

The baseline is deliberately interpretable. That made it possible to see diversity collapse, grammar over-promotion risk, and encoder underperformance.

## 8. Tokenizer and Variable-Duration Motion Primitives

Implemented in:

- `marginal_value/models/tokenizer.py`
- `marginal_value/tokenization/patches.py`
- `marginal_value/tokenization/modal_tokenizer.py`
- `marginal_value/tokenization/transform_existing.py`

Two-layer tokenizer:

1. `PatchVQTokenizer`
   - numpy k-means stand-in for VQ
   - dependency-light, good enough for pipeline validation
   - replaceable later with a true GPU VQ module

2. `MotionBPE`
   - mines frequent adjacent token sequences
   - creates variable-duration primitives
   - enforces maximum primitive duration
   - scores candidates with count, length, token diversity, and boundary-context entropy

The theory tested here:

Fixed-time tokens can shred natural factory motion primitives. A reach, pause, gait cycle, inspection, or repetitive tool rhythm may not fit a single fixed duration. MotionBPE lets common local token sequences become reusable primitives while keeping duration explicit.

The tokenizer pipeline now has tests for:

- patch extraction
- VQ encode behavior
- BPE primitive selection
- duration bounds
- transform jobs
- Modal tokenizer entrypoint configuration

## 9. Grammar Modeling

Implemented in:

- `marginal_value/models/grammar_lm.py`
- `marginal_value/tokenization/modal_grammar.py`
- `marginal_value/eval/grammar_ablation_eval.py`
- `marginal_value/eval/motion_phrase_holdout_eval.py`

Current grammar model is n-gram based, not a transformer LM.

Features emitted include:

- `token_nll_mean`
- `token_nll_p90`
- `token_nll_p95`
- `transition_nll_mean`
- `transition_nll_p95`
- `rare_bigram_fraction`
- `rare_trigram_fraction`
- `rare_phrase_fraction`
- `longest_unseen_phrase_len`

Grammar was tested as:

- diagnostics-only
- direct promotion feature
- quality-gated score variant
- motion-phrase holdout signal
- leave-cluster ablation signal

The main useful variant became:

```text
quality_gated_grammar
```

Why it matters:

The grammar features are the closest implemented piece to the original differentiated system: "new workflow composed of familiar atoms." They capture temporal composition novelty rather than only distance in handcrafted feature space.

## 10. Grammar Score Promotion

Implemented in `apply_grammar_score_promotion`.

The score variant currently used for the strongest path is:

```text
quality_gated_grammar
```

It uses grammar surprisal only when:

- grammar features are present
- quality is above threshold
- new-batch support gate passes

It records:

- `phase_a_ranker_score`
- `phase_a_final_score`
- `grammar_score`
- `grammar_score_component`
- `grammar_promotion_delta`
- `grammar_promotion_applied`

This lets audits distinguish clips that were already high under Phase A from clips promoted because of rare temporal composition.

## 11. Corruption Negatives

Implemented in:

- `marginal_value/ranking/modal_baseline_rank.py`
- `marginal_value/eval/ablation_eval.py`
- `marginal_value/eval/metrics.py`

Synthetic corruption negatives are injected into ranking evaluation. Supported modes:

- flatline
- spike
- saturation
- jitter

Each corruption candidate receives:

- `label = 0`
- `is_corruption = true`
- `quality_score = 0.05`
- `corruption_mode`
- `source_sample_id`

Metrics now include:

- `corruption_rate`
- `corruption_rate@K`

Why this was added:

The project goal is marginal data value, not anomaly detection. Corruptions can be very novel in embedding space. The ranking system must prove it does not place obvious artifacts in the top K.

## 12. Learned Ranker

Implemented in:

- `marginal_value/models/learned_linear_ranker.py`
- `marginal_value/eval/learned_ranker_eval.py`
- `modal_learned_ranker.py`
- `marginal_value/ranking/modal_baseline_rank.py`

The implemented learned ranker is a leakage-audited linear/centroid model, not LightGBM LambdaRank.

Why linear first:

- dependency-light
- easy to serialize as JSON
- fast to evaluate on Modal
- useful as a leakage-audited production wiring test

Leakage prevention:

- forbidden columns exclude label, rank, reason code, heldout identifiers, split, score outputs, and duplicate/artifact flags
- coverage by label is tracked
- excluded features are reported

Observed result:

The learned ranker artifact was produced:

```text
/artifacts/ranking/window_mean_std_grammar_promoted/learned_ranker/learned_ranker_model_full.json
```

But eval showed the existing `final_score` beat `learned_linear`:

| Variant | nDCG@100 |
| --- | ---: |
| existing `final_score` | 0.5621 |
| learned linear | 0.5255 |

Decision:

Do not promote the learned linear ranker as production default. Keep the integration path, but use it only after it beats the current score on pseudo-holdout validation.

## 13. Diversity and Shadow Reranking

Implemented in:

- `marginal_value/ranking/baseline_ranker.py`
- `marginal_value/eval/shadow_ranking_eval.py`
- `marginal_value/eval/rerank_eval.py`
- `configs/shadow_quality_gated_grammar_diversity.json`

Diversity approaches implemented:

- MMR
- cluster bonus
- cluster cap
- cluster round-robin
- parent-cluster cap
- parent-prefix plus child-fill hybrid cap

The user explicitly called out diversity collapse risk. We added shadow evaluation variants such as:

- `cap1_top100`
- `cap2_top100`
- `cap2_top200`
- `cap3_top200`
- `cap4_top200`
- `cap5_top100`
- `round_robin`

Selection criteria in the current diversity config:

```text
candidate_top_k: 200
min_positive_fraction: 0.9
min_unique_clusters: 30
max_low_quality_count: 0
```

The current new-split quality-gated grammar ranking config uses:

```text
reranker_method: cluster_cap
cluster_cap_top_k: 200
cluster_max_per_cluster: 8
```

The follow-up parent-cap config is:

```text
configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_parentcap.json
ranking.cluster_cap_key: new_cluster_parent_id
```

The stricter parent-cap run showed an important failure mode: parent diversity improves early, but the cap exhausts quickly because most parent clusters are tiny. After fallback, the giant parent cluster returns. We then added a safer hybrid:

```text
configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_hybrid75.json
reranker_method: parent_prefix_cluster_cap
prefix_cluster_cap_top_k: 75
prefix_cluster_cap_key: new_cluster_parent_id
cluster_cap_key: new_cluster_id
cluster_cap_min_quality: 0.45
```

The hybrid gives parent diversity where it matters most early in the ranking, then fills using the better child-cluster cap order. It also prevents the diversity cap from promoting low-quality singleton corruptions.

## 14. Ranking Audits

Implemented in:

- `marginal_value/eval/ranking_audit.py`
- `marginal_value/eval/reason_threshold_grid.py`
- `marginal_value/eval/score_calibration_eval.py`

Audits track:

- top-K reason-code counts
- top-K quality
- low-quality counts
- cluster diversity
- grammar surprisal examples
- top movers after shadow reranking
- score calibration variants

This was important because many failures were not visible from nDCG alone.

## 15. Modal Logging and Execution Safety

Implemented across Modal entrypoints using:

- `marginal_value/logging_utils.py`

Modal jobs now log structured events and progress markers for:

- local dispatch start/done
- split audit
- manifest load
- quality progress
- tokenizer fit progress
- grammar feature progress
- training step progress
- checkpoint write and verification
- eval embedding progress
- artifact write completion

Modal training entrypoint:

- `modal_train.py`

The local entrypoint always runs:

1. remote split audit
2. remote smoke training
3. remote validation if requested
4. remote full training only if requested

This was added because we wanted remote jobs to fail early before spending full GPU time.

## 16. Encoder Work

The encoder work went through two stages.

### 16.1 Original Encoder

Original training path:

- masked reconstruction style MLP
- Modal H100 training
- checkpoint written and verified

Artifact:

```text
/artifacts/checkpoints/ssl_encoder_pretrain_only.pt
```

Observed full eval:

- effective rank about `1.02`
- mean pairwise cosine distance about `0.014`

Interpretation:

The representation was collapsed or nearly collapsed. It passed checkpoint mechanics but failed as a useful embedding space.

Decision:

Do not use original encoder embeddings for ranking.

### 16.2 Anti-Collapse Encoder

Implemented in:

- `marginal_value/models/ssl_encoder.py`
- `marginal_value/training/feature_scaler.py`
- `marginal_value/training/torch_train.py`
- `marginal_value/eval/modal_encoder_eval.py`

Key changes:

- normalized feature scaler fit only on pretrain rows
- explicit `normalized_vicreg_mlp` architecture
- embedding head separate from reconstruction head
- masked reconstruction retained
- VICReg invariance, variance, and covariance losses added
- augmentations added:
  - Gaussian noise
  - timestep dropout
  - channel dropout
- checkpoint now saves:
  - `encoder_config`
  - `feature_scaler`
  - `loss_components`

Config:

- `configs/modal_training_anticollapse.json`

Training settings:

```text
gpu: H100
batch_size: 32
max_steps: 2000
smoke_steps: 2
validation_steps: 150
embedding_dim: 128
d_model: 128
mask_probability: 0.2
losses:
  masked_reconstruction: 1.0
  vicreg_invariance: 1.0
  vicreg_variance: 1.0
  vicreg_covariance: 0.04
```

Validation checkpoint:

```text
/artifacts/checkpoints/validation_encoder_normalized_vicreg_pretrain_only.pt
```

Validation full eval:

| Metric | Value |
| --- | ---: |
| effective rank | 5.2979 |
| mean pairwise cosine distance | 0.9976 |
| encoder mean kNN d1 | 0.1401 |

Full checkpoint:

```text
/artifacts/checkpoints/ssl_encoder_normalized_vicreg_pretrain_only.pt
```

Full training final values:

| Metric | Value |
| --- | ---: |
| last loss | 1.9323 |
| last reconstruction loss | 0.1107 |
| last embedding std mean | 0.0888 |

Full encoder eval artifact:

```text
/artifacts/eval/encoder_normalized_vicreg_full/encoder_eval_report_full.json
```

Full encoder eval:

| Metric | Value |
| --- | ---: |
| acceptance passed | true |
| effective rank | 2.9747 |
| mean pairwise cosine distance | 1.0007 |
| encoder mean kNN d1 | 0.0479 |
| baseline mean kNN d1 | 0.6898 |

Interpretation:

The anti-collapse encoder fixed the geometry enough to pass acceptance. However, effective rank dropped from validation to full, and the ranking smoke showed it is not useful as a replacement representation.

### 16.3 Encoder Artifact Ranking Smoke

Implemented support for:

```text
ranking.representation = "encoder_artifact"
```

Files:

- `marginal_value/ranking/modal_baseline_rank.py`
- `marginal_value/ranking/config.py`
- `configs/baseline_ranking_encoder_anticollapse.json`

The ranking job can now load precomputed `.npy` encoder embeddings plus manifest CSVs and preserve requested row order.

Smoke result for `encoder_artifact`:

| Metric | Value |
| --- | ---: |
| nDCG@100 | 0.2100 |
| precision@100 | 0.12 |
| corruption_rate@100 | 0.48 |
| cluster_diversity@100 | 0.4167 |

Decision:

Stop this branch before full ranking eval. Do not promote encoder embeddings into final ranking yet.

Theory after this result:

The encoder is no longer collapsed, but its objective is not aligned with marginal-data-value ranking. Longer VICReg/reconstruction training is unlikely to solve this by itself. The next encoder step should be ranking-aware or pseudo-label-aware, not just more self-supervised training.

## 17. New-Split Quality-Gated Grammar Run

A real new-split ranking run exists locally in:

- `data/submissions/ranking_report_quality_gated_grammar_new.json`
- `data/submissions/submission_quality_gated_grammar_new.csv`
- `data/submissions/diagnostics_quality_gated_grammar_new.csv`

Config:

- `configs/baseline_ranking_new_quality_gated_grammar.json`

Important setup:

```text
support_split: pretrain
query_split: new
representation: window_mean_std_pool
grammar score_variant: quality_gated_grammar
reranker_method: cluster_cap
cluster_cap_top_k: 200
cluster_max_per_cluster: 4
corruption_eval enabled
```

Reported full metrics:

| Metric | Value |
| --- | ---: |
| n_query | 2000 |
| n_support | 5000 |
| n_negative | 500 |
| nDCG@10 | 0.7850 |
| nDCG@50 | 0.5763 |
| nDCG@100 | 0.7390 |
| nDCG@200 | 0.8431 |
| precision@10 | 0.70 |
| precision@50 | 0.52 |
| precision@100 | 0.76 |
| precision@200 | 0.88 |
| cluster_diversity@100 | 0.25 |
| cluster_diversity@200 | 0.1080 |
| mean quality | 0.9675 |
| low-quality fraction | 0.01 |
| new-batch clusters | 32 |
| max cluster size | 1500 |

Important caveat:

These metrics are from the pseudo-eval/candidate machinery around the new run, not an external challenge score. They suggest the ranking path is useful, but diversity still needs careful review before treating the output as final.

## 18. Tests Added

The local suite reached:

```text
156 passed
```

Test areas include:

- pipeline scoring and submission formatting
- quality and preprocessing behavior
- split manifests and new-split cache
- tokenizer and grammar pipelines
- Modal tokenizer and grammar entrypoints
- Modal training config validation
- encoder eval config and acceptance wiring
- feature scaler correctness
- baseline ranking and reason-code logic
- corruption negatives and corruption_rate@K metrics
- learned ranker feature leakage checks
- learned-ranker production wiring
- shadow ranking and diversity selection
- ranking audits
- score calibration
- motion-phrase holdout
- rerank eval

Representative test files:

- `tests/test_pipeline.py`
- `tests/test_baseline_ranking.py`
- `tests/test_feature_scaler.py`
- `tests/test_encoder_eval.py`
- `tests/test_modal_training.py`
- `tests/test_learned_ranker_eval.py`
- `tests/test_shadow_ranking_eval.py`
- `tests/test_score_calibration_eval.py`
- `tests/test_motion_phrase_holdout_eval.py`

## 19. Main Decisions and Why

### Decision: Use handcrafted pool as the current production representation

Reason:

It is stable, cheap, interpretable, and currently beats the trained encoder in ranking smoke tests.

### Decision: Add grammar before trusting encoder complexity

Reason:

The plan's differentiator is temporal composition novelty. The n-gram grammar path gave a way to test that idea without waiting for a transformer LM.

### Decision: Use `quality_gated_grammar` instead of raw grammar surprisal

Reason:

Raw grammar surprise can be caused by corruption, bad mounting, or timestamp issues. Quality and support gating make it more like useful novelty and less like anomaly detection.

### Decision: Add corruption negatives to ranking eval

Reason:

Without corruption negatives, the system could look good on held-out novelty while still ranking bad sensors highly.

### Decision: Keep learned ranker wiring but do not promote current learned model

Reason:

The implemented learned linear ranker did not beat the existing final score. Production should not switch just because a learned model exists.

### Decision: Stop encoder-artifact full ranking after bad smoke

Reason:

The smoke result had poor precision and high corruption rate. Running full eval after that would waste compute and risk confusing the project direction.

### Decision: Avoid final submission for now

Reason:

The new-split run exists, but we still want a better diversity validation and a clear final selected variant before treating it as the submission path.

## 20. Current Gaps Against Original Plan

| Original plan item | Current gap |
| --- | --- |
| PatchTST/TS2Vec-style self-supervised encoder | Current SSL encoder is an MLP over cached features, not a full PatchTST/TS2Vec architecture. |
| True VQ tokenizer | Current VQ is numpy k-means stand-in. |
| Transformer grammar LM | Current grammar model is n-gram. |
| LightGBM LambdaRank | Current learned ranker is linear/centroid. |
| FAISS | Exact cosine kNN is used because the subset size is manageable. |
| Mount-conditioned novelty | Not implemented. |
| Video distillation | Intentionally cut. |
| Full external submission | Deferred. |

## 21. What We Learned

1. The evaluation harness is the most valuable part of the system.

   Pseudo-holdouts, corruption negatives, diversity audits, and reason-code diagnostics caught issues that raw novelty would have hidden.

2. Geometry checks are necessary but not sufficient for encoders.

   The anti-collapse encoder passed effective-rank and pairwise-distance gates, but failed ranking smoke. Representation health does not equal marginal-data-value usefulness.

3. Quality gating is not optional.

   The encoder-artifact smoke had `corruption_rate@100 = 0.48`, which is exactly the failure mode the plan warned about.

4. Grammar features are useful, but only when calibrated.

   `quality_gated_grammar` is more defensible than raw surprisal because it requires quality and batch support.

5. Diversity must be treated as a first-class ranking objective.

   The new batch had a max cluster size of 1500, so naive sorting can collapse into one repeated workflow. Cluster caps and shadow reranking are necessary.

6. More GPU training is not automatically the next best step.

   The next encoder experiment should change the objective, not merely train the same objective longer.

## 22. Recommended Next Steps

### Step 1: Finish diversity selection for the current best path

Run or re-run the diversity shadow eval on `quality_gated_grammar` variants:

```text
configs/shadow_quality_gated_grammar_diversity.json
```

Target:

```text
precision@200 >= 0.9
unique_clusters@200 >= 30
low_quality_count@200 = 0
```

If no variant satisfies all constraints, choose the best tradeoff and document it.

### Step 2: Promote a single final ranking config

Once diversity is selected, create one final config that clearly names the selected path:

```text
window_mean_std_pool + quality_gated_grammar + selected cluster diversity
```

Do not include encoder embeddings unless they improve validation.

### Step 3: Do a ranking-aware encoder experiment

Instead of more VICReg-only training, try one cheap Modal experiment:

- use current anti-collapse encoder as initialization or frozen feature producer
- train a projection/head with leave-cluster and motion-phrase pseudo-labels
- optimize pairwise separation or supervised contrastive loss
- evaluate with the same ranking smoke before any full run

Acceptance should include:

```text
ranking smoke beats window_mean_std_pool
corruption_rate@100 remains low
precision@100 improves
diversity does not collapse
```

### Step 4: Upgrade learned ranker only after feature value is proven

Once the feature table is stable, replace the linear ranker with LightGBM LambdaRank or `rank_xendcg` if the dependency is acceptable in Modal.

Do not use forbidden columns or pseudo-label identifiers as features.

### Step 5: Final new-split ranking and diagnostics

Generate:

- final submission CSV
- diagnostics CSV
- ranking audit report
- ablation summary
- failure-mode notes

Only then treat the output as final.

## 23. Useful Commands

Local tests:

```bash
python3 -m unittest discover -s tests
```

Modal anti-collapse training:

```bash
.venv/bin/modal run modal_train.py --config-path configs/modal_training_anticollapse.json --run-full
```

Encoder eval:

```bash
.venv/bin/modal run modal_eval.py --config-path configs/eval_encoder_anticollapse_full.json
```

Encoder artifact ranking smoke:

```bash
.venv/bin/modal run modal_rank.py --config-path configs/baseline_ranking_encoder_anticollapse.json
```

Quality-gated grammar new ranking:

```bash
.venv/bin/modal run modal_rank.py --config-path configs/baseline_ranking_new_quality_gated_grammar.json
```

Shadow diversity eval:

```bash
.venv/bin/modal run modal_shadow_ranking.py --config-path configs/shadow_quality_gated_grammar_diversity.json
```

## 24. Current Project State in One Sentence

We have a working, audited marginal-data-value ranking pipeline whose best current signal is handcrafted IMU support plus quality-gated motion grammar and cluster-aware diversity; the learned encoder infrastructure works mechanically and no longer collapses geometrically, but it needs a ranking-aware objective before it should replace the handcrafted representation.

## 25. External Skeptical Review and Conservative Score Blend

We asked Claude Code on Opus for a skeptical review of the current system and gave it the full project context, including the original marginal-data-value plan, the implemented pipeline, the encoder failure, the full new-split run, and the finalized submission mapping.

Claude's main criticism was that the perfect pseudo-eval metrics should not be trusted as external performance evidence. In the new-split candidate eval, `new` rows are treated as positives while sampled `pretrain` rows are treated as negatives, and `pretrain` is also the old support corpus. That makes `precision@K = 1.0` and `nDCG@K = 1.0` more of a support-membership sanity check than a hard proof of challenge readiness.

The more important criticism was that `quality_gated_grammar` with `score_weight = 1.0` fully replaced the phase-A marginal-value score with grammar score. That made the ranking defensible as a rare-motion-grammar picker, but less defensible as the original objective:

```text
quality x old-corpus novelty x new-batch support x diverse coverage
```

Claude recommended a conservative ablation with lower grammar influence, starting at `score_weight = 0.3`, and comparing top-K novelty, support, quality, diversity, and overlap against the previous `score_weight = 1.0` run.

### Runs

Previous candidate:

```text
config: configs/baseline_ranking_new_quality_gated_grammar_worker_coverage.json
submission artifacts: data/submissions/worker_coverage_final/
grammar score_weight: 1.0
```

Conservative blend:

```text
config: configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03.json
audit config: configs/ranking_audit_quality_gated_grammar_new_worker_coverage_w03.json
submission artifacts: data/submissions/worker_coverage_w03/
grammar score_weight: 0.3
```

The `w0.3` run completed on Modal:

- smoke rank run completed
- full rank run completed
- audit run completed
- all recent Modal apps stopped with zero tasks
- local unit tests passed: `178 tests OK`

### Top-K Comparison

The perfect pseudo-eval metrics were unchanged and are not decision-grade. The useful comparison is the diagnostic top-K behavior.

Top 100:

| Metric | `w1.0` | `w0.3` |
|---|---:|---:|
| mean quality | 0.997699 | 0.988345 |
| min quality | 0.971291 | 0.878218 |
| low-quality fraction | 0.000000 | 0.000000 |
| corruption fraction | 0.000000 | 0.000000 |
| unique clusters | 29 | 42 |
| largest cluster fraction | 0.08 | 0.08 |
| mean old support novelty | 0.033047 | 0.172365 |
| mean new-batch support | 0.952562 | 0.820842 |
| mean grammar score | 0.805055 | 0.777509 |

Top 200:

| Metric | `w1.0` | `w0.3` |
|---|---:|---:|
| mean quality | 0.997804 | 0.993436 |
| min quality | 0.971291 | 0.878218 |
| low-quality fraction | 0.000000 | 0.000000 |
| corruption fraction | 0.000000 | 0.000000 |
| unique clusters | 32 | 54 |
| largest cluster fraction | 0.04 | 0.04 |
| mean old support novelty | 0.028274 | 0.105597 |
| mean new-batch support | 0.962433 | 0.883981 |
| mean grammar score | 0.778278 | 0.765013 |

Top-K overlap between the two runs:

| K | overlap |
|---:|---:|
| 10 | 2 / 10 |
| 25 | 7 / 25 |
| 50 | 11 / 50 |
| 100 | 29 / 100 |
| 200 | 81 / 200 |
| 500 | 200 / 500 |

### Interpretation

The `w1.0` run is very clean, but too grammar-dominated. It selects extremely high-quality, well-supported clips, but the top-ranked set has much lower old-corpus novelty.

The `w0.3` run better matches the original marginal-data-value objective. It keeps quality and corruption safety acceptable while selecting clips with substantially higher old-support novelty and broader cluster coverage:

- top-100 old-support novelty is about `5.2x` higher than `w1.0`
- top-200 old-support novelty is about `3.7x` higher than `w1.0`
- unique clusters@100 improves from `29` to `42`
- unique clusters@200 improves from `32` to `54`

The cost is that `w0.3` gives up some new-batch support and has a lower minimum top-K quality, but it does not cross the configured low-quality threshold and corruption count remains zero.

### Current Recommendation

Promote `w0.3` over `w1.0` as the current best candidate if submitting from this project state.

The reason is not that its pseudo-eval score is better. The reason is that its diagnostic behavior is more aligned with the challenge's actual stated goal: ranking clips by marginal value against the old corpus, not merely selecting high-quality rare grammar under the current tokenizer.

Use one of these finalized CSVs depending on the expected ID column:

```text
data/submissions/worker_coverage_w03/submission_worker_coverage_w03_worker_id.csv
data/submissions/worker_coverage_w03/submission_worker_coverage_w03_new_worker_id.csv
```

Both finalized files have:

- 2,000 rows
- 2,000 unique manifest-resolved IDs
- ranks exactly `1..2000`
- non-increasing scores
- no internal 64-character hash IDs

## 26. Skeptical System Audit

After seeing how much Claude's `score_weight = 0.3` suggestion changed the ranking, we ran a skeptical audit of the whole current system and documented the findings here:

```text
docs/skeptical_system_audit.md
```

The most important correction from that audit is that diversity must be judged using both post-split cluster IDs and original parent-cluster IDs. The `w0.3` run remains better than `w1.0`, but its top-200 has:

```text
post-split clusters: 54
parent clusters: 30
largest parent-cluster fraction: 0.54
```

So the candidate is improved, but still concentrated. The next step should not be bigger modeling. It should be:

1. treat `hybrid75` as the current best reranker candidate
2. add raw-signal corruption negatives
3. implement real feature-space subclustering for the largest parent cluster if we need another diversity gain

## 27. Physical-Source Full-Support Expansion

Claude's review also exposed a more basic coverage problem: the old-support cache used by ranking contained only about `13,273` cached pretrain clips, not the full accessible old corpus. We therefore paused model work and expanded the old-support cache against the actual physical source mirror on Modal.

### Source Inventory

We added source inventory tooling:

```text
marginal_value/data/source_inventory.py
modal_source_inventory.py
configs/source_inventory_observe_full.json
tests/test_source_inventory.py
```

The inventory found:

| Item | Count |
|---|---:|
| original pretrain manifest URLs | 200,000 |
| physical/extracted pretrain URLs available on Modal | 68,210 |
| physical pretrain workers | 8,483 |
| new split URLs available | 2,000 |

Important nuance: the original plan's `1.15M` old windows are still theoretical sliding windows over `10,000 x 1h`. The mounted Modal source currently exposes `68,210` physical pretrain clips, not all `1.15M` derived 180s windows.

### Cache Expansion

We added support-cache sharding and malformed-source tolerance:

```text
marginal_value/data/cache_support_split.py
modal_cache_support_split.py
configs/cache_pretrain_physical_source_all.json
configs/cache_pretrain_source_existing_all.json
```

The first full sharded run exposed two real malformed source cases. The cache builder now logs and skips malformed clips instead of killing the whole run.

Final Modal run:

```text
https://modal.com/apps/vishkrish200/main/ap-7LllAkmPfr6aHsoZ3SQwtb
```

Result:

| Metric | Value |
|---|---:|
| selected physical pretrain URLs | 68,210 |
| newly written this run | 14,474 |
| reused/skipped from existing cache | 53,734 |
| missing source | 0 |
| malformed source | 2 |
| feature files written this run | 14,474 |

Malformed examples:

```text
worker04447/clip013.txt: No IMU samples found
worker03833/clip005.txt: missing acc/gyro or six numeric channels
```

### Coverage Audit

We added a physical-source audit config:

```text
configs/support_coverage_audit_physical_source.json
```

The audit initially hit a `900s` Modal timeout while checking NPZ feature files, so `modal_support_coverage_audit.py` now uses a `3600s` timeout.

Final Modal audit:

```text
https://modal.com/apps/vishkrish200/main/ap-yR84Edp9LUnLkPW3jK7IhS
```

Pulled local report:

```text
data/audits/support_coverage_physical_source/support_coverage_audit_full.json
```

Coverage:

| Split | Manifest URLs | Source Exists | Raw Present | Feature Present | Cached Both |
|---|---:|---:|---:|---:|---:|
| new | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 |
| pretrain physical source | 68,210 | 68,210 | 68,210 | 68,208 | 68,208 |

Feature sanity sample:

| Metric | Value |
|---|---:|
| inspected feature files | 5,000 |
| failed feature files | 0 |
| feature dimension | 75 |
| windows per clip | 35 |
| estimated total 10s feature windows | 2,387,280 |

The volume also contains `3,305` orphan cached feature/raw files from older experiments. These are harmless if downstream jobs use the physical-source manifest, because `build_split_manifest` only selects rows present in the configured manifest and with both raw+feature files.

### Physical-Support Ranking Run

We added a full-support ranking config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75.json
```

This keeps the current best candidate family unchanged:

```text
window_mean_std_pool
+ quality_gated_grammar
+ parent_prefix_cluster_cap / hybrid75
```

The only major change is old support:

```text
pretrain_manifest: cache/manifests/pretrain_physical_source_urls.txt
```

Modal smoke ranking:

```text
https://modal.com/apps/vishkrish200/main/ap-42eDee75hqyfwJHAEUSpOe
```

Modal full ranking:

```text
https://modal.com/apps/vishkrish200/main/ap-hIhVOqNjDyetTtWPkT6Uzu
```

Pulled local artifacts:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75/baseline_ranking_report_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75/baseline_submission_val_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75/baseline_diagnostics_val_full.csv
```

Ranking report:

| Metric | Value |
|---|---:|
| support clips used | 68,208 |
| query clips used | 2,000 |
| pseudo-negative clips | 500 |
| query grammar matched | 2,000 / 2,000 |
| query mean quality | 0.9675 |
| query low-quality fraction | 0.0100 |
| raw new clusters before split | 32 |
| largest raw cluster before split | 1,500 |
| clusters after large-cluster split | 56 |
| max split cluster size | 78 |

The internal candidate pseudo-eval reported:

```text
nDCG@100 = 0.9546
precision@100 = 0.95
precision@200 = 0.87
corruption_rate@100 = 0.0
```

But this metric remains structurally too easy because positives are `new` rows and negatives are sampled `pretrain` rows while novelty is computed against pretrain.

### Physical-Support Ranking Audit

We added:

```text
configs/ranking_audit_quality_gated_grammar_new_physical_source_hybrid75.json
```

Modal audit:

```text
https://modal.com/apps/vishkrish200/main/ap-wDt3b96sS40eBFO9V7lO7e
```

Pulled local audit:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75/baseline_audit_full.json
```

Top-K diagnostics:

| K | mean quality | low quality | split clusters | largest split cluster frac | parent clusters | largest parent frac |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9857 | 0 | 10 | 0.10 | 9 | 0.20 |
| 25 | 0.9884 | 0 | 21 | 0.12 | 18 | 0.20 |
| 50 | 0.9876 | 0 | 37 | 0.16 | 27 | 0.16 |
| 100 | 0.9875 | 0 | 43 | 0.08 | 32 | 0.46 |
| 200 | 0.9928 | 0 | 56 | 0.04 | 32 | 0.53 |

Interpretation:

- The full-support run is now genuinely ranking against the accessible old corpus, not the small worker-coverage support subset.
- Top-200 split-cluster diversity is strong: `56` clusters.
- Original parent-cluster concentration remains the main caveat: top-200 has `106/200` clips from the largest parent cluster.
- Top-K quality is excellent and zero clips under the configured low-quality threshold appear in top-200.
- Grammar is matched for all new query rows, but pretrain grammar features still cover only the older worker-coverage support subset. Candidate pseudo-eval grammar promotion therefore skips negatives and should not be treated as proof.

### Current Status After Full-Support Expansion

The current best candidate is:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75.json
```

This is a material improvement over the previous candidate because old-support novelty is now computed against `68,208` physical pretrain clips instead of about `13k`.

It is still not perfect:

1. parent-cluster concentration remains high at top-200
2. candidate pseudo-eval is still structurally too easy
3. corruption eval still uses embedding-space synthetic corruptions with hard-coded low quality
4. grammar/tokenizer artifacts have not yet been refit on the full `68,208` physical support set

Local tests after these changes:

```text
195 tests OK
```

## 28. Hardening Pass: Feature-Space Diversity, Raw Corruptions, Physical Leave-Cluster Eval

This pass was triggered by a skeptical review of the previous "best" run. The review found three real issues:

1. the `new` versus `pretrain` candidate eval was structurally too easy
2. the mega-cluster split inflated diversity by assigning subclusters with score round-robin order
3. the corruption eval used embedding perturbations with hard-coded low quality, so it mostly proved that quality was used in the score

The goal of this pass was not to add model capacity. It was to make the current candidate harder to fool.

### Code Changes

Large-cluster splitting now uses feature-space subclustering instead of score round-robin:

```text
marginal_value/ranking/baseline_ranker.py
```

The new default split method is `feature_kmeans`. It can use the actual query embeddings, keeps score out of the split by default, reports whether embedding features were used, and reports fallback splits separately.

Raw-signal corruption validation was added:

```text
marginal_value/ranking/modal_baseline_rank.py
```

When `corruption_eval.raw_signal = true`, the Modal job reads real raw IMU JSONL rows, applies corruptions to the raw signal, recomputes window features and embeddings, and then recomputes quality from the corrupted raw signal. Corruption quality is no longer hard-coded.

The quality gate was strengthened:

```text
marginal_value/preprocessing/quality.py
```

The first raw-corruption full run exposed sparse impossible spikes that still had quality near `0.95`. We added `max_abs_value` and `extreme_value_fraction` features plus penalties for sparse impossible spikes and extreme absolute values.

A harder old-corpus validation was added:

```text
marginal_value/eval/physical_leave_cluster_eval.py
modal_physical_leave_cluster_eval.py
configs/physical_leave_cluster_eval.json
```

This samples old physical-source clips, clusters them, removes held-out clusters from support, and asks whether the ranking recovers those held-out old motion regions. It is much closer to the planned leave-cluster-out evaluation than the easy `new` versus `pretrain` pseudo-eval.

### Hardened Configs

The new hardened candidate config is:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr.json
```

The matching audit config is:

```text
configs/ranking_audit_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr.json
```

This candidate keeps the production family:

```text
window_mean_std_pool
+ quality_gated_grammar
+ physical-source support
+ hybrid75 reranking
+ feature-space mega-cluster split
+ raw-signal corruption eval
```

The trained SSL encoder remains disabled for the production candidate.

### Modal Runs

All expensive ranking and validation work ran on Modal, not locally.

Smoke ranking before the quality fix:

```text
https://modal.com/apps/vishkrish200/main/ap-lwZRWTVnW8coKlFuiyEwhf
```

Physical leave-cluster smoke:

```text
https://modal.com/apps/vishkrish200/main/ap-r2MpamYxOQV8uzn2nt5pdM
```

First full hardened ranking before the quality fix:

```text
https://modal.com/apps/vishkrish200/main/ap-ERIZ3HOApLUvwtK5SS8Zo5
```

That run used full physical support but exposed raw-corruption leakage:

```text
corruption_rate@100 = 0.12
top candidate-eval corruptions were sparse spike / saturation rows
```

Smoke ranking after the quality fix:

```text
https://modal.com/apps/vishkrish200/main/ap-TUQ7jDsSbe6bJWEELhfGEy
```

This confirmed that raw corruptions were now caught:

```text
raw corruption max quality = 0.275
corruption_rate@10 = 0.0
corruption_rate@50 = 0.0
corruption_rate@100 = 0.0
```

Final full hardened ranking after the quality fix:

```text
https://modal.com/apps/vishkrish200/main/ap-RxwRg5l0qnnQ5s909bSSTH
```

Final audit after the quality fix:

```text
https://modal.com/apps/vishkrish200/main/ap-cziX14JosBL05oMDjz7LHo
```

Physical leave-cluster full eval:

```text
https://modal.com/apps/vishkrish200/main/ap-GqhtpdHBkGvOIpI8MtfwaE
```

### Final Hardened Ranking Artifacts

Pulled local artifacts:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_ranking_report_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_audit_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_submission_val_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_diagnostics_val_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_ranking_val_candidates_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/baseline_quality_metadata_full.csv
```

Full ranking metrics:

| Metric | Value |
|---|---:|
| support clips used | 68,208 |
| query clips used | 2,000 |
| pseudo-negative clips | 500 |
| nDCG@10 | 0.9364 |
| precision@10 | 0.9000 |
| nDCG@50 | 0.9639 |
| precision@50 | 0.9600 |
| nDCG@100 | 0.9463 |
| precision@100 | 0.9400 |
| nDCG@200 | 0.8697 |
| precision@200 | 0.8450 |
| corruption_rate@10 | 0.0000 |
| corruption_rate@50 | 0.0000 |
| corruption_rate@100 | 0.0000 |
| corruption_rate@200 | 0.0000 |
| raw-corruption quality mean | 0.1081 |
| raw-corruption quality max | 0.2809 |

Query cluster handling:

| Metric | Value |
|---|---:|
| raw new clusters before split | 32 |
| largest raw cluster before split | 1,500 |
| parent clusters split | 2 |
| clusters after feature split | 56 |
| max split cluster size | 78 |
| fallback split parent clusters | 0 |
| embedding features used | true |

Final audit summary:

| Check | Value |
|---|---:|
| submission rows | 2,000 |
| ranks contiguous | true |
| scores nonincreasing | true |
| duplicate worker count | 0 |
| top-10 mean quality | 0.9857 |
| top-100 low-quality count | 0 |

Reason-code counts over the full 2,000-row ranking:

| Reason code | Count |
|---|---:|
| HIGH_NOVELTY_SINGLETON | 2 |
| RARE_TEMPORAL_COMPOSITION | 211 |
| RARE_MOTION_PRIMITIVES | 6 |
| REDUNDANT_KNOWN_WORKFLOW | 1,761 |
| LOW_QUALITY | 20 |

Top-K audit after the hardening fix:

| K | mean quality | low quality | split clusters | largest split frac | parent clusters | largest parent frac | corruption negatives |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9857 | 0 | 10 | 0.10 | 9 | 0.20 | 0 |
| 25 | 0.9884 | 0 | 20 | 0.12 | 18 | 0.20 | 0 |
| 50 | 0.9876 | 0 | 37 | 0.16 | 27 | 0.16 | 0 |
| 100 | 0.9876 | 0 | 46 | 0.08 | 32 | 0.44 | 0 |
| 200 | 0.9930 | 0 | 56 | 0.04 | 32 | 0.53 | 0 |

Candidate-eval top-K after the quality fix:

| K | positive fraction | corruption negatives | low quality | unique clusters | largest cluster frac |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.940 | 0 | 0 | 41 | 0.08 |
| 200 | 0.845 | 0 | 0 | 59 | 0.04 |

### Physical Leave-Cluster Evaluation

Pulled local artifact:

```text
data/eval/physical_leave_cluster/window_mean_std_physical_source/physical_leave_cluster_eval_full.json
```

Setup:

| Metric | Value |
|---|---:|
| sampled old physical-source clips | 20,000 |
| source clusters | 64 |
| folds | 4 |

Mean metrics:

| Metric | Value |
|---|---:|
| mean nDCG@100 | 0.5935 |
| mean precision@100 | 0.6950 |
| mean nDCG@200 | 0.7348 |
| mean precision@200 | 0.8213 |
| mean quality | 0.9741 |
| low-quality fraction | 0.0068 |

Fold metrics:

| Fold | Held-out clusters | nDCG@100 | precision@100 |
|---:|---|---:|---:|
| 0 | `[0]` | 0.5849 | 0.7100 |
| 1 | `[40]` | 0.5493 | 0.6600 |
| 2 | `[11]` | 0.6534 | 0.7100 |
| 3 | `[19]` | 0.5863 | 0.7000 |

Interpretation:

- This is a more meaningful validation than `new` versus `pretrain` candidate eval.
- The ranking is not perfect, but it is recovering held-out physical old-corpus motion clusters at useful rates.
- The physical leave-cluster result is the better evidence that the system is learning old-corpus support structure rather than just split membership.

### What Was Addressed

The earlier caveats changed as follows:

| Previous caveat | Status |
|---|---|
| support was only about 13k cached clips | addressed: physical-source support uses 68,208 clips |
| mega-cluster diversity was inflated by score round-robin | addressed: feature-space k-means split with fallback reporting |
| corruption eval used embedding perturbations and hard-coded low quality | addressed: raw-signal corruptions recompute features and quality |
| sparse raw corruptions slipped through quality | addressed: quality gate now penalizes extreme values and sparse impossible spikes |
| candidate eval was structurally too easy | partially addressed: still reported, but no longer treated as the main proof; physical leave-cluster eval added |

### Remaining Caveats

The current candidate is much more defensible than the previous full-support run, but it is not magically guaranteed to score well on the hidden challenge metric.

Remaining risks:

1. Parent-cluster concentration still exists. Top-200 has `0.53` of clips from the largest original parent cluster, even though split-cluster diversity is strong.
2. The grammar/tokenizer artifacts are still from the earlier lightweight pipeline, not a full refit over all `68,208` physical-source support clips.
3. The production representation is still `window_mean_std_pool`, not the planned trained SSL encoder.
4. The final scoring function is still a hand-engineered quality-gated grammar score, not a learned LambdaRank model.
5. Manual trace plots for the top 25 clips are still recommended before final submission.

### Current Recommendation

The current best candidate is now:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr.json
```

This should replace the previous `physical_source_hybrid75` candidate as the main working candidate.

It is not the full original research plan. It is the best pragmatic submission path so far:

```text
quality-gated grammar
+ handcrafted pooled representation
+ full physical-source old support
+ raw-corruption validation
+ feature-space diversity control
+ physical leave-cluster sanity check
```

Local tests after this hardening pass:

```text
201 tests OK
```

## 29. Top-Clip Visual Audit

After hardening the ranking metrics, we added a human-audit artifact so the top-ranked clips can be inspected directly instead of trusting aggregate scores.

### Why

The remaining question was qualitative:

```text
Do the top-ranked clips actually look like useful workflow novelty,
or are they mostly one-off bumps, stationary segments, or scoring quirks?
```

This is especially important because top-200 still has meaningful parent-cluster concentration.

### Implementation

Added:

```text
marginal_value/eval/top_clip_visual_audit.py
modal_top_clip_visual_audit.py
configs/top_clip_visual_audit_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr.json
tests/test_top_clip_visual_audit.py
```

The audit reads the hardened ranking outputs:

```text
baseline_diagnostics_val_{mode}.csv
baseline_quality_metadata_{mode}.csv
```

It joins ranked rows back to raw IMU via `sample_id`, loads the raw JSONL from `/data/cache/raw`, and writes:

```text
top_clip_visual_audit_report_{mode}.json
top_clip_visual_audit_index_{mode}.html
top_clip_visual_audit_plot_index_{mode}.csv
top_clip_visual_audit_top_rows_{mode}.csv
top_clip_visual_audit_dominant_parent_rows_{mode}.csv
top_clip_trace_plots_{mode}/*.png
```

Each PNG includes:

- raw accel axes
- raw gyro axes
- accel/gyro norm
- accel/gyro jerk
- rank, score, quality, reason code, parent cluster, split cluster
- old novelty, grammar score, new-batch support, raw recomputed quality diagnostics

The audit selects:

```text
top 50 ranked clips
+ diverse examples from the dominant parent cluster in top-200
```

Duplicates are removed before plotting.

### Modal Runs

Smoke visual audit:

```text
https://modal.com/apps/vishkrish200/main/ap-UluvwdbHDpM3yRAKOen54R
```

Full visual audit:

```text
https://modal.com/apps/vishkrish200/main/ap-10k2MxdxF4r25lPX6kOw3g
```

This was CPU-bound plotting, not model training. It ran on Modal with mounted data and artifact volumes.

### Local Artifacts

Pulled local folder:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/visual_audit/
```

Important files:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/visual_audit/top_clip_visual_audit_report_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/visual_audit/top_clip_visual_audit_index_full.html
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/visual_audit/top_clip_visual_audit_plot_index_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr/visual_audit/top_clip_trace_plots_full/
```

### Results

Full audit:

| Metric | Value |
|---|---:|
| selected clips | 67 |
| plots written | 67 |
| plot errors | 0 |
| dominant parent cluster | `0` |
| top-10 mean quality | 0.9857 |
| top-100 largest parent fraction | 0.44 |

Top-K summary from the visual audit:

| K | mean quality | parent clusters | largest parent frac | split clusters | largest split frac |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.9857 | 9 | 0.20 | 10 | 0.10 |
| 25 | 0.9884 | 18 | 0.20 | 20 | 0.12 |
| 50 | 0.9876 | 27 | 0.16 | 37 | 0.16 |
| 100 | 0.9876 | 32 | 0.44 | 46 | 0.08 |
| 200 | 0.9930 | 32 | 0.53 | 56 | 0.04 |

Dominant parent cluster in top-200:

| Metric | Value |
|---|---:|
| parent id | `0` |
| count in top-200 | 106 |
| fraction in top-200 | 0.53 |
| split clusters represented | 20 |
| mean quality | 0.9984 |
| mean score | 0.4050 |

Dominant-parent reason codes:

| Reason code | Count |
|---|---:|
| RARE_TEMPORAL_COMPOSITION | 88 |
| REDUNDANT_KNOWN_WORKFLOW | 18 |

### Early Visual Read

The audit surfaced one thing worth treating seriously:

- Rank 1 is a high-quality `HIGH_NOVELTY_SINGLETON`, but visually it is mostly stationary with a couple of sharp events.
- Across the top 50, high-stationary clips are not dominant:
  - `stationary_fraction > 0.5`: 5 / 50
  - `stationary_fraction > 0.8`: 3 / 50
  - `stationary_fraction > 0.9`: 2 / 50
- Across the top 50:
  - `spike_rate > 0.005`: 10 / 50
  - `max_abs_value > 20`: 6 / 50

Interpretation:

The top-50 is not broadly collapsing to stationary artifacts, but the very top singleton behavior still deserves a scoring/reranking guard. A reasonable next adjustment is to downweight `HIGH_NOVELTY_SINGLETON` clips with very high stationarity unless they also have strong new-batch support or grammar evidence.

Local tests after adding the visual audit:

```text
203 tests OK
```

## 30. Stationary Singleton Guard

The first visual audit surfaced one specific failure mode: the old rank-1 clip was a `HIGH_NOVELTY_SINGLETON` with very high stationarity and only a couple of sharp events. It was high quality, but operationally it looked less convincing than a supported or grammar-backed workflow pattern.

### Change

Added a narrow score guard:

```text
marginal_value/ranking/baseline_ranker.py
marginal_value/ranking/modal_baseline_rank.py
marginal_value/ranking/config.py
```

New config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json
```

Matching audit configs:

```text
configs/ranking_audit_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json
configs/top_clip_visual_audit_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json
```

The guard uses raw quality metadata carried into the ranking diagnostics:

```text
stationary_fraction
spike_rate
max_abs_value
extreme_value_fraction
flatline_fraction
saturation_fraction
```

Guard rule:

```text
if HIGH_NOVELTY_SINGLETON-like
and stationary_fraction >= 0.90
and new_density_score < 0.35
and grammar_score_component < 0.85:
    ranker_score *= 0.35
```

This intentionally does not punish:

- supported new-batch patterns
- strong grammar-surprisal patterns
- moving singleton novelty

### Modal Runs

Smoke ranking:

```text
https://modal.com/apps/vishkrish200/main/ap-L0bx4mxFDgae1xXyJix2pj
```

Full guarded ranking:

```text
https://modal.com/apps/vishkrish200/main/ap-mZAYEUYM2oWwsdaRtwxnP4
```

Guarded ranking audit:

```text
https://modal.com/apps/vishkrish200/main/ap-7tY3bHQwVCpsdRVaTQ8fw9
```

Guarded visual audit:

```text
https://modal.com/apps/vishkrish200/main/ap-38VtL7h3WjEpfDOzs61KeN
```

### Local Artifacts

Pulled local folder:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/
```

Key files:

```text
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/baseline_ranking_report_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/baseline_audit_full.json
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/baseline_submission_val_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/baseline_diagnostics_val_full.csv
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/visual_audit/top_clip_visual_audit_index_full.html
data/ranking/window_mean_std_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard/visual_audit/top_clip_visual_audit_report_full.json
```

### Results

The guard fired exactly once in the full query set:

| Field | Value |
|---|---:|
| guarded query rows | 1 / 2,000 |
| guarded candidate-eval rows | 0 / 2,628 |
| old guarded clip rank | 1 |
| new guarded clip rank | 62 |
| old guarded final score | 0.7254 |
| new guarded final score | 0.2539 |
| guarded stationary fraction | 0.9291 |
| guarded old novelty score | 1.0000 |
| guarded grammar score | 0.7427 |
| guarded new density score | 0.0000 |

Main candidate metrics were unchanged:

| Metric | Value |
|---|---:|
| nDCG@10 | 0.9364 |
| precision@10 | 0.9000 |
| nDCG@50 | 0.9639 |
| precision@50 | 0.9600 |
| nDCG@100 | 0.9463 |
| precision@100 | 0.9400 |
| nDCG@200 | 0.8697 |
| precision@200 | 0.8450 |
| corruption_rate@100 | 0.0000 |
| corruption_rate@200 | 0.0000 |

Audit summary:

| Metric | Value |
|---|---:|
| submission rows | 2,000 |
| score nonincreasing | true |
| top-10 mean quality | 0.9880 |
| top-100 low-quality count | 0 |
| visual plots written | 67 |
| visual plot errors | 0 |

Top-50 visual-audit stationary counts improved:

| Check | Before guard | After guard |
|---|---:|---:|
| stationary_fraction > 0.5 | 5 / 50 | 4 / 50 |
| stationary_fraction > 0.8 | 3 / 50 | 2 / 50 |
| stationary_fraction > 0.9 | 2 / 50 | 1 / 50 |

Top-10 reason-code distribution after guard:

| Reason code | Count |
|---|---:|
| HIGH_NOVELTY_SINGLETON | 1 |
| RARE_MOTION_PRIMITIVES | 2 |
| RARE_TEMPORAL_COMPOSITION | 5 |
| REDUNDANT_KNOWN_WORKFLOW | 2 |

Top-200 parent concentration did not change:

```text
largest parent fraction@200 = 0.53
```

Interpretation:

The guard fixed the specific visual-audit issue without broad score churn. It is a conservative improvement over the previous hardened candidate. The remaining risk is still parent-cluster concentration, not stationary singleton collapse.

Local tests after adding the guard:

```text
208 tests OK
```

## 31. Diversity Skepticism Pass: Parent Cap and Tiered Child Caps

After the stationary singleton guard, the main unresolved risk was diversity collapse inside the top ranks. The guarded candidate had good quality and corruption behavior, but the top 200 still concentrated heavily in one parent motion cluster:

```text
largest parent fraction@200 = 0.53
dominant parent count@200 = 106 / 200
```

We tested three diversity variants on Modal, all using:

- `window_mean_std_pool`
- `quality_gated_grammar`
- physical-source pretrain support manifest
- feature-space large-cluster splitting, not score round-robin
- raw-signal corruption eval
- stationary singleton guard

### Variant A: parentcap200_p8

Config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_parentcap200_p8.json
```

Modal runs:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-3fQXedVdjpHCk6TwDtV3TH
full:  https://modal.com/apps/vishkrish200/main/ap-SLEn95icu7PDEbnnfenIE1
audit: https://modal.com/apps/vishkrish200/main/ap-xsgDiNSLqeorykt7s5nRa3
visual:https://modal.com/apps/vishkrish200/main/ap-EPEffz2Vsx9nk9djZXH2Bq
```

Result:

| Metric | Value |
|---|---:|
| precision@100 | 0.9500 |
| nDCG@100 | 0.9537 |
| corruption_rate@100 | 0.0000 |
| top-100 parent largest fraction | 0.46 |
| top-200 parent largest fraction | 0.695 |

Interpretation:

This failed the intended diversity fix. Extending the parent prefix cap to 200 caused fallback inside the parent-prefix stage. Because only about 62 high-quality slots are available before all small parent clusters are exhausted, the remaining top-200 slots collapsed back into a dominant parent. This variant should not be promoted.

### Variant B: tiered_childcap2_5

Implemented a new reranker:

```text
tiered_cluster_cap
```

It enforces:

```text
top 50:  max 2 per child/subcluster
top 200: max 5 per child/subcluster
```

Config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5.json
```

Modal runs:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-VO57FQ2b8N2dXnTgOH80M7
full:  https://modal.com/apps/vishkrish200/main/ap-WxaW4Lwkt0rPGIcy95wfOr
audit: https://modal.com/apps/vishkrish200/main/ap-i3cvGFKlfoqZDfCi3P5otQ
visual:https://modal.com/apps/vishkrish200/main/ap-AIjULqZLSsKsRCzMSSgYOE
```

Result:

| Metric | Value |
|---|---:|
| precision@100 | 0.9000 |
| nDCG@100 | 0.9163 |
| corruption_rate@100 | 0.0000 |
| top-50 child max count | 2 |
| top-100 child max count | 5 |
| top-100 parent largest fraction | 0.33 |
| top-200 parent largest fraction | 0.50 |

Interpretation:

This improved parent concentration substantially at top 100 and kept the safety metrics clean, but the top-200 child cap could only hold through rank 170. With the existing 75-sized subclusters, max-5 capacity was only 170 high-quality slots, so ranks 171-200 had to fallback.

### Variant C: tiered_childcap2_5_subcluster40

Changed large-cluster split granularity:

```text
target_subcluster_size: 75 -> 40
```

This increases child-cluster capacity so the tiered max-5 rule can fill the whole top 200 without fallback.

Config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40.json
```

Modal runs:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-ffflSePs6ii0HLygPUBizn
full:  https://modal.com/apps/vishkrish200/main/ap-B2n88U4Rtz3NokBTrtzHuS
audit: https://modal.com/apps/vishkrish200/main/ap-qQZrQSjpG4HOcfvvtK3Fo3
visual:https://modal.com/apps/vishkrish200/main/ap-OIDgYT9uhaUSMGD9kNjrkZ
```

Result:

| Metric | Value |
|---|---:|
| support clips | 68,208 |
| query clips | 2,000 |
| post-split query clusters | 78 |
| precision@100 | 0.9600 |
| nDCG@100 | 0.9623 |
| corruption_rate@100 | 0.0000 |
| top-100 low-quality count | 0 |
| top-10 mean quality | 0.9880 |
| top-50 child max count | 2 |
| top-100 child max count | 5 |
| top-200 child max count | 5 |
| top-100 parent largest fraction | 0.50 |
| top-200 parent largest fraction | 0.56 |

Interpretation:

This is the best candidate by pseudo ranking metrics and child-cluster diversity. It exactly satisfies the child-cluster cap through top 200 with no fallback. It does not solve parent-level concentration; parent concentration is worse than the original stationary-guard candidate, because the finer subclusters give the dominant parent more legitimate child slots.

### Current Decision

If the selection criterion is:

```text
precision@100 >= 0.90
corruption_rate@100 = 0
top-100 low-quality count = 0
unique child clusters@200 >= 30
child max count@200 <= 5
```

then `tiered_childcap2_5_subcluster40` is the current best candidate.

If the selection criterion prioritizes parent-level diversity, the original `stationary_guard` candidate is still safer:

```text
stationary_guard parent largest fraction@100 = 0.44
stationary_guard parent largest fraction@200 = 0.53
subcluster40 parent largest fraction@100 = 0.50
subcluster40 parent largest fraction@200 = 0.56
```

Do not promote `parentcap200_p8`.

Local tests after these changes:

```text
61 ranking tests OK
compileall OK
```

## 32. Source-Blocked Validation Redesign

The review after the diversity pass concluded that the largest remaining problem is evaluation, not modeling. The new-vs-pretrain candidate eval is still structurally easy, and physical leave-cluster eval is useful but still labels positives by the same feature-space abstraction used by the ranker.

Added a harder validation lane:

```text
marginal_value/eval/source_blocked_eval.py
modal_source_blocked_eval.py
configs/source_blocked_eval.json
configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
tests/test_source_blocked_eval.py
```

The protocol:

1. sample old physical-source pretrain rows
2. parse source groups from manifest URLs, usually `workerNNNNN`
3. cluster old rows in `window_mean_std_pool` feature space
4. choose held-out feature clusters and held-out source groups for each fold
5. remove the held-out novel region from support
6. remove all candidate rows from support, including negatives
7. rank:
   - positives from held-out source groups inside held-out feature clusters
   - hard negatives from source-covered regions still represented in support
   - optional raw-signal corruption negatives

This means positives and negatives are both old-corpus rows, unlike the submission candidate eval. The evaluator also reuses the production ranking machinery: grammar promotion, stationary singleton guard, large-cluster split, and parent-prefix or tiered cluster-cap reranking.

The first comparison should be:

```bash
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval.json --smoke
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json --smoke
```

Then full Modal runs if smoke passes:

```bash
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval.json
.venv/bin/modal run modal_source_blocked_eval.py --config-path configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
```

Decision rule:

- use source-blocked metrics as the primary validation signal
- keep raw corruption@100 and low-quality@100 at zero
- choose a challenger over `stationary_guard` only if it improves blocked validation without worsening parent/source concentration

### Source-Blocked Smoke Results

Both first smoke runs completed on Modal.

Current safer candidate:

```text
config: configs/source_blocked_eval.json
Modal run: https://modal.com/apps/vishkrish200/main/ap-L9TKuzrnf0HTPFGsbzmVok
artifact: /artifacts/eval/source_blocked/window_mean_std_physical_source/source_blocked_eval_smoke.json
```

Smoke summary:

| Metric | Value |
|---|---:|
| rows sampled | 512 |
| source groups | 485 |
| source clusters | 12 |
| mean precision@100 | 0.3400 |
| mean nDCG@100 | 0.3575 |
| corruption@100 / @200 | 0.0000 / 0.0000 |

Child-diversity challenger:

```text
config: configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
Modal run: https://modal.com/apps/vishkrish200/main/ap-3cOARTruuWnxmHYnbwuf8d
artifact: /artifacts/eval/source_blocked/window_mean_std_physical_source_tiered_childcap2_5_subcluster40/source_blocked_eval_smoke.json
```

Smoke summary:

| Metric | Value |
|---|---:|
| rows sampled | 512 |
| source groups | 485 |
| source clusters | 12 |
| mean precision@100 | 0.3300 |
| mean nDCG@100 | 0.3545 |
| corruption@100 / @200 | 0.0000 / 0.0000 |

Interpretation:

The source-blocked smoke is doing its job: metrics are much less comfortable than the old candidate eval. On smoke, `subcluster40` does not beat the safer `stationary_guard`-style config. Full runs are still needed before treating this as final evidence, but the first signal supports keeping `stationary_guard` as the default.

### Source-Blocked Full Results

Both full source-blocked runs completed on Modal over 20,000 old physical-source rows, 5,437 parsed source groups, and 64 source clusters.

Current safer candidate:

```text
config: configs/source_blocked_eval.json
Modal run: https://modal.com/apps/vishkrish200/main/ap-1mLZqeVa7NoGswra05c1rC
artifact: /artifacts/eval/source_blocked/window_mean_std_physical_source/source_blocked_eval_full.json
candidates: /artifacts/eval/source_blocked/window_mean_std_physical_source/source_blocked_candidates_full.jsonl
```

Full summary:

| Metric | Value |
|---|---:|
| rows sampled | 20,000 |
| source groups | 5,437 |
| source clusters | 64 |
| mean precision@100 | 0.2500 |
| mean nDCG@100 | 0.2149 |
| corruption@100 / @200 | 0.0000 / 0.0000 |

Child-diversity challenger:

```text
config: configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json
Modal run: https://modal.com/apps/vishkrish200/main/ap-RyLA2XWehIEQXNfnxAtiWK
artifact: /artifacts/eval/source_blocked/window_mean_std_physical_source_tiered_childcap2_5_subcluster40/source_blocked_eval_full.json
candidates: /artifacts/eval/source_blocked/window_mean_std_physical_source_tiered_childcap2_5_subcluster40/source_blocked_candidates_full.jsonl
```

Full summary:

| Metric | Value |
|---|---:|
| rows sampled | 20,000 |
| source groups | 5,437 |
| source clusters | 64 |
| mean precision@100 | 0.3250 |
| mean nDCG@100 | 0.2897 |
| corruption@100 / @200 | 0.0000 / 0.0000 |

Interpretation:

At full scale, `tiered_childcap2_5_subcluster40` is better on source-blocked recovery than the safer parent-prefix candidate. This changes the candidate comparison: the challenger is no longer just a pseudo-metric winner, it also wins the first harder blocked validation pass. The unresolved risk is still parent/source concentration on the real new split, so the next decision should compare full source-blocked wins against new-split parent concentration and top-K source diversity before submission.

## 33. Submission-Readiness Decision Audit

Decision memo:

```text
docs/submission_readiness_decision_2026-04-26.md
```

Compact JSON evidence was pulled locally from Modal into:

```text
data/decision_audit_2026_04_26/raw/
```

Reviewed:

- full source-blocked reports for `stationary_guard` and `tiered_childcap2_5_subcluster40`
- full new-split ranking reports
- full ranking audit reports
- full visual audit reports

Final decision from this audit:

```text
PROMOTE tiered_childcap2_5_subcluster40 as the current submission candidate.
```

Reason:

- `subcluster40` wins source-blocked P@100 on every fold
- mean source-blocked P@100 improves from `0.250` to `0.325`
- mean source-blocked nDCG@100 improves from `0.215` to `0.290`
- top-100 and top-200 corruption remain zero
- top-100 and top-200 low-quality counts remain zero
- child-cluster diversity is materially better
- parent concentration is worse but not disqualifying: top-100 largest parent rises from `0.44` to `0.50`, and top-200 largest parent rises from `0.53` to `0.56`
- the dominant parent spans many split child clusters, so the risk looks like broad workflow concentration rather than one uncapped duplicate mode

Current leading config:

```text
configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40.json
```

Do not train more models. Do not build a hybrid parent-cap variant unless manual inspection of the dominant parent visual audit shows repeated trivial motion or sensor artifacts.

## 34. Final Subcluster40 Packaging

Final package:

```text
data/submissions/subcluster40_2026_04_26/
```

Finalized CSVs:

```text
data/submissions/subcluster40_2026_04_26/submission_subcluster40_worker_id.csv
data/submissions/subcluster40_2026_04_26/submission_subcluster40_new_worker_id.csv
```

Source artifacts pulled locally:

```text
data/final_subcluster40_2026_04_26/raw/baseline_submission_val_full.csv
data/final_subcluster40_2026_04_26/raw/baseline_diagnostics_val_full.csv
data/final_subcluster40_2026_04_26/raw/new_urls.txt
data/final_subcluster40_2026_04_26/visual_audit/
```

Checks passed:

- 2,000 rows
- 2,000 unique public sample IDs
- no duplicate IDs
- no internal 64-character hash IDs in finalized files
- ranks exactly `1..2000`
- scores non-increasing
- top-100 low-quality count is `0`
- top-100 minimum quality is `0.878218`

Visual sanity:

- downloaded the subcluster40 visual audit HTML, plot index, top rows, dominant-parent rows, and trace plots
- built contact sheets for top-ranked and dominant-parent examples
- dominant parent still exists as a concentration risk, but inspected traces are varied and do not show one repeated flatline, saturation, or obvious sensor-artifact mode

## 35. Scientific Validity Reassessment

New document:

```text
docs/scientific_validity_plan_2026-04-27.md
```

Current scientific status:

```text
plausible but unproven
```

The external reviews and post-review base-rate check weakened confidence in source-blocked validation. `subcluster40` still beats `stationary_guard`, but its mean source-blocked P@100 is only about equal to the fold positive-rate baseline:

| Candidate | mean P@100 | mean fold positive baseline | mean P@100 minus baseline |
|---|---:|---:|---:|
| `stationary_guard` | 0.250 | about 0.320 | -0.070 |
| `subcluster40` | 0.325 | about 0.320 | +0.005 |

This means source-blocked validation is not proof of marginal data value. The next validation should test whether selected top-K clips improve coverage of a held-out target set under representations the ranker did not optimize.

Proposed next eval:

```text
marginal coverage validation
```

Core question:

```text
If selected clips are added to old support, does an independently held-out target set become better covered than under random/quality/diverse baselines?
```

The eval should measure coverage gain in multiple representations:

- current `window_mean_std_pool`
- temporal-order features
- frequency/raw-shape features
- grammar/token features

Do not claim scientific validity until the ranker beats simple baselines in non-ranker representations.

## 36. Marginal Coverage Validation Implementation

Implemented the falsification eval:

```text
marginal_value/eval/marginal_coverage_eval.py
modal_marginal_coverage_eval.py
configs/marginal_coverage_eval_subcluster40.json
tests/test_marginal_coverage_eval.py
```

Protocol:

- hold out physical-source worker/source groups from old support
- split held-out rows into candidate and target sets
- add source-covered distractors to the candidate pool
- rank candidates with the production `subcluster40` stack
- compare top-K coverage gain against `random_high_quality`, `quality_only`, `old_novelty_only`, `new_density_only`, `final_score_only`, and `diverse_source_cluster`
- measure nearest-neighbor coverage gain on `window_mean_std_pool`, `temporal_order`, `raw_shape_stats`, and `grammar_features`

Local checks passed:

```text
python3 -m unittest tests.test_source_blocked_eval tests.test_marginal_coverage_eval
python3 -m compileall marginal_value/eval/marginal_coverage_eval.py modal_marginal_coverage_eval.py
```

Remote smoke command:

```text
.venv/bin/modal run modal_marginal_coverage_eval.py --config-path configs/marginal_coverage_eval_subcluster40.json --smoke
```

Remote smoke artifacts:

```text
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_report_smoke.json
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_candidates_smoke.jsonl
```

Local pulled copy:

```text
data/marginal_coverage_smoke_2026_04_27/marginal_coverage_report_smoke.json
data/marginal_coverage_smoke_2026_04_27/marginal_coverage_candidates_smoke.jsonl
```

Smoke warning:

| Policy | K | window mean/std rel gain | temporal-order rel gain | raw-shape rel gain |
|---|---:|---:|---:|---:|
| `ranker` | 100 | 0.2125 | 0.0180 | 0.0387 |
| `random_high_quality` | 100 | 0.1973 | 0.0709 | 0.0642 |
| `quality_only` | 100 | 0.3773 | 0.0659 | 0.1378 |
| `diverse_source_cluster` | 100 | 0.0360 | 0.0469 | 0.1031 |

The smoke result is not submission-grade evidence because it is only one fold, but it is a warning: `subcluster40` did not beat simple baselines in the non-ranker temporal/raw representations. Full Modal marginal-coverage validation is now the next decision point.

Implementation note:

- the uncapped full config attempted to enumerate `68,208` physical-source cached rows, which is too expensive for the first raw-shape validation pass
- the decision-run config is capped at `20,000` rows, matching the source-blocked eval scale
- the first 20k run reached `14,000 / 20,000` raw-shape rows before the Modal local client disconnected; the rerun keeps `20,000` rows but caps `raw_shape_max_samples` at `1,800` per clip so it can finish reliably

Full remote command:

```text
.venv/bin/modal run --detach modal_marginal_coverage_eval.py --config-path configs/marginal_coverage_eval_subcluster40.json
```

Remote full artifacts:

```text
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_report_full.json
/artifacts/eval/marginal_coverage/subcluster40/marginal_coverage_candidates_full.jsonl
```

Local pulled copy:

```text
data/marginal_coverage_full_2026_04_27/marginal_coverage_report_full.json
data/marginal_coverage_full_2026_04_27/marginal_coverage_candidates_full.jsonl
```

Full run:

| Field | Value |
|---|---:|
| rows | 20,000 |
| source groups | 5,437 |
| folds | 4 |
| fold candidate count range | 756-828 |
| fold target count range | 320-365 |

Mean relative coverage gain:

| Policy | K | window mean/std | temporal-order | raw-shape |
|---|---:|---:|---:|---:|
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
subcluster40 is not scientifically validated by marginal-coverage testing.
```

It beats random high-quality in temporal-order coverage, but it does not beat simple baselines in two non-ranker representations. It loses raw-shape coverage to random high-quality and quality-only, and it loses K=100 temporal/raw/window coverage to old-novelty-only. The current stack should be treated as a plausible heuristic, not as a proven marginal-value algorithm.

## 37. Simple Baseline Reset Planning

Claude review artifact:

```text
.omx/artifacts/claude-simple-baseline-redesign-review-20260427.md
```

Reset plan:

```text
docs/simple_baseline_reset_plan_2026-04-27.md
```

Core reframing:

```text
Pick new IMU clips that are clean enough to be usable and cover behavior not already well represented in old support.
```

Recommended next candidates:

- `quality_gated_old_novelty`: hard quality gate, then rank by old-support novelty
- `quality_gated_old_novelty_sourcecap`: same ranking with one simple source cap as a redundancy-control challenger
- `quality_score * old_novelty_score`: keep only as a comparator, not the primary scientific design

Important implementation detail:

- quality is now treated as a gate, not as the primary score
- novelty is the ranking criterion after the gate
- grammar, density, feature subclustering, tiered child caps, and stationary score guards are out of the critical path

## 38. Quality-Gated Old-Novelty Reset Implementation And Eval

Implemented the deterministic reset family:

```text
marginal_value/ranking/baseline_ranker.py
marginal_value/ranking/modal_baseline_rank.py
marginal_value/eval/marginal_coverage_eval.py
configs/marginal_coverage_eval_qgate_oldnovelty.json
configs/baseline_ranking_new_qgate_oldnovelty_knn5.json
configs/baseline_ranking_new_quality_only.json
tests/test_baseline_ranking.py
tests/test_marginal_coverage_eval.py
```

Main rule:

```text
pass_i = quality_score_i >= threshold
rank by pass_i desc, old_novelty_score desc, quality_score desc, sample_id asc
```

The full Modal marginal-coverage run completed:

```text
.venv/bin/modal run modal_marginal_coverage_eval.py --config-path configs/marginal_coverage_eval_qgate_oldnovelty.json
```

Remote artifacts:

```text
/artifacts/eval/marginal_coverage/qgate_oldnovelty/marginal_coverage_report_full.json
/artifacts/eval/marginal_coverage/qgate_oldnovelty/marginal_coverage_candidates_full.jsonl
```

Local pulled report:

```text
data/modal_reports/qgate_oldnovelty/marginal_coverage_report_full.json
```

Full run:

| Field | Value |
|---|---:|
| rows | 20,000 |
| source groups | 5,437 |
| folds | 4 |

Mean relative coverage gain on primary non-ranker aggregate:

| Policy | K | temporal-order | raw-shape | primary average |
|---|---:|---:|---:|---:|
| `random_high_quality` | 100 | 0.0034 | 0.0618 | 0.0326 |
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 |
| `old_novelty_only` | 100 | 0.0173 | 0.0572 | 0.0372 |
| `quality_gated_old_novelty_q45` | 100 | 0.0173 | 0.0572 | 0.0372 |
| `quality_gated_old_novelty_q45_sourcecap2` | 100 | 0.0157 | 0.0573 | 0.0365 |
| `random_high_quality` | 200 | 0.0063 | 0.0955 | 0.0509 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 |
| `old_novelty_only` | 200 | 0.0199 | 0.0703 | 0.0451 |
| `quality_gated_old_novelty_q45` | 200 | 0.0199 | 0.0705 | 0.0452 |
| `quality_gated_old_novelty_q45_sourcecap2` | 200 | 0.0181 | 0.0702 | 0.0441 |

Decision:

```text
quality_gated_old_novelty is a useful control, not a validated winner.
```

The quality gate improves minimum selected quality but barely changes coverage because old-novelty top ranks are already mostly high quality. Sourcecap2 lowers concentration but also slightly hurts coverage. The strongest supported policy on the primary temporal+raw aggregate is currently `quality_only` at K=100 and K=200. Old-novelty remains best in ranker-space and temporal coverage, but its raw-shape weakness is real.

Execution configs now exist for both honest submission-capable candidates:

```text
configs/baseline_ranking_new_quality_only.json
configs/baseline_ranking_new_qgate_oldnovelty_knn5.json
```

Quality-only remote smoke execution passed:

```text
.venv/bin/modal run modal_rank.py --config-path configs/baseline_ranking_new_quality_only.json --smoke
```

Smoke artifacts:

```text
/artifacts/ranking/window_mean_std_quality_only_new/baseline_ranking_report_smoke.json
/artifacts/ranking/window_mean_std_quality_only_new/baseline_submission_val_smoke.csv
```

Next scientific move:

```text
Do not add grammar/density/subclusters back. Decide whether to use quality_only as the honest baseline, or redesign the novelty representation because current old-support novelty is not improving non-ranker coverage enough.
```

## 39. Representation-Specific Novelty And Media Audit Reset

Added representation-specific novelty controls to the marginal-coverage eval:

```text
old_novelty_temporal_order
old_novelty_raw_shape_stats
quality_gated_old_novelty_temporal_order_q{threshold}
quality_gated_old_novelty_raw_shape_stats_q{threshold}
```

The goal was diagnostic, not leaderboard tuning: test whether novelty itself is failing, or whether the original `window_mean_std_pool` novelty representation is the wrong geometry for held-out coverage.

Full Modal run:

```text
.venv/bin/modal run modal_marginal_coverage_eval.py --config-path configs/marginal_coverage_eval_qgate_oldnovelty.json
```

Remote artifact:

```text
/artifacts/eval/marginal_coverage/qgate_oldnovelty/marginal_coverage_report_full.json
```

Local reports:

```text
data/modal_reports/qgate_oldnovelty/marginal_coverage_report_full.json
docs/qgate_oldnovelty_scientific_diagnostic_2026-04-27.md
```

Key full-run result:

| Policy | K | temporal-order | raw-shape | primary average | min quality |
|---|---:|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 | 1.000 |
| `quality_gated_old_novelty_raw_shape_stats_q45` | 100 | 0.0029 | 0.1893 | 0.0961 | 0.499 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 100 | 0.0030 | 0.1688 | 0.0859 | 0.865 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 | 0.999 |
| `quality_gated_old_novelty_raw_shape_stats_q45` | 200 | 0.0098 | 0.2020 | 0.1059 | 0.479 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 200 | 0.0099 | 0.1824 | 0.0962 | 0.856 |
| `quality_only` | 400 | 0.0112 | 0.1341 | 0.0726 | 0.997 |
| `quality_gated_old_novelty_raw_shape_stats_q45` | 400 | 0.0178 | 0.2134 | 0.1156 | 0.477 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 400 | 0.0183 | 0.1913 | 0.1048 | 0.856 |

Interpretation:

```text
Novelty is not dead. Window-space novelty was the weak part.
Raw-shape old-support novelty is the first simple method that robustly beats quality_only on the primary temporal+raw aggregate, including with a strict q85 quality gate.
```

Important caveat:

```text
The raw-shape win is mostly raw-shape coverage. It is not a temporal-order breakthrough at K=50/100.
```

Executable ranking support was added for:

```text
ranking.representation = "raw_shape_stats"
ranking.representation = "temporal_order"
```

New submission-capable configs:

```text
configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_knn5.json
configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_knn5.json
configs/baseline_ranking_new_qgate_oldnovelty_temporal_knn5.json
```

Raw-shape q85 smoke execution passed:

```text
.venv/bin/modal run modal_rank.py --config-path configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_knn5.json --smoke
```

Smoke artifacts:

```text
/artifacts/ranking/raw_shape_stats_qgate85_oldnovelty_knn5_new/baseline_ranking_report_smoke.json
/artifacts/ranking/raw_shape_stats_qgate85_oldnovelty_knn5_new/baseline_submission_val_smoke.csv
```

Media audit:

```text
.venv/bin/modal run modal_top_clip_visual_audit.py --config-path configs/top_clip_visual_audit_qgate_oldnovelty_raw_shape_q85.json --smoke
```

Local pulled smoke report:

```text
data/modal_reports/qgate_raw_shape_q85_visual_audit/top_clip_visual_audit_report_smoke.json
```

Media finding:

```text
q85 removes the obvious max-abs 108 rank-1 artifact seen under q45, but the top-5 smoke set still contains stationary-heavy clips.
```

Next scientific move:

```text
Do not reintroduce grammar/density/subclusters. The next small test should be raw-shape q85 old-novelty with one explicit physical validity gate for stationary-heavy clips, compared against raw-shape q85 and quality_only under the same marginal-coverage eval and visual audit.
```

## 40. Scientific/Media Reset: q85 Raw-Shape Novelty Plus Stationary Validity

Added a minimal physical-validity gate to the deterministic quality-gated old-novelty family:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
rank remaining rows by old-support novelty in raw_shape_stats
```

This is not a score cocktail and does not reintroduce grammar, density, subclusters, or tiered caps. It is a hard validity filter motivated by the visual/media audit finding that q85 still admitted stationary-heavy top clips.

New executable ranking config:

```text
configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_knn5.json
```

New visual-audit config:

```text
configs/top_clip_visual_audit_qgate_oldnovelty_raw_shape_q85_stat90.json
```

The marginal-coverage eval now includes stat90 policies such as:

```text
quality_gated_old_novelty_raw_shape_stats_q85_stat90
```

Full Modal eval completed:

```text
.venv/bin/modal run modal_marginal_coverage_eval.py --config-path configs/marginal_coverage_eval_qgate_oldnovelty.json
```

Remote run:

```text
https://modal.com/apps/vishkrish200/main/ap-kCy0GHEvqEDw8dXKQzE9ah
```

Key result:

| Policy | K | temporal-order | raw-shape | primary average | min quality | stationary >0.90 |
|---|---:|---:|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 | 1.000 | 0.035 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 100 | 0.0030 | 0.1688 | 0.0859 | 0.865 | 0.015 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 100 | 0.0030 | 0.1688 | 0.0859 | 0.868 | 0.000 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 | 0.999 | 0.024 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 200 | 0.0099 | 0.1824 | 0.0962 | 0.856 | 0.010 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 200 | 0.0099 | 0.1824 | 0.0962 | 0.860 | 0.000 |
| `quality_only` | 400 | 0.0112 | 0.1341 | 0.0726 | 0.997 | 0.021 |
| `quality_gated_old_novelty_raw_shape_stats_q85` | 400 | 0.0183 | 0.1913 | 0.1048 | 0.856 | 0.006 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 400 | 0.0183 | 0.1913 | 0.1048 | 0.859 | 0.000 |

Interpretation:

```text
q85_stat90 preserves the raw-shape q85 coverage result while removing high-stationary selections from the measured top-K summaries. This is now the best scientific/media compromise among the simple deterministic methods.
```

Remaining caveat:

```text
The win is still mostly raw-shape coverage. Temporal-order marginality is weak at K=50/100, so do not claim broad temporal behavior discovery yet.
```

## 41. Scientific/Media Reset: Add Spike Validity Gate

The full visual audit for q85_stat90 found that the stationary gate worked, but the top-ranked clip still had a large acceleration spike:

```text
rank 1 max_abs_value ~= 142
```

That is a media and physical-validity failure mode, so the next minimal gate was added:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
rank remaining rows by old-support novelty in raw_shape_stats
```

This remains a deterministic filter-plus-novelty rule. It does not use grammar, density, feature subclustering, tiered caps, learned labels, or weighted score blending.

New executable ranking config:

```text
configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_abs60_knn5.json
```

New visual-audit config:

```text
configs/top_clip_visual_audit_qgate_oldnovelty_raw_shape_q85_stat90_abs60.json
```

Full Modal runs:

```text
marginal coverage: https://modal.com/apps/vishkrish200/main/ap-40N9yy9tHS0Eo4Z8xQEnyY
ranking:           https://modal.com/apps/vishkrish200/main/ap-ER6Qlrl2DlkzpGVs1YJKmA
visual audit:      https://modal.com/apps/vishkrish200/main/ap-TFz1kLpWMDmVnkBo6uNhaY
```

Local artifacts:

```text
data/modal_reports/qgate_oldnovelty/marginal_coverage_report_full.json
docs/qgate_oldnovelty_scientific_diagnostic_2026-04-27.md
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_visual_audit/top_clip_visual_audit_report_full.json
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_visual_audit/top_clip_visual_audit_index_full.html
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_visual_audit/top_clip_trace_plots_full/
```

Key marginal-coverage result:

| Policy | K | temporal-order | raw-shape | primary average | max abs | max abs >60 | stationary >0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 | 23.0 | 0.000 | 0.035 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 100 | 0.0030 | 0.1688 | 0.0859 | 106.3 | 0.043 | 0.000 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60` | 100 | 0.0030 | 0.1658 | 0.0844 | 52.4 | 0.000 | 0.000 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 | 29.4 | 0.000 | 0.024 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 200 | 0.0099 | 0.1824 | 0.0962 | 106.8 | 0.025 | 0.000 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60` | 200 | 0.0099 | 0.1788 | 0.0944 | 53.6 | 0.000 | 0.000 |
| `quality_only` | 400 | 0.0112 | 0.1341 | 0.0726 | 37.4 | 0.000 | 0.021 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90` | 400 | 0.0183 | 0.1913 | 0.1048 | 106.8 | 0.014 | 0.000 |
| `quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60` | 400 | 0.0183 | 0.1880 | 0.1031 | 53.6 | 0.000 | 0.000 |

Full new-split visual audit:

| K | mean quality | min quality | max stationary | max abs | parent largest fraction | unique parent clusters |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9920 | 0.9750 | 0.5589 | 52.21 | 0.10 | 10 |
| 50 | 0.9762 | 0.8744 | 0.7831 | 52.97 | 0.02 | 50 |
| 100 | 0.9762 | 0.8565 | 0.8072 | 56.49 | 0.07 | 91 |
| 200 | 0.9772 | 0.8506 | 0.8972 | 56.49 | 0.29 | 123 |

Interpretation:

```text
q85_stat90_abs60 is the current best science/media compromise. It loses only a small amount of raw-shape coverage versus q85_stat90, while removing all max_abs >60 and stationary >0.90 selections in the measured top-K summaries.
```

Remaining caveats:

```text
1. The coverage win is still mostly raw-shape coverage, not temporal-order discovery.
2. Parent/source concentration reappears by K=200: parent cluster 0 is 29% of top 200. The next small test should be one simple parent/source cap, not a return to grammar/density/subclusters.
3. Candidate-eval P@100 remains only a sanity metric because it is still new-vs-pretrain, not decision-grade validation.
```

## 42. Scientific/Media Reset: Simple Cluster Cap Diagnostic

Added one deterministic diversity-control variant on top of the current simple rule:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
rank passing rows by old-support novelty in raw_shape_stats
cap early selections at 2 per new_cluster_id
```

This is still intentionally small. It does not reintroduce grammar, density, learned rankers, feature subclustering, tiered child caps, or weighted score cocktails.

New executable configs:

```text
configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_abs60_clustercap2_knn5.json
configs/top_clip_visual_audit_qgate_oldnovelty_raw_shape_q85_stat90_abs60_clustercap2.json
```

Full Modal runs:

```text
ranking:      https://modal.com/apps/vishkrish200/main/ap-YTqKlnk7sFhrL5jUvBFsrL
visual audit: https://modal.com/apps/vishkrish200/main/ap-xCPC8gheFyvLFafjUBqSNH
```

Local artifacts:

```text
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_clustercap2_visual_audit/baseline_ranking_report_full.json
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_clustercap2_visual_audit/baseline_diagnostics_val_full.csv
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_clustercap2_visual_audit/top_clip_visual_audit_report_full.json
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_clustercap2_visual_audit/top_clip_visual_audit_index_full.html
data/modal_reports/qgate_raw_shape_q85_stat90_abs60_clustercap2_visual_audit/top_clip_trace_plots_full/
```

Candidate-eval sanity result:

```text
precision@100 = 0.86
ndcg@100 = 0.8703
```

Do not treat that as decision-grade validation; it is still new-vs-pretrain.

Full new-split visual audit:

| Variant | K | mean quality | min quality | max stationary | max abs | parent largest fraction | unique parent clusters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q85_stat90_abs60` | 100 | 0.9762 | 0.8565 | 0.8072 | 56.49 | 0.07 | 91 |
| `q85_stat90_abs60_clustercap2` | 100 | 0.9781 | 0.8565 | 0.7831 | 56.49 | 0.02 | 96 |
| `q85_stat90_abs60` | 200 | 0.9772 | 0.8506 | 0.8972 | 56.49 | 0.29 | 123 |
| `q85_stat90_abs60_clustercap2` | 200 | 0.9781 | 0.8506 | 0.8972 | 56.49 | 0.235 | 135 |

Interpretation:

```text
clustercap2 is a useful media/top100 diversity control. It reduces top100 parent concentration from 7% to 2% and slightly improves unique parent coverage without hurting quality or the physical validity gates.
```

Important limitation:

```text
It does not fully solve K=200 concentration. After q85/stat90/abs60, only 146 passing rows are selected before the cap fallback, across 135 passing clusters; the top200 therefore must include fallback rows from the largest cluster unless we relax quality/validity gates, stop filling top200 from capped fallbacks, or introduce a stronger source/worker cap with reliable metadata.
```

Implementation hardening:

```text
quality-gated source caps now reject missing explicit cap keys instead of silently falling back to per-sample IDs. If source_cap_key is new_cluster_parent_id and no parent annotation exists, it falls back to new_cluster_id, matching the cluster identity used by the visual audit.
```

Current decision:

```text
Use q85_stat90_abs60 as the scientific marginal-coverage-supported baseline.
Use q85_stat90_abs60_clustercap2 as the current media/top100 diversity candidate.
Do not call clustercap2 a scientific improvement until it is run through marginal-coverage validation with the capped ranking policy.
```

## 43. Scientific/Media Reset: Marginal Coverage for Cluster Cap

The marginal-coverage eval now explicitly measures capped eval-representation old-novelty policies. This was necessary because the earlier `source_caps` setting only applied to the default old-novelty policy, not to the actual raw-shape q85/stat90/abs60 candidate.

Implementation changes:

```text
quality_gated_old_novelty.source_cap_key = "new_cluster_id"
quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60_clustercap2
```

For eval-representation policies, cluster IDs are computed in that same representation before the cap is applied. So `raw_shape_stats_*_clustercap2` uses raw-shape candidate clusters, matching the ranking candidate more closely than a window-space cap would.

Full Modal run:

```text
smoke: https://modal.com/apps/vishkrish200/main/ap-3amKfFCcmc3ezLcEsQW2Qg
full:  https://modal.com/apps/vishkrish200/main/ap-mBiDhqGfyRSwaVjMRCn5pm
```

Local artifacts:

```text
data/modal_reports/qgate_oldnovelty/marginal_coverage_report_full.json
data/modal_reports/qgate_oldnovelty/marginal_coverage_candidates_full.jsonl
```

Key marginal-coverage comparison:

| Policy | K | temporal-order | raw-shape | primary average | min quality | stationary >0.90 | max abs >60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 | 1.000 | 0.035 | 0.000 |
| `q85_stat90_abs60` | 100 | 0.0030 | 0.1658 | 0.0844 | 0.864 | 0.000 | 0.000 |
| `q85_stat90_abs60_clustercap2` | 100 | 0.0030 | 0.1682 | 0.0856 | 0.864 | 0.000 | 0.000 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 | 0.999 | 0.024 | 0.000 |
| `q85_stat90_abs60` | 200 | 0.0099 | 0.1788 | 0.0944 | 0.860 | 0.000 | 0.000 |
| `q85_stat90_abs60_clustercap2` | 200 | 0.0099 | 0.1792 | 0.0946 | 0.860 | 0.000 | 0.000 |
| `quality_only` | 400 | 0.0112 | 0.1341 | 0.0726 | 0.997 | 0.021 | 0.000 |
| `q85_stat90_abs60` | 400 | 0.0183 | 0.1880 | 0.1031 | 0.859 | 0.000 | 0.000 |
| `q85_stat90_abs60_clustercap2` | 400 | 0.0183 | 0.1882 | 0.1032 | 0.859 | 0.000 | 0.000 |

Paired primary deltas for `q85_stat90_abs60_clustercap2`:

| Baseline | K | mean delta | fold wins | fold ties | folds |
|---|---:|---:|---:|---:|---:|
| `q85_stat90_abs60` | 50 | -0.0013 | 0 | 0 | 4 |
| `q85_stat90_abs60` | 100 | +0.0012 | 4 | 0 | 4 |
| `q85_stat90_abs60` | 200 | +0.0002 | 2 | 1 | 4 |
| `q85_stat90_abs60` | 400 | +0.0001 | 1 | 2 | 4 |
| `quality_only` | 100 | +0.0439 | 4 | 0 | 4 |
| `old_novelty_only` | 100 | +0.0483 | 4 | 0 | 4 |

Interpretation:

```text
clustercap2 is now scientifically acceptable as a neutral-to-slightly-positive diversity control, not a major marginal-coverage breakthrough.
```

Decision update:

```text
Use q85_stat90_abs60_clustercap2 as the current science/media candidate for top100-style usage because it improves media diversity and does not materially hurt marginal coverage.
Keep q85_stat90_abs60 as the ablation/control because it is the simpler uncapped baseline.
Do not claim the cap solves K=200 concentration; the visual audit still shows residual concentration at K=200.
```

## 44. Scientific Soundness Verdict Layer

The project now has an explicit verdict layer that turns the marginal-coverage report into claim-level gates. This was added because the current candidate ranks in `raw_shape_stats`; therefore raw-shape coverage is no longer an independent validation space. The evaluator must distinguish:

```text
artifact-safe raw-shape/media selection
from
broad held-out behavior discovery
```

Implementation:

```text
marginal_value/eval/scientific_soundness.py
tests/test_scientific_soundness.py
```

Generated artifacts:

```text
docs/scientific_soundness_verdict_2026-04-27.md
data/modal_reports/qgate_oldnovelty/scientific_soundness_verdict_full.json
```

Real full-report verdict for `q85_stat90_abs60_clustercap2`:

| Claim / Gate | Status | Key result |
|---|---|---|
| `artifact_safe_raw_shape_supported` | pass | primary aggregate beats simple controls and physical validity gates pass |
| `broad_behavior_discovery_supported` | fail | temporal-order coverage loses to `old_novelty_only` |
| `primary_vs_simple_controls` | pass | weakest mean delta `+0.0379` vs `quality_only` at K=200, `4/4` fold wins |
| `physical_validity` | pass | max stationary>0.90 = `0.000`, max abs>60 = `0.000`, min quality = `0.850` |
| `uncapped_regression` | pass | weakest mean delta `+0.0002` vs uncapped at K=200 |
| `independent_temporal_vs_controls` | fail | weakest mean delta `-0.0143` vs `old_novelty_only` at K=100, `0/4` fold wins |

Current scientific interpretation:

```text
q85_stat90_abs60_clustercap2 is defensible as an artifact-safe raw-shape/media selector.
It is not yet scientifically supported as broad behavior discovery.
The next claim-closing experiment should target independent temporal behavior coverage or a downstream/blind held-out test on Modal.
```

## 45. Temporal-Novelty Probe Against The Failed Gate

The full marginal-coverage report already includes temporal-order old-novelty variants, so we checked whether a simple temporal-order version of the same quality/validity/cap idea closes the failed broad-behavior gate.

Candidate probed:

```text
quality_gated_old_novelty_temporal_order_q85_stat90_abs60_clustercap2
```

Comparison against the current raw-shape candidate and simple controls:

| Policy | K | temporal-order | raw-shape | primary average | stationary >0.90 | max abs >60 |
|---|---:|---:|---:|---:|---:|---:|
| `quality_only` | 100 | 0.0059 | 0.0775 | 0.0417 | 0.035 | 0.000 |
| `old_novelty_only` | 100 | 0.0173 | 0.0572 | 0.0372 | 0.105 | 0.005 |
| `temporal_q85_stat90_abs60_clustercap2` | 100 | 0.0159 | 0.0389 | 0.0274 | 0.000 | 0.000 |
| `raw_q85_stat90_abs60_clustercap2` | 100 | 0.0030 | 0.1682 | 0.0856 | 0.000 | 0.000 |
| `quality_only` | 200 | 0.0098 | 0.1035 | 0.0566 | 0.024 | 0.000 |
| `old_novelty_only` | 200 | 0.0199 | 0.0703 | 0.0451 | 0.061 | 0.003 |
| `temporal_q85_stat90_abs60_clustercap2` | 200 | 0.0198 | 0.0975 | 0.0587 | 0.000 | 0.000 |
| `raw_q85_stat90_abs60_clustercap2` | 200 | 0.0099 | 0.1792 | 0.0946 | 0.000 | 0.000 |
| `quality_only` | 400 | 0.0112 | 0.1341 | 0.0726 | 0.021 | 0.000 |
| `old_novelty_only` | 400 | 0.0217 | 0.1223 | 0.0720 | 0.033 | 0.003 |
| `temporal_q85_stat90_abs60_clustercap2` | 400 | 0.0227 | 0.1359 | 0.0793 | 0.000 | 0.000 |
| `raw_q85_stat90_abs60_clustercap2` | 400 | 0.0183 | 0.1882 | 0.1032 | 0.000 | 0.000 |

Verdict:

```text
The temporal-order candidate is artifact-safe, and it becomes competitive by K=200/K=400, but it does not beat old_novelty_only at K=100 in temporal-order coverage.
It also loses too much raw-shape coverage at K=100.
Therefore it does not repair the broad-behavior claim.
```

Current next scientific move:

```text
Do not add a weighted raw+temporal hybrid yet.
First decide whether the project claim is raw-shape/media utility or broad behavior discovery.
If the claim is broad behavior discovery, the next meaningful experiment is a blind/downstream evaluation or a new independent temporal representation with predeclared gates.
```

## 46. External Hidden-Test Selector Interface

The repo now includes a deterministic external selector entrypoint so an outside evaluator can test the current algorithm without exposing hidden targets or evaluation features.

Implementation:

```text
marginal_value/select.py
tests/test_external_selector.py
docs/external_hidden_test_protocol_2026-04-27.md
```

Command:

```bash
python3 -m marginal_value.select \
  --old-support old_support.csv \
  --candidate-pool candidate_pool.csv \
  --output ranked_candidates.csv
```

Equivalent installed command:

```bash
marginal-value select \
  --old-support old_support.csv \
  --candidate-pool candidate_pool.csv \
  --output ranked_candidates.csv
```

Input manifests require:

```text
sample_id,raw_path
```

`raw_path` may point to JSONL IMU rows or per-clip CSV files. Relative paths are resolved against the manifest directory.

Frozen default selector:

```text
representation = raw_shape_stats
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
old-support novelty = mean cosine distance to k=5 nearest old-support clips
candidate cluster cap = 2 per new_cluster_id
cluster threshold = 0.985 cosine similarity
```

This is intentionally a selector, not an evaluator. It does not accept:

```text
hidden target rows
labels
evaluation embeddings
target-source metadata
```

Scientific interpretation:

```text
This makes the current narrow claim externally testable.
It does not upgrade the broad active-learning claim.
The evaluator should measure hidden-target coverage or downstream utility using representations/metrics not controlled by this selector.
```

## 47. Detached Cross-Representation Generalization Matrix

The next scientific check is running on Modal in detached mode so the remote job should survive local laptop sleep/disconnect.

Config:

```text
configs/marginal_coverage_eval_cross_rep_matrix.json
```

Launch command:

```bash
.venv/bin/modal run --detach modal_marginal_coverage_eval.py \
  --config-path configs/marginal_coverage_eval_cross_rep_matrix.json
```

Modal run:

```text
https://modal.com/apps/vishkrish200/main/ap-YuVEuf3zUNZQ3TIs3na5FY
```

The app was verified as:

```text
state = ephemeral (detached)
tasks = 1
```

This run is intentionally small in method space:

```text
quality threshold = q85
validity gate = stationary_fraction <= 0.90 and max_abs_value <= 60.0
candidate cap = 2 per new_cluster_id
K = 50, 100, 200, 400
folds = 4
max_rows = 20,000
```

Ranking/novelty representations tested:

```text
window_mean_std_pool
temporal_order
raw_shape_stats
window_shape_stats
```

Coverage representations measured:

```text
window_mean_std_pool
temporal_order
raw_shape_stats
window_shape_stats
```

Expected artifacts:

```text
/artifacts/eval/marginal_coverage/cross_rep_matrix/marginal_coverage_report_full.json
/artifacts/eval/marginal_coverage/cross_rep_matrix/marginal_coverage_candidates_full.jsonl
```

Scientific purpose:

```text
Build a representation-by-representation generalization matrix.
If a representation only wins in itself, the claim remains representation-specific.
If one ranking representation improves multiple independent coverage spaces, that is stronger evidence for broad behavior discovery.
```

## 48. Cross-Representation Matrix Result

The detached Modal cross-representation run completed successfully.

Artifacts downloaded locally:

```text
data/modal_reports/cross_rep_matrix/marginal_coverage_report_full.json
data/modal_reports/cross_rep_matrix/marginal_coverage_candidates_full.jsonl
```

Run dimensions:

```text
mode = full
rows = 20,000
source groups = 5,437
folds = 4
representations = window_mean_std_pool, temporal_order, raw_shape_stats, window_shape_stats
```

Key K=100 result:

| Policy | temporal | raw-shape | window-shape | Interpretation |
|---|---:|---:|---:|---|
| `quality_only` | 0.0059 | 0.0775 | 0.0279 | strong quality/raw control |
| `old_novelty_only` | 0.0173 | 0.0572 | 0.0327 | good temporal/window-shape control |
| `raw_shape_q85_stat90_abs60_clustercap2` | 0.0030 | 0.1682 | 0.0043 | raw-shape specialist, poor generalization |
| `temporal_q85_stat90_abs60_clustercap2` | 0.0159 | 0.0389 | 0.0229 | temporal-safe, weak raw |
| `window_shape_q85_stat90_abs60_clustercap2` | 0.0175 | 0.0994 | 0.0329 | best balanced candidate |

Key K=200 result:

| Policy | temporal | raw-shape | window-shape |
|---|---:|---:|---:|
| `quality_only` | 0.0098 | 0.1035 | 0.0326 |
| `old_novelty_only` | 0.0199 | 0.0703 | 0.0340 |
| `raw_shape_q85_stat90_abs60_clustercap2` | 0.0099 | 0.1792 | 0.0089 |
| `temporal_q85_stat90_abs60_clustercap2` | 0.0198 | 0.0975 | 0.0328 |
| `window_shape_q85_stat90_abs60_clustercap2` | 0.0198 | 0.1189 | 0.0332 |

Interpretation:

```text
The raw-shape candidate should be treated as a raw-shape/media specialist.
The window-shape candidate is now the main scientific candidate because it is much more balanced across temporal, raw-shape, and window-shape coverage.
```

## 49. Challenge-Matched External Selector

The external selector now matches the challenge framing more directly.

Implementation update:

```text
marginal_value/select.py
tests/test_external_selector.py
docs/submission_contract.md
```

Default selector changed from raw-shape specialist to:

```text
window_shape_stats_q85_stat90_abs60_clustercap2
```

Challenge-mode behavior:

```text
old support rows are segmented into non-overlapping 180-second support clips
new candidate rows are ranked as candidate clips
old-support novelty is mean cosine distance to k=5 nearest old support clips
quality gate is quality_score >= 0.85
physical gates are stationary_fraction <= 0.90 and max_abs_value <= 60.0
new-batch diversity cap is 2 per new_cluster_id
```

Supported manifest formats:

```text
CSV with sample_id,raw_path
CSV with worker_id,path/file_path fallback
one-path-per-line text manifest
one-URL-per-line text manifest for JSONL URLs
```

Command:

```bash
python3 -m marginal_value.select \
  --old-support pretrain_paths_or_urls.txt \
  --candidate-pool new_paths_or_urls.txt \
  --output ranked_candidates.csv
```

The selector still does not accept hidden targets, labels, eval embeddings, or target-source metadata.
