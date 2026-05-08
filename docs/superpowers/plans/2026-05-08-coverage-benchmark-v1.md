# Blind Target Coverage Benchmark V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only offline active-learning benchmark that tests whether a candidate acquisition policy improves blind held-out target coverage, using cross-view evaluation and hard hygiene checks.

**Architecture:** Keep this additive under `marginal_value/active_benchmark/`. Reuse `BenchmarkClip`, `EpisodeSpec`, embedding stacking, and source-blocked split concepts. Add a new coverage runner/report layer instead of changing the submitted selector or the existing benchmark defaults.

**Tech Stack:** Python dataclasses, NumPy nearest-neighbor distances, existing unittest suite, JSON/Markdown reports for the first smoke milestone.

---

## Scientific Contract

- [ ] Treat each episode as `support S`, `candidate pool P`, and blind `target T`.
- [ ] Policies may only see `S` and `P`; only oracle diagnostics may inspect `T`.
- [ ] Evaluate selected prefixes `A@K` by whether `S ∪ A@K` lowers nearest-neighbor distance from each target item in `T`.
- [ ] Mark eval rows as primary only when the eval feature family does not overlap the policy selector feature family.
- [ ] Keep TS2Vec as a selector/eval diagnostic, not proof by itself.
- [ ] Block source-group overlap across support, candidate, and target roles.

## Implementation Steps

- [ ] Add synthetic coverage benchmark tests first.
  - [ ] Support clips live near origin.
  - [ ] Candidate pool contains a high-quality near duplicate, a valid far target-covering clip, and an invalid far artifact/low-quality clip.
  - [ ] Blind targets live near the valid far candidate.
  - [ ] Assert TS2Vec novelty/k-center pick the valid far clip.
  - [ ] Assert quality-only picks the high-quality near duplicate and gets low target coverage gain.
  - [ ] Assert invalid clips are never selected when valid alternatives exist.
  - [ ] Assert same-view TS2Vec eval rows are diagnostic, not primary.

- [ ] Add `coverage_runner.py`.
  - [ ] Define coverage benchmark config/result row dataclasses.
  - [ ] Implement v1 policies: `random_valid_v1`, `quality_stratified_random_v1`, `quality_only_v1`, `ts2vec_support_novelty_v1`, `ts2vec_kcenter_v1`, `submitted_full_replay_v1`, and optional `oracle_greedy_eval_view_v1`.
  - [ ] Apply quality, stationary, max-abs, and artifact gates before deployable policy selection.
  - [ ] Produce selected rows, metric rows, and policy summaries for fixed budgets.

- [ ] Add cross-view coverage metrics.
  - [ ] Primary metric: mean relative nearest-neighbor coverage gain, `(d0 - dK) / max(d0, eps)`.
  - [ ] Include absolute distance gain, mean target NN distance before/after, tail gain, tau coverage gain, selected invalid rate, selected duplicate count, target leakage count, and out-of-pool count.
  - [ ] Keep oracle metrics clearly labeled as diagnostics.

- [ ] Add `coverage_reports.py`.
  - [ ] Write JSON and Markdown reports for local smoke runs.
  - [ ] Make report rows explicit enough to later convert to Parquet/CSV for GCP sweeps without changing benchmark semantics.

- [ ] Export the new runner/report helpers from `marginal_value.active_benchmark`.

- [ ] Run focused tests, then the full local suite.

## Acceptance Before Real GCP Runs

- [ ] The synthetic fixture passes and shows the expected policy ordering: target-covering novelty/k-center beats quality-only on blind primary coverage.
- [ ] Same-view TS2Vec evaluation is flagged non-primary for TS2Vec selector policies.
- [ ] Invalid/low-quality/artifact candidates are not selected by deployable v1 policies when valid alternatives exist.
- [ ] Reports contain enough rows to audit selected IDs, coverage metrics, feature-family overlap, and hygiene failures.
- [ ] The full local suite still passes.

## Do Not Do Yet

- [ ] Do not retrain TS2Vec or any downstream model.
- [ ] Do not launch a GPU job.
- [ ] Do not call this a learned query policy.
- [ ] Do not mutate the clean submitted repo.
- [ ] Do not use the target set for deployable policy selection except in explicitly labeled oracle diagnostics.
