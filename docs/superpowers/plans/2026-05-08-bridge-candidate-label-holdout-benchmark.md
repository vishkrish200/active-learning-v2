# Bridge Candidate Label-Holdout Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded offline benchmark that tests whether acquisition selects hidden target-family bridge candidates and thereby makes a cheap downstream probe identifiable.

**Architecture:** Reuse the existing coverage benchmark path under `marginal_value/active_benchmark/` because it already supports fixed K budgets, policy comparison, hygiene rows, oracle diagnostics, reports, and the GCP launcher. Extend the source-family label-holdout episode builder so candidate pools can include explicit bridge groups from one or more held-out target families, then add downstream bridge metrics to the coverage supervised smoke report.

**Tech Stack:** Python dataclasses, NumPy, existing unittest suite, JSON/Markdown reports, CPU-only GCP launcher config.

---

### Task 1: Bridge Episode Controls

**Files:**
- Modify: `marginal_value/active_benchmark/splits.py`
- Modify: `scripts/offline_active_benchmark_from_urls.py`
- Modify: `scripts/offline_coverage_benchmark_from_urls.py`
- Test: `tests/test_offline_active_benchmark.py`

- [x] **Step 1: Write failing tests**
  - Assert `build_source_family_label_holdout_episodes(..., target_families_per_episode=2, target_candidate_groups_per_episode=4)` creates target groups from two families.
  - Assert candidate groups include bridge groups from both target families.
  - Assert support groups exclude all target families.
  - Assert support, candidate, and target groups are disjoint.

- [x] **Step 2: Verify red**
  - Run `python3 -m unittest tests.test_offline_active_benchmark.OfflineActiveBenchmarkTests.test_source_family_label_holdout_can_hold_out_multiple_target_families`
  - Expected: fail because `target_families_per_episode` is not supported.

- [x] **Step 3: Implement minimal episode support**
  - Add `target_families_per_episode` to `build_source_family_label_holdout_episodes`.
  - Allocate target groups and target-candidate bridge groups across held-out families.
  - Add matching CLI args and pass-through helpers.

- [x] **Step 4: Verify green**
  - Re-run the focused test.

### Task 2: Target-Family Oracle

**Files:**
- Modify: `marginal_value/active_benchmark/coverage_runner.py`
- Test: `tests/test_active_coverage_benchmark.py`

- [x] **Step 1: Write failing tests**
  - Add synthetic clips where only one candidate belongs to the held-out target family.
  - Assert `oracle_greedy_target_family_v1` selects the bridge candidate at `K=1`.
  - Assert oracle rows are marked `uses_target_for_selection=True` and non-primary for coverage.

- [x] **Step 2: Verify red**
  - Run `python3 -m unittest tests.test_active_coverage_benchmark.ActiveCoverageBenchmarkTests.test_target_family_oracle_selects_bridge_candidates`
  - Expected: fail because policy is unsupported.

- [x] **Step 3: Implement minimal oracle**
  - Add `oracle_greedy_target_family_v1` to supported coverage policies.
  - Add source-family label config fields to `CoverageBenchmarkConfig`.
  - Rank candidates by newly discovered target-family labels, then quality, then id.

- [x] **Step 4: Verify green**
  - Re-run the focused test.

### Task 3: Bridge Metrics And Reports

**Files:**
- Modify: `marginal_value/active_benchmark/downstream_coverage_smoke.py`
- Test: `tests/test_downstream_supervised_smoke.py`

- [x] **Step 1: Write failing tests**
  - Assert downstream coverage rows include `candidate_bridge_count`, `selected_bridge_count`, `target_family_count`, and `target_family_discovery_rate`.
  - Assert budget `0` has discovery `0.0`.
  - Assert selecting a target-family bridge makes known-target fraction nonzero and discovery positive.
  - Assert report decision includes benchmark-validity gates and compares top policy against `quality_stratified_random_v1`.

- [x] **Step 2: Verify red**
  - Run `python3 -m unittest tests.test_downstream_supervised_smoke.DownstreamSupervisedSmokeTests.test_bridge_metrics_measure_target_family_discovery`
  - Expected: fail because bridge metrics are absent.

- [x] **Step 3: Implement minimal metrics**
  - Infer labels once using existing source-family label helper.
  - For each episode/policy/budget, compute target labels, candidate bridge labels, selected bridge labels, discovery rate, and known-target fraction.
  - Add summary means and Markdown rows for discovery.

- [x] **Step 4: Verify green**
  - Re-run the focused test.

### Task 4: Bounded GCP Pilot Config

**Files:**
- Create: `configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_pilot.json`
- Modify: `scripts/launch_coverage_benchmark_gcp.py`
- Test: `tests/test_coverage_benchmark_from_urls.py`

- [x] **Step 1: Write failing tests**
  - Assert config is CPU-only, one seed, no TS2Vec retraining, no downstream training.
  - Assert source-family label-holdout is used with 36 candidate clips via 12 groups and 3 clips per group.
  - Assert `oracle_greedy_target_family_v1`, `quality_stratified_random_v1`, and `ts2vec_kcenter_v1` are present.
  - Assert acceptance gates include oracle discovery, known-target movement, BA movement, and no leakage.

- [x] **Step 2: Verify red**
  - Run `python3 -m unittest tests.test_coverage_benchmark_from_urls.CoverageBenchmarkFromUrlsTests.test_bridge_pilot_config_matches_advisor_gate`
  - Expected: fail because config does not exist.

- [x] **Step 3: Implement config and launcher pass-through**
  - Add target-family pass-through flags to startup script.
  - Create the one-seed, CPU-only pilot config with bounded rows/groups.

- [x] **Step 4: Verify green**
  - Re-run focused config and launcher tests.

### Task 5: Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused suite**
  - Run `python3 -m unittest tests.test_offline_active_benchmark tests.test_active_coverage_benchmark tests.test_downstream_supervised_smoke tests.test_coverage_benchmark_from_urls`

- [x] **Step 2: Compile touched scripts/modules**
  - Run `python3 -m py_compile marginal_value/active_benchmark/splits.py marginal_value/active_benchmark/coverage_runner.py marginal_value/active_benchmark/downstream_coverage_smoke.py scripts/offline_active_benchmark_from_urls.py scripts/offline_coverage_benchmark_from_urls.py scripts/launch_coverage_benchmark_gcp.py`

- [ ] **Step 3: Report the exact pilot command**
  - Provide the dry-run command first, then the launch command only after the user wants to spend GCP credits.

