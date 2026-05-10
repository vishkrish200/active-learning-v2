# Coverage Benchmark Decision Report

## Decision

- downstream training: `hold`
- large training: `hold`
- bounded downstream canary: `hold`
- tiny frozen supervised probe: `pending-audit`
- TS2Vec retraining: `no`
- next CPU gate: `oracle-plus-episode-bootstrap`
- read: keep this in CPU/offline benchmark mode until oracle, confidence, direct-control, and stability gates pass.

## Run Shape

- reports: `3`
- independent final episodes: `18`
- final budget: `4`
- gain records: `324`
- candidate pool size: `36-36 clips, median 36.0`

## Final Leaderboard

| rank | policy | mean final gain | median | CI95 low | CI95 high | episodes | target-used |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `oracle_greedy_target_family_v1` | 0.1299 | 0.1217 | 0.0796 | 0.1852 | 18 | Y |
| 2 | `window_kcenter_v1` | 0.1240 | 0.0814 | 0.0753 | 0.1776 | 18 | N |
| 3 | `submitted_full_replay_v1` | 0.1192 | 0.0870 | 0.0713 | 0.1682 | 18 | N |
| 4 | `ts2vec_kcenter_v1` | 0.1119 | 0.1017 | 0.0612 | 0.1840 | 18 | N |
| 5 | `quality_stratified_random_v1` | 0.0628 | 0.0400 | 0.0231 | 0.1191 | 18 | N |
| 6 | `quality_only_v1` | 0.0445 | 0.0078 | 0.0084 | 0.0993 | 18 | N |

## Expanded Training Unlock Criteria

- scope: Unlock only a bounded frozen-feature downstream canary; keep TS2Vec retraining and large neural training held.
- focal policy: `ts2vec_kcenter_v1`
- baseline policy: `quality_stratified_random_v1`
- per-pool episodes >= `80`
- candidate clips max >= `72`
- mean delta vs baseline >= `0.0400`
- CI95 low vs baseline >= `0.0250`
- win fraction vs baseline >= `0.7000`
- replay-null p-value <= `0.0100`
- window-kcenter mean delta >= `0.0150` with CI95 low > `0.0000`
- submitted-full replay CI95 low > `0.0000`

## Paired Delta Vs Baseline

| policy | baseline | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| `oracle_greedy_target_family_v1` | `quality_stratified_random_v1` | 0.0672 | 0.0538 | 0.0381 | 0.0988 | 0.6667 | 18 |
| `window_kcenter_v1` | `quality_stratified_random_v1` | 0.0612 | 0.0338 | 0.0231 | 0.1006 | 0.5556 | 18 |
| `submitted_full_replay_v1` | `quality_stratified_random_v1` | 0.0564 | 0.0415 | 0.0233 | 0.0945 | 0.7222 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | 0.0491 | 0.0541 | 0.0281 | 0.0727 | 0.7778 | 18 |
| `quality_only_v1` | `quality_stratified_random_v1` | -0.0183 | 0.0000 | -0.0415 | -0.0015 | 0.3333 | 18 |

## Direct Paired Control Comparisons

| policy | comparator | mean delta | median delta | CI95 low | CI95 high | win frac | loss frac | tie frac | paired episodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ts2vec_kcenter_v1` | `quality_only_v1` | 0.0674 | 0.0740 | 0.0422 | 0.0946 | 0.8333 | 0.1667 | 0.0000 | 18 |
| `window_kcenter_v1` | `quality_stratified_random_v1` | 0.0612 | 0.0338 | 0.0240 | 0.1029 | 0.5556 | 0.3333 | 0.1111 | 18 |
| `submitted_full_replay_v1` | `quality_stratified_random_v1` | 0.0564 | 0.0415 | 0.0231 | 0.0906 | 0.7222 | 0.2222 | 0.0556 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | 0.0491 | 0.0541 | 0.0265 | 0.0719 | 0.7778 | 0.2222 | 0.0000 | 18 |
| `ts2vec_kcenter_v1` | `submitted_full_replay_v1` | -0.0073 | -0.0022 | -0.0473 | 0.0321 | 0.3889 | 0.6111 | 0.0000 | 18 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | -0.0121 | -0.0006 | -0.0542 | 0.0294 | 0.4444 | 0.5556 | 0.0000 | 18 |

## Deprecated Raw Oracle Capture

Raw oracle capture is kept only as a debug metric. It is a ratio over episode-level oracle headroom and is unstable when the denominator is small or negative.

| policy | mean capture | median capture | CI95 low | CI95 high | episodes |
|---|---:|---:|---:|---:|---:|
| `ts2vec_kcenter_v1` | 4.2905 | 0.3975 | 0.1306 | 12.2487 | 12 |
| `quality_only_v1` | 0.1075 | 0.0000 | -0.3909 | 0.7797 | 12 |
| `window_kcenter_v1` | -1.6619 | 0.2843 | -6.2804 | 0.9410 | 12 |
| `submitted_full_replay_v1` | -2.3265 | 0.1070 | -8.0908 | 0.8248 | 12 |

## Stable Oracle Diagnostics

- oracle policy: `oracle_greedy_target_family_v1`
- baseline source: `quality_stratified_random_v1`
- headroom epsilon: `0.0100`
- oracle headroom mean: `0.0672`
- oracle headroom median: `0.0538`
- oracle headroom CI95: `0.0342` to `0.1019`
- oracle headroom positive fraction: `0.6111`
- oracle headroom near-zero fraction: `0.3333`

| policy | oracle gap mean | oracle gap CI95 low | positive-headroom capture median | bounded capture mean | negative capture frac | overshoot frac | positive-headroom episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `window_kcenter_v1` | 0.0060 | -0.0395 | 0.5685 | 0.4465 | 0.2727 | 0.2727 | 11 |
| `submitted_full_replay_v1` | 0.0108 | -0.0305 | 0.1712 | 0.4401 | 0.2727 | 0.2727 | 11 |
| `ts2vec_kcenter_v1` | 0.0181 | -0.0312 | 0.3158 | 0.3340 | 0.3636 | 0.0909 | 11 |
| `quality_only_v1` | 0.0855 | 0.0441 | 0.0000 | 0.1128 | 0.4545 | 0.0000 | 11 |

## Oracle Diagnostics

- The exact coverage oracle maximizes the exact final-budget coverage objective; smaller budget prefixes are deterministic ordering diagnostics.
- The target-family oracle is a discovery diagnostic, not an exact ceiling for reported coverage gain.
- configured oracle policy: `oracle_greedy_target_family_v1`
- exact coverage oracle: `None`
- target-family discovery oracle: `oracle_greedy_target_family_v1`
- top deployable policy: `window_kcenter_v1`

## Acquisition Stability

- focal policy: `ts2vec_kcenter_v1`
- baseline source: `quality_stratified_random_v1`
- positive seed fraction: `1.0000`
- positive fold fraction: `1.0000`
- min leave-one-seed-out delta: `0.0361`
- min leave-one-fold-out delta: `0.0424`
- largest single episode contribution fraction: `0.1813`

## Selected Set Audit

| policy | selected | invalid rate | duplicate frac | mean quality | max artifact | unique source groups | top source frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oracle_greedy_target_family_v1` | 72 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 27 | 0.1389 |
| `quality_only_v1` | 72 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 20 | 0.1528 |
| `quality_stratified_random_v1` | 72 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 28 | 0.0972 |
| `submitted_full_replay_v1` | 72 | 0.0000 | 0.0000 | 0.9926 | 0.0178 | 35 | 0.0833 |
| `ts2vec_kcenter_v1` | 72 | 0.0000 | 0.0000 | 0.9929 | 0.0178 | 34 | 0.0972 |
| `window_kcenter_v1` | 72 | 0.0000 | 0.0000 | 0.9930 | 0.0126 | 35 | 0.0972 |

## Downstream Bridge Proxy

This is a source-family pseudo-label frozen-feature proxy on predeclared heads, not real challenge-label downstream proof.

- downstream training: `hold_large_training`
- bounded frozen canary: `hold`
- large training: `hold`
- TS2Vec retraining: `no`
- read: bridge proxy is aggregated with episode-level paired intervals; use it as an identifiability gate, not real challenge-label downstream proof.
- focal policy: `ts2vec_kcenter_v1`
- top deployable policy by balanced accuracy: `ts2vec_kcenter_v1`
- independent final episodes: `18`
- final budget: `4`

### Downstream Final Means

| policy | episodes | discovery | known target frac | balanced accuracy gain | NLL reduction |
|---|---:|---:|---:|---:|---:|
| `oracle_greedy_target_family_v1` | 18 | 1.0000 | 1.0000 | 0.3179 | 22.6409 |
| `quality_stratified_random_v1` | 18 | 0.5556 | 0.5556 | 0.1883 | 12.4478 |
| `ts2vec_kcenter_v1` | 18 | 0.6944 | 0.6944 | 0.1605 | 14.7057 |
| `submitted_full_replay_v1` | 18 | 0.7778 | 0.7778 | 0.1520 | 15.0257 |
| `quality_only_v1` | 18 | 0.4167 | 0.4167 | 0.1451 | 9.4572 |
| `window_kcenter_v1` | 18 | 0.6944 | 0.6944 | 0.1404 | 13.6186 |

### Paired Downstream Deltas Vs Baseline

| policy | metric | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| `oracle_greedy_target_family_v1` | `target_family_discovery_rate` | 0.4444 | 0.5000 | 0.3056 | 0.6111 | 0.7222 | 18 |
| `oracle_greedy_target_family_v1` | `after_known_target_fraction` | 0.4444 | 0.5000 | 0.2778 | 0.5833 | 0.7222 | 18 |
| `oracle_greedy_target_family_v1` | `balanced_accuracy_gain` | 0.1296 | 0.0764 | 0.0687 | 0.2045 | 0.8889 | 18 |
| `oracle_greedy_target_family_v1` | `nll_reduction` | 10.1931 | 10.6521 | 6.7842 | 13.5942 | 0.9444 | 18 |
| `quality_only_v1` | `target_family_discovery_rate` | -0.1389 | 0.0000 | -0.3056 | 0.0000 | 0.1111 | 18 |
| `quality_only_v1` | `after_known_target_fraction` | -0.1389 | 0.0000 | -0.2778 | 0.0000 | 0.1111 | 18 |
| `quality_only_v1` | `balanced_accuracy_gain` | -0.0432 | -0.0625 | -0.1011 | 0.0209 | 0.2778 | 18 |
| `quality_only_v1` | `nll_reduction` | -2.9906 | -2.4588 | -6.1646 | 0.4705 | 0.3333 | 18 |
| `submitted_full_replay_v1` | `target_family_discovery_rate` | 0.2222 | 0.5000 | 0.0833 | 0.3611 | 0.5556 | 18 |
| `submitted_full_replay_v1` | `after_known_target_fraction` | 0.2222 | 0.5000 | 0.0556 | 0.3611 | 0.5556 | 18 |
| `submitted_full_replay_v1` | `balanced_accuracy_gain` | -0.0363 | -0.0347 | -0.0810 | 0.0054 | 0.2778 | 18 |
| `submitted_full_replay_v1` | `nll_reduction` | 2.5779 | 6.3194 | -0.2490 | 5.4062 | 0.6111 | 18 |
| `ts2vec_kcenter_v1` | `target_family_discovery_rate` | 0.1389 | 0.0000 | -0.0556 | 0.3056 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `after_known_target_fraction` | 0.1389 | 0.0000 | -0.0278 | 0.2778 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `balanced_accuracy_gain` | -0.0278 | 0.0000 | -0.0942 | 0.0394 | 0.3889 | 18 |
| `ts2vec_kcenter_v1` | `nll_reduction` | 2.2579 | 2.2815 | -1.8564 | 6.2477 | 0.5556 | 18 |
| `window_kcenter_v1` | `target_family_discovery_rate` | 0.1389 | 0.0000 | -0.0278 | 0.3056 | 0.4444 | 18 |
| `window_kcenter_v1` | `after_known_target_fraction` | 0.1389 | 0.0000 | -0.0278 | 0.3056 | 0.4444 | 18 |
| `window_kcenter_v1` | `balanced_accuracy_gain` | -0.0478 | -0.0139 | -0.1057 | 0.0054 | 0.2778 | 18 |
| `window_kcenter_v1` | `nll_reduction` | 1.1708 | 0.9619 | -2.2121 | 4.5508 | 0.5000 | 18 |

### Direct Downstream Controls

| policy | comparator | metric | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `ts2vec_kcenter_v1` | `quality_only_v1` | `target_family_discovery_rate` | 0.2778 | 0.0000 | 0.1389 | 0.4444 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `quality_only_v1` | `after_known_target_fraction` | 0.2778 | 0.0000 | 0.1389 | 0.4444 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `quality_only_v1` | `balanced_accuracy_gain` | 0.0154 | 0.0139 | -0.0579 | 0.0803 | 0.5000 | 18 |
| `ts2vec_kcenter_v1` | `quality_only_v1` | `nll_reduction` | 5.2485 | 2.4223 | 1.9573 | 8.7817 | 0.6667 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | `target_family_discovery_rate` | 0.1389 | 0.0000 | -0.0278 | 0.3056 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | `after_known_target_fraction` | 0.1389 | 0.0000 | -0.0278 | 0.3056 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | `balanced_accuracy_gain` | -0.0278 | 0.0000 | -0.1003 | 0.0394 | 0.3889 | 18 |
| `ts2vec_kcenter_v1` | `quality_stratified_random_v1` | `nll_reduction` | 2.2579 | 2.2815 | -1.7245 | 6.0753 | 0.5556 | 18 |
| `ts2vec_kcenter_v1` | `submitted_full_replay_v1` | `target_family_discovery_rate` | -0.0833 | 0.0000 | -0.2500 | 0.0556 | 0.1667 | 18 |
| `ts2vec_kcenter_v1` | `submitted_full_replay_v1` | `after_known_target_fraction` | -0.0833 | 0.0000 | -0.2500 | 0.0833 | 0.1667 | 18 |
| `ts2vec_kcenter_v1` | `submitted_full_replay_v1` | `balanced_accuracy_gain` | 0.0085 | 0.0000 | -0.0486 | 0.0640 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `submitted_full_replay_v1` | `nll_reduction` | -0.3200 | -0.0089 | -3.5398 | 3.0525 | 0.5000 | 18 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | `target_family_discovery_rate` | 0.0000 | 0.0000 | -0.1944 | 0.1944 | 0.3333 | 18 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | `after_known_target_fraction` | 0.0000 | 0.0000 | -0.1667 | 0.1944 | 0.3333 | 18 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | `balanced_accuracy_gain` | 0.0201 | 0.0000 | -0.0633 | 0.1003 | 0.4444 | 18 |
| `ts2vec_kcenter_v1` | `window_kcenter_v1` | `nll_reduction` | 1.0871 | 1.6396 | -2.7601 | 4.9357 | 0.6111 | 18 |

## Budget Curves

| budget | policy | mean gain | episodes |
|---:|---|---:|---:|
| 1 | `oracle_greedy_target_family_v1` | 0.1150 | 18 |
| 1 | `window_kcenter_v1` | 0.0788 | 18 |
| 1 | `ts2vec_kcenter_v1` | 0.0606 | 18 |
| 1 | `submitted_full_replay_v1` | 0.0588 | 18 |
| 1 | `quality_only_v1` | 0.0230 | 18 |
| 1 | `quality_stratified_random_v1` | 0.0155 | 18 |
| 2 | `oracle_greedy_target_family_v1` | 0.1263 | 18 |
| 2 | `window_kcenter_v1` | 0.1007 | 18 |
| 2 | `submitted_full_replay_v1` | 0.0839 | 18 |
| 2 | `ts2vec_kcenter_v1` | 0.0811 | 18 |
| 2 | `quality_stratified_random_v1` | 0.0421 | 18 |
| 2 | `quality_only_v1` | 0.0300 | 18 |
| 4 | `oracle_greedy_target_family_v1` | 0.1299 | 18 |
| 4 | `window_kcenter_v1` | 0.1240 | 18 |
| 4 | `submitted_full_replay_v1` | 0.1192 | 18 |
| 4 | `ts2vec_kcenter_v1` | 0.1119 | 18 |
| 4 | `quality_stratified_random_v1` | 0.0628 | 18 |
| 4 | `quality_only_v1` | 0.0445 | 18 |

## Gates

| gate | status | read |
|---|---|---|
| episode_count | warn | 18 independent final episodes; require at least 20 before downstream spend. |
| large_training_baseline_delta_ci | warn | window_kcenter_v1 vs quality_stratified_random_v1 CI low is 0.0231; require > 0.04 before large downstream spend. |
| oracle_present | pass | oracle ceiling is present. |
| oracle_headroom_sanity | warn | oracle_greedy_target_family_v1 headroom median is 0.0538 with positive fraction 0.6111; require positive median and >= 0.70 positive fraction. |
| tiny_probe_random_delta | warn | window_kcenter_v1 vs quality_stratified_random_v1: CI low 0.0231, median 0.0338, win fraction 0.5556; tiny-probe threshold is CI low >= 0.02, median >= 0.025, win fraction >= 0.70. |
| tiny_probe_replay_null | warn | window_kcenter_v1 replay percentile , +1 p-value , replay-mean CI low . |
| tiny_probe_direct_controls | warn | ts2vec_kcenter_v1 must beat window_kcenter_v1 and submitted_full_replay_v1 by paired mean and median. |
| tiny_probe_stability | pass | ts2vec_kcenter_v1 stability: positive seed fraction 1.0000, positive fold fraction 1.0000, min leave-one-seed 0.0361, min leave-one-fold 0.0424. |
| deployable_top_policy | pass | top non-oracle policy is window_kcenter_v1. |

## Notes

- Confidence intervals are episode-level paired bootstraps, not row-level bootstraps.
- Oracle rows are non-deployable ceilings because they inspect the blind target set.
- Gains use primary eval views but still exclude same-feature selector/eval shortcuts.
