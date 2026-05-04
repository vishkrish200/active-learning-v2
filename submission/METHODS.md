# Method Note

We rank newly arrived IMU clips with an exact full-support TS2Vec / exact-window
blended k-center selector with artifact-aware trace rerank.

Each new candidate clip is embedded in two views:

- a TS2Vec-style temporal encoder representation;
- a handcrafted `window_mean_std_pool` representation.

For each view, candidate novelty is measured against the old corpus. In the
submitted public-candidate run, both old-support views cover all 200,000 old
clips.

The two novelty scores are normalized and blended with `alpha = 0.5`. The
selector then applies quality and physical-validity gates:

```text
quality_score >= 0.85
stationary_fraction <= 0.90
max_abs_value <= 60.0
```

The final ordering uses k-center-style redundancy control so top-ranked clips
are both under-covered by the old corpus and nonredundant within the new batch.
After ranking, an artifact-aware trace rerank demotes likely sensor-artifact
traces below plausible-motion traces.

This is a deterministic geometric acquisition selector on top of a trained
TS2Vec representation. It is not a claim of validated clean TS2Vec active
learning, and it is not a promoted learned ranker.

## Validation Summary

The exact full-support run used:

```text
TS2Vec old support: 200000 / 200000
window-stat old support: 200000 / 200000
new candidates: 2000 / 2000
```

The exact-full output stayed highly consistent with the previous promoted
partial-support package:

```text
top10_overlap: 0.900
top50_overlap: 0.940
top100_overlap: 0.970
rank_correlation: 0.9572
```

Artifact-gated top-50 hygiene:

```text
quality_fail_rate: 0.000
physical_fail_rate: 0.000
spike_fail_rate: 0.000
trace_artifact_fail_rate: 0.000
unique_clusters: 50
```

The external held-out evaluator remains the real test.
