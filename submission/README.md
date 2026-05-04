# Active Learning Challenge Submission

## Primary File

Submit this file first:

```text
ranked_new_clips.csv
```

It contains exactly 2,000 ranked clips with columns:

```text
rank,score,new_worker_id
```

`rank=1` is the clip selected as most valuable/new-information-rich by the
selector.

Backup ID variants are included:

```text
ranked_new_clips_new_worker_id.csv
ranked_new_clips_worker_id.csv
```

`ranked_new_clips.csv` and `ranked_new_clips_new_worker_id.csv` are identical.
Use `ranked_new_clips_worker_id.csv` only if the evaluator expects the internal
manifest-hash ID instead of the public `sampleNNNNNN`-style ID.

## What This Submission Contains

This package contains both:

```text
1. the ranked output for the public 2,000 new clips;
2. the algorithm/run instructions for scoring a held-out new manifest.
```

The public CSV is the easiest way to evaluate the clips already listed on the
challenge page. The held-out runner is the algorithm path if the evaluator wants
to run the same method on another new-candidate manifest.

## How Eddy Can Test The 2,000 Provided Clips

For the public 2,000 new clips, Eddy does not need to run our code.

The evaluator can:

1. Load `ranked_new_clips.csv`.
2. Join rows by `new_worker_id` to the same 2,000 public candidate clips.
3. Evaluate any top-K prefix, for example top 10, 50, 100, or 200.
4. Measure downstream gain, hidden coverage, redundancy, or any internal utility
   metric using that ordered list.

This is the intended CSV-only submission path.

## How Eddy Can Test The Algorithm

If the held-out test uses the same public 200,000-clip old corpus and a
different hidden new-candidate manifest, use:

```text
RUN_ON_HELDOUT_NEW.md
```

That path runs:

```text
old manifest + hidden new manifest -> ranked CSV
```

It reuses the exact 200,000-clip old-support TS2Vec cache, computes
TS2Vec/window embeddings for the hidden new clips, rebuilds exact window support,
then emits the same CSV schema.

If the evaluator changes the old corpus itself, exact TS2Vec support for that
new old corpus must be recomputed. The same runner also emits an exact-window
fallback that does not depend on old-support TS2Vec embeddings.

## Method Summary

The submitted ranking is:

```text
artifact-gate exact-full TS2Vec / exact-window blend
```

It combines:

- TS2Vec novelty against all 200,000 old-support clips;
- `window_mean_std_pool` novelty against all 200,000 old-support clips;
- quality and physical-validity gates;
- k-center-style redundancy control within the new batch;
- artifact-aware trace reranking.

The final package ranks:

```text
old TS2Vec support: 200000 / 200000 clips
old window support: 200000 / 200000 clips
new candidates: 2000 / 2000 clips
```

## Main Evidence

The exact full-support ranking report is:

```text
exact_selector_report.json
```

It records:

```text
n_left_support: 200000
n_right_support: 200000
n_query: 2000
ranking_mode: full_left_exact_window_right
```

The artifact-gate hygiene report is:

```text
selector_report.json
```

Top-50 artifact-gated hygiene:

```text
quality_fail_rate: 0.000
physical_fail_rate: 0.000
spike_fail_rate: 0.000
trace_artifact_fail_rate: 0.000
unique_clusters: 50
```

## Files

```text
ranked_new_clips.csv                  primary submission
ranked_new_clips_new_worker_id.csv    explicit new_worker_id duplicate
ranked_new_clips_worker_id.csv        backup internal worker_id variant
diagnostics.csv                       full per-clip diagnostic table
selector_config.json                  method/config claim
feature_schema.json                   output schema
selector_report.json                  artifact-gate report
exact_selector_report.json            exact full-support ranking report
validation_report.json                internal active-loop validation report
final_package_report.json             package readiness report
METHODS.md                            short method note
RUN_ON_HELDOUT_NEW.md                 how to rerun for hidden new clips
SUBMISSION_MANIFEST.json              file sizes and checksums
```

## What Not To Claim

Do not describe this as:

- validated clean TS2Vec active learning;
- a learned active-acquisition policy;
- semantic workflow discovery.

The honest claim is:

```text
Exact full-support TS2Vec/window geometric acquisition selector with
artifact-aware trace rerank.
```
