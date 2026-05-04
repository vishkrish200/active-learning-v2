# Submission Email Draft

Subject:

```text
Active learning challenge - IMU novelty ranking submission
```

Body:

```text
Hi Eddy,

I'm sending my submission for the active-learning IMU challenge.

The attached package contains the ranked output for the public 2,000 new IMU
clips. The primary file is ranked_new_clips.csv, with columns:

rank,score,new_worker_id

The method is an exact full-support TS2Vec/window geometric acquisition
selector with quality gates, k-center redundancy control, and artifact-aware
trace reranking. It compares each new clip against all 200,000 old-support clips
in both a trained TS2Vec representation and a handcrafted window-stat
representation.

The repo also includes a held-out-new runner for the likely evaluation mode:
same public 200k old corpus plus a different hidden new-candidate manifest. See
RUN_ON_HELDOUT_NEW.md for the exact command flow. If the old corpus itself is
changed, the exact TS2Vec old-support cache needs to be recomputed, and the
runner also emits an exact-window fallback.

I frame this as a novelty-ranking / marginal-data-value acquisition selector,
not as a downstream retraining proof.

Best,
Vishnu
```
