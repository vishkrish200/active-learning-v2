# Full-Support Data-Shard Plan

Date: 2026-05-02

## Decision

Do not immediately start a full-support TS2Vec rerun using the current small-file
Modal-volume layout.

The evidence says the next real scientific improvement is a data-layer change,
not another H100 job:

- fixed-crop TS2Vec did not clearly beat the current checkpoint on the
  8-episode medium eval;
- 25k support sampling has broad worker coverage, but top-K rankings move under
  different support seeds;
- prior full-support H100 attempts were dominated by small-file IO and cache
  bookkeeping rather than GPU math.

## Goal

Make exact full-support scoring feasible by packing the old and new clips into
large sequential shard files.

The desired end state:

```text
old_support_000.npz
old_support_001.npz
...
new_candidates.npz
```

Each shard should contain:

```text
sample_id
worker_id
source_group_id
split
url
quality_score
stationary_fraction
max_abs_value
window_mean_std_pool
raw_shape_stats
normalized_imu or fixed-length windows
```

## Why This Matters

H100 accelerates TS2Vec forward passes. It does not fix:

- opening hundreds of thousands of tiny JSONL files;
- parsing JSON line by line;
- loading many tiny `.npz` feature files;
- repeatedly rebuilding cache keys and file manifests.

Large shards turn the bottleneck into sequential reads and batched tensor
inference, which is the shape H100 can actually accelerate.

## Proposed Build Order

1. Build a shard writer for old support and new candidates.
2. Store metadata and cheap features in the same shard as the clip IDs.
3. Store either normalized IMU arrays or fixed windows suitable for TS2Vec
   inference.
4. Add a shard reader that yields batches without per-clip file opens.
5. Re-run TS2Vec full-support embedding from shards.
6. Re-run final blend with exact 200k support for both TS2Vec and window views.
7. Compare exact full-support ranking against the 3-seed budgeted stability
   result.

## Acceptance Gates

Do not promote a full-support rerun just because it is more expensive. Promote
only if it passes these gates:

| Gate | Target |
| --- | --- |
| Full old-support coverage | 200,000 / 200,000 clips |
| New-candidate coverage | 2,000 / 2,000 clips |
| Top-K hygiene | quality fail 0.00, physical fail 0.00 at K=10/K=50 |
| Active-loop eval | no regression vs current blend at K=10 and K=50 |
| Stability | better top-50/top-100 agreement than sampled-support runs |

## Near-Term Alternative

Before implementing shards, a cheaper improvement is a consensus selector:

- run the budgeted final selector over 3-5 seeded 25k support samples;
- aggregate by mean rank or Borda score;
- keep hard hygiene gates;
- emit a consensus final ranking.

This directly addresses the observed top-K sampling sensitivity and costs much
less than a full data-layer rewrite.
