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

## Implemented Logging Contract

The shard builder writes both Modal logs and durable artifact logs. A smoke run
must be reviewed before starting the full build.

Durable artifact paths:

```text
/artifacts/active/full_support_shards/window_stats_v1/full_support_shard_progress_{mode}.jsonl
/artifacts/active/full_support_shards/window_stats_v1/full_support_shard_report_{mode}.json
/artifacts/active/full_support_shards/window_stats_v1/full_support_shards_{mode}.json
```

The progress JSONL includes:

- `selection_ready`: selected clip count, shard count, representations, quality
  metadata coverage;
- `quality_metadata_incomplete`: emitted early if any selected clip is missing
  quality fields;
- `shard_start`: shard index and shard size;
- `embedding_progress`: per-shard feature progress inherited from the embedding
  cache layer;
- `shard_write`: completed clips, completed fraction, throughput, ETA, and
  shard path;
- `report_write`: running report refresh after each shard;
- `done`: final manifest/report/progress paths.

The Modal wrapper blocks for smoke runs only. Full runs are spawned/detached
after smoke, so the local client can disconnect while progress continues to be
written to the artifact volume.

Smoke run on 2026-05-02:

```text
Modal app: ap-y1NtTOuVrU6NUOOoWmlz0e
Mode: smoke
Selected clips: 256
Shards: 1
Elapsed: 71.5s
Throughput: 3.58 clips/s
Quality metadata coverage: 0 / 256 clips with all fields
Full build launched: no
```

The zero quality-metadata coverage is intentional evidence from the smoke, not
a silent pass. Do not treat the full shard output as quality-complete until a
real `data.quality_metadata` source is attached or a full quality-metadata
build is run.

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
