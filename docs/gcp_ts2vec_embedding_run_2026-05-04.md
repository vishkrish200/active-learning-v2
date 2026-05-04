# GCP TS2Vec Full-Support Embedding Setup - 2026-05-04

## Purpose

Prepare a Google Cloud path for the scientific upgrade from partial old-support
TS2Vec to full old-support TS2Vec embeddings.

The goal is not to retrain TS2Vec yet. The immediate goal is to embed the full
old support manifest with the current promoted checkpoint, write resumable
shards to GCS, and later use those shards as exact TS2Vec support.

## GCP Project

```text
project_id: project-afdaec96-592b-4bd3-9ee
project_number: 802636843791
billing_account: billingAccounts/016CDF-499943-8D4002
budget: GPU Spend, INR 10000/month, alerts at 50%, 90%, 100%
bucket: gs://active-learning-v2-802636843791
bucket_location: us-central1
```

Budget alerts are not hard spending caps. GPU jobs still need explicit runtime
limits, checkpointing, and manual cleanup checks.

## Quota Status

The first L4 smoke test failed because global GPU quota was zero:

```text
Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```

Quota preferences were then submitted:

| Quota | Scope | Preferred | Granted after request |
|---|---|---:|---:|
| `GPUS-ALL-REGIONS-per-project` | global | 1 | 1 |
| `CPUS-ALL-REGIONS-per-project` | global | 32 | 32 |
| `GPUS-PER-GPU-FAMILY-per-project-region` | `NVIDIA_H100`, `us-central1` | 1 | 0 |
| `PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region` | `us-central1` | 1 | 1 |

This means:

- L4 is usable now.
- Spot/preemptible H100 quota is granted, but H100 capacity is currently stocked
  out in `us-central1-a`, `us-central1-b`, and `us-central1-c`.
- Standard H100 quota is not granted yet.

## Smoke Tests

L4 smoke passed on `g2-standard-4` in `us-central1-a`.

```text
GPU: NVIDIA L4
memory: 23034 MiB
driver: 580.126.20
```

The smoke VM was deleted after the check. No VM was left running.

H100 spot smoke was attempted on `a3-highgpu-1g` in `us-central1-a/b/c`.
All attempts failed with resource pool stockout. No H100 VM was created.

## Staged Artifacts

The current promoted checkpoint was exported from the Modal artifact volume and
uploaded to GCS:

```text
gs://active-learning-v2-802636843791/checkpoints/ts2vec_candidate_eval/ts2vec_best.pt
sha256: 1776e2adc1cc473e8f5650d95860475d9f2b869999703e49fa3a575225ec9d59
```

The committed source archive and selected configs were also uploaded:

```text
gs://active-learning-v2-802636843791/source/active-learning-v2-codex-ts2vec-pipeline-de5629b6d2c0.tar.gz
gs://active-learning-v2-802636843791/configs/active_embedding_precompute_ts2vec_full_new.json
gs://active-learning-v2-802636843791/configs/active_exact_window_blend_rank.json
gs://active-learning-v2-802636843791/configs/final_package_artifact_gate.json
```

After adding the GCP runner, a fresh source archive was uploaded from commit
`3eb4ce57b0a0`:

```text
gs://active-learning-v2-802636843791/source/active-learning-v2-codex-ts2vec-pipeline-3eb4ce57b0a0.tar.gz
```

A later smoke run exposed that old-corpus clip filenames repeat across workers
(`worker00001/clip001.txt`, `worker00002/clip001.txt`, ...). The GCP runner now
uses the repo's canonical SHA-256 manifest URL sample IDs instead of filename
stems. The patched source archive used for the passing smoke/probe is:

```text
gs://active-learning-v2-802636843791/source/active-learning-v2-codex-ts2vec-pipeline-sampleidfix.tar.gz
```

## GCP-Native Embedding Runner

Added:

```text
marginal_value/gcp/ts2vec_embed_manifest.py
```

The runner reads a local, HTTPS, or `gs://` manifest of raw JSONL clips, embeds
each clip with a local TS2Vec checkpoint, writes `shard_*.npz` files with
`rep__ts2vec`, and optionally uploads each completed shard plus a manifest to
GCS.

Example smoke command on a GPU VM:

```bash
python -m marginal_value.gcp.ts2vec_embed_manifest \
  --manifest https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain_urls.txt \
  --checkpoint /workspace/checkpoints/ts2vec_candidate_eval/ts2vec_best.pt \
  --output-dir /workspace/out/ts2vec_old_support_smoke \
  --gcs-output gs://active-learning-v2-802636843791/embeddings/ts2vec_old_support_smoke \
  --device cuda \
  --limit 128 \
  --shard-size 64 \
  --encode-batch-size 32 \
  --load-workers 16
```

Example full old-support command:

```bash
python -m marginal_value.gcp.ts2vec_embed_manifest \
  --manifest https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain_urls.txt \
  --checkpoint /workspace/checkpoints/ts2vec_candidate_eval/ts2vec_best.pt \
  --output-dir /workspace/out/ts2vec_old_support_full \
  --gcs-output gs://active-learning-v2-802636843791/embeddings/ts2vec_old_support_full \
  --device cuda \
  --shard-size 1024 \
  --encode-batch-size 64 \
  --load-workers 16
```

## Runtime Fix

The stock GCP Deep Learning VM PyTorch image was not usable for the TS2Vec
forward pass. It imported CUDA successfully, but failed during real model
execution with:

```text
Invalid handle. Cannot load symbol cublasLtCreate
```

The working path is to keep the GCP NVIDIA driver image, but create a clean
Python virtualenv and install a stable PyTorch CUDA wheel:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
```

Verified on L4:

```text
torch: 2.5.1+cu124
torch.version.cuda: 12.4
torch.cuda.is_available: True
GPU matmul: OK
```

## Completed GCP Probes

### 16-clip smoke

```text
output: gs://active-learning-v2-802636843791/embeddings/ts2vec_old_support_stabletorch_smoke
status: passed
n_clips: 16
n_shards: 2
```

### 128-clip full-length smoke

```text
output: gs://active-learning-v2-802636843791/embeddings/ts2vec_old_support_stabletorch_smoke_128
log: gs://active-learning-v2-802636843791/logs/ts2vec_stabletorch_smoke_128_fixed_ids.log
status: passed
n_clips: 128
n_shards: 2
sample_ids_unique: 128 / 128
embedding_shape_per_shard: (64, 320)
finite_embeddings: true
elapsed_seconds: ~15
```

### 5,000-clip throughput probe

```text
output: gs://active-learning-v2-802636843791/embeddings/ts2vec_old_support_stabletorch_probe_5k
log: gs://active-learning-v2-802636843791/logs/ts2vec_stabletorch_probe_5k.log
status: passed
n_clips: 5000
n_shards: 10
sample_ids_unique: 5000 / 5000
embedding_dim: 320
finite_embeddings: true
elapsed_seconds: 301.796
```

The 5k probe ran at roughly 1,000 full-length clips per minute on one L4 after
the Python environment was already built.

## Full-Run Estimate

At the measured 5k-probe throughput, embedding all 200,000 old-support clips on
one L4 is approximately:

```text
200000 / 5000 * 301.796s = ~3.35 hours active embedding time
```

Allowing for startup, PyTorch install, occasional GCS retries, and validation,
a practical single-L4 estimate is:

```text
time: ~4-5 hours
cost: likely single-digit USD to low-teens USD, depending on exact G2/L4 pricing
```

The next run should be the full old-support job, not more smoke tests, provided
it uses:

- clean `torch==2.5.1+cu124` virtualenv;
- canonical hashed sample IDs;
- shard-level GCS uploads;
- log upload on exit;
- explicit VM deletion after completion.
