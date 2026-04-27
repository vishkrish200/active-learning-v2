# Marginal Data Value Ranking for IMU Clips

This project scores new IMU clips by expected marginal dataset value, not by raw anomaly.

The first runnable slice is deliberately dependency-light:

- streaming-style CSV loading and validation
- IMU preprocessing and quality scoring
- deterministic handcrafted IMU clip encoder baseline
- exact cosine kNN old-corpus support features
- new-batch support features
- variable-duration motion primitive discovery with motion-BPE-style phrase mining
- n-gram motion grammar surprisal features
- quality-gated marginal value ranker
- MMR diversity reranking
- submission and diagnostics CSV writers

Heavy training components from the full plan, such as PatchTST/TS2Vec pretraining, FAISS, VQ, transformer grammar LMs, and LightGBM LambdaRank, are represented by replaceable module boundaries so they can be added once the real challenge data and dependency stack are available.

## Quick Test

```bash
python3 -m unittest discover -s tests
```

## Rank CSV Directories

Each worker CSV should contain six IMU channels. Timestamp columns are optional but recommended.

```bash
marginal-value rank \
  --existing-dir data/raw/existing \
  --new-dir data/raw/new \
  --submission-out data/submissions/submission.csv \
  --diagnostics-out data/submissions/diagnostics.csv
```

Expected minimal submission columns:

```text
worker_id,rank,score,quality_score,reason_code
```

## Modal-Only Training

Do not run actual training on the local Mac. Local commands are only for validation and packaging checks.

First validate the config locally:

```bash
python3 -m marginal_value.cli validate-training --config configs/modal_training.json
```

Make sure the Modal CLI is installed and authenticated:

```bash
python3 -m pip install modal
modal setup
```

Then run the Modal smoke training job on an H100. This performs a tiny optimizer-step check remotely and does not launch the full job:

```bash
modal run modal_train.py --config-path configs/modal_training.json
```

After the remote smoke result looks healthy, run the validation job. This performs a larger remote-only training pass and verifies checkpoint write/read against the Modal artifacts volume:

```bash
modal run modal_train.py --config-path configs/modal_training.json --run-validation
```

After validation passes, launch full training on Modal:

```bash
modal run modal_train.py --config-path configs/modal_training.json --run-full
```

The `modal_train.py` entrypoint always calls `remote_smoke_train.remote(...)` before `remote_full_train.remote(...)`, so full training is not sent until the remote H100 smoke path has passed. Training data is read from the Modal volume named by `configs/modal_training.json` at `/data/cache/features/*.npz`, and checkpoints are written under `/artifacts`.
