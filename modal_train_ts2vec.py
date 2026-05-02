from __future__ import annotations

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "ts2vec-training"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==2.2.6",
        "torch==2.8.0",
    )
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 6,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def train(
    manifest_path: str = "/data/cache/manifests/pretrain_full_cached_urls.txt",
    checkpoint_dir: str = "/artifacts/checkpoints/ts2vec",
    n_epochs: int = 20,
    batch_size: int = 8,
    crop_min_len: int = 150,
    crop_max_len: int = 600,
    max_steps_per_epoch: int | None = None,
    collapse_sample_size: int = 1000,
    training_log_path: str = "/artifacts/docs/ts2vec_training_log.md",
    alpha: float = 0.1,
    instance_warmup_epochs: int = 0,
    collapse_pooling: str = "aggregation",
    max_temporal_positions: int = 64,
    collapse_min_effective_rank: float = 10.0,
    collapse_max_mean_pairwise_cosine: float = 0.9,
) -> dict:
    from marginal_value.training.train_ts2vec import train_ts2vec

    log_event("modal_train_ts2vec", "remote_start", manifest_path=manifest_path, checkpoint_dir=checkpoint_dir)
    result = train_ts2vec(
        manifest_path=manifest_path,
        checkpoint_dir=checkpoint_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        crop_min_len=crop_min_len,
        crop_max_len=crop_max_len,
        max_steps_per_epoch=max_steps_per_epoch,
        collapse_sample_size=collapse_sample_size,
        training_log_path=training_log_path,
        alpha=alpha,
        instance_warmup_epochs=instance_warmup_epochs,
        collapse_pooling=collapse_pooling,
        max_temporal_positions=max_temporal_positions,
        collapse_min_effective_rank=collapse_min_effective_rank,
        collapse_max_mean_pairwise_cosine=collapse_max_mean_pairwise_cosine,
        device="cuda",
    )
    artifacts_volume.commit()
    log_event("modal_train_ts2vec", "remote_done", **result)
    return result


@app.local_entrypoint()
def main(
    n_epochs: int = 20,
    batch_size: int = 8,
    crop_min_len: int = 150,
    crop_max_len: int = 600,
    max_steps_per_epoch: int | None = None,
    collapse_sample_size: int = 1000,
    training_log_path: str = "/artifacts/docs/ts2vec_training_log.md",
    alpha: float = 0.1,
    instance_warmup_epochs: int = 0,
    collapse_pooling: str = "aggregation",
    max_temporal_positions: int = 64,
    collapse_min_effective_rank: float = 10.0,
    collapse_max_mean_pairwise_cosine: float = 0.9,
    manifest_path: str = "/data/cache/manifests/pretrain_full_cached_urls.txt",
    checkpoint_dir: str = "/artifacts/checkpoints/ts2vec",
) -> None:
    result = train.remote(
        manifest_path=manifest_path,
        checkpoint_dir=checkpoint_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        crop_min_len=crop_min_len,
        crop_max_len=crop_max_len,
        max_steps_per_epoch=max_steps_per_epoch,
        collapse_sample_size=collapse_sample_size,
        training_log_path=training_log_path,
        alpha=alpha,
        instance_warmup_epochs=instance_warmup_epochs,
        collapse_pooling=collapse_pooling,
        max_temporal_positions=max_temporal_positions,
        collapse_min_effective_rank=collapse_min_effective_rank,
        collapse_max_mean_pairwise_cosine=collapse_max_mean_pairwise_cosine,
    )
    print(f"Remote TS2Vec training completed: {result}")
