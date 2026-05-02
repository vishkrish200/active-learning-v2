from __future__ import annotations

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "ts2vec-rank-probe"
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
    timeout=1800,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def probe(
    manifest_path: str = "/data/cache/manifests/pretrain_full_cached_urls.txt",
    n_clips: int = 32,
    clip_len: int = 300,
    output_path: str = "/artifacts/docs/ts2vec_layer_rank_probe.json",
    seed: int = 20260430,
    random_start: bool = True,
) -> list[dict]:
    from marginal_value.models.ts2vec_diagnostics import run_layer_rank_probe

    log_event(
        "modal_probe_ts2vec_ranks",
        "remote_start",
        manifest_path=manifest_path,
        n_clips=n_clips,
        clip_len=clip_len,
        seed=seed,
        random_start=random_start,
    )
    results = run_layer_rank_probe(
        manifest_path=manifest_path,
        n_clips=n_clips,
        clip_len=clip_len,
        device="cuda",
        output_path=output_path,
        seed=seed,
        random_start=random_start,
    )
    artifacts_volume.commit()
    for row in results:
        log_event("modal_probe_ts2vec_ranks", "layer", **row)
    log_event("modal_probe_ts2vec_ranks", "remote_done", output_path=output_path)
    return results


@app.local_entrypoint()
def main(
    manifest_path: str = "/data/cache/manifests/pretrain_full_cached_urls.txt",
    n_clips: int = 32,
    clip_len: int = 300,
    output_path: str = "/artifacts/docs/ts2vec_layer_rank_probe.json",
    seed: int = 20260430,
    random_start: bool = True,
) -> None:
    results = probe.remote(
        manifest_path=manifest_path,
        n_clips=n_clips,
        clip_len=clip_len,
        output_path=output_path,
        seed=seed,
        random_start=random_start,
    )
    for row in results:
        print(
            f"{row['layer']:>22}: "
            f"rank={row['effective_rank']:7.4f}  "
            f"cosine={row['mean_pairwise_cosine']:7.4f}  "
            f"std={row['activation_std']:8.6f}  "
            f"mean_abs={row['activation_mean_abs']:8.6f}"
        )
