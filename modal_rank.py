from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.ranking.config import load_ranking_config, validate_ranking_config


APP_NAME = "marginal-value-baseline-ranking"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=3600,
    cpu=16,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_baseline_rank(config: dict, smoke: bool = False) -> dict:
    from marginal_value.ranking.modal_baseline_rank import run_baseline_ranking

    log_event("modal_rank", "remote_baseline_rank_start", smoke=smoke)
    validate_ranking_config(config)
    result = run_baseline_ranking(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_rank", "remote_baseline_rank_done", **result)
    return result


@app.local_entrypoint()
def rank_baseline(config_path: str = "configs/baseline_ranking.json", smoke: bool = False) -> None:
    config = load_ranking_config(Path(config_path))
    validate_ranking_config(config)
    log_event("modal_rank", "local_dispatch_start", smoke=smoke)
    result = remote_baseline_rank.remote(config, smoke=smoke)
    print(f"Remote baseline ranking completed: {result}")
    log_event("modal_rank", "local_dispatch_done", smoke=smoke)
