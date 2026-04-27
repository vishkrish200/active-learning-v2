from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.eval.encoder_eval import load_eval_config, validate_eval_config


APP_NAME = "marginal-value-encoder-eval"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==2.2.6",
        "pandas==2.2.3",
        "torch==2.8.0",
    )
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_encoder_eval(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.modal_encoder_eval import run_encoder_eval

    log_event("modal_eval", "remote_encoder_eval_start", smoke=smoke)
    validate_eval_config(config)
    result = run_encoder_eval(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_eval", "remote_encoder_eval_done", **result)
    return result


@app.local_entrypoint()
def eval_encoder(config_path: str = "configs/eval_encoder.json", smoke: bool = False) -> None:
    config = load_eval_config(Path(config_path))
    validate_eval_config(config)
    log_event("modal_eval", "local_dispatch_start", smoke=smoke)
    result = remote_encoder_eval.remote(config, smoke=smoke)
    print(f"Remote encoder eval completed: {result}")
    log_event("modal_eval", "local_dispatch_done", smoke=smoke)
