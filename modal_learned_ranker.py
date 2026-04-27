from __future__ import annotations

import json
from pathlib import Path

import modal

from marginal_value.eval.learned_ranker_eval import run_learned_ranker_eval, validate_learned_ranker_config
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-learned-ranker"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=900,
    volumes={"/artifacts": artifacts_volume},
)
def remote_learned_ranker_eval(config: dict) -> dict:
    log_event("modal_learned_ranker", "remote_start")
    validate_learned_ranker_config(config)
    result = run_learned_ranker_eval(config)
    artifacts_volume.commit()
    log_event("modal_learned_ranker", "remote_done", **result)
    return result


@app.local_entrypoint()
def learned_ranker_eval(config_path: str = "configs/learned_ranker_eval.json") -> None:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_learned_ranker_config(config)
    log_event("modal_learned_ranker", "local_dispatch_start")
    result = remote_learned_ranker_eval.remote(config)
    print(f"Remote learned ranker eval completed: {result}")
    log_event("modal_learned_ranker", "local_dispatch_done")
