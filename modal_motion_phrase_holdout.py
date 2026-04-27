from __future__ import annotations

import json
from pathlib import Path

import modal

from marginal_value.eval.motion_phrase_holdout_eval import (
    run_motion_phrase_holdout_eval,
    validate_motion_phrase_holdout_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-motion-phrase-holdout"
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
def remote_motion_phrase_holdout_eval(config: dict, smoke: bool = False) -> dict:
    log_event("modal_motion_phrase_holdout", "remote_start", smoke=smoke)
    validate_motion_phrase_holdout_config(config)
    result = run_motion_phrase_holdout_eval(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_motion_phrase_holdout", "remote_done", **result)
    return result


@app.local_entrypoint()
def motion_phrase_holdout(
    config_path: str = "configs/motion_phrase_holdout_eval.json",
    smoke: bool = False,
) -> None:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_motion_phrase_holdout_config(config)
    log_event("modal_motion_phrase_holdout", "local_dispatch_start", config_path=config_path, smoke=smoke)
    result = remote_motion_phrase_holdout_eval.remote(config, smoke=smoke)
    print(f"Remote motion phrase holdout completed: {result}")
    log_event("modal_motion_phrase_holdout", "local_dispatch_done")
