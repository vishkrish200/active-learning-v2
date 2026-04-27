from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.eval.score_calibration_eval import (
    load_score_calibration_config,
    run_score_calibration_eval,
    validate_score_calibration_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-score-calibration"
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
    timeout=600,
    volumes={"/artifacts": artifacts_volume},
)
def remote_score_calibration_eval(config: dict) -> dict:
    log_event("modal_score_calibration", "remote_start")
    validate_score_calibration_config(config)
    result = run_score_calibration_eval(config)
    artifacts_volume.commit()
    log_event("modal_score_calibration", "remote_done", **result)
    return result


@app.local_entrypoint()
def score_calibration_eval(config_path: str = "configs/score_calibration_eval.json") -> None:
    config = load_score_calibration_config(Path(config_path))
    validate_score_calibration_config(config)
    log_event("modal_score_calibration", "local_dispatch_start", config_path=config_path)
    result = remote_score_calibration_eval.remote(config)
    print(f"Remote score calibration eval completed: {result}")
    log_event("modal_score_calibration", "local_dispatch_done")
