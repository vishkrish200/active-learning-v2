from __future__ import annotations

import json
from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-marginal-coverage-eval"
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
def remote_marginal_coverage_eval(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.marginal_coverage_eval import run_marginal_coverage_eval

    log_event("modal_marginal_coverage_eval", "remote_start", smoke=smoke)
    result = run_marginal_coverage_eval(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_marginal_coverage_eval", "remote_done", **result)
    return result


@app.local_entrypoint()
def marginal_coverage_eval(
    config_path: str = "configs/marginal_coverage_eval_subcluster40.json",
    smoke: bool = False,
) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_marginal_coverage_eval", "local_dispatch_start", smoke=smoke, config_path=config_path)
    result = remote_marginal_coverage_eval.remote(config, smoke=smoke)
    print(f"Remote marginal coverage eval completed: {result}")
    log_event("modal_marginal_coverage_eval", "local_dispatch_done", smoke=smoke)
