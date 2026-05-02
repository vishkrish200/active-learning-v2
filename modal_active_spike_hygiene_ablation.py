from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.active.spike_hygiene_ablation import (
    load_spike_hygiene_ablation_config,
    validate_spike_hygiene_ablation_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-active-spike-hygiene-ablation"
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
    timeout=1800,
    cpu=8,
    memory=16384,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_active_spike_hygiene_ablation(config: dict, smoke: bool = False) -> dict:
    from marginal_value.active.spike_hygiene_ablation import run_spike_hygiene_ablation

    log_event("modal_active_spike_hygiene_ablation", "remote_start", smoke=smoke)
    validate_spike_hygiene_ablation_config(config)
    result = run_spike_hygiene_ablation(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_active_spike_hygiene_ablation", "remote_done", **result)
    return result


@app.local_entrypoint()
def active_spike_hygiene_ablation(
    config_path: str = "configs/active_spike_hygiene_ablation_exact_window.json",
    smoke: bool = False,
) -> None:
    config = load_spike_hygiene_ablation_config(Path(config_path))
    validate_spike_hygiene_ablation_config(config)
    log_event("modal_active_spike_hygiene_ablation", "local_dispatch_start", smoke=smoke, config_path=config_path)
    result = remote_active_spike_hygiene_ablation.remote(config, smoke=smoke)
    print(f"Remote active spike hygiene ablation completed: {result}")
    log_event("modal_active_spike_hygiene_ablation", "local_dispatch_done", smoke=smoke)
