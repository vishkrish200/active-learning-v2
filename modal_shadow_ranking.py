from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.eval.shadow_ranking_eval import (
    load_shadow_ranking_config,
    run_shadow_ranking_eval,
    validate_shadow_ranking_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-shadow-ranking"
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
def remote_shadow_ranking_eval(config: dict) -> dict:
    log_event("modal_shadow_ranking", "remote_start")
    validate_shadow_ranking_config(config)
    result = run_shadow_ranking_eval(config)
    artifacts_volume.commit()
    log_event("modal_shadow_ranking", "remote_done", **result)
    return result


@app.local_entrypoint()
def shadow_ranking_eval(config_path: str = "configs/shadow_quality_gated_grammar.json") -> None:
    config = load_shadow_ranking_config(Path(config_path))
    validate_shadow_ranking_config(config)
    log_event("modal_shadow_ranking", "local_dispatch_start", config_path=config_path)
    result = remote_shadow_ranking_eval.remote(config)
    print(f"Remote shadow ranking eval completed: {result}")
    log_event("modal_shadow_ranking", "local_dispatch_done")
