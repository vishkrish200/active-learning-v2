from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.eval.rerank_eval import load_rerank_eval_config, run_rerank_eval, validate_rerank_eval_config
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-rerank-eval"
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
def remote_rerank_eval(config: dict) -> dict:
    log_event("modal_rerank_eval", "remote_start")
    validate_rerank_eval_config(config)
    result = run_rerank_eval(config)
    artifacts_volume.commit()
    log_event("modal_rerank_eval", "remote_done", **result)
    return result


@app.local_entrypoint()
def rerank_eval(config_path: str = "configs/rerank_eval_grammar_promoted.json") -> None:
    config = load_rerank_eval_config(Path(config_path))
    validate_rerank_eval_config(config)
    log_event("modal_rerank_eval", "local_dispatch_start", config_path=config_path)
    result = remote_rerank_eval.remote(config)
    print(f"Remote rerank eval completed: {result}")
    log_event("modal_rerank_eval", "local_dispatch_done")
