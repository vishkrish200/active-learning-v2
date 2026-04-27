from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.eval.grammar_ablation_eval import load_grammar_ablation_config, validate_grammar_ablation_config
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-grammar-ablation"
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
def remote_grammar_ablation(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.grammar_ablation_eval import run_grammar_ablation

    log_event("modal_grammar_ablation", "remote_grammar_ablation_start", smoke=smoke)
    validate_grammar_ablation_config(config)
    result = run_grammar_ablation(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_grammar_ablation", "remote_grammar_ablation_done", **result)
    return result


@app.function(
    image=image,
    timeout=600,
    volumes={"/artifacts": artifacts_volume},
)
def remote_leave_cluster_out_ablation(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.grammar_ablation_eval import run_leave_cluster_out_ablation

    log_event("modal_grammar_ablation", "remote_leave_cluster_out_start", smoke=smoke)
    validate_grammar_ablation_config(config)
    result = run_leave_cluster_out_ablation(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_grammar_ablation", "remote_leave_cluster_out_done", **result)
    return result


@app.local_entrypoint()
def grammar_ablation(
    config_path: str = "configs/grammar_ablation.json",
    smoke: bool = False,
    task: str = "candidate",
) -> None:
    config = load_grammar_ablation_config(Path(config_path))
    validate_grammar_ablation_config(config)
    log_event("modal_grammar_ablation", "local_dispatch_start", smoke=smoke, task=task)
    if task == "candidate":
        result = remote_grammar_ablation.remote(config, smoke=smoke)
    elif task == "leave_cluster_out":
        result = remote_leave_cluster_out_ablation.remote(config, smoke=smoke)
    else:
        raise ValueError("task must be 'candidate' or 'leave_cluster_out'.")
    print(f"Remote grammar ablation completed: {result}")
    log_event("modal_grammar_ablation", "local_dispatch_done", smoke=smoke, task=task)
