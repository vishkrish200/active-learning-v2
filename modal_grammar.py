from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.tokenization.config import load_grammar_config, validate_grammar_config


APP_NAME = "marginal-value-motion-grammar"
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
    gpu="H100",
    timeout=1800,
    volumes={"/artifacts": artifacts_volume},
)
def remote_grammar_smoke(config: dict) -> dict:
    from marginal_value.tokenization.modal_grammar import run_grammar_pipeline

    log_event("modal_grammar", "remote_grammar_smoke_start")
    validate_grammar_config(config)
    result = run_grammar_pipeline(config, smoke=True)
    artifacts_volume.commit()
    log_event("modal_grammar", "remote_grammar_smoke_done", **result)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/artifacts": artifacts_volume},
)
def remote_grammar_full(config: dict) -> dict:
    from marginal_value.tokenization.modal_grammar import run_grammar_pipeline

    log_event("modal_grammar", "remote_grammar_full_start")
    validate_grammar_config(config)
    result = run_grammar_pipeline(config, smoke=False)
    artifacts_volume.commit()
    log_event("modal_grammar", "remote_grammar_full_done", **result)
    return result


@app.local_entrypoint()
def grammar(
    config_path: str = "configs/modal_grammar.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    config = load_grammar_config(Path(config_path))
    validate_grammar_config(config)
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    log_event("modal_grammar", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)

    if skip_smoke:
        log_event("modal_grammar", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_grammar_smoke.remote(config)
        print(f"Remote grammar smoke completed: {smoke_result}")

    if not run_full:
        print("Full grammar scoring was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_grammar", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return

    full_result = remote_grammar_full.remote(config)
    print(f"Remote grammar full completed: {full_result}")
    log_event("modal_grammar", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
