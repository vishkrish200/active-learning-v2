from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-tokenizer-transform"
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
    timeout=1200,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_tokenizer_transform(config: dict, smoke: bool = False) -> dict:
    from marginal_value.tokenization.transform_existing import run_existing_tokenizer_transform

    log_event("modal_tokenizer_transform", "remote_start", smoke=smoke)
    result = run_existing_tokenizer_transform(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_tokenizer_transform", "remote_done", **result)
    return result


@app.local_entrypoint()
def transform_tokens(
    config_path: str = "configs/modal_tokenizer_new_transform.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_tokenizer_transform", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_tokenizer_transform", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_tokenizer_transform.remote(config, smoke=True)
        print(f"Remote tokenizer transform smoke completed: {smoke_result}")
    if not run_full:
        print("Full tokenizer transform was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_tokenizer_transform", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_tokenizer_transform.remote(config, smoke=False)
    print(f"Remote tokenizer transform full completed: {full_result}")
    log_event("modal_tokenizer_transform", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
