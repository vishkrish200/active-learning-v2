from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "imu2clip-style-training"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6", "torch==2.8.0", "scipy==1.14.1")
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 8,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_train_imu2clip_style(config: dict, mode: str = "smoke") -> dict:
    from marginal_value.training.imu2clip_style_train import train_imu2clip_style_encoder

    run_id = str(config.get("execution", {}).get("run_id", ""))
    log_event(
        "modal_train_imu2clip_style",
        "remote_train_start",
        mode=mode,
        run_id=run_id,
        gpu=str(config.get("execution", {}).get("gpu", "H100")),
    )
    result = train_imu2clip_style_encoder(config, mode=mode)
    artifacts_volume.commit()
    log_event("modal_train_imu2clip_style", "remote_train_done", **result)
    return result


@app.function(
    image=image,
    timeout=3600 * 3,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_preload_imu2clip_windows(config: dict, mode: str = "validation") -> dict:
    from marginal_value.training.imu2clip_style_train import preload_imu2clip_style_windows

    run_id = str(config.get("execution", {}).get("run_id", ""))
    log_event("modal_train_imu2clip_style", "remote_preload_start", run_id=run_id, mode=mode)
    result = preload_imu2clip_style_windows(config, mode=mode)
    artifacts_volume.commit()
    log_event("modal_train_imu2clip_style", "remote_preload_done", **result)
    return result


@app.function(
    image=image,
    timeout=3600 * 2,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_preload_imu2clip_audit_clips(config: dict) -> dict:
    from marginal_value.eval.imu2clip_embedding_audit import preload_embedding_audit_clips

    run_id = str(config.get("execution", {}).get("run_id", ""))
    audit_config = dict(config.get("audit", {}))
    output_path = str(audit_config.get("preloaded_clips_path", "/artifacts/cache/imu2clip_style/audit_clips.npz"))
    log_event(
        "modal_train_imu2clip_style",
        "remote_audit_preload_start",
        run_id=run_id,
        output_path=output_path,
        n_clips=int(audit_config.get("n_clips", 500)),
    )
    result = preload_embedding_audit_clips(
        manifest_path=str(audit_config.get("manifest_path", config["data"]["manifest"])),
        n_clips=int(audit_config.get("n_clips", 500)),
        output_path=output_path,
    )
    artifacts_volume.commit()
    payload = dict(result)
    payload["run_id"] = run_id
    log_event("modal_train_imu2clip_style", "remote_audit_preload_done", **payload)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 10,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_train_and_audit_imu2clip_style(config: dict, mode: str = "validation") -> dict:
    from marginal_value.eval.imu2clip_embedding_audit import run_embedding_audit
    from marginal_value.training.imu2clip_style_train import train_imu2clip_style_encoder

    if mode not in {"validation", "full"}:
        raise ValueError("remote_train_and_audit_imu2clip_style only supports validation or full modes.")
    audit_config = dict(config.get("audit", {}))
    run_id = str(config.get("execution", {}).get("run_id", ""))
    log_event(
        "modal_train_imu2clip_style",
        "remote_train_audit_start",
        run_id=run_id,
        mode=mode,
        gpu=str(config.get("execution", {}).get("gpu", "H100")),
    )
    train_result = train_imu2clip_style_encoder(config, mode=mode)
    artifacts_volume.commit()
    checkpoint_path = str(train_result.get("checkpoint_path", ""))
    if checkpoint_path == "":
        raise RuntimeError("Training did not produce a checkpoint for audit.")
    output_path = str(audit_config.get("output_path", "/artifacts/reports/imu2clip_embedding_audit.md"))
    audit_report = run_embedding_audit(
        manifest_path=str(audit_config.get("manifest_path", config["data"]["manifest"])),
        checkpoint_path=checkpoint_path,
        n_clips=int(audit_config.get("n_clips", 500)),
        output_path=output_path,
        preloaded_clips_path=str(audit_config.get("preloaded_clips_path", "")) or None,
    )
    artifacts_volume.commit()
    result = {
        "mode": mode,
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "audit_output_path": output_path,
        "effective_rank": _audit_metric(audit_report, "effective_rank"),
        "mean_pairwise_cosine": _audit_metric(audit_report, "mean_pairwise_cosine"),
        "train": train_result,
        "audit": audit_report,
    }
    log_event(
        "modal_train_imu2clip_style",
        "remote_train_audit_done",
        mode=mode,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        audit_output_path=output_path,
        effective_rank=result["effective_rank"],
        mean_pairwise_cosine=result["mean_pairwise_cosine"],
    )
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 2,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_audit_imu2clip_style(config: dict, checkpoint_path: str) -> dict:
    from marginal_value.eval.imu2clip_embedding_audit import run_embedding_audit

    audit_config = dict(config.get("audit", {}))
    run_id = str(config.get("execution", {}).get("run_id", ""))
    output_path = str(audit_config.get("output_path", "/artifacts/reports/imu2clip_embedding_audit.md"))
    log_event(
        "modal_train_imu2clip_style",
        "remote_audit_start",
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        n_clips=int(audit_config.get("n_clips", 500)),
    )
    report = run_embedding_audit(
        manifest_path=str(audit_config.get("manifest_path", config["data"]["manifest"])),
        checkpoint_path=checkpoint_path,
        n_clips=int(audit_config.get("n_clips", 500)),
        output_path=output_path,
        preloaded_clips_path=str(audit_config.get("preloaded_clips_path", "")) or None,
    )
    artifacts_volume.commit()
    log_event(
        "modal_train_imu2clip_style",
        "remote_audit_done",
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        effective_rank=_audit_metric(report, "effective_rank"),
        mean_pairwise_cosine=_audit_metric(report, "mean_pairwise_cosine"),
    )
    return report


@app.local_entrypoint()
def main(
    config_path: str = "configs/imu2clip_style_training.json",
    run_validation: bool = False,
    run_full: bool = False,
    skip_smoke: bool = False,
    spawn_remote: bool = False,
    audit_checkpoint_path: str = "",
) -> None:
    if skip_smoke and not (run_validation or run_full or audit_checkpoint_path):
        raise ValueError("--skip-smoke is only valid when launching validation, full training, or an audit.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _with_run_tracking(config, config_path=config_path)
    run_id = str(config["execution"]["run_id"])
    log_event(
        "modal_train_imu2clip_style",
        "local_dispatch_start",
        run_id=run_id,
        config_path=config_path,
        run_validation=run_validation,
        run_full=run_full,
        skip_smoke=skip_smoke,
        spawn_remote=spawn_remote,
        audit_checkpoint_path=audit_checkpoint_path,
        gpu=str(config["execution"].get("gpu", "H100")),
    )
    if audit_checkpoint_path:
        if bool(config.get("audit", {}).get("preload_before_gpu", False)):
            preload_result = remote_preload_imu2clip_audit_clips.remote(config)
            audit_config = dict(config.get("audit", {}))
            audit_config["preloaded_clips_path"] = preload_result["preloaded_clips_path"]
            config["audit"] = audit_config
            log_event(
                "modal_train_imu2clip_style",
                "local_audit_preload_done",
                run_id=run_id,
                preloaded_clips_path=preload_result["preloaded_clips_path"],
                n_clips=preload_result["n_clips"],
            )
        if spawn_remote:
            function_call = remote_audit_imu2clip_style.spawn(config, checkpoint_path=audit_checkpoint_path)
            log_event(
                "modal_train_imu2clip_style",
                "remote_audit_spawned",
                run_id=run_id,
                function_call_id=function_call.object_id,
                dashboard_url=function_call.get_dashboard_url(),
                checkpoint_path=audit_checkpoint_path,
            )
            print("Spawned detached IMU2CLIP-style audit.")
            print(f"run_id={run_id}")
            print(f"function_call_id={function_call.object_id}")
            print(f"dashboard_url={function_call.get_dashboard_url()}")
            return
        audit = remote_audit_imu2clip_style.remote(config, checkpoint_path=audit_checkpoint_path)
        print(f"Checkpoint audit completed: effective_rank={_audit_metric(audit, 'effective_rank')}")
        return
    if spawn_remote:
        if not (run_validation or run_full):
            raise ValueError("--spawn-remote requires --run-validation or --run-full.")
        mode = "full" if run_full else "validation"
        if bool(config.get("data", {}).get("preload_before_gpu", False)):
            preload_result = remote_preload_imu2clip_windows.remote(config, mode=mode)
            data_config = dict(config.get("data", {}))
            data_config["preloaded_windows_path"] = preload_result["preloaded_windows_path"]
            data_config["preload_windows"] = False
            config["data"] = data_config
            log_event(
                "modal_train_imu2clip_style",
                "local_preload_done",
                run_id=run_id,
                mode=mode,
                preloaded_windows_path=preload_result["preloaded_windows_path"],
                windows=preload_result["windows"],
            )
        function_call = remote_train_and_audit_imu2clip_style.spawn(config, mode=mode)
        log_event(
            "modal_train_imu2clip_style",
            "remote_train_audit_spawned",
            run_id=run_id,
            mode=mode,
            function_call_id=function_call.object_id,
            dashboard_url=function_call.get_dashboard_url(),
        )
        print(f"Spawned detached IMU2CLIP-style {mode} train+audit.")
        print(f"run_id={run_id}")
        print(f"function_call_id={function_call.object_id}")
        print(f"dashboard_url={function_call.get_dashboard_url()}")
        return
    if skip_smoke:
        log_event("modal_train_imu2clip_style", "local_smoke_skipped", run_id=run_id, reason="previous_smoke_passed")
    else:
        smoke_result = remote_train_imu2clip_style.remote(config, mode="smoke")
        print(f"Remote IMU2CLIP-style smoke training completed: {smoke_result}")
    if run_validation:
        validation_result = remote_train_imu2clip_style.remote(config, mode="validation")
        print(f"Remote IMU2CLIP-style validation training completed: {validation_result}")
        if validation_result.get("checkpoint_path"):
            audit = remote_audit_imu2clip_style.remote(config, checkpoint_path=str(validation_result["checkpoint_path"]))
            print(f"Validation checkpoint audit completed: effective_rank={_audit_metric(audit, 'effective_rank')}")
    if run_full:
        full_result = remote_train_imu2clip_style.remote(config, mode="full")
        print(f"Remote IMU2CLIP-style full training completed: {full_result}")
        if full_result.get("checkpoint_path"):
            audit = remote_audit_imu2clip_style.remote(config, checkpoint_path=str(full_result["checkpoint_path"]))
            print(f"Full checkpoint audit completed: effective_rank={_audit_metric(audit, 'effective_rank')}")
    log_event("modal_train_imu2clip_style", "local_dispatch_done", run_id=run_id, run_validation=run_validation, run_full=run_full)


def _with_run_tracking(config: dict, *, config_path: str) -> dict:
    output = dict(config)
    execution = dict(output.get("execution", {}))
    execution.setdefault("gpu", "H100")
    execution["run_id"] = str(execution.get("run_id") or f"imu2clip-style-{uuid4().hex[:12]}")
    execution["config_path"] = config_path
    output["execution"] = execution
    return output


def _audit_metric(report: dict, key: str) -> float:
    collapse = report.get("collapse")
    if isinstance(collapse, dict) and key in collapse:
        return float(collapse[key])
    return float(report[key])
