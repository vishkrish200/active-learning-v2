from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import modal

from marginal_value.logging_utils import log_event, log_progress


APP_NAME = "imagebind-imu-embedding"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

IMAGEBIND_REPO = "https://github.com/facebookresearch/ImageBind.git"
IMAGEBIND_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "numpy==2.2.6",
        "scipy==1.14.1",
        "torch==2.8.0",
        "timm==1.0.15",
        "ftfy==6.3.1",
        "regex==2024.11.6",
        "iopath==0.1.10",
        "einops==0.8.1",
    )
    .run_commands(f"git clone --depth 1 {IMAGEBIND_REPO} /opt/ImageBind")
    .run_commands("python -c \"from pathlib import Path; Path('/opt/ImageBind/imagebind/__init__.py').write_text('')\"")
    .env({"PYTHONPATH": "/opt/ImageBind"})
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=3600 * 2,
    cpu=8,
    memory=32768,
    volumes={"/artifacts": artifacts_volume},
)
def remote_prepare_imagebind_imu(
    checkpoint_dir: str = "/artifacts/checkpoints/imagebind",
    run_id: str = "",
) -> dict[str, object]:
    import torch
    from imagebind.models import imagebind_model

    run_id = run_id or f"imagebind-imu-{uuid4().hex[:12]}"
    checkpoint_path = Path(checkpoint_dir) / "imagebind_huge.pth"
    log_event(
        "modal_embed_imagebind_imu",
        "prepare_start",
        run_id=run_id,
        checkpoint_path=str(checkpoint_path),
    )
    model = imagebind_model.imagebind_huge(pretrained=False)
    del model
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        log_event("modal_embed_imagebind_imu", "checkpoint_download_start", run_id=run_id)
        torch.hub.download_url_to_file(IMAGEBIND_CHECKPOINT_URL, str(checkpoint_path), progress=True)
        log_event("modal_embed_imagebind_imu", "checkpoint_download_done", run_id=run_id)
    artifacts_volume.commit()
    result = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_gb": checkpoint_path.stat().st_size / 1.0e9,
    }
    log_event("modal_embed_imagebind_imu", "prepare_done", **result)
    return result


@app.function(
    image=image,
    timeout=3600 * 2,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_preload_imagebind_imu_audit_clips(
    manifest_path: str,
    n_clips: int = 500,
    output_path: str = "/artifacts/cache/imagebind_imu/audit_clips_500.npz",
    run_id: str = "",
) -> dict[str, object]:
    from marginal_value.eval.imagebind_imu_embedding_audit import preload_imagebind_imu_audit_clips

    run_id = run_id or f"imagebind-imu-{uuid4().hex[:12]}"
    log_event(
        "modal_embed_imagebind_imu",
        "audit_preload_start",
        run_id=run_id,
        manifest_path=manifest_path,
        n_clips=int(n_clips),
        output_path=output_path,
    )
    result = preload_imagebind_imu_audit_clips(
        manifest_path=manifest_path,
        n_clips=int(n_clips),
        output_path=output_path,
        run_id=run_id,
    )
    artifacts_volume.commit()
    log_event("modal_embed_imagebind_imu", "audit_preload_done", **result)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 4,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_audit_imagebind_imu(
    manifest_path: str,
    output_path: str = "/artifacts/reports/imagebind_imu_embedding_audit.md",
    n_clips: int = 500,
    preloaded_clips_path: str = "/artifacts/cache/imagebind_imu/audit_clips_500.npz",
    checkpoint_dir: str = "/artifacts/checkpoints/imagebind",
    batch_size: int = 32,
    run_id: str = "",
) -> dict[str, object]:
    from marginal_value.eval.imagebind_imu_embedding_audit import run_imagebind_imu_embedding_audit

    run_id = run_id or f"imagebind-imu-{uuid4().hex[:12]}"
    log_event(
        "modal_embed_imagebind_imu",
        "audit_start",
        run_id=run_id,
        gpu="H100",
        manifest_path=manifest_path,
        output_path=output_path,
        n_clips=int(n_clips),
        preloaded_clips_path=preloaded_clips_path,
        checkpoint_dir=checkpoint_dir,
        batch_size=int(batch_size),
    )
    report = run_imagebind_imu_embedding_audit(
        manifest_path=manifest_path,
        n_clips=int(n_clips),
        output_path=output_path,
        preloaded_clips_path=preloaded_clips_path or None,
        checkpoint_dir=checkpoint_dir,
        batch_size=int(batch_size),
        run_id=run_id,
    )
    artifacts_volume.commit()
    log_event(
        "modal_embed_imagebind_imu",
        "audit_done",
        run_id=run_id,
        output_path=output_path,
        effective_rank=float(report["effective_rank"]),
        mean_pairwise_cosine=float(report["mean_pairwise_cosine"]),
        passed_embedding_audit=bool(report["passed_embedding_audit"]),
    )
    return report


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 4,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def embed_corpus(
    manifest_path: str,
    output_path: str,
    checkpoint_dir: str = "/artifacts/checkpoints/imagebind",
    batch_size: int = 32,
    run_id: str = "",
) -> dict[str, object]:
    import numpy as np

    from marginal_value.models.imagebind_imu_inference import ImageBindIMUInference
    from marginal_value.preprocessing.quality import load_modal_jsonl_imu

    run_id = run_id or f"imagebind-imu-{uuid4().hex[:12]}"
    records = _read_manifest(manifest_path)
    log_event(
        "modal_embed_imagebind_imu",
        "embed_start",
        run_id=run_id,
        gpu="H100",
        manifest_path=manifest_path,
        output_path=output_path,
        n_clips=len(records),
        checkpoint_dir=checkpoint_dir,
        batch_size=int(batch_size),
    )
    inference = ImageBindIMUInference(device="cuda", checkpoint_dir=checkpoint_dir, batch_size=int(batch_size))
    rows: list[np.ndarray] = []
    clip_ids: list[str] = []
    progress_every = max(1, min(100, len(records) // 100 or 1))
    for index, record in enumerate(records, start=1):
        clip, _timestamps = load_modal_jsonl_imu(_resolve_raw_path(record["path"]))
        rows.append(inference.encode_clip(clip))
        clip_ids.append(record["clip_id"])
        log_progress(
            "modal_embed_imagebind_imu",
            "embed_progress",
            index=index,
            total=len(records),
            every=progress_every,
            run_id=run_id,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(rows).astype("float32") if rows else np.empty((0, 1024), dtype="float32")
    np.save(output, matrix)
    index_path = output.with_suffix(".clip_ids.json")
    index_path.write_text(json.dumps({"clip_ids": clip_ids}, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    result = {
        "run_id": run_id,
        "output_path": str(output),
        "index_path": str(index_path),
        "n_clips": len(clip_ids),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
    }
    log_event("modal_embed_imagebind_imu", "embed_done", **result)
    return result


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                raw_path = str(record.get("raw_path") or record.get("path") or record.get("url") or "")
                clip_id = str(record.get("sample_id") or record.get("clip_id") or Path(raw_path).stem or index)
            else:
                raw_path = line
                clip_id = Path(line).stem or str(index)
            rows.append({"clip_id": clip_id, "path": raw_path})
    return rows


def _resolve_raw_path(path_or_url: str) -> Path:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://") or path_or_url.startswith("gs://"):
        from marginal_value.data.split_manifest import hash_manifest_url

        return Path("/data/cache/raw") / f"{hash_manifest_url(path_or_url)}.jsonl"
    return Path(path_or_url)


@app.local_entrypoint()
def main(
    manifest_path: str = "/data/cache/manifests/pretrain_full_cached_urls.txt",
    checkpoint_dir: str = "/artifacts/checkpoints/imagebind",
    audit_preload_path: str = "/artifacts/cache/imagebind_imu/audit_clips_500.npz",
    audit_output_path: str = "/artifacts/reports/imagebind_imu_embedding_audit.md",
    embedding_output_path: str = "/artifacts/embeddings/imagebind_imu_pretrain.npy",
    n_audit_clips: int = 500,
    batch_size: int = 32,
    run_audit: bool = False,
    run_full_embed: bool = False,
    skip_prepare: bool = False,
    skip_preload: bool = False,
) -> None:
    run_id = f"imagebind-imu-{uuid4().hex[:12]}"
    log_event(
        "modal_embed_imagebind_imu",
        "local_dispatch_start",
        run_id=run_id,
        run_audit=run_audit,
        run_full_embed=run_full_embed,
        skip_prepare=skip_prepare,
        skip_preload=skip_preload,
    )
    if not skip_prepare:
        print(f"Preparing ImageBind checkpoint/import on CPU for run {run_id}...")
        print(remote_prepare_imagebind_imu.remote(checkpoint_dir=checkpoint_dir, run_id=run_id))
    if not skip_preload:
        print(f"Preloading audit clips on CPU for run {run_id}...")
        print(
            remote_preload_imagebind_imu_audit_clips.remote(
                manifest_path=manifest_path,
                n_clips=n_audit_clips,
                output_path=audit_preload_path,
                run_id=run_id,
            )
        )
    if run_audit:
        print(f"Running H100 ImageBind IMU audit for run {run_id}...")
        print(
            remote_audit_imagebind_imu.remote(
                manifest_path=manifest_path,
                output_path=audit_output_path,
                n_clips=n_audit_clips,
                preloaded_clips_path=audit_preload_path,
                checkpoint_dir=checkpoint_dir,
                batch_size=batch_size,
                run_id=run_id,
            )
        )
    else:
        print("H100 audit was not launched. Re-run with --run-audit after reviewing CPU prepare/preload logs.")
    if run_full_embed:
        print(f"Running H100 full embedding for run {run_id}...")
        print(
            embed_corpus.remote(
                manifest_path=manifest_path,
                output_path=embedding_output_path,
                checkpoint_dir=checkpoint_dir,
                batch_size=batch_size,
                run_id=run_id,
            )
        )
    elif run_audit:
        print("Full embedding was not launched. Only run it after the audit passes.")
    log_event("modal_embed_imagebind_imu", "local_dispatch_done", run_id=run_id)
