from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "imu2clip-embedding"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6", "scipy==1.14.1", "torch==2.8.0")
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 4,
    memory=16384,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def embed_corpus(
    manifest_path: str,
    checkpoint_path: str,
    output_path: str,
    representation: str = "imu2clip",
    batch_size: int = 32,
) -> dict[str, object]:
    import numpy as np

    from marginal_value.models.imu2clip_inference import IMU2CLIPInference
    from marginal_value.preprocessing.quality import load_modal_jsonl_imu

    if representation not in {"imu2clip", "imu2clip_multiscale"}:
        raise ValueError("representation must be 'imu2clip' or 'imu2clip_multiscale'.")
    inference = IMU2CLIPInference(checkpoint_path=checkpoint_path, device="cuda", batch_size=batch_size)
    records = _read_manifest(manifest_path)
    rows: list[np.ndarray] = []
    clip_ids: list[str] = []
    for record in records:
        clip, _timestamps = load_modal_jsonl_imu(_resolve_raw_path(record["path"]))
        if representation == "imu2clip":
            embedding = inference.encode_clip(clip)
        else:
            embedding = inference.encode_clip_multiscale(clip)
        rows.append(embedding)
        clip_ids.append(record["clip_id"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(rows).astype("float32") if rows else np.empty((0, 512), dtype="float32")
    np.save(output, matrix)
    index_path = output.with_suffix(".clip_ids.json")
    index_path.write_text(json.dumps({"clip_ids": clip_ids}, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return {
        "output_path": str(output),
        "index_path": str(index_path),
        "n_clips": len(clip_ids),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
    }


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
    checkpoint_path: str = "/artifacts/checkpoints/imu2clip/imu_encoder.pt",
    output_path: str = "/artifacts/embeddings/imu2clip_pretrain.npy",
    representation: str = "imu2clip",
    batch_size: int = 32,
) -> None:
    result = embed_corpus.remote(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        representation=representation,
        batch_size=batch_size,
    )
    print(f"IMU2CLIP embedding completed: {result}")
