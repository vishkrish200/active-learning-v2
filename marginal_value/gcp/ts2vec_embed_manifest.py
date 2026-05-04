from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.request import urlopen

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.logging_utils import log_event
from marginal_value.models.ts2vec_inference import TS2VecInference
from marginal_value.preprocessing.quality import _sample_from_jsonl_record


DEFAULT_OLD_MANIFEST = "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain_urls.txt"
DEFAULT_NEW_MANIFEST = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val_urls.txt"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    start = time.time()
    urls = _read_manifest(args.manifest)
    if args.limit is not None:
        urls = urls[: int(args.limit)]
    if not urls:
        raise ValueError(f"No URLs found in manifest: {args.manifest}")

    sample_ids = [_sample_id_from_url(url) for url in urls]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(
            "Manifest sample_id collision detected. "
            "TS2Vec support shards require one stable unique sample_id per URL."
        )
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / args.shard_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    log_event(
        "gcp_ts2vec_embed_manifest",
        "start",
        manifest=args.manifest,
        n_urls=len(urls),
        output_dir=str(output_dir),
        gcs_output=args.gcs_output or "",
        device=args.device,
        shard_size=args.shard_size,
        load_workers=args.load_workers,
        encode_batch_size=args.encode_batch_size,
    )

    encoder = TS2VecInference(args.checkpoint, device=args.device)
    manifest_shards: list[dict[str, object]] = []
    completed = _completed_shards(output_dir / args.manifest_name)
    shard_count = (len(urls) + int(args.shard_size) - 1) // int(args.shard_size)

    for shard_index in range(shard_count):
        start_idx = shard_index * int(args.shard_size)
        end_idx = min(len(urls), start_idx + int(args.shard_size))
        shard_name = f"shard_{shard_index:05d}.npz"
        relpath = f"{args.shard_dir_name}/{shard_name}"
        local_shard = shard_dir / shard_name
        shard_urls = urls[start_idx:end_idx]
        shard_ids = sample_ids[start_idx:end_idx]
        if shard_index in completed and local_shard.exists():
            log_event(
                "gcp_ts2vec_embed_manifest",
                "shard_skip_existing",
                shard_index=shard_index,
                n_shards=shard_count,
                path=str(local_shard),
                n_clips=len(shard_ids),
            )
        else:
            matrix = _embed_url_batch(
                encoder,
                shard_urls,
                load_workers=int(args.load_workers),
                encode_batch_size=int(args.encode_batch_size),
                max_samples=args.max_samples,
            )
            _write_shard(local_shard, shard_ids, matrix)
            log_event(
                "gcp_ts2vec_embed_manifest",
                "shard_write",
                shard_index=shard_index + 1,
                n_shards=shard_count,
                path=str(local_shard),
                n_clips=len(shard_ids),
            )
            if args.gcs_output:
                _gcloud_cp(local_shard, f"{args.gcs_output.rstrip('/')}/{relpath}")

        manifest_shards.append(
            {
                "index": int(shard_index),
                "path": relpath,
                "sample_ids": shard_ids,
                "n_clips": int(len(shard_ids)),
            }
        )
        _write_manifest(
            output_dir / args.manifest_name,
            sample_ids=sample_ids,
            shards=manifest_shards,
            args=vars(args),
        )
        if args.gcs_output:
            _gcloud_cp(output_dir / args.manifest_name, f"{args.gcs_output.rstrip('/')}/{args.manifest_name}")

    elapsed = time.time() - start
    log_event(
        "gcp_ts2vec_embed_manifest",
        "done",
        n_urls=len(urls),
        n_shards=shard_count,
        elapsed_seconds=round(elapsed, 3),
        output_dir=str(output_dir),
        gcs_output=args.gcs_output or "",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed an IMU JSONL manifest with a TS2Vec checkpoint.")
    parser.add_argument("--manifest", default=DEFAULT_OLD_MANIFEST, help="Local path, https URL, or gs:// manifest.")
    parser.add_argument("--checkpoint", required=True, help="Local TS2Vec checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Local output directory for shard files.")
    parser.add_argument("--gcs-output", default="", help="Optional gs:// prefix to upload shards and manifest.")
    parser.add_argument("--manifest-name", default="ts2vec_embedding_shards.json")
    parser.add_argument("--shard-dir-name", default="ts2vec_embedding_shards")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--load-workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests.")
    args = parser.parse_args(argv)
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size must be positive.")
    if int(args.encode_batch_size) <= 0:
        raise ValueError("--encode-batch-size must be positive.")
    if int(args.load_workers) <= 0:
        raise ValueError("--load-workers must be positive.")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("--limit must be positive when provided.")
    return args


def _read_manifest(path_or_url: str) -> list[str]:
    text = _read_text(path_or_url)
    urls = [line.strip() for line in text.splitlines() if line.strip()]
    return [line for line in urls if not line.startswith("#")]


def _read_text(path_or_url: str) -> str:
    if path_or_url.startswith("gs://"):
        return subprocess.check_output(["gcloud", "storage", "cat", path_or_url], text=True)
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(path_or_url).read_text(encoding="utf-8")


def _sample_id_from_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise ValueError(f"Could not derive sample_id from URL: {url}")
    return hash_manifest_url(value)


def _embed_url_batch(
    encoder: TS2VecInference,
    urls: Sequence[str],
    *,
    load_workers: int,
    encode_batch_size: int,
    max_samples: int | None,
) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=int(load_workers)) as pool:
        clips = list(pool.map(lambda url: _load_jsonl_imu_url(url, max_samples=max_samples), urls))
    return encoder.encode_batch(clips, batch_size=int(encode_batch_size)).astype("float32")


def _load_jsonl_imu_url(url: str, *, max_samples: int | None) -> np.ndarray:
    samples: list[list[float]] = []
    if url.startswith("gs://"):
        text = subprocess.check_output(["gcloud", "storage", "cat", url], text=True)
        lines: Iterable[str] = text.splitlines()
    elif url.startswith(("http://", "https://")):
        with urlopen(url, timeout=120) as response:
            lines = (line.decode("utf-8") for line in response)
            return _samples_from_lines(lines, max_samples=max_samples)
    else:
        lines = Path(url).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        line = line.strip()
        if not line:
            continue
        samples.append(_sample_from_jsonl_record(json.loads(line)))
    if not samples:
        raise ValueError(f"No IMU samples found in {url}")
    return np.asarray(samples, dtype=np.float32)


def _samples_from_lines(lines: Iterable[str], *, max_samples: int | None) -> np.ndarray:
    samples: list[list[float]] = []
    for line in lines:
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        line = line.strip()
        if not line:
            continue
        samples.append(_sample_from_jsonl_record(json.loads(line)))
    if not samples:
        raise ValueError("No IMU samples found in streamed JSONL.")
    return np.asarray(samples, dtype=np.float32)


def _write_shard(path: Path, sample_ids: Sequence[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez(tmp_path, sample_ids=np.asarray(sample_ids, dtype=str), rep__ts2vec=np.asarray(matrix, dtype="float32"))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_manifest(
    path: Path,
    *,
    sample_ids: Sequence[str],
    shards: Sequence[Mapping[str, object]],
    args: Mapping[str, object],
) -> None:
    metadata = {
        "version": "active_embedding_cache_v1",
        "sample_ids": list(sample_ids),
        "representations": ["ts2vec"],
        "sample_rate": 30.0,
        "raw_shape_max_samples": None,
        "representation_options": {
            "ts2vec_checkpoint_path": str(args.get("checkpoint", "")),
            "ts2vec_multiscale_window_sizes": [900, 2700],
            "ts2vec_multiscale_stride_ratio": 0.5,
            "ts2vec_multiscale_pool": "max",
            "ts2vec_max_samples": args.get("max_samples"),
        },
    }
    payload = {
        "metadata": metadata,
        "n_clips": int(len(sample_ids)),
        "n_shards": int(len(shards)),
        "shards": list(shards),
        "gcp_runner": {
            "manifest": args.get("manifest", ""),
            "device": args.get("device", ""),
            "shard_size": args.get("shard_size", ""),
            "encode_batch_size": args.get("encode_batch_size", ""),
            "load_workers": args.get("load_workers", ""),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _completed_shards(manifest_path: Path) -> set[int]:
    if not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed: set[int] = set()
    for shard in payload.get("shards", []):
        if isinstance(shard, Mapping):
            completed.add(int(shard["index"]))
    return completed


def _gcloud_cp(local_path: Path, gcs_path: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", str(local_path), gcs_path], check=True)


if __name__ == "__main__":
    main()
