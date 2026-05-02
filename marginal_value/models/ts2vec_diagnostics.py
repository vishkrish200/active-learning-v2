from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def run_layer_rank_probe(
    *,
    manifest_path: str,
    n_clips: int = 32,
    clip_len: int = 300,
    device: str = "cuda",
    hidden_dims: int = 64,
    output_dims: int = 320,
    output_path: str | None = None,
    seed: int = 20260430,
    random_start: bool = True,
) -> list[dict[str, float | str]]:
    import json
    import torch

    from marginal_value.models.ts2vec_encoder import TS2VecEncoder

    torch.manual_seed(int(seed))
    batch = load_probe_batch(
        manifest_path,
        n_clips=int(n_clips),
        clip_len=int(clip_len),
        seed=int(seed),
        random_start=bool(random_start),
    )
    encoder = TS2VecEncoder(input_dims=int(hidden_dims), hidden_dims=int(hidden_dims), output_dims=int(output_dims))
    encoder = encoder.to(device)
    results = probe_layer_ranks(encoder, batch, device=device)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    return results


def run_handcrafted_rank_probe(
    *,
    manifest_path: str,
    n_clips: int = 128,
    clip_len: int = 300,
    seed: int = 20260430,
    random_start: bool = True,
    output_path: str | None = None,
) -> list[dict[str, float | str]]:
    import json

    from marginal_value.ranking.baseline_ranker import raw_shape_stats_embedding

    batch = load_probe_batch(
        manifest_path,
        n_clips=int(n_clips),
        clip_len=int(clip_len),
        seed=int(seed),
        random_start=bool(random_start),
    )
    rows = [
        ("raw_mean_std_max", _handcrafted_raw_mean_std_max(batch)),
        ("raw_shape_stats", np.vstack([raw_shape_stats_embedding(clip, sample_rate=30.0) for clip in batch])),
    ]
    results = [_summarize_matrix(name, values) for name, values in rows]

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    return results


def load_probe_batch(
    manifest_path: str | Path,
    *,
    n_clips: int,
    clip_len: int,
    seed: int = 20260430,
    random_start: bool = True,
) -> np.ndarray:
    from marginal_value.data.split_manifest import read_manifest_urls
    from marginal_value.training.train_ts2vec import _load_manifest_clip, _normalize_clip, _pad_to_length

    manifest = Path(manifest_path)
    urls = read_manifest_urls(manifest)
    if not urls:
        raise ValueError(f"No clips found in manifest: {manifest}")

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(urls, size=int(n_clips), replace=len(urls) < int(n_clips))

    clips = []
    for url in selected:
        clip = _normalize_clip(_load_manifest_clip(manifest, url))
        clip = _pad_to_length(clip, int(clip_len))
        if random_start and len(clip) > int(clip_len):
            start = int(rng.integers(0, len(clip) - int(clip_len) + 1))
        else:
            start = 0
        clip = clip[start : start + int(clip_len)]
        clips.append(clip.astype("float32", copy=False))
    if len(clips) < int(n_clips):
        raise ValueError(f"Requested {n_clips} clips, only loaded {len(clips)}.")
    return np.stack(clips).astype("float32", copy=False)


def probe_layer_ranks(encoder: Any, batch: np.ndarray, *, device: str = "cuda") -> list[dict[str, float | str]]:
    """Measure clip-level effective rank at each major TS2Vec encoder stage."""
    import torch

    encoder.eval()
    values = torch.tensor(np.asarray(batch, dtype=np.float32), dtype=torch.float32, device=device)
    activations: dict[str, Any] = {}
    hooks = []
    activations["raw_input"] = values.detach().cpu()
    if values.shape[-1] == 6:
        from marginal_value.models.ts2vec_encoder import expand_imu_features

        activations["expanded_input"] = expand_imu_features(values).detach().cpu()

    def remember(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            activations[name] = output.detach().cpu()

        return hook

    modules: list[tuple[str, Any]] = [
        ("input_projection", encoder.input_projection),
        ("hidden_projection", encoder.hidden_projection),
    ]
    modules.extend((f"block_{idx:02d}", block) for idx, block in enumerate(encoder.blocks))
    modules.append(("output_projection", encoder.output_projection))
    modules.append(("aggregation_projection", encoder.aggregation_head.projection))

    try:
        for name, module in modules:
            hooks.append(module.register_forward_hook(remember(name)))
        with torch.no_grad():
            final = encoder(values)
        activations["aggregation_output"] = final.detach().cpu()
    finally:
        for hook in hooks:
            hook.remove()

    results: list[dict[str, float | str]] = []
    for name in _ordered_names(modules, include_final=True):
        if name not in activations:
            continue
        results.append(_summarize_activation(name, activations[name], pooling="mean"))
        if activations[name].ndim == 3:
            results.append(_summarize_activation(f"{name}:stats", activations[name], pooling="stats"))
    return results


def _ordered_names(modules: Iterable[tuple[str, Any]], *, include_final: bool) -> list[str]:
    names = ["raw_input", "expanded_input"]
    names.extend(name for name, _module in modules)
    if include_final:
        names.append("aggregation_output")
    return names


def _summarize_activation(name: str, activation: Any, *, pooling: str) -> dict[str, float | str]:
    import torch

    values = activation.float()
    if values.ndim == 3:
        if pooling == "mean":
            pooled = values.mean(dim=1)
        elif pooling == "stats":
            pooled = torch.cat(
                [
                    values.mean(dim=1),
                    values.max(dim=1).values,
                    values.std(dim=1, unbiased=False),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
    elif values.ndim == 2:
        pooled = values
    else:
        raise ValueError(f"Unsupported activation shape for {name}: {tuple(values.shape)}")

    pooled_np = pooled.numpy().astype("float64", copy=False)
    return _summarize_matrix(name, pooled_np) | {
        "activation_std": round(float(values.std(unbiased=False).item()), 6),
        "activation_mean_abs": round(float(values.abs().mean().item()), 6),
    }


def _summarize_matrix(name: str, values: np.ndarray) -> dict[str, float | str]:
    matrix = np.asarray(values, dtype="float64")
    return {
        "layer": name,
        "effective_rank": round(_effective_rank(matrix), 4),
        "mean_pairwise_cosine": round(_mean_pairwise_cosine(matrix), 4),
    }


def _handcrafted_raw_mean_std_max(batch: np.ndarray) -> np.ndarray:
    values = np.asarray(batch, dtype="float64")
    return np.concatenate(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ],
        axis=1,
    )


def _effective_rank(values: np.ndarray) -> float:
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values * singular_values
    probabilities = power / max(float(np.sum(power)), 1.0e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1.0e-12)))
    return float(np.exp(entropy))


def _mean_pairwise_cosine(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 1.0
    normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1.0e-12)
    similarities = normalized @ normalized.T
    mask = ~np.eye(len(values), dtype=bool)
    return float(np.mean(similarities[mask]))
