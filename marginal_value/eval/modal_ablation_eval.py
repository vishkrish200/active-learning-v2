from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from marginal_value.data.split_manifest import build_split_manifest, select_split
from marginal_value.eval.ablation_eval import summarize_ranked_scores
from marginal_value.eval.encoder_eval import cosine_knn
from marginal_value.eval.modal_encoder_eval import _build_model
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.submit.make_submission import diversity_rerank


def run_ablation_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    import torch

    log_event("ablation_eval", "start", smoke=smoke)
    rng = np.random.default_rng(int(config["eval"].get("seed", 17)))
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_event("ablation_eval", "manifest_load_start", root=str(data_root))
    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    support_rows = select_split(manifest, config["splits"]["support_split"])
    positive_rows = select_split(manifest, config["splits"]["positive_split"])
    negative_rows = select_split(manifest, config["splits"]["negative_split"])

    if smoke:
        support_rows = support_rows[: int(config["eval"].get("smoke_support_samples", 256))]
        positive_rows = positive_rows[: int(config["eval"].get("smoke_positive_samples", 64))]
        negative_rows = negative_rows[-int(config["eval"].get("smoke_negative_samples", 64)) :]
    else:
        negative_size = min(int(config["eval"].get("negative_sample_size", len(positive_rows))), len(negative_rows))
        negative_indices = rng.choice(len(negative_rows), size=negative_size, replace=False)
        negative_rows = [negative_rows[int(idx)] for idx in negative_indices]
    log_event(
        "ablation_eval",
        "manifest_ready",
        n_manifest=len(manifest),
        n_support=len(support_rows),
        n_positive=len(positive_rows),
        n_negative=len(negative_rows),
    )

    log_event("ablation_eval", "checkpoint_load_start", checkpoint_path=config["checkpoint"]["path"])
    checkpoint = torch.load(config["checkpoint"]["path"], map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("data", {}).get("train_split") != "pretrain":
        raise RuntimeError("Ablation eval requires the pretrain-only checkpoint.")

    model = _load_encoder_model(checkpoint, feature_dim=int(config["data"]["feature_dim"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log_event("ablation_eval", "model_ready", device=str(device))

    support = _load_embeddings(model, support_rows, device=device, label="support")
    positives = _load_embeddings(model, positive_rows, device=device, label="positive")
    negatives = _load_embeddings(model, negative_rows, device=device, label="negative")

    log_event("ablation_eval", "candidate_combine_start")
    candidates = _combine_candidates(positives, negatives)
    k_values = [int(k) for k in config["eval"]["k_values"]]
    report = {
        "mode": "smoke" if smoke else "full",
        "n_support": len(support_rows),
        "n_positive": len(positive_rows),
        "n_negative": len(negative_rows),
        "support_split": config["splits"]["support_split"],
        "positive_split": config["splits"]["positive_split"],
        "negative_split": config["splits"]["negative_split"],
        "baselines": {},
    }

    log_event("ablation_eval", "baseline_scores_start", n_candidates=len(candidates["labels"]))
    baseline_scores = {
        "random": rng.random(len(candidates["labels"])),
        "clip_features_knn": _novelty_scores(support["clip"], candidates["clip"]),
        "window_mean_pool_knn": _novelty_scores(support["window_mean"], candidates["window_mean"]),
        "window_mean_std_pool_knn": _novelty_scores(support["window_mean_std"], candidates["window_mean_std"]),
        "encoder_hidden_knn": _novelty_scores(support["encoder"], candidates["encoder"]),
        "encoder_reconstruction_error": candidates["reconstruction_error"],
        "clip_knn_plus_candidate_density": _combine_novelty_density(
            _novelty_scores(support["clip"], candidates["clip"]),
            candidates["clip"],
        ),
    }

    for name, scores in baseline_scores.items():
        log_event("ablation_eval", "baseline_metric_start", baseline=name)
        report["baselines"][name] = summarize_ranked_scores(
            scores=np.asarray(scores, dtype=float),
            labels=candidates["labels"],
            clusters=candidates["clusters"],
            k_values=k_values,
        )

    mmr_rows = [
        {"worker_id": str(idx), "final_score": float(score)}
        for idx, score in enumerate(baseline_scores["clip_knn_plus_candidate_density"])
    ]
    reranked = diversity_rerank(mmr_rows, candidates["clip"], lambda_redundancy=0.25)
    mmr_order = np.array([int(row["worker_id"]) for row in reranked], dtype=int)
    report["baselines"]["clip_knn_density_mmr"] = summarize_ranked_scores(
        scores=np.arange(len(mmr_order), 0, -1),
        labels=candidates["labels"][mmr_order],
        clusters=candidates["clusters"][mmr_order],
        k_values=k_values,
    )

    output_suffix = "smoke" if smoke else "full"
    report_path = output_dir / f"ablation_report_{output_suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_candidate_scores(output_dir / f"candidate_scores_{output_suffix}.csv", candidates, baseline_scores)
    log_event("ablation_eval", "artifacts_written", output_dir=str(output_dir), suffix=output_suffix)

    best = max(
        report["baselines"].items(),
        key=lambda item: item[1].get("ndcg@100", item[1].get("ndcg@50", 0.0)),
    )
    result = {
        "mode": report["mode"],
        "n_support": report["n_support"],
        "n_positive": report["n_positive"],
        "n_negative": report["n_negative"],
        "best_by_ndcg100": best[0],
        "best_ndcg100": best[1].get("ndcg@100", best[1].get("ndcg@50", 0.0)),
        "report_path": str(report_path),
    }
    log_event("ablation_eval", "done", **result)
    return result


def _load_encoder_model(checkpoint: dict[str, Any], *, feature_dim: int):
    training = checkpoint.get("config", {}).get("training", {})
    model = _build_model(feature_dim=feature_dim, d_model=int(training.get("d_model", 128)))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _load_embeddings(model, rows, *, device, label: str) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    clip = []
    window_mean = []
    window_mean_std = []
    encoder = []
    reconstruction = []
    total = len(rows)
    log_event("ablation_eval", "embed_start", label=label, total=total)
    progress_every = max(1, total // 10)
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            with np.load(row.feature_path) as data:
                clip_features = np.asarray(data["clip_features"], dtype="float32")
                window_features = np.asarray(data["window_features"], dtype="float32")
            values = torch.tensor(window_features[None, :, :], dtype=torch.float32, device=device)
            prediction = model(values)
            reconstruction.append(float(F.smooth_l1_loss(prediction, values).detach().cpu()))
            clip.append(clip_features)
            window_mean.append(np.mean(window_features, axis=0))
            window_mean_std.append(np.concatenate([np.mean(window_features, axis=0), np.std(window_features, axis=0)]))
            encoder.append(model.encode(values).detach().cpu().numpy()[0])
            log_progress("ablation_eval", "embed_progress", index=index, total=total, every=progress_every, label=label)
    log_event("ablation_eval", "embed_done", label=label, total=total)
    return {
        "clip": np.vstack(clip),
        "window_mean": np.vstack(window_mean),
        "window_mean_std": np.vstack(window_mean_std),
        "encoder": np.vstack(encoder),
        "reconstruction_error": np.asarray(reconstruction, dtype=float),
    }


def _combine_candidates(positives: dict[str, np.ndarray], negatives: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    combined = {
        key: np.vstack([positives[key], negatives[key]])
        for key in ("clip", "window_mean", "window_mean_std", "encoder")
    }
    labels = np.concatenate([np.ones(len(positives["clip"]), dtype=int), np.zeros(len(negatives["clip"]), dtype=int)])
    combined["labels"] = labels
    combined["clusters"] = _simple_clusters(combined["clip"])
    combined["reconstruction_error"] = np.concatenate(
        [positives["reconstruction_error"], negatives["reconstruction_error"]]
    )
    return combined


def _novelty_scores(support: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    distances, _indices = cosine_knn(support, candidates, k=min(10, len(support)))
    return np.mean(distances, axis=1)


def _combine_novelty_density(novelty: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, np.nan)
    density = np.nanmean(np.maximum(sims, 0.0), axis=1)
    density = np.nan_to_num(density, nan=0.0)
    novelty_norm = _minmax(novelty)
    density_norm = _minmax(density)
    return 0.75 * novelty_norm + 0.25 * density_norm


def _simple_clusters(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int)
    dominant = np.argmax(np.abs(values), axis=1)
    signs = (values[np.arange(len(values)), dominant] >= 0).astype(int)
    return dominant * 2 + signs


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values)) if len(values) else 0.0
    if span < 1.0e-12:
        return np.zeros_like(values)
    return (values - float(np.min(values))) / span


def _write_candidate_scores(path: Path, candidates: dict[str, np.ndarray], baseline_scores: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["candidate_index", "label", "cluster"] + sorted(baseline_scores)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, label in enumerate(candidates["labels"]):
            row = {
                "candidate_index": idx,
                "label": int(label),
                "cluster": int(candidates["clusters"][idx]),
            }
            for name, scores in baseline_scores.items():
                row[name] = float(scores[idx])
            writer.writerow(row)
