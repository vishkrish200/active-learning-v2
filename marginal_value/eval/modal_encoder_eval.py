from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from marginal_value.data.split_manifest import build_split_manifest, select_split
from marginal_value.eval.encoder_eval import evaluate_retrieval, write_eval_report
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.ssl_encoder import build_ssl_encoder, encoder_config_from_training
from marginal_value.training.feature_scaler import load_feature_scaler


def run_encoder_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    import torch

    log_event("encoder_eval", "start", smoke=smoke)
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_event("encoder_eval", "manifest_load_start", root=str(data_root))
    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    support_rows = select_split(manifest, config["splits"]["support_split"])
    query_rows = select_split(manifest, config["splits"]["query_split"])

    if smoke:
        support_rows = support_rows[: int(config["eval"].get("smoke_support_samples", 128))]
        query_rows = query_rows[: int(config["eval"].get("smoke_query_samples", 64))]
    log_event(
        "encoder_eval",
        "manifest_ready",
        n_manifest=len(manifest),
        n_support=len(support_rows),
        n_query=len(query_rows),
    )

    checkpoint_path = Path(config["checkpoint"]["path"])
    log_event("encoder_eval", "checkpoint_load_start", checkpoint_path=str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("data", {}).get("train_split") != "pretrain":
        raise RuntimeError("Checkpoint was not trained with train_split='pretrain'.")
    if checkpoint_config.get("data", {}).get("holdout_split") != "val":
        raise RuntimeError("Checkpoint does not preserve val as holdout_split.")

    d_model = int(checkpoint_config.get("training", {}).get("d_model", 128))
    feature_dim = int(config["data"]["feature_dim"])
    checkpoint_encoder_config = checkpoint.get("encoder_config") or encoder_config_from_training(checkpoint_config)
    architecture = str(checkpoint_encoder_config.get("architecture", "legacy_reconstruction_mlp"))
    model = _build_model(
        feature_dim=feature_dim,
        d_model=d_model,
        encoder_config=checkpoint_encoder_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = load_feature_scaler(checkpoint.get("feature_scaler"), feature_dim=feature_dim)
    log_event(
        "encoder_eval",
        "model_ready",
        device=str(device),
        feature_dim=feature_dim,
        d_model=d_model,
        architecture=architecture,
        scaler_files=scaler.n_files if scaler is not None else 0,
        scaler_windows=scaler.n_windows if scaler is not None else 0,
    )

    support_encoder, support_baseline = _embed_rows(model, support_rows, device=device, scaler=scaler, label="support")
    query_encoder, query_baseline = _embed_rows(model, query_rows, device=device, scaler=scaler, label="query")

    k_values = [int(k) for k in config["eval"]["k_values"]]
    log_event("encoder_eval", "metrics_start", k_values=k_values)
    report = evaluate_retrieval(
        support_encoder=support_encoder,
        query_encoder=query_encoder,
        support_baseline=support_baseline,
        query_baseline=query_baseline,
        k_values=k_values,
    )
    report["mode"] = "smoke" if smoke else "full"
    report["support_split"] = config["splits"]["support_split"]
    report["query_split"] = config["splits"]["query_split"]
    report["checkpoint_path"] = str(checkpoint_path)
    report["encoder_architecture"] = architecture
    report["acceptance"] = _acceptance_report(report, config)

    suffix = "smoke" if smoke else "full"
    np.save(output_dir / f"support_encoder_embeddings_{suffix}.npy", support_encoder)
    np.save(output_dir / f"query_encoder_embeddings_{suffix}.npy", query_encoder)
    np.save(output_dir / f"support_baseline_embeddings_{suffix}.npy", support_baseline)
    np.save(output_dir / f"query_baseline_embeddings_{suffix}.npy", query_baseline)
    _write_rows(output_dir / f"support_manifest_{suffix}.csv", support_rows)
    _write_rows(output_dir / f"query_manifest_{suffix}.csv", query_rows)
    write_eval_report(report, output_dir / f"encoder_eval_report_{suffix}.json")
    log_event("encoder_eval", "artifacts_written", output_dir=str(output_dir), suffix=suffix)

    result = {
        "mode": report["mode"],
        "n_support": report["n_support"],
        "n_query": report["n_query"],
        "encoder_mean_knn_d1": report["encoder"]["mean_knn_d1"],
        "baseline_mean_knn_d1": report["baseline"]["mean_knn_d1"],
        "encoder_effective_rank": report["encoder"]["diagnostics"]["effective_rank"],
        "encoder_mean_pairwise_cosine_distance": report["encoder"]["diagnostics"]["mean_pairwise_cosine_distance"],
        "acceptance_passed": report["acceptance"]["passed"],
        "report_path": str(output_dir / f"encoder_eval_report_{suffix}.json"),
    }
    log_event("encoder_eval", "done", **result)
    return result


def _build_model(*, feature_dim: int, d_model: int, encoder_config: dict[str, Any]):
    import torch.nn as nn

    architecture = str(encoder_config.get("architecture", "legacy_reconstruction_mlp"))
    if architecture == "normalized_vicreg_mlp":
        return build_ssl_encoder(
            feature_dim=feature_dim,
            d_model=d_model,
            embedding_dim=int(encoder_config.get("embedding_dim", 128)),
            dropout=float(encoder_config.get("dropout", 0.1)),
        )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, feature_dim),
            )

        def forward(self, values):
            return self.net(values)

        def encode(self, values):
            hidden = self.net[:4](values)
            return hidden.mean(dim=1)

    return Model()


def _embed_rows(model, rows, *, device, scaler, label: str):
    import torch

    encoder_embeddings = []
    baseline_embeddings = []
    total = len(rows)
    log_event("encoder_eval", "embed_start", label=label, total=total)
    progress_every = max(1, total // 10)
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            with np.load(row.feature_path) as data:
                window_features = np.asarray(data["window_features"], dtype="float32")
                clip_features = np.asarray(data["clip_features"], dtype="float32")
            if scaler is not None:
                window_features = scaler.transform(window_features)
            values = torch.tensor(window_features[None, :, :], dtype=torch.float32, device=device)
            embedding = model.encode(values).detach().cpu().numpy()[0]
            encoder_embeddings.append(embedding)
            baseline_embeddings.append(clip_features)
            log_progress("encoder_eval", "embed_progress", index=index, total=total, every=progress_every, label=label)
    log_event("encoder_eval", "embed_done", label=label, total=total)
    return np.vstack(encoder_embeddings), np.vstack(baseline_embeddings)


def _acceptance_report(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    acceptance_config = config.get("acceptance", {})
    if not isinstance(acceptance_config, dict) or not acceptance_config:
        return {"enabled": False, "passed": True, "checks": {}}
    diagnostics = report["encoder"]["diagnostics"]
    checks = {
        "min_effective_rank": {
            "threshold": float(acceptance_config.get("min_effective_rank", 0.0)),
            "value": float(diagnostics["effective_rank"]),
        },
        "min_mean_pairwise_cosine_distance": {
            "threshold": float(acceptance_config.get("min_mean_pairwise_cosine_distance", 0.0)),
            "value": float(diagnostics["mean_pairwise_cosine_distance"]),
        },
    }
    for check in checks.values():
        check["passed"] = bool(check["value"] >= check["threshold"])
    return {
        "enabled": True,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
        "checks": checks,
    }


def _write_rows(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "url", "raw_path", "feature_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "url": row.url,
                    "raw_path": str(row.raw_path),
                    "feature_path": str(row.feature_path),
                }
            )
