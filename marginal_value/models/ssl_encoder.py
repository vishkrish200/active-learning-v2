from __future__ import annotations

from typing import Any


def build_ssl_encoder(
    *,
    feature_dim: int,
    d_model: int,
    embedding_dim: int,
    dropout: float = 0.1,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NormalizedVicregEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            self.temporal = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.post_temporal = nn.LayerNorm(d_model)
            self.reconstruction_head = nn.Linear(d_model, feature_dim)
            self.embedding_head = nn.Sequential(
                nn.Linear(d_model, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
            )

        def hidden_tokens(self, values):
            hidden = self.input(values)
            temporal = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
            return self.post_temporal(hidden + temporal)

        def forward(self, values):
            return self.reconstruct(values)

        def reconstruct(self, values):
            return self.reconstruction_head(self.hidden_tokens(values))

        def encode(self, values):
            hidden = self.hidden_tokens(values)
            pooled = hidden.mean(dim=1)
            embedding = self.embedding_head(pooled)
            return F.normalize(embedding, dim=-1)

    return NormalizedVicregEncoder()


def encoder_config_from_training(config: dict[str, Any]) -> dict[str, Any]:
    encoder = config.get("encoder", {})
    if not isinstance(encoder, dict):
        encoder = {}
    return {
        "architecture": str(encoder.get("architecture", "normalized_vicreg_mlp")),
        "embedding_dim": int(encoder.get("embedding_dim", 128)),
        "dropout": float(encoder.get("dropout", 0.1)),
        "normalization": dict(encoder.get("normalization", {"enabled": True})),
        "losses": dict(encoder.get("losses", {})),
        "augmentation": dict(encoder.get("augmentation", {})),
    }
