from __future__ import annotations


def expand_imu_features(values):
    """Expand raw acc/gyro channels with physically meaningful IMU features."""
    import torch

    if values.ndim != 3:
        raise ValueError("IMU feature expansion input must be shaped (B, T, C).")
    if values.shape[-1] != 6:
        raise ValueError(f"Expected 6 IMU channels for feature expansion, got {values.shape[-1]}.")
    acc = values[:, :, :3]
    gyro = values[:, :, 3:]
    acc_mag = torch.linalg.norm(acc, dim=-1, keepdim=True)
    gyro_mag = torch.linalg.norm(gyro, dim=-1, keepdim=True)
    cross = torch.linalg.cross(acc, gyro, dim=-1)
    acc_centered = acc - torch.mean(acc, dim=1, keepdim=True)
    return torch.cat([values, acc_mag, gyro_mag, cross, acc_centered], dim=-1)


class AggregationHead:
    """Mean/max/std temporal aggregation projected to a clip embedding."""

    def __new__(
        cls,
        *,
        hidden_dims: int = 64,
        output_dims: int = 320,
    ):
        import torch
        import torch.nn as nn

        class _AggregationHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if hidden_dims <= 0 or output_dims <= 0:
                    raise ValueError("hidden_dims and output_dims must be positive.")
                self.hidden_dims = int(hidden_dims)
                self.output_dims = int(output_dims)
                self.projection = nn.Linear(3 * int(hidden_dims), int(output_dims))

            def forward(self, values):
                if values.ndim != 3:
                    raise ValueError("AggregationHead input must be shaped (B, T, D).")
                mean = torch.mean(values, dim=1)
                maximum = torch.max(values, dim=1).values
                std = torch.std(values, dim=1, unbiased=False)
                return self.projection(torch.cat([mean, maximum, std], dim=-1))

        return _AggregationHead()


class NonlinearInputProjection:
    """Pointwise nonlinear IMU projection before the temporal encoder."""

    def __new__(
        cls,
        *,
        input_channels: int = 6,
        input_dims: int = 64,
        expand_imu: bool = True,
    ):
        import torch.nn as nn

        class _NonlinearInputProjection(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if input_channels <= 0 or input_dims <= 0:
                    raise ValueError("input_channels and input_dims must be positive.")
                self.input_channels = int(input_channels)
                self.input_dims = int(input_dims)
                self.expand_imu = bool(expand_imu and input_channels == 6)
                projection_channels = 14 if self.expand_imu else int(input_channels)
                self.projection = nn.Sequential(
                    nn.Linear(projection_channels, int(input_dims)),
                    nn.GELU(),
                    nn.Linear(int(input_dims), int(input_dims)),
                )

            def forward(self, values):
                if values.ndim != 3:
                    raise ValueError("NonlinearInputProjection input must be shaped (B, T, C).")
                if values.shape[-1] != self.input_channels:
                    raise ValueError(f"Expected {self.input_channels} input channels, got {values.shape[-1]}.")
                if self.expand_imu:
                    values = expand_imu_features(values)
                return self.projection(values)

        return _NonlinearInputProjection()


class TS2VecEncoder:
    """Dilated causal temporal CNN encoder for TS2Vec-style IMU pretraining."""

    def __new__(
        cls,
        *,
        input_dims: int = 64,
        hidden_dims: int = 64,
        output_dims: int = 320,
        n_layers: int = 10,
        input_channels: int = 6,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _CausalConvBlock(nn.Module):
            def __init__(self, channels: int, dilation: int) -> None:
                super().__init__()
                self.left_padding = int(dilation * 2)
                self.conv = nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    dilation=int(dilation),
                    padding=0,
                )
                self.activation = nn.GELU()
                self.norm = nn.LayerNorm(channels)

            def forward(self, values):
                residual = values
                x = values.transpose(1, 2)
                x = F.pad(x, (self.left_padding, 0))
                x = self.conv(x).transpose(1, 2)
                x = self.activation(x)
                return self.norm(residual + x)

        class _TS2VecEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if input_channels <= 0:
                    raise ValueError("input_channels must be positive.")
                if input_dims <= 0 or hidden_dims <= 0 or output_dims <= 0:
                    raise ValueError("input_dims, hidden_dims, and output_dims must be positive.")
                if n_layers <= 0:
                    raise ValueError("n_layers must be positive.")
                self.input_dims = int(input_dims)
                self.hidden_dims = int(hidden_dims)
                self.output_dims = int(output_dims)
                self.n_layers = int(n_layers)
                self.input_channels = int(input_channels)
                self.input_projection = NonlinearInputProjection(
                    input_channels=input_channels,
                    input_dims=input_dims,
                )
                self.hidden_projection = (
                    nn.Identity()
                    if input_dims == hidden_dims
                    else nn.Linear(input_dims, hidden_dims)
                )
                self.blocks = nn.ModuleList(
                    _CausalConvBlock(hidden_dims, dilation=2**idx)
                    for idx in range(n_layers)
                )
                self.output_projection = nn.Linear(hidden_dims, output_dims)
                self.aggregation_head = AggregationHead(hidden_dims=hidden_dims, output_dims=output_dims)

            def _encode_hidden(self, values):
                if values.ndim != 3:
                    raise ValueError("TS2VecEncoder input must be shaped (B, T, C).")
                if values.shape[-1] != self.input_channels:
                    raise ValueError(f"Expected {self.input_channels} input channels, got {values.shape[-1]}.")
                hidden = self.hidden_projection(self.input_projection(values))
                layers = []
                for block in self.blocks:
                    hidden = block(hidden)
                    layers.append(hidden)
                return hidden, layers

            def encode_sequence(self, values, *, return_layers: bool = False):
                hidden, layers = self._encode_hidden(values)
                output = self.output_projection(hidden)
                if return_layers:
                    return [*layers, output]
                return output

            def forward(self, values, *, return_layers: bool = False):
                if return_layers:
                    return self.encode_sequence(values, return_layers=True)
                hidden, _layers = self._encode_hidden(values)
                return self.aggregation_head(hidden)

        return _TS2VecEncoder()
