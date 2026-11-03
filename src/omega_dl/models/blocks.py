"""Reusable convolutional blocks."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for model construction")


if nn is not None:

    class SpatialProjection(nn.Module):
        def __init__(self, n_bands: int, channels: int = 260, spatial_per_band: int = 40):
            super().__init__()
            self.n_bands = n_bands
            self.channels = channels
            self.projection = nn.Conv1d(n_bands * channels, n_bands * spatial_per_band, 1, groups=n_bands, bias=False)
            self.norm = nn.BatchNorm1d(n_bands * spatial_per_band)

        def forward(self, x):
            return self.norm(self.projection(x))


    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, pool: int, dropout: float):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.MaxPool1d(pool),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.block(x)


    class StatisticsPooling(nn.Module):
        def forward(self, x):
            return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=1)

else:

    class SpatialProjection:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class TemporalBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class StatisticsPooling:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()
