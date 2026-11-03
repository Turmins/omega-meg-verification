"""Siamese 1D CNN encoder used for open-set biometric embeddings."""

from __future__ import annotations

from typing import Sequence

from omega_dl.models.blocks import SpatialProjection, StatisticsPooling, TemporalBlock, _require_torch, nn, torch


if nn is not None:

    class SiameseCNN(nn.Module):
        def __init__(
            self,
            n_bands: int = 3,
            channels: int = 260,
            spatial_per_band: int = 40,
            temporal_widths: Sequence[int] = (96, 128, 128, 160),
            kernel: int = 7,
            dilations: Sequence[int] = (1, 1, 2, 4),
            pool: int = 4,
            dropout: Sequence[float] = (0.10, 0.10, 0.15, 0.15),
            embedding_dim: int = 128,
        ):
            super().__init__()
            if len(temporal_widths) != 4 or len(dilations) != 4 or len(dropout) != 4:
                raise ValueError("The encoder requires four temporal blocks")
            self.n_bands = n_bands
            self.spatial = SpatialProjection(n_bands, channels, spatial_per_band)
            layers = []
            in_channels = n_bands * spatial_per_band
            for width, dilation, drop in zip(temporal_widths, dilations, dropout):
                layers.append(TemporalBlock(in_channels, width, kernel, dilation, pool, drop))
                in_channels = width
            self.temporal = nn.Sequential(*layers)
            self.pooling = StatisticsPooling()
            self.embedding = nn.Sequential(
                nn.Linear(2 * temporal_widths[-1], embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )

        def forward(self, x):
            x = x.float()
            z = self.embedding(self.pooling(self.temporal(self.spatial(x))))
            return torch.nn.functional.normalize(z, dim=1)

else:

    class SiameseCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()
