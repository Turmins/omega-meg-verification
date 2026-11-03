"""Optional convolutional encoder-decoder feature extractor."""

from __future__ import annotations

from omega_dl.models.blocks import _require_torch, nn, torch


if nn is not None:

    class ConvAutoencoder(nn.Module):
        def __init__(self, n_channels: int, bottleneck_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, 64, 7, stride=4, padding=3), nn.GELU(),
                nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.GELU(),
            )
            self.to_code = nn.Linear(128 * 80, bottleneck_dim)
            self.from_code = nn.Linear(bottleneck_dim, 128 * 80)
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2), nn.GELU(),
                nn.ConvTranspose1d(64, n_channels, 8, stride=4, padding=2),
            )

        def encode(self, x):
            hidden = self.encoder(x.float())
            return self.to_code(hidden.flatten(1))

        def forward(self, x):
            code = self.encode(x)
            hidden = self.from_code(code).view(x.shape[0], 128, 80)
            return self.decoder(hidden), code

else:

    class ConvAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()
