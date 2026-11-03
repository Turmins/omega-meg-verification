"""Training-only auxiliary head and exponential moving average wrapper."""

from __future__ import annotations

from copy import deepcopy

from omega_dl.models.blocks import _require_torch, nn, torch


if nn is not None:

    class AuxiliaryClassifier(nn.Module):
        def __init__(self, embedding_dim: int = 128, n_classes: int = 40):
            super().__init__()
            self.classifier = nn.Linear(embedding_dim, n_classes)

        def forward(self, z):
            return self.classifier(z)


    class EMA:
        def __init__(self, model: nn.Module, decay: float = 0.999):
            self.decay = decay
            self.model = deepcopy(model).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        @torch.no_grad()
        def update(self, model: nn.Module) -> None:
            for target, source in zip(self.model.parameters(), model.parameters()):
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            for target, source in zip(self.model.buffers(), model.buffers()):
                target.copy_(source)

else:

    class AuxiliaryClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

    class EMA:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()
