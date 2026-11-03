"""EEGNet-inspired signal encoder and edge-aware graph message passing."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class EEGNetEncoder(nn.Module):
    """Three-band EEGNet adaptation with a normalized metric-learning output."""
    def __init__(self, embedding_dim=64, temporal_filters=8, depth_multiplier=2,
                 pointwise_filters=16, temporal_kernel=33, dropout=0.25):
        super().__init__()
        width = temporal_filters * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(3, temporal_filters, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), stride=(1, 2), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(temporal_filters, width, (260, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(width), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(dropout),
            nn.Conv2d(width, width, (1, 17), padding=(0, 8), groups=width, bias=False),
            nn.Conv2d(width, pointwise_filters, 1, bias=False),
            nn.BatchNorm2d(pointwise_filters), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.AvgPool2d((1, 5)), nn.Flatten(),
        )
        self.project = nn.Linear(pointwise_filters * 8, embedding_dim)

    def forward(self, signal):
        if signal.ndim != 4 or signal.shape[1:] != (3, 260, 1280):
            raise ValueError("Expected batch x 3 bands x 260 sensors x 1280 samples")
        return F.normalize(self.project(self.features(signal)), dim=-1)


class MessageLayer(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Sequential(nn.Linear(9, hidden), nn.ELU(), nn.Linear(hidden, hidden))
        self.update = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, neighbors, edges):
        batch = torch.arange(h.shape[0], device=h.device)[:, None, None]
        other = h[batch, neighbors.long()]
        messages = F.elu(self.neighbor(other) + self.edge(edges)).mean(dim=2)
        change = self.dropout(F.elu(self.update(torch.cat((h, messages), dim=-1))))
        return F.elu(self.norm(h + change))


class GraphEncoder(nn.Module):
    """Sensor identity features retain channel correspondence across graphs."""
    def __init__(self, embedding_dim=64, hidden=32, sensor_embedding=8,
                 layers=2, dropout=0.15):
        super().__init__()
        self.sensor = nn.Embedding(260, sensor_embedding)
        self.input = nn.Linear(12 + sensor_embedding, hidden)
        self.layers = nn.ModuleList(MessageLayer(hidden, dropout) for _ in range(layers))
        self.project = nn.Linear(2 * hidden, embedding_dim)

    def forward(self, nodes, neighbors, edges):
        if nodes.ndim != 3 or nodes.shape[1:] != (260, 12):
            raise ValueError("Expected batch x 260 sensors x 12 node features")
        if neighbors.shape[:2] != nodes.shape[:2] or edges.shape != (*neighbors.shape, 9):
            raise ValueError("Incompatible graph edge shapes")
        identity = self.sensor.weight.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        h = F.elu(self.input(torch.cat((nodes, identity), dim=-1)))
        for layer in self.layers:
            h = layer(h, neighbors, edges)
        pooled = torch.cat((h.mean(dim=1), h.std(dim=1, unbiased=False)), dim=-1)
        return F.normalize(self.project(pooled), dim=-1)


def make_model(name, config):
    if name == "eegnet":
        return EEGNetEncoder(config["embedding_dim"], **config["eegnet"])
    if name == "graph":
        return GraphEncoder(config["embedding_dim"], **config["graph"])
    raise ValueError(f"Unknown encoder: {name}")


def prototype_loss(z, subjects, windows, temperature):
    """Support and query are always different sessions; optimize both directions."""
    if z.shape[0] != subjects * 2 * windows or subjects < 2 or temperature <= 0:
        raise ValueError("Invalid cross-session training episode")
    z = z.reshape(subjects, 2, windows, -1)
    labels = torch.arange(subjects, device=z.device).repeat_interleave(windows)
    losses = []
    for support, query in ((0, 1), (1, 0)):
        prototypes = F.normalize(z[:, support].mean(dim=1), dim=-1)
        queries = z[:, query].reshape(subjects * windows, -1)
        losses.append(F.cross_entropy(queries @ prototypes.T / temperature, labels))
    return sum(losses) / 2

