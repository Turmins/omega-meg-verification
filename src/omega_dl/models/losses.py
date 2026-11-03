"""Cosine semi-hard triplet and auxiliary classification losses."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object
    F = None


def cross_recording_masks(participant_ids, recording_ids, sessions):
    if torch is None:
        raise RuntimeError("PyTorch is required for mining")
    same_subject = participant_ids[:, None] == participant_ids[None, :]
    different_recording = recording_ids[:, None] != recording_ids[None, :]
    different_session = sessions[:, None] != sessions[None, :]
    positive = same_subject & different_recording & different_session
    negative = ~same_subject
    positive.fill_diagonal_(False)
    negative.fill_diagonal_(False)
    return positive, negative


def semi_hard_triplet_loss(distance: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.2, random_negative: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch is required for the loss")
    values = []
    active = []
    generator = torch.Generator(device=distance.device).manual_seed(0)
    for anchor in range(distance.shape[0]):
        pos = torch.where(positive[anchor])[0]
        neg = torch.where(negative[anchor])[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if random_negative:
            choice = torch.randint(len(neg), (), generator=generator, device=distance.device)
            neg_value = distance[anchor, neg[choice]]
        else:
            candidate = distance[anchor, neg]
            pos_max = distance[anchor, pos].max()
            semi = candidate[(candidate > pos_max) & (candidate < pos_max + margin)]
            neg_value = semi.min() if len(semi) else candidate.min()
        pos_value = distance[anchor, pos].max()
        loss = F.relu(pos_value - neg_value + margin)
        values.append(loss)
        active.append((loss > 0).float())
    if not values:
        return distance.sum() * 0.0, 0.0
    return torch.stack(values).mean(), float(torch.stack(active).mean().item())


def classification_loss(logits: Tensor, labels: Tensor, label_smoothing: float = 0.1):
    if F is None:
        raise RuntimeError("PyTorch is required for classification loss")
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
