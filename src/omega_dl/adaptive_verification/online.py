"""Label-free, score-before-update verification with serializable templates."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from omega_dl.verification.scoring import normalize


@dataclass
class AdaptiveTemplate:
    anchor: np.ndarray
    threshold: float
    update_threshold: float
    policy: str = "guarded"
    alpha: float = 0.05
    anchor_weight: float = 0.1
    current: np.ndarray | None = None
    updates: int = 0
    seen: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.policy not in {"static", "accepted", "guarded"}:
            raise ValueError("Unknown update policy")
        if not 0 < self.alpha <= 1 or not 0 <= self.anchor_weight <= 1:
            raise ValueError("Invalid update coefficients")
        if not np.isfinite([self.threshold, self.update_threshold]).all() or self.update_threshold < self.threshold:
            raise ValueError("Update threshold must be finite and at least acceptance threshold")
        self.anchor = normalize(self.anchor).copy()
        if self.anchor.ndim != 1:
            raise ValueError("A template must be a vector")
        self.current = self.anchor.copy() if self.current is None else normalize(self.current).copy()
        if self.current.shape != self.anchor.shape:
            raise ValueError("Template dimensions differ")

    def verify(self, embedding, event_id=None):
        if event_id is not None and event_id in self.seen:
            raise ValueError("This event was already consumed")
        z = normalize(embedding)
        if z.shape != self.anchor.shape:
            raise ValueError("Query and template dimensions differ")
        score, anchor_score = float(self.current @ z), float(self.anchor @ z)
        accepted = score >= self.threshold
        updated = bool(accepted and (self.policy == "accepted" or (
            self.policy == "guarded" and score >= self.update_threshold
            and anchor_score >= self.update_threshold)))
        if updated:
            candidate = normalize((1 - self.alpha) * self.current + self.alpha * z)
            if self.policy == "guarded":
                candidate = normalize((1 - self.anchor_weight) * candidate + self.anchor_weight * self.anchor)
            self.current = candidate
            self.updates += 1
        if event_id is not None:
            self.seen.add(event_id)
        return {"score": score, "anchor_score": anchor_score, "accepted": bool(accepted),
                "updated": updated, "updates_so_far": self.updates,
                "drift_from_anchor": float(1 - np.clip(self.current @ self.anchor, -1, 1))}

    def state(self):
        return {"anchor": self.anchor.tolist(), "current": self.current.tolist(),
                "threshold": self.threshold, "update_threshold": self.update_threshold,
                "policy": self.policy, "alpha": self.alpha, "anchor_weight": self.anchor_weight,
                "updates": self.updates, "seen": sorted(self.seen)}

    @classmethod
    def restore(cls, value):
        return cls(**{**value, "anchor": np.asarray(value["anchor"]),
                     "current": np.asarray(value["current"]), "seen": set(value["seen"])})


def strict_update_threshold(labels, scores, acceptance, far):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not 0 <= far < 1 or not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes and finite development scores are required")
    negative = np.sort(scores[labels == 0])[::-1]
    budget = int(np.floor(far * len(negative)))
    boundary = float(np.nextafter(negative[budget], np.inf))
    return max(float(acceptance), boundary)
