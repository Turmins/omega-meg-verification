"""Pairwise verification scores, development-only thresholds and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms < 1e-12) or not np.isfinite(x).all():
        raise ValueError("Invalid or zero template")
    return x / norms


def phase_template(edges):
    return np.exp(1j * np.angle(np.asarray(edges, dtype=np.complex128).mean(axis=0)))


def score_frame(features, records, ids, duration, starts, kind, fold, stage):
    by_role = {(r.participant_id, r.role): r.recording_id for r in records}
    count = duration // 10
    gallery, probes, query_meta = [], [], []
    for subject in ids:
        enrollment = features[by_role[subject, "enroll"]]
        gallery.append(phase_template(enrollment).reshape(-1) if kind == "phase" else normalize(np.asarray(enrollment).mean(axis=0)))
        values = features[by_role[subject, "probe"]]
        for start in starts:
            block = values[start // 10:start // 10 + count]
            if len(block) != count:
                raise ValueError("Incomplete query interval")
            probes.append(phase_template(block).reshape(-1) if kind == "phase" else normalize(np.asarray(block).mean(axis=0)))
            query_meta.append((subject, start, by_role[subject, "probe"]))
    gallery, probes = np.stack(gallery), np.stack(probes)
    scores = (gallery @ probes.conj().T).real
    if kind == "phase":
        scores /= gallery.shape[1]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    rows = []
    for i, claimed in enumerate(ids):
        for j, (source, start, recording_id) in enumerate(query_meta):
            rows.append({"fold": fold, "stage": stage, "claimed_id": claimed, "source_id": source, "probe_recording_id": recording_id, "start_seconds": start, "duration_seconds": duration, "is_genuine": int(claimed == source), "score": float(scores[i, j])})
    return pd.DataFrame(rows)


def _validated(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Finite scores and both binary classes are required")
    return labels, scores


def choose_threshold(labels, scores):
    labels, scores = _validated(labels, scores)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_labels)[ends]]
    fp = np.r_[0, np.cumsum(1 - sorted_labels)[ends]]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    far = fp / np.sum(labels == 0)
    frr = 1 - tp / np.sum(labels == 1)
    hter = (far + frr) / 2
    index = np.lexsort((-thresholds, hter, np.abs(far - frr)))[0]
    return {"threshold": float(thresholds[index]), "development_FAR": float(far[index]), "development_FRR": float(frr[index]), "development_EER_approx": float(hter[index]), "n_genuine": int(np.sum(labels)), "n_impostor": int(np.sum(labels == 0)), "rule": "score >= threshold; min abs(FAR-FRR), then HTER, then highest threshold"}


def evaluate_scores(labels, scores, threshold):
    labels, scores = _validated(labels, scores)
    accept = scores >= threshold
    far = float(accept[labels == 0].mean())
    frr = float((~accept[labels == 1]).mean())
    return {"FAR": far, "FRR": frr, "TAR": 1 - frr, "HTER": (far + frr) / 2, "ROC_AUC": float(roc_auc_score(labels, scores)), "EER_diagnostic": choose_threshold(labels, scores)["development_EER_approx"], "n_genuine": int(np.sum(labels)), "n_impostor": int(np.sum(labels == 0))}
