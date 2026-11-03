"""Matched static comparisons and label-free sequential adaptation experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from omega_dl.verification.scoring import score_frame, normalize, choose_threshold, evaluate_scores
from .online import AdaptiveTemplate, strict_update_threshold

POLICIES = ("static", "accepted", "guarded")


def comparisons(features, records, ids, enrollment, duration, kind, fold, stage):
    enrollment_records = {r.recording_id for r in records if r.role == "enroll"}
    adjusted = {key: value[:enrollment // 10] if key in enrollment_records else value
                for key, value in features.items()}
    starts = list(range(0, 241 - duration, duration))
    return score_frame(adjusted, records, ids, duration, starts, kind, fold, stage)


def calibrate(features, records, ids, config, kind, fold):
    thresholds, frames = {}, []
    for enrollment in config["enrollment_seconds"]:
        for duration in config["durations"]:
            frame = comparisons(features, records, ids, enrollment, duration, kind, fold, "development")
            entry = choose_threshold(frame.is_genuine, frame.score)
            entry["update_threshold"] = strict_update_threshold(
                frame.is_genuine, frame.score, entry["threshold"],
                config["adaptation"]["update_development_far"])
            entry["empirical_update_gate_FAR"] = float((frame.loc[frame.is_genuine == 0, "score"]
                                                         >= entry["update_threshold"]).mean())
            thresholds[f"{enrollment}:{duration}"] = entry
            frame["enrollment_seconds"] = enrollment
            frames.append(frame)
    return thresholds, pd.concat(frames, ignore_index=True)


def schedule(ids, duration, repetition, seed):
    """Independent reference state for each claimed identity; chronological slots."""
    rng = np.random.default_rng(seed + duration * 101 + repetition * 10007)
    starts = list(range(0, 241 - duration, duration))
    events = []
    for claimed in ids:
        genuine = np.array([True] * (len(starts) // 2) + [False] * (len(starts) // 2))
        rng.shuffle(genuine)
        others = [subject for subject in ids if subject != claimed]
        for step, (start, is_genuine) in enumerate(zip(starts, genuine)):
            source = claimed if is_genuine else str(rng.choice(others))
            events.append({"claimed_id": claimed, "source_id": source, "step": step,
                           "start_seconds": start, "duration_seconds": duration,
                           "is_genuine": int(is_genuine), "repetition": repetition})
    return events


def stream_scores(features, records, ids, config, thresholds, fold, method, stage="test"):
    by_role = {(r.participant_id, r.role): features[r.recording_id] for r in records}
    rows = []
    for enrollment in config["enrollment_seconds"]:
        anchors = {s: normalize(by_role[s, "enroll"][:enrollment // 10].mean(axis=0)) for s in ids}
        for duration in config["durations"]:
            entry = thresholds[f"{enrollment}:{duration}"]
            for repetition in range(config["stream_repetitions"]):
                events = schedule(ids, duration, repetition, config["seed"] + fold * 100003)
                for policy in POLICIES:
                    states = {s: AdaptiveTemplate(anchors[s], entry["threshold"], entry["update_threshold"],
                              policy, config["adaptation"]["alpha"], config["adaptation"]["anchor_weight"]) for s in ids}
                    for event in events:
                        start = event["start_seconds"] // 10
                        vector = normalize(by_role[event["source_id"], "probe"][start:start + duration // 10].mean(axis=0))
                        # Ground truth never reaches the verifier or its update rule.
                        decision = states[event["claimed_id"]].verify(vector, str(event["step"]))
                        rows.append({**event, **decision, "enrollment_seconds": enrollment,
                                     "method": method, "fold": fold, "stage": stage, "policy": policy,
                                     "threshold": entry["threshold"], "update_threshold": entry["update_threshold"]})
    return pd.DataFrame(rows)


def summarize_streams(frame):
    results = []
    for keys, rows in frame.groupby(["fold", "method", "enrollment_seconds", "duration_seconds", "policy"]):
        genuine = rows.is_genuine.to_numpy().astype(bool)
        accepted = rows.accepted.to_numpy().astype(bool)
        updated = rows.updated.to_numpy().astype(bool)
        far = float(accepted[~genuine].mean())
        frr = float((~accepted[genuine]).mean())
        results.append(dict(zip(("fold", "method", "enrollment_seconds", "duration_seconds", "policy"), keys))
                       | {"FAR": far, "FRR": frr, "HTER": (far + frr) / 2,
                          "n_genuine": int(genuine.sum()), "n_impostor": int((~genuine).sum()),
                          "false_accepts": int((accepted & ~genuine).sum()),
                          "false_rejects": int((~accepted & genuine).sum()),
                          "updates": int(updated.sum()), "impostor_updates": int((updated & ~genuine).sum()),
                          "mean_final_drift": float(rows.groupby(["claimed_id", "repetition"]).tail(1).drift_from_anchor.mean())})
    return pd.DataFrame(results)


def static_metrics(frame, thresholds, method, fold):
    result, decisions = [], []
    for (enrollment, duration), rows in frame.groupby(["enrollment_seconds", "duration_seconds"]):
        threshold = thresholds[f"{enrollment}:{duration}"]["threshold"]
        rows = rows.copy()
        rows["method"], rows["threshold"] = method, threshold
        rows["accepted"] = rows.score >= threshold
        decisions.append(rows)
        result.append({"fold": fold, "method": method, "enrollment_seconds": int(enrollment),
                       "duration_seconds": int(duration),
                       **evaluate_scores(rows.is_genuine, rows.score, threshold),
                       "false_accepts": int(((rows.is_genuine == 0) & rows.accepted).sum()),
                       "false_rejects": int(((rows.is_genuine == 1) & ~rows.accepted).sum())})
    return pd.concat(decisions, ignore_index=True), pd.DataFrame(result)
