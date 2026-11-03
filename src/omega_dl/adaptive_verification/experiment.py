"""Train both encoders on identical cross-session prototype episodes."""
from __future__ import annotations
import gc
import math
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from omega_dl.data.index import load_recording_index
from omega_dl.paths import resolve_data_root
from omega_dl.seeding import seed_everything
from omega_dl.verification.protocol import save_json, read_json, now
from .models import make_model, prototype_loss
from .protocol import fold_path, require_frozen
from .evaluation import comparisons, calibrate, stream_scores, summarize_streams, static_metrics


def records_for(base, p, fold, stage, root=None):
    if stage == "test":
        if root is None:
            raise RuntimeError("Test requires an output directory and frozen models")
        require_frozen(root, p)
    key = {"train": "train_subjects", "development": "val_subjects", "test": "test_subjects"}[stage]
    ids = p["folds"][fold][key]
    records = load_recording_index(resolve_data_root(base["config"]["data_root"]))
    return [r for r in records if r.participant_id in ids], ids


def episode(records, ids, rng, subjects, windows):
    positions = {(r.participant_id, r.role): i for i, r in enumerate(records)}
    indices = []
    for subject in rng.choice(ids, size=subjects, replace=False):
        enroll, probe = (positions[subject, role] for role in ("enroll", "probe"))
        if records[enroll].session == records[probe].session:
            raise ValueError("Support and query must come from different sessions")
        for position in (enroll, probe):
            indices.extend(position * 24 + int(w) for w in rng.choice(24, size=windows, replace=False))
    return indices


def weights(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


@torch.no_grad()
def embed(model, store, records, method, device, batch_size=16):
    model.eval()
    chunks = []
    for start in range(0, len(records) * 24, batch_size):
        indices = list(range(start, min(start + batch_size, len(records) * 24)))
        chunks.append(model(*store.batch(records, indices, method, device)).cpu().numpy())
    values = np.concatenate(chunks)
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
        raise RuntimeError("Invalid embeddings")
    return {r.recording_id: values[i * 24:(i + 1) * 24] for i, r in enumerate(records)}


def train_model(root, p, base, store, fold, method):
    path = fold_path(root, fold, method)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "training_summary.json").exists():
        if read_json(path / "training_summary.json")["protocol_sha256"] != p["protocol_sha256"]:
            raise ValueError("Existing training belongs to another protocol")
        print(f"{method} fold {fold}: completed training reused", flush=True)
        return
    c, t = p["config"], p["config"]["train"]
    seed = c["seed"] + 100 * fold
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(t["cpu_threads"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records, ids = records_for(base, p, fold, "train")
    dev_records, dev_ids = records_for(base, p, fold, "development")
    for record in records + dev_records:
        store.arrays(record, method)
    model = make_model(method, c).to(device)
    weights(path / "initial.pt", model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    rng = np.random.default_rng(seed)
    history, best, best_epoch, stale = [], -1., 0, 0
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        losses = []
        for batch in range(t["batches_per_epoch"]):
            step = (epoch - 1) * t["batches_per_epoch"] + batch
            total = t["epochs"] * t["batches_per_epoch"]
            factor = min(1., (step + 1) / (2 * t["batches_per_epoch"]))
            factor *= 0.1 + 0.9 * (1 + math.cos(math.pi * step / total)) / 2
            for group in optimizer.param_groups:
                group["lr"] = t["learning_rate"] * factor
            indices = episode(records, ids, rng, t["subjects_per_batch"], t["windows_per_session"])
            inputs = store.batch(records, indices, method, device)
            z = model(*inputs)
            loss = prototype_loss(z, t["subjects_per_batch"], t["windows_per_session"], t["temperature"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started}
        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            features = embed(model, store, dev_records, method, device)
            frame = comparisons(features, dev_records, dev_ids, t["selection_enrollment_seconds"],
                                t["selection_duration"], "cnn", fold, "development")
            auc = float(roc_auc_score(frame.is_genuine, frame.score))
            row["development_AUC"] = auc
            if auc > best + 1e-12:
                best, best_epoch, stale = auc, epoch, 0
                weights(path / "best.pt", model)
            else:
                stale += 1
            print(f"{method} fold {fold} epoch {epoch}: loss={row['loss']:.4f}, dev AUC={auc:.4f}, best={best_epoch}, {row['elapsed_seconds']:.0f}s", flush=True)
        history.append(row)
        pd.DataFrame(history).to_csv(path / "training.csv", index=False, float_format="%.17g")
        if stale >= t["patience"]:
            break
    save_json(path / "training_summary.json", {"protocol_sha256": p["protocol_sha256"], "model": method,
              "fold": fold, "seed": seed, "epochs": epoch, "best_epoch": best_epoch,
              "best_development_AUC": best, "parameter_count": sum(x.numel() for x in model.parameters()),
              "elapsed_seconds": time.monotonic() - started, "device": device,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "completed_utc": now()})
    del model, optimizer, inputs, z, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def neural_features(root, p, store, records, fold, method, initial=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(method, p["config"]).to(device)
    name = "initial.pt" if initial else "best.pt"
    model.load_state_dict(torch.load(fold_path(root, fold, method) / name, map_location=device, weights_only=True))
    result = embed(model, store, records, method, device)
    del model
    return result


def calibrate_model(root, p, base, store, fold, method):
    path = fold_path(root, fold, method)
    if (path / "thresholds.json").exists() and (path / "initial_thresholds.json").exists():
        return
    records, ids = records_for(base, p, fold, "development")
    features = neural_features(root, p, store, records, fold, method)
    entries, frame = calibrate(features, records, ids, p["config"], "cnn", fold)
    np.savez_compressed(path / "development_embeddings.npz", **features)
    frame.to_csv(path / "development_scores.csv", index=False, float_format="%.17g")
    save_json(path / "thresholds.json", {"protocol_sha256": p["protocol_sha256"],
              "source_stage": "development", "thresholds": entries})
    # Calibrate an independently scored untrained control using the same rules.
    initial = neural_features(root, p, store, records, fold, method, initial=True)
    initial_entries, _ = calibrate(initial, records, ids, p["config"], "cnn", fold)
    save_json(path / "initial_thresholds.json", {"source_stage": "development", "thresholds": initial_entries})
    print(f"{method} fold {fold}: development thresholds ready", flush=True)


def calibrate_phase(root, p, base, store, fold):
    path = fold_path(root, fold) / "phase_thresholds.json"
    if path.exists():
        return
    records, ids = records_for(base, p, fold, "development")
    features = {r.recording_id: store.arrays(r, "phase")[0] for r in records}
    entries, _ = calibrate(features, records, ids, p["config"], "phase", fold)
    save_json(path, {"source_stage": "development", "thresholds": entries})


def evaluate_fold(root, p, base, store, fold):
    require_frozen(root, p)
    path = fold_path(root, fold)
    if (path / "evaluation_complete.json").exists():
        return
    records, ids = records_for(base, p, fold, "test", root)
    static_frames, metric_frames, streams = [], [], []
    for method in p["config"]["models"] + ["phase"]:
        if method == "phase":
            features = {r.recording_id: store.arrays(r, "phase")[0] for r in records}
            thresholds = read_json(path / "phase_thresholds.json")["thresholds"]
        else:
            features = neural_features(root, p, store, records, fold, method)
            np.savez_compressed(fold_path(root, fold, method) / "test_embeddings.npz", **features)
            thresholds = read_json(fold_path(root, fold, method) / "thresholds.json")["thresholds"]
            streams.append(stream_scores(features, records, ids, p["config"], thresholds, fold, method))
        frames = []
        for enrollment in p["config"]["enrollment_seconds"]:
            for duration in p["config"]["durations"]:
                frame = comparisons(features, records, ids, enrollment, duration,
                                    "phase" if method == "phase" else "cnn", fold, "test")
                frame["enrollment_seconds"] = enrollment
                frames.append(frame)
        decisions, metrics = static_metrics(pd.concat(frames), thresholds, method, fold)
        static_frames.append(decisions)
        metric_frames.append(metrics)
        if method != "phase":
            initial = neural_features(root, p, store, records, fold, method, initial=True)
            initial_thresholds = read_json(fold_path(root, fold, method) / "initial_thresholds.json")["thresholds"]
            frames = []
            for enrollment in p["config"]["enrollment_seconds"]:
                for duration in p["config"]["durations"]:
                    frame = comparisons(initial, records, ids, enrollment, duration, "cnn", fold, "test")
                    frame["enrollment_seconds"] = enrollment
                    frames.append(frame)
            decisions, metrics = static_metrics(pd.concat(frames), initial_thresholds, method + "_initial", fold)
            static_frames.append(decisions)
            metric_frames.append(metrics)
    pd.concat(static_frames).to_csv(path / "static_scores.csv", index=False, float_format="%.17g")
    pd.concat(metric_frames).to_csv(path / "static_metrics.csv", index=False, float_format="%.17g")
    stream_frame = pd.concat(streams, ignore_index=True)
    stream_frame.to_csv(path / "stream_scores.csv", index=False, float_format="%.17g")
    summarize_streams(stream_frame).to_csv(path / "stream_metrics.csv", index=False, float_format="%.17g")
    save_json(path / "evaluation_complete.json", {"protocol_sha256": p["protocol_sha256"], "completed_utc": now()})
    print(f"fold {fold}: both encoders, static controls and sequential updates evaluated", flush=True)
