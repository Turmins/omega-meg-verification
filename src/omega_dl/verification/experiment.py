"""Train six subject-disjoint encoders, calibrate, then unlock test scoring."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from omega_dl.models.siamese_cnn import SiameseCNN
from omega_dl.models.losses import cross_recording_masks
from omega_dl.seeding import seed_everything
from omega_dl.verification.preprocessing import WindowCache
from omega_dl.verification.protocol import fold_dir, freeze_test, now, read_json, records_for_stage, require_test_frozen, save_json
from omega_dl.verification.scoring import choose_threshold, evaluate_scores, score_frame


def model_for(config):
    return SiameseCNN(n_bands=3, **config["model"])


def save_weights(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(path)


def load_windows(cache, records, device):
    shape = (len(records) * 24, 780, 1280)
    storage = device
    if str(device).startswith("cuda") and math.prod(shape) * 2 > torch.cuda.mem_get_info()[0] * 0.70:
        storage = "cpu"
    tensor = torch.empty(shape, dtype=torch.float16, device=storage)
    for i, record in enumerate(records):
        values = np.array(cache.open(record, "cnn"), copy=True)
        tensor[i * 24:(i + 1) * 24].copy_(torch.from_numpy(values))
    return tensor


@torch.no_grad()
def embed(model, windows, records, device):
    model.eval()
    result = []
    for start in range(0, len(windows), 24):
        result.append(model(windows[start:start + 24].to(device=device, dtype=torch.float32)).cpu().numpy())
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite embeddings")
    return {r.recording_id: values[i * 24:(i + 1) * 24] for i, r in enumerate(records)}


def triplet_loss(z, labels, recording_ids, sessions, margin, warmup):
    positive, negative = cross_recording_masks(labels, recording_ids, sessions)
    if not torch.all(positive.any(dim=1)) or not torch.all(negative.any(dim=1)):
        raise ValueError("Every anchor must have a cross-session positive and a negative")
    distances = 1 - z.float() @ z.float().T
    hardest_positive = distances.masked_fill(~positive, -torch.inf).max(dim=1).values
    if warmup:
        index = torch.rand_like(distances).masked_fill(~negative, 2).argmin(dim=1)
        selected_negative = distances.gather(1, index[:, None]).squeeze(1)
    else:
        semi = negative & (distances > hardest_positive[:, None]) & (distances < hardest_positive[:, None] + margin)
        candidate = distances.masked_fill(~semi, torch.inf).min(dim=1).values
        fallback = distances.masked_fill(~negative, torch.inf).min(dim=1).values
        selected_negative = torch.where(torch.isfinite(candidate), candidate, fallback)
    loss = F.relu(hardest_positive - selected_negative + margin)
    return loss.mean(), (loss > 0).float().mean()


def sample_batch(records, ids, rng, p_subjects):
    indexes, labels, rec_ids, sessions = [], [], [], []
    session_map = {s: i for i, s in enumerate(sorted({r.session for r in records}))}
    positions = {s: [i for i, r in enumerate(records) if r.participant_id == s] for s in ids}
    for label in rng.choice(len(ids), size=p_subjects, replace=False):
        chosen = positions[ids[label]]
        if len(chosen) != 2 or records[chosen[0]].session == records[chosen[1]].session:
            raise ValueError("Training identity requires two distinct sessions")
        for rec in chosen:
            for window in rng.choice(24, size=2, replace=False):
                indexes.append(rec * 24 + int(window))
                labels.append(int(label))
                rec_ids.append(rec)
                sessions.append(session_map[records[rec].session])
    return indexes, labels, rec_ids, sessions


def train_fold(run_root, protocol, fold, cache):
    path = fold_dir(run_root, fold)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "training_summary.json").exists():
        summary = read_json(path / "training_summary.json")
        if summary["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Completed fold belongs to a different protocol")
        print(f"fold {fold}: existing completed training reused", flush=True)
        return
    c, tc = protocol["config"], protocol["config"]["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(c["seed"] + fold)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(tc["cpu_threads"])
    rng = np.random.default_rng(c["seed"] + fold)
    train_records = records_for_stage(run_root, protocol, fold, "train")
    dev_records = records_for_stage(run_root, protocol, fold, "development")
    for i, record in enumerate(train_records + dev_records):
        cache.ensure(record)
        if (i + 1) % 10 == 0:
            print(f"fold {fold}: verified/built {i + 1}/100 training/development record caches", flush=True)
    train_ids = protocol["folds"][fold]["train_subjects"]
    dev_ids = protocol["folds"][fold]["val_subjects"]
    x = load_windows(cache, train_records, device)
    dev_x = load_windows(cache, dev_records, device)
    model = model_for(c).to(device)
    save_weights(path / "untrained.pt", model)
    auxiliary = torch.nn.Linear(c["model"]["embedding_dim"], len(train_ids)).to(device)
    parameters = list(model.parameters()) + list(auxiliary.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * tc["batches_per_epoch"]
    warmup_steps = tc["warmup_epochs"] * tc["batches_per_epoch"]
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    start_time = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        losses, active = [], []
        for batch in range(tc["batches_per_epoch"]):
            step = (epoch - 1) * tc["batches_per_epoch"] + batch
            factor = (step + 1) / max(warmup_steps, 1) if step < warmup_steps else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))
            for group in optimizer.param_groups:
                group["lr"] = tc["lr"] * factor
            indices, labels, recs, sessions = sample_batch(train_records, train_ids, rng, tc["p_subjects"])
            values = x[torch.tensor(indices, device=x.device)].to(device=device, dtype=torch.float32)
            values *= (torch.rand(values.shape[0], values.shape[1], 1, device=device) >= tc["channel_dropout"])
            values += torch.randn_like(values) * tc["gaussian_noise"]
            labels, recs, sessions = [torch.tensor(v, device=device) for v in [labels, recs, sessions]]
            z = model(values)
            triplet, fraction = triplet_loss(z, labels, recs, sessions, tc["margin"], epoch <= tc["warmup_epochs"])
            loss = triplet + tc["aux_weight"] * F.cross_entropy(auxiliary(z), labels, label_smoothing=tc["label_smoothing"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, tc["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            active.append(float(fraction.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "active_triplet_fraction": float(np.mean(active)), "elapsed_seconds": time.monotonic() - start_time}
        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            embeddings = embed(model, dev_x, dev_records, device)
            frame = score_frame(embeddings, dev_records, dev_ids, tc["selection_duration"], c["query_starts"], "cnn", fold, "development")
            auc = float(roc_auc_score(frame.is_genuine, frame.score))
            spread = float(np.concatenate(list(embeddings.values())).std(axis=0).mean())
            row.update(development_AUC_30s=auc, embedding_dimension_std=spread)
            if not np.isfinite(spread) or spread < 1e-7:
                raise RuntimeError("Collapsed or invalid development embeddings")
            if auc > best_auc + 1e-12:
                best_auc, best_epoch, stale = auc, epoch, 0
                save_weights(path / "best.pt", model)
            else:
                stale += 1
            print(f"fold {fold}, epoch {epoch}: loss={row['loss']:.4f}, development AUC={auc:.4f}, best epoch={best_epoch}, elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        history.append(row)
        pd.DataFrame(history).to_csv(path / "training.csv", index=False, float_format="%.17g")
        if stale >= tc["patience_evaluations"]:
            break
    save_weights(path / "last.pt", model)
    save_json(path / "training_summary.json", {"protocol_sha256": protocol["protocol_sha256"], "fold": fold, "seed": c["seed"] + fold, "base_seed": c["seed"], "epochs_completed": epoch, "steps": epoch * tc["batches_per_epoch"], "best_epoch": best_epoch, "best_development_AUC_30s": best_auc, "parameter_count": sum(p.numel() for p in model.parameters()), "device": device, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0, "elapsed_seconds": time.monotonic() - start_time, "completed_utc": now(), "checkpoint_ties": "earliest epoch wins"})
    del model, auxiliary, optimizer, parameters, x, dev_x, values, z, loss, triplet
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_frames(run_root, protocol, fold, stage, cache):
    # The barrier is checked before any test record/cache/embedding is accessed.
    records = records_for_stage(run_root, protocol, fold, stage)
    c = protocol["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = protocol["folds"][fold]["test_subjects" if stage == "test" else "val_subjects"]
    for record in records:
        cache.ensure(record)
    windows = load_windows(cache, records, device)
    path = fold_dir(run_root, fold)
    all_frames = []
    for method in ["phase", "cnn", "untrained"]:
        if method == "phase":
            features = {r.recording_id: cache.open(r, "phase") for r in records}
        else:
            model = model_for(c).to(device)
            weight_name = "best.pt" if method == "cnn" else "untrained.pt"
            model.load_state_dict(torch.load(path / weight_name, map_location=device, weights_only=True))
            features = embed(model, windows, records, device)
            np.savez_compressed(path / f"{stage}_{method}_embeddings.npz", **features)
            del model
        for duration in c["durations"]:
            frame = score_frame(features, records, ids, duration, c["query_starts"], "phase" if method == "phase" else "cnn", fold, stage)
            frame["method"] = method
            all_frames.append(frame)
        del features
    del windows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(all_frames, ignore_index=True)


def calibrate_fold(run_root, protocol, fold, cache):
    path = fold_dir(run_root, fold)
    if (path / "test_lock.json").exists():
        require_test_frozen(run_root, protocol, fold)
        print(f"fold {fold}: existing development calibration reused", flush=True)
        return
    frame = stage_frames(run_root, protocol, fold, "development", cache)
    thresholds = {}
    for (method, duration), rows in frame.groupby(["method", "duration_seconds"]):
        thresholds[f"{method}:{duration}"] = choose_threshold(rows.is_genuine, rows.score)
    frame.to_csv(path / "development_scores.csv", index=False, float_format="%.17g")
    save_json(path / "thresholds.json", {"protocol_sha256": protocol["protocol_sha256"], "fold": fold, "source_stage": "development", "thresholds": thresholds})
    freeze_test(run_root, protocol, fold)
    print(f"fold {fold}: thresholds frozen; test remains unevaluated", flush=True)


def evaluate_fold(run_root, protocol, fold, cache):
    require_test_frozen(run_root, protocol, fold)
    path = fold_dir(run_root, fold)
    if (path / "evaluation_complete.json").exists():
        print(f"fold {fold}: existing test result reused", flush=True)
        return
    thresholds = read_json(path / "thresholds.json")["thresholds"]
    frame = stage_frames(run_root, protocol, fold, "test", cache)
    summaries, output = [], []
    for (method, duration), rows in frame.groupby(["method", "duration_seconds"]):
        threshold = thresholds[f"{method}:{duration}"]["threshold"]
        rows = rows.copy()
        rows["threshold"] = threshold
        rows["accepted"] = (rows.score >= threshold).astype(int)
        output.append(rows)
        summaries.append({"fold": fold, "method": method, "duration_seconds": int(duration), "threshold": threshold, **evaluate_scores(rows.is_genuine, rows.score, threshold)})
    pd.concat(output).to_csv(path / "test_scores.csv", index=False, float_format="%.17g")
    pd.DataFrame(summaries).to_csv(path / "metrics.csv", index=False, float_format="%.17g")
    save_json(path / "evaluation_complete.json", {"protocol_sha256": protocol["protocol_sha256"], "fold": fold, "test_attempts_per_method_duration": 400, "completed_utc": now()})
    print(f"fold {fold}: test evaluation complete", flush=True)
