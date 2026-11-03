"""Frozen configuration, subject splits, provenance, and test access barrier."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from omega_dl.data.index import load_recording_index
from omega_dl.data.splits import assert_no_leakage, make_folds
from omega_dl.data.validate import validate_dataset
from omega_dl.paths import extension_root, inside_extension, resolve_data_root, sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = inside_extension(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(c):
    expected = {"version", "data_root", "sample_rate", "channels", "central_seconds", "context_seconds", "window_seconds", "bands", "durations", "query_starts", "split_seed", "seed", "normalization", "clip_mad", "model", "train"}
    if set(c) != expected:
        raise ValueError(f"Unsupported or missing configuration keys: {set(c) ^ expected}")
    fixed = dict(version="verification_v1", sample_rate=128, channels=260, central_seconds=240, context_seconds=10, window_seconds=10, bands=["theta", "alpha", "beta"], durations=[10, 30, 60], query_starts=[0, 60, 120, 180], normalization="window_channel_mad")
    if any(c[k] != v for k, v in fixed.items()):
        raise ValueError("This version requires the frozen 10-second / three-band protocol")
    model_keys = {"spatial_per_band", "temporal_widths", "kernel", "dilations", "pool", "dropout", "embedding_dim"}
    train_keys = {"epochs", "batches_per_epoch", "p_subjects", "windows_per_recording", "lr", "weight_decay", "warmup_epochs", "margin", "aux_weight", "label_smoothing", "grad_clip", "eval_every", "patience_evaluations", "selection_duration", "channel_dropout", "gaussian_noise", "cpu_threads"}
    if set(c["model"]) != model_keys or set(c["train"]) != train_keys:
        raise ValueError("Unknown or missing model/training setting")
    if c["train"]["selection_duration"] != 30 or c["train"]["windows_per_recording"] != 2:
        raise ValueError("Selection duration and cross-session sampler are frozen")
    for name in ["epochs", "batches_per_epoch", "p_subjects", "eval_every", "patience_evaluations", "cpu_threads"]:
        if not isinstance(c["train"][name], int) or c["train"][name] <= 0:
            raise ValueError(f"Expected positive integer: {name}")
    if c["train"]["epochs"] < c["train"]["eval_every"] or not 2 <= c["train"]["p_subjects"] <= 40:
        raise ValueError("Training must contain a validation event and at least two identities")
    if not 0 < c["clip_mad"] <= 20:
        raise ValueError("Invalid clipping threshold")
    return c


def source_hashes():
    root = extension_root()
    paths = list((root / "src" / "omega_dl" / "verification").glob("*.py"))
    paths = [p for p in paths if p.name not in {"reporting.py"}]
    paths += list((root / "src" / "omega_dl" / "models").glob("*.py"))
    paths += [root / "src" / "omega_dl" / "data" / name for name in ["index.py", "splits.py"]]
    paths += [root / "src" / "omega_dl" / "seeding.py"]
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}


def prepare(config_path, run_root):
    run_root = inside_extension(run_root)
    config = validate_config(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    protocol_path = run_root / "protocol.json"
    if protocol_path.exists():
        saved = load_protocol(run_root)
        if saved["config"] != config:
            raise ValueError("Configuration changed; use a new output directory")
        return saved
    data_root = resolve_data_root(config["data_root"])
    validation = validate_dataset(data_root, check_arrays=True)
    validation.pop("records")
    records = load_recording_index(data_root)
    folds = make_folds(records, config["split_seed"])
    assert_no_leakage(records, folds)
    metadata = {name: sha256_file(data_root / name) for name in ["recording_index.csv", "channels.txt", "protocol_FROZEN.json", "cache_manifest.csv"]}
    sources = {r.recording_id: {"filename": r.cache_filename, "sha256": sha256_file(data_root / "cache" / r.cache_filename)} for r in records}
    payload = {"config": config, "config_sha256": digest(config), "folds": [asdict(f) for f in folds], "metadata_sha256": metadata, "recordings": sources, "code_sha256": source_hashes(), "validation": validation}
    payload["protocol_sha256"] = digest(payload)
    payload["created_utc"] = now()
    save_json(protocol_path, payload)
    return payload


def load_protocol(run_root, check_code=True):
    protocol = read_json(Path(run_root) / "protocol.json")
    unsigned = {k: v for k, v in protocol.items() if k not in {"protocol_sha256", "created_utc"}}
    if digest(unsigned) != protocol["protocol_sha256"]:
        raise ValueError("Protocol manifest has been modified")
    validate_config(protocol["config"])
    data_root = resolve_data_root(protocol["config"]["data_root"])
    if any(sha256_file(data_root / k) != v for k, v in protocol["metadata_sha256"].items()):
        raise ValueError("Source metadata has changed")
    if check_code and source_hashes() != protocol["code_sha256"]:
        raise ValueError("Experiment code changed after protocol freeze; use a new run directory")
    return protocol


def fold_dir(run_root, fold):
    return inside_extension(Path(run_root) / "folds" / f"fold_{fold:02d}")


def freeze_test(run_root, protocol, fold):
    path = fold_dir(run_root, fold)
    files = ["best.pt", "untrained.pt", "thresholds.json", "training_summary.json"]
    lock = {"fold": fold, "protocol_sha256": protocol["protocol_sha256"], "files_sha256": {name: sha256_file(path / name) for name in files}, "frozen_utc": now()}
    save_json(path / "test_lock.json", lock)
    return lock


def require_test_frozen(run_root, protocol, fold):
    path = fold_dir(run_root, fold)
    if not (path / "test_lock.json").exists():
        raise RuntimeError("Test access requires frozen checkpoints and development thresholds")
    lock = read_json(path / "test_lock.json")
    if lock["fold"] != fold or lock["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("Test lock belongs to another protocol/fold")
    if any(sha256_file(path / name) != value for name, value in lock["files_sha256"].items()):
        raise RuntimeError("Frozen calibration or checkpoint has changed")
    return lock


def records_for_stage(run_root, protocol, fold, stage):
    if stage not in {"train", "development", "test"}:
        raise ValueError(stage)
    if stage == "test":
        require_test_frozen(run_root, protocol, fold)
    key = {"train": "train_subjects", "development": "val_subjects", "test": "test_subjects"}[stage]
    ids = set(protocol["folds"][fold][key])
    records = load_recording_index(resolve_data_root(protocol["config"]["data_root"]))
    return [r for r in records if r.participant_id in ids]
