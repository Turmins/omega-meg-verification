"""Separate v2 provenance and an all-model, all-fold test barrier."""
from __future__ import annotations
from pathlib import Path
import yaml
from omega_dl.paths import extension_root, inside_extension, sha256_file
from omega_dl.verification.protocol import read_json, save_json, digest, now, load_protocol as load_base

CORE = ("models.py", "features.py", "online.py", "protocol.py", "experiment.py", "evaluation.py")


def code_hashes():
    root = extension_root()
    paths = [root / "src/omega_dl/adaptive_verification" / name for name in CORE]
    paths += [root / "src/omega_dl/paths.py"]
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def validate_config(c):
    keys = {"version", "base_run", "models", "seed", "embedding_dim", "enrollment_seconds",
            "durations", "stream_repetitions", "graph_neighbors", "eegnet", "graph", "train", "adaptation"}
    if set(c) != keys or c["version"] != "adaptive_v2" or c["models"] != ["eegnet", "graph"]:
        raise ValueError("Unsupported v2 configuration")
    if c["enrollment_seconds"] != [30, 240] or c["durations"] != [10, 30, 60]:
        raise ValueError("This protocol requires both enrollment lengths and all three query lengths")
    for name in ("embedding_dim", "stream_repetitions"):
        if not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(name)
    if not 1 <= c["graph_neighbors"] < 260 or c["eegnet"]["temporal_kernel"] % 2 != 1:
        raise ValueError("Invalid graph degree or temporal kernel")
    t = c["train"]
    for name in ("epochs", "batches_per_epoch", "windows_per_session", "eval_every", "patience", "cpu_threads"):
        if not isinstance(t[name], int) or t[name] <= 0:
            raise ValueError(name)
    if not 2 <= t["subjects_per_batch"] <= 40 or t["windows_per_session"] > 24:
        raise ValueError("Invalid training episode")
    if t["temperature"] <= 0 or t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimization setting")
    a = c["adaptation"]
    if not 0 < a["alpha"] <= 1 or not 0 <= a["anchor_weight"] <= 1 or not 0 <= a["update_development_far"] < 1:
        raise ValueError("Invalid update policy")
    return c


def prepare(config_path, root):
    root = inside_extension(root)
    c = validate_config(yaml.safe_load(Path(config_path).read_text()))
    base_root = inside_extension(extension_root() / c["base_run"])
    if root == base_root or base_root in root.parents:
        raise ValueError("New outputs must be separate from the frozen v1 run")
    base = load_base(base_root)
    if (root / "protocol.json").exists():
        p, _ = load(root)
        if p["config"] != c:
            raise ValueError("Configuration differs; use a new run directory")
        return p, base
    payload = {"config": c, "base_protocol_sha256": base["protocol_sha256"],
               "folds": base["folds"], "code_sha256": code_hashes(),
               "evaluation_status": "Exploratory reuse of an already studied two-session dataset",
               "stream_design": "Five balanced chronological streams per claim; nonoverlapping probe windows; score before update"}
    payload["protocol_sha256"] = digest(payload)
    payload["created_utc"] = now()
    save_json(root / "protocol.json", payload)
    return payload, base


def load(root):
    p = read_json(root / "protocol.json")
    unsigned = {k: v for k, v in p.items() if k not in {"protocol_sha256", "created_utc"}}
    if digest(unsigned) != p["protocol_sha256"] or code_hashes() != p["code_sha256"]:
        raise ValueError("V2 protocol or core code changed; use a new output directory")
    validate_config(p["config"])
    base = load_base(inside_extension(extension_root() / p["config"]["base_run"]))
    if base["protocol_sha256"] != p["base_protocol_sha256"]:
        raise ValueError("Base protocol changed")
    return p, base


def fold_path(root, fold, method=None):
    if fold not in range(6):
        raise ValueError("Fold must be in 0..5")
    path = inside_extension(root / "folds" / f"fold_{fold:02d}")
    return path if method is None else inside_extension(path / method)


def locked_files(root):
    result = []
    for fold in range(6):
        result.append(fold_path(root, fold) / "phase_thresholds.json")
        for method in ("eegnet", "graph"):
            result += [fold_path(root, fold, method) / name for name in
                       ("best.pt", "initial.pt", "training_summary.json", "thresholds.json", "initial_thresholds.json")]
    return result


def freeze_all(root, protocol):
    files = locked_files(root)
    if not all(path.exists() for path in files):
        return False
    lock_path = root / "test_lock.json"
    if lock_path.exists():
        require_frozen(root, protocol)
        return True
    save_json(lock_path, {"protocol_sha256": protocol["protocol_sha256"], "frozen_utc": now(),
                         "files_sha256": {str(p.relative_to(root)): sha256_file(p) for p in files}})
    return True


def require_frozen(root, protocol):
    path = root / "test_lock.json"
    if not path.exists():
        raise RuntimeError("All twelve models and development thresholds must be frozen before test access")
    lock = read_json(path)
    expected_names = {str(p.relative_to(root)) for p in locked_files(root)}
    if lock["protocol_sha256"] != protocol["protocol_sha256"] or set(lock["files_sha256"]) != expected_names:
        raise RuntimeError("Incomplete or incompatible test lock")
    if any(sha256_file(root / name) != value for name, value in lock["files_sha256"].items()):
        raise RuntimeError("A frozen checkpoint or threshold changed")
    return lock
