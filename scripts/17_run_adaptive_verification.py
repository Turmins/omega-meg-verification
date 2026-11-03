"""Run isolated EEGNet and graph verification experiments on the existing data."""
from __future__ import annotations
import argparse
import os
import sys
from _bootstrap import ROOT
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
def configure_runtime():
    for name in ("TEMP", "TMP", "MPLCONFIGDIR", "TORCH_HOME"):
        folder = ROOT / "outputs/adaptive_runtime" / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
import torch
from omega_dl.paths import inside_extension
from omega_dl.provenance import Invocation, run_training
from omega_dl.adaptive_verification.protocol import prepare, freeze_all, require_frozen, fold_path
from omega_dl.adaptive_verification.features import FeatureStore
from omega_dl.adaptive_verification.experiment import train_model, evaluate_fold
from omega_dl.calibration_io import calibrate_model, calibrate_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptive_v2.yaml")
    parser.add_argument("--out", default="outputs/adaptive_v2")
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "report", "all"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    configure_runtime()
    if len(set(args.folds)) != len(args.folds) or any(f not in range(6) for f in args.folds):
        parser.error("Folds must be unique integers in 0..5")
    root = inside_extension(ROOT / args.out)
    with Invocation(root, args.stage, vars(args)) as invocation:
        p, base = prepare(ROOT / args.config, root)
        torch.set_num_threads(p["config"]["train"]["cpu_threads"])
        invocation.bind_protocol(p)
        print(f"V2 protocol: {p['protocol_sha256']}", flush=True)
        store = FeatureStore(inside_extension(ROOT / p["config"]["base_run"]), base, root, p["config"]["graph_neighbors"])
        if args.stage in ("train", "all"):
            for fold in args.folds:
                for method in p["config"]["models"]:
                    path = fold_path(root, fold, method)
                    run_training(invocation, train_model, path / "training_summary.json",
                                 [path / name for name in ("best.pt", "initial.pt")],
                                 fold=fold, model=method, seed=p["config"]["seed"] + 100 * fold,
                                 args=(root, p, base, store, fold, method))
                    calibrate_model(root, p, base, store, fold, method)
                calibrate_phase(root, p, base, store, fold)
            if freeze_all(root, p):
                print("All twelve encoders and all development calibrations frozen", flush=True)
        if args.stage in ("evaluate", "all"):
            require_frozen(root, p)
            for fold in args.folds:
                evaluate_fold(root, p, base, store, fold)
        if args.stage in ("report", "all"):
            from omega_dl.adaptive_verification.reporting import build_report
            build_report(root, p)
        print("Requested stages complete", flush=True)


if __name__ == "__main__":
    main()
