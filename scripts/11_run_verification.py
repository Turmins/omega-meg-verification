"""Run the educational verification experiment entirely inside the extension."""

from __future__ import annotations

import argparse
import os
import sys

from _bootstrap import ROOT

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
def configure_runtime():
    for name in ["TEMP", "TMP", "MPLCONFIGDIR", "TORCH_HOME"]:
        location = ROOT / "outputs" / "runtime" / name.lower()
        location.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(location)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from omega_dl.paths import inside_extension
from omega_dl.verification.protocol import prepare, require_test_frozen, fold_dir
from omega_dl.provenance import Invocation, run_training
from omega_dl.verification.preprocessing import WindowCache
from omega_dl.verification.experiment import calibrate_fold, evaluate_fold, train_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/verification_v1.yaml")
    parser.add_argument("--out", default="outputs/verification_v1")
    parser.add_argument("--stage", choices=["all", "prepare", "train", "evaluate"], required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    args = parser.parse_args()
    configure_runtime()
    if len(set(args.folds)) != len(args.folds) or any(f not in range(6) for f in args.folds):
        parser.error("Folds must be unique integers in 0..5")
    out = inside_extension(ROOT / args.out)
    with Invocation(out, args.stage, vars(args)) as invocation:
        protocol = prepare(ROOT / args.config, out)
        invocation.bind_protocol(protocol)
        print(f"Protocol frozen: {protocol['protocol_sha256']}", flush=True)
        if args.stage == "prepare":
            return
        cache = WindowCache(out, protocol)
        if args.stage in {"train", "all"}:
            for fold in args.folds:
                path = fold_dir(out, fold)
                run_training(invocation, train_fold, path / "training_summary.json",
                             [path / name for name in ("best.pt", "untrained.pt", "last.pt")],
                             fold=fold, model="cnn_v1", seed=protocol["config"]["seed"] + fold,
                             args=(out, protocol, fold, cache))
                calibrate_fold(out, protocol, fold, cache)
        if args.stage in {"evaluate", "all"}:
            for fold in range(6):
                require_test_frozen(out, protocol, fold)
            for fold in args.folds:
                evaluate_fold(out, protocol, fold, cache)
        print("Requested stages completed", flush=True)


if __name__ == "__main__":
    main()
