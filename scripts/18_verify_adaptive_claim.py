"""Run actual encoder inference and optionally persist a personal template."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from _bootstrap import ROOT
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from omega_dl.paths import inside_extension, sha256_file
from omega_dl.verification.protocol import read_json, save_json
from omega_dl.verification.scoring import normalize
from omega_dl.adaptive_verification.protocol import load, require_frozen, fold_path
from omega_dl.adaptive_verification.features import FeatureStore
from omega_dl.adaptive_verification.models import make_model
from omega_dl.adaptive_verification.online import AdaptiveTemplate
from omega_dl.adaptive_verification.experiment import records_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/adaptive_v2")
    parser.add_argument("--model", choices=("eegnet", "graph"), required=True)
    parser.add_argument("--fold", type=int, choices=range(6), required=True)
    parser.add_argument("--claimed", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--enrollment", type=int, choices=(30, 240), default=30)
    parser.add_argument("--duration", type=int, choices=(10, 30, 60), default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--policy", choices=("static", "accepted", "guarded"), default="guarded")
    parser.add_argument("--profile", help="A JSON filename under the run's profiles directory")
    parser.add_argument("--reset", action="store_true", help="Explicitly replace a saved profile with fresh enrollment")
    args = parser.parse_args()
    if args.start < 0 or args.start + args.duration > 240 or args.start % args.duration:
        parser.error("Queries must be nonoverlapping intervals on the chosen duration grid")
    if args.reset and not args.profile:
        parser.error("--reset requires --profile")
    root = inside_extension(ROOT / args.out)
    p, base = load(root)
    require_frozen(root, p)
    records, ids = records_for(base, p, args.fold, "test", root)
    if args.claimed not in ids or args.source not in ids:
        parser.error("Claimed and source identities must be held out in this fold")
    checkpoint = fold_path(root, args.fold, args.model) / "best.pt"
    binding = {"protocol_sha256": p["protocol_sha256"], "checkpoint_sha256": sha256_file(checkpoint),
               "model": args.model, "fold": args.fold, "claimed_id": args.claimed,
               "enrollment_seconds": args.enrollment, "duration_seconds": args.duration, "policy": args.policy}
    profile_path = None
    previous = None
    if args.profile:
        if Path(args.profile).name != args.profile or Path(args.profile).suffix != ".json":
            parser.error("Profile must be a plain JSON filename")
        profile_path = inside_extension(root / "profiles" / args.profile)
        if profile_path.exists() and not args.reset:
            previous = read_json(profile_path)
            if previous["binding"] != binding:
                parser.error("Saved profile belongs to another model, claim, or protocol")
            if args.start <= previous["last_start_seconds"]:
                parser.error("A persistent evaluation profile must advance chronologically")
    torch.set_num_threads(p["config"]["train"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.model, p["config"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    store = FeatureStore(inside_extension(ROOT / p["config"]["base_run"]), base, root, p["config"]["graph_neighbors"])
    enrollment_record = next(r for r in records if r.participant_id == args.claimed and r.role == "enroll")
    query_record = next(r for r in records if r.participant_id == args.source and r.role == "probe")

    def encode(record, indices):
        with torch.no_grad():
            batches = [model(*store.batch([record], indices[start:start + 16], args.model, device)).cpu().numpy()
                       for start in range(0, len(indices), 16)]
        return normalize(np.concatenate(batches).mean(axis=0))

    calibration = read_json(fold_path(root, args.fold, args.model) / "thresholds.json")["thresholds"][f"{args.enrollment}:{args.duration}"]
    if previous:
        state = AdaptiveTemplate.restore(previous["template"])
    else:
        anchor = encode(enrollment_record, list(range(args.enrollment // 10)))
        state = AdaptiveTemplate(anchor, calibration["threshold"], calibration["update_threshold"], args.policy,
                                 p["config"]["adaptation"]["alpha"], p["config"]["adaptation"]["anchor_weight"])
    query = encode(query_record, list(range(args.start // 10, (args.start + args.duration) // 10)))
    event = f"{query_record.recording_id}:{args.start}:{args.duration}"
    result = state.verify(query, event)
    if profile_path:
        save_json(profile_path, {"binding": binding, "template": state.state(), "last_start_seconds": args.start})
    output = {**binding, "source_id": args.source, "start_seconds": args.start, "device": device,
              "mode": "fresh encoder inference from verified preprocessed windows", **result,
              "threshold": state.threshold, "update_threshold": state.update_threshold,
              "ground_truth_for_explanation_only": args.claimed == args.source,
              "profile": str(profile_path.relative_to(ROOT)) if profile_path else None}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
