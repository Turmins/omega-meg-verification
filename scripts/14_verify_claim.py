"""Run one held-out claim directly from enrollment and permitted query samples."""
from __future__ import annotations
import argparse
import json
import sys
from _bootstrap import ROOT
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from omega_dl.paths import inside_extension, resolve_data_root, sha256_file
from omega_dl.verification.experiment import model_for
from omega_dl.verification.preprocessing import EDGES, extract_window, filter_window, normalize_window, window_coherency
from omega_dl.verification.protocol import fold_dir, load_protocol, read_json, records_for_stage, require_test_frozen, save_json
from omega_dl.verification.scoring import normalize, phase_template


def extract_features(record, indices, protocol, model, device):
    path = resolve_data_root(protocol["config"]["data_root"]) / "cache" / record.cache_filename
    if sha256_file(path) != protocol["recordings"][record.recording_id]["sha256"]:
        raise ValueError("Source recording differs from the frozen input")
    raw = np.load(path, mmap_mode="r")
    cnn, phase = [], []
    with threadpool_limits(limits=2):
        for index in indices:
            bands, edges = [], []
            window = extract_window(raw, index)
            for band in protocol["config"]["bands"]:
                filtered = filter_window(window, band)
                bands.append(normalize_window(filtered, protocol["config"]["clip_mad"])[0])
                edges.append(window_coherency(filtered, band)[EDGES])
            cnn.append(np.concatenate(bands))
            phase.append(np.stack(edges))
    with torch.no_grad():
        embedding = model(torch.from_numpy(np.stack(cnn)).to(device=device, dtype=torch.float32)).cpu().numpy()
    return phase_template(np.stack(phase)).reshape(-1), normalize(embedding.mean(axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/verification_v1")
    parser.add_argument("--fold", type=int, required=True, choices=range(6))
    parser.add_argument("--claimed", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--duration", type=int, choices=[10, 30, 60], default=30)
    parser.add_argument("--start", type=int, choices=[0, 60, 120, 180], default=0)
    parser.add_argument("--save", help="Optional JSON output path inside the extension")
    args = parser.parse_args()
    run_root = inside_extension(ROOT / args.out)
    protocol = load_protocol(run_root)
    require_test_frozen(run_root, protocol, args.fold)
    ids = protocol["folds"][args.fold]["test_subjects"]
    if args.claimed not in ids or args.source not in ids:
        parser.error("Both claimed participant and query source must be test IDs in this fold")
    records = records_for_stage(run_root, protocol, args.fold, "test")
    enrollment = next(r for r in records if r.participant_id == args.claimed and r.role == "enroll")
    query = next(r for r in records if r.participant_id == args.source and r.role == "probe")
    path = fold_dir(run_root, args.fold)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = model_for(protocol["config"]).to(device)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
    model.eval()
    enrollment_phase, enrollment_cnn = extract_features(enrollment, range(24), protocol, model, device)
    query_phase, query_cnn = extract_features(query, range(args.start // 10, (args.start + args.duration) // 10), protocol, model, device)
    scores = {"phase": float(np.vdot(query_phase, enrollment_phase).real / len(query_phase)), "cnn": float(enrollment_cnn @ query_cnn)}
    thresholds = read_json(path / "thresholds.json")["thresholds"]
    saved = pd.read_csv(path / "test_scores.csv", float_precision="round_trip")
    saved = saved[(saved.claimed_id == args.claimed) & (saved.source_id == args.source) & (saved.duration_seconds == args.duration) & (saved.start_seconds == args.start)]
    results = {}
    for method, score in scores.items():
        threshold = thresholds[f"{method}:{args.duration}"]["threshold"]
        accepted = score >= threshold
        row = saved[saved.method == method]
        if len(row) != 1:
            raise ValueError("No unique saved evaluation row for the claim")
        reference = row.iloc[0]
        error = abs(score - float(reference.score))
        if error > 5e-6 or accepted != bool(reference.accepted):
            raise RuntimeError(f"{method} inference disagrees with the saved evaluation")
        results[method] = {"score": score, "threshold": threshold, "accepted": bool(accepted), "decision": "accept" if accepted else "reject", "absolute_difference_from_saved_score": error}
    output = {"protocol_sha256": protocol["protocol_sha256"], "mode": "inference from original recordings", "fold": args.fold, "claimed_id": args.claimed, "source_id": args.source, "duration_seconds": args.duration, "start_seconds": args.start, "genuine_for_explanation_only": args.claimed == args.source, "device": device, "matches_saved_decisions": True, "results": results}
    if args.save:
        save_json(inside_extension(ROOT / args.save), output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
