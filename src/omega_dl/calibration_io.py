"""Additional development output IO for new v2 runs; frozen core is unchanged."""
from __future__ import annotations
import numpy as np

from omega_dl.adaptive_verification import experiment as core
from omega_dl.adaptive_verification.evaluation import calibrate
from omega_dl.adaptive_verification.protocol import fold_path
from omega_dl.verification.protocol import read_json, save_json


def fresh_targets(paths):
    if any(path.exists() for path in paths):
        raise FileExistsError('Existing calibration output found; use a new run')


def save_frame(path, frame):
    with path.open('x', encoding='utf-8', newline='') as handle:
        frame.to_csv(handle, index=False, float_format='%.17g')


def calibrate_model(root, p, base, store, fold, method):
    path = fold_path(root, fold, method)
    threshold_files = [path / name for name in ('thresholds.json', 'initial_thresholds.json')]
    if all(file.exists() for file in threshold_files):
        return  # Historical reuse is not a request to backfill missing artifacts.
    if any(file.exists() for file in threshold_files):
        raise FileExistsError('Partial calibration exists; use a new run instead of overwriting it')
    fresh_targets([path / name for name in ('development_embeddings.npz', 'development_scores.csv',
                                           'initial_development_scores.csv')])
    records, ids = core.records_for(base, p, fold, 'development')
    features = core.neural_features(root, p, store, records, fold, method)
    entries, frame = calibrate(features, records, ids, p['config'], 'cnn', fold)
    with (path / 'development_embeddings.npz').open('xb') as handle:
        np.savez_compressed(handle, **features)
    save_frame(path / 'development_scores.csv', frame)
    initial = core.neural_features(root, p, store, records, fold, method, initial=True)
    initial_entries, initial_frame = calibrate(initial, records, ids, p['config'], 'cnn', fold)
    save_frame(path / 'initial_development_scores.csv', initial_frame)
    save_json(path / 'thresholds.json', {'protocol_sha256': p['protocol_sha256'],
              'source_stage': 'development', 'thresholds': entries})
    save_json(path / 'initial_thresholds.json', {'source_stage': 'development', 'thresholds': initial_entries})


def calibrate_phase(root, p, base, store, fold):
    path = fold_path(root, fold)
    if (path / 'phase_thresholds.json').exists():
        return
    fresh_targets([path / 'phase_development_scores.csv'])
    records, ids = core.records_for(base, p, fold, 'development')
    features = {r.recording_id: store.arrays(r, 'phase')[0] for r in records}
    entries, frame = calibrate(features, records, ids, p['config'], 'phase', fold)
    save_frame(path / 'phase_development_scores.csv', frame)
    save_json(path / 'phase_thresholds.json', {'source_stage': 'development', 'thresholds': entries})
