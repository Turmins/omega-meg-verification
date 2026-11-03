from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from omega_dl import calibration_io as io
from omega_dl import paths

def test_new_v2_calibration_saves_phase_and_initial_scores_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'extension_root', lambda: tmp_path)
    records = [SimpleNamespace(participant_id=s, role=role, recording_id=f'{s}-{role}')
               for s in ['a', 'b'] for role in ['enroll', 'probe']]
    features = {r.recording_id: np.tile(np.eye(2)[0 if r.participant_id == 'a' else 1], (24, 1)) for r in records}
    monkeypatch.setattr(io.core, 'records_for', lambda *args: (records, ['a', 'b']))
    monkeypatch.setattr(io.core, 'neural_features', lambda *args, **kwargs: features)
    p = {'protocol_sha256': 'fixture', 'config': {'enrollment_seconds': [30, 240], 'durations': [10, 30, 60],
                                                'adaptation': {'update_development_far': .01}}}
    root = tmp_path / 'run'; folder = root / 'folds/fold_00/eegnet'; folder.mkdir(parents=True)
    io.calibrate_model(root, p, {}, None, 0, 'eegnet')
    for name in ['development_scores.csv', 'initial_development_scores.csv']:
        assert len(pd.read_csv(folder / name)) == 288
    phases = {r.recording_id: np.full((24, 3, 2), 1 if r.participant_id == 'a' else -1, dtype=complex) for r in records}
    store = SimpleNamespace(arrays=lambda record, kind: (phases[record.recording_id],))
    io.calibrate_phase(root, p, {}, store, 0)
    assert len(pd.read_csv(folder.parent / 'phase_development_scores.csv')) == 288
    before = (folder / 'development_scores.csv').read_bytes()
    io.calibrate_model(root, p, {}, None, 0, 'eegnet')
    assert (folder / 'development_scores.csv').read_bytes() == before
    (folder / 'initial_thresholds.json').unlink()
    with pytest.raises(FileExistsError, match='Partial calibration'):
        io.calibrate_model(root, p, {}, None, 0, 'eegnet')
