"""Entry point wiring is exercised with fixtures, without experiments."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

from omega_dl.paths import extension_root
from omega_dl import provenance


@pytest.mark.parametrize('name', ['11_run_verification.py', '17_run_adaptive_verification.py'])
@pytest.mark.parametrize('stage', ['prepare', 'evaluate'])
def test_runner_invocation_records_and_legacy_environment(tmp_path, monkeypatch, name, stage):
    root = extension_root()
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('p0_runner_fixture', root / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'inside_extension', lambda p: Path(p))
    monkeypatch.setattr(provenance, 'inside_extension', lambda p: Path(p))
    monkeypatch.setattr(provenance, 'current_environment', lambda: {'fixture_environment': 'now'})
    protocol = {'protocol_sha256': 'frozen', 'code_sha256': {},
                'config': {'train': {'cpu_threads': 1}, 'base_run': 'fixture', 'graph_neighbors': 12}}
    legacy = tmp_path / 'environment.json'
    legacy.write_bytes(b'{"python":"historical"}\n')
    monkeypatch.setattr(sys, 'argv', [name, '--out', str(tmp_path), '--stage', stage, '--folds', '0'])
    barriers, evaluations = [], []
    if name.startswith('11'):
        monkeypatch.setattr(module, 'prepare', lambda *a: protocol)
        monkeypatch.setattr(module, 'WindowCache', lambda *a: object())
        monkeypatch.setattr(module, 'require_test_frozen', lambda *a: barriers.append(a[-1]))
        monkeypatch.setattr(module, 'evaluate_fold', lambda *a: evaluations.append(a[2]))
    else:
        monkeypatch.setattr(module, 'prepare', lambda *a: (protocol, {}))
        monkeypatch.setattr(module, 'FeatureStore', lambda *a: object())
        monkeypatch.setattr(module, 'require_frozen', lambda *a: barriers.append(True))
        monkeypatch.setattr(module, 'evaluate_fold', lambda *a: evaluations.append(a[-1]))
    before = legacy.read_bytes()
    module.main()
    assert legacy.read_bytes() == before
    records = list((tmp_path / 'invocations').glob('*/start.json'))
    assert len(records) == 1
    start = json.loads(records[0].read_text())
    assert start['stage'] == stage and start['environment'] == {'fixture_environment': 'now'}
    assert not (tmp_path / 'training_provenance').exists()
    if stage == 'evaluate':
        assert barriers and evaluations == [0]
    else:
        assert not barriers and not evaluations
