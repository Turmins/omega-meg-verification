import json
from pathlib import Path

import pytest

from omega_dl import provenance as prov


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(prov, 'inside_extension', lambda path: Path(path).resolve())
    monkeypatch.setattr(prov, 'source_snapshot', lambda: {'core.py': 'frozen', 'io.py': 'current'})
    monkeypatch.setattr(prov, 'current_environment', lambda: {'python': 'new', 'device': 'fixture'})


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize('stage', ['prepare', 'evaluate', 'report', 'train'])
def test_invocations_preserve_legacy_environment_and_do_not_claim_training(tmp_path, isolated, stage):
    legacy = tmp_path / 'environment.json'
    legacy.write_bytes(b'{"python":"historic"}\n')
    before = legacy.read_bytes()
    ids = []
    for environment in ['current-A', 'current-B']:
        prov.current_environment = lambda: {'python': environment}
        with prov.Invocation(tmp_path, stage, {'stage': stage}) as invocation:
            invocation.bind_protocol({'protocol_sha256': 'frozen', 'config': {}, 'code_sha256': {}})
            ids.append(invocation.id)
    assert len(set(ids)) == 2
    for identifier in ids:
        start = read(tmp_path / 'invocations' / identifier / 'start.json')
        assert start['legacy_environment']['interpretation'].endswith('training environment unknown')
        assert start['provenance_type'] == 'invocation'
        assert read(tmp_path / 'invocations' / identifier / 'finish.json')['status'] == 'completed'
    assert legacy.read_bytes() == before
    assert not (tmp_path / 'training_provenance').exists()


def test_unique_record_collision_and_atomic_failure_do_not_replace(tmp_path, isolated, monkeypatch):
    with prov.Invocation(tmp_path, 'prepare', {}, invocation_id='fixed') as first:
        original = (first.path / 'start.json').read_bytes()
    with pytest.raises(FileExistsError):
        with prov.Invocation(tmp_path, 'report', {}, invocation_id='fixed'):
            pass
    assert (first.path / 'start.json').read_bytes() == original
    path = tmp_path / 'immutable.json'
    prov.write_once(path, {'original': True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prov.write_once(path, {'original': False})
    assert path.read_bytes() == before
    monkeypatch.setattr(prov.os, 'link', lambda *args: (_ for _ in ()).throw(PermissionError('fixture')))
    with pytest.raises(PermissionError):
        prov.write_once(tmp_path / 'failed.json', {})
    assert not (tmp_path / 'failed.json').exists()
    assert not list(tmp_path.rglob('*.partial'))


def test_failed_invocation_and_training_attempt_never_create_training_provenance(tmp_path, isolated):
    def failing():
        raise RuntimeError('fixture failure before optimization')
    with pytest.raises(RuntimeError):
        with prov.Invocation(tmp_path, 'train', {}) as invocation:
            invocation.bind_protocol({'protocol_sha256': 'frozen', 'config': {}, 'code_sha256': {}})
            prov.run_training(invocation, failing, tmp_path / 'summary.json', [],
                              fold=0, model='fixture', seed=1, args=())
    assert read(invocation.path / 'finish.json')['status'] == 'failed'
    assert not (tmp_path / 'training_provenance').exists()
    assert read(invocation.path / 'event_0002.json')['kind'] == 'training_failed'


def test_new_confirmed_artifact_binding_and_reuse_link(tmp_path, isolated, monkeypatch):
    # A fixture writer substitutes the trainer; no optimization or neural model runs.
    summary = tmp_path / 'summary.json'
    weight = tmp_path / 'best.pt'
    def completed_fixture():
        monkeypatch.setattr(prov, 'current_environment', lambda: {'threads': 'trainer_effective'})
        summary.write_text(json.dumps({'protocol_sha256': 'frozen', 'epochs': 1, 'completed_utc': 'fixture'}))
        weight.write_bytes(b'fixture checkpoint bytes')
    p = {'protocol_sha256': 'frozen', 'config': {'seed': 1}, 'code_sha256': {}}
    with prov.Invocation(tmp_path, 'train', {}) as invocation:
        invocation.bind_protocol(p)
        prov.run_training(invocation, completed_fixture, summary, [weight],
                          fold=0, model='fixture', seed=1, args=())
    files = list((tmp_path / 'training_provenance').glob('*.json'))
    assert len(files) == 1
    record = read(files[0])
    assert record['provenance_type'] == 'confirmed_completed_training'
    assert record['invocation_id'] == invocation.id
    assert record['environment'] == {'threads': 'trainer_effective'}
    assert record['invocation_start_environment'] == {'python': 'new', 'device': 'fixture'}
    assert record['artifacts_sha256']['best.pt'] == prov.sha256_file(weight)
    before = files[0].read_bytes()
    with prov.Invocation(tmp_path, 'train', {}) as reused:
        reused.bind_protocol(p)
        prov.run_training(reused, lambda: None, summary, [weight],
                          fold=0, model='fixture', seed=1, args=())
    event = read(reused.path / 'event_0001.json')
    assert event['kind'] == 'training_reuse' and len(event['training_records']) == 1
    assert files[0].read_bytes() == before


def test_legacy_training_reuse_stays_unknown(tmp_path, isolated):
    summary = tmp_path / 'summary.json'
    summary.write_text('{"protocol_sha256":"historic"}')
    with prov.Invocation(tmp_path, 'train', {}) as invocation:
        prov.run_training(invocation, lambda: None, summary, [], fold=0, model='fixture', seed=1, args=())
    event = read(invocation.path / 'event_0001.json')
    assert event['training_environment'] == 'unknown' and event['training_records'] == []
    assert not (tmp_path / 'training_provenance').exists()


@pytest.mark.parametrize('identifier', ['', '.', '..', '../outside'])
def test_invocation_id_rejects_escaping_or_empty_identifier(tmp_path, isolated, identifier):
    if identifier == '':
        # Empty explicit IDs select the normal UUID path, which is safe.
        assert prov.Invocation(tmp_path, 'prepare', {}, identifier).id
    else:
        with pytest.raises(ValueError):
            prov.Invocation(tmp_path, 'prepare', {}, identifier)
