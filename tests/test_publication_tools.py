from pathlib import Path
import json
import subprocess
import sys
import pytest
from tools import check_integrity, restore_results
from tools._common import sha256

ROOT = Path(__file__).resolve().parents[1]

def test_every_frozen_core_and_required_reference_byte_is_preserved():
    registry = json.loads((ROOT / 'artifacts/FROZEN_CORE.json').read_text('utf-8'))
    assert len(registry['core_sha256']) == 21
    for name, expected in {**registry['core_sha256'], **registry['reference_sha256']}.items():
        assert sha256(ROOT / name) == expected

@pytest.mark.parametrize('defect', ['modify', 'delete', 'add'])
def test_integrity_detects_changes_and_never_resets_baseline(tmp_path, monkeypatch, defect):
    checkout = tmp_path / 'checkout'; checkout.mkdir()
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    source = inputs / 'signal'; source.write_bytes(b'original')
    monkeypatch.setattr(check_integrity, 'ROOT', checkout)
    check_integrity.snapshot([inputs], 'before.json')
    before = (checkout / 'before.json').read_bytes()
    assert check_integrity.verify(checkout / 'before.json') == 1
    with pytest.raises(FileExistsError):
        check_integrity.snapshot([inputs], 'before.json')
    if defect == 'modify': source.write_bytes(b'changed')
    elif defect == 'delete': source.unlink()
    else: (inputs / 'extra').write_bytes(b'new')
    with pytest.raises(ValueError): check_integrity.verify(checkout / 'before.json')
    assert (checkout / 'before.json').read_bytes() == before

def test_restore_preflight_rejects_mismatch_before_any_destination_write(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'; publication = tmp_path / 'publication'
    archive.mkdir(); publication.mkdir()
    monkeypatch.setattr(restore_results, 'ROOT', publication)
    rel = 'outputs/verification_v1/protocol.json'
    p = archive / rel; p.parent.mkdir(parents=True); p.write_bytes(b'original')
    files = {rel: {'sha256': '0' * 64, 'bytes': 8}}
    with pytest.raises(ValueError, match='hash mismatch'): restore_results.restore(archive, files)
    assert not list(publication.iterdir())

def test_restore_preserves_occupied_files_and_rejects_path_escape(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'; publication = tmp_path / 'publication'
    archive.mkdir(); publication.mkdir()
    monkeypatch.setattr(restore_results, 'ROOT', publication)
    rel = 'outputs/verification_v1/protocol.json'
    source = archive / rel; source.parent.mkdir(parents=True); source.write_bytes(b'original')
    target = publication / rel; target.parent.mkdir(parents=True); target.write_bytes(b'keep')
    files = {rel: {'sha256': sha256(source), 'bytes': 8}}
    with pytest.raises(FileExistsError): restore_results.restore(archive, files)
    assert target.read_bytes() == b'keep'
    with pytest.raises(ValueError): restore_results.restore(archive, {'../escape': files[rel]})

@pytest.mark.parametrize('name', ['11_run_verification.py', '12_report_verification.py', '17_run_adaptive_verification.py'])
def test_help_has_no_filesystem_side_effects_and_training_stage_is_explicit(name):
    runtime = ROOT / 'outputs'
    before = sorted(str(p.relative_to(ROOT)) for p in runtime.rglob('*')) if runtime.exists() else []
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / name), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and 'usage:' in result.stdout
    after = sorted(str(p.relative_to(ROOT)) for p in runtime.rglob('*')) if runtime.exists() else []
    assert after == before
    if name != '12_report_verification.py':
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / name)], capture_output=True, text=True)
        assert result.returncode == 2 and '--stage' in result.stderr

