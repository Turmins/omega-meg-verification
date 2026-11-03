"""Local replay import must fail closed without touching source records."""
from pathlib import Path
import importlib.util
import json
import sys
import pytest


@pytest.fixture
def importer(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools" / "prepare_saved_demo.py"
    spec = importlib.util.spec_from_file_location("clean_saved_demo_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    return module, root


def invoke(module, archive, output, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_saved_demo.py", "--archive-root", str(archive), "--output-dir", str(output)])
    module.main()


def test_missing_archive_creates_no_output(importer, tmp_path, monkeypatch):
    module, _ = importer
    output = tmp_path / "new-output"
    with pytest.raises(FileNotFoundError):
        invoke(module, tmp_path / "missing", output, monkeypatch)
    assert not output.exists()


def test_output_cannot_write_into_original_archive(importer, tmp_path, monkeypatch):
    module, _ = importer
    archive = tmp_path / "archive"
    archive.mkdir()
    with pytest.raises(ValueError, match="original archive"):
        invoke(module, archive, archive / "new-output", monkeypatch)
    assert list(archive.iterdir()) == []


def test_occupied_output_is_preserved(importer, tmp_path, monkeypatch):
    module, _ = importer
    archive = tmp_path / "archive"
    archive.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        invoke(module, archive, output, monkeypatch)
    assert sentinel.read_bytes() == b"preserve"


def test_changed_input_hash_rejects_before_output(importer, tmp_path, monkeypatch):
    module, root = importer
    archive = tmp_path / "archive"
    source = archive / "outputs/verification_v1/report/demo.html"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic changed input")
    manifest = root / "artifacts/EXTERNAL_ARTIFACTS.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"saved_V1_demo_files": {"demo.html": {
        "source_relative_path": "outputs/verification_v1/report/demo.html",
        "sha256": "0" * 64, "size_bytes": source.stat().st_size}}}))
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="hash/size mismatch"):
        invoke(module, archive, output, monkeypatch)
    assert not output.exists()
    assert source.read_bytes() == b"synthetic changed input"
