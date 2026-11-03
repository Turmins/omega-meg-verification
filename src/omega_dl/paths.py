"""Path resolution and write-scope guards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def extension_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(value: str | Path, base_dir: Path | None = None) -> Path:
    base = (base_dir or extension_root()).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {candidate}")
    return candidate


def inside_extension(path: str | Path) -> Path:
    root = extension_root().resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Writes are restricted to the extension directory: {resolved}") from exc
    return resolved


def ensure_output_tree() -> dict[str, Path]:
    root = extension_root()
    paths = {
        "logs": root / "outputs" / "logs",
        "checkpoints": root / "outputs" / "checkpoints",
        "metrics": root / "outputs" / "metrics",
        "figures": root / "outputs" / "figures",
        "caches": root / "outputs" / "caches",
        "runs": root / "outputs" / "runs",
    }
    for path in paths.values():
        inside_extension(path).mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_external_files(root: Path) -> Iterable[Path]:
    extension = extension_root().resolve()
    for current, dirs, files in os.walk(root, followlinks=True):
        current_path = Path(current)
        dirs[:] = [d for d in dirs if (current_path / d).resolve() != extension]
        for name in files:
            path = current_path / name
            if extension not in path.resolve().parents:
                yield path


def external_snapshot(project_root: Path | None = None) -> dict[str, dict[str, int | str]]:
    root = (project_root or extension_root().parent).resolve()
    snapshot: dict[str, dict[str, int | str]] = {}
    extension = extension_root().resolve()
    for path in _iter_external_files(root):
        if extension in path.resolve().parents:
            continue
        stat = path.stat()
        snapshot[str(path.relative_to(root))] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        }
    return snapshot


def write_json(path: Path, value: object) -> None:
    target = inside_extension(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
