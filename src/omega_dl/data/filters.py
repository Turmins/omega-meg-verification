"""Baseline-compatible full-recording bandpass and central crop."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from omega_dl.paths import extension_root

TARGET_SFREQ = 128.0
PAD_SAMPLES = 1280
EXPECTED_RAW_SHAPE = (260, 33280)
EXPECTED_ANALYSIS_SHAPE = (260, 30720)


@lru_cache(maxsize=1)
def baseline_bands() -> dict[str, tuple[float, float]]:
    path = extension_root() / "baseline_reference" / "omega_phasor_student_multiband_T240_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target == "BANDS" and value is not None:
            parsed = ast.literal_eval(value)
            return {key: (float(pair[0]), float(pair[1])) for key, pair in parsed.items()}
    raise RuntimeError("Could not extract BANDS from the reference program")


BANDS = baseline_bands()


def validate_band_names(names: list[str]) -> None:
    unknown = [name for name in names if name not in BANDS]
    if unknown:
        raise ValueError(f"Unknown bands: {unknown}; available bands: {list(BANDS)}")


def load_raw(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.shape != EXPECTED_RAW_SHAPE or array.dtype != np.float32:
        raise ValueError(f"{path.name}: expected float32 {EXPECTED_RAW_SHAPE}, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: non-finite values")
    return np.asarray(array, dtype=np.float64)


def bandpass_and_crop(x_padded: np.ndarray, band: str | tuple[float, float], order: int = 4) -> np.ndarray:
    if isinstance(band, str):
        band = BANDS[band]
    sos = signal.butter(order, list(band), btype="bandpass", fs=TARGET_SFREQ, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(x_padded, dtype=np.float64), axis=1)
    cropped = filtered[:, PAD_SAMPLES:-PAD_SAMPLES]
    if cropped.shape != EXPECTED_ANALYSIS_SHAPE:
        raise ValueError(f"Expected cropped shape {EXPECTED_ANALYSIS_SHAPE}, got {cropped.shape}")
    cropped = cropped - cropped.mean(axis=1, keepdims=True)
    if not np.isfinite(cropped).all():
        raise ValueError("Non-finite values after filtering")
    return np.asarray(cropped, dtype=np.float64)
