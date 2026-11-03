"""Window-local filtering, MAD normalization and signed read-only caches."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal
from threadpoolctl import threadpool_limits

from omega_dl.paths import inside_extension, resolve_data_root, sha256_file
from omega_dl.verification.protocol import digest, read_json, save_json

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
EDGES = np.triu_indices(260, 1)


def extract_window(raw, index):
    start = 1280 + index * 1280
    if not 0 <= index < 24:
        raise ValueError("Window index outside central 240 seconds")
    window = np.array(raw[:, start:start + 1280], dtype=np.float64, copy=True)
    if window.shape != (260, 1280) or not np.isfinite(window).all():
        raise ValueError("Invalid MEG window")
    return window


def filter_window(window, band):
    if window.shape[-1] != 1280 or not np.isfinite(window).all():
        raise ValueError("Expected a finite ten-second window")
    sos = signal.butter(4, BANDS[band], btype="bandpass", fs=128.0, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(window, dtype=np.float64), axis=1, padtype="odd")
    return filtered - filtered.mean(axis=1, keepdims=True)


def normalize_window(filtered, clip=8.0):
    median = np.median(filtered, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(filtered - median), axis=1, keepdims=True)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-30):
        raise ValueError("Flat or invalid channel; no silent attempt removal")
    values = (filtered - median) / scale
    clipped = float(np.mean(np.abs(values) > clip))
    normalized = np.clip(values, -clip, clip).astype(np.float16)
    if not np.isfinite(normalized).all() or float(normalized.astype(np.float32).std()) <= 0.5:
        raise ValueError("Unsafe normalized cache")
    return normalized, clipped


def window_coherency(filtered, band):
    lo, hi = BANDS[band]
    freqs = np.fft.rfftfreq(256, 1 / 128)
    mask = (freqs >= lo - 1e-12) & (freqs <= hi + 1e-12)
    hann = signal.windows.hann(256, sym=False)
    segments = np.stack([filtered[:, start:start + 256] for start in range(0, 1280 - 256 + 1, 128)], axis=1)
    segments -= segments.mean(axis=2, keepdims=True)
    spectra = np.fft.rfft(segments * hann, axis=2)[..., mask]
    spectra = spectra.reshape(filtered.shape[0], -1)
    csd = spectra @ spectra.conj().T / spectra.shape[1]
    power = csd.diagonal().real.copy()
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise ValueError("Invalid spectral power")
    g = csd / np.sqrt(power[:, None] * power[None, :])
    return (0.5 * (g + g.conj().T)).astype(np.complex64)


class WindowCache:
    def __init__(self, run_root, protocol):
        self.root = inside_extension(Path(run_root) / "cache" / protocol["protocol_sha256"][:16])
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol
        self.checked = set()

    def signature(self, record):
        return {"protocol_sha256": self.protocol["protocol_sha256"], "source": self.protocol["recordings"][record.recording_id], "recording_id": record.recording_id, "shape": [24, 780, 1280], "normalization": "window_channel_mad", "clip_mad": self.protocol["config"]["clip_mad"]}

    def ensure(self, record):
        folder = inside_extension(self.root / record.recording_id)
        expected = self.signature(record)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["signature"] != expected:
                raise ValueError("Cache signature mismatch")
            if record.recording_id not in self.checked:
                if any(sha256_file(folder / name) != value for name, value in manifest["files_sha256"].items()):
                    raise ValueError("Cache checksum mismatch")
                self.checked.add(record.recording_id)
            return folder
        raw_path = resolve_data_root(self.protocol["config"]["data_root"]) / "cache" / record.cache_filename
        if sha256_file(raw_path) != expected["source"]["sha256"]:
            raise ValueError("Raw input changed since protocol freeze")
        raw = np.load(raw_path, mmap_mode="r")
        if raw.shape != (260, 33280) or raw.dtype != np.float32:
            raise ValueError("Invalid raw shape/dtype")
        folder.mkdir(parents=True, exist_ok=True)
        x = np.lib.format.open_memmap(folder / "cnn.npy.partial", mode="w+", dtype=np.float16, shape=(24, 780, 1280))
        g = np.lib.format.open_memmap(folder / "phase.npy.partial", mode="w+", dtype=np.complex64, shape=(24, 3, len(EDGES[0])))
        clipping = []
        with threadpool_limits(limits=2):
            for window_index in range(24):
                window = extract_window(raw, window_index)
                for band_index, band in enumerate(self.protocol["config"]["bands"]):
                    filtered = filter_window(window, band)
                    normalized, clipped = normalize_window(filtered, self.protocol["config"]["clip_mad"])
                    x[window_index, band_index * 260:(band_index + 1) * 260] = normalized
                    g[window_index, band_index] = window_coherency(filtered, band)[EDGES]
                    clipping.append(clipped)
        x.flush()
        g.flush()
        del x, g, raw
        for name in ["cnn.npy", "phase.npy"]:
            (folder / (name + ".partial")).replace(folder / name)
        save_json(manifest_path, {"signature": expected, "files_sha256": {name: sha256_file(folder / name) for name in ["cnn.npy", "phase.npy"]}, "clipped_fraction_mean": float(np.mean(clipping)), "clipped_fraction_max": float(np.max(clipping))})
        self.checked.add(record.recording_id)
        return folder

    def open(self, record, kind):
        if kind not in {"cnn", "phase"}:
            raise ValueError(kind)
        folder = self.ensure(record)
        return np.load(folder / f"{kind}.npy", mmap_mode="r")
