#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
omega_phasor_student_multiband_T240_v1.py
==========================================

OMEGA MEG student pipeline: multiband phase-only phasor fingerprinting.

INPUT DATASET
-------------
This script reads the compact broadband student export produced by
make_omega_student_broadband_cache_v1_FIXED.py.

Expected data-root contents:

    data-root/
        cache/
            120 *.npy files
        recording_index.csv
        cache_manifest.csv
        channels.txt
        protocol_FROZEN.json
        README.txt

Each NPY file must contain:

    260 channels x 33280 samples
    float32
    fs = 128 Hz
    duration = 260 s

The 260 s interval consists of:

    10 s context + 240 s central analysis interval + 10 s context

The student export is BROADBAND. No delta/theta/alpha/beta/gamma filtering
has been applied before this script.

SCIENTIFIC PROTOCOL
-------------------
For each requested band:

1. Apply a zero-phase 4th-order Butterworth bandpass to the full 260 s.
2. Discard 10 s from both ends, retaining the central 240 s.
3. Divide the 240 s signal into four non-overlapping 60 s windows.
4. Within each 60 s window estimate a band-limited cross-spectrum using:
       periodic Hann window
       2 s segments = 256 samples
       50% overlap
       FFT length = 256
5. Average complex cross-spectral outer products over segments/frequencies,
   then average the four 60 s windows.
6. Form complex coherency

       G = D^(-1/2) S D^(-1/2),  D = diag(S).

7. Keep only inter-channel phase:

       phi_ij = angle(G_ij),
       P_ij   = exp(i phi_ij).

8. Compare enrollment A with probe B using circular cosine:

       score(A,B) = mean_{i<j} cos(phi_ij^A - phi_ij^B).

Prespecified bands:

    delta      1-4 Hz
    theta      4-8 Hz
    alpha      8-13 Hz
    beta       13-30 Hz
    low_gamma  30-45 Hz

No classifier, metric learning, edge threshold, rank scan, or band tuning.

SHARED DERIVED CACHE
--------------------
The first time a recording/band is processed, its complex coherency matrix
is saved under:

    data-root/_derived_phasor_cache_T240_v1/<band>/

Later runs reuse this cache automatically. Thus after single-band runs have
been completed, multiband/fusion runs are much faster and do not repeat the
spectral computation.

Use --force only when intentionally recomputing derived coherency caches.

OUTPUTS
-------
For each requested band:

    score_phase_<band>_circular_cosine.csv
    verification_scores_<band>.csv

Combined outputs:

    multiband_phasor_summary.csv
    subject_ranks_all_bands.csv
    spectral_diagnostics_all_bands.csv
    recording_index_used.csv
    analysis_protocol_FROZEN.json

The summary contains CMC@1, CMC@5, mean/median rank, verification AUC,
and genuine-score statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, signal
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# Frozen scientific constants
# -----------------------------------------------------------------------------

PROTOCOL_VERSION = "omega_phasor_student_multiband_T240_v1"
EXPECTED_EXPORT_PROTOCOL = "omega_student_broadband_T260_v1"

EXPECTED_N_PAIRS = 60
EXPECTED_N_RECORDINGS = 120
EXPECTED_N_CHANNELS = 260

TARGET_SFREQ = 128.0
PAD_SEC = 10.0
T_ANALYSIS_SEC = 240.0
EXPORT_SEC = T_ANALYSIS_SEC + 2.0 * PAD_SEC

EXPECTED_PADDED_SAMPLES = int(round(EXPORT_SEC * TARGET_SFREQ))      # 33280
EXPECTED_ANALYSIS_SAMPLES = int(round(T_ANALYSIS_SEC * TARGET_SFREQ))  # 30720
EXPECTED_PADDED_SHAPE = (EXPECTED_N_CHANNELS, EXPECTED_PADDED_SAMPLES)
EXPECTED_ANALYSIS_SHAPE = (EXPECTED_N_CHANNELS, EXPECTED_ANALYSIS_SAMPLES)

BUTTER_ORDER = 4

N_WINDOWS = 4
WINDOW_SEC = 60.0
WINDOW_SAMPLES = int(round(WINDOW_SEC * TARGET_SFREQ))               # 7680

WELCH_SEG_SEC = 2.0
WELCH_SEG_SAMPLES = int(round(WELCH_SEG_SEC * TARGET_SFREQ))         # 256
WELCH_OVERLAP = 0.50
WELCH_STEP = int(round(WELCH_SEG_SAMPLES * (1.0 - WELCH_OVERLAP)))   # 128

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


# -----------------------------------------------------------------------------
# Dataset validation
# -----------------------------------------------------------------------------

def _sha256_text_object(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def analysis_signature() -> str:
    """Hash only scientific parameters that determine the derived G cache."""
    obj = {
        "protocol": PROTOCOL_VERSION,
        "target_sfreq": TARGET_SFREQ,
        "pad_sec": PAD_SEC,
        "analysis_sec": T_ANALYSIS_SEC,
        "butter_order": BUTTER_ORDER,
        "n_windows": N_WINDOWS,
        "window_sec": WINDOW_SEC,
        "welch_seg_sec": WELCH_SEG_SEC,
        "welch_overlap": WELCH_OVERLAP,
        "welch_nfft": WELCH_SEG_SAMPLES,
        "band_endpoints": "inclusive",
        "bands": BANDS,
    }
    return _sha256_text_object(obj)


def load_student_dataset(data_root: Path) -> Tuple[pd.DataFrame, List[str], dict]:
    data_root = data_root.resolve()

    protocol_path = data_root / "protocol_FROZEN.json"
    index_path = data_root / "recording_index.csv"
    channels_path = data_root / "channels.txt"
    cache_dir = data_root / "cache"

    required = [protocol_path, index_path, channels_path]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required student dataset file missing: {path}")
    if not cache_dir.is_dir():
        raise RuntimeError(f"Student dataset cache directory missing: {cache_dir}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    checks = {
        "protocol_version": EXPECTED_EXPORT_PROTOCOL,
        "n_pairs": EXPECTED_N_PAIRS,
        "n_recordings": EXPECTED_N_RECORDINGS,
        "n_channels": EXPECTED_N_CHANNELS,
        "target_sfreq_Hz": TARGET_SFREQ,
        "export_sec": EXPORT_SEC,
        "export_shape": list(EXPECTED_PADDED_SHAPE),
        "export_dtype": "float32",
    }
    for key, expected in checks.items():
        got = protocol.get(key)
        if got != expected:
            raise RuntimeError(
                f"Student dataset protocol mismatch for {key!r}: "
                f"expected {expected!r}, got {got!r}"
            )

    channels = [
        line.strip()
        for line in channels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(channels) != EXPECTED_N_CHANNELS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_CHANNELS} channels, found {len(channels)}"
        )
    if len(set(channels)) != len(channels):
        raise RuntimeError("channels.txt contains duplicate channel names")

    index = pd.read_csv(index_path)

    needed_cols = {
        "participant_id",
        "pair_index",
        "role",
        "session",
        "run",
        "cache_filename",
    }
    missing_cols = sorted(needed_cols - set(index.columns))
    if missing_cols:
        raise RuntimeError(
            f"recording_index.csv is missing columns: {missing_cols}"
        )

    if len(index) != EXPECTED_N_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_RECORDINGS} index rows, found {len(index)}"
        )
    if index["participant_id"].nunique() != EXPECTED_N_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_PAIRS} unique participants in index"
        )

    role_counts = index["role"].value_counts().to_dict()
    if role_counts.get("enroll", 0) != EXPECTED_N_PAIRS:
        raise RuntimeError("Expected exactly 60 enrollment recordings")
    if role_counts.get("probe", 0) != EXPECTED_N_PAIRS:
        raise RuntimeError("Expected exactly 60 probe recordings")

    # Stable order: pair index, enrollment first, probe second.
    role_order = pd.Categorical(
        index["role"], categories=["enroll", "probe"], ordered=True
    )
    index = (
        index.assign(_role_order=role_order)
        .sort_values(["pair_index", "_role_order"], kind="stable")
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )

    index["student_npy_path"] = index["cache_filename"].map(
        lambda s: str((cache_dir / str(s)).resolve())
    )

    missing_files = [
        p for p in index["student_npy_path"] if not Path(p).is_file()
    ]
    if missing_files:
        raise RuntimeError(
            f"Missing {len(missing_files)} student NPY files. "
            f"First missing file: {missing_files[0]}"
        )

    return index, channels, protocol


def validate_all_npy_files(index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for i, row in index.iterrows():
        p = Path(str(row["student_npy_path"]))
        x = np.load(p, mmap_mode="r", allow_pickle=False)
        ok_shape = tuple(x.shape) == EXPECTED_PADDED_SHAPE
        ok_dtype = x.dtype == np.float32
        finite = bool(np.isfinite(x).all())

        rows.append(
            {
                "participant_id": row["participant_id"],
                "role": row["role"],
                "cache_filename": p.name,
                "shape": str(tuple(x.shape)),
                "dtype": str(x.dtype),
                "finite": finite,
                "valid": bool(ok_shape and ok_dtype and finite),
                "file_bytes": int(p.stat().st_size),
            }
        )
        if verbose:
            print(
                f"[{i+1:03d}/{len(index):03d}] {p.name} | "
                f"shape={tuple(x.shape)} dtype={x.dtype} finite={finite}"
            )

    d = pd.DataFrame(rows)
    if not bool(d["valid"].all()):
        bad = d.loc[~d["valid"]]
        raise RuntimeError(
            "Student dataset integrity check failed. First bad rows:\n"
            + bad.head().to_string(index=False)
        )
    return d


# -----------------------------------------------------------------------------
# Signal processing
# -----------------------------------------------------------------------------

def load_broadband_npy(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if x.shape != EXPECTED_PADDED_SHAPE:
        raise RuntimeError(
            f"{path.name}: expected shape {EXPECTED_PADDED_SHAPE}, got {x.shape}"
        )
    if x.dtype != np.float32:
        raise RuntimeError(
            f"{path.name}: expected float32, got {x.dtype}"
        )
    if not np.isfinite(x).all():
        raise RuntimeError(f"{path.name}: contains non-finite values")

    # Filtering is deliberately carried out in float64.
    return np.asarray(x, dtype=np.float64)


def bandpass_and_crop(
    x_padded: np.ndarray,
    band: Tuple[float, float],
) -> np.ndarray:
    lo, hi = band

    sos = signal.butter(
        BUTTER_ORDER,
        [lo, hi],
        btype="bandpass",
        fs=TARGET_SFREQ,
        output="sos",
    )

    xb = signal.sosfiltfilt(sos, x_padded, axis=1)

    pad_n = int(round(PAD_SEC * TARGET_SFREQ))
    xb = xb[:, pad_n:-pad_n]

    if xb.shape != EXPECTED_ANALYSIS_SHAPE:
        raise RuntimeError(
            f"Unexpected filtered/cropped shape {xb.shape}; "
            f"expected {EXPECTED_ANALYSIS_SHAPE}"
        )

    # Same convention as the research pipeline.
    xb = xb - xb.mean(axis=1, keepdims=True)

    if not np.isfinite(xb).all():
        raise RuntimeError("Non-finite samples after bandpass/cropping")

    return np.asarray(xb, dtype=np.float64)


def welch_band_csd_window(
    xw: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[np.ndarray, int, int]:
    if xw.shape != (EXPECTED_N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"Unexpected 60-s window shape: {xw.shape}")

    xw = np.asarray(xw, dtype=np.float64)
    xw = xw - xw.mean(axis=1, keepdims=True)

    starts = np.arange(
        0,
        WINDOW_SAMPLES - WELCH_SEG_SAMPLES + 1,
        WELCH_STEP,
        dtype=int,
    )
    nseg = len(starts)

    hann = signal.windows.hann(WELCH_SEG_SAMPLES, sym=False).astype(np.float64)

    freqs = np.fft.rfftfreq(WELCH_SEG_SAMPLES, d=1.0 / TARGET_SFREQ)
    fmask = (freqs >= lo - 1e-12) & (freqs <= hi + 1e-12)
    find = np.where(fmask)[0]
    nfreq = len(find)

    if nfreq < 1:
        raise RuntimeError(f"No FFT bins in band {lo}-{hi} Hz")

    Z = np.empty(
        (EXPECTED_N_CHANNELS, nseg * nfreq),
        dtype=np.complex128,
    )

    pos = 0
    for start in starts:
        seg = xw[:, start : start + WELCH_SEG_SAMPLES]
        seg = seg - seg.mean(axis=1, keepdims=True)
        F = np.fft.rfft(seg * hann[None, :], axis=1)
        z = F[:, find]
        Z[:, pos : pos + nfreq] = z
        pos += nfreq

    S = (Z @ Z.conj().T) / float(Z.shape[1])
    S = 0.5 * (S + S.conj().T)
    return S, nseg, nfreq


def coherency_from_band_signal(
    x: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[np.ndarray, dict]:
    Ssum = np.zeros(
        (EXPECTED_N_CHANNELS, EXPECTED_N_CHANNELS),
        dtype=np.complex128,
    )

    nseg = None
    nfreq = None

    for w in range(N_WINDOWS):
        a = w * WINDOW_SAMPLES
        b = a + WINDOW_SAMPLES
        Sw, ns, nf = welch_band_csd_window(x[:, a:b], lo, hi)
        Ssum += Sw
        nseg = ns
        nfreq = nf

    S = Ssum / float(N_WINDOWS)
    S = 0.5 * (S + S.conj().T)

    p = np.real(np.diag(S))
    bad = (~np.isfinite(p)) | (p <= 0)
    if np.any(bad):
        idx = np.where(bad)[0]
        raise RuntimeError(
            f"Non-positive spectral diagonal at channels {idx[:20].tolist()}"
        )

    invsqrt = 1.0 / np.sqrt(p)
    G = S * invsqrt[:, None] * invsqrt[None, :]
    G = 0.5 * (G + G.conj().T)
    np.fill_diagonal(G, 1.0 + 0j)

    herm_rel = float(
        linalg.norm(G - G.conj().T, ord="fro")
        / max(linalg.norm(G, ord="fro"), np.finfo(float).tiny)
    )

    eig = linalg.eigvalsh(G, check_finite=False)
    iu = np.triu_indices(EXPECTED_N_CHANNELS, k=1)
    abs_edges = np.abs(G[iu])

    diagnostics = {
        "welch_segments_per_60s_window": int(nseg),
        "frequency_bins": int(nfreq),
        "spectral_observations_per_window": int(nseg * nfreq),
        "G_hermitian_relative_residual": herm_rel,
        "G_eigen_min": float(eig[0]),
        "G_eigen_median": float(np.median(eig)),
        "G_eigen_max": float(eig[-1]),
        "G_abs_edge_mean": float(np.mean(abs_edges)),
        "G_abs_edge_median": float(np.median(abs_edges)),
        "G_abs_edge_q95": float(np.quantile(abs_edges, 0.95)),
    }

    return G.astype(np.complex64), diagnostics


# -----------------------------------------------------------------------------
# Shared derived coherency cache
# -----------------------------------------------------------------------------

def recording_id(row: pd.Series) -> str:
    participant = str(row["participant_id"])
    role = str(row["role"])
    session = str(row["session"])
    run = int(row["run"])
    return f"{participant}__{role}__{session}__run-{run:02d}"


def derived_cache_path(cache_root: Path, band: str, row: pd.Series) -> Path:
    return cache_root / band / f"{recording_id(row)}.npz"


def load_derived_cache(path: Path, band: str) -> Tuple[np.ndarray, dict]:
    try:
        with np.load(path, allow_pickle=False) as a:
            sig = str(a["analysis_signature"].item())
            stored_band = str(a["band"].item())
            G = np.asarray(a["G"], dtype=np.complex128)
            diagnostics = json.loads(str(a["diagnostics_json"].item()))
    except Exception as exc:
        raise RuntimeError(f"Cannot read derived cache {path}: {exc}") from exc

    if sig != analysis_signature():
        raise RuntimeError(
            f"Derived cache protocol mismatch: {path}. "
            "Delete the derived cache or use --force."
        )
    if stored_band != band:
        raise RuntimeError(f"Derived cache band mismatch in {path}")
    if G.shape != (EXPECTED_N_CHANNELS, EXPECTED_N_CHANNELS):
        raise RuntimeError(f"Bad G shape in {path}: {G.shape}")
    if not np.isfinite(G.real).all() or not np.isfinite(G.imag).all():
        raise RuntimeError(f"Non-finite G in {path}")

    return G, diagnostics


def save_derived_cache(
    path: Path,
    band: str,
    G: np.ndarray,
    diagnostics: dict,
    row: pd.Series,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "analysis_signature": np.asarray(analysis_signature()),
        "band": np.asarray(band),
        "recording_id": np.asarray(recording_id(row)),
        "source_cache_filename": np.asarray(str(row["cache_filename"])),
        "G": np.asarray(G, dtype=np.complex64),
        "diagnostics_json": np.asarray(json.dumps(diagnostics, sort_keys=True)),
    }

    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)

    # Ordinary load (not mmap) avoids Windows rename locking.
    with np.load(tmp, allow_pickle=False) as a:
        test_shape = tuple(a["G"].shape)
        test_sig = str(a["analysis_signature"].item())

    if test_shape != (EXPECTED_N_CHANNELS, EXPECTED_N_CHANNELS):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary derived cache validation failed: {tmp}")
    if test_sig != analysis_signature():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary derived cache signature failed: {tmp}")

    tmp.replace(path)


# -----------------------------------------------------------------------------
# Biometric scoring
# -----------------------------------------------------------------------------

def aligned_indices(index: pd.DataFrame):
    e = index[index["role"] == "enroll"].sort_values("pair_index", kind="stable")
    p = index[index["role"] == "probe"].sort_values("pair_index", kind="stable")

    if len(e) != EXPECTED_N_PAIRS or len(p) != EXPECTED_N_PAIRS:
        raise RuntimeError("Expected 60 enrollment and 60 probe rows")

    if not np.array_equal(e["pair_index"].to_numpy(), p["pair_index"].to_numpy()):
        raise RuntimeError("Enrollment/probe pair_index mismatch")

    if not np.array_equal(
        e["participant_id"].astype(str).to_numpy(),
        p["participant_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Enrollment/probe participant mismatch")

    return (
        e.index.to_numpy(dtype=int),
        p.index.to_numpy(dtype=int),
        e["participant_id"].astype(str).tolist(),
        e["pair_index"].to_numpy(dtype=int),
    )


def score_phase_circular(G_e: np.ndarray, G_p: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(EXPECTED_N_CHANNELS, k=1)

    phi_e = np.angle(G_e[:, iu[0], iu[1]])
    phi_p = np.angle(G_p[:, iu[0], iu[1]])

    P_e = np.exp(1j * phi_e)
    P_p = np.exp(1j * phi_p)

    score = np.real(P_e @ P_p.conj().T) / float(P_e.shape[1])
    return np.asarray(score, dtype=np.float64)


def biometric_metrics(
    score: np.ndarray,
    participants: List[str],
    pair_index: np.ndarray,
    band: str,
) -> Tuple[dict, pd.DataFrame]:
    n = len(participants)

    if score.shape != (n, n):
        raise RuntimeError(f"Unexpected score shape for {band}: {score.shape}")
    if not np.isfinite(score).all():
        raise RuntimeError(f"Non-finite scores for {band}")

    ranks = np.empty(n, dtype=int)
    genuine = np.diag(score).copy()

    # Rows = enrollment gallery; columns = probes.
    for j in range(n):
        order = np.argsort(-score[:, j], kind="stable")
        ranks[j] = int(np.where(order == j)[0][0]) + 1

    labels = np.eye(n, dtype=int).ravel()
    auc = float(roc_auc_score(labels, score.ravel()))

    summary = {
        "band": band,
        "band_lo_Hz": BANDS[band][0],
        "band_hi_Hz": BANDS[band][1],
        "representation": "Phase(G)",
        "matcher": "phasor_circular_cosine",
        "N": n,
        "CMC1": float(np.mean(ranks <= 1)),
        "CMC5": float(np.mean(ranks <= 5)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "verification_AUC": auc,
        "genuine_score_mean": float(np.mean(genuine)),
        "genuine_score_median": float(np.median(genuine)),
        "genuine_score_sd": float(np.std(genuine, ddof=1)),
    }

    ranks_df = pd.DataFrame(
        {
            "participant_id": participants,
            "pair_index": pair_index,
            "band": band,
            "rank": ranks,
            "genuine_score": genuine,
        }
    )

    return summary, ranks_df


def save_score_matrix(score: np.ndarray, participants: List[str], path: Path) -> None:
    d = pd.DataFrame(score, index=participants, columns=participants)
    d.index.name = "enroll_participant"
    d.to_csv(path)


def verification_score_table(score: np.ndarray, participants: List[str], band: str) -> pd.DataFrame:
    n = len(participants)
    rows = []
    for i, enroll_id in enumerate(participants):
        for j, probe_id in enumerate(participants):
            rows.append(
                {
                    "band": band,
                    "enroll_participant": enroll_id,
                    "probe_participant": probe_id,
                    "is_genuine": int(i == j),
                    "score": float(score[i, j]),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main reusable analysis API
# -----------------------------------------------------------------------------

def make_analysis_protocol(
    data_root: Path,
    out_dir: Path,
    cache_root: Path,
    selected_bands: List[str],
    export_protocol: dict,
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_export_protocol": export_protocol.get("protocol_version"),
        "source_export_shape": export_protocol.get("export_shape"),
        "source_export_dtype": export_protocol.get("export_dtype"),
        "pairs": EXPECTED_N_PAIRS,
        "recordings": EXPECTED_N_RECORDINGS,
        "channels": EXPECTED_N_CHANNELS,
        "target_sfreq_Hz": TARGET_SFREQ,
        "padding_sec_each_side": PAD_SEC,
        "analysis_sec": T_ANALYSIS_SEC,
        "selected_bands": {b: list(BANDS[b]) for b in selected_bands},
        "bandpass": {
            "family": "Butterworth",
            "order": BUTTER_ORDER,
            "application": "scipy.signal.sosfiltfilt",
            "phase": "zero-phase forward-backward",
        },
        "spectral_estimator": {
            "recording_windows": N_WINDOWS,
            "window_sec": WINDOW_SEC,
            "welch_segment_sec": WELCH_SEG_SEC,
            "welch_overlap_fraction": WELCH_OVERLAP,
            "welch_window": "periodic Hann",
            "nfft_samples": WELCH_SEG_SAMPLES,
            "frequency_resolution_Hz": TARGET_SFREQ / WELCH_SEG_SAMPLES,
            "band_endpoints": "inclusive",
        },
        "coherency": "G = D^(-1/2) S D^(-1/2)",
        "representation": "unit phasors exp(i angle(G_ij))",
        "matcher": "mean circular cosine over strict-upper-triangle edges",
        "no_edge_threshold": True,
        "no_classifier": True,
        "no_metric_learning": True,
        "no_rank_scan": True,
        "analysis_signature": analysis_signature(),
        "paths": {
            "data_root": str(data_root.resolve()),
            "out_dir": str(out_dir.resolve()),
            "shared_cache_root": str(cache_root.resolve()),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }


def run_band_analysis(
    data_root: Path,
    out_dir: Path,
    selected_bands: List[str],
    cache_root: Path | None = None,
    force: bool = False,
    quiet_cache: bool = False,
) -> dict:
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()

    invalid = [b for b in selected_bands if b not in BANDS]
    if invalid:
        raise RuntimeError(f"Unknown bands: {invalid}; allowed: {list(BANDS)}")
    if len(selected_bands) != len(set(selected_bands)):
        raise RuntimeError("Duplicate bands requested")
    if not selected_bands:
        raise RuntimeError("At least one band must be requested")

    if cache_root is None:
        cache_root = data_root / "_derived_phasor_cache_T240_v1"
    cache_root = cache_root.resolve()

    index, channels, export_protocol = load_student_dataset(data_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    protocol = make_analysis_protocol(
        data_root, out_dir, cache_root, selected_bands, export_protocol
    )
    (out_dir / "analysis_protocol_FROZEN.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )

    index.to_csv(out_dir / "recording_index_used.csv", index=False)

    print("\n============================================")
    print("OMEGA STUDENT MULTIBAND PHASOR FINGERPRINT")
    print("============================================")
    print(f"data root:    {data_root}")
    print(f"pairs:        {EXPECTED_N_PAIRS}")
    print(f"recordings:   {EXPECTED_N_RECORDINGS}")
    print(f"channels:     {EXPECTED_N_CHANNELS}")
    print(f"input:        {EXPORT_SEC:g} s broadband @ {TARGET_SFREQ:g} Hz")
    print(f"analysis:     central {T_ANALYSIS_SEC:g} s")
    print(f"Welch:        {WELCH_SEG_SEC:g} s Hann, {100*WELCH_OVERLAP:g}% overlap")
    print(f"derived cache:{cache_root}")
    print("bands:")
    for b in selected_bands:
        lo, hi = BANDS[b]
        print(f"  {b:10s} {lo:g}-{hi:g} Hz")
    print("matcher:      phase circular cosine")
    print("classifier:   none")

    # One matrix array per requested band. 5 bands together are still modest
    # (~325 MB as complex64 for 120 x 260 x 260).
    G_all = {
        band: np.empty(
            (EXPECTED_N_RECORDINGS, EXPECTED_N_CHANNELS, EXPECTED_N_CHANNELS),
            dtype=np.complex64,
        )
        for band in selected_bands
    }

    diag_rows = []

    for i, row in index.iterrows():
        rec_id = recording_id(row)
        print(f"\n[{i+1:03d}/{EXPECTED_N_RECORDINGS:03d}] {rec_id}")

        # Check which requested bands already have a reusable G cache.
        G_by_band: Dict[str, np.ndarray] = {}
        diag_by_band: Dict[str, dict] = {}
        missing_bands: List[str] = []

        for band in selected_bands:
            cp = derived_cache_path(cache_root, band, row)
            if cp.is_file() and not force:
                try:
                    G, d = load_derived_cache(cp, band)
                    G_by_band[band] = G
                    diag_by_band[band] = d
                    if not quiet_cache:
                        print(f"  {band:10s} CACHE REUSE")
                except RuntimeError as exc:
                    print(f"  {band:10s} CACHE INVALID -> recompute ({exc})")
                    missing_bands.append(band)
            else:
                missing_bands.append(band)

        if missing_bands:
            x_padded = load_broadband_npy(Path(str(row["student_npy_path"])))

            for band in missing_bands:
                lo, hi = BANDS[band]
                print(f"  {band:10s} COMPUTE {lo:g}-{hi:g} Hz", flush=True)

                xb = bandpass_and_crop(x_padded, (lo, hi))
                G, d = coherency_from_band_signal(xb, lo, hi)

                cp = derived_cache_path(cache_root, band, row)
                save_derived_cache(cp, band, G, d, row)

                G_by_band[band] = np.asarray(G, dtype=np.complex128)
                diag_by_band[band] = d

                print(
                    f"               bins={d['frequency_bins']:2d} | "
                    f"|G|med={d['G_abs_edge_median']:.4f} | "
                    f"eigmin={d['G_eigen_min']:.3e} | SAVED"
                )

            del x_padded

        for band in selected_bands:
            G_all[band][i] = np.asarray(G_by_band[band], dtype=np.complex64)
            lo, hi = BANDS[band]
            d = diag_by_band[band]
            diag_rows.append(
                {
                    "recording_id": rec_id,
                    "participant_id": row["participant_id"],
                    "pair_index": int(row["pair_index"]),
                    "role": row["role"],
                    "session": row["session"],
                    "run": int(row["run"]),
                    "cache_filename": row["cache_filename"],
                    "band": band,
                    "band_lo_Hz": lo,
                    "band_hi_Hz": hi,
                    **d,
                }
            )

    diagnostics_df = pd.DataFrame(diag_rows)
    diagnostics_df.to_csv(
        out_dir / "spectral_diagnostics_all_bands.csv", index=False
    )

    ei, pi, participants, pair_index = aligned_indices(index)

    summary_rows = []
    rank_tables = []
    scores: Dict[str, np.ndarray] = {}

    print("\n============================================")
    print("BAND RESULTS")
    print("============================================")

    for band in selected_bands:
        Ge = np.asarray(G_all[band][ei], dtype=np.complex128)
        Gp = np.asarray(G_all[band][pi], dtype=np.complex128)

        score = score_phase_circular(Ge, Gp)
        scores[band] = score

        save_score_matrix(
            score,
            participants,
            out_dir / f"score_phase_{band}_circular_cosine.csv",
        )
        verification_score_table(score, participants, band).to_csv(
            out_dir / f"verification_scores_{band}.csv", index=False
        )

        summary, ranks = biometric_metrics(score, participants, pair_index, band)
        summary_rows.append(summary)
        rank_tables.append(ranks)

        print(
            f"{band:10s} | AUC={summary['verification_AUC']:.6f} | "
            f"CMC1={summary['CMC1']:.6f} | CMC5={summary['CMC5']:.6f} | "
            f"mean_rank={summary['mean_rank']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    ranks_df = pd.concat(rank_tables, ignore_index=True)

    summary_df.to_csv(out_dir / "multiband_phasor_summary.csv", index=False)
    ranks_df.to_csv(out_dir / "subject_ranks_all_bands.csv", index=False)

    print("\nSaved summary:")
    print(out_dir / "multiband_phasor_summary.csv")
    print("\nDONE.")

    return {
        "index": index,
        "participants": participants,
        "pair_index": pair_index,
        "scores": scores,
        "summary": summary_df,
        "ranks": ranks_df,
        "diagnostics": diagnostics_df,
        "cache_root": cache_root,
        "channels": channels,
        "export_protocol": export_protocol,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Student dataset directory containing cache/, recording_index.csv, "
            "channels.txt, and protocol_FROZEN.json."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        default=Path("results") / PROTOCOL_VERSION,
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Shared derived coherency cache. Default: "
            "<data-root>/_derived_phasor_cache_T240_v1"
        ),
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=list(BANDS.keys()),
        choices=list(BANDS.keys()),
        help="Bands to compute. Default: all five prespecified bands.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute derived coherency caches for requested bands.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all 120 student NPY files and stop without analysis.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.validate_only:
            index, _, protocol = load_student_dataset(args.data_root)
            print("=== STUDENT DATASET VALIDATION ===")
            print(f"protocol: {protocol.get('protocol_version')}")
            d = validate_all_npy_files(index, verbose=True)
            print("\nVALIDATION PASSED")
            print(f"valid files: {int(d['valid'].sum())}/{len(d)}")
            print(
                f"total size: {d['file_bytes'].sum()/1e9:.3f} GB decimal "
                f"({d['file_bytes'].sum()/(1024**3):.3f} GiB)"
            )
            return 0

        run_band_analysis(
            data_root=args.data_root,
            out_dir=args.out_dir,
            selected_bands=list(args.bands),
            cache_root=args.cache_root,
            force=bool(args.force),
        )
        return 0

    except Exception as exc:
        print(f"\nSTOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
