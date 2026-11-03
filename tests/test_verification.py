"""Behavioral tests for the actual verification boundary and score protocol."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from omega_dl.data.index import Recording
from omega_dl.paths import extension_root
from omega_dl.verification.protocol import digest, freeze_test, read_json, require_test_frozen, save_json, validate_config
from omega_dl.verification.preprocessing import extract_window, filter_window, normalize_window, window_coherency, WindowCache
from omega_dl.verification.scoring import choose_threshold, evaluate_scores, phase_template, score_frame
from omega_dl.verification.experiment import model_for, sample_batch, stage_frames, triplet_loss


def config():
    import yaml
    return yaml.safe_load((extension_root() / "configs" / "verification_v1.yaml").read_text())


def test_large_window_storage_falls_back_to_cpu_without_integer_overflow():
    from omega_dl.verification.experiment import load_windows
    with patch("torch.cuda.mem_get_info", return_value=(4_000_000_000, 8_000_000_000)), patch("torch.empty", side_effect=RuntimeError("allocation intercepted")) as allocation:
        with pytest.raises(RuntimeError, match="allocation intercepted"):
            load_windows(None, [None] * 80, "cuda")
        assert allocation.call_args.kwargs["device"] == "cpu"


def records(n=10):
    return [Recording(f"sub-{i:04d}", i, role, f"ses-{j + 1:02d}", 1, f"sub-{i:04d}_{role}.npy", "") for i in range(n) for j, role in enumerate(["enroll", "probe"])]


def test_query_features_do_not_read_outside_the_window():
    rng = np.random.default_rng(14)
    raw = rng.normal(scale=1e-13, size=(260, 33280)).astype(np.float32)
    original = filter_window(extract_window(raw, 6), "alpha")
    a, b = 1280 + 6 * 1280, 1280 + 7 * 1280
    raw[:, :a] = 1e6
    raw[:, b:] = -1e6
    actual = filter_window(extract_window(raw, 6), "alpha")
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(normalize_window(actual)[0], normalize_window(original)[0])


def test_normalization_is_per_channel_over_time():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(5, 1280)) * np.arange(1, 6)[:, None] * 1e-13
    normalized, _ = normalize_window(x)
    assert np.allclose(np.median(normalized.astype(float), axis=1), 0, atol=1e-3)
    assert np.all(np.std(normalized.astype(float), axis=1) > 0.8)
    with pytest.raises(ValueError):
        normalize_window(np.zeros((5, 1280)))


def test_coherency_and_circular_score_match_independent_calculation():
    rng = np.random.default_rng(4)
    x = filter_window(rng.normal(size=(4, 1280)) * 1e-13, "alpha")
    from scipy.signal.windows import hann
    freqs = np.fft.rfftfreq(256, 1 / 128)
    mask = (freqs >= 8) & (freqs <= 13)
    z = []
    for start in range(0, 1025, 128):
        segment = x[:, start:start + 256]
        z.append(np.fft.rfft((segment - segment.mean(axis=1, keepdims=True)) * hann(256, sym=False), axis=1)[:, mask])
    z = np.concatenate(z, axis=1)
    csd = np.einsum("ik,jk->ij", z, z.conj()) / z.shape[1]
    p = csd.diagonal().real
    expected = csd / np.sqrt(p[:, None] * p[None, :])
    actual = window_coherency(x, "alpha")
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    a, b = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
    score = (np.exp(1j * a) * np.exp(-1j * b)).real.mean()
    assert score == pytest.approx(np.cos(a - b).mean())


def test_attempt_counts_and_duration_aggregation():
    recs = records()
    ids = sorted({r.participant_id for r in recs})
    features = {r.recording_id: np.tile(np.eye(10)[ids.index(r.participant_id)], (24, 1)) for r in recs}
    for duration in [10, 30, 60]:
        frame = score_frame(features, recs, ids, duration, [0, 60, 120, 180], "cnn", 0, "test")
        assert len(frame) == 400 and frame.is_genuine.sum() == 40
        assert np.all(frame.loc[frame.is_genuine == 1, "score"] == 1)
        assert np.all(frame.loc[frame.is_genuine == 0, "score"] == 0)


@pytest.mark.parametrize("scores", [[0.9, 0.8, 0.2, 0.1], [0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.8, 0.9]])
def test_threshold_rule_against_brute_force(scores):
    labels = np.array([1, 1, 0, 0])
    scores = np.array(scores)
    candidates = np.r_[np.unique(scores), np.nextafter(max(scores), np.inf)]
    def key(t):
        decisions = scores >= t
        far = decisions[labels == 0].mean()
        frr = (~decisions[labels == 1]).mean()
        return abs(far - frr), (far + frr) / 2, -t
    threshold = choose_threshold(labels, scores)["threshold"]
    assert threshold == min(candidates, key=key)
    assert evaluate_scores(labels, scores, threshold)["HTER"] == pytest.approx(key(threshold)[1])


def test_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.1, np.nan])


def test_actual_evaluator_refuses_test_before_cache_construction(tmp_path):
    protocol = {"protocol_sha256": "x", "config": config()}
    with patch("omega_dl.verification.protocol.inside_extension", side_effect=lambda p: Path(p)), patch.object(WindowCache, "open", side_effect=AssertionError("must not read test cache")):
        with pytest.raises(RuntimeError, match="Test access requires"):
            stage_frames(tmp_path, protocol, 0, "test", object())


def test_frozen_threshold_change_blocks_test(tmp_path):
    protocol = {"protocol_sha256": "x"}
    folder = tmp_path / "folds" / "fold_00"
    folder.mkdir(parents=True)
    for name in ["best.pt", "untrained.pt", "thresholds.json", "training_summary.json"]:
        (folder / name).write_text("original")
    with patch("omega_dl.verification.protocol.inside_extension", side_effect=lambda p: Path(p)):
        freeze_test(tmp_path, protocol, 0)
        require_test_frozen(tmp_path, protocol, 0)
        (folder / "thresholds.json").write_text("changed")
        with pytest.raises(RuntimeError, match="has changed"):
            require_test_frozen(tmp_path, protocol, 0)


def test_cache_does_not_accept_signature_or_content_mismatch(tmp_path):
    r = records(1)[0]
    p = {"protocol_sha256": "abcdef", "config": config(), "recordings": {r.recording_id: {"sha256": "raw", "filename": r.cache_filename}}}
    with patch("omega_dl.verification.preprocessing.inside_extension", side_effect=lambda v: Path(v)):
        cache = WindowCache(tmp_path, p)
        folder = cache.root / r.recording_id
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"signature": {}}))
        with pytest.raises(ValueError, match="signature"):
            cache.open(r, "cnn")
        (folder / "cnn.npy").write_bytes(b"corrupt")
        (folder / "manifest.json").write_text(json.dumps({"signature": cache.signature(r), "files_sha256": {"cnn.npy": "wrong"}}))
        with pytest.raises(ValueError, match="checksum"):
            cache.open(r, "cnn")


def test_batch_and_loss_have_cross_session_positives():
    recs = records(4)
    ids = sorted({r.participant_id for r in recs})
    indices, labels, recording_ids, sessions = sample_batch(recs, ids, np.random.default_rng(5), 4)
    assert len(indices) == 16 and len(set(indices)) == 16
    z = F_normalized = torch.nn.functional.normalize(torch.randn(16, 8, requires_grad=True), dim=1)
    for warmup in [True, False]:
        loss, active = triplet_loss(z, torch.tensor(labels), torch.tensor(recording_ids), torch.tensor(sessions), .2, warmup)
        assert torch.isfinite(loss) and 0 <= active <= 1
        loss.backward(retain_graph=True)


def test_compact_model_shape_and_parameter_count():
    model = model_for(config()).eval()
    with torch.no_grad():
        result = model(torch.randn(2, 780, 1280))
    assert result.shape == (2, 128)
    assert sum(p.numel() for p in model.parameters()) < 150000
    assert torch.allclose(result.norm(dim=1), torch.ones(2), atol=1e-5)


def test_strict_configuration():
    c = config()
    validate_config(c)
    c["unused_option"] = True
    with pytest.raises(ValueError):
        validate_config(c)
