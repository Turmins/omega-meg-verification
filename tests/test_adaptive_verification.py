"""Tests for graph semantics, session separation, adaptation and frozen access."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omega_dl.adaptive_verification.models import EEGNetEncoder, GraphEncoder, prototype_loss
from omega_dl.adaptive_verification.features import graph_features
from omega_dl.adaptive_verification.online import AdaptiveTemplate, strict_update_threshold
from omega_dl.adaptive_verification.experiment import episode, records_for
from omega_dl.adaptive_verification.evaluation import schedule
from omega_dl.adaptive_verification.protocol import locked_files
from omega_dl.verification.preprocessing import EDGES
from omega_dl.verification.protocol import load_protocol
from omega_dl.paths import extension_root


def test_eegnet_forward_and_gradients():
    torch.set_num_threads(2)
    model = EEGNetEncoder()
    z = model(torch.randn(2, 3, 260, 1280))
    assert z.shape == (2, 64)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(2))
    z[:, 0].sum().backward()
    assert model.features[0].weight.grad.abs().sum() > 0


def test_graph_preserves_signed_phase_and_window_independence():
    phases = np.arange(260) * 0.017
    matrix = np.exp(1j * (phases[:, None] - phases[None, :])).astype(np.complex64)
    values = np.tile(matrix[EDGES][None, None], (2, 3, 1))
    nodes, neighbors, edges = graph_features(values, 259)
    forward = np.flatnonzero(neighbors[0, 0] == 1)[0]
    reverse = np.flatnonzero(neighbors[0, 1] == 0)[0]
    np.testing.assert_allclose(edges[0, 0, forward, :3], edges[0, 1, reverse, :3])
    np.testing.assert_allclose(edges[0, 0, forward, 3:6], -edges[0, 1, reverse, 3:6])
    assert all(i not in neighbors[0, i] for i in range(260))
    changed = values.copy()
    changed[1] *= 0.1j
    for before, after in zip((nodes, neighbors, edges), graph_features(changed, 259)):
        np.testing.assert_array_equal(before[0], after[0])


def test_graph_encoder_uses_edge_features():
    model = GraphEncoder().eval()
    nodes = torch.randn(2, 260, 12)
    adjacent = (torch.arange(260)[:, None] + torch.arange(1, 4)[None]) % 260
    adjacent = adjacent[None].expand(2, -1, -1)
    edges = torch.randn(2, 260, 3, 9, requires_grad=True)
    z = model(nodes, adjacent, edges)
    assert z.shape == (2, 64)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(2))
    z[:, 0].sum().backward()
    assert edges.grad.abs().sum() > 0


def test_decision_precedes_update_and_rejections_leave_template_unchanged():
    state = AdaptiveTemplate(np.array([1., 0.]), .75, .9, "accepted", .2)
    result = state.verify(np.array([.8, .6]), "first")
    assert result["score"] == pytest.approx(.8)
    assert result["accepted"] and result["updated"]
    assert float(state.current @ np.array([.8, .6])) > result["score"]
    before = state.current.copy()
    assert not state.verify(np.array([-1., 0.]), "second")["updated"]
    np.testing.assert_array_equal(before, state.current)


def test_guarded_update_also_requires_original_anchor_agreement():
    state = AdaptiveTemplate(np.array([1., 0.]), .7, .9, current=np.array([.8, .6]))
    result = state.verify(np.array([.8, .6]))
    assert result["accepted"] and not result["updated"]


def test_static_policy_and_persistent_replay_guard():
    state = AdaptiveTemplate(np.array([1., 0.]), .7, .9, "static")
    assert not state.verify(np.array([1., 0.]), "record:0:30")["updated"]
    restored = AdaptiveTemplate.restore(state.state())
    with pytest.raises(ValueError, match="already consumed"):
        restored.verify(np.array([1., 0.]), "record:0:30")
    np.testing.assert_array_equal(state.current, restored.current)


def test_update_threshold_handles_ties_without_exceeding_budget():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([.9, .9, .8, .95])
    threshold = strict_update_threshold(labels, scores, .7, 1 / 3)
    assert threshold > .9
    assert (scores[labels == 0] >= threshold).sum() <= 1


def test_prototype_training_episode_uses_distinct_sessions():
    records = [SimpleNamespace(participant_id=s, role=role, session=role)
               for s in ("a", "b", "c") for role in ("enroll", "probe")]
    indices = episode(records, ["a", "b", "c"], np.random.default_rng(3), 3, 2)
    for group in np.array(indices).reshape(3, 4):
        assert records[group[0] // 24].participant_id == records[group[2] // 24].participant_id
        assert records[group[0] // 24].session != records[group[2] // 24].session
        assert len(set(group)) == 4
    vectors = torch.nn.functional.normalize(torch.randn(12, 8), dim=1).requires_grad_()
    loss = prototype_loss(vectors, 3, 2, .1)
    loss.backward()
    assert torch.isfinite(loss) and vectors.grad.abs().sum() > 0


def test_streams_are_balanced_chronological_and_reproducible():
    events = schedule(["a", "b", "c"], 30, 0, 123)
    assert events == schedule(["a", "b", "c"], 30, 0, 123)
    for claimed in ("a", "b", "c"):
        own = [e for e in events if e["claimed_id"] == claimed]
        assert [e["start_seconds"] for e in own] == list(range(0, 240, 30))
        assert sum(e["is_genuine"] for e in own) == 4
        assert all(bool(e["is_genuine"]) == (e["source_id"] == claimed) for e in own)


def test_test_data_is_not_read_without_global_freeze(tmp_path):
    with pytest.raises(RuntimeError, match="twelve"):
        records_for({}, {}, 0, "test", tmp_path)


def test_initial_controls_are_also_in_the_frozen_file_set(tmp_path, monkeypatch):
    import omega_dl.paths as paths
    monkeypatch.setattr(paths, "extension_root", lambda: tmp_path)
    names = locked_files(tmp_path)
    assert sum(p.name == "initial_thresholds.json" for p in names) == 12


@pytest.mark.external_artifacts
def test_existing_v1_protocol_is_still_loadable():
    run = extension_root() / "outputs/verification_v1"
    if not (run / "protocol.json").is_file():
        pytest.skip("Historical frozen V1 run is an external artifact; see docs/EXTERNAL_ARTIFACTS.md")
    assert load_protocol(run)["config"]["version"] == "verification_v1"
