"""Tests for the world-space line evidence accumulator (line_evidence)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch_import_guard = pytest.importorskip("torch")  # noqa: F841 (parity w/ lib)

from beamng_autopilot.vision.line_evidence import (  # noqa: E402
    LineEvidenceAccumulator,
)
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402

W, H = 320, 240
CAM = CameraModel(offset=np.array([0.0, 1.2, 1.4]),
                  fwd_local=np.array([0.0, 1.0, 0.0]),
                  up_local=np.array([0.0, 0.0, 1.0]),
                  fov_deg=65.0, width=W, height=H)

# A straight "painted line" on the ground, 10-14 m ahead, slightly left.
WORLD_LINE = np.column_stack([
    np.linspace(10.0, 14.0, 40), np.full(40, 0.6)])


def _project_mask(world_pts, pos, heading):
    pos3 = np.array([pos[0], pos[1], 0.0])
    u, v, valid = CAM.project(np.column_stack([
        world_pts[:, 0], world_pts[:, 1], np.zeros(len(world_pts))]),
        pos3, heading)
    m = np.zeros((H, W), dtype=bool)
    u = np.asarray(u)[valid].astype(int)
    v = np.asarray(v)[valid].astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m[v[ok], u[ok]] = True
    return m


def test_two_sightings_survive_a_dropout():
    acc = LineEvidenceAccumulator()
    t = [100.0, 100.5, 101.0]
    # two frames see the line while driving straight
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=t[0])
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=t[1])
    # third frame: total dropout (empty current mask)
    fused = acc.fuse(np.zeros((H, W), dtype=bool),
                     CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=t[2])
    assert fused.any(), "2-hit evidence must fill a single-frame dropout"


def test_single_frame_noise_never_propagates():
    acc = LineEvidenceAccumulator()
    noise = _project_mask(np.array([[12.0, -2.5]]), (0.0, 0.0, 0.0), 0.0)
    acc.update(noise, CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=100.0)
    fused = acc.fuse(np.zeros((H, W), dtype=bool),
                     CAM, (0.5, 0.0, 0.0), 0.0, 0.0, now=100.5)
    assert not fused.any(), "one-frame false positives must not persist"


def test_reset_clears_all_state():
    acc = LineEvidenceAccumulator()
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=100.0)
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=100.5)
    acc.reset()
    assert not acc._cells
    fused = acc.fuse(np.zeros((H, W), dtype=bool),
                     CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=200.0)
    assert not fused.any()


def test_stale_evidence_decays_away():
    acc = LineEvidenceAccumulator()
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=100.0)
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=100.5)
    # long gap: decay + MAX_AGE_S must retire the evidence
    fused = acc.fuse(np.zeros((H, W), dtype=bool),
                     CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=200.0)
    assert not fused.any()


def _same_cell_points(monkeypatch):
    from beamng_autopilot.vision import line_evidence
    points = np.array([[10.01, 1.01], [10.02, 1.02], [10.03, 1.03]])
    monkeypatch.setattr(line_evidence, "_back_project_many",
                        lambda *a: (points, np.ones(3, dtype=bool)))
    mask = np.ones((1, 3), dtype=bool)
    return mask


def test_many_pixels_in_one_frame_count_as_one_sighting(monkeypatch):
    mask = _same_cell_points(monkeypatch)
    acc = LineEvidenceAccumulator()
    acc.update(mask, None, (0., 0.), 0., now=100.)
    assert len(acc._cells) == 1
    assert next(iter(acc._cells.values()))[0] == 1.0
    assert acc.fused_support().shape == (0, 2)
    acc.update(mask, None, (0., 0.), 0., now=100.5)
    assert len(acc.fused_support()) == 1


def test_repeated_timestamp_cannot_confirm_a_single_frame(monkeypatch):
    mask = _same_cell_points(monkeypatch)
    acc = LineEvidenceAccumulator()
    for _ in range(3):
        acc.update(mask, None, (0., 0.), 0., now=100.)
    assert acc.fused_support().shape == (0, 2)


def test_clock_restart_drops_previous_episode(monkeypatch):
    mask = _same_cell_points(monkeypatch)
    acc = LineEvidenceAccumulator()
    acc.update(mask, None, (0., 0.), 0., now=100.)
    acc.update(mask, None, (0., 0.), 0., now=100.5)
    assert len(acc.fused_support()) == 1
    acc.update(mask, None, (0., 0.), 0., now=1.)
    assert acc.fused_support().shape == (0, 2)


def test_long_gap_cannot_confirm_old_single_sighting(monkeypatch):
    mask = _same_cell_points(monkeypatch)
    acc = LineEvidenceAccumulator()
    acc.update(mask, None, (0., 0.), 0., now=100.)
    acc.update(mask, None, (0., 0.), 0., now=110.)
    assert acc.fused_support().shape == (0, 2)


def test_invalid_timestamp_is_rejected(monkeypatch):
    mask = _same_cell_points(monkeypatch)
    acc = LineEvidenceAccumulator()
    with pytest.raises(ValueError, match="timestamp"):
        acc.update(mask, None, (0., 0.), 0., now=float("nan"))


def test_nonfinite_projected_points_do_not_poison_history(monkeypatch):
    from beamng_autopilot.vision import line_evidence
    points = np.array([[10.01, 1.01], [np.nan, 0.], [0., np.inf]])
    monkeypatch.setattr(line_evidence, "_back_project_many",
                        lambda *a: (points, np.ones(3, dtype=bool)))
    acc = LineEvidenceAccumulator()
    acc.update(np.ones((1, 3), bool), None, (0., 0.), 0., now=100.)
    assert len(acc._cells) == 1
    assert acc.fused_support().shape == (0, 2)


# ---------------------------------------------------------------------------
# Dual confidence + continuous-loss expiry (plan phase E3)
# ---------------------------------------------------------------------------

def test_dual_confidence_separates_fresh_from_held_evidence():
    acc = LineEvidenceAccumulator()
    t = [200.0, 200.5, 201.0]
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=t[0])
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=t[1])
    # frame 3 observes the line again: most supported cells are refreshed
    # by THIS frame (the mask is thin and the car moved, so a few cells
    # keep an older sighting - the share is honest, not forced to 1.0)
    mask, info = acc.fuse_with_confidence(
        _project_mask(WORLD_LINE, (2.0, 0.0, 0.0), 0.0),
        CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=t[2])
    assert 0.5 < info["current_confidence"] <= 1.0
    assert info["n_current"] > 0
    assert info["n_current"] + info["n_history"] == info["n_supported"]
    assert info["history_confidence"] > 0.0
    assert info["expired"] is False
    # frame 4 is a total dropout: the SAME support is now held history
    mask4, info4 = acc.fuse_with_confidence(
        np.zeros((H, W), dtype=bool),
        CAM, (3.0, 0.0, 0.0), 0.0, 0.0, now=t[2] + 0.5)
    assert mask4.any()
    assert info4["current_confidence"] == pytest.approx(0.0)
    assert info4["n_current"] == 0
    assert info4["n_history"] == info4["n_supported"]
    assert info4["since_observation_s"] == pytest.approx(0.5)
    # held evidence is trusted less than fresh evidence
    assert info4["history_confidence"] < info["history_confidence"]


def test_history_expires_after_continuous_loss():
    """连续丢线超过阈值后必须失效 (plan E3)."""
    from beamng_autopilot.vision.line_evidence import MAX_AGE_S
    acc = LineEvidenceAccumulator()
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=100.0)
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=100.5)
    _mask, info = acc.fuse_with_confidence(
        np.zeros((H, W), dtype=bool), CAM, (2.0, 0.0, 0.0), 0.0, 0.0,
        now=100.5 + MAX_AGE_S + 0.5)
    assert info["expired"] is True
    assert info["n_supported"] == 0
    assert info["history_confidence"] == 0.0
    assert info["since_observation_s"] > MAX_AGE_S


def test_confidence_is_json_safe_and_zero_without_evidence():
    import json
    acc = LineEvidenceAccumulator()
    info = acc.confidence()
    assert info["current_confidence"] == 0.0
    assert info["history_confidence"] == 0.0
    assert info["n_supported"] == 0
    json.dumps(info)


def test_fuse_still_returns_the_mask_unchanged():
    """The provenance-aware path must not alter fuse()'s contract."""
    acc = LineEvidenceAccumulator()
    t = [300.0, 300.5]
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=t[0])
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=t[1])
    acc_a = LineEvidenceAccumulator()
    acc_b = LineEvidenceAccumulator()
    for a in (acc_a, acc_b):
        a.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
                 CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=t[0])
        a.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
                 CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=t[1])
    empty = np.zeros((H, W), dtype=bool)
    plain = acc_a.fuse(empty, CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=301.0)
    aware, _info = acc_b.fuse_with_confidence(
        empty, CAM, (2.0, 0.0, 0.0), 0.0, 0.0, now=301.0)
    assert np.array_equal(plain, aware)


def test_support_mask_projects_history_without_voting():
    """The far-zone rule needs history read WITHOUT adding to it."""
    acc = LineEvidenceAccumulator()
    acc.update(_project_mask(WORLD_LINE, (0.0, 0.0, 0.0), 0.0),
               CAM, (0.0, 0.0, 0.0), 0.0, 0.0, now=100.0)
    acc.update(_project_mask(WORLD_LINE, (1.0, 0.0, 0.0), 0.0),
               CAM, (1.0, 0.0, 0.0), 0.0, 0.0, now=100.5)
    cells_before = dict(acc._cells)
    support = acc.support_mask((H, W), CAM, (2.0, 0.0, 0.0), 0.0, 0.0)
    assert support.any(), "accumulated evidence must project back"
    assert acc._cells == cells_before, "support_mask must not vote"
    assert acc._last_observation_t == 100.5, "nor count as an observation"
    # an empty accumulator projects nothing at all
    empty = LineEvidenceAccumulator()
    assert not empty.support_mask((H, W), CAM, (0.0, 0.0, 0.0), 0.0).any()
    # and a missing camera model degrades to an empty mask, not a crash
    assert not acc.support_mask((H, W), None, (0.0, 0.0, 0.0), 0.0).any()
