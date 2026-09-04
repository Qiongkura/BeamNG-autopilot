"""Offline tests for the live E2E runtime (drive-loop wiring pieces)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from beamng_autopilot.neural.e2e_runtime import (
    E2ERuntime, ego_path_to_world,
)
from beamng_autopilot.neural.e2e_torch import E2ENetTorch
from beamng_autopilot.bev_fusion import BEVFeatureMap


def test_ego_path_to_world_inverts_dataset_transform() -> None:
    """The inverse of the dataset's world->ego transform round-trips."""
    pos = np.array([100.0, 200.0])
    heading = 0.7
    world = np.array([[100.0, 200.0],
                      [101.0, 201.0],
                      [101.5, 202.0]])
    rel = world - pos
    c, s = np.cos(heading), np.sin(heading)
    ego = np.stack([rel[:, 0] * c + rel[:, 1] * s,
                    -rel[:, 0] * s + rel[:, 1] * c], axis=1)
    back = ego_path_to_world(ego, pos, heading)
    assert back is not None
    assert np.allclose(back, world, atol=1e-5)


def test_ego_path_to_world_matches_dataset_forward_at_zero() -> None:
    pos = np.array([5.0, -3.0])
    heading = 0.0
    out = ego_path_to_world(np.array([[0.0, 0.0], [2.0, 0.5]]),
                            pos, heading)
    assert out is not None
    assert np.allclose(out, [[5.0, -3.0], [7.0, -2.5]])


def test_ego_path_to_world_rejects_bad_input() -> None:
    assert ego_path_to_world(None, [0.0, 0.0], 0.0) is None
    assert ego_path_to_world(np.zeros((1,)), [0.0, 0.0], 0.0) is None


def _tiny_ckpt(tmp_path, history: int = 1):
    # Legacy checkpoint format: single-channel occupancy raster, and the
    # bev_channels key absent (pre-v3 recordings) - the runtime must fall
    # back to bev_channels=1 and still load the state dict.
    net = E2ENetTorch(grid_n=16, n_waypoints=8, history=history,
                      bev_channels=1).eval()
    p = tmp_path / "tiny.pt"
    torch.save({
        "model": net.state_dict(),
        "grid_n": 16,
        "n_waypoints": 8,
        "history": history,
        "img_h": 32,
        "img_w": 48,
        "epoch": 1,
    }, p)
    return p


def test_e2e_runtime_step_pipeline(tmp_path) -> None:
    weights = _tiny_ckpt(tmp_path, history=1)
    rt = E2ERuntime(weights, device="cpu")
    assert rt.loaded
    assert rt.history == 1
    assert rt.n_waypoints == 8

    def make_out(h, w, bev_n=16):
        frame = (np.random.default_rng(0).integers(
            0, 256, size=(h, w, 3)).astype(np.uint8))
        masks = {
            "road": np.ones((h, w), dtype=bool),
            "line": np.zeros((h, w), dtype=bool),
        }
        masks["line"][h // 2, :] = True
        bev = np.zeros((bev_n, bev_n), dtype=np.float32)
        bev[:, bev_n // 2] = 1.0
        return SimpleNamespace(
            frame=frame, bev=bev,
            head_outputs={"semantic": SimpleNamespace(masks=masks)})

    pos = np.array([10.0, 20.0])
    heading = 0.4
    path, action, ms = rt.step(make_out(60, 80), pos, heading, 3.0)
    assert path is not None and path.shape == (8, 2)
    assert np.isfinite(path).all()
    assert action is not None and action.shape == (2,)
    assert ms >= 0.0

    # temporal buffer holds history+1 frames and pads at the start; a
    # second frame changes only the newest buffer slot.
    path2, _, _ = rt.step(make_out(60, 80), pos, heading, 3.2)
    assert path2 is not None and path2.shape == (8, 2)
    assert len(rt._buf) == 2

    rt.reset()
    assert len(rt._buf) == 0
    path3, _, _ = rt.step(make_out(60, 80), pos, heading, 3.0)
    assert path3 is not None and path3.shape == (8, 2)


def test_e2e_runtime_disabled_without_checkpoint() -> None:
    rt = E2ERuntime(None, device="cpu")
    assert not rt.loaded
    assert rt.step(None, [0.0, 0.0], 0.0, 0.0) == (None, None, 0.0)


def _tiny_ckpt_multichannel(tmp_path, history: int = 1):
    net = E2ENetTorch(grid_n=16, n_waypoints=8, history=history,
                      bev_channels=4).eval()
    p = tmp_path / "tiny_mc.pt"
    torch.save({
        "model": net.state_dict(),
        "grid_n": 16,
        "n_waypoints": 8,
        "history": history,
        "bev_channels": 4,
        "img_h": 32,
        "img_w": 48,
        "epoch": 1,
    }, p)
    return p


def test_e2e_runtime_multichannel_vector_input(tmp_path) -> None:
    """A 4-channel checkpoint consumes the fused feature map live."""
    weights = _tiny_ckpt_multichannel(tmp_path)
    rt = E2ERuntime(weights, device="cpu")
    assert rt.loaded and rt.bev_channels == 4

    h, w, n = 60, 80, 16
    frame = np.random.default_rng(0).integers(
        0, 256, size=(h, w, 3)).astype(np.uint8)
    masks = {"road": np.ones((h, w), dtype=bool),
             "line": np.zeros((h, w), dtype=bool)}
    bev = np.zeros((n, n), dtype=np.float32)
    bev[:, n // 2] = 1.0
    fm = BEVFeatureMap(n=n, res=0.5)
    fm.logodds["lane"][4:12, n // 2] = 3.0   # strong lane-line evidence
    out = SimpleNamespace(
        frame=frame, bev=bev, drivable=np.ones((n, n), dtype=np.uint8),
        feature_map=fm,
        head_outputs={"semantic": SimpleNamespace(masks=masks)})

    vec = rt._vector_input(out, bev)
    assert vec.shape == (4, n, n)
    assert float(vec[2, 4, n // 2]) > 0.8    # lane channel from fmap
    path, action, ms = rt.step(out, np.array([0.0, 0.0]), 0.0, 3.0)
    assert path is not None and path.shape == (8, 2)
    assert np.isfinite(path).all() and action is not None
    assert ms >= 0.0

    # Degraded tick without fusion still runs via the fallback synthesis.
    out2 = SimpleNamespace(
        frame=frame, bev=bev, drivable=np.ones((n, n), dtype=np.uint8),
        feature_map=None,
        head_outputs={"semantic": SimpleNamespace(masks=masks)})
    path2, _, _ = rt.step(out2, np.array([0.0, 0.0]), 0.0, 3.0)
    assert path2 is not None and path2.shape == (8, 2)
    assert np.isfinite(path2).all()


def test_e2e_runtime_label_from_semantic_head() -> None:
    h, w = 12, 16
    masks = {
        "road": np.zeros((h, w), dtype=bool),
        "line": np.zeros((h, w), dtype=bool),
    }
    masks["road"][2:10, 3:14] = True
    masks["line"][5:7, :] = True
    out = SimpleNamespace(head_outputs={
        "semantic": SimpleNamespace(masks=masks)})
    lab = E2ERuntime._label_from_outputs(out.head_outputs, (h, w))
    assert lab is not None
    assert set(np.unique(lab)) <= {0, 1, 2}
    assert np.all(lab[5:7, :] == 2)
    assert np.all(lab[2:10, 3:14] >= 1)
    assert np.all(lab[0, :] == 0)
