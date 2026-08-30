"""Offline tests for the real (torch) E2E network and shadow dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from beamng_autopilot.neural import (
    E2ENetTorch,
    ShadowMultimodalDataset,
)
from beamng_autopilot.neural.dataset import GRID_N, N_WAYPOINTS, _has_wedge_restart


def _tiny_episode(tmp_path, n=4, with_rgb=True):
    """Build a minimal version-2 shadow .npz episode."""
    p = tmp_path / "ep.npz"
    t = np.arange(n, dtype=np.float64)
    x = np.linspace(0, 10, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    hdg = np.zeros(n, dtype=np.float64)
    bev = np.zeros((n, GRID_N, GRID_N), dtype=np.float32)
    bev[:, GRID_N // 2, :] = 1.0  # a free corridor
    traj = np.zeros((n, 25, 2), dtype=np.float64)
    for i in range(n):
        traj[i, :, 0] = np.linspace(0, 12, 25)
        traj[i, :, 1] = 0.2 * np.sin(np.linspace(0, 3, 25))
    rgb = np.zeros((n, 32, 48, 3), dtype=np.uint8)
    rgb[:, :, :, :] = np.arange(n, dtype=np.uint8)[:, None, None, None]
    label = np.zeros((n, 32, 48), dtype=np.uint8)
    label[:, 10:22, :] = 1
    np.savez_compressed(
        p,
        version=np.int64(2), t=t, x=x, y=y, heading=hdg,
        speed=np.full(n, 3.0), throttle=np.full(n, 0.35),
        brake=np.zeros(n),
        steer=np.zeros(n), bev=bev, drivable=np.ones((n, GRID_N, GRID_N),
                                                     dtype=np.uint8),
        trajectory=traj, trajectory_ok=np.ones(n, dtype=bool),
        target_speed=np.full(n, 5.0),
        lane_src=np.array(["semantic"] * n, dtype=object),
        cost=np.zeros(n), kind=np.array(["arc"] * n, dtype=object),
        rgb=rgb, label=label, quality=np.ones(n, dtype=np.float32),
        meta=b"{}")
    return p


def test_torch_forward_shapes() -> None:
    net = E2ENetTorch()
    rgb = torch.zeros(2, 3, 240, 320)
    label = torch.zeros(2, 1, 240, 320)
    bev = torch.zeros(2, 1, GRID_N, GRID_N)
    traj, action = net(rgb, label, bev)
    assert traj.shape == (2, N_WAYPOINTS, 2)
    assert action.shape == (2, 2)
    assert torch.isfinite(traj).all() and torch.isfinite(action).all()


def test_torch_forward_optional_modalities() -> None:
    net = E2ENetTorch()
    rgb = torch.zeros(1, 3, 120, 160)
    traj, action = net(rgb)  # label/bev zero-filled
    assert traj.shape == (1, N_WAYPOINTS, 2)
    assert action.shape == (1, 2)


def test_torch_trains_steps(tmp_path) -> None:
    net = E2ENetTorch().eval()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rgb = torch.rand(4, 3, 64, 64)
    label = (torch.rand(4, 1, 64, 64) > 0.5).float()
    bev = torch.rand(4, 1, GRID_N, GRID_N)
    traj_t = torch.randn(4, N_WAYPOINTS, 2) * 2.0
    act_t = torch.randn(4, 2) * 0.5
    def loss():
        traj_p, act_p = net(rgb, label, bev)
        return ((traj_p - traj_t) ** 2).mean() + \
            ((act_p - act_t) ** 2).mean()
    l0 = float(loss().detach())
    for _ in range(5):
        opt.zero_grad()
        loss().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    l1 = float(loss().detach())
    assert l1 < l0, (l0, l1)


def test_torch_checkpoint_roundtrip(tmp_path) -> None:
    net = E2ENetTorch()
    ckpt = {"model": net.state_dict(), "grid_n": GRID_N,
            "n_waypoints": N_WAYPOINTS, "img_h": 120, "img_w": 160,
            "epoch": 3, "val_loss": 0.12}
    out = tmp_path / "best.pt"
    torch.save(ckpt, out)
    net2 = E2ENetTorch(grid_n=GRID_N, n_waypoints=N_WAYPOINTS).eval()
    net2.load_state_dict(torch.load(out, map_location="cpu")["model"])
    rgb = torch.zeros(1, 3, 64, 64)
    net.eval()
    a, b = net(rgb)
    c, d = net2(rgb)
    assert torch.allclose(a, c) and torch.allclose(b, d)


def test_dataset_filters_and_shapes(tmp_path) -> None:
    p = _tiny_episode(tmp_path, n=4)
    ds = ShadowMultimodalDataset([p], min_quality=0.0, min_speed=0.0,
                                 img_h=32, img_w=48)
    assert len(ds) == 4
    obs, traj, mask, act = ds[0]
    rgb, label, bev = obs
    assert rgb.shape == (3, 32, 48)
    assert label.shape == (1, 32, 48)
    assert bev.shape == (1, GRID_N, GRID_N)
    assert traj.shape == (N_WAYPOINTS, 2)
    assert mask.shape == (N_WAYPOINTS,)
    assert act.shape == (2,)
    ds2 = ShadowMultimodalDataset([p], min_quality=0.0, min_speed=9.0,
                                  img_h=32, img_w=48)
    assert len(ds2) == 0  # every frame filtered by the speed gate


def test_dataset_keeps_action_only_frames(tmp_path) -> None:
    """A frame without a feasible trajectory is kept (action-only)."""
    p = _tiny_episode(tmp_path, n=4)
    # rewrite one frame's trajectory to all-NaN (trajectory_ok=False)
    with np.load(p, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    data["trajectory"][2] = np.nan
    data["trajectory_ok"][2] = False
    p2 = tmp_path / "ep_bad.npz"
    np.savez_compressed(p2, **data)
    ds = ShadowMultimodalDataset([p2], min_quality=0.0, min_speed=0.0,
                                 img_h=32, img_w=48)
    assert len(ds) == 4
    obs, traj, mask, act = ds[2]
    assert float(mask.sum()) == 0.0   # no valid waypoints
    assert torch.isfinite(traj).all()
    assert act.shape == (2,)
    obs, traj, mask0, act = ds[0]
    rgb, label, bev = obs
    assert rgb.shape == (3, 32, 48)
    assert label.shape == (1, 32, 48)
    assert bev.shape == (1, GRID_N, GRID_N)
    assert traj.shape == (N_WAYPOINTS, 2)
    assert float(mask0.sum()) == N_WAYPOINTS  # healthy frame fully valid
    assert act.shape == (2,)


def test_wedge_restart_detection() -> None:
    # mid-run stop followed by more driving = wedge restart -> bad
    spd_wedge = np.array([0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0,
                          5.0, 6.0, 6.0], dtype=float)
    assert _has_wedge_restart(spd_wedge)
    # end-zone stop with no driving after it = good (end-stop)
    spd_end = np.array([5.0, 6.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       dtype=float)
    assert not _has_wedge_restart(spd_end)
    # pure cruise, no stops
    spd_cruise = np.full(20, 6.0)
    assert not _has_wedge_restart(spd_cruise)


def test_dataset_drops_wedge_episodes(tmp_path) -> None:
    p_good = _tiny_episode(tmp_path, n=4)
    # a second episode with a mid-run stop followed by more driving
    n = 10
    p2 = tmp_path / "ep_wedge.npz"
    np.savez_compressed(
        p2,
        version=np.int64(2), t=np.arange(n, dtype=np.float64),
        x=np.zeros(n), y=np.zeros(n), heading=np.zeros(n),
        speed=np.array([6, 6, 0, 0, 0, 0, 6, 6, 6, 6], dtype=np.float64),
        throttle=np.full(n, 0.35), brake=np.zeros(n), steer=np.zeros(n),
        bev=np.zeros((n, GRID_N, GRID_N), dtype=np.float32),
        drivable=np.ones((n, GRID_N, GRID_N), dtype=np.uint8),
        trajectory=np.zeros((n, 25, 2)), trajectory_ok=np.ones(n, bool),
        target_speed=np.full(n, 5.0),
        lane_src=np.array(["semantic"] * n, dtype=object),
        cost=np.zeros(n), kind=np.array(["arc"] * n, dtype=object),
        rgb=np.zeros((n, 32, 48, 3), dtype=np.uint8),
        label=np.zeros((n, 32, 48), dtype=np.uint8),
        quality=np.ones(n, dtype=np.float32), meta=b"{}")
    ds = ShadowMultimodalDataset([p_good, p2], min_quality=0.0,
                                 min_speed=0.0, drop_wedge_episodes=True,
                                 img_h=32, img_w=48)
    assert len(ds) == 4  # only the good episode survives


def test_dataset_collate(tmp_path) -> None:
    p = _tiny_episode(tmp_path, n=4)
    ds = ShadowMultimodalDataset([p], min_quality=0.0,
                                 img_h=32, img_w=48)
    obs, trajs, masks, acts = ShadowMultimodalDataset.collate(
        [ds[i] for i in range(2)])
    rgb, label, bev = obs
    assert rgb.shape == (2, 3, 32, 48)
    assert label.shape == (2, 1, 32, 48)
    assert bev.shape == (2, 1, GRID_N, GRID_N)
    assert trajs.shape == (2, N_WAYPOINTS, 2)
    assert masks.shape == (2, N_WAYPOINTS)
    assert acts.shape == (2, 2)
