# -*- coding: utf-8 -*-
"""BEAMNG_PAIR_WIDTH_NEAR：世界系配对宽度取近场（默认关）。

远场回投误差随距离放大：试车场 40m 实线在 s>25m 读到 5.1–6.7m
（GT 3.8m），全线中位数把宽度带偏到 4.95m。本测试用"近场平行、远场
外漂"的合成线对，钉住开/关两种口径的差异，并确认默认行为不变。
"""
import numpy as np

from beamng_autopilot.lane import pairing as P


def _cand(xs, ys):
    world = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
    pos = np.array([0.0, 0.0])
    fwd = np.array([0.0, 1.0])
    left = np.array([-1.0, 0.0])
    proj = P._boundary_projection(world, pos, fwd, left)
    return P._LineCandidate(
        proj=proj, span=float(proj[-1, 0] - proj[0, 0]), conf=0.8,
        kind="solid", color="white",
        med_lat=float(np.median(proj[:, 1])), score=0.5, world=world)


def _pair_width(monkeypatch, near: bool):
    ys = np.arange(4.0, 45.0, 1.0)
    left = _cand(np.full_like(ys, 1.8), ys)
    # 右线近场 -1.8；12m 之后线性外漂到 -5.5（模拟远场回投误差）
    drift = np.clip((ys - 12.0) / 33.0, 0.0, 1.0)
    right = _cand(-1.8 - 3.7 * drift, ys)
    if near:
        monkeypatch.setenv("BEAMNG_PAIR_WIDTH_NEAR", "1")
    else:
        monkeypatch.delenv("BEAMNG_PAIR_WIDTH_NEAR", raising=False)
    dbg = {}
    best = P._best_vision_pair([left, right], [], np.array([0.0, 0.0]),
                               np.array([0.0, 1.0]), 1.0, 64, debug=dbg)
    assert best is not None
    return float(best[4]), dbg


def test_near_width_ignores_far_drift(monkeypatch):
    w_near, dbg = _pair_width(monkeypatch, near=True)
    w_all, _ = _pair_width(monkeypatch, near=False)
    assert abs(w_near - 3.6) < 0.25, w_near
    assert w_all > w_near + 0.5, (w_all, w_near)
    assert abs(float(dbg["pair_width_near_m"]) - w_near) < 1e-6


def test_near_width_coarse_station_step_falls_back_to_nearest(monkeypatch):
    """粗站点步长下 10m 内不足 3 站：退化为"最近 3 个有效站"，仍不取全线。"""
    ys = np.arange(4.0, 45.0, 1.0)
    left = _cand(np.full_like(ys, 1.8), ys)
    drift = np.clip((ys - 12.0) / 33.0, 0.0, 1.0)
    right = _cand(-1.8 - 3.7 * drift, ys)
    monkeypatch.setenv("BEAMNG_PAIR_WIDTH_NEAR", "1")
    dbg = {}
    best = P._best_vision_pair([left, right], [], np.array([0.0, 0.0]),
                               np.array([0.0, 1.0]), 6.0, 64, debug=dbg)
    assert best is not None
    assert int(dbg["pair_width_near_n"]) >= 3
    # 粗步长下最近 3 站可能落到 10-16m（仍有轻微外漂），关键是不能回到
    # 全线中位数（该线对的全线中位 >4.3m）。
    assert 3.4 < float(best[4]) < 4.2


def test_near_width_default_off(monkeypatch):
    monkeypatch.delenv("BEAMNG_PAIR_WIDTH_NEAR", raising=False)
    _w, dbg = _pair_width(monkeypatch, near=False)
    assert "pair_width_near_m" not in dbg
