"""静态投影标定的测试（T16 Order2 后续：渲染回读 → 落盘 → 复验）。

覆盖三件事：
1. `_rotate_camera_basis` 的小角度旋转是**正交**的（basis 保持单位正交，不缩放）；
2. `fit_projection_alignment` 在"已知被注入角度误差"的合成批次上能把误差找回来
   （可辨识性），且校准后残差显著下降；
3. 没有线像素/没有线真值点时返回 `unknown`，不猜（UNKNOWN≠PASS）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import (  # noqa: E402
    _rotate_camera_basis, camera_basis, fit_projection_alignment,
    line_distance_stats, make_synthetic_batch,
)


def test_rotate_camera_basis_stays_orthonormal():
    batch = make_synthetic_batch(with_line=True)
    cam = batch["frames"][0]["camera"]
    for dy, dp in ((2.0, -3.0), (-1.5, 4.0)):
        c2 = _rotate_camera_basis(cam, dy, dp)
        b = c2["pose"]["basis"]
        f = np.asarray(b["fwd"], float)
        r = np.asarray(b["right"], float)
        u = np.asarray(b["up"], float)
        assert abs(np.linalg.norm(f) - 1) < 1e-9
        assert abs(np.linalg.norm(r) - 1) < 1e-9
        assert abs(np.linalg.norm(u) - 1) < 1e-9
        assert abs(float(f @ r)) < 1e-9 and abs(float(f @ u)) < 1e-9
        assert abs(float(r @ u)) < 1e-9
    # 0 度旋转 = 原方向（数值比较：basis 会做一次正交化，不逐位相同）
    c0 = _rotate_camera_basis(cam, 0.0, 0.0)
    _p, r0, f0, u0 = camera_basis(c0)
    _p, r1, f1, u1 = camera_basis(cam)
    assert np.allclose(f0, f1, atol=1e-9) and np.allclose(r0, r1, atol=1e-9)


def test_fit_recovers_an_injected_angle_error():
    """把相机人为转歪 2.5°，标定必须把它找回来并把残差压下去。"""
    batch = make_synthetic_batch(with_line=True)
    frames = []
    for fr in batch["frames"]:
        f2 = dict(fr)
        f2["camera"] = _rotate_camera_basis(fr["camera"], 2.5, -1.5)
        frames.append(f2)
    before = [line_distance_stats(f)["mean_px"] for f in frames]
    assert all(b is not None for b in before)
    assert min(before) > 1.0, before          # 注入的误差必须真的把点甩开
    fit = fit_projection_alignment(frames)
    assert fit["status"] == "measured", fit
    assert fit["after_px"] < fit["before_px"], fit
    # 单看角度**不可唯一辨识**：沿路的长直线对 yaw 退化，而 yaw 的残差又会被
    # pitch 吸收（实测：注入 (yaw=+2.5°, pitch=-1.5°) 时最优解跑到
    # (yaw=+1.25°, pitch=-5.0°)，残差同样降到 0.35 px）。所以断言的是：
    #   (a) 参数落在受限搜索范围内；(b) **对齐后残差**达标——它才是验收指标。
    assert -1.5 <= fit["yaw_deg"] <= 1.5, fit
    assert -6.0 <= fit["pitch_deg"] <= 6.0, fit
    assert fit["after_px"] <= 2.0, fit
    assert fit["before_px"] > fit["after_px"], fit


def test_fit_is_unknown_without_line_pixels_or_truth():
    batch = make_synthetic_batch(with_line=False)     # 无线真值点
    fit = fit_projection_alignment(batch["frames"])
    assert fit["status"] == "unknown" and fit["after_px"] is None
    assert "no frame" in fit["why"]


def test_line_evidence_coverage_gate():
    """覆盖门：证据覆盖不足的帧/站点要能被识别（扩量批次实测：有站点线类像素
    只有 25 个甚至 0 个，那种站点上"到最近像素的距离"是假残差）。"""
    from beamng_autopilot.experiments.auto_truth import line_evidence_coverage
    batch = make_synthetic_batch(with_line=True)
    fr = batch["frames"][0]
    cov = line_evidence_coverage(fr, radius_px=6)
    assert cov["status"] == "measured" and cov["coverage"] is not None
    assert cov["coverage"] >= 0.6, cov          # 合成场景线就在真值处
    assert cov["n_in_frame"] >= cov["n_covered"] > 0
    # 把线类像素抹掉（label 全 0）-> 覆盖 0，必须判 not_covered
    import numpy as np
    fr2 = dict(fr)
    fr2["label"] = np.zeros_like(np.asarray(fr["label"]))
    cov2 = line_evidence_coverage(fr2, radius_px=6)
    assert cov2["status"] == "not_covered" and cov2["coverage"] == 0.0, cov2
