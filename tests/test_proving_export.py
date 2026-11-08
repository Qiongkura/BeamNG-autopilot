# -*- coding: utf-8 -*-
"""试车场导出：built 绑定几何 / 漆色组件标签 / A1 门禁 / 溯源。

几何用真实 CameraModel + 合成渲染图（不依赖引擎），与
tests/test_proving_diag_a1.py 同一套确定性夹具。
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

ORANGE = (255, 128, 0)
W, H = 1024, 800


def _mod():
    spec = importlib.util.spec_from_file_location(
        "mpc_export", str(ROOT / "scripts" / "m5_proving_export.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _camera():
    from beamng_autopilot.vision.projection import CameraModel
    return CameraModel(
        offset=np.array([0.0, 1.216, 1.386]),
        fwd_local=np.array([0.0, 0.99994, -0.0112]),
        up_local=np.array([0.0, 0.0112, 0.99994]),
        fov_deg=120, width=W, height=H)


def _nodes(lat_m: float) -> np.ndarray:
    xs = np.arange(5.0, 26.0, 1.0)
    return np.column_stack([xs, np.full(len(xs), lat_m), np.zeros(len(xs))])


def _truth(nodes: np.ndarray, role: str) -> list[dict]:
    return [{"world": [float(x), float(y), float(z)], "class": 2, "role": role}
            for x, y, z in nodes]


def _built_doc(right_lat: float = -2.7) -> dict:
    """与 right_lane_lats(3.5, 0.8) 一致的双线 built 记录（右线可改标定）。"""
    left_lat = right_lat + 3.6
    ln, rn = _nodes(left_lat), _nodes(right_lat)
    return {"scene": "S1", "lines": [
        {"id": "m5_line_S1_left", "role": "left", "width_m": 0.15,
         "lateral_m": round(left_lat, 3), "nodes": ln.tolist(),
         "truth_points": _truth(ln, "left")},
        {"id": "m5_line_S1_right", "role": "right", "width_m": 0.15,
         "lateral_m": round(right_lat, 3), "nodes": rn.tolist(),
         "truth_points": _truth(rn, "right")},
    ]}


def _geom(right_lat: float = -2.7):
    m = _mod()
    return m.geometry_from_built(m.validate_built_lines(_built_doc(right_lat)))


def _paint_strip(img: np.ndarray, cam, lat_m: float, thickness: int = 5):
    pts = _nodes(lat_m)
    u, v, ok = cam.project(pts, np.array([0.0, 0.0, 0.0]), 0.0)
    for i in range(len(pts) - 1):
        if ok[i] and ok[i + 1] and np.isfinite(u[i]) and np.isfinite(u[i + 1]):
            cv2.line(img, (int(u[i]), int(v[i])),
                     (int(u[i + 1]), int(v[i + 1])), ORANGE, thickness)
    return img


# ── built 校验（不变契约）─────────────────────────────────────────────────
def test_built_line_validation_never_claims_rendered_truth():
    m = _mod()
    source = {"lines": [
        {"role": "right", "nodes": [[1, 0, 2], [1, 3, 2]], "width_m": .15},
        {"role": "left", "nodes": [[0, 0, 2], [0, 3, 2]], "width_m": .2},
    ]}
    lines = m.validate_built_lines(source)
    assert [x["role"] for x in lines] == ["left", "right"]
    assert lines[0]["width_m"] == .2
    lines[0]["nodes"][0, 0] = 99
    assert source["lines"][1]["nodes"][0][0] == 0


@pytest.mark.parametrize("record", [None, {}, {"lines": []},
    {"lines": [{"role": "left", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1}]},
    {"lines": [{"role": "left", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1},
               {"role": "left", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1}]},
    {"lines": [{"role": "left", "nodes": [[0, 0, 0], [0, 0, 0]], "width_m": 1},
               {"role": "right", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1}]},
    {"lines": [{"role": "left", "nodes": [[0, 0, 0], [np.nan, 0, 0]], "width_m": 1},
               {"role": "right", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1}]},
    {"lines": [{"role": "left", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 0},
               {"role": "right", "nodes": [[0, 0, 0], [1, 0, 0]], "width_m": 1}]},
])
def test_invalid_built_line_records_rejected(record):
    with pytest.raises(ValueError):
        _mod().validate_built_lines(record)


# ── built 绑定几何（A1：不重算）───────────────────────────────────────────
def test_geometry_from_built_keeps_declared_lateral():
    """built 里声明右线 -2.0（变体②）→ 几何必须用 -2.0，不得重算回 -2.7。"""
    m = _mod()
    geom = m.geometry_from_built(m.validate_built_lines(_built_doc(-2.0)))
    right = next(g for g in geom if g["role"] == "right")
    assert abs(right["center"][0, 1] - (-2.0)) < 1e-6
    left = next(g for g in geom if g["role"] == "left")
    assert abs(left["center"][0, 1] - 1.6) < 1e-6


def test_geometry_rails_parallel_and_truth_bound():
    m = _mod()
    geom = _geom()
    for g in geom:
        sep = np.linalg.norm(g["left"][0, :2] - g["right"][0, :2])
        assert abs(sep - m.LINE_WIDTH_M) < 1e-6
        assert g["truth_points"], "truth points must come from built record"
    assert geom[0]["center"][0, 1] > geom[1]["center"][0, 1]


# ── 漆色组件标签 + A1 门禁 ────────────────────────────────────────────────
def test_label_uses_rendered_paint_and_records_right_offset():
    """左线对齐、右线按标定 -0.85m 渲染 → 双侧组件入标签、右偏移被记录。"""
    m = _mod()
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9)              # 左线：渲染=节点
    img = _paint_strip(img, cam, -2.7 - 0.85)      # 右线：渲染 -0.85m 偏移
    g = m.label_and_gate_frame(img, cam, np.array([0.0, 0.0, 0.0]), 0.0,
                               _geom(-2.7))
    assert g["ok"], g["reasons"]
    assert (g["line"] == 2).sum() > 0
    assert abs(g["metrics"]["left_offset_m"]) <= 0.2
    assert abs(g["metrics"]["right_offset_m"] - (-0.85)) < 0.2
    assert g["metrics"]["corridor_ratio"] >= m.CORRIDOR_MIN_RATIO


def test_gate_rejects_when_component_too_thick():
    """30px 粗橙块（>25px 门）不能入线标签，帧被拒。"""
    m = _mod()
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9, thickness=31)
    img = _paint_strip(img, cam, -2.7 - 0.85, thickness=31)
    g = m.label_and_gate_frame(img, cam, np.array([0.0, 0.0, 0.0]), 0.0,
                               _geom(-2.7))
    assert not g["ok"]
    assert any("component_span_or_missing" in r for r in g["reasons"])
    assert not (g["line"] == 2).any()


def test_gate_rejects_when_right_offset_beyond_calibration():
    """右线渲染偏移 1.6m（>1.0m 门）→ 帧被拒且右线组件不入标签。"""
    m = _mod()
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9)
    img = _paint_strip(img, cam, -2.7 - 1.6)
    g = m.label_and_gate_frame(img, cam, np.array([0.0, 0.0, 0.0]), 0.0,
                               _geom(-2.7))
    assert not g["ok"]
    assert any(r.startswith("right_offset=") for r in g["reasons"])
    assert g["metrics"]["corridor_ratio"] < m.CORRIDOR_MIN_RATIO


def test_road_label_covers_lane_between_built_lines():
    m = _mod()
    cam = _camera()
    img = np.full((H, W, 3), 200, np.uint8)
    img = _paint_strip(img, cam, 0.9)
    img = _paint_strip(img, cam, -2.7 - 0.85)
    g = m.label_and_gate_frame(img, cam, np.array([0.0, 0.0, 0.0]), 0.0,
                               _geom(-2.7))
    assert g["road"].sum() >= (g["line"] == 2).sum()
    # 车道中心（两线中点 lat -0.9）应落在 road 内
    pts = _nodes(-0.9)
    u, v, ok = cam.project(pts, np.array([0.0, 0.0, 0.0]), 0.0)
    mid_ok = [i for i in range(len(pts)) if ok[i] and np.isfinite(u[i])]
    assert mid_ok
    i = mid_ok[len(mid_ok) // 2]
    assert g["road"][int(v[i]), int(u[i])] == 1


def test_paint_mask_accepts_orange_and_rejects_grey():
    m = _mod()
    img = np.zeros((20, 20, 3), np.uint8)
    img[5:10, 5:10] = [255, 128, 0]          # 橙
    img[12:16, 12:16] = [128, 128, 128]      # 灰
    mask = m.paint_mask(img)
    assert mask[7, 7] and not mask[13, 13]


# ── 位姿去重（不变契约）───────────────────────────────────────────────────
def test_pose_duplicate_checks_all_accepted_and_wraps_angle():
    m = _mod()
    accepted = [([0.0, 0.0], np.pi - 0.005), ([3.0, 0.0], 0.0)]
    match = m.find_pose_duplicate([0.1, 0.0], -np.pi + 0.005, accepted)
    assert match is not None and match[0] == 0  # A -> B -> A, ±π
    assert match[1] == pytest.approx(0.1)
    assert match[2] == pytest.approx(0.01)
    assert m.find_pose_duplicate([1.0, 0.0], np.pi, accepted) is None
    assert m.find_pose_duplicate([0.0, 0.0], np.pi - 0.1, accepted) is None


@pytest.mark.parametrize("pos, heading", [([np.nan, 0], 0), ([0, np.inf], 0),
                                           ([0, 0], np.nan), ([1], 0)])
def test_pose_duplicate_rejects_invalid_input(pos, heading):
    with pytest.raises(ValueError):
        _mod().find_pose_duplicate(pos, heading, [])


def test_select_pose_candidates_audit_revisit_and_invalid():
    m = _mod()
    records = [
        {"source_id": "a", "pos": [0, 0], "heading": np.pi - .005},
        {"source_id": "b", "pos": [2, 0], "heading": 0},
        {"source_id": "revisit", "pos": [.1, 0], "heading": -np.pi + .005},
        {"source_id": "invalid", "pos": [np.nan, 0], "heading": 0},
        {"source_id": "rotation", "pos": [0, 0], "heading": 0},
    ]
    accepted, rejected = m.select_pose_candidates(records)
    assert [x["source_id"] for x in accepted] == ["a", "b", "rotation"]
    assert rejected[0]["matched_source_id"] == "a"
    assert rejected[0]["distance_m"] == pytest.approx(.1)
    assert rejected[0]["angle_rad"] == pytest.approx(.01)
    assert rejected[1] == {"source_id": "invalid", "reason": "invalid_pose"}
    with pytest.raises(ValueError, match="source_id"):
        m.select_pose_candidates([records[0], records[0]])


def test_pose_repeat_thresholds_are_strict_enough_for_static_frames():
    m = _mod()
    assert m.POSE_MIN_XY_M >= 0.5, m.POSE_MIN_XY_M
    assert m.POSE_MIN_HEADING_RAD >= 0.02, m.POSE_MIN_HEADING_RAD
    assert 1.0 > m.POSE_MIN_XY_M


# ── 导出流程：溯源 / 覆盖保护 / quarantine 保留 ───────────────────────────
def _make_capture(tmp_path: Path, n_frames: int = 4) -> Path:
    """合成一个采集格：probe.frames + capture.json + built.json。"""
    m = _mod()
    cam = _camera()
    src = tmp_path / "probe.frames"
    src.mkdir(parents=True)
    for i in range(n_frames):
        img = np.full((H, W, 3), 200, np.uint8)
        # 每帧沿 x 平移车辆位置，避免位姿去重误杀
        pos = np.array([i * 1.0, 0.0, 0.0])
        pts = _nodes(0.9)
        u, v, ok = cam.project(pts, pos, 0.0)
        for k in range(len(pts) - 1):
            if ok[k] and ok[k + 1] and np.isfinite(u[k]) and np.isfinite(u[k + 1]):
                cv2.line(img, (int(u[k]), int(v[k])),
                         (int(u[k + 1]), int(v[k + 1])), ORANGE, 5)
        pts_r = _nodes(-2.7 - 0.85)
        u, v, ok = cam.project(pts_r, pos, 0.0)
        for k in range(len(pts_r) - 1):
            if ok[k] and ok[k + 1] and np.isfinite(u[k]) and np.isfinite(u[k + 1]):
                cv2.line(img, (int(u[k]), int(v[k])),
                         (int(u[k + 1]), int(v[k + 1])), ORANGE, 5)
        np.savez_compressed(src / f"frame_{i:05d}.npz",
                            frame=img, pos=pos, heading=0.0,
                            cam=np.array([cam], dtype=object), t=float(i))
    (tmp_path / "capture.json").write_text(json.dumps(
        {"schema": "proving_capture_v1", "site": "S1", "cell": "x",
         "split": "train", "status": "ok",
         "env": {"tod_after": "12:00:00", "weather": "rainy"}}),
        encoding="utf-8")
    (tmp_path / "built.json").write_text(
        json.dumps(_built_doc(-2.7)), encoding="utf-8")
    return tmp_path


def _wire_dirs(monkeypatch, tmp_path: Path):
    m = _mod()
    cap = tmp_path / "cap" / "S1" / "cellx"
    cap.mkdir(parents=True)
    pool = tmp_path / "pool"
    pool.mkdir()
    _make_capture(cap)
    monkeypatch.setattr(m, "CAP_DIR", tmp_path / "cap")
    monkeypatch.setattr(m, "POOL_DIR", pool)
    return m, cap, pool, cap / "built.json"


def test_export_cell_writes_pool_with_provenance(tmp_path, monkeypatch):
    m, cap, pool, built = _wire_dirs(monkeypatch, tmp_path)
    rc = m.export_cell("S1", "cellx", str(built), target_frames=3)
    assert rc == 0
    out = pool / "S1" / "cellx"
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "proving_pool_v2"
    assert manifest["pool_status"] == "pilot"
    assert manifest["training_qualified"] is False
    assert manifest["kept"] == 3
    assert manifest["built_source"]["sha256"] == m._sha256_file(built)
    assert len(manifest["frames"]) == 3
    for f in manifest["frames"]:
        assert len(f["sha256"]) == 64
        assert (out / f["file"]).is_file()
    # 每帧标签非空、road 覆盖 line
    z = np.load(out / manifest["frames"][0]["file"], allow_pickle=False)
    assert (z["line"] == 2).sum() > 0
    assert z["road"].sum() >= (z["line"] == 2).sum()


def test_export_refuses_overwrite_and_force_preserves_quarantine(tmp_path, monkeypatch):
    m, cap, pool, built = _wire_dirs(monkeypatch, tmp_path)
    assert m.export_cell("S1", "cellx", str(built), target_frames=3) == 0
    out = pool / "S1" / "cellx"
    # 造一份 quarantine 证据，重导出后必须仍在
    q = out / "quarantine"
    q.mkdir(exist_ok=True)
    (q / "rejected_evidence.npz").write_bytes(b"quarantine-evidence")
    with pytest.raises(SystemExit, match="force"):
        m.export_cell("S1", "cellx", str(built), target_frames=3)
    # manifest 未被触碰
    assert json.loads((out / "manifest.json").read_text("utf-8"))["kept"] == 3
    assert m.export_cell("S1", "cellx", str(built), target_frames=3,
                         force=True) == 0
    stales = list((out / "stale").glob("*"))
    assert stales, "old artifacts must be moved into stale/"
    assert (q / "rejected_evidence.npz").read_bytes() == b"quarantine-evidence"
    assert json.loads((out / "manifest.json").read_text("utf-8"))["kept"] == 3


def test_export_requires_built_geometry(tmp_path, monkeypatch):
    m, cap, pool, built = _wire_dirs(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="缺构建记录"):
        m.export_cell("S1", "cellx", str(tmp_path / "missing.json"),
                      target_frames=3)


def test_resolve_attempt_frames_prefers_evidence_ready(tmp_path):
    """§6.2 attempt 布局：只从 EVIDENCE_READY 的 attempt 取帧。

    INCOMPLETE attempt 的残留帧不得当证据；旧布局 probe.frames 仍兼容。
    """
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpe_attempt", str(root / "scripts" / "m5_proving_export.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cell = tmp_path / "S1" / "noon_clear"
    bad = cell / "attempt_aaa"
    good = cell / "attempt_bbb"
    for att in (bad, good):
        (att / "probe.frames").mkdir(parents=True)
        (att / "probe.frames" / "frame_00000.npz").write_bytes(b"x")
    (bad / "attempt_result.json").write_text('{"status": "INCOMPLETE"}',
                                             encoding="utf-8")
    (good / "attempt_result.json").write_text('{"status": "EVIDENCE_READY"}',
                                              encoding="utf-8")
    import os
    import time
    now = time.time()
    os.utime(bad, (now - 100, now - 100))
    os.utime(good, (now, now))

    frames, rec = m.resolve_attempt_frames(cell)
    assert frames == good / "probe.frames"
    assert rec["status"] == "EVIDENCE_READY"

    # 只有 INCOMPLETE → 拒绝
    import shutil
    shutil.rmtree(good)
    with pytest.raises(SystemExit, match="无可用帧"):
        m.resolve_attempt_frames(cell)

    # 旧布局兼容
    legacy = cell / "probe.frames"
    legacy.mkdir(parents=True)
    (legacy / "frame_00000.npz").write_bytes(b"x")
    frames, rec = m.resolve_attempt_frames(cell)
    assert frames == legacy and rec is None


def test_component_scan_handles_unprojectable_chain_points():
    """回归：链点被 _project 判为无效（depth<=0.1）时不得 IndexError。

    2026-11-08 实测：退化分支已按 j 索引过 depth，末尾又索引一次 →
    index 105 out of bounds for axis 0 with size 7（近处链点不可投影）。
    """
    import importlib.util
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpe_scan", str(root / "scripts" / "m5_proving_export.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from beamng_autopilot.vision.projection import CameraModel

    cam = CameraModel(offset=np.array([0.0, 1.216, 1.386]),
                      fwd_local=np.array([0.0, 0.99994, -0.0112]),
                      up_local=np.array([0.0, 0.0112, 0.99994]),
                      fov_deg=120.0, width=1024, height=800)
    # 链从车后 5m 延伸到车前 25m：车后点 depth<0.1 → 无效 → 退化分支
    s = np.linspace(-5.0, 25.0, 60)
    dense = np.column_stack([s, np.full_like(s, 0.9), np.zeros_like(s)])
    line = {"dense": dense, "normal": np.array([0.0, 1.0])}
    px = np.array([[500.0, 600.0], [512.0, 610.0], [520.0, 620.0]])
    scan = m._component_scan(cam, np.array([0.0, 0.0, 1.5]), 0.0, line, px)
    assert scan["delta_m"] is not None
    assert scan["depths"] is not None and len(scan["depths"]) == len(px)
