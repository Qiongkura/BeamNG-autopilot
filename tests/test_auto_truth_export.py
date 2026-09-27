"""`m5_auto_truth_export.py` 的测试（T16 §4.3：把合格批次落成带凭证的自动真值）。

用合成帧（`auto_truth.make_synthetic_batch`）走完整链路，不碰游戏：
1. 两道门：annotation 覆盖不足 -> 隔离（带原因）；几何残差超限 -> 隔离；
2. 逐点认证：校对不上的点被丢弃、丢弃量可查（不拿它们换 ok=True）；
3. 导出产物：训练包结构（npz=colour+label、meta.json、annotation.json），
   `label_source` **只在复核通过时**才是 engine_verified；
4. 调色板离线重建（缺它会直接 PALETTE_CHANGED，实测踩到）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import make_synthetic_batch  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_auto_truth_export", ROOT / "scripts" / "m5_auto_truth_export.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_auto_truth_export"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frames_from_synthetic(with_line: bool = True) -> tuple:
    """合成批次的帧 -> 导出用的帧 dict 列表（补 path，转成 list）。"""
    batch = make_synthetic_batch(with_line=with_line, n_frames=2)
    frames = []
    for i, fr in enumerate(batch["frames"]):
        frames.append({"path": Path(f"frame_{i:05d}.npz"),
                       "rgb": np.asarray(fr["rgb"]),
                       "annotation": np.asarray(fr["annotation"]),
                       "label": np.asarray(fr["label"]),
                       "depth": np.asarray(fr["depth"], dtype=np.float32),
                       "camera": fr["camera"]})
    truth = [dict(p) for p in batch["frames"][0]["truth_points"]]
    return frames, truth


def _rec(frames: list, *, after_px: float = 1.0, with_lines: bool = True):
    """构造最小 scene 记录（真实报告里的字段子集）。"""
    lines = ([{"role": "left", "truth_points": []}] if with_lines else [])
    return {"generated": {"lines": lines, "line_generated": bool(lines)},
            "camera_alignment": {"per_frame": [{"after_px": after_px}
                                               for _ in frames]}}


def test_palette_rebuild_registers_observed_colours():
    m = _load()
    frames, _truth = _frames_from_synthetic()
    pal = m.palette_from_frames(frames)
    assert pal.get("sha") and pal.get("road") and pal.get("line")
    classes = pal.get("classes") or {}
    # 合成画面里出现过的颜色都应登记（否则 UNKNOWN_CLASS）
    ann = np.asarray(frames[0]["annotation"])[:, :, :3].reshape(-1, 3)
    for row in np.unique(ann, axis=0)[:5]:
        c = tuple(int(v) for v in row)
        assert any(list(v) == list(c) for v in classes.values()), c


def test_certify_points_keeps_only_checkable_points():
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    cams = [f["camera"] for f in frames]
    cert = m.certify_points(truth, frames, cams, radius_px=2)
    assert cert["n_total"] == len(truth) * len(frames)
    assert sum(len(k) for k in cert["kept_by_frame"]) > 0, cert
    assert cert["drops"]["no_line_within_radius"] >= 0
    # 把线抹掉（label 清 0）-> 全部丢弃，且原因是"2 px 内没有线"
    blank = [dict(f, label=np.zeros_like(np.asarray(f["label"]))) for f in frames]
    cert2 = m.certify_points(truth, blank, cams, radius_px=2)
    assert sum(len(k) for k in cert2["kept_by_frame"]) == 0
    assert cert2["drops"]["no_line_within_radius"] > 0


def test_export_writes_training_package_and_credential(tmp_path):
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    rec = _rec(frames)
    res = m.export_scene(rec, truth, frames, out_root=tmp_path,
                         scene="known_line_a0s0", anchor=0, map_name="italy",
                         generator={"version": "test", "script_sha16": "abc123"})
    view = Path(res["dir"])
    assert view.is_dir() and (view / "frame_00000.npz").is_file()
    npz = np.load(view / "frame_00000.npz", allow_pickle=False)
    assert "colour" in npz and "label" in npz          # 训练器要的键
    meta = json.loads((view.parent / "meta.json").read_text(encoding="utf-8"))
    for k in ("map_name", "source_id", "map_name_source"):
        assert meta.get(k), k
    assert meta["source_id"].startswith("m5auto_a")    # 场景族 = 锚点
    cred = json.loads((view.parent / "annotation.json").read_text(encoding="utf-8"))
    assert cred.get("truth_contract") == "v1"
    prov = cred.get("truth_provenance") or {}
    assert (prov.get("report") or {}).get("verifier_version")
    assert (prov.get("labels") or {}).get("label_sha")
    # 合成批次复核通过 -> 才允许 engine_verified
    assert cred.get("label_source") == "engine_verified", cred.get("why")
    assert res["verified"] is True


def test_scene_without_line_truth_is_isolated(tmp_path):
    m = _load()
    frames, _truth = _frames_from_synthetic(with_line=False)
    batch = tmp_path / "batch"
    (batch / "frames").mkdir(parents=True)
    rep = {"scenes": {"known_no_line": {"generated": {"lines": []},
                                        "camera_alignment": {"per_frame": []}}}}
    (batch / "scene_report.json").write_text(json.dumps(rep), encoding="utf-8")
    ev = m.evaluate_scene(batch, "known_no_line", rep["scenes"]["known_no_line"],
                          min_coverage=0.6, max_after_px=2.0)
    assert ev["eligible"] is False
    assert any("no line truth points" in r for r in ev["reasons"]), ev


def test_low_coverage_and_bad_geometry_are_isolated(tmp_path):
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    batch = tmp_path / "batch"
    (batch / "frames").mkdir(parents=True)
    # 落盘帧（evaluate_scene 会自己读 frames 目录）——**label 清空**模拟
    # "annotation 没覆盖这条线"（合成场景本身覆盖=1.0，不能用它测覆盖门）
    for i, fr in enumerate(frames):
        np.savez_compressed(batch / "frames" / f"s_{i:05d}.npz",
                            rgb=fr["rgb"], annotation=fr["annotation"],
                            label=np.zeros_like(np.asarray(fr["label"])),
                            depth_raw=fr["depth"],
                            camera_json=json.dumps(fr["camera"]))
    rec = {"generated": {"lines": [{"role": "left", "truth_points": truth}]},
           "camera_alignment": {"per_frame": [{"after_px": 1.0}] * 2}}
    ev = m.evaluate_scene(batch, "s", rec, min_coverage=0.6, max_after_px=2.0)
    assert ev["eligible"] is False and ev["coverage"] is not None
    assert any("coverage" in r for r in ev["reasons"]), ev
    # 覆盖够（恢复 label）但几何残差超限 -> 仍隔离
    for i, fr in enumerate(frames):
        np.savez_compressed(batch / "frames" / f"s_{i:05d}.npz",
                            rgb=fr["rgb"], annotation=fr["annotation"],
                            label=fr["label"], depth_raw=fr["depth"],
                            camera_json=json.dumps(fr["camera"]))
    ev_ok = m.evaluate_scene(batch, "s", rec, min_coverage=0.6,
                             max_after_px=2.0)
    assert ev_ok["eligible"] is True, ev_ok
    rec_bad = {"generated": {"lines": [{"role": "left", "truth_points": truth}]},
               "camera_alignment": {"per_frame": [{"after_px": 9.0}] * 2}}
    ev2 = m.evaluate_scene(batch, "s", rec_bad, min_coverage=0.0,
                           max_after_px=2.0)
    assert ev2["eligible"] is False
    assert any("residual" in r for r in ev2["reasons"]), ev2
