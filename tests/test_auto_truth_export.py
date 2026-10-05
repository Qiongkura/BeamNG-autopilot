"""`m5_auto_truth_export.py` 的测试（T16 §4.3：把合格批次落成带凭证的自动真值）。

用合成帧（`auto_truth.make_synthetic_batch`）走完整链路，不碰游戏：
1. 门：annotation 覆盖不足 -> 隔离（带原因）；几何残差（均值/中位）超限 ->
   隔离；横向档位超带 -> 实例被排除；真值点处 RGB 不是该颜色的漆 -> 隔离；
2. 逐点认证：校对不上的点被丢弃、丢弃量可查（不拿它们换 ok=True）；
3. 导出产物：训练包结构（npz=colour+label、meta.json、annotation.json），
   `label_source` **只在复核通过时**才是 engine_verified；
4. 调色板离线重建（缺它会直接 PALETTE_CHANGED，实测踩到）；
5. 负例包用**自己的锚点**命名（旧代码把 anchor 写在负例分支后面：首个负例
   直接 UnboundLocalError，后面的负例沿用上一站锚点）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import math
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
    # 判定顺序：先看有没有落盘帧，再看"无线声明"——两者都必须被隔离
    assert any(("no dumped frames" in r) or ("line-free" in r)
               or ("no line truth points" in r) for r in ev["reasons"]), ev
    # 给了帧、但没有 line_generated=False 声明 -> 仍隔离（不得凭"没画线"当负例）
    fr0 = _frames_from_synthetic(with_line=False)[0][0]
    np.savez_compressed(batch / "frames" / "known_no_line_00000.npz",
                        rgb=fr0["rgb"], annotation=fr0["annotation"],
                        label=fr0["label"], depth_raw=fr0["depth"],
                        camera_json=json.dumps(fr0["camera"]))
    ev2 = m.evaluate_scene(batch, "known_no_line",
                           rep["scenes"]["known_no_line"],
                           min_coverage=0.6, max_after_px=2.0)
    assert ev2["eligible"] is False
    assert any("did not declare this scene line-free" in r
               for r in ev2["reasons"]), ev2


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


def _dump_frames(batch: Path, name: str, frames: list, *, rgb=None) -> None:
    """把帧落盘到 ``batch/frames/<name>_*.npz``（``rgb`` 可整体替换）。"""
    (batch / "frames").mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        np.savez_compressed(
            batch / "frames" / f"{name}_{i:05d}.npz",
            rgb=(np.asarray(fr["rgb"]) if rgb is None else rgb),
            annotation=fr["annotation"], label=fr["label"],
            depth_raw=fr["depth"], camera_json=json.dumps(fr["camera"]))


def test_scene_anchor_parses_site_names():
    m = _load()
    assert m.scene_anchor("known_line_a0s0") == 0
    assert m.scene_anchor("slope_curve_a11s3") == 11
    assert m.scene_anchor("no_anchor_here") == 0
    assert m.scene_anchor("") == 0


def test_line_free_package_uses_its_own_anchor(tmp_path):
    """负例包目录名/meta 必须用自己的锚点（旧代码沿用上一站锚点，实测踩到）。"""
    m = _load()
    frames, _truth = _frames_from_synthetic(with_line=False)
    out = tmp_path / "out"
    res = m.export_line_free_package(frames, out_root=out,
                                     scene="known_no_line_a3s1", anchor=3,
                                     map_name="italy", generator={"version": "t"},
                                     evidence={"line_px": 0, "appearance_like_px": 0})
    assert "m5auto_a3_known_no_line_a3s1" in res["dir"], res
    meta = json.loads((out / "m5auto_a3_known_no_line_a3s1" /
                       "meta.json").read_text(encoding="utf-8"))
    assert meta["source_id"] == "m5auto_a3", meta


def test_main_first_scene_line_free_does_not_crash_and_names_anchor(tmp_path,
                                                                   monkeypatch):
    """端到端回归：批次**第一站**就是负例（旧代码在这里 UnboundLocalError）。"""
    m = _load()
    frames, _t = _frames_from_synthetic(with_line=False)
    batch = tmp_path / "batch"
    _dump_frames(batch, "known_no_line_a3s1", frames)
    rep = {"generator": {"version": "t", "script_sha16": "x"},
           "scene_types": ["known_no_line"],
           "scenes": {"known_no_line_a3s1": {
               "generated": {"lines": [], "line_generated": False},
               "camera_alignment": {"per_frame": []}}}}
    (batch / "scene_report.json").write_text(json.dumps(rep), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "m5_auto_truth_export.py", "--batch", str(batch), "--out", str(out)])
    assert m.main() == 0
    blob = json.loads((out / "export_report.json").read_text(encoding="utf-8"))
    assert [e["scene"] for e in blob["exported"]] == ["known_no_line_a3s1"], blob
    assert (out / "m5auto_a3_known_no_line_a3s1" / "front_main" /
            "frame_00000.npz").is_file()


def test_lateral_band_excludes_far_instance(tmp_path):
    """|lat| 超带的实例被排除：只剩 1.2 档时场景仍可认证。"""
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    batch = tmp_path / "batch"
    _dump_frames(batch, "s", frames)
    rec = {"generated": {"lines": [
               {"role": "near_left", "material": "italy_road_markings_line_thin",
                "lateral_m": 1.2, "truth_points": truth},
               {"role": "near_right", "material": "italy_road_markings_line_thin",
                "lateral_m": 2.4, "truth_points": []}]},
           "camera_alignment": {"per_frame": [{"after_px": 0.5}] * 2}}
    ev = m.evaluate_scene(batch, "s", rec, min_coverage=0.6, max_after_px=2.0,
                          max_line_lat_m=2.0)
    assert ev["eligible"] is True, ev
    assert [x["role"] for x in ev["excluded_instances"]] == ["near_right"]
    assert ev["certified_roles"] == ["near_left"]
    # 全部实例都在带外 -> 隔离（不得当"没有线实例"静默通过）
    rec2 = {"generated": {"lines": [
                {"role": "far_left", "material": "x", "lateral_m": 3.5,
                 "truth_points": truth}]},
            "camera_alignment": {"per_frame": [{"after_px": 0.5}] * 2}}
    ev2 = m.evaluate_scene(batch, "s", rec2, min_coverage=0.6, max_after_px=2.0,
                           max_line_lat_m=2.0)
    assert ev2["eligible"] is False
    assert any("lateral band" in r for r in ev2["reasons"]), ev2


def test_residual_median_gate_rejects_one_bad_frame(tmp_path):
    """均值被坏帧掩盖时中位门要拦住：0.5/0.5/9.0 均值 3.3 中位 0.5 -> 两道都拒。"""
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    batch = tmp_path / "batch"
    _dump_frames(batch, "s", frames)
    rec = {"generated": {"lines": [{"role": "left", "lateral_m": 1.8,
                                    "truth_points": truth}]},
           "camera_alignment": {"per_frame": [{"after_px": 1.2}, {"after_px": 1.2},
                                              {"after_px": 1.4}]}}
    ev = m.evaluate_scene(batch, "s", rec, min_coverage=0.6, max_after_px=2.0,
                          max_after_median_px=1.0)
    assert ev["eligible"] is False
    assert any("median" in r for r in ev["reasons"]), ev
    assert ev["after_px_median"] == 1.2


def test_paint_agreement_gate_rejects_label_without_paint(tmp_path):
    """标签在 RGB 不是漆的地方写线 -> 拒（旧代码只查 annotation，会放行）。"""
    m = _load()
    frames, truth = _frames_from_synthetic(with_line=True)
    batch = tmp_path / "batch"
    # RGB 换成"没有漆"的深灰：label 仍是合成批次的线类
    _dump_frames(batch, "s", frames,
                 rgb=np.full_like(np.asarray(frames[0]["rgb"]), 60))
    rec = {"generated": {"lines": [{"role": "left", "lateral_m": 1.8,
                                    "material": "italy_road_markings_line_thin",
                                    "truth_points": truth}]},
           "camera_alignment": {"per_frame": [{"after_px": 0.5}] * 2}}
    ev = m.evaluate_scene(batch, "s", rec, min_coverage=0.6, max_after_px=2.0,
                          min_paint_agreement=0.9)
    assert ev["eligible"] is False
    assert ev["paint_agreement_pooled"] == 0.0
    assert any("paint appearance" in r for r in ev["reasons"]), ev
    # 正常合成帧（线是暖白 238/230/120）-> 过
    _dump_frames(batch, "s", frames)
    ev2 = m.evaluate_scene(batch, "s", rec, min_coverage=0.6, max_after_px=2.0,
                           min_paint_agreement=0.9)
    assert ev2["eligible"] is True, ev2
    assert ev2["paint_agreement_pooled"] >= 0.9


def test_paint_criterion_and_branches():
    m = _load()
    assert m.PAINT_CRITERION == "achromatic_bright_or_yellow"
    rgb = np.full((8, 8, 3), 70, dtype=np.uint8)          # 沥青灰
    # 亮的无彩脊（开发集人工线像素 p10≈154 以上）-> 漆
    rgb[4, 4] = (162, 162, 162)
    assert m.paint_like(rgb, 4, 4) is True
    # 黄漆 -> 漆；蓝漆不认证（与蓝遮挡车分不开，宁可隔离）
    rgb[4, 4] = (205, 190, 90)
    assert m.paint_like(rgb, 4, 4) is True
    rgb[4, 4] = (60, 80, 200)
    assert m.paint_like(rgb, 4, 4) is False
    # 暗沥青 -> 不是漆
    rgb[4, 4] = (80, 78, 76)
    assert m.paint_like(rgb, 4, 4) is False
    assert m.paint_like(rgb, -3, 4) is None      # 画面外：UNKNOWN，不算失败
    assert m.paint_like(rgb, 4, 40) is None
    # 相机背后的点：project_point 出 NaN，先判有限再取整（实测踩到 ValueError）
    assert m.paint_like(rgb, float("nan"), 4.0) is None
    assert m.paint_like(rgb, 4.0, float("nan")) is None


def _struct_rec(frames: list, *, structures: list, after_px: float = 0.5):
    return {"generated": {"lines": [], "line_generated": False,
                          "structures": structures},
            "camera_alignment": {"per_frame": [{"after_px": after_px}
                                               for _ in frames]}}


def test_structure_negative_is_gated_by_annotation_not_appearance(tmp_path):
    """结构负例：门是 **annotation 线类像素为 0**；外观只上报、不判定。

    机制（T16 §16.1）：模型在墙/护栏上多画线，所以要放"像线的非漆结构"。
    为什么外观不能当门（2026-09-30 实测）：`line_evidence_mask(mode="appearance")`
    在**已验证的无线负例包**上就有 13k–68k px/8 帧（亮铺装/亮碎石都命中"白"），
    拿它当门会把负例通道杀光。旧代码的 `appearance_line_check`（需要真值点）
    对无线场景恒返回 0——一个恒不生效的门，已去掉。
    """
    m = _load()
    frames, _t = _frames_from_synthetic(with_line=False)
    batch = tmp_path / "batch"
    _dump_frames(batch, "s", frames)
    structs = [{"id": "m5_wall_s_0", "kind": "wall_stone", "shape": "x.dae",
                "pos": [0.0, 0.0, 0.0]}]
    ev = m.evaluate_scene(batch, "s", _struct_rec(frames, structures=structs),
                          min_coverage=0.8, max_after_px=2.0)
    assert ev["eligible"] is True, ev
    assert ev["line_px_in_frames"] == 0
    # 结构被记进判定（导出侧据此写进凭证），外观分解是**证据**不是门
    assert ev["structures"] and ev["structures"][0]["kind"] == "wall_stone"
    assert ev["structure_appearance"]["n_structures"] == 1
    assert "outside" in ev["structure_appearance"]
    # 没声明结构的普通负例照样过（门只看 annotation），但判定里不该有结构键
    ev2 = m.evaluate_scene(batch, "s", _struct_rec(frames, structures=[]),
                           min_coverage=0.8, max_after_px=2.0)
    assert ev2["eligible"] is True
    assert not ev2.get("structures")
    # 外观数**必须上报**（哪怕不判定）：否则读报告的人以为外观被查过
    assert ev["appearance_like_px"] is not None

def test_structure_negative_still_rejects_line_class_pixels(tmp_path):
    """结构负例也不能有 annotation 线类像素（引擎说这是漆线就不是负例）。"""
    m = _load()
    frames, _t = _frames_from_synthetic(with_line=True)   # label 里有线类
    batch = tmp_path / "batch"
    _dump_frames(batch, "s", frames)
    structs = [{"id": "x", "kind": "wall_stone", "shape": "x.dae",
                "pos": [0.0, 0.0, 0.0]}]
    ev = m.evaluate_scene(batch, "s", _struct_rec(frames, structures=structs),
                          min_coverage=0.8, max_after_px=2.0)
    assert ev["eligible"] is False
    assert any("line-class pixels present" in r for r in ev["reasons"]), ev


def test_negative_credential_carries_its_own_evidence(tmp_path):
    """负例凭证必须能自证：`truth_kind` + `line_free_evidence` 不得被丢掉。

    实测踩到（2026-09-30）：`_normalize_provenance` 的 labels 白名单里没有这两键，
    导出侧写进去的"零线像素 + 已声明非漆结构 + 外观归因"在凭证里全部消失，
    读凭证的人无法判断"凭什么算合格负例"（尤其结构负例：像线的外观来自
    已声明结构，这条证据必须随凭证走）。
    """
    m = _load()
    frames, _t = _frames_from_synthetic(with_line=False)
    out = tmp_path / "out"
    ev = {"line_px": 0, "appearance_like_px": 1234,
          "structures": [{"id": "m5_wall_x_0", "kind": "wall_stone",
                          "shape": "/levels/italy/.../italy_wall_stone_"
                                   "bricktop.dae"}],
          "structure_appearance": {"inside": 1000, "outside": 234,
                                   "radius_px": 60, "n_structures": 1}}
    m.export_line_free_package(frames, out_root=out, scene="known_no_line_a3s1",
                               anchor=3, map_name="italy",
                               generator={"version": "t"}, evidence=ev)
    cred = json.loads((out / "m5auto_a3_known_no_line_a3s1" /
                       "annotation.json").read_text(encoding="utf-8"))
    lab = (cred.get("truth_provenance") or {}).get("labels") or {}
    assert lab.get("truth_kind") == "line_free", lab
    got = lab.get("line_free_evidence") or {}
    assert got.get("line_px") == 0
    assert got.get("structures") and got["structures"][0]["kind"] == "wall_stone"
    assert (got.get("structure_appearance") or {}).get("inside") == 1000


def test_camera_block_from_frames_matches_render_pose():
    """导出包必须带 `cameras` 块（探针建相机模型用；缺它整 run 判 UNKNOWN）。

    实测踩到：R3 首轮验收 8/8 帧 UNKNOWN、C=0/R=0。局部系用**录制时的世界
    位姿**换算（不猜约定）：合成相机沿站点方向看且带 -12° 俯角，故
    `fwd_local` 应≈(0, cos12°, -sin12°)、`up_local`≈(0, sin12°, cos12°)；
    offset 用采集/生成共用的 CAMERA_POS。
    """
    m = _load()
    frames, _t = _frames_from_synthetic(with_line=False)
    site_dir = [1.0, 0.0, 0.0]
    blk = m.camera_block_from_frames(frames, site_dir)
    assert "front_main" in blk, blk
    cam = blk["front_main"]
    assert cam["fov_deg"] == 65.0 and cam["width"] > 0 and cam["height"] > 0
    fwd = cam["fwd"]
    up = cam["up"]
    # 合成相机的 fwd 沿站点方向（+x）且带 -12° 俯角 -> 车体系里是 (0,cos,-sin)
    c, sn = math.cos(math.radians(12.0)), math.sin(math.radians(12.0))
    assert abs(fwd[0]) < 1e-3 and abs(fwd[1] - c) < 1e-3, fwd
    assert abs(fwd[2] + sn) < 1e-3, fwd
    assert abs(up[0]) < 1e-3 and abs(up[1] - sn) < 1e-3, up
    assert abs(up[2] - c) < 1e-3, up
    # 缺相机/坏方向：返回空表（调用方写不出块 -> 探针判 UNKNOWN，不猜）
    assert m.camera_block_from_frames([], site_dir) == {}
    assert m.camera_block_from_frames(frames, [0.0, 0.0, 0.0]) == {}

def test_exported_meta_carries_road_identity():
    """导出包 meta 必须带**可机读的路段身份**（road_id/station_m/mid）。

    为什么（2026-10-05 最终集审计踩到）：封存最终集要判"是否与训练同路段
    （相邻站点）"，而 meta 原先没有路段标识，只能靠目录名里的锚点编号——各批次
    锚点编号互不可比（各批自己的 offset），于是"锚点不重叠"这条判据**静默失效**
    （过滤器 0 排除）。测试：导出后用生成侧路段值对账，缺失即失败。
    """
    import importlib.util
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_ate_road", root / "scripts" / "m5_auto_truth_export.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_ate_road"] = m
    spec.loader.exec_module(m)
    src = (_P(__file__).resolve().parents[1] / "scripts"
           / "m5_auto_truth_export.py").read_text(encoding="utf-8")
    # 正例包从 rec 取；负例包没有 rec -> 走可选参数 site，由 main 传 rec["scene"]
    assert '_scene_block(rec).get("road_id")' in src
    # rec["scene"] 是字符串的批次不能崩（实测崩过 6 个场景）
    assert "def _scene_block(rec) -> dict:" in src
    assert "isinstance(sc, dict)" in src
    assert 'site.get("road_id") if isinstance(site, dict) else None' in src
    assert 'site=(rec.get("scene") or {})' in src, "main 必须把路段身份传给负例导出"
    assert '"station_m"' in src and '"mid"' in src
    assert "site: dict | None = None" in src, "负例导出的 site 必须是可选参数（老调用不受影响）"

def test_scene_block_tolerates_string_scene_field():
    """``rec["scene"]`` 是字符串时按空处理（不崩）——实测 6 个场景因此被隔离过。"""
    import importlib.util
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_ate_block", root / "scripts" / "m5_auto_truth_export.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_ate_block"] = m
    spec.loader.exec_module(m)
    assert m._scene_block({"scene": {"road_id": "1.0"}}) == {"road_id": "1.0"}
    assert m._scene_block({"scene": "known_line_a0s0"}) == {}
    assert m._scene_block({"scene": None}) == {}
    assert m._scene_block({}) == {}
    assert m._scene_block(None) == {}

def test_capture_path_reaps_its_own_game_processes():
    """采集脚本收尾必须关掉**本次新起**的游戏进程（只关连接不够）。

    实测（2026-10-05）：连续四轮受控场景采集后机器上积了 9 个 BeamNG 实例
    （conn.close() 只关连接），把 collect 的资源门打红（"采集前没有游戏进程"
    断言失败），并会拖慢后续实验（2026-09-25 同款缺陷：一局挂 1.5 小时，
    推理 p95 漂 7.6 倍）。必须复用采集侧助手（只杀"新起 + 创建时间可验证"的
    pid，不碰用户会话），不得自己手写 kill。
    """
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text(encoding="utf-8")
    assert "close_started_game" in src, "收尾必须关自己起的游戏"
    assert "game_pids()" in src, "要有'采集前'的进程基线（差集才安全）"
    assert "launched_after=_launched_after" in src, "要传启动时刻做所有权校验"
    assert "taskkill" not in src and "Stop-Process" not in src,         "不得自己手写 kill（所有权校验在助手里）"

def test_fsd_drive_cleanup_has_vis_index_initialised():
    """驾驶收尾路径引用的 `_vis_index` 必须在入口就初始化（实测崩溃 + 掩盖错误）。

    2026-10-05 实测：非 --vis 的运行走到收尾就 `UnboundLocalError` ——
    ① 把真正的失败原因（山道场景的 lane placement ABORT）盖掉；
    ② 收尾中断，**游戏进程不被关闭**（本机因此积过 9 个 BeamNG 实例，
    把 tests/test_seg_collect_unattended.py 的资源门打红）。
    """
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "beamng_autopilot"
           / "fsd_drive.py").read_text(encoding="utf-8")
    i_init = src.index("_vis_index: list[dict] = []")
    i_use = src.index("if (_vis_index or []) and args.out:")
    assert i_init < i_use, "初始化必须在收尾引用之前（同一函数、入口处）"
    # 入口初始化应与 hist 同处（连接建立之前），不是只在 --vis 分支里
    i_hist = src.index("hist: list[dict] = []")
    assert abs(i_init - i_hist) < 400, "应与 hist 一起在入口初始化"
