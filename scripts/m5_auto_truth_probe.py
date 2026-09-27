"""自动真值探针：先证明"能自动产/能自动拒"，再决定扩量（T16 方案 §4.1/§4.3）。

纯逻辑模式（默认，不碰游戏）跑五类已知场景与五个反例：

* 场景：已知有线 / 已知无线（生成器声明未生成漆线且无内嵌漆线纹理）/ 遮挡 /
  坡面曲线 / 材质混合——每个都要 ``verify_batch`` 过；
* 反例：相机翻转 / 尺寸变化 / 时间错帧 / 遮挡物插入 / 标签篡改——每个都必须被
  对应的拒绝码拒掉（``inject`` 只注入一类错误）。

退出码：``0`` 仅当"五类全过 + 五反例全被对应码拒绝"；否则非 0（报告里能看到
哪一个没做到）。**探针失败时自动换场景/隔离该资产，不盲目扩量**（方案 §4.3：
"通过后使用现有 engine_verified 来源；失败时隔离并自动换场景，不盲目扩量"）。

Tech 模式（``--runtime tech``，本开发窗口不执行，代码完整留给串行阶段）：
薄包装，连 BeamNGpy 1.35.1 采集 RGB/annotation/depth，喂同一个 ``verify_batch``。
缺 BeamNGpy、连不上、没有真值点、深度换算自检不过，都**大声失败**（非 0 退出 +
明确信息），不把 UNKNOWN 当通过。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_auto_truth_probe.py --runtime pure
    .venv\\Scripts\\python.exe scripts\\m5_auto_truth_probe.py --runtime tech \\
        --map italy --vehicle etk800 --truth-json assets\\probe_truth_italy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.auto_truth import (  # noqa: E402
    CAMERA_FLIP, LABEL_TAMPERED, OCCLUSION_INSERT, RESOLUTION_MISMATCH,
    TIME_MISMATCH, TRUTH_CONTRACT, VERIFIER_VERSION, frame_content_shas,
    inject, make_synthetic_batch, palette_sha, verify_batch,
)
from beamng_autopilot_tech.annotations import (  # noqa: E402
    annotation_palette, to_label,
)

#: 五类已知场景（name -> make_synthetic_batch 参数）
SCENES: tuple[tuple[str, dict], ...] = (
    ("known_line", {"with_line": True}),
    ("known_no_line", {"with_line": False}),
    ("occluded_line", {"with_line": True, "with_occluder": True}),
    ("slope_curve", {"with_line": True, "curve_deg": 18.0, "slope_deg": 6.0}),
    ("material_mix", {"with_line": True, "second_line": True,
                      "material_mix": True}),
)

#: 五个反例（kind -> 必须出现的拒绝码）
COUNTEREXAMPLES: tuple[tuple[str, str], ...] = (
    ("flip", CAMERA_FLIP),
    ("resize", RESOLUTION_MISMATCH),
    ("time_shift", TIME_MISMATCH),
    ("occlude", OCCLUSION_INSERT),
    ("tamper", LABEL_TAMPERED),
)


def run_pure(*, width: int = 192, height: int = 144, n_frames: int = 2) -> dict:
    """跑五类场景 + 五个反例，返回可 JSON 化的报告。"""
    scenes: list[dict] = []
    for name, kwargs in SCENES:
        batch = make_synthetic_batch(width=width, height=height,
                                     n_frames=n_frames, **kwargs)
        report = verify_batch(batch)
        scenes.append({
            "name": name, "ok": bool(report["ok"]),
            "rejections": sorted({r["code"] for r in report["rejections"]}),
            "channels": {k: v["status"] for k, v in report["channels"].items()},
            "evidence": {
                "projection_checked": report["stats"]["projection"]["checked"],
                "projection_mismatches": report["stats"]["projection"]["mismatches"],
                "occlusion_occluded_px": report["stats"]["occlusion"]["occluded_px"],
                "n_frames": report["stats"]["n_frames"],
            },
        })
    base = make_synthetic_batch(width=width, height=height, n_frames=n_frames,
                                with_line=True)
    counterexamples: list[dict] = []
    for kind, expected in COUNTEREXAMPLES:
        batch = inject(base, kind)
        report = verify_batch(batch)
        codes = sorted({r["code"] for r in report["rejections"]})
        counterexamples.append({
            "kind": kind, "expected_code": expected, "rejected_codes": codes,
            "ok": bool(expected in codes and not report["ok"]),
        })
    n_scenes_ok = sum(1 for s in scenes if s["ok"])
    n_rejected = sum(1 for c in counterexamples if c["ok"])
    return {
        "runtime": "pure",
        "verifier_version": VERIFIER_VERSION,
        "truth_contract": TRUTH_CONTRACT,
        "ok": bool(n_scenes_ok == len(scenes)
                   and n_rejected == len(counterexamples)),
        "n_scenes_ok": n_scenes_ok, "n_scenes": len(scenes),
        "n_counterexamples_rejected": n_rejected,
        "n_counterexamples": len(counterexamples),
        "scenes": scenes,
        "counterexamples": counterexamples,
        "note": ("探针失败时自动换场景、不盲目扩量（方案 §4.3）；纯逻辑结论只覆盖"
                 "合成场景，不代替 Tech 实机能力"),
    }


# ── Tech 模式（薄包装；本开发窗口不执行） ──────────────────────────────────
def _depth_to_meters(raw, *, near: float, far: float, mode: str) -> np.ndarray:
    """把 BeamNGpy 的深度缓冲转成沿光轴深度（米）。

    ``mode="ndc"``（默认）按 OpenGL 非线性深度反解；``mode="linear"`` 按
    ``near + d*(far-near)``。**这个换算必须由实机自检确认**（见
    `_tech_depth_sanity`）：不确定就大声失败，不把错的深度喂给遮挡判据。
    """
    d = np.asarray(raw, dtype=np.float64)
    if d.size == 0:
        return d
    if float(np.nanmax(d)) > 1.5:
        # 已经是米（部分配置 integer_depth=False + 非共享内存会返回 [0,1]，
        # 但有的路径直接给米）：原样返回，仍由自检把关。
        return d
    if mode == "linear":
        return near + d * (far - near)
    z = (2.0 * near * far) / (far + near - (2.0 * d - 1.0) * (far - near))
    return z


def _tech_depth_sanity(depth: np.ndarray, *, cam_height_m: float = 1.386,
                       pitch_deg: float = -0.64) -> tuple[bool, str]:
    """实机深度自检：画面下缘中央应看到车头前 ~1.4 m 高的地面。

    返回 ``(ok, why)``。不通过时探针必须非 0 退出并提示重新标定深度换算，
    而不是带病跑判定。
    """
    if depth is None or np.size(depth) == 0:
        return False, "depth buffer is empty"
    h, w = np.asarray(depth).shape[:2]
    rows = slice(int(h * 0.75), h)
    cols = slice(int(w * 0.4), int(w * 0.6))
    patch = np.asarray(depth, dtype=np.float64)[rows, cols]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return False, "no valid depth in the lower-centre patch"
    value = float(np.median(patch))
    expected = float(cam_height_m) / max(0.2, np.sin(np.radians(-pitch_deg)))
    if not (0.2 * expected <= value <= 4.0 * expected):
        return False, (f"lower-centre depth median {value:.2f} m is outside "
                       f"[{0.2 * expected:.2f}, {4.0 * expected:.2f}] m "
                       f"(expected about {expected:.2f} m for the calibrated "
                       "mount); convert/calibrate the depth buffer before "
                       "trusting occlusion checks")
    return True, f"lower-centre depth median {value:.2f} m (expected ~{expected:.2f} m)"


def _load_truth_points(path: str | None) -> list[dict]:
    """读真值点 JSON；文件缺失/不可解析返回空表（调用方据此大声失败）。"""
    if not path:
        return []
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[auto-truth] 读不了 --truth-json {path}: "
              f"{type(exc).__name__}: {exc}")
        return []
    pts = blob.get("truth_points") if isinstance(blob, dict) else blob
    return [p for p in (pts or []) if isinstance(p, dict)]


def run_tech(args) -> dict:
    """连 BeamNGpy 采 ``--frames`` 帧 RGB/annotation/depth，喂同一个 ``verify_batch``。

    真实/连接问题一律返回 ``ok=False`` + ``error``（调用方非 0 退出），不静默降级。
    标签由 **annotation + 会话调色板** 经 ``annotations.to_label`` 得到，
    ``label_source="engine_annotation"``（冻结文档 §3.5：这是**不完整**来源，
    资格会是 unverified_labels/unknown，不会因为本探针跑通就变 verified）。
    """
    checks: list[str] = []
    try:
        from beamng_autopilot.connector import BeamNGConnector
        from beamng_autopilot_tech.providers import (
            CAMERA_DIR, CAMERA_FOV_DEG, CAMERA_POS, CAMERA_UP,
            check_graphics_quality,
        )
    except ImportError as exc:
        return {"runtime": "tech", "ok": False,
                "error": f"BeamNGpy/连接器不可用：{type(exc).__name__}: {exc}",
                "hint": "安装 requirements 里的 beamngpy 1.35.1 后再跑 --runtime tech"}
    truth_points = _load_truth_points(args.truth_json)
    if not truth_points:
        return {"runtime": "tech", "ok": False,
                "error": "--truth-json 未提供或为空：没有已知 3D 真值点，"
                         "投影/遮挡不可判（UNKNOWN 不是通过）",
                "hint": "先用受控场景脚本导出漆线/路面世界点，再跑本探针"}
    conn = BeamNGConnector(host=args.host, port=args.port)
    cam = None
    try:
        check_graphics_quality(conn.user_dir)
        conn.open(launch=bool(args.launch))
        from beamngpy.sensors import Camera
        width, height = int(args.width), int(args.height)
        near, far = 0.05, 150.0
        name = f"m5_auto_truth_probe_{abs(hash(str(conn.user_dir))) % 10000}"
        with conn.io_lock:
            cam = Camera(name, conn.bng, conn.vehicle, requested_update_time=0.05,
                         pos=CAMERA_POS, dir=CAMERA_DIR, up=CAMERA_UP,
                         resolution=(width, height),
                         field_of_view_y=CAMERA_FOV_DEG,
                         near_far_planes=(near, far),
                         is_using_shared_memory=True,
                         is_render_colours=True, is_render_annotations=True,
                         is_render_depth=True, is_visualised=False,
                         integer_depth=False, postprocess_depth=False)
        frames: list[dict] = []
        palette = None
        for i in range(max(1, int(args.frames))):
            if i:
                with conn.io_lock:
                    conn.bng.control.step(int(args.step))
            with conn.io_lock:
                data = cam.poll()
            colour = data.get("colour")
            ann = data.get("annotation")
            raw_depth = data.get("depth")
            if colour is None or ann is None or raw_depth is None:
                return {"runtime": "tech", "ok": False,
                        "error": "BeamNGpy 未返回 colour/annotation/depth（检查"
                                 " annotations 模式与 GPU prepass buffer）"}
            rgb = np.ascontiguousarray(np.asarray(colour), dtype=np.uint8)
            annotation = np.ascontiguousarray(np.asarray(ann), dtype=np.uint8)
            depth_m = _depth_to_meters(raw_depth, near=near, far=far,
                                       mode=args.depth_mode)
            ok, why = _tech_depth_sanity(depth_m)
            if not ok:
                return {"runtime": "tech", "ok": False,
                        "error": f"深度换算自检失败：{why}",
                        "hint": "试 --depth-mode linear；仍不过就说明这一配置的"
                                "深度语义与假设不同，必须重新标定后再用遮挡判据"}
            checks.append(why)
            # 相机位姿：直接问传感器（引擎给的世界位姿），不猜车辆坐标约定。
            try:
                cam_pos = [float(v) for v in cam.get_position()]
                fwd = np.asarray(cam.get_direction(), dtype=float)
            except Exception as exc:                          # noqa: BLE001
                return {"runtime": "tech", "ok": False,
                        "error": f"读不到相机世界位姿（{type(exc).__name__}: {exc}）",
                        "hint": "BeamNGpy 需要传感器已 attach 且场景已加载"}
            fwd = fwd / max(1e-12, float(np.linalg.norm(fwd)))
            up_world = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up_world)
            right = right / max(1e-12, float(np.linalg.norm(right)))
            up = np.cross(right, fwd)
            if palette is None:
                palette = _tech_palette(conn, annotation)
            label = to_label(annotation, road_colors=palette["road"],
                             line_colors=palette["line"])
            label = np.ascontiguousarray(label, dtype=np.uint8)
            fid = f"tech_{i:05d}"
            src_sha, lab_sha = frame_content_shas(rgb, label)
            frames.append({
                "frame_id": fid, "timestamp": float(i) * 0.05,
                "channel_ids": {"rgb": fid, "annotation": fid, "depth": fid},
                "camera": {"name": "front_main", "width": width,
                           "height": height, "fov_y_deg": float(CAMERA_FOV_DEG),
                           "pose": {"pos": cam_pos,
                                    "basis": {"right": [float(v) for v in right],
                                              "fwd": [float(v) for v in fwd],
                                              "up": [float(v) for v in up]}}},
                "rgb": rgb, "annotation": annotation,
                "depth": np.ascontiguousarray(depth_m, dtype=np.float32),
                "label": label, "label_sha": lab_sha,
                "source_image_sha": src_sha,
                "truth_points": [dict(p) for p in truth_points],
            })
        run_id = f"m5_auto_truth_probe_{args.map or 'unknown'}"
        batch = {"frames": frames,
                 "scene": {"map": str(args.map), "segment": str(args.segment),
                           "run_id": run_id,
                           "scene_seed": int(args.scene_seed),
                           "game_version": "beamng_tech",
                           "renderer": str(args.renderer),
                           # 真实地图的漆线生成/纹理内嵌情况未知：不写 False，
                           # 否则会把"没测"当成"已证明无线"。
                           "line_generated": None,
                           "line_texture_embedded": None,
                           "shoulder_defined": None,
                           "roles_defined": bool(
                               any(p.get("role") for p in truth_points)),
                           "road_defined": True,
                           "label_source": "engine_annotation"},
                 "palette": palette,
                 "run": {"id": run_id, "scene_seed": int(args.scene_seed),
                         "game_version": "beamng_tech",
                         "renderer": str(args.renderer)}}
        report = verify_batch(batch)
        measured = sorted(k for k, v in report["channels"].items()
                          if v["status"] == "measured")
        checked = report["stats"]["projection"]["checked"]
        # 退出码语义：只有"没有拒绝码 + 至少一个通道 measured + 投影真的核过"
        # 才算通过；否则报告里写清为什么（UNKNOWN 不当 PASS）。
        why = []
        if not report["ok"]:
            why.append(f"rejections={sorted({r['code'] for r in report['rejections']})}")
        if not measured:
            why.append("no channel measured (engine annotation is not verified truth)")
        if checked <= 0:
            why.append("no truth point could be projected/compared")
        return {"runtime": "tech", "ok": bool(report["ok"] and measured and checked),
                "why": "; ".join(why),
                "map": args.map, "n_frames": len(frames),
                "truth_points": len(truth_points),
                "depth_checks": checks,
                "measured_channels": measured,
                "projection_checked": checked,
                "verifier_version": VERIFIER_VERSION,
                "truth_contract": TRUTH_CONTRACT,
                "rejections": report["rejections"],
                "channels": {k: v["status"] for k, v in report["channels"].items()},
                "stats": report["stats"]}
    finally:
        if cam is not None:
            try:
                with conn.io_lock:
                    cam.remove()
            except Exception:                                # noqa: BLE001
                pass
        try:
            conn.close()
        except Exception:                                    # noqa: BLE001
            pass


def _tech_palette(conn, annotation: np.ndarray) -> dict:
    """Tech 调色板：``bng.get_annotations()`` 为准，再登记画面实际出现的颜色。"""
    raw = None
    try:
        with conn.io_lock:
            raw = conn.bng.get_annotations()
    except Exception:                                        # noqa: BLE001
        raw = None
    palette = annotation_palette(raw) if raw else annotation_palette(None)
    palette = {"version": palette.get("version"), "source": palette.get("source"),
               "classes": dict(palette.get("classes") or {}),
               "road": palette.get("road"), "line": palette.get("line")}
    # 实际出现的颜色也要登记，否则 UNKNOWN_CLASS 会把"未登记"和"没登记"混在一起
    flat = np.asarray(annotation)[:, :, :3].reshape(-1, 3)
    for row in (np.unique(flat, axis=0).tolist() if flat.size else []):
        color = tuple(int(v) for v in row)
        palette["classes"].setdefault(
            f"observed_{color[0]:02x}{color[1]:02x}{color[2]:02x}", list(color))
    palette["sha"] = palette_sha(palette)
    return palette


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("pure", "tech"), default="pure")
    ap.add_argument("--out", default=None, help="报告输出路径（默认 stdout）")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--frames", type=int, default=2)
    # Tech 专用（本轮不执行）
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=64256)
    ap.add_argument("--launch", action="store_true",
                    help="允许 BeamNGpy 自行启动游戏（默认只连已在跑的实例）")
    ap.add_argument("--map", default="")
    ap.add_argument("--segment", default="probe")
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--renderer", default="beamng_tech")
    ap.add_argument("--truth-json", default=None,
                    help="Tech 模式的已知 3D 真值点（没有它探针大声失败）")
    ap.add_argument("--depth-mode", choices=("ndc", "linear"), default="ndc")
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args(argv)

    if args.runtime == "tech":
        report = run_tech(args)
    else:
        report = run_pure(width=args.width, height=args.height,
                          n_frames=args.frames)
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[auto-truth] -> {args.out}")
    else:
        print(text)
    if args.runtime == "pure":
        print(f"[auto-truth] 场景 {report['n_scenes_ok']}/{report['n_scenes']} 通过，"
              f"反例 {report['n_counterexamples_rejected']}/"
              f"{report['n_counterexamples']} 被拒")
    else:
        print(f"[auto-truth] tech ok={report.get('ok')} "
              f"error={report.get('error', '')}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
