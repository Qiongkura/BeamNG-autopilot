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
import math
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


def _runtime_port(args) -> int:
    """运行时端口：与其它 Tech 入口同一来源（`config.runtime_port("tech")`）。

    实测踩到：写死 64256（BeamNGpy 默认）而本机 tech 运行在 64257，
    探针会在错误的端口上连接失败——看起来像"游戏没跑"，其实是端口来源不一致。
    """
    if getattr(args, "port", None):
        return int(args.port)
    from beamng_autopilot import config
    return int(config.runtime_port("tech"))


def _runtime_home(args) -> str:
    from beamng_autopilot import config
    return str(config.runtime_home("tech"))

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
def _vehicle_frame(state):
    """车辆世界基（前/左/上，右手，Z-up）：只用 `state.dir` + 世界上方向。"""
    fwd = np.asarray(getattr(state, "dir", None), dtype=float).ravel()[:3]
    n = float(np.linalg.norm(fwd))
    if not np.isfinite(n) or n < 1e-9:
        h = float(getattr(state, "heading", 0.0))
        fwd = np.array([math.cos(h), math.sin(h), 0.0])
    else:
        fwd = fwd / n
    up_w = np.array([0.0, 0.0, 1.0])
    left = np.cross(up_w, fwd)
    ln = float(np.linalg.norm(left))
    left = left / ln if ln > 1e-9 else np.array([0.0, 1.0, 0.0])
    up = np.cross(fwd, left)
    return fwd, left, up


def _camera_dir_arg(desired_world_dir, state):
    """把"想要的世界朝向"换成 ``Camera(dir=...)`` 该传的向量。

    实测约定（2026-09-27，BeamNG.tech v0.38.5 + BeamNGpy 1.35.1，受控探针
    `logs/.../camera_convention.json`）：``Camera(dir=r)`` 的实际世界朝向是
    ``R_vehicle · R_z(+90°) · r``——传 ``(1,0,0)`` 得到车体左向、``(0,1,0)``
    得到车体后向、``(0,-1,0)`` 才是车头前向。传参前必须反解：
    ``r = R_z(-90°) · R_vehicle⁻¹ · d_world``。

    不这么算的后果实测过：按世界方向直接传，相机朝侧后方，真值点全部投影到
    画面外/相机后（"18 个点全跳过"），看起来像标签问题。
    """
    fwd, left, up = _vehicle_frame(state)
    w = np.asarray(desired_world_dir, dtype=float).ravel()[:3]
    n = float(np.linalg.norm(w))
    if n < 1e-9:
        raise ValueError("desired camera direction must be nonzero")
    w = w / n
    local = np.array([float(w @ fwd), float(w @ left), float(w @ up)])
    # R_z(-90°)：(x, y, z) -> (y, -x, z)
    return (float(local[1]), float(-local[0]), float(local[2]))


def _calibrate_depth(depth_m: np.ndarray, camera: dict, truth_points: list,
                     *, window: int = 2) -> dict:
    """用**独立几何**标定深度缓冲：真值点的真实距离 vs 该像素的深度读数。

    为什么不用"挂载高度 + 俯仰角"推期望值：那是假设（实测两次都对不上——
    同一配置下车载朝向读到 4.00 m、道路对齐读到 1.00 m，谁也没法从假设判断
    哪个是"对的"）。真值点的世界坐标给出**真实距离**（相机位姿已知），
    读同一像素的深度值，最小二乘拟合 ``true_m = a*raw + b``：斜率≈1、截距≈0
    就是"uint8 米制"；其它系数照样能用（报告里写清），
    拟合残差大就说明缓冲与几何不一致 -> 遮挡判据不可用（大声失败）。

    返回 ``{"n", "a", "b", "residual_m", "mode", "samples", "why"}``。
    """
    import math as _math

    from beamng_autopilot.experiments.auto_truth import (_point_depth,
                                                         project_point)
    cam = camera or {}
    pose = cam.get("pose") or {}
    pos = np.asarray(pose.get("pos") or [np.nan] * 3, dtype=float)
    h, w = np.asarray(depth_m).shape[:2]
    xs: list[float] = []
    ys: list[float] = []
    samples: list[dict] = []
    for p in truth_points or []:
        try:
            u, v = project_point(p.get("world"), cam)
            d_true = float(_point_depth(p.get("world"), cam))
        except (TypeError, ValueError):
            continue
        if not (_math.isfinite(u) and _math.isfinite(v) and d_true > 0.5):
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            continue
        patch = np.asarray(depth_m)[max(0, y - window):y + window + 1,
                                    max(0, x - window):x + window + 1]
        raw = float(np.median(patch))
        if raw <= 0:
            continue
        xs.append(raw)
        ys.append(d_true)
        samples.append({"class": p.get("class"), "raw": round(raw, 2),
                        "true_m": round(d_true, 2),
                        "ratio": round(d_true / raw, 3) if raw else None})
    out = {"n": len(xs), "a": None, "b": None, "residual_m": None,
           "mode": "uncalibrated", "samples": samples[:8],
           "why": "fewer than 3 usable truth points for a fit"}
    if len(xs) < 3:
        return out
    A = np.vstack([np.asarray(xs, dtype=float), np.ones(len(xs))]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(ys, dtype=float), rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * np.asarray(xs, dtype=float) + b
    res = np.abs(pred - np.asarray(ys, dtype=float))
    out.update(a=a, b=b,
               residual_m={"mean": round(float(res.mean()), 3),
                           "max": round(float(res.max()), 3)},
               mode=("uint8_meters" if abs(a - 1.0) <= 0.15 and abs(b) <= 1.0
                     else "custom_linear"),
               why="")
    return out


def _buffers_ready(data) -> tuple[bool, str]:
    """一帧缓冲是否已渲染（预热判据）：RGB/annotation/depth 都不能是全 0。

    实测（2026-09-27，italy，BeamNG.tech v0.38.5）：刚挂相机时的共享内存缓冲
    全 0；有效帧的 depth 是 **uint8 米制距离**（下缘中央 ~4 m、整帧中位 ~27 m、
    远端饱和 255），且必须 ``is_using_shared_memory=True``（非共享配置实测
    50% 像素为 0）。全 0 缓冲按 NDC 反解会得到 near=0.05 m，看着像"深度标定
    错了"——所以预热判据先查"有没有渲染过"，再谈换算。
    """
    for ch in ("colour", "annotation", "depth"):
        v = data.get(ch)
        if v is None:
            return False, f"{ch} missing"
        a = np.asarray(v)
        if a.size == 0:
            return False, f"{ch} empty"
        if float(np.nanmax(a)) <= 0.0:
            return False, f"{ch} is all zeros (not rendered yet)"
    return True, "buffers rendered"


def _depth_to_meters(raw, *, near: float, far: float, mode: str) -> np.ndarray:
    """把 BeamNGpy 的深度缓冲转成沿光轴深度（米）。

    ``mode="ndc"``（默认）按 OpenGL 非线性深度反解；``mode="linear"`` 按
    ``near + d*(far-near)``。**这个换算必须由实机自检确认**（见
    `_tech_depth_sanity`）：不确定就大声失败，不把错的深度喂给遮挡判据。

    实机实测（2026-09-27，italy）：共享内存路径直接给 **uint8 米制距离**
    （下缘中央 ~4 m 与相机几何一致），走下面的"已经是米"分支原样返回；
    全 0 的未渲染缓冲会被预热拦下（见 `_buffers_ready`），不会到这里。
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


def _load_truth_blob(path: str | None) -> dict:
    """真值 JSON 的完整内容（含生成时的自车位姿）；读不到返回空 dict。"""
    if not path:
        return {}
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return blob if isinstance(blob, dict) else {}


def _pose_mismatch(truth_blob: dict, conn) -> str | None:
    """采帧位姿 vs 生成真值位姿；不一致返回原因（调用方非 0 退出）。

    为什么必须查：真值点是**生成时**自车位姿下的几何。车被挪过、换了 spawn、
    或连到了另一个会话时，投影对不上会伪装成"标签错误"——那是假发现。
    """
    ego = (truth_blob or {}).get("ego_pose")
    if not isinstance(ego, dict) or not ego.get("pos"):
        return None                      # 生成侧没记位姿：不阻断，报告里注明
    try:
        st = conn.get_state()
    except Exception as exc:                              # noqa: BLE001
        return f"读不到当前位姿（{type(exc).__name__}: {exc}）"
    p0 = np.asarray(ego["pos"], dtype=float)[:2]
    p1 = np.asarray(st.pos, dtype=float)[:2]
    d = float(np.hypot(*(p1 - p0)))
    if d > 2.0:
        return (f"车辆位置与生成真值时相差 {d:.1f} m（>2 m）：真值几何属于"
                f"另一个位姿，先回到生成位置或重新生成真值")
    h0 = float(ego.get("heading_rad") or 0.0)
    h1 = float(getattr(st, "heading", 0.0))
    dh = abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
    if np.degrees(dh) > 5.0:
        return (f"车辆朝向与生成真值时相差 {np.degrees(dh):.1f}°（>5°）："
                f"重新生成真值再跑探针")
    return None


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
    conn = BeamNGConnector(host=args.host, port=_runtime_port(args),
                           home=_runtime_home(args))
    cam = None
    try:
        check_graphics_quality(conn.user_dir)
        conn.open(launch=bool(args.launch))
        # 车辆：附到运行中会话的车（探针不控车，只读状态 + 挂传感器）。
        # 没有可附着的车才加载场景；那时下面的位姿检查会拦住"用错几何的真值"。
        if getattr(conn, "vehicle", None) is None:
            try:
                conn.attach_vehicle(already_open=True)
            except Exception:                                # noqa: BLE001
                conn.load_scenario()
        # 采帧位姿必须与生成真值时的位姿一致（否则投影对不上是假发现）
        truth_blob = _load_truth_blob(args.truth_json)
        _pm = _pose_mismatch(truth_blob, conn)
        if _pm:
            return {"runtime": "tech", "ok": False,
                    "error": f"位姿一致性检查失败：{_pm}",
                    "hint": "真值点由 scripts/m5_tech_truth_points.py 在生成时"
                            "的位姿下采样；回到那个位姿或重新生成"}
        from beamngpy.sensors import Camera
        width, height = int(args.width), int(args.height)
        near, far = 0.05, 150.0
        # 测试台架朝向：真值 JSON 带 road_dir 时**按道路方向**定向（不是按车头）。
        # 实测踩到两件事：(a) 车停的朝向可能与道路差 45°；(b) Camera(dir=r) 的实际
        # 世界朝向是 R_v·R_z(+90°)·r（见 `_camera_dir_arg`）——直接传世界方向会
        # 朝侧后方。这里两件都处理，读回后还会校验（偏差 >10° 大声失败）。
        _rd = truth_blob.get("road_dir")
        cam_dir, cam_up, frame_mode = CAMERA_DIR, CAMERA_UP, "vehicle_mount"
        want_world_dir = None
        if (isinstance(_rd, (list, tuple)) and len(_rd) >= 2
                and float(np.linalg.norm(np.asarray(_rd, dtype=float)[:2])) > 1e-6):
            want_world_dir = np.asarray(_rd, dtype=float).ravel()[:3]
            cam_dir = _camera_dir_arg(want_world_dir, conn.get_state())
            cam_up = (0.0, 0.0, 1.0)
            frame_mode = "road_aligned"
        name = f"m5_auto_truth_probe_{abs(hash(str(conn.user_dir))) % 10000}"
        with conn.io_lock:
            cam = Camera(name, conn.bng, conn.vehicle, requested_update_time=0.05,
                         pos=CAMERA_POS, dir=cam_dir, up=cam_up,
                         resolution=(width, height),
                         field_of_view_y=CAMERA_FOV_DEG,
                         near_far_planes=(near, far),
                         is_using_shared_memory=True,
                         is_render_colours=True, is_render_annotations=True,
                         is_render_depth=True, is_visualised=False,
                         integer_depth=False, postprocess_depth=False)
        frames: list[dict] = []
        palette = None
        depth_raw_stats: dict = {}
        depth_calib: dict = {}
        # 预热：相机刚挂上时共享内存缓冲还没渲染，第一次 poll 会拿到全 0 的
        # 深度/黑图（实测：全 0 缓冲按 NDC 反解出 near=0.05 m，看着像"深度标定
        # 错了"，其实是**异步旧帧**）。这里先推进仿真并丢弃无效帧，有界重试。
        warm: list[str] = []
        for _try in range(6):
            with conn.io_lock:
                conn.bng.control.step(max(1, int(args.step)))
            with conn.io_lock:
                data = cam.poll()
            ok_w, why_w = _buffers_ready(data)
            warm.append(why_w)
            if ok_w:
                break
        else:
            return {"runtime": "tech", "ok": False,
                    "error": "相机缓冲预热失败（6 次重试后仍无有效帧）："
                             + "; ".join(warm[-3:]),
                    "hint": "深度/彩色缓冲未渲染：检查共享内存与 GPU prepass；"
                            "不要把全 0 缓冲当标定错误"}
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
            # 相机位姿：直接问传感器（引擎给的世界位姿），不猜车辆坐标约定。
            try:
                cam_pos = [float(v) for v in cam.get_position()]
                fwd = np.asarray(cam.get_direction(), dtype=float)
            except Exception as exc:                          # noqa: BLE001
                return {"runtime": "tech", "ok": False,
                        "error": f"读不到相机世界位姿（{type(exc).__name__}: {exc}）",
                        "hint": "BeamNGpy 需要传感器已 attach 且场景已加载"}
            fwd = fwd / max(1e-12, float(np.linalg.norm(fwd)))
            # 读回校验：道路对齐时，相机实际世界朝向必须≈road_dir（否则几何
            # 对不上会伪装成标签错误——实测就是在这里抓到 +90° 约定差）。
            if want_world_dir is not None:
                _w = want_world_dir / max(1e-12, float(
                    np.linalg.norm(want_world_dir)))
                _cos = float(np.clip(fwd @ _w, -1.0, 1.0))
                _ang = math.degrees(math.acos(_cos))
                if _ang > 10.0:
                    return {"runtime": "tech", "ok": False,
                            "error": f"相机朝向读回校验失败：请求道路方向 "
                                     f"{[round(float(v), 3) for v in _w]}，"
                                     f"实际 {[round(float(v), 3) for v in fwd]}"
                                     f"（差 {_ang:.1f}° >10°）",
                            "hint": "Camera(dir=) 的约定可能又变了：用 "
                                    "logs/.../camera_convention_probe.py 重测约定"}
                checks.append(f"camera direction matches road_dir "
                              f"(off by {_ang:.1f}°)")
            up_world = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up_world)
            right = right / max(1e-12, float(np.linalg.norm(right)))
            up = np.cross(right, fwd)
            cam_dict = {"name": "front_main", "width": width,
                        "height": height, "fov_y_deg": float(CAMERA_FOV_DEG),
                        "pose": {"pos": cam_pos,
                                 "basis": {"right": [float(v) for v in right],
                                           "fwd": [float(v) for v in fwd],
                                           "up": [float(v) for v in up]}}}
            if palette is None:
                palette = _tech_palette(conn, annotation)
            label = to_label(annotation, road_colors=palette["road"],
                             line_colors=palette["line"])
            label = np.ascontiguousarray(label, dtype=np.uint8)
            fid = f"tech_{i:05d}"
            src_sha, lab_sha = frame_content_shas(rgb, label)
            # 落盘原始帧（在深度/标定门槛**之前**）：一次采集、离线迭代校验与
            # 标定，避免为每个判据反复占用游戏（方案 §4.4：小批次可重复）。
            if getattr(args, "dump_dir", None):
                _dd = Path(args.dump_dir)
                _dd.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    _dd / f"frame_{i:05d}.npz", rgb=rgb, annotation=annotation,
                    depth_raw=np.asarray(raw_depth), label=label,
                    camera_json=json.dumps(cam_dict),
                    truth_json=json.dumps(truth_points),
                    palette_json=json.dumps(palette or {}))
            # 深度标定优先：用**真值点的独立几何**拟合"读数 -> 米"，
            # 而不是猜挂载高度/俯仰角（实测两种朝向给出 4.00/1.00 m 两个读数，
            # 假设法分不出谁对）。拟合不可用（<3 点）才退回挂载假设自检。
            calib = _calibrate_depth(np.asarray(raw_depth), cam_dict,
                                     truth_points)
            if calib["n"] >= 3 and calib["mode"] != "uncalibrated":
                if calib["residual_m"]["mean"] > 1.0:
                    return {"runtime": "tech", "ok": False,
                            "error": "深度标定残差过大（真值点几何 vs 深度读数"
                                     f"不一致：{calib['residual_m']}）——遮挡"
                                     "判据不可用，先查渲染配置",
                            "depth_calibration": calib}
                depth_m = (float(calib["a"]) * np.asarray(raw_depth, dtype=float)
                           + float(calib["b"]))
                checks.append(
                    f"depth calibrated from truth points: mode={calib['mode']} "
                    f"a={calib['a']:.3f} b={calib['b']:.3f} "
                    f"residual_mean={calib['residual_m']['mean']:.3f} m "
                    f"(n={calib['n']})")
            else:
                depth_m = _depth_to_meters(raw_depth, near=near, far=far,
                                           mode=args.depth_mode)
                ok, why = _tech_depth_sanity(depth_m)
                if not ok:
                    return {"runtime": "tech", "ok": False,
                            "error": f"深度换算自检失败（且真值点不足以标定）：{why}",
                            "hint": "需要 ≥3 个可见真值点来自标定，或先用 "
                                    "--depth-mode linear 试；都不行必须重新标定"
                                    "后再用遮挡判据",
                            "depth_calibration": calib}
                checks.append(why)
            if not depth_raw_stats:
                _rd = np.asarray(raw_depth)
                depth_raw_stats = {
                    "dtype": str(_rd.dtype), "min": float(np.nanmin(_rd)),
                    "median": float(np.nanmedian(_rd)),
                    "max": float(np.nanmax(_rd)),
                    "lower_center_median_m": float(np.nanmedian(
                        np.asarray(depth_m)[int(height * 0.75):,
                                            int(width * 0.4):int(width * 0.6)]))}
            if not depth_calib:
                depth_calib = calib
            frames.append({
                "frame_id": fid, "timestamp": float(i) * 0.05,
                "channel_ids": {"rgb": fid, "annotation": fid, "depth": fid},
                "camera": cam_dict,
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
        # 逐点诊断（为什么 checked=0 要能直接看出来）：投影坐标、是否在画面内、
        # 该像素的标签值/深度。`skipped` 的三种原因（相机后/画面外/类别非法）
        # 只有在这里才分得清——没有它，"18 个点全跳过"无法归因。
        point_diag: list[dict] = []
        try:
            from beamng_autopilot.experiments.auto_truth import project_point
            _f = frames[0]
            _cam = _f["camera"]
            _lab = np.asarray(_f["label"])
            _dep = np.asarray(_f["depth"])
            _h, _w = _lab.shape[:2]
            for p in (_f.get("truth_points") or [])[:12]:
                u, v = project_point(p.get("world"), _cam)
                row = {"class": p.get("class"), "role": p.get("role"),
                       "u": None if not np.isfinite(u) else round(float(u), 1),
                       "v": None if not np.isfinite(v) else round(float(v), 1),
                       "behind_or_invalid": bool(not (np.isfinite(u)
                                                      and np.isfinite(v)))}
                if not row["behind_or_invalid"]:
                    x, y = int(round(u)), int(round(v))
                    row["in_frame"] = bool(0 <= x < _w and 0 <= y < _h)
                    if row["in_frame"]:
                        row["label"] = int(_lab[y, x])
                        row["depth_m"] = round(float(_dep[y, x]), 2)
                point_diag.append(row)
        except Exception as exc:                              # noqa: BLE001
            point_diag.append({"error": f"{type(exc).__name__}: {exc}"})
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
                "camera_frame": frame_mode,
                "truth_points": len(truth_points),
                "depth_checks": checks,
                "depth_raw": depth_raw_stats,
                "depth_calibration": depth_calib,
                "measured_channels": measured,
                "projection_checked": checked,
                "verifier_version": VERIFIER_VERSION,
                "truth_contract": TRUTH_CONTRACT,
                "rejections": report["rejections"],
                "point_diag": point_diag,
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
    ap.add_argument("--port", type=int, default=None,
                    help="覆盖运行时端口（默认取 config.runtime_port('tech')）")
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
    ap.add_argument("--dump-dir", default=None,
                    help="把原始帧（rgb/annotation/depth_raw/label/camera/真值点）"
                         "落盘到该目录：一次采集后可离线迭代校验与标定")
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
