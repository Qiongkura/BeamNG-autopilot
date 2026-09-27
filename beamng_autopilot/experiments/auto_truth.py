"""自动真值生产与校验库（T16 P1 核心；纯 numpy，无游戏依赖）。

为什么存在（方案 `docs/T16_AUTONOMOUS_LEARNING_ROADMAP_20260927.md` §4）：
"免人工标注"的前提是**自动产出的真值本身可被证伪**。本模块把"一批模拟器
真值是否合法"写成纯函数：帧身份、独立投影、调色板、标签 hash、遮挡、
相机翻转、逐通道资格，以及把结论写进 sidecar（凭证契约 v1，见冻结文档
`docs/T16_ORDER0_FREEZE_20260927.md` §3.4）。

纪律（与冻结文档一致）：

* 不确定一律记 ``unknown`` / ``unknown_px``，**绝不当通过**（UNKNOWN≠PASS）；
* 无线负例只能来自"生成器声明未生成漆线且无内嵌漆线纹理"，**不得**用
  "模型没预测线"或空标签证明无线；
* ``pseudo`` 是弱监督，永不作为真值（rank 表沿用 ``experiments/labels.py``）；
* ``ok=False`` 时任何调用方不得写出 ``label_source="engine_verified"``。

数据契约：

* batch = ``{"frames": [frame, ...], "scene": {...}, "palette": {...}}``；
* frame 至少含 ``frame_id`` / ``timestamp`` / ``camera`` / ``rgb`` / ``annotation``
  / ``depth`` / ``label`` / ``label_sha`` / ``source_image_sha``；可选
  ``truth_points``（生成器给出的已知 3D 点，``class ∈ {1,2}``，
  ``role ∈ {"left","right",None}``）。``rgb`` 是 HxWx3 uint8 视觉图，
  ``annotation`` 是 HxWx3 uint8 调色板图，``depth`` 是 HxW float（米），
  ``label`` 是 HxW uint8 ∈ {0,1,2,255}；
* camera = ``{"name","width","height","fov_y_deg" 或 3x3 "K",
  "pose": {"pos":[x,y,z], "rot":[roll,pitch,yaw] 或 4 元四元数}}``。
  世界系 Z 向上，``rot`` 为弧度；相机基 = (right, fwd, up)，与 BeamNG 车辆
  朝向约定一致（heading 0 → fwd=+X、right=-Y）。**本模块自己实现针孔投影，
  不复用 ``vision/projection.py``**——校验的意义就是另一套独立实现；
* ``depth`` 约定 = **沿光轴深度（米）**（不是欧氏射线距离）。若数据源给的是
  射线距离，把 batch/frame 的 ``depth_convention`` 设成 ``"ray"``，本模块按
  像素射线角换算；0 或负值 = 无效（未命中任何面）。

本模块不做 I/O（除 `write_truth_credentials` 写 sidecar），不 import beamngpy。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

# 复用既有实现，不另写第二套：annotation->label 转换器只有 annotations.py 一个。
from beamng_autopilot.experiments.labels import (  # noqa: F401  (公开复用点)
    IGNORE, PAINT_SOURCE_RANK, label_sha16,
)
from beamng_autopilot_tech.annotations import to_label

# ── 契约与拒绝码（冻结文档 §3.4；调用方按常量比对，不要写字面量） ──────────
TRUTH_CONTRACT = "v1"
VERIFIER_VERSION = "auto_truth/1"

CAMERA_FLIP = "CAMERA_FLIP"
RESOLUTION_MISMATCH = "RESOLUTION_MISMATCH"
TIME_MISMATCH = "TIME_MISMATCH"
ASYNC_STALE_FRAME = "ASYNC_STALE_FRAME"
OCCLUSION_INSERT = "OCCLUSION_INSERT"
LABEL_TAMPERED = "LABEL_TAMPERED"
BLACK_FRAME = "BLACK_FRAME"
PALETTE_CHANGED = "PALETTE_CHANGED"
UNKNOWN_CLASS = "UNKNOWN_CLASS"
PROJECTION_MISMATCH = "PROJECTION_MISMATCH"

#: 全部拒绝码（报告/测试用来防漏接）。
REJECTION_CODES = (CAMERA_FLIP, RESOLUTION_MISMATCH, TIME_MISMATCH,
                   ASYNC_STALE_FRAME, OCCLUSION_INSERT, LABEL_TAMPERED,
                   BLACK_FRAME, PALETTE_CHANGED, UNKNOWN_CLASS,
                   PROJECTION_MISMATCH)

CHANNELS = ("LINE", "ROAD", "SHOULDER", "ROLE")
CHANNEL_STATUSES = ("measured", "not_applicable", "unknown", "unverified_labels")

#: 默认容差：真值点投影半径（像素）、遮挡判定余量（米）、道路像素下限。
PROJECTION_RADIUS_PX = 2
OCCLUSION_MARGIN_M = 0.5
ROAD_MIN_PX = 200
#: 逐帧标签取值域；出现其它值 = 篡改。
LABEL_VALUES = (0, 1, 2, 255)


# ── 基础工具 ────────────────────────────────────────────────────────────────
def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _sha16_text(text: str) -> str:
    return _sha16(text.encode("utf-8"))


def _vec3(value, what: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be three numbers") from exc
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} must be three finite numbers, got {value!r}")
    return arr


def _dilate(mask: np.ndarray, rounds: int = 1) -> np.ndarray:
    """4 邻域膨胀；用于覆盖亚像素漆线与投影残差（生成器与校验器都不猜中心）。"""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(rounds)):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


# ── 相机与独立针孔投影 ──────────────────────────────────────────────────────
def basis_from_euler(roll: float, pitch: float, yaw: float):
    """Euler(弧度) -> 世界系相机基 (right, fwd, up)。

    约定：先绕世界 Z 偏航（heading 0 = +X），再绕相机 right 俯仰（正 = 抬头），
    最后绕相机 fwd 滚转（正 = 右倾）。基础三轴与仓库车辆约定一致：
    fwd0=[cos yaw, sin yaw, 0]，right0=[sin yaw, -cos yaw, 0]，up0=[0,0,1]
    （right0 × fwd0 = up0，右手系）。
    """
    c_y, s_y = math.cos(float(yaw)), math.sin(float(yaw))
    c_p, s_p = math.cos(float(pitch)), math.sin(float(pitch))
    c_r, s_r = math.cos(float(roll)), math.sin(float(roll))
    fwd0 = np.array([c_y, s_y, 0.0])
    right0 = np.array([s_y, -c_y, 0.0])
    up0 = np.array([0.0, 0.0, 1.0])
    fwd = c_p * fwd0 + s_p * up0
    up_t = -s_p * fwd0 + c_p * up0
    right = c_r * right0 + s_r * up_t
    up = -s_r * right0 + c_r * up_t
    return right, fwd, up


def basis_from_quat(q):
    """四元数 (x, y, z, w) -> 世界系相机基。

    约定：四元数把标准基 (X=right, Y=fwd, Z=up) 旋到世界系（列向量即世界
    轴）。BeamNG 车辆四元数使用逆/共轭约定且 local Y 指向车后，转成本模块
    基时需要 ``diag(1,-1,1)`` 与 ``-w``（与 ``vision/projection.py`` 的实测
    结论一致）；Tech 探针既能走这条路，也能直接用传感器给出的 dir/up。
    """
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:4])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]))
    x, y, z, w = x / n, y / n, z / n, w / n
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    right = rot @ np.array([1.0, 0.0, 0.0])
    fwd = rot @ np.array([0.0, 1.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])
    return right, fwd, up


def euler_from_basis(right, fwd):
    """世界系相机基 -> ``rot=[roll,pitch,yaw]``（弧度）。生成器/Tech 采集用。"""
    r = _vec3(right, "camera right")
    f = _vec3(fwd, "camera fwd")
    f = f / max(1e-12, float(np.linalg.norm(f)))
    yaw = math.atan2(float(f[1]), float(f[0]))
    pitch = math.asin(float(np.clip(f[2], -1.0, 1.0)))
    fwd0 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up0 = np.array([0.0, 0.0, 1.0])
    right0 = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up_t = -math.sin(pitch) * fwd0 + math.cos(pitch) * up0
    roll = math.atan2(float(r @ up_t), float(r @ right0))
    return roll, pitch, yaw


def camera_basis(camera):
    """世界系相机基 ``(C, right, fwd, up)``；``pose`` 缺失时的缺省 = 原点直视 +X。"""
    if not isinstance(camera, Mapping):
        raise ValueError("camera must be a mapping")
    pose = camera.get("pose")
    pose = pose if isinstance(pose, Mapping) else {}
    pos = _vec3(pose.get("pos", (0.0, 0.0, 0.0)), "camera pose.pos")
    basis = pose.get("basis")
    if isinstance(basis, Mapping):
        # 采集侧已经拿到世界系 dir/up 时直接用（Tech 传感器路径）；仍做正交化。
        fwd = _vec3(basis.get("fwd"), "camera pose.basis.fwd")
        right = _vec3(basis.get("right"), "camera pose.basis.right")
        fwd = fwd / float(np.linalg.norm(fwd))
        right = right - float(right @ fwd) * fwd
        right = right / max(1e-12, float(np.linalg.norm(right)))
        up = np.cross(right, fwd)
        return pos, right, fwd, up
    rot = pose.get("rot", (0.0, 0.0, 0.0))
    arr = np.asarray(rot, dtype=float).reshape(-1)
    if arr.size == 4:
        right, fwd, up = basis_from_quat(arr)
    elif arr.size == 3:
        right, fwd, up = basis_from_euler(arr[0], arr[1], arr[2])
    else:
        raise ValueError("camera pose.rot must have 3 (roll,pitch,yaw) or 4 "
                         "(x,y,z,w) numbers")
    return pos, right, fwd, up


def camera_intrinsics(camera):
    """``(fx, fy, cx, cy)``；显式 3x3 K 优先，否则由 width/height + fov_y_deg 推。"""
    if not isinstance(camera, Mapping):
        raise ValueError("camera must be a mapping")
    k = camera.get("K")
    if k is not None:
        k = np.asarray(k, dtype=float)
        if k.shape != (3, 3):
            raise ValueError("camera K must be 3x3")
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        if fx <= 0 or fy <= 0:
            raise ValueError("camera K must have positive focal lengths")
        return fx, fy, cx, cy
    width = float(camera.get("width") or 0)
    height = float(camera.get("height") or 0)
    fov = camera.get("fov_y_deg")
    if width <= 0 or height <= 0:
        raise ValueError("camera needs width/height (or a 3x3 K)")
    if fov is None:
        raise ValueError("camera needs fov_y_deg (or a 3x3 K)")
    fov = float(fov)
    if not 0.0 < fov < 180.0:
        raise ValueError("camera fov_y_deg must be in (0, 180)")
    # 方形像素：fx = fy = (H/2)/tan(fov/2)，主点 = 图像中心（与仓库标定一致）。
    fy = (height / 2.0) / math.tan(math.radians(fov) / 2.0)
    return fy, fy, width / 2.0, height / 2.0


def project_point(world, camera) -> tuple[float, float]:
    """世界点 -> 像素 ``(u, v)``；在相机后方返回 ``(nan, nan)``（不猜）。"""
    c, right, fwd, _up = camera_basis(camera)
    fx, fy, cx, cy = camera_intrinsics(camera)
    p = _vec3(world, "truth point world")
    d = p - c
    depth = float(d @ fwd)
    if depth <= 1e-6:
        return float("nan"), float("nan")
    u = cx + fx * float(d @ right) / depth
    v = cy - fy * float(d @ _up) / depth
    return u, v


def _point_depth(world, camera) -> float:
    c, _right, fwd, _up = camera_basis(camera)
    return float((_vec3(world, "truth point world") - c) @ fwd)


def _nearer_surface(depth: np.ndarray, x: int, y: int, d_line: float,
                    margin: float = OCCLUSION_MARGIN_M,
                    radius: int = 1):
    """该像素附近是否有**明确更近**的面（遮挡证据）。

    为什么用窗口最大值而不是单像素：地面深度是行的陡函数——靠近地平线处一行
    就相差数米（30 m 处约 5 m/行），投影点四舍五入到相邻行会凭空得到"更近"的
    深度。遮挡物是**成片**的：只有窗口里最远的有效深度仍然明显近于漆线，才算
    有遮挡证据；边界像素因此保守放过（不判码、记 unknown）。
    返回 ``None`` = 窗口内没有有效深度（不可判）。
    """
    h, w = depth.shape[:2]
    x0, x1 = max(0, x - int(radius)), min(w, x + int(radius) + 1)
    y0, y1 = max(0, y - int(radius)), min(h, y + int(radius) + 1)
    window = np.asarray(depth[y0:y1, x0:x1], dtype=float)
    valid = window[np.isfinite(window) & (window > 0)]
    if valid.size == 0:
        return None
    return bool(float(valid.max()) < float(d_line) - float(margin))


def _axis_depth(frame) -> np.ndarray | None:
    """把 ``depth`` 归一到沿光轴深度；无效（<=0）原样保留。"""
    raw = frame.get("depth")
    if raw is None:
        return None
    depth = np.asarray(raw, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return None
    conv = str(frame.get("depth_convention") or "axis").lower()
    if conv not in ("ray", "euclidean", "ray_distance"):
        return depth
    camera = frame.get("camera")
    if not isinstance(camera, Mapping):
        return depth
    try:
        _c, _r, _f, _u = camera_basis(camera)
        fx, fy, cx, cy = camera_intrinsics(camera)
    except ValueError:
        return depth
    h, w = depth.shape[:2]
    u = np.arange(w, dtype=float)[None, :]
    v = np.arange(h, dtype=float)[:, None]
    xr = (u - cx) / fx
    yr = -(v - cy) / fy
    # cos(射线与光轴夹角) = 1/sqrt(1+xr^2+yr^2)；轴深 = 射线距离 * cos。
    cos_theta = 1.0 / np.sqrt(1.0 + xr * xr + yr * yr)
    return depth * cos_theta


def _truth_points(frame) -> list[dict]:
    pts = frame.get("truth_points")
    return [p for p in pts if isinstance(p, Mapping)] if isinstance(pts, list) else []


def _line_truth_points(frame) -> list[dict]:
    return [p for p in _truth_points(frame) if int(p.get("class", 0)) == 2]


# ── 1. 投影校验 ─────────────────────────────────────────────────────────────
def verify_projection(frame, *, radius_px: int = PROJECTION_RADIUS_PX) -> dict:
    """逐真值点核对投影位置半径内的像素类别（线点=2、路面点=1）。

    跳过规则（保守，只跳"本来就看不清"的点，绝不放过矛盾）：

    * 投影在画面外/相机后方 → ``skipped``；
    * 命中像素是 ignore(255)：标签明确说"这一块不判"，不算 mismatch；
    * 该像素的实测深度明显近于真值点深度 → 点被遮挡，不算 mismatch。

    ``ok`` 要求 ``checked>0``：没有可核对的点 = ``status="unknown"``，
    调用方不得当通过（UNKNOWN≠PASS）。
    """
    camera = frame.get("camera")
    label = frame.get("label")
    pts = _truth_points(frame)
    out = {"ok": False, "checked": 0, "mismatches": 0, "residual_px": None,
           "status": "unknown", "skipped": 0, "ignored": 0, "occluded": 0,
           "unknown_depth": 0, "why": "", "details": []}
    if not pts:
        out["why"] = "frame declares no truth_points: projection unverifiable"
        return out
    if camera is None or label is None:
        out["why"] = "frame has neither camera nor label"
        return out
    lab = np.asarray(label)
    if lab.ndim != 2 or lab.size == 0:
        out["why"] = f"label must be a nonempty 2-D array, got {lab.shape}"
        return out
    h, w = lab.shape[:2]
    depth = _axis_depth(frame)
    if depth is not None and depth.shape != lab.shape:
        depth = None  # 深度分辨率对不上时不做遮挡跳过（由分辨率检查拒帧）
    residuals: list[float] = []
    details: list[dict] = []
    for i, p in enumerate(pts):
        cls = int(p.get("class", 0))
        if cls not in (1, 2):
            out["skipped"] += 1
            continue
        try:
            u, v = project_point(p.get("world"), camera)
            d_pt = _point_depth(p.get("world"), camera)
        except (TypeError, ValueError):
            out["skipped"] += 1
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            out["skipped"] += 1
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            out["skipped"] += 1
            continue
        if int(lab[y, x]) == IGNORE:
            out["ignored"] += 1
            continue
        if depth is not None:
            nearer = _nearer_surface(depth, x, y, d_pt)
            if nearer is None:
                out["unknown_depth"] += 1
            elif nearer:
                out["occluded"] += 1
                continue
        x0, x1 = max(0, x - int(radius_px)), min(w, x + int(radius_px) + 1)
        y0, y1 = max(0, y - int(radius_px)), min(h, y + int(radius_px) + 1)
        window = lab[y0:y1, x0:x1]
        ys, xs = np.nonzero(window == cls)
        out["checked"] += 1
        if len(ys):
            gy, gx = y0 + ys, x0 + xs
            dist = np.hypot(gy - v, gx - u)
            residuals.append(float(dist.min()))
        else:
            out["mismatches"] += 1
            if len(details) < 5:
                seen = sorted(int(t) for t in np.unique(window))
                details.append({"i": i, "class": cls, "u": round(u, 2),
                                "v": round(v, 2), "label_window": seen,
                                "role": p.get("role")})
    out["details"] = details
    if residuals:
        arr = np.asarray(residuals, dtype=float)
        out["residual_px"] = {"n": int(arr.size),
                              "mean": round(float(arr.mean()), 3),
                              "p95": round(float(np.percentile(arr, 95)), 3),
                              "max": round(float(arr.max()), 3)}
    out["ok"] = bool(out["checked"] > 0 and out["mismatches"] == 0)
    if out["checked"] == 0:
        out["status"] = "unknown"
        out["why"] = ("no truth point could be compared (all skipped/ignored/"
                      "occluded): projection unknown, not verified")
    elif out["mismatches"]:
        out["status"] = "mismatch"
        out["why"] = (f"{out['mismatches']}/{out['checked']} truth points have no "
                      f"declared class within {int(radius_px)} px")
    else:
        out["status"] = "measured"
        out["why"] = f"{out['checked']} truth points matched within {int(radius_px)} px"
    return out


# ── 2. 帧身份 ───────────────────────────────────────────────────────────────
def frame_identity_ok(frame, *, prev_timestamp=None) -> list[str]:
    """帧身份拒绝码（空 = 通过）。

    * RGB/annotation/depth 各自的 ``frame_id`` 不一致 → ``ASYNC_STALE_FRAME``
      （异步采集拿到的旧帧：三路不是同一次曝光）；
    * 时间戳缺失或相对上一帧倒退 → ``TIME_MISMATCH``（同曝光多视角共享
      时间戳是合法的，因此只拒绝倒退，不要求严格递增）；
    * 三路分辨率互不相同，或与相机声明的 width/height 不符 →
      ``RESOLUTION_MISMATCH``（尺寸变化；用 depth 单路缩放即可触发）。
    """
    codes: list[str] = []
    ids: dict = {}
    for key in ("channel_ids", "frame_ids", "channel_frames"):
        val = frame.get(key)
        if isinstance(val, Mapping):
            ids.update({str(k): val[k] for k in val})
    for name in ("rgb", "annotation", "depth", "colour"):
        val = frame.get(f"{name}_frame_id")
        if val is not None:
            ids.setdefault(name, val)
    fid = frame.get("frame_id")
    if fid is None and not ids:
        codes.append(ASYNC_STALE_FRAME)
    else:
        distinct = {str(v) for v in ids.values() if v is not None}
        if len(distinct) > 1:
            codes.append(ASYNC_STALE_FRAME)
        if fid is not None and distinct and str(fid) not in distinct:
            codes.append(ASYNC_STALE_FRAME)

    ts = frame.get("timestamp")
    if ts is None:
        codes.append(TIME_MISMATCH)
    else:
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            codes.append(TIME_MISMATCH)
            ts = None
        if ts is not None and prev_timestamp is not None:
            try:
                if ts < float(prev_timestamp):
                    codes.append(TIME_MISMATCH)
            except (TypeError, ValueError):
                codes.append(TIME_MISMATCH)

    shapes = {}
    for name in ("rgb", "annotation", "depth"):
        arr = frame.get(name)
        if arr is None:
            shapes[name] = None
            continue
        a = np.asarray(arr)
        if a.ndim < 2 or a.size == 0:
            shapes[name] = None
            continue
        shapes[name] = (int(a.shape[0]), int(a.shape[1]))
    present = [s for s in shapes.values() if s is not None]
    if len(present) != 3 or len(set(present)) > 1:
        codes.append(RESOLUTION_MISMATCH)
    cam = frame.get("camera")
    if isinstance(cam, Mapping) and shapes.get("rgb") is not None:
        cw, ch = cam.get("width"), cam.get("height")
        if cw and ch and (int(ch), int(cw)) != shapes["rgb"]:
            codes.append(RESOLUTION_MISMATCH)
    return codes


# ── 3. 黑图 ─────────────────────────────────────────────────────────────────
def detect_black_frame(rgb, *, min_mean: float = 2.0) -> bool:
    """均值 < ``min_mean`` 的 RGB（含缺失/空数组）= 黑图/陈旧共享内存读。"""
    if rgb is None:
        return True
    arr = np.asarray(rgb)
    if arr.size == 0:
        return True
    if not np.issubdtype(arr.dtype, np.number):
        return True
    return float(np.mean(arr, dtype=np.float64)) < float(min_mean)


# ── 4. 调色板 ───────────────────────────────────────────────────────────────
def palette_classes(palette) -> dict:
    """归一化 ``{类名: (颜色三元组, ...)}``；兼容 annotation_palette 的分类形态。"""
    if not isinstance(palette, Mapping):
        return {}
    out: dict = {}
    raw = palette.get("classes")
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            colors = _normalize_colors(value)
            if colors:
                out[str(name)] = colors
    for name in ("road", "line", "background", "shoulder", "occluder", "sky"):
        if name in palette and name not in out:
            colors = _normalize_colors(palette.get(name))
            if colors:
                out[name] = colors
    return out


def _normalize_colors(value):
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and all(isinstance(v, (int, float, np.integer, np.floating))
                                   for v in value):
            value = [value]
        out = []
        for item in value:
            try:
                rgb = tuple(int(v) for v in item)
            except (TypeError, ValueError):
                continue
            if len(rgb) == 3 and all(0 <= v <= 255 for v in rgb):
                out.append(rgb)
        return tuple(out)
    return ()


def palette_sha(palette) -> str:
    """调色板内容摘要（sha256 前 16 位）：版本 + 类名-颜色映射，稳定可复算。"""
    classes = palette_classes(palette)
    payload = {
        "version": str((palette or {}).get("version") or "") if isinstance(palette, Mapping) else "",
        "classes": {name: [list(c) for c in sorted(colors)]
                    for name, colors in sorted(classes.items())},
    }
    return _sha16_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def check_palette(palette, recorded) -> list[str]:
    """调色板一致性拒绝码。

    * ``PALETTE_CHANGED``：版本/类名-颜色映射/摘要与记录不一致，或调色板
      本身缺类、自报摘要与内容对不上（调色板变了，颜色就不再可比）；
    * ``UNKNOWN_CLASS``：记录里出现**未登记类色**（``recorded["colors_seen"]``
      或 ``recorded["unknown_colors"]`` 中有不在调色板里的颜色）。

    ``recorded=None`` = 没有独立记录，只做调色板自洽检查（不猜"没变"，
    也不凭空拒绝；调用方要看清报告里的 ``recorded`` 状态）。
    """
    codes: list[str] = []
    if not isinstance(palette, Mapping):
        return [PALETTE_CHANGED]
    classes = palette_classes(palette)
    if not classes:
        return [PALETTE_CHANGED]
    declared_sha = palette.get("sha") or palette.get("hash")
    if declared_sha and str(declared_sha) not in (palette_sha(palette),):
        codes.append(PALETTE_CHANGED)
    if not isinstance(recorded, Mapping):
        return codes
    rec_classes = palette_classes(recorded)
    if rec_classes:
        if rec_classes != classes:
            codes.append(PALETTE_CHANGED)
    rec_version = recorded.get("version")
    if rec_version and str(rec_version) != str(palette.get("version") or ""):
        codes.append(PALETTE_CHANGED)
    rec_sha = recorded.get("sha") or recorded.get("hash")
    if rec_sha and str(rec_sha) != palette_sha(palette):
        codes.append(PALETTE_CHANGED)
    known = {tuple(c) for colors in classes.values() for c in colors}
    seen = list(recorded.get("colors_seen") or []) + list(
        recorded.get("unknown_colors") or [])
    unknown = []
    for color in seen:
        try:
            rgb = tuple(int(v) for v in color)
        except (TypeError, ValueError):
            continue
        if len(rgb) != 3:
            continue
        if rgb not in known:
            unknown.append(rgb)
    if unknown:
        codes.append(UNKNOWN_CLASS)
    return sorted(set(codes))


def annotation_color_census(ann, *, max_px: int = 200_000) -> list[tuple[int, int, int]]:
    """annotation 图里出现过的颜色（去重；超采样上限时按步长抽样，仍覆盖空间）。"""
    arr = np.asarray(ann) if ann is not None else None
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return []
    flat = arr.reshape(-1, arr.shape[2])[:, :3]
    if flat.shape[0] > int(max_px):
        step = int(math.ceil(flat.shape[0] / float(max_px)))
        flat = flat[::step]
    return [tuple(int(v) for v in row) for row in
            np.unique(flat, axis=0).tolist()]


# ── 5. 标签 hash 与取值 ─────────────────────────────────────────────────────
def check_label(frame) -> list[str]:
    """``label_sha`` 与内容一致、取值 ∈ {0,1,2,255}；否则 ``LABEL_TAMPERED``。"""
    label = frame.get("label")
    if label is None:
        return [LABEL_TAMPERED]
    lab = np.asarray(label)
    if lab.ndim != 2 or lab.size == 0 or lab.dtype != np.uint8:
        return [LABEL_TAMPERED]
    values = {int(v) for v in np.unique(lab).tolist()}
    if not values.issubset(set(LABEL_VALUES)):
        return [LABEL_TAMPERED]
    recorded = frame.get("label_sha")
    if not recorded:
        # 没有 hash = 无法核对（label_sha 是凭证必填项），不当通过。
        return [LABEL_TAMPERED]
    full = hashlib.sha256(np.ascontiguousarray(lab, dtype=np.uint8).tobytes()).hexdigest()
    prefix = label_sha16(lab)          # 复用 labels.py：仓库的 16 位口径
    got = str(recorded).strip().lower()
    if got not in (full, full[:16], prefix):
        return [LABEL_TAMPERED]
    return []


def frame_content_shas(rgb, label) -> tuple[str, str]:
    """``(source_image_sha, label_sha)``（各 16 位，仓库 content/label sha16 口径）。"""
    src = ""
    if rgb is not None:
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.size:
            src = _sha16(arr.tobytes())
    lab = ""
    if label is not None:
        arr = np.ascontiguousarray(np.asarray(label, dtype=np.uint8))
        if arr.size:
            lab = label_sha16(arr)
    return src, lab


# ── 6. 遮挡 ─────────────────────────────────────────────────────────────────
def _line_truth_samples(frame, *, min_gap_px: float = 0.0,
                        max_samples: int = 6000) -> list[tuple[float, float, float]]:
    """沿漆线真值采样 ``(u, v, 轴深)``：相邻同 role 点之间按像素步长线性插值。

    为什么插值：生成器沿漆线每隔 ~1 m 给一个真值点，点与点之间的遮挡物必须有
    证据可查——只用稀疏点会漏掉中间被挡住的线段。直线段投影仍是直线，插值点
    的世界坐标线性插值后再投影，深度取插值点的轴深（不用像素深度近似）。
    """
    camera = frame.get("camera")
    pts = _line_truth_points(frame)
    if camera is None or not pts:
        return []
    groups: dict = {}
    for p in pts:
        try:
            u, v = project_point(p.get("world"), camera)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        try:
            d = _point_depth(p.get("world"), camera)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(d) or d <= 0:
            continue
        key = str(p.get("role")) if p.get("role") else "anon"
        groups.setdefault(key, []).append((float(u), float(v), float(d),
                                           np.asarray(p.get("world"), dtype=float)))
    out: list[tuple[float, float, float]] = []
    for key, items in groups.items():
        items.sort(key=lambda t: t[0])
        for i, (u, v, d, _w) in enumerate(items):
            out.append((u, v, d))
            if i + 1 >= len(items) or len(out) >= max_samples:
                continue
            u2, v2, d2, w2 = items[i + 1]
            w1 = items[i][3]
            span = math.hypot(u2 - u, v2 - v)
            if span > 400.0:
                # 像素跨度太大 = 两点之间可能根本不是同一段线（例如不同车道的
                # 无 role 点混在一起）：不插值，避免编造几何。
                continue
            steps = max(0, int(span / max(1.0, float(min_gap_px) or 1.0)))
            steps = min(steps, 400, max_samples - len(out))
            for s in range(1, steps):
                t = s / float(steps)
                w = (1.0 - t) * w1 + t * w2
                try:
                    su, sv = project_point(w, camera)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(su) and math.isfinite(sv)):
                    continue
                # 直线段的世界点插值 -> 轴深 = 两端轴深的线性插值（旋转无关）。
                sd = (1.0 - t) * d + t * d2
                out.append((su, sv, sd))
        if len(out) >= max_samples:
            break
    return out


def occlusion_audit(frame, *, margin_m: float = OCCLUSION_MARGIN_M) -> dict:
    """遮挡审计：几何上"比漆线更近"的像素不得标成线/背景，应为 ignore(255)。

    判据（保守：只在 depth 明确支持时判定）：

    * 沿漆线真值采样点，取实测轴深 ``d_meas < d_line - margin_m`` 的像素为
      **被遮挡像素**；depth 无效或不足则记 ``unknown_px``；
    * 被遮挡像素的 label 是 2 → ``OCCLUSION_INSERT``（把看不见的线标出来）；
      label 是 0 → ``OCCLUSION_INSERT``（把遮挡当背景，同样制造了假负例）；
      label 是 255 = 正确（该区域 ignore，不计漏检）；
      label 是 1 或其它 → 保守记 ``unknown_px``，不判码。
    """
    out = {"status": "unknown", "codes": [], "occluded_px": 0,
           "line_marked_px": 0, "background_marked_px": 0, "ignored_px": 0,
           "unknown_px": 0, "samples": 0, "why": ""}
    label = frame.get("label")
    if label is None:
        out["why"] = "no label: occlusion cannot be judged"
        return out
    lab = np.asarray(label)
    if lab.ndim != 2 or lab.size == 0:
        out["why"] = "label is not a nonempty 2-D array"
        return out
    if not _line_truth_points(frame):
        out["status"] = "not_applicable"
        out["why"] = ("no line truth points in this frame: there is no known line "
                      "that an occluder could hide")
        return out
    depth = _axis_depth(frame)
    if depth is None or depth.shape != lab.shape:
        out["status"] = "unknown"
        out["unknown_px"] = int(lab.size)
        out["why"] = ("depth is missing or has a different resolution: occlusion "
                      "cannot be judged (recorded as unknown)")
        return out
    h, w = lab.shape[:2]
    samples = _line_truth_samples(frame)
    out["samples"] = len(samples)
    for u, v, d_line in samples:
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            continue
        nearer = _nearer_surface(depth, x, y, d_line, margin=float(margin_m))
        if nearer is None:
            out["unknown_px"] += 1
            continue
        if not nearer:
            continue                      # 没有更近的面：这一像素的可见性无争议
        out["occluded_px"] += 1
        value = int(lab[y, x])
        if value == 2:
            out["line_marked_px"] += 1
        elif value == 0:
            out["background_marked_px"] += 1
        elif value == IGNORE:
            out["ignored_px"] += 1
        else:
            out["unknown_px"] += 1
    if out["line_marked_px"]:
        out["codes"].append(OCCLUSION_INSERT)
    if out["background_marked_px"]:
        out["codes"].append(OCCLUSION_INSERT)
    out["codes"] = sorted(set(out["codes"]))
    if out["occluded_px"]:
        out["status"] = "measured"
        out["why"] = (f"{out['occluded_px']} occluded line pixels: "
                      f"ignore={out['ignored_px']}, line={out['line_marked_px']}, "
                      f"background={out['background_marked_px']}, "
                      f"other={out['unknown_px']}")
    else:
        out["status"] = "unknown"
        out["why"] = ("depth never showed a nearer surface on the line truth "
                      "samples: no occlusion evidence either way")
    return out


def check_occlusion(frame) -> list[str]:
    """遮挡拒绝码（见 `occlusion_audit` 的判据）。"""
    return list(occlusion_audit(frame)["codes"])


# ── 7. 相机翻转 ─────────────────────────────────────────────────────────────
def _mirror_class_match(lab: np.ndarray, u: float, v: float, expected: int) -> bool:
    h, w = lab.shape[:2]
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(int(lab[y, x]) == int(expected))


def flip_audit(frame) -> dict:
    """镜像检测。

    主判据（有 ``truth_points`` 且带 role）：真值点属于模型侧的世界几何，投影
    位置是"正确渲染应该出现的位置"；若类别只出现在**水平镜像位置**
    （``W-1-u``）而正向位置对不上，则渲染画面被镜像 = 相机翻转。

    退化判据（无可用真值点）：RGB 与 annotation 的结构（逐列能量）互相镜像
    —— 只镜像了一路的采集错误。两路**一起**被镜像时该判据无效（那需要真值点
    或录制侧相机标定），此时返回 unknown 而不是硬猜。
    """
    lab = frame.get("label")
    lab = np.asarray(lab) if lab is not None else None
    out = {"flipped": False, "source": "unknown", "direct_votes": 0,
           "mirror_votes": 0, "why": ""}
    pts = _line_truth_points(frame)
    camera = frame.get("camera")
    if lab is not None and lab.ndim == 2 and pts and camera is not None:
        try:
            _c, _r, _f, _u = camera_basis(camera)
            _, _, cx, _cy = camera_intrinsics(camera)
        except ValueError:
            cx = None
        if cx is not None:
            h, w = lab.shape[:2]
            for p in pts:
                try:
                    u, v = project_point(p.get("world"), camera)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(u) and math.isfinite(v)):
                    continue
                if abs(u - cx) < 8.0:            # 中心附近镜像前后都在中心，无判别力
                    continue
                if not (0 <= u < w and 0 <= v < h):
                    continue
                if _mirror_class_match(lab, u, v, 2):
                    out["direct_votes"] += 1
                if _mirror_class_match(lab, (w - 1) - u, v, 2):
                    out["mirror_votes"] += 1
        if out["direct_votes"] or out["mirror_votes"]:
            out["source"] = "truth_points"
            out["flipped"] = bool(out["mirror_votes"] > out["direct_votes"])
            out["why"] = (f"role/line truth votes: direct={out['direct_votes']}, "
                          f"mirrored={out['mirror_votes']}")
            return out
    # 退化判据：RGB 与 annotation 的逐列结构相关性
    rgb = frame.get("rgb")
    ann = frame.get("annotation")
    if rgb is None or ann is None or lab is None:
        out["why"] = "no truth points and no rgb/annotation pair: unknown"
        return out
    palette = frame.get("palette") or {}
    classes = palette_classes(palette)
    line_colors = {tuple(c) for c in classes.get("line", ())}
    ann_arr = np.asarray(ann)
    if ann_arr.ndim != 3 or ann_arr.shape[:2] != lab.shape[:2]:
        out["why"] = "annotation shape does not match label"
        return out
    n_line = int(np.isin(lab, [2]).sum())
    if n_line and not line_colors:
        # 调色板缺席时用"label==2 处的 annotation 颜色"当漆线色（数据自身可判），
        # 仍不引入第二种线检测器。
        mask2 = (lab == 2)
        colors = np.unique(ann_arr[:, :, :3][mask2], axis=0)
        line_colors = {tuple(int(v) for v in c) for c in colors[:8].tolist()}
    prof_ann = np.zeros(lab.shape[1], dtype=float)
    if line_colors:
        mask = np.zeros(lab.shape[:2], dtype=bool)
        for c in line_colors:
            mask |= (ann_arr[:, :, :3] == np.asarray(c, dtype=np.uint8)).all(axis=2)
        prof_ann = mask.mean(axis=0)
    if not prof_ann.any():
        # 结构缺失（没有漆线颜色可分辨）时只报 unknown，不硬猜。
        out["why"] = ("no separable line colour in the annotation: mirror "
                      "consistency not measurable")
        return out
    rgb_arr = np.asarray(rgb, dtype=float)
    lum = rgb_arr[:, :, :3].mean(axis=2) if rgb_arr.ndim == 3 else rgb_arr
    # RGB 的"亮线"结构：逐列的高亮像素比例（与 annotation 线掩码同构），
    # 不做形态学，避免引入第二种 line 检测器。
    thresh = float(np.percentile(lum, 99.0))
    prof_rgb = (lum >= max(thresh, 1.0)).mean(axis=0)

    def _corr(a, b) -> float:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    direct = _corr(prof_rgb, prof_ann)
    mirrored = _corr(prof_rgb, prof_ann[::-1])
    out["source"] = "mirror_consistency"
    out["direct_votes"] = int(round(direct * 100))
    out["mirror_votes"] = int(round(mirrored * 100))
    out["flipped"] = bool(mirrored - direct > 0.05 and mirrored > 0.0)
    out["why"] = (f"rgb/annotation column-profile correlation: direct={direct:.3f}, "
                  f"mirrored={mirrored:.3f} "
                  "(only catches one channel mirrored against the other)")
    return out


def detect_camera_flip(frame) -> bool:
    """相机翻转（见 `flip_audit`）。"""
    return bool(flip_audit(frame)["flipped"])


# ── 8. 逐通道资格 ───────────────────────────────────────────────────────────
def _channel(status: str, why: str, n: int) -> dict:
    assert status in CHANNEL_STATUSES, status
    return {"status": status, "why": why, "n": int(n)}


def _frame_label_counts(label) -> dict:
    lab = np.asarray(label) if label is not None else None
    if lab is None or lab.ndim != 2:
        return {"road": 0, "line": 0, "ignore": 0, "background": 0, "total": 0}
    road = int((lab == 1).sum())
    line = int((lab == 2).sum())
    ignore = int((lab == IGNORE).sum())
    total = int(lab.size)
    return {"road": road, "line": line, "ignore": ignore,
            "background": total - road - line - ignore, "total": total}


def channel_eligibility(batch, *, frame_info=None) -> dict:
    """逐通道资格：``{channel: {status, why, n}}``，通道 = LINE/ROAD/SHOULDER/ROLE。

    某通道合格**不替其它通道背书**：每个通道各自的证据、各自的状态。无线场景
    （生成器声明未生成漆线且无内嵌漆线纹理）的 LINE = ``not_applicable`` 并
    写明依据；标签来源是 pseudo/agent 档时记 ``unverified_labels``（沿用
    ``labels.PAINT_SOURCE_RANK``：弱监督永不作为真值）。
    """
    batch = batch if isinstance(batch, Mapping) else {}
    frames = [f for f in (batch.get("frames") or []) if isinstance(f, Mapping)]
    scene = batch.get("scene") if isinstance(batch.get("scene"), Mapping) else {}
    palette = batch.get("palette") if isinstance(batch.get("palette"), Mapping) else {}
    classes = palette_classes(palette)
    rank = PAINT_SOURCE_RANK.get(str(scene.get("label_source")
                                     or batch.get("label_source") or ""), "absent")
    info = frame_info if isinstance(frame_info, dict) else {}
    n = len(frames)
    label_counts = [_frame_label_counts(f.get("label")) for f in frames]
    line_px = sum(c["line"] for c in label_counts)
    road_px = sum(c["road"] for c in label_counts)
    n_road_frames = sum(1 for c in label_counts if c["road"] >= ROAD_MIN_PX)
    n_line_truth_frames = sum(1 for f in frames if _line_truth_points(f))
    n_line_verified = sum(1 for i in range(len(frames))
                          if (info.get(i) or {}).get("projection_line_verified"))
    n_role_frames = sum(1 for f in frames if _role_evidence(f))
    both_roles = bool(frames) and all(
        any(str(p.get("role")) == "left" for p in _line_truth_points(f))
        and any(str(p.get("role")) == "right" for p in _line_truth_points(f))
        for f in frames)

    line_generated = scene.get("line_generated")
    embedded = scene.get("line_texture_embedded")
    declared_no_line = line_generated is False and embedded is False

    if scene.get("road_defined") is False:
        road = _channel("not_applicable",
                        "scene declares road_defined=False: no road surface in "
                        "this batch", n)
    elif n == 0:
        road = _channel("unknown", "batch has no frames", 0)
    elif rank in ("pseudo", "agent"):
        road = _channel("unverified_labels",
                        f"road labels come from a weak source rank={rank}: "
                        "usable for research arms, never gating truth",
                        n_road_frames)
    elif n_road_frames == n and road_px >= ROAD_MIN_PX:
        road = _channel("measured",
                        f"road class present in {n_road_frames}/{n} frames "
                        f"({road_px} px total) with palette checks available",
                        n_road_frames)
    else:
        road = _channel("unknown",
                        f"road class present in only {n_road_frames}/{n} frames "
                        f"(>= {ROAD_MIN_PX} px per frame required)",
                        n_road_frames)

    if declared_no_line and line_px == 0 and n_line_truth_frames == 0:
        line = _channel("not_applicable",
                        "generator declares line_generated=False and "
                        "line_texture_embedded=False, and no frame carries line "
                        "pixels or line truth points: the line channel has no "
                        "subject (this is NOT proven by model output)",
                        n)
    elif declared_no_line and (line_px or n_line_truth_frames):
        line = _channel("unknown",
                        "scene declares no generated line and no embedded line "
                        f"texture, but frames carry {line_px} line px / "
                        f"{n_line_truth_frames} line-truth frames: the declaration "
                        "contradicts the labels, so neither claim is verified",
                        max(line_px, n_line_truth_frames))
    elif n_line_verified and rank == "verified":
        line = _channel("measured",
                        f"line truth points projected and matched in "
                        f"{n_line_verified}/{n} frames, label source rank=verified",
                        n_line_verified)
    elif rank in ("pseudo", "agent", "unreliable", "absent"):
        evidence = (f"; {n_line_verified}/{n} frames have matching truth-point "
                    "projections" if n_line_verified else "")
        line = _channel("unverified_labels",
                        f"line labels exist but source rank={rank}: engine/"
                        "agent/pseudo labels do not prove visibility or "
                        f"completeness (see labels.PAINT_SOURCE_RANK){evidence}",
                        n_line_verified or n)
    else:
        line = _channel("unknown",
                        "line labels are declared verified but no truth point "
                        "projection evidence was collected for them",
                        n)

    n_road_type = sum(1 for f in frames if _road_type_present(f))
    n_shoulder_px = sum(1 for f in frames
                        if _shoulder_pixels(f, classes))
    shoulder_defined = scene.get("shoulder_defined")
    if n_road_type and n_road_type == n:
        shoulder = _channel("measured",
                            f"per-pixel road-type map present in {n_road_type}/{n} "
                            "frames (asphalt/gravel/shoulder separable)", n_road_type)
    elif shoulder_defined is True and "shoulder" in classes and n_shoulder_px:
        if rank in ("pseudo", "agent"):
            shoulder = _channel("unverified_labels",
                                f"shoulder class present in the palette but labels "
                                f"come from rank={rank}", n_shoulder_px)
        else:
            shoulder = _channel("measured",
                                f"scene defines separate shoulder material and the "
                                f"annotation marks it in {n_shoulder_px}/{n} frames",
                                n_shoulder_px)
    elif shoulder_defined is False and "shoulder" not in classes:
        shoulder = _channel("not_applicable",
                            "scene declares shoulder_defined=False and the palette "
                            "has no shoulder class: no pavement/shoulder split to "
                            "measure", n)
    else:
        shoulder = _channel("unknown",
                            "shoulder/pavement split needs its own material or "
                            "road-type evidence; neither is complete in this batch",
                            n_shoulder_px)

    if line["status"] == "not_applicable":
        role = _channel("not_applicable",
                        "line channel is not_applicable, so left/right roles have "
                        "no subject in this batch", n)
    elif rank in ("pseudo", "agent", "unreliable", "absent"):
        role = _channel("unverified_labels",
                        f"role labels come from rank={rank}: machine/pseudo roles "
                        "are not gating truth", n_role_frames)
    elif both_roles and n_role_frames == n:
        role = _channel("measured",
                        f"both left and right role-labelled line truth in "
                        f"{n_role_frames}/{n} frames", n_role_frames)
    else:
        role = _channel("unknown",
                        f"role evidence incomplete: {n_role_frames}/{n} frames with "
                        "role-labelled line truth (both left and right required)",
                        n_role_frames)
    return {"LINE": line, "ROAD": road, "SHOULDER": shoulder, "ROLE": role}


def _role_evidence(frame) -> bool:
    return any(p.get("role") in ("left", "right") for p in _line_truth_points(frame))


def _road_type_present(frame) -> bool:
    rt = frame.get("road_type")
    if rt is None:
        return False
    arr = np.asarray(rt)
    return bool(arr.ndim == 2 and arr.size and
                {int(v) for v in np.unique(arr).tolist()} <= {1, 2, 3})


def _shoulder_pixels(frame, classes: dict) -> int:
    ann = frame.get("annotation")
    if ann is None or "shoulder" not in classes:
        return 0
    arr = np.asarray(ann)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0
    total = 0
    for color in classes["shoulder"]:
        total += int((arr[:, :, :3] == np.asarray(color, dtype=np.uint8)).all(axis=2).sum())
    return total


# ── 9. 批次汇总 ─────────────────────────────────────────────────────────────
def _safe(fn, code: str, frame_id, rejections: list, stats: dict):
    """检查函数内部异常 = 该检查不成立（记拒绝码 + 错误明细），绝不当通过。"""
    try:
        return fn()
    except Exception as exc:                                  # noqa: BLE001
        rejections.append({"code": code, "frame_id": frame_id,
                           "detail": f"check raised {type(exc).__name__}: {exc}"})
        stats.setdefault("errors", []).append(
            {"frame_id": frame_id, "code": code,
             "error": f"{type(exc).__name__}: {exc}"})
        return None


def verify_batch(batch, *, expected=None) -> dict:
    """批次校验汇总（契约见模块 docstring）。

    返回 ``{ok, rejections, channels, stats}``：

    * ``rejections`` 每项 ``{"code","frame_id","detail"}``，``code`` 取模块常量；
    * ``channels`` 来自 `channel_eligibility`（逐通道独立，不合格不替别人背书）；
    * ``stats`` 给证据计数（投影 checked/mismatch、遮挡 unknown_px、黑图数、
      逐帧 sha 等）——没有证据的通道在报告里必须是 unknown，不是通过。

    ``expected`` 可选：``{"channels": {...}}`` 或直接 ``{"LINE": "measured"}``。
    与实测资格不符 → ``ok=False`` 且记 ``stats["expected_mismatches"]``（不编造
    新的拒绝码：拒绝码只有冻结文档列出的十个）。
    """
    rejections: list[dict] = []
    stats: dict = {"verifier_version": VERIFIER_VERSION, "errors": [],
                   "n_frames": 0, "code_counts": {}, "frames": []}
    batch = batch if isinstance(batch, Mapping) else {}
    frames = [f for f in (batch.get("frames") or []) if isinstance(f, Mapping)]
    stats["n_frames"] = len(frames)
    scene = batch.get("scene") if isinstance(batch.get("scene"), Mapping) else {}
    palette = batch.get("palette") if isinstance(batch.get("palette"), Mapping) else {}
    recorded = batch.get("recorded_palette")

    def _add(code: str, frame_id, detail: str) -> None:
        rejections.append({"code": code, "frame_id": frame_id, "detail": detail})
        stats["code_counts"][code] = stats["code_counts"].get(code, 0) + 1

    base_palette_codes = check_palette(palette, recorded) if palette else [PALETTE_CHANGED]
    stats["palette"] = {"version": str(palette.get("version") or ""),
                        "sha": palette_sha(palette) if palette else "",
                        "recorded": bool(isinstance(recorded, Mapping)),
                        "codes": sorted(set(base_palette_codes))}
    for code in sorted(set(base_palette_codes)):
        _add(code, None, "batch palette check failed")

    prev_ts = None
    proj_checked = proj_mismatch = proj_ignored = proj_occluded = proj_skipped = 0
    black_frames = 0
    occ_occluded = occ_unknown = 0
    area_road = area_line = area_ignore = area_total = 0
    frame_info: dict = {}
    for i, frame in enumerate(frames):
        # 逐帧检查需要看到批次调色板（flip 的退化判据要从类名-颜色映射取漆线色）；
        # 只做浅拷贝，数组本体不复制也不改。
        frame = dict(frame)
        frame["palette"] = palette
        fid = frame.get("frame_id")
        ts = frame.get("timestamp")
        # 2. 帧身份（异步旧帧 / 时间错帧 / 尺寸变化）
        identity_detail = {
            ASYNC_STALE_FRAME: "rgb/annotation/depth frame ids differ (async "
                               "stale frame) or frame_id missing",
            TIME_MISMATCH: "timestamp missing or earlier than the previous frame",
            RESOLUTION_MISMATCH: "rgb/annotation/depth resolutions differ or do "
                                 "not match the declared camera size",
        }
        for code in _safe(lambda f=frame, p=prev_ts: frame_identity_ok(
                f, prev_timestamp=p), RESOLUTION_MISMATCH, fid, rejections,
                stats) or []:
            _add(code, fid, identity_detail.get(code, "frame identity check failed"))
        if ts is not None:
            try:
                prev_ts = float(ts)
            except (TypeError, ValueError):
                pass
        # 3. 黑图
        if _safe(lambda f=frame: detect_black_frame(f.get("rgb")), BLACK_FRAME,
                 fid, rejections, stats):
            black_frames += 1
            _add(BLACK_FRAME, fid, "rgb mean below threshold: black/stale frame")
        # 4. 调色板：逐帧查未登记类色
        census = _safe(lambda f=frame: annotation_color_census(f.get("annotation")),
                       UNKNOWN_CLASS, fid, rejections, stats)
        if census is not None:
            frame_palette_codes = _safe(
                lambda p=palette, c=census: check_palette(
                    p, {"classes": palette_classes(p), "version": palette.get("version"),
                        "colors_seen": c}),
                UNKNOWN_CLASS, fid, rejections, stats) or []
            for code in sorted(set(frame_palette_codes) - set(base_palette_codes)):
                _add(code, fid, f"annotation carries colours not registered in the "
                                f"palette: {census[:6]}")
        # 5. 标签 hash 与取值
        for code in _safe(lambda f=frame: check_label(f), LABEL_TAMPERED,
                          fid, rejections, stats) or []:
            _add(code, fid, "label_sha/values inconsistent with the label content")
        # 1. 投影
        rep = _safe(lambda f=frame: verify_projection(f), PROJECTION_MISMATCH,
                    fid, rejections, stats)
        if rep is not None:
            proj_checked += rep["checked"]
            proj_mismatch += rep["mismatches"]
            proj_ignored += rep["ignored"]
            proj_occluded += rep["occluded"]
            proj_skipped += rep["skipped"]
            frame_info[i] = {
                "projection_checked": rep["checked"],
                "projection_mismatches": rep["mismatches"],
                "projection_line_verified": bool(
                    any(int(p.get("class", 0)) == 2 for p in _line_truth_points(frame))
                    and rep["checked"] > 0 and rep["mismatches"] == 0),
                "projection_status": rep["status"],
            }
            if rep["checked"] > 0 and rep["mismatches"] > 0:
                _add(PROJECTION_MISMATCH, fid,
                     f"{rep['mismatches']}/{rep['checked']} truth points have no "
                     f"declared class within {PROJECTION_RADIUS_PX} px")
        # 7. 翻转
        audit = _safe(lambda f=frame: flip_audit(f), CAMERA_FLIP, fid,
                      rejections, stats)
        if audit is not None:
            frame_info.setdefault(i, {})["flip"] = audit
            if audit["flipped"]:
                _add(CAMERA_FLIP, fid, audit["why"])
        # 6. 遮挡
        occ = _safe(lambda f=frame: occlusion_audit(f), OCCLUSION_INSERT, fid,
                    rejections, stats)
        if occ is not None:
            occ_occluded += occ["occluded_px"]
            occ_unknown += occ["unknown_px"]
            frame_info.setdefault(i, {})["occlusion"] = occ
            for code in occ["codes"]:
                _add(code, fid, occ["why"])
        src_sha, lab_sha = frame_content_shas(frame.get("rgb"), frame.get("label"))
        counts = _frame_label_counts(frame.get("label"))
        area_road += counts["road"]
        area_line += counts["line"]
        area_ignore += counts["ignore"]
        area_total += counts["total"]
        cam = frame.get("camera") if isinstance(frame.get("camera"), Mapping) else {}
        stats["frames"].append({
            "frame_id": fid, "timestamp": ts,
            "camera": str(cam.get("name") or ""),
            "source_image_sha": str(frame.get("source_image_sha") or src_sha),
            "label_sha": str(frame.get("label_sha") or lab_sha),
            "label_px": counts,
            "projection": frame_info.get(i, {}).get("projection_status", ""),
        })

    channels = channel_eligibility(batch, frame_info=frame_info)
    # 有效面积与拒绝率分开统计（冻结文档 §3.4）：ignore 像素既不算假阳也不算
    # 漏检，但它有多少必须看得见，不能混进"有效面积"里。
    area = {"road_px": area_road, "line_px": area_line, "ignore_px": area_ignore,
            "total_px": area_total,
            "valid_px": area_total - area_ignore,
            "ignore_frac": round(area_ignore / area_total, 5) if area_total else None}
    rejected_frames = {str(r["frame_id"]) for r in rejections
                       if r.get("frame_id") is not None}
    stats.update({
        "black_frames": black_frames,
        "valid_area": area,
        "rejection_rate": {"frames_rejected": len(rejected_frames),
                           "frames_total": len(frames),
                           "rejections": len(rejections)},
        "projection": {"checked": proj_checked, "mismatches": proj_mismatch,
                       "ignored": proj_ignored, "occluded": proj_occluded,
                       "skipped": proj_skipped, "radius_px": PROJECTION_RADIUS_PX},
        "occlusion": {"occluded_px": occ_occluded, "unknown_px": occ_unknown,
                      "margin_m": OCCLUSION_MARGIN_M},
        "scene": dict(scene),
        "generator": dict(batch.get("generator") or {}),
        "asset": dict(batch.get("asset") or {}),
        "run": dict(batch.get("run") or {}),
    })

    expected_mismatches: list[dict] = []
    if isinstance(expected, Mapping):
        want = expected.get("channels") if isinstance(expected.get("channels"), Mapping) \
            else {k: v for k, v in expected.items() if k in CHANNELS}
        for name, status in want.items():
            got = (channels.get(name) or {}).get("status")
            if got != status:
                expected_mismatches.append({"channel": name, "expected": str(status),
                                            "got": str(got)})
        exp_frames = expected.get("n_frames")
        if exp_frames is not None and int(exp_frames) != len(frames):
            expected_mismatches.append({"channel": "n_frames",
                                        "expected": int(exp_frames),
                                        "got": len(frames)})
    stats["expected_mismatches"] = expected_mismatches
    stats["empty_batch"] = not frames
    stats["evidence"] = {
        "projection_checked_points": proj_checked,
        "occlusion_occluded_px": occ_occluded,
        "line_truth_frames": sum(1 for f in frames if _line_truth_points(f)),
    }
    # 空批次不是"没有拒绝码"就能通过：没有任何帧 = 没有任何证据。
    ok = bool(frames) and not rejections and not expected_mismatches
    return {"ok": ok, "rejections": rejections, "channels": channels,
            "stats": stats, "verifier_version": VERIFIER_VERSION,
            "truth_contract": TRUTH_CONTRACT}


# ── 10. 凭证写入 ────────────────────────────────────────────────────────────
def _digest(values) -> str:
    vals = [str(v) for v in values if str(v)]
    if not vals:
        return ""
    return _sha16_text("|".join(sorted(vals)))


def _normalize_provenance(provenance, batch_report, frames, channels) -> dict:
    src = provenance if isinstance(provenance, Mapping) else {}
    stats = batch_report.get("stats") if isinstance(batch_report.get("stats"), Mapping) else {}

    def _block(name: str, keys: tuple) -> dict:
        given = src.get(name) if isinstance(src.get(name), Mapping) else {}
        return {k: given.get(k) for k in keys}

    generator = _block("generator", ("name", "version", "sha"))
    asset = _block("asset", ("map", "segment", "sha"))
    run = _block("run", ("id", "scene_seed", "game_version", "renderer"))
    camera = _block("camera", ("name", "calibration_sha", "frame_ids"))
    labels = _block("labels", ("source_image_sha", "label_sha",
                               "channel_valid_area", "unknown_reason"))
    report = _block("report", ("test_report_sha", "verifier_version", "verified"))

    if not generator.get("name"):
        generator["name"] = str((stats.get("generator") or {}).get("name") or "")
    if not generator.get("sha"):
        generator["sha"] = str((stats.get("generator") or {}).get("sha") or "")
    for key in ("map", "segment", "sha"):
        if not asset.get(key):
            asset[key] = (stats.get("asset") or {}).get(key)
    for key in ("id", "scene_seed", "game_version", "renderer"):
        if not run.get(key):
            run[key] = (stats.get("run") or {}).get(key) or (stats.get("scene") or {}).get(key)
    if not camera.get("frame_ids"):
        camera["frame_ids"] = [f.get("frame_id") for f in frames]
    if not camera.get("name"):
        names = [str(f.get("camera") or "") for f in frames]
        camera["name"] = next((n for n in names if n), "")
    if not labels.get("label_sha"):
        labels["label_sha"] = _digest(f.get("label_sha") for f in frames)
    if not labels.get("source_image_sha"):
        labels["source_image_sha"] = _digest(f.get("source_image_sha") for f in frames)
    if not isinstance(labels.get("channel_valid_area"), Mapping):
        labels["channel_valid_area"] = {name: (channels.get(name) or {}).get("status", "")
                                        for name in CHANNELS}
    if labels.get("unknown_reason") is None:
        labels["unknown_reason"] = ""
    if not report.get("test_report_sha"):
        report["test_report_sha"] = _sha16_text(json.dumps(
            batch_report, sort_keys=True, default=str))
    if not report.get("verifier_version"):
        report["verifier_version"] = str(
            batch_report.get("verifier_version") or stats.get("verifier_version")
            or VERIFIER_VERSION)
    report["verified"] = bool(report.get("verified") is True)
    return {"generator": generator, "asset": asset, "run": run,
            "camera": camera, "labels": labels, "report": report}


def write_truth_credentials(d: str | Path, *, batch_report, provenance) -> dict:
    """写/合并 ``<dir>/annotation.json``（凭证契约 v1，冻结文档 §3.4）。

    * ``truth_contract="v1"`` + ``truth_provenance``（generator/asset/run/camera/
      labels/report 六块，``report.verifier_version`` 必有）；
    * 逐帧记录 ``source_image_sha`` / ``label_sha``；
    * ``label_source="engine_verified"`` **只有**在 ``batch_report["ok"] is True``
      **且** ``provenance["report"]["verified"] is True`` 时才写；否则写 ``""``
      （无来源声明）并在 ``why`` 里说明原因。写入的 ``report.verified`` 是两者
      的与——校验没过时不得留下一个"已验证"的 sidecar。
    """
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "annotation.json"
    existing: dict = {}
    if path.is_file():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                existing = blob
        except (OSError, ValueError):
            existing = {}
    batch_report = batch_report if isinstance(batch_report, Mapping) else {}
    provenance = provenance if isinstance(provenance, Mapping) else {}
    stats = batch_report.get("stats") if isinstance(batch_report.get("stats"), Mapping) else {}
    channels = batch_report.get("channels") if isinstance(batch_report.get("channels"), Mapping) else {}
    frames = [f for f in (stats.get("frames") or batch_report.get("frames") or [])
              if isinstance(f, Mapping)]
    frames = [{"frame_id": f.get("frame_id"), "timestamp": f.get("timestamp"),
               "camera": f.get("camera"),
               "source_image_sha": str(f.get("source_image_sha") or ""),
               "label_sha": str(f.get("label_sha") or "")} for f in frames]

    ok = bool(batch_report.get("ok") is True)
    got_claim = bool((provenance.get("report") or {}).get("verified") is True) \
        if isinstance(provenance.get("report"), Mapping) else False
    verified = bool(ok and got_claim)
    prov = _normalize_provenance(provenance, batch_report, frames, channels)
    prov["report"]["verified"] = verified
    if verified:
        why = "batch_report.ok=True and provenance.report.verified=True"
    elif not ok:
        codes = sorted({str(r.get("code")) for r in (batch_report.get("rejections") or [])
                        if isinstance(r, Mapping)})
        why = (f"batch_report.ok is False (rejections={codes or 'none'}): "
               "label_source stays empty; engine_verified is never written "
               "without a passing report")
    else:
        why = ("provenance.report.verified is not True: the generator did not "
               "claim a verified batch, so label_source stays empty")
    if not prov["report"].get("verifier_version"):
        prov["report"]["verifier_version"] = VERIFIER_VERSION

    old_frames = existing.get("frames") if isinstance(existing.get("frames"), list) else []
    new_ids = {str(f.get("frame_id")) for f in frames}
    merged_frames = [f for f in old_frames
                     if not isinstance(f, Mapping)
                     or str(f.get("frame_id") or "") not in new_ids] + frames
    out = dict(existing)
    previous = str(existing.get("label_source") or "")
    if previous and previous != ("engine_verified" if verified else ""):
        out["previous_label_source"] = previous
    out.update({
        "label_source": "engine_verified" if verified else "",
        "truth_contract": TRUTH_CONTRACT,
        "truth_provenance": prov,
        "frames": merged_frames,
        "why": why,
    })
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ── 11. 反例生成器（测试与探针自检；不参与判定逻辑） ────────────────────────
#: 合成场景的调色板颜色（annotation 语义色）与 RGB 观感色。RGB 只是"和几何
#: 一致的另一路渲染"，用于镜像退化判据与黑图检测，不追求照片级。
_SYNTH_ANN = {
    "background": (0, 0, 0),
    "road": (128, 128, 128),
    "line": (240, 240, 64),
    "shoulder": (168, 120, 72),
    "grass": (56, 120, 48),
    "sky": (88, 144, 216),
    "occluder": (200, 32, 32),
}
_SYNTH_RGB = {
    "background": (0, 0, 0),
    "road": (72, 72, 76),
    "line": (238, 230, 120),
    "shoulder": (150, 120, 86),
    "grass": (58, 110, 52),
    "sky": (110, 150, 190),
    "occluder": (120, 40, 40),
}


def _road_frame(curve_deg: float, slope_deg: float):
    """局部道路坐标基（世界系）：``a_u`` 沿路向前、``a_v`` 车左、``a_w`` 向上。"""
    yaw, slope = math.radians(float(curve_deg)), math.radians(float(slope_deg))
    a_u = np.array([math.cos(yaw) * math.cos(slope),
                    math.sin(yaw) * math.cos(slope), math.sin(slope)])
    a_v = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    a_w = np.cross(a_u, a_v)
    return a_u, a_v, a_w


def _render_synthetic(*, width, height, fov_y_deg, cam_pitch_deg, cam_height_m,
                      curve_deg, slope_deg, bands, lines, occluder,
                      cam_offset_m=(0.0, 0.0)):
    """局部道路坐标系里逐像素求交，返回三路图 + 真值点 + 相机记录。

    ``cam_offset_m=(前向, 横向)``：同一场景的不同曝光只在相机位姿上有差别，
    几何（道路/漆线/遮挡盒）不动——这样多帧批次内容各不相同，而全部真值点
    仍然描述同一批世界几何。
    """
    fx = fy = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    a_u, a_v, a_w = _road_frame(curve_deg, slope_deg)
    off_fwd, off_lat = (float(v) for v in cam_offset_m)
    cam_pos = cam_height_m * a_w + off_fwd * a_u + off_lat * a_v
    pitch = math.radians(cam_pitch_deg)
    fwd = math.cos(pitch) * a_u + math.sin(pitch) * a_w
    up = -math.sin(pitch) * a_u + math.cos(pitch) * a_w
    right = -a_v

    us = np.arange(width, dtype=float)
    vs = np.arange(height, dtype=float)
    xr = (us - cx) / fx
    yr = -(vs - cy) / fy
    d = (fwd[None, None, :] + xr[None, :, None] * right[None, None, :]
         + yr[:, None, None] * up[None, None, :])       # d·fwd == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = d @ a_w
        t_ground = -cam_height_m / denom
    valid_g = (denom < -1e-9) & (t_ground > 0)
    hit_g = cam_pos[None, None, :] + t_ground[..., None] * d
    du = hit_g @ a_u
    dv = hit_g @ a_v

    cls = np.full((height, width), "sky", dtype=object)
    cls[valid_g] = "grass"
    for lo, hi, name in bands:
        cls[valid_g & (dv >= lo) & (dv < hi)] = name
    line_mask = np.zeros((height, width), dtype=bool)
    for spec in lines:
        line_mask |= (valid_g & (np.abs(dv - spec["v"]) <= spec["half"])
                      & (du >= spec["u0"]) & (du <= spec["u1"]))
    cls[line_mask] = "line"

    hit_box = np.zeros((height, width), dtype=bool)
    t_enter = np.full((height, width), np.inf)
    if occluder is not None:
        u0, u1, v0, v1, w0, w1 = occluder
        o_l = np.array([off_fwd, off_lat, cam_height_m])
        dl = np.stack([d @ a_u, d @ a_v, d @ a_w], axis=-1)
        tmin = np.full((height, width), -np.inf)
        tmax = np.full((height, width), np.inf)
        for axis, (lo, hi) in enumerate(((u0, u1), (v0, v1), (w0, w1))):
            comp = dl[:, :, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - o_l[axis]) / comp
                t2 = (hi - o_l[axis]) / comp
            lo_t = np.minimum(t1, t2)
            hi_t = np.maximum(t1, t2)
            zero = np.abs(comp) < 1e-12
            lo_t = np.where(zero, np.where((lo - o_l[axis]) <= 0, np.inf, -np.inf), lo_t)
            hi_t = np.where(zero, np.where((hi - o_l[axis]) >= 0, np.inf, -np.inf), hi_t)
            tmin = np.maximum(tmin, lo_t)
            tmax = np.minimum(tmax, hi_t)
        hit_box = (tmin <= tmax) & (tmax > 0)
        t_enter = np.where(hit_box, np.maximum(tmin, 0.0), np.inf)

    ground_first = valid_g & (~hit_box | (t_ground <= t_enter))
    box_first = hit_box & (~valid_g | (t_enter < t_ground))
    hidden = box_first & _dilate(line_mask & valid_g, 2)
    visible_line = ground_first & _dilate(line_mask, 1)
    cls[box_first] = "occluder"
    cls[visible_line] = "line"

    depth = np.zeros((height, width), dtype=float)
    depth[ground_first] = t_ground[ground_first]
    depth[box_first] = t_enter[box_first]
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

    ann = np.zeros((height, width, 3), dtype=np.uint8)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for name, color in _SYNTH_ANN.items():
        ann[cls == name] = color
        rgb[cls == name] = _SYNTH_RGB[name]
    ignore = (cls == "sky") | hidden | (depth <= 0)
    return {"ann": ann, "rgb": rgb, "depth": depth, "ignore": ignore}


def make_synthetic_batch(*, width: int = 192, height: int = 144,
                         fov_y_deg: float = 65.0, cam_pitch_deg: float = -12.0,
                         cam_height_m: float = 1.4, n_frames: int = 2,
                         step_m: float = 2.0, lateral_step_m: float = 0.25,
                         with_line: bool = True,
                         second_line: bool = False, with_occluder: bool = False,
                         curve_deg: float = 0.0, slope_deg: float = 0.0,
                         material_mix: bool = False,
                         label_source: str = "engine_verified",
                         camera_name: str = "front_main",
                         timestamp0: float = 1000.0, dt: float = 0.05,
                         scene_seed: int = 0) -> dict:
    """造一个几何已知的合成场景（有/无线、遮挡、坡面曲线、材质混合）。

    真值由生成器自己的几何定义产生：漆线段的世界点（class=2，带 left/right
    role）与路面点（class=1）。渲染与校验共用同一套针孔定义，但校验针对的是
    **图与声明是否一致**——翻转/缩放/遮挡插入/篡改都会被图与真值的矛盾抓住。
    """
    road_half = 3.0 if material_mix else 3.5
    bands = [(-road_half, road_half, "road")]
    if material_mix:
        bands = [(-3.0, 3.0, "road"), (-4.6, -3.0, "shoulder"),
                 (3.0, 4.6, "shoulder")]
    lines: list[dict] = []
    if with_line:
        lines.append({"v": 0.9, "half": 0.08, "u0": 1.0, "u1": 34.0,
                      "role": "left"})
    if with_line and second_line:
        lines.append({"v": -0.9, "half": 0.08, "u0": 1.0, "u1": 34.0,
                      "role": "right"})
    occluder = (8.5, 11.0, -1.3, 1.3, 0.0, 1.25) if with_occluder else None
    a_u, a_v, a_w = _road_frame(curve_deg, slope_deg)
    pitch = math.radians(cam_pitch_deg)
    roll, pitch_r, yaw_r = euler_from_basis(
        -a_v, math.cos(pitch) * a_u + math.sin(pitch) * a_w)
    palette = {"version": "synthetic_v1", "source": "auto_truth.generator",
               "classes": {name: list(color) for name, color in _SYNTH_ANN.items()}}
    palette["sha"] = palette_sha(palette)
    truth_points: list[dict] = []
    for spec in lines:
        u = 2.5
        while u <= 32.0:
            world = u * a_u + spec["v"] * a_v
            truth_points.append({"world": [float(x) for x in world], "class": 2,
                                 "role": spec["role"]})
            u += 1.0
    u = 3.0
    while u <= 32.0:
        for v in (0.0, 1.7, -1.7):
            world = u * a_u + v * a_v
            truth_points.append({"world": [float(x) for x in world], "class": 1,
                                 "role": None})
        u += 3.0
    road_colors = [_SYNTH_ANN["road"]]
    line_colors = [_SYNTH_ANN["line"]]
    frames: list[dict] = []
    for i in range(int(n_frames)):
        # 同一场景的不同曝光：只动相机位姿（前进 + 微横移），几何与真值不动。
        render = _render_synthetic(
            width=width, height=height, fov_y_deg=fov_y_deg,
            cam_pitch_deg=cam_pitch_deg, cam_height_m=cam_height_m,
            curve_deg=curve_deg, slope_deg=slope_deg, bands=bands, lines=lines,
            occluder=occluder,
            cam_offset_m=(float(step_m) * i, float(lateral_step_m) * i))
        cam_pos = (cam_height_m * a_w) + (float(step_m) * i) * a_u \
            + (float(lateral_step_m) * i) * a_v
        cam = {"name": f"{camera_name}",
               "width": int(width), "height": int(height),
               "fov_y_deg": float(fov_y_deg),
               "pose": {"pos": [float(x) for x in cam_pos],
                        "rot": [float(roll), float(pitch_r), float(yaw_r)]}}
        label = to_label(render["ann"], road_colors=road_colors,
                         line_colors=line_colors)
        label[render["ignore"]] = IGNORE
        label = np.ascontiguousarray(label, dtype=np.uint8)
        fid = f"{camera_name}_{scene_seed:02d}_{i:05d}"
        rgb = np.ascontiguousarray(render["rgb"], dtype=np.uint8)
        src_sha, lab_sha = frame_content_shas(rgb, label)
        frames.append({
            "frame_id": fid,
            "timestamp": float(timestamp0) + i * float(dt),
            "channel_ids": {"rgb": fid, "annotation": fid, "depth": fid},
            "camera": cam,
            "rgb": rgb,
            "annotation": np.ascontiguousarray(render["ann"], dtype=np.uint8),
            "depth": np.ascontiguousarray(render["depth"], dtype=np.float32),
            "label": label,
            "label_sha": lab_sha,
            "source_image_sha": src_sha,
            "truth_points": [dict(p) for p in truth_points],
        })
    segment = ("probe_curve" if curve_deg or slope_deg else
               "probe_occluded" if with_occluder else
               "probe_material" if material_mix else
               "probe_straight" if with_line else "probe_noline")
    batch = {
        "frames": frames,
        "scene": {"map": "synthetic_grid", "segment": segment,
                  "run_id": f"synthetic_{segment}_{scene_seed}",
                  "scene_seed": int(scene_seed),
                  "game_version": "synthetic", "renderer": "raycast_v1",
                  "line_generated": bool(with_line),
                  "line_texture_embedded": False,
                  "shoulder_defined": bool(material_mix),
                  "roles_defined": bool(second_line),
                  "road_defined": True,
                  "materials": [name for _lo, _hi, name in bands],
                  "label_source": str(label_source)},
        "palette": palette,
        "generator": {"name": "auto_truth.make_synthetic_batch",
                      "version": VERIFIER_VERSION,
                      "sha": _sha16_text(json.dumps(
                          {"segment": segment, "seed": int(scene_seed),
                           "w": int(width), "h": int(height),
                           "curve": curve_deg, "slope": slope_deg},
                          sort_keys=True))},
        "asset": {"map": "synthetic_grid", "segment": segment,
                  "sha": _sha16_text(segment)},
        "run": {"id": f"synthetic_{segment}_{scene_seed}",
                "scene_seed": int(scene_seed), "game_version": "synthetic",
                "renderer": "raycast_v1"},
    }
    return batch


def _copy_batch(batch: dict) -> dict:
    out = dict(batch)
    out["frames"] = []
    for frame in batch.get("frames") or []:
        new = dict(frame)
        for key in ("rgb", "annotation", "depth", "label", "road_type"):
            if new.get(key) is not None:
                new[key] = np.array(new[key], copy=True)
        if new.get("truth_points"):
            new["truth_points"] = [dict(p) for p in new["truth_points"]]
        out["frames"].append(new)
    out["scene"] = dict(batch.get("scene") or {})
    return out


def inject(batch, kind: str, **kwargs):
    """在合法批次上注入**恰好一类**错误（反例生成器，供测试/探针自检）。

    ==============  ==========================================================
    ``"flip"``       水平镜像 rgb/annotation/depth/label（真值点保持世界几何）
    ``"resize"``     只把 depth 缩一半（陈旧/错位缓冲）：三路分辨率不一致
    ``"time_shift"`` 把第 2 帧时间戳调到第 1 帧之前（时间错帧）
    ``"occlude"``    在漆线中段插入更近的遮挡物；``keep_line=True``（默认）把
                     被挡区域仍标成线(2)，``keep_line=False`` 改标背景(0)
    ``"tamper"``     把顶部天空行改成 class 2 但不更新 ``label_sha``
    ==============  ==========================================================
    """
    kind = str(kind)
    out = _copy_batch(batch)
    if kind == "flip":
        for frame in out["frames"]:
            for key in ("rgb", "annotation", "depth", "label"):
                frame[key] = np.ascontiguousarray(frame[key][:, ::-1])
        _refresh_shas(out)
        return out
    if kind == "resize":
        factor = int(kwargs.get("factor", 2))
        for frame in out["frames"]:
            frame["depth"] = np.ascontiguousarray(frame["depth"][::factor, ::factor])
        return out
    if kind == "time_shift":
        frames = out["frames"]
        if len(frames) >= 2:
            frames[1]["timestamp"] = float(frames[0]["timestamp"]) - 0.1
        elif frames:
            frames[0]["timestamp"] = None
        return out
    if kind == "occlude":
        keep_line = bool(kwargs.get("keep_line", True))
        for frame in out["frames"]:
            label = frame.get("label")
            if label is None or not _line_truth_points(frame):
                continue
            lab = np.array(label, copy=True)
            samples = _line_truth_samples(frame)
            if not samples:
                continue
            samples.sort(key=lambda t: t[0])
            lo = int(len(samples) * 0.35)
            hi = max(lo + 1, int(len(samples) * 0.65))
            region = np.zeros(lab.shape, dtype=bool)
            h, w = lab.shape[:2]
            for u, v, _d in samples[lo:hi]:
                x, y = int(round(u)), int(round(v))
                if 0 <= x < w and 0 <= y < h:
                    region[y, x] = True
            region = _dilate(region, 2)
            depth = np.array(frame["depth"], dtype=np.float32, copy=True)
            # 遮挡物 = 一条更近的面：按插值采样点的 0.45 倍轴深写入，区域其余
            # 像素沿用各自原深度按同比例拉近（保证"更近"在任何点都成立）。
            depth[region] = np.where(depth[region] > 0, depth[region] * 0.45, np.inf)
            for s_u, s_v, s_d in samples[lo:hi]:
                x, y = int(round(s_u)), int(round(s_v))
                if 0 <= x < w and 0 <= y < h:
                    depth[y, x] = float(s_d) * 0.45
            frame["depth"] = np.ascontiguousarray(depth)
            # 遮挡物同时出现在视觉图与 annotation（保持"两路描述同一场景"），
            # 颜色是调色板已登记的 occluder 类。
            for key, colors in (("rgb", _SYNTH_RGB), ("annotation", _SYNTH_ANN)):
                img = np.array(frame[key], copy=True)
                img[region] = colors["occluder"]
                frame[key] = img
            if not keep_line:
                lab[region & (lab == 2)] = 0
                frame["label"] = lab
                frame["label_sha"] = label_sha16(lab)
            _refresh_shas(out, labels=False)
        return out
    if kind == "tamper":
        rows = int(kwargs.get("rows", 4))
        for frame in out["frames"]:
            lab = np.array(frame["label"], copy=True)
            lab[:rows, :] = 2
            frame["label"] = lab
            # 故意不更新 label_sha：内容与记录 hash 不符 = 篡改
        return out
    raise ValueError(f"unknown injection kind {kind!r}")


def _refresh_shas(batch: dict, *, labels: bool = True) -> None:
    for frame in batch.get("frames") or []:
        src, lab = frame_content_shas(frame.get("rgb"), frame.get("label"))
        frame["source_image_sha"] = src
        if labels:
            frame["label_sha"] = lab
