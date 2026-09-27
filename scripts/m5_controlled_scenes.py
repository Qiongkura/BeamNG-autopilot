"""五类受控场景的可复现生成入口（T16 §4.1/§4.2 路线 B）。

为什么要它：原生地图的线通道实测无覆盖（italy road 45070 的 annotation 线类
像素为 0，见 `docs/T16_ORDER1_2_REPORT_20260927.md` §5），所以线真值不能指望
地图自带贴花。本脚本在 **italy** 上用**地图自己的 DecalRoad 材质**程序化生成
五类受控场景，真值来自**生成侧的定义**（每条线的节点链 + 宽度 + 左右角色），
渲染结果再用 `verify_batch` 校验：

| 场景 | 生成物 | 真值要点 |
|---|---|---|
| known_line | 白线（左界）+ 黄线（右界） | 2 条线实例，role=left/right |
| known_no_line | 不生成任何线 | `line_generated=False`（对照） |
| occluded_line | 白线 + 遮挡车（真车模型，停在线上 8 m 处） | 被挡点由深度判遮挡，不计漏检 |
| slope_curve | 白线 + 黄线（选在弯/坡最大的站点） | 记录实测曲率与坡度 |
| material_mix | 白线 + 土肩/碎石干扰带 + 蓝线 | 干扰材质不得被判成线 |

关键约束（方案 §4.2）：几何是**真实可见表面**——线用与地图相同的
`italy_road_markings_*` 贴花材质，不是"突出路面的粗立方体"；一张场景里
五条线各自独立（实例可分辨），不依赖"整张沥青纹理内嵌漆线"。

校验证据：`verify_batch(line_evidence="appearance")`——贴花不被 annotation
标注时，用**经深度遮挡校验的投影 + RGB 外观**作证据（方案 §4.2 的替代路径），
同时并列输出 annotation 模式的对照结果，读的人能分清两种证据。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_controlled_scenes.py --out logs\\experiments\\t16_scenes
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402
from beamng_autopilot.experiments.auto_truth import (  # noqa: E402
    TRUTH_CONTRACT, VERIFIER_VERSION, frame_content_shas, palette_sha,
    verify_batch,
)

GENERATOR_VERSION = "controlled_scenes_v1"
#: 采集锚点（italy）：与现有采集/探针同一片区域，便于对照
ANCHOR = (729.63, 763.91)
#: 五个站点沿道路链的间隔（米）：远到互不进入对方画面
SITE_SPACING_M = 120.0

#: 地图自己的 DecalRoad 材质（从 levels/italy 的 items.level.json 统计得到，
#: 见 docs/T16_ORDER1_2_REPORT_20260927.md §5.2）：生成场景与地图同材质，
#: 不引入"目标域外"的外观。
MAT_PAVE = "italy_asphalt_overlay_light"
MAT_LINE_WHITE = "italy_road_markings_line_thin"
MAT_LINE_YELLOW = "italy_road_markings_line_thin_yellow"
MAT_LINE_BLUE = "italy_road_markings_line_thin_blue"
MAT_GRAVEL = "m_dirt_road_gravels"

LINE_WIDTH_M = 0.15
#: 车道半宽（米）：受控场景里"行驶车道"是生成出来的——线放在 ±LAT_LANE_HALF。
#: 不用地图路面的外缘：外缘常被边缘贴花覆盖（annotation 里是背景/非路面），
#: 真值点会落在铺装之外（实测 3.65 m 处 label=0）。车道内部才是可控的。
LAT_LANE_HALF = 1.8
#: 横向符号约定（与 `_line_nodes` 的 left2d 一致）：**正 = 车体左**。
#: 实测踩到：写成 -1 让"左线"跑到右侧，翻转审计按 role 投票直接报 CAMERA_FLIP。
GRAVEL_WIDTH_M = 2.6
OCCLUDER_MODEL = "pickup"
OCCLUDER_AHEAD_M = 8.0

#: 场景规格：每条线的 (材质, 横向偏移符号, role)。横向偏移按"相对路面中线的
#: 米数"给：车道宽的一半即车道分界。
SCENES: dict[str, dict] = {
    "known_line": {"lines": [("white", +1, "left"), ("yellow", -1, "right")]},
    "known_no_line": {"lines": []},
    "occluded_line": {"lines": [("white", +1, "left")], "occluder": True},
    "slope_curve": {"lines": [("white", +1, "left"), ("yellow", -1, "right")]},
    "material_mix": {"lines": [("white", +1, "left"), ("blue", -1, "right")],
                     "gravel": True},
}
_MAT = {"white": MAT_LINE_WHITE, "yellow": MAT_LINE_YELLOW,
        "blue": MAT_LINE_BLUE}


def _load_probe():
    """复用探针的相机/深度/预热实现（同仓库 scripts 模块，不复制一份）。

    载入后立刻校验接口存在：实测踩到——探针的相机常量是在函数内 import 的，
    模块级没有，于是**游戏启动+场景加载全跑完**才在采帧处报 AttributeError，
    白花一次启动。这里提前失败。
    """
    spec = importlib.util.spec_from_file_location(
        "m5_auto_truth_probe", ROOT / "scripts" / "m5_auto_truth_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_auto_truth_probe"] = mod
    spec.loader.exec_module(mod)
    needed = ("_camera_dir_arg", "_buffers_ready", "_tech_palette",
              "_runtime_port", "_runtime_home", "to_label")
    missing = [n for n in needed if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(f"探针模块缺少接口：{missing}（接线漂移，先修这里）")
    return mod


def _load_truth_export():
    """复用真值点导出的 road network 工具（同一份搜索/插值口径）。"""
    spec = importlib.util.spec_from_file_location(
        "m5_tech_truth_points", ROOT / "scripts" / "m5_tech_truth_points.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_tech_truth_points"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


# ── 站点几何（沿 road network 走）─────────────────────────────────────────────
def _walk_chain(edges, i0, t0, step_dir, *, max_hops=600):
    """沿一条 DecalRoad 的边链走到端点，返回 ``(i, t, 走过的米数)``。

    实测：italy 的一条 DecalRoad 只有 ~250 m（单条贴花路），所以站点规划必须
    先知道链有多长、再决定间距，而不是假定"想走多远就有多远"。
    """
    i, t, walked, hops = i0, float(t0), 0.0, 0
    while hops < max_hops and 0 <= i < len(edges) - 1:
        a = np.asarray(edges[i]["middle"], dtype=float)
        b = np.asarray(edges[i + 1]["middle"], dtype=float)
        seg = float(np.linalg.norm(b[:2] - a[:2]))
        if seg < 1e-6:
            i += step_dir
            t = 0.0 if step_dir > 0 else 1.0
            hops += 1
            continue
        remaining = seg * (1.0 - t if step_dir > 0 else t)
        walked += remaining
        t = 1.0 if step_dir > 0 else 0.0
        i += step_dir
        t = 0.0 if step_dir > 0 else 1.0
        hops += 1
    if not (0 <= i < len(edges) - 1):
        # 走过头了：回退一格，位置取"走行方向的最后一端"
        # （正向 = 末段终点 t=1；反向 = 首段起点 t=0）
        return i - step_dir, (0.0 if step_dir < 0 else 1.0), walked
    return i, t, walked


def plan_sites(conn, *, n_sites: int, spacing_m: float) -> list[dict]:
    """沿车辆所在道路链布站：先定位链起点，再按可达长度自适应间距。

    返回每站的几何与生成参数；链太短（连最小间距都放不下）时报错，不返回
    不足的站点（"五类场景"缺一类就不是完成）。
    """
    texp = _load_truth_export()
    st = conn.get_state()
    pos = np.asarray(st.pos, dtype=float)
    fwd = np.array([math.cos(float(st.heading)), math.sin(float(st.heading)), 0.0])
    with conn.io_lock:
        roads = conn.bng.scenario.get_road_network(include_edges=True,
                                                  drivable_only=True)
    found = texp._nearest_edge(roads, pos)
    if found is None:
        raise RuntimeError("no drivable road edge near the ego")
    rid, meta, i0, t0, edges = found

    def lat_of(p_xy) -> float:
        left2d = np.array([-fwd[1], fwd[0]])
        return float((np.asarray(p_xy, dtype=float)[:2] - pos[:2]) @ left2d)

    row_a, row_b = edges[i0], edges[i0 + 1]
    L0 = texp._interp(row_a["left"], row_b["left"], t0)
    R0 = texp._interp(row_a["right"], row_b["right"], t0)
    lat_left, lat_right = lat_of(L0), lat_of(R0)
    if lat_left < lat_right:
        lat_left, lat_right = lat_right, lat_left
    half = 0.5 * float(lat_left - lat_right)
    ab = (np.asarray(edges[i0 + 1]["middle"], dtype=float)[:2]
          - np.asarray(edges[i0]["middle"], dtype=float)[:2])
    step_dir = 1 if float(ab @ fwd[:2]) >= 0 else -1

    # 1) 先走到链的**起点**（反向），再正向量出全长
    i_s, t_s, _back = _walk_chain(edges, i0, t0, -step_dir)
    _i_e, _t_e, total_len = _walk_chain(edges, i_s, t_s, step_dir)
    span = float(total_len)
    if span < 60.0:
        raise RuntimeError(f"道路链只有 {span:.0f} m：放不下 {n_sites} 个站点")
    spacing = min(float(spacing_m), max(30.0, span / max(1, int(n_sites) - 1)))
    if span / max(1, int(n_sites) - 1) < 30.0:
        raise RuntimeError(f"道路链 {span:.0f} m 放不下 {n_sites} 个站点"
                           f"（最小间距 30 m）")

    # 2) 从起点正向布站：每 spacing 米一个
    sites: list[dict] = []
    i, t, walked, hops = i_s, float(t_s), 0.0, 0
    next_station = 0.0
    while len(sites) < int(n_sites) and hops < 600 and 0 <= i < len(edges) - 1:
        a = np.asarray(edges[i]["middle"], dtype=float)
        b = np.asarray(edges[i + 1]["middle"], dtype=float)
        seg = float(np.linalg.norm(b[:2] - a[:2]))
        if seg < 1e-6:
            i += step_dir
            t = 0.0 if step_dir > 0 else 1.0
            hops += 1
            continue
        if walked + 1e-9 >= next_station:
            mid = texp._interp(a, b, t)
            # 方向取**路段方向**（含坡度、按走行方向），不用 3 m 前视：
            # 实测踩到——极短路段上前视被 clamp 到同一点，得到零向量，
            # 该站点直接失败（"desired camera direction must be nonzero"）。
            d3 = _unit((b - a) * float(step_dir))
            if float(np.linalg.norm(d3)) < 1e-9:
                hops += 1
                i += step_dir
                t = 0.0 if step_dir > 0 else 1.0
                continue
            sites.append({
                "station_m": round(walked, 2),
                "edge_index": int(i), "t": round(float(t), 4),
                "mid": [float(v) for v in mid],
                "dir": [float(v) for v in d3],
                "lat_left": lat_left, "lat_right": lat_right,
                "half_width_m": round(half, 3),
                "lanes_left": int(meta.get("lanesLeft") or 0),
                "lanes_right": int(meta.get("lanesRight") or 0),
                "road_id": str(rid),
                "chain_len_m": round(span, 1), "spacing_m": round(spacing, 1),
            })
            next_station += spacing
        remaining = seg * (1.0 - t if step_dir > 0 else t)
        step = max(0.5, min(next_station - walked, remaining))
        t += step_dir * step / seg
        walked += step
        while (step_dir > 0 and t >= 1.0) or (step_dir < 0 and t <= 0.0):
            t -= step_dir * 1.0
            i += step_dir
            hops += 1
            if not (0 <= i < len(edges) - 1):
                break
    if len(sites) < int(n_sites):
        raise RuntimeError(f"只走到 {len(sites)} 个站点（需要 {n_sites}）："
                           f"道路链 {span:.0f} m、间距 {spacing:.0f} m")
    # 每站的 z 剖面：路面沿链爬升/下降（实测 11.76% 坡上 34 m 差 4 m——
    # 用站点恒定 z 会让真值点整体飘出路面，投影距离 5–6 px）。
    for s_ in sites:
        prof: list[list[float]] = []
        i_, t_, acc = s_["edge_index"], float(s_["t"]), 0.0
        while acc <= 36.0 and 0 <= i_ < len(edges) - 1:
            a = np.asarray(edges[i_]["middle"], dtype=float)
            b = np.asarray(edges[i_ + 1]["middle"], dtype=float)
            seg = float(np.linalg.norm(b[:2] - a[:2]))
            if seg < 1e-6:
                i_ += step_dir
                t_ = 0.0 if step_dir > 0 else 1.0
                continue
            p_ = texp._interp(a, b, t_)
            prof.append([round(acc, 3), float(p_[2])])
            remain = seg * (1.0 - t_ if step_dir > 0 else t_)
            step_ = min(3.0, max(0.5, remain))
            t_ += step_dir * step_ / seg
            acc += step_
            while (step_dir > 0 and t_ >= 1.0) or (step_dir < 0 and t_ <= 0.0):
                t_ -= step_dir * 1.0
                i_ += step_dir
                if not (0 <= i_ < len(edges) - 1):
                    break
        s_["z_profile"] = prof
    for s_ in sites:
        i_, t_ = s_["edge_index"], s_["t"]
        a = np.asarray(edges[i_]["middle"], dtype=float)
        b = np.asarray(edges[i_ + 1]["middle"], dtype=float)
        p0 = texp._interp(a, b, max(0.0, t_ - 0.15))
        p1 = texp._interp(a, b, min(1.0, t_ + 0.15))
        d = p1 - p0
        s_["slope_pct"] = round(float(d[2] / max(1e-6, np.linalg.norm(d[:2]))) * 100, 2)
    for k in range(1, len(sites)):
        d0 = np.asarray(sites[k - 1]["dir"], dtype=float)[:2]
        d1 = np.asarray(sites[k]["dir"], dtype=float)[:2]
        cos = float(np.clip(d0 @ d1 / max(1e-9, np.linalg.norm(d0)
                                          * np.linalg.norm(d1)), -1, 1))
        sites[k]["curve_deg"] = round(math.degrees(math.acos(cos)), 2)
    sites[0]["curve_deg"] = 0.0
    return sites


def _z_at(site: dict, s_m: float) -> float:
    """沿里程的**路面高度**：优先用站点 z 剖面（road network 的真实高程），
    没有剖面时退回站点 z（平地近似）。"""
    prof = site.get("z_profile") or []
    if not prof:
        return float(np.asarray(site["mid"], dtype=float)[2])
    if s_m <= prof[0][0]:
        return float(prof[0][1])
    for k in range(1, len(prof)):
        if s_m <= prof[k][0]:
            a, b = prof[k - 1], prof[k]
            f = 0.0 if b[0] <= a[0] else (s_m - a[0]) / (b[0] - a[0])
            return float(a[1] + f * (b[1] - a[1]))
    return float(prof[-1][1])


def _line_nodes(site: dict, lat_m: float, *, span_m: float = 34.0,
                step_m: float = 1.0) -> list[list[float]]:
    """一条线/铺装的节点链：沿站点方向按 ``step_m`` 前推，横向 ``lat_m``，
    z 取该里程处的**路面高程**（坡道上必须跟着爬，否则真值点飘出路面）。"""
    mid = np.asarray(site["mid"], dtype=float)
    d = _unit(site["dir"])
    left2d = np.array([-d[1], d[0]])
    out: list[list[float]] = []
    s = 2.0
    while s <= span_m:
        p = mid + d * s + np.array([left2d[0], left2d[1], 0.0]) * lat_m
        out.append([float(p[0]), float(p[1]), _z_at(site, s)])
        s += step_m
    return out


def _line_truth(nodes: list[list[float]], role: str, *, per_m: float = 1.7) -> list[dict]:
    """线实例的真值点：沿节点链等距取样（class=2，带 role）。"""
    pts: list[dict] = []
    for i in range(len(nodes) - 1):
        a = np.asarray(nodes[i], dtype=float)
        b = np.asarray(nodes[i + 1], dtype=float)
        n = max(1, int(np.linalg.norm(b - a) / per_m))
        for k in range(n):
            p = a + (b - a) * (k / n)
            pts.append({"world": [float(v) for v in p], "class": 2, "role": role})
    return pts


# ── 场景构建 ─────────────────────────────────────────────────────────────────
def build_scenario(conn, sites: list[dict], *, scene_names: list[str]):
    """把五类站点全部生成进**一个** Scenario（一次加载，站点相距 120 m）。"""
    from beamngpy import Scenario, Vehicle
    from beamngpy.scenario.road import Road
    from beamngpy.misc.quat import angle_to_quat

    scen = Scenario("italy", "m5_controlled_scenes")
    site_of = {name: sites[k] for k, name in enumerate(scene_names)}
    record: dict = {"sites": {}, "line_instances": [], "vehicles": {}}

    def _add_road(material: str, rid: str, nodes: list, width: float,
                  priority: int) -> None:
        # interpolate=False：贴花按节点间的**直线段**渲染。实测踩到——用
        # Catmull-Rom 插值时渲染线在节点之间鼓出，与"节点连线"的真值点错开
        # 2 px 以上，判定全成 PROJECTION_MISMATCH。关掉插值后定义与渲染一致。
        road = Road(material=material, rid=rid, interpolate=False,
                    default_width=float(width), render_priority=int(priority))
        road.add_nodes(*[tuple(n) for n in nodes])
        scen.add_road(road)

    for k, name in enumerate(scene_names):
        spec = SCENES[name]
        site = sites[k]
        rec = {"station_m": site["station_m"], "road_id": site["road_id"],
               "mid": site["mid"], "dir": site["dir"],
               "half_width_m": site["half_width_m"],
               "slope_pct": site["slope_pct"], "curve_deg": site["curve_deg"],
               "line_generated": bool(spec["lines"]),
               "line_texture_embedded": False,
               "materials": {}, "lines": [], "gravel": None}
        # 车道边界（真值几何：铺装边界 = 路面边缘；线在车道分界上）
        for kind, sign, role in spec["lines"]:
            lat = sign * LAT_LANE_HALF
            nodes = _line_nodes(site, lat)
            rid = f"m5_line_{name}_{role}"
            _add_road(_MAT[kind], rid, nodes, LINE_WIDTH_M, 30 + k)
            rec["materials"][rid] = _MAT[kind]
            inst = {"id": rid, "role": role, "material": _MAT[kind],
                    "width_m": LINE_WIDTH_M, "lateral_m": round(lat, 3),
                    "nodes": nodes, "truth_points": _line_truth(nodes, role)}
            rec["lines"].append(inst)
            record["line_instances"].append({"scene": name, **inst})
        if spec.get("gravel"):
            nodes = _line_nodes(site, +(site["half_width_m"] + 1.6))
            rid = f"m5_gravel_{name}"
            _add_road(MAT_GRAVEL, rid, nodes, GRAVEL_WIDTH_M, 20 + k)
            rec["materials"][rid] = MAT_GRAVEL
            rec["gravel"] = {"id": rid, "material": MAT_GRAVEL,
                             "width_m": GRAVEL_WIDTH_M, "nodes": nodes}
        record["sites"][name] = rec

    # ego：停在第一个站点，朝站点方向
    site0 = sites[0]
    yaw_deg = -math.degrees(math.atan2(site0["dir"][1], site0["dir"][0])) - 90.0
    ego = Vehicle("ego", model="etk800", color="Red")
    scen.add_vehicle(ego, pos=tuple(site0["mid"]), rot_quat=angle_to_quat(
        (0.0, 0.0, yaw_deg)), cling=True)
    record["vehicles"]["ego"] = {"pos": site0["mid"], "yaw_deg": yaw_deg}

    # 遮挡车：只在 occluded_line 站点前方（真车模型，真实遮挡）
    occ_site = site_of.get("occluded_line")
    if occ_site is not None:
        mid = np.asarray(occ_site["mid"], dtype=float)
        d = _unit(occ_site["dir"])
        left2d = np.array([-d[1], d[0]])
        pos = mid + d * OCCLUDER_AHEAD_M + np.array(
            [left2d[0], left2d[1], 0.0]) * LAT_LANE_HALF   # 正=左，压在左线上
        yaw = -math.degrees(math.atan2(d[1], d[0])) - 90.0
        blocker = Vehicle("blocker", model=OCCLUDER_MODEL, color="Blue")
        scen.add_vehicle(blocker, pos=(float(pos[0]), float(pos[1]),
                                       float(mid[2])), rot_quat=angle_to_quat(
            (0.0, 0.0, yaw)), cling=True)
        record["vehicles"]["blocker"] = {
            "pos": [float(pos[0]), float(pos[1]), float(mid[2])],
            "ahead_m": OCCLUDER_AHEAD_M, "model": OCCLUDER_MODEL}
    return scen, record


def capture_site(conn, probe, *, site: dict, n_frames: int, width: int,
                 height: int, out_dir: Path, tag: str,
                 frame_truth: list | None = None) -> tuple:
    """在一个站点采帧（相机按站点方向对齐），返回 ``(frames, palette, meta)``。"""
    from beamngpy.sensors import Camera
    from beamng_autopilot_tech.providers import CAMERA_FOV_DEG, CAMERA_POS
    _d = np.asarray(site["dir"], dtype=float)
    if float(np.linalg.norm(_d)) < 1e-9:
        raise RuntimeError(f"站点 {site.get('station_m')} m 的方向是零向量："
                           "规划阶段就该修（road network 的极短路段）")
    cam_dir = probe._camera_dir_arg(_d, conn.get_state())
    name = f"m5_scene_{tag}"
    with conn.io_lock:
        cam = Camera(name, conn.bng, conn.vehicle, requested_update_time=0.05,
                     pos=CAMERA_POS, dir=cam_dir, up=(0.0, 0.0, 1.0),
                     resolution=(width, height),
                     field_of_view_y=CAMERA_FOV_DEG,
                     near_far_planes=(0.05, 150.0),
                     is_using_shared_memory=True, is_render_colours=True,
                     is_render_annotations=True, is_render_depth=True,
                     is_visualised=False, integer_depth=False,
                     postprocess_depth=False)
    frames: list[dict] = []
    palette = None
    try:
        for _try in range(6):
            with conn.io_lock:
                conn.bng.control.step(3)
            with conn.io_lock:
                data = cam.poll()
            ok, _why = probe._buffers_ready(data)
            if ok:
                break
        else:
            raise RuntimeError("相机缓冲预热失败")
        depth_raw_stats: dict = {}
        for i in range(max(1, int(n_frames))):
            if i:
                try:                       # 保持驻车，别让车滑
                    conn.vehicle.control(throttle=0.0, brake=1.0,
                                         parkingbrake=1.0)
                except Exception:                            # noqa: BLE001
                    pass
                with conn.io_lock:
                    conn.bng.control.step(2)
            with conn.io_lock:
                data = cam.poll()
            rgb = np.ascontiguousarray(np.asarray(data["colour"]), dtype=np.uint8)
            ann = np.ascontiguousarray(np.asarray(data["annotation"]), dtype=np.uint8)
            raw_depth = np.asarray(data["depth"])
            # 相机位姿：**用车辆状态独立重建**，不信 cam.get_direction()。
            # 实测（2026-09-27，五类场景 8 帧）：get_direction() 给出的方向
            # 在坡道上与渲染光轴差 ~2–6°，且随站点坡度/帧间车辆姿态变化
            # （11.5% 坡站点需要 -6° 修正，平地站点只要 -2°）——投影校验会被
            # 这个偏差污染成 PROJECTION_MISMATCH。车辆状态带 dir/up（含俯仰与
            # 侧倾），挂载偏移在车体系里已知，重建的位姿与渲染一致。
            st = conn.get_state()
            fwd_w = _unit(np.asarray(st.dir, dtype=float))
            up_w = _unit(np.asarray(st.up, dtype=float))
            left_w = _unit(np.cross(up_w, fwd_w))
            # 车体系：x=左, y=-前, z=上（与 Camera(dir=) 的参数约定一致，见
            # `m5_auto_truth_probe._camera_dir_arg` 的实测记录）
            from beamng_autopilot_tech.providers import CAMERA_POS as _CPOS
            cam_pos = [float(v) for v in (
                np.asarray(st.pos, dtype=float) + float(_CPOS[0]) * left_w
                + (-float(_CPOS[1])) * fwd_w + float(_CPOS[2]) * up_w)]
            rd = _unit(np.asarray(site["dir"], dtype=float))
            d_local = np.array([float(rd @ left_w), -float(rd @ fwd_w),
                                float(rd @ up_w)])
            fwd = _unit(d_local[0] * left_w + (-d_local[1]) * fwd_w
                        + d_local[2] * up_w)
            right = _unit(np.cross(fwd, up_w))
            up = np.cross(right, fwd)
            # 传感器读回（只作记录：两者的角度差就是上面那条发现的证据）
            try:
                _read_fwd = _unit(np.asarray(cam.get_direction(), dtype=float))
                _read_pos = [float(v) for v in cam.get_position()]
            except Exception:                                 # noqa: BLE001
                _read_fwd, _read_pos = None, None
            cam_dict = {"name": "front_main", "width": width, "height": height,
                        "fov_y_deg": float(CAMERA_FOV_DEG),
                        "pose": {"pos": cam_pos,
                                 "basis": {"right": [float(v) for v in right],
                                           "fwd": [float(v) for v in fwd],
                                           "up": [float(v) for v in up]}}}
            if palette is None:
                palette = probe._tech_palette(conn, ann)
            label = np.ascontiguousarray(probe.to_label(
                ann, road_colors=palette["road"],
                line_colors=palette["line"]), dtype=np.uint8)
            fid = f"{tag}_{i:05d}"
            src_sha, lab_sha = frame_content_shas(rgb, label)
            if not depth_raw_stats:
                depth_raw_stats = {"dtype": str(raw_depth.dtype),
                                   "min": float(np.nanmin(raw_depth)),
                                   "median": float(np.nanmedian(raw_depth)),
                                   "max": float(np.nanmax(raw_depth))}
            # 深度转米：遮挡判据要米，而原始缓冲是 uint8 自定义映射。用
            # **真值点几何**标定（与探针同一实现），不可用才退回探针的换算。
            _cal = probe._calibrate_depth(
                raw_depth, cam_dict,
                [dict(p) for p in (frame_truth or [])])
            if _cal.get("n", 0) >= 3 and _cal.get("a") is not None:
                depth_m = (float(_cal["a"]) * np.asarray(raw_depth, dtype=float)
                           + float(_cal["b"]))
            else:
                depth_m = probe._depth_to_meters(raw_depth, near=0.05, far=150.0,
                                                 mode="ndc")
            frames.append({"frame_id": fid, "timestamp": 1000.0 + i * 0.05,
                           "channel_ids": {"rgb": fid, "annotation": fid,
                                           "depth": fid},
                           "camera": cam_dict, "rgb": rgb, "annotation": ann,
                           "depth": np.ascontiguousarray(depth_m, dtype=np.float32),
                           "label": label,
                           "label_sha": lab_sha, "source_image_sha": src_sha,
                           "truth_points": [],
                           "camera_readback": {
                               "pos": _read_pos,
                               "fwd": (None if _read_fwd is None
                                       else [float(v) for v in _read_fwd]),
                               "dir_gap_deg": (None if _read_fwd is None else
                                               round(float(np.degrees(np.arccos(
                                                   float(np.clip(fwd @ _read_fwd,
                                                                 -1, 1))))), 2)),
                               "depth_calibration": _cal}})
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            np.savez_compressed(
                out_dir / f"{tag}_{i:05d}.npz", rgb=fr["rgb"],
                annotation=fr["annotation"], depth_raw=np.asarray(fr["depth"]),
                label=fr["label"], camera_json=json.dumps(fr["camera"]))
    finally:
        try:
            with conn.io_lock:
                cam.remove()
        except Exception:                                    # noqa: BLE001
            pass
    return frames, palette, {"depth_raw": depth_raw_stats}


def run_scene(conn, probe, *, name: str, site: dict, rec: dict, truth_points: list,
              n_frames: int, width: int, height: int, out_dir: Path) -> dict:
    """一个站点的完整流程：teleport → 采帧 → 两种证据校验 → 记录。"""
    # 站点对齐的朝向（与生成时的 ego 朝向同一约定）
    heading = math.atan2(site["dir"][1], site["dir"][0])
    mid = site["mid"]
    conn.safe_teleport(float(mid[0]), float(mid[1]),
                       heading_deg=math.degrees(heading))
    # 驻车：坡道上不刹住，车会在两帧之间滑动——采帧时的姿态与我读到的
    # 状态不一致（异步旧帧），投影会差几个像素（实测坡站点 ~4° 的假修正）。
    try:
        conn.vehicle.control(throttle=0.0, brake=1.0, parkingbrake=1.0)
    except Exception:                                        # noqa: BLE001
        pass
    conn.step(6)
    _st = conn.get_state()
    _delta = float(np.hypot(np.asarray(_st.pos, dtype=float)[0] - float(mid[0]),
                            np.asarray(_st.pos, dtype=float)[1] - float(mid[1])))
    frames, palette, cam_meta = capture_site(
        conn, probe, site=site, n_frames=n_frames, width=width, height=height,
        out_dir=out_dir, tag=name, frame_truth=truth_points)
    for fr in frames:
        fr["truth_points"] = [dict(p) for p in truth_points]
    batch = {
        "frames": frames,
        "scene": {"map": "italy", "segment": rec.get("road_id", ""),
                  "run_id": f"m5_controlled_{name}",
                  "scene_seed": 0, "game_version": "beamng_tech",
                  "renderer": "beamng_tech",
                  "line_generated": bool(rec.get("line_generated")),
                  "line_texture_embedded": False,
                  "shoulder_defined": bool(rec.get("gravel")),
                  "roles_defined": bool(rec.get("lines")),
                  "road_defined": True,
                  "label_source": "engine_annotation"},
        "palette": palette,
        "generator": {"name": "m5_controlled_scenes.py",
                      "version": GENERATOR_VERSION},
        "asset": {"map": "italy", "materials": dict(rec.get("materials") or {})},
        "run": {"id": f"m5_controlled_{name}", "scene_seed": 0,
                "game_version": "beamng_tech", "renderer": "beamng_tech"},
    }
    rep_app = verify_batch(batch, line_evidence="appearance")
    rep_ann = verify_batch(batch, line_evidence="annotation")
    return {
        "scene": name,
        "station_m": site["station_m"],
        "teleport_delta_m": round(_delta, 2),
        "ego_pos": [float(v) for v in np.asarray(_st.pos, dtype=float)[:3]],
        "curve_deg": site.get("curve_deg"), "slope_pct": site.get("slope_pct"),
        "half_width_m": site["half_width_m"],
        "n_truth_points": len(truth_points),
        "n_line_points": sum(1 for p in truth_points
                             if int(p.get("class") or 0) == 2),
        "truth_contract": TRUTH_CONTRACT, "verifier_version": VERIFIER_VERSION,
        "generated": rec,
        "camera_meta": cam_meta,
        "verification_appearance": rep_app,
        "verification_annotation": rep_ann,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attach", action="store_true", help="只连已跑的实例")
    ap.add_argument("--map", default="italy")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES),
                    help="要跑的场景（默认全部五类）")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--spacing-m", type=float, default=SITE_SPACING_M)
    args = ap.parse_args()

    probe = _load_probe()
    out = Path(args.out) if args.out else (
        config.LOGS_DIR / "experiments"
        / f"t16_scenes_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    conn = BeamNGConnector(str(args.map), "etk800",
                           port=probe._runtime_port(args),
                           home=probe._runtime_home(args))
    report: dict = {"generator": {"version": GENERATOR_VERSION,
                                  "script_sha16": _sha16(Path(__file__))},
                    "map": str(args.map), "anchor": list(ANCHOR),
                    "scenes": {}, "errors": []}
    conn.open(launch=not args.attach)
    try:
        # 规划站点要读 road network，需要先有**已加载的地图 + 一辆车**：
        # 新启动的实例只有空场景，attach 会失败——失败就自己加载一次
        # （italy 默认出生点），之后再换成我们生成的受控场景。
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:                                    # noqa: BLE001
            conn.load_scenario()
        if getattr(conn, "vehicle", None) is None:
            conn.load_scenario()
        # 站点：按场景顺序沿路取站（slope_curve 用弯/坡最大的站）
        sites = plan_sites(conn, n_sites=int(args.sites),
                           spacing_m=float(args.spacing_m))
        order = list(args.scenes)
        if "slope_curve" in order and len(sites) > 1:
            best = max(range(1, len(sites)),
                       key=lambda k: abs(sites[k].get("curve_deg") or 0)
                       + abs(sites[k].get("slope_pct") or 0))
            sites[best], sites[order.index("slope_curve")] = \
                sites[order.index("slope_curve")], sites[best]
        report["sites"] = sites
        scen, rec = build_scenario(conn, sites, scene_names=order)
        report["scenario"] = {"name": scen.name, "level": str(scen.level),
                              "vehicles": rec["vehicles"],
                              "materials": {n: rec["sites"][n]["materials"]
                                            for n in rec["sites"]}}
        with conn.io_lock:
            scen.make(conn.bng)
            conn.bng.scenario.load(scen)
            conn.bng.scenario.start()
        time.sleep(2.0)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:                                    # noqa: BLE001
            pass
        for k, name in enumerate(order):
            site = sites[k]
            rec_site = rec["sites"][name]
            truth: list[dict] = []
            for inst in rec_site["lines"]:
                truth += inst["truth_points"]
            # 路面点（class 1）：行驶车道中心，供路面通道对照
            mid = np.asarray(site["mid"], dtype=float)
            d = _unit(site["dir"])
            for s in (6.0, 12.0, 18.0):
                p = mid + d * s
                truth.append({"world": [float(p[0]), float(p[1]),
                                        _z_at(site, s)],
                              "class": 1, "role": None})
            try:
                res = run_scene(conn, probe, name=name, site=site, rec=rec_site,
                                truth_points=truth, n_frames=int(args.frames),
                                width=int(args.width), height=int(args.height),
                                out_dir=out / "frames")
                report["scenes"][name] = res
                ap_ = res["verification_appearance"]["stats"]["appearance"]
                print(f"[scenes] {name}: 线点 {res['n_line_points']} | "
                      f"外观 checked={ap_['checked']} like={ap_['line_like']} "
                      f"not_like={ap_['not_line_like']} occ={ap_['occluded']} | "
                      f"annotation 线类像素="
                      f"{res['verification_annotation']['stats']['valid_area']['line_px']}",
                      flush=True)
            except Exception as exc:                          # noqa: BLE001
                report["errors"].append({"scene": name,
                                         "error": f"{type(exc).__name__}: {exc}"})
                print(f"[scenes] {name} 失败：{type(exc).__name__}: {exc}",
                      flush=True)
        (out / "scene_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8")
        # 逐场景独立记录（凭证形态，方案 §4.3）
        for name, res in report["scenes"].items():
            blob = {"truth_contract": TRUTH_CONTRACT,
                    "label_source": "",     # 未经验证器验证前不声明来源
                    "generator": report["generator"],
                    "asset": {"map": str(args.map)},
                    "scene": res["generated"],
                    "verification": {
                        "verifier_version": res["verifier_version"],
                        "appearance": res["verification_appearance"]["stats"],
                        "annotation": res["verification_annotation"]["stats"]},
                    "camera": {"frames": res["camera_meta"]}}
            (out / f"scene_{name}.json").write_text(
                json.dumps(blob, indent=1, ensure_ascii=False, default=str),
                encoding="utf-8")
        print(f"[scenes] 产物 -> {out}")
    finally:
        try:
            conn.close()
        except Exception:                                    # noqa: BLE001
            pass
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
