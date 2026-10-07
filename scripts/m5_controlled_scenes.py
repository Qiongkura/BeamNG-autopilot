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
    TRUTH_CONTRACT, VERIFIER_VERSION, _rotate_camera_basis,
    fit_projection_alignment, frame_content_shas, line_distance_stats,
    line_evidence_coverage, palette_sha, verify_batch,
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

#: 地图档案：键 = 地图名，值 = 该地图自带的 DecalRoad 材质名。
#: 为什么需要：线材质必须来自**该地图自己的**贴花，否则生成物与地图不同材质，
#: 等于引入目标域外的外观（方案 §4.2）。italy 是历史批次用的地图，逐位保留；
#: west_coast_usa 的材质来自 2026-10-07 对 `content/levels/*.zip` 的离线扫描
#: （`items.level.json` 里 line_white 3112 次、line_yellow 1470 次、
#: line_dashed_short 215 次），用于第二张地图的复测（审查 E4）。
#: 其余地图（gridmap_v2/smallgrid/utah/east_coast_usa/jungle_rock_island）扫描
#: 未见成规模的白/黄线贴花，故不建档案；要用得先找材质再补档案。
MAP_PROFILES: dict[str, dict] = {
    "italy": {"pave": MAT_PAVE, "white": MAT_LINE_WHITE, "yellow": MAT_LINE_YELLOW,
              "blue": MAT_LINE_BLUE, "gravel": MAT_GRAVEL,
              "anchor": ANCHOR, "note": "历史批次地图（与既有采集同一片区域）"},
    "west_coast_usa": {"pave": "road_asphalt_2lane", "white": "line_white",
                       "yellow": "line_yellow", "blue": "line_dashed_short",
                       "gravel": "m_dirt_road_gravels", "anchor": None,
                       "note": "第二张地图（审查 E4）：材质为离线扫描所得"},
}


def apply_map_profile(map_name: str, *, overrides: dict | None = None) -> dict:
    """把模块级材质常量切到该地图的档案（返回实际生效的档案，供报告记录）。

    未建档案的地图：只有显式给了 ``--mat-*`` 才允许继续（否则材质名是 italy 的，
    生成的"线"会用错材质——宁可报错也不要静默生成错东西）。
    """
    global MAT_PAVE, MAT_LINE_WHITE, MAT_LINE_YELLOW, MAT_LINE_BLUE, MAT_GRAVEL, _MAT
    prof = dict(MAP_PROFILES.get(map_name) or {})
    ov = {k: v for k, v in (overrides or {}).items() if v}
    if not prof and not ov:
        raise SystemExit(
            f"地图 {map_name!r} 没有材质档案（已知：{sorted(MAP_PROFILES)}）；"
            f"请先用 --mat-white/--mat-yellow/--mat-pave 显式给出该地图的材质名")
    eff = {"pave": ov.get("pave") or prof.get("pave"),
           "white": ov.get("white") or prof.get("white"),
           "yellow": ov.get("yellow") or prof.get("yellow"),
           "blue": ov.get("blue") or prof.get("blue"),
           "gravel": ov.get("gravel") or prof.get("gravel"),
           "anchor": prof.get("anchor"),
           "note": (prof.get("note") or "") + ("（含 CLI 覆盖）" if ov else "")}
    missing = [k for k in ("pave", "white", "yellow", "gravel") if not eff[k]]
    if missing:
        raise SystemExit(f"地图 {map_name!r} 档案缺材质：{missing}")
    MAT_PAVE, MAT_LINE_WHITE = eff["pave"], eff["white"]
    MAT_LINE_YELLOW, MAT_LINE_BLUE = eff["yellow"], eff["blue"]
    MAT_GRAVEL = eff["gravel"]
    _MAT = {"white": MAT_LINE_WHITE, "yellow": MAT_LINE_YELLOW, "blue": MAT_LINE_BLUE}
    return eff

LINE_WIDTH_M = 0.15
#: 车道半宽（米）：受控场景里"行驶车道"是生成出来的——线放在 ±LAT_LANE_HALF。
#: 不用地图路面的外缘：外缘常被边缘贴花覆盖（annotation 里是背景/非路面），
#: 真值点会落在铺装之外（实测 3.65 m 处 label=0）。车道内部才是可控的。
LAT_LANE_HALF = 1.8
#: **实测的开发集线位约定**（2026-09-28，`scripts/m5_lane_geometry_measure.py`
#: 在 8 个 reviewed 开发目录上反投影人工标签）：池化 261k 线像素，
#: 质量集中在**自车右侧 −0.2…−0.7 m**（自车基本骑在线上——采集器沿 roadnet
#: 中线行驶），另一条线在 +1.5…+3 m；地面高度 ±0.5 m 的假设误差只让中位移
#: ~0.13 m，结论稳。上一轮自动真值用对称 ±1.8 m 生成，身份率 +0.028 但
#: **角色一致率掉到 0.609**——约定不一致是主嫌疑。生成按实测约定：
#: ``LINE_LATERAL_M`` 给"近线/远线"两个位置（车体系左为正，近线在右侧 = role
#: right），`--line-convention measured|symmetric` 切换以便做单因子对照。
LINE_LATERAL_M = {"near": -0.4, "far": +2.1}
#: 实测的**参考角色词表**（2026-09-28，8 个开发目录 174 个参考线实例，用探针
#: 自己的 `engine_lines`+`assign_roles` 在人工标签上统计）：
#: ``straddled 13.8% / near_right 28.7% / near_left 32.2% / far_left 12.1% /
#: far_right 13.2%``。对称 ±1.8 m 只覆盖 near_left+near_right（≈61%），
#: 缺 straddled（|lat|≤0.5，探针 `STRADDLE_M`）与 far_*（同侧第二条线）
#: ——上一轮角色一致率 0.609 的机械解释。
#: `measured` 约定按词表放线（位置 = 车体系左为正，夹在铺装内）：
#: 角色横向位置（米，车体系左为正；夹在**已验证覆盖**的 ±1.8 m 带内）。
LINE_ROLE_LATERAL_M = {"straddled": -0.4, "near_left": +1.8, "near_right": -1.8}
#: **按铺装宽度的相对线位**（`--line-convention relative`）：线位 = frac × 半宽。
#: 依据（2026-09-28 匹配几何扫描）：候选比参考线系统**外偏** 1.0–1.3 m
#: （far_left +1.32 / far_right −1.05），而假线平均路外占比 0.67——怀疑模型学到
#: 了"绝对 ±1.8 m"的先验，开发集窄路上就画到铺装外。相对比例让"线在铺装内的
#: 相对位置"成为可迁移先验（宽路上与绝对 ±1.8 m 等价：0.45×4.0 = 1.8）。
#: straddled 的比例压到 |lat| ≤ 0.45 m，保证仍判 straddled（STRADDLE_M=0.5）。
LINE_RELATIVE_FRAC = {"straddled": -0.10, "near_left": +0.45,
                      "near_right": -0.45}
#: **混合密度循环**（按站点序号轮换）。两个实测约束一起满足：
#: * **密度**：开发集人工标签是 **2.3 条/帧**（174 实例/76 帧）；单线批次
#:   （密度 1.0）把模型教成"每帧一条线"，身份率崩到 0.206（2026-09-28 实测）；
#: * **词表**：对称 ±1.8 m 只覆盖 near_left+near_right（61%），缺 straddled
#:   （13.8%，|lat|<=STRADDLE_M=0.5）；
#: * **覆盖**：单线站点覆盖 1.00、3–4 线站点 0.46–0.59（一个全局 yaw/pitch 标定
#:   对不齐多条线）——所以每站最多 2 条线。
#: 本循环平均密度 1.75 条/站，含 1/4 的 straddled 站，兼顾三者。
MIXED_DENSITY_CYCLE = (
    ("near_left", "near_right"),
    ("near_left", "near_right"),
    ("straddled",),
    ("near_left", "near_right"),
)
#: **多档线位**（米，车体系左为正）：按站点序号轮换档位，覆盖更宽的线位先验。
#: 依据（2026-09-29）：固定 ±1.8 m 的模型在开发集上候选-参考偏差 sd 1.19 m、
#: far_* 候选系统外偏 1.0–1.3 m -> 线位先验太窄。多档（1.2/1.8/2.4）让模型见到
#: 更宽的位置分布；每个档位仍夹在铺装内（|lat| <= half-0.3），窄路上自动收缩。
#: straddled 档固定 -0.4 m（|lat|<=STRADDLE_M=0.5 保证角色可判）。
LINE_TIER_M = (1.2, 1.8, 2.4)
#: **开发集线位分布驱动**（`--line-convention devdist`）：位置与角色配比按
#: 开发集**人工标签**实测来放，修 §16.1/§16.4 的"线位差 1–8 m"匹配层损失。
#:
#: 实测依据：
#: * 实例词表（174 实例，§2）：straddled 13.8% / near_left 32.2% /
#:   near_right 28.7% / far_left 12.1% / far_right 13.2%；
#: * 横向直方图（249k 近场线像素，§1）：主峰 **−0.75…0.00（41.8%）**，
#:   近线 ±1.5…2.5，远线 ±3…6（左侧远端 +3…+6 有 17.3%）；
#: * 密度 2.3 条/帧（174 实例/76 帧）。
#:
#: 6 步循环（每站 2 条 → 密度 2.0），角色必须与探针 `assign_roles` 的规则
#: 一致（**同侧按由近到远**定 near_/far_）——所以 far_* 只能出现在"同侧两条"
#: 的站上；`tests/test_controlled_scenes.py` 用探针自己的 `assign_roles`
#: 逐站校验声明的角色，避免"生成侧词表"与"评价侧词表"漂移。
LINE_DEVDIST_CYCLE = (
    (("near_left", +1.8), ("near_right", -1.8)),
    (("straddled", -0.4), ("near_left", +2.0)),
    (("near_left", +1.7), ("far_left", +4.0)),
    (("straddled", -0.3), ("near_right", -1.9)),
    (("near_right", -1.7), ("far_right", -4.0)),
    (("near_left", +2.2), ("far_left", +4.2)),
)
#: 远线（|lat| ≥ 3 m）需要**宽铺装**才放得下（生成时仍按 `|lat| <= half-0.3`
#: 夹紧）：窄路上远线会被夹回近线带，等于没生成 far_* 质量。`--anchor-min-half-width`
#: 让锚点挑选只取够宽的路段。
LINE_FAR_MIN_HALF_M = 4.6
#: **`pairfar`：同侧近+远成对扩量**（`--line-convention pairfar`）。依据
#: （2026-09-30，T16 §21）：近/远档配比臂在 6 seed 下**确认改善**身份率 +0.0214、
#: 精度 +0.1147、IoU +0.0828、路外假线 −32%——而该臂只有 **16 帧**成对数据。
#: 本节把它扩量看剂量-效应：每个有线站点都放**同侧一对**（内=近、外=远，
#: 满足探针 `assign_roles` 的"同侧由近到远"），两侧交替，6 步循环。
LINE_PAIRFAR_CYCLE = (
    (("near_left", +1.7), ("far_left", +4.0)),
    (("near_right", -1.7), ("far_right", -4.0)),
    (("near_left", +2.0), ("far_left", +4.2)),
    (("near_right", -1.9), ("far_right", -4.2)),
    (("near_left", +1.8), ("far_left", +3.8)),
    (("near_right", -1.8), ("far_right", -3.8)),
)
#: 成对扩量要求半宽 ≥ 4.5 m（远线 |lat| 最大 4.2 + 0.3 夹紧余量——
#: 测试用这条不变式校验常量，写 4.3 时当场被抓住）
LINE_PAIRFAR_MIN_HALF_M = 4.5
#: 为什么是 ±1.8 m 而不是实测的 ±2.1/−2.6 m：实测那些位置在同一锚点上
#: annotation 覆盖只有 0.54（对称 ±1.8 m 是 0.81）——**被标注的铺装带比 roadnet
#: 的半宽窄**，靠外的线落在标注之外。覆盖门是硬门，所以位置夹回 ±1.8 m；
#: 本批次的**新东西是 straddled**（自车骑线），这正是词表里此前完全缺失的角色。
LINE_CONVENTION = "symmetric"   # 由 --line-convention 覆盖
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
    # **结构负例**（T16 §16.1 的机制修复）：身份率的损失是模型在墙/护栏这类
    # "像线的非漆结构"上多画线。这类场景**不生成漆线**（line_generated=False），
    # 但在路侧放地图自带的静态物件（石墙 / 护栏）——它们在引擎 annotation 里是
    # **非路面、非线**（GUARD_RAIL/BACKGROUND），外观却是亮的、细长的，正是
    # "亮≠线"的对比样本。实机探针（2026-09-30）验证过：能加、能渲染、annotation
    # 非线非路面。
    "structure_negative": {"lines": [],
                           "structures": ["wall_stone", "guardrail"]},
}
_MAT = {"white": MAT_LINE_WHITE, "yellow": MAT_LINE_YELLOW,
        "blue": MAT_LINE_BLUE}

#: 结构负例用的地图自带静态物件：``shapeName`` 取自 levels/italy 的
#: items.level.json（石墙 442 处、护栏 149 处），**不引入域外资产**。
#: (shapeName, 横向符号(+左/-右), 距铺装边多少米)
STRUCT_SHAPES: dict[str, tuple] = {
    "wall_stone": ("/levels/italy/art/shapes/buildings/"
                   "italy_wall_stone_bricktop.dae", +1.0, 0.9),
    "guardrail": ("/levels/italy/art/shapes/buildings/"
                  "italy_guardrails_railing.dae", -1.0, 0.9),
    "jersey": ("art/shapes/objects/jerseybarrier_3m.dae", -1.0, 2.6),
}
#: 结构沿站点摆放：从 6 m 起、每 4 m 一个、共 7 个（覆盖 6–30 m，与线链同跨度）
STRUCT_FIRST_M = 6.0
STRUCT_STEP_M = 4.0
STRUCT_COUNT = 7


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


def read_road_network(conn) -> dict:
    """一次取回可行驶 road network（布站/挑锚点共用，避免重复查询）。"""
    with conn.io_lock:
        return conn.bng.scenario.get_road_network(include_edges=True,
                                                  drivable_only=True)


def pick_anchors(roads: dict, *, n_anchors: int,
                 min_sep_m: float = 150.0,
                 min_half_width_m: float = 0.0) -> list[dict]:
    """从 road network 里挑 **n 条互不相邻** 的道路作锚点（§4.4：先加路段）。

    每条道路取中段一行为锚点：位置 = 该行 middle，方向 = 相邻行方向。
    锚点间最小间距 ``min_sep_m``：同一段路的不同站点算同一"路段/场景族"，
    扩量要先加**路段**再加近邻帧。

    ``min_half_width_m > 0`` 时只取**铺装够宽**的路段（锚点行的 left/right
    横向间距的一半 ≥ 该值）：`devdist` 约定的远线（|lat| ≥ 3 m）在窄路上会被
    `|lat| <= half-0.3` 夹回近线带，等于没生成 far_* 质量。没有够宽的路段时
    返回空表（调用方如实报"挑不出"，不偷偷放宽）。
    """
    cands: list[dict] = []
    for rid, meta in (roads or {}).items():
        if not isinstance(meta, dict):
            continue
        edges = meta.get("edges")
        if not isinstance(edges, list) or len(edges) < 3:
            continue
        k = len(edges) // 2
        a = np.asarray(edges[k]["middle"], dtype=float)
        b = np.asarray(edges[min(k + 1, len(edges) - 1)]["middle"], dtype=float)
        d = _unit(b - a)
        if float(np.linalg.norm(d)) < 1e-9:
            continue
        half = None
        try:                       # 锚点行的铺装半宽（与 plan_sites_at 同口径）
            row = edges[k]
            mid2 = np.asarray(row["middle"], dtype=float)[:2]
            l2 = np.asarray(row["left"], dtype=float)[:2]
            r2 = np.asarray(row["right"], dtype=float)[:2]
            # 半宽 = 左右边界点间距的一半（不是"各自到中线的距离之差"——
            # 实测踩到：左右对称时那个式子恒等于 0，宽路全被过滤掉）
            half = 0.5 * float(np.linalg.norm(l2 - r2))
        except Exception:                                    # noqa: BLE001
            half = None
        if (float(min_half_width_m) > 0.0
                and (half is None or half < float(min_half_width_m))):
            continue
        cands.append({"road_id": str(rid), "pos": [float(v) for v in a],
                      "dir": [float(v) for v in d],
                      "n_rows": len(edges),
                      "half_width_m": (None if half is None
                                       else round(float(half), 3))})
    # 长的道路优先（给布站留空间），再按最小间距去重
    cands.sort(key=lambda c: -c["n_rows"])
    out: list[dict] = []
    for c in cands:
        if len(out) >= int(n_anchors):
            break
        if any(float(np.hypot(np.asarray(c["pos"], float)[0]
                              - np.asarray(o["pos"], float)[0],
                              np.asarray(c["pos"], float)[1]
                              - np.asarray(o["pos"], float)[1])) < float(min_sep_m)
               for o in out):
            continue
        out.append(c)
    return out


def plan_sites_at(texp, roads, pos, fwd, *, n_sites: int,
                  spacing_m: float) -> list[dict]:
    """在给定锚点（位置 + 朝向）所在道路链上布站。

    返回每站的几何与生成参数；链太短（连最小间距都放不下）时报错，不返回
    不足的站点（"五类场景"缺一类就不是完成）。
    """
    pos = np.asarray(pos, dtype=float)
    fwd = _unit(np.asarray(fwd, dtype=float))
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
                "anchor_pos": [float(v) for v in pos[:2]],
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


def plan_sites(conn, *, n_sites: int, spacing_m: float) -> list[dict]:
    """沿**车辆所在**道路链布站（旧入口，保留给单锚点场景）。"""
    texp = _load_truth_export()
    st = conn.get_state()
    pos = np.asarray(st.pos, dtype=float)
    fwd = np.array([math.cos(float(st.heading)), math.sin(float(st.heading)), 0.0])
    return plan_sites_at(texp, read_road_network(conn), pos, fwd,
                         n_sites=n_sites, spacing_m=spacing_m)


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
    # 从 4 m 起：2 m 处 0.2 m 的横向误差就是 ~4 px（最近处像素放大最大），
    # 而且那么近的点常在画面下缘之外；真值点只放在"可核对"的距离上。
    s = 4.0
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
def assign_scene_types(a_sites: list, types_order: list) -> list[str]:
    """给一个锚点的站点分配场景类型：按顺序轮转；slope_curve 换到弯/坡最大的站。

    **站点数少于类型数时不越界**（实测踩到：``--sites 3`` 时
    ``types_order.index("slope_curve") == 2`` 恰好等于长度，交换时 IndexError）。
    交换的是**站点**（类型按位置走），所以返回的 types 与 a_sites 一一对应。
    """
    types = [types_order[i % len(types_order)] for i in range(len(a_sites))]
    if "slope_curve" in types and len(a_sites) > 1:
        j = types.index("slope_curve")
        best = max(range(len(a_sites)),
                   key=lambda k: (abs(a_sites[k].get("curve_deg") or 0)
                                  + abs(a_sites[k].get("slope_pct") or 0)))
        if best != j:
            a_sites[j], a_sites[best] = a_sites[best], a_sites[j]
    return types


def build_scenario(conn, sites: list[dict], *, scene_types: list[str],
                   scene_names: list[str]):
    """把全部站点生成进**一个** Scenario（一次加载；扩量后站点分布在多条路段）。

    ``scene_types`` 是五类规格的键（决定生成什么），``scene_names`` 是每条站点的
    唯一名（报告/凭证的键）——扩量后同一类场景会在不同路段重复出现。
    """
    from beamngpy import Scenario, Vehicle
    from beamngpy.scenario.road import Road
    from beamngpy.scenario.scenario_object import ScenarioObject
    from beamngpy.misc.quat import angle_to_quat

    scen = Scenario("italy", "m5_controlled_scenes")
    site_of = {name: sites[k] for k, name in enumerate(scene_names)}
    record: dict = {"sites": {}, "line_instances": [], "vehicles": {}}
    occ_site = next((sites[k] for k, t in enumerate(scene_types)
                     if t == "occluded_line"), None)

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
        spec = SCENES[scene_types[k]]
        site = sites[k]
        rec = {"station_m": site["station_m"], "road_id": site["road_id"],
               "mid": site["mid"], "dir": site["dir"],
               "half_width_m": site["half_width_m"],
               "slope_pct": site["slope_pct"], "curve_deg": site["curve_deg"],
               "line_generated": bool(spec["lines"]),
               "line_texture_embedded": False,
               "materials": {}, "lines": [], "gravel": None}
        # 车道边界（真值几何：铺装边界 = 路面边缘；线在车道分界上）
        _specs = spec["lines"]
        _devlat: dict | None = None      # devdist/pairfar：按声明横向放线（见下）
        if LINE_CONVENTION in ("devdist", "pairfar") and spec["lines"]:
            # **开发集线位分布驱动 / 同侧成对扩量**：每站取循环里的 (role, lat) 对
            # （角色与探针 assign_roles 同规则，测试用探针自己校验）。
            _cyc = (LINE_PAIRFAR_CYCLE if LINE_CONVENTION == "pairfar"
                    else LINE_DEVDIST_CYCLE)
            _pair = _cyc[k % len(_cyc)]
            _devlat = {rn: float(lat) for rn, lat in _pair}
            _specs = [("yellow" if rn.endswith("_left") else "white",
                       1 if float(lat) >= 0 else -1, rn)
                      for rn, lat in _pair]
        elif LINE_CONVENTION in ("measured", "relative", "tiers") and spec["lines"]:
            # 按参考词表放线：位置来自 LINE_ROLE_TARGETS，夹在铺装内
            # （|lat| > half-0.3 的线会落在铺装外、annotation 不覆盖 ->
            # 资格门必失败），并去掉彼此 <1 m 的重复线。
            # 混合密度：按站点序号取一组角色（多数站 2 条，少数 straddled）
            _roles = MIXED_DENSITY_CYCLE[k % len(MIXED_DENSITY_CYCLE)]
            if LINE_CONVENTION == "tiers":
                # 多档线位：档位也按站点轮换（角色不变），横向在下面按档位算
                _tier = float(LINE_TIER_M[k % len(LINE_TIER_M)])
            _specs = [("yellow" if rn.endswith("_left") else "white",
                       1 if float(LINE_ROLE_LATERAL_M[rn]) >= 0 else -1, rn)
                      for rn in _roles]
        for kind, sign, role in _specs:
            if LINE_CONVENTION in ("devdist", "pairfar"):
                # 声明线位直接用；仍夹在铺装内（|lat| <= half-0.3），
                # **夹紧量记进实例**（`lat_clamped_m`）——远线在窄路上被夹回
                # 近线带时，报告里必须能看出来，否则"生成了 far_*"是假的。
                half = float(site["half_width_m"])
                lat = float((_devlat or {}).get(role, sign * LAT_LANE_HALF))
                _clamped = max(-(half - 0.3), min(half - 0.3, lat))
                _lat_clamp = round(_clamped - lat, 3)
                lat = _clamped
            elif LINE_CONVENTION == "tiers":
                half = float(site["half_width_m"])
                if role == "straddled":
                    lat = -0.4
                else:
                    lat = (_tier if role == "near_left" else -_tier)
                lat = max(-(half - 0.3), min(half - 0.3, lat))
            elif LINE_CONVENTION == "relative":
                half = float(site["half_width_m"])
                lat = float(LINE_RELATIVE_FRAC.get(role, -0.10)) * half
                lat = max(-(half - 0.3), min(half - 0.3, lat))
                if role == "straddled":
                    lat = max(-0.45, min(0.45, lat))
            elif LINE_CONVENTION == "measured":
                lat = float(LINE_ROLE_LATERAL_M.get(role, -0.4))
                lat = max(-(float(site["half_width_m"]) - 0.3),
                          min(float(site["half_width_m"]) - 0.3, lat))
            else:
                lat = sign * LAT_LANE_HALF
            nodes = _line_nodes(site, lat)
            rid = f"m5_line_{name}_{role}"
            _add_road(_MAT[kind], rid, nodes, LINE_WIDTH_M, 30 + k)
            rec["materials"][rid] = _MAT[kind]
            inst = {"id": rid, "role": role, "material": _MAT[kind],
                    "width_m": LINE_WIDTH_M, "lateral_m": round(lat, 3),
                    "nodes": nodes, "truth_points": _line_truth(nodes, role)}
            if LINE_CONVENTION in ("devdist", "pairfar"):
                # 声明线位 vs 实际线位：窄路上远线被夹回近线带时必须可见
                inst["declared_lateral_m"] = round(
                    float((_devlat or {}).get(role, lat)), 3)
                inst["lat_clamped_m"] = round(float(_lat_clamp), 3)
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

    # 遮挡车：只在第一个 occluded_line 站点前方（真车模型，真实遮挡）
    if occ_site is not None:
        mid = np.asarray(occ_site["mid"], dtype=float)
        d = _unit(occ_site["dir"])
        left2d = np.array([-d[1], d[0]])
        if LINE_CONVENTION in ("relative", "tiers"):
            _occ_lat = -0.4
        elif LINE_CONVENTION == "measured":
            _occ_lat = LINE_ROLE_LATERAL_M["straddled"]
        else:
            _occ_lat = LAT_LANE_HALF
        pos = mid + d * OCCLUDER_AHEAD_M + np.array(
            [left2d[0], left2d[1], 0.0]) * _occ_lat   # 正=左；压在该侧线上
        yaw = -math.degrees(math.atan2(d[1], d[0])) - 90.0
        blocker = Vehicle("blocker", model=OCCLUDER_MODEL, color="Blue")
        scen.add_vehicle(blocker, pos=(float(pos[0]), float(pos[1]),
                                       float(mid[2])), rot_quat=angle_to_quat(
            (0.0, 0.0, yaw)), cling=True)
        record["vehicles"]["blocker"] = {
            "pos": [float(pos[0]), float(pos[1]), float(mid[2])],
            "ahead_m": OCCLUDER_AHEAD_M, "model": OCCLUDER_MODEL}

    # 结构负例：路侧放地图自带的静态物件（石墙 / 护栏）。**不是**漆线，
    # annotation 里也不是线类——它们的作用是教"亮的细长结构 ≠ 线"。
    # 每个物件都记进 record，导出侧据此把"像线的外观"归因到已声明结构上。
    for k, name in enumerate(scene_names):
        spec = SCENES[scene_types[k]]
        site = sites[k]
        for skind in spec.get("structures") or []:
            if skind not in STRUCT_SHAPES:
                raise KeyError(f"未登记的结构种类 {skind!r}："
                               f"可选 {sorted(STRUCT_SHAPES)}")
            shape, sign, off = STRUCT_SHAPES[skind]
            mid = np.asarray(site["mid"], dtype=float)
            d = _unit(site["dir"])
            left2d = np.array([-d[1], d[0]])
            lat = float(sign) * (float(site["half_width_m"]) + float(off))
            yaw = -math.degrees(math.atan2(d[1], d[0])) - 90.0
            rec_s = record["sites"][name]
            rec_s["structures"] = []
            for j in range(int(STRUCT_COUNT)):
                s = STRUCT_FIRST_M + STRUCT_STEP_M * j
                p = mid + d * s + np.array([left2d[0], left2d[1], 0.0]) * lat
                z = _z_at(site, s)
                oid = f"m5_{skind}_{name}_{j}"
                scen.add_object(ScenarioObject(
                    oid, oid, "TSStatic",
                    pos=(float(p[0]), float(p[1]), float(z)),
                    scale=(1.0, 1.0, 1.0),
                    rot_quat=angle_to_quat((0.0, 0.0, yaw)),
                    shapeName=shape))
                rec_s["structures"].append({
                    "id": oid, "kind": skind, "shape": shape,
                    "lateral_m": round(lat, 3), "station_m": round(s, 2),
                    "pos": [float(p[0]), float(p[1]), float(z)]})
    return scen, record


def capture_site(conn, probe, *, site: dict, n_frames: int, width: int,
                 height: int, out_dir: Path, tag: str,
                 frame_truth: list | None = None,
                 step_m: float = 0.0) -> tuple:
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
    # 挡位/踏板读回：这一项以前不在证据里，于是"车留在 R 挡"只能靠人肉发现。
    _el = None
    try:
        from beamngpy.sensors import Electrics
        _el = Electrics()
        conn.vehicle.attach_sensor(f"t16_el_{tag}", _el)
    except Exception:                                        # noqa: BLE001
        _el = None
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
                try:                       # 保持驻车 + 前进挡，别让车滑/挂 R
                    conn.vehicle.control(throttle=0.0, brake=1.0,
                                         parkingbrake=1.0, gear=1)
                except Exception:                            # noqa: BLE001
                    pass
                # 逐帧沿站点方向前进 step_m（默认 0 = 原地多帧）：静止多帧是
                # **重复画面**（方案 §4.4 明确"先加路段再加近邻帧"），
                # 采序列才是不同的视角/位置。
                _step_m = float(step_m or 0.0)
                if _step_m > 0.0:
                    _d = _unit(np.asarray(site["dir"], dtype=float))
                    _p = np.asarray(st.pos, dtype=float) + _d * _step_m
                    try:
                        conn.safe_teleport(float(_p[0]), float(_p[1]),
                                           heading_deg=math.degrees(
                                               math.atan2(_d[1], _d[0])))
                        conn.vehicle.control(throttle=0.0, brake=1.0,
                                             parkingbrake=1.0, gear=1)
                    except Exception:                        # noqa: BLE001
                        pass
                with conn.io_lock:
                    conn.bng.control.step(2)
            # 异步旧帧检测（方案 §4.1）：poll 前后各读一次位姿，漂移超过阈值就
            # 说明这一帧渲染时的姿态与我记的不一致（实测：帧间标定差可达 5°，
            # 就是它）。有界重试，仍漂移就如实记录，不假装一致。
            drift_m = drift_deg = None
            for _try in range(4):
                st_pre = conn.get_state()
                with conn.io_lock:
                    data = cam.poll()
                st_post = conn.get_state()
                dp = float(np.linalg.norm(np.asarray(st_post.pos, dtype=float)
                                          - np.asarray(st_pre.pos, dtype=float)))
                d0 = _unit(np.asarray(st_pre.dir, dtype=float))
                d1 = _unit(np.asarray(st_post.dir, dtype=float))
                dd = float(np.degrees(np.arccos(
                    float(np.clip(d0 @ d1, -1.0, 1.0)))))
                drift_m, drift_deg = dp, dd
                if dp <= 0.02 and dd <= 0.05:
                    break
                with conn.io_lock:
                    conn.bng.control.step(1)
            rgb = np.ascontiguousarray(np.asarray(data["colour"]), dtype=np.uint8)
            ann = np.ascontiguousarray(np.asarray(data["annotation"]), dtype=np.uint8)
            raw_depth = np.asarray(data["depth"])
            # 相机位姿：**用车辆状态独立重建**，不信 cam.get_direction()。
            # 实测（2026-09-27，五类场景 8 帧）：get_direction() 给出的方向
            # 在坡道上与渲染光轴差 ~2–6°，且随站点坡度/帧间车辆姿态变化
            # （11.5% 坡站点需要 -6° 修正，平地站点只要 -2°）——投影校验会被
            # 这个偏差污染成 PROJECTION_MISMATCH。车辆状态带 dir/up（含俯仰与
            # 侧倾），挂载偏移在车体系里已知，重建的位姿与渲染一致。
            st = st_pre          # 用 poll **前**的姿态（最接近渲染时刻）
            fwd_w = _unit(np.asarray(st.dir, dtype=float))
            up_w = _unit(np.asarray(st.up, dtype=float))
            left_w = _unit(np.cross(up_w, fwd_w))
            # 车体系：x=左, y=-前, z=上（与 Camera(dir=) 的参数约定一致，见
            # `m5_auto_truth_probe._camera_dir_arg` 的实测记录）
            # 位置用**传感器读回**（权威）：实测（2026-09-27 扩量批次）我按
            # 车体系 + CAMERA_POS 重建的位置与读回差 ~0.36 m——10 m 处就是 ~4 px，
            # 正好是扩量后残差 3–5 px 的量级。方向仍用车辆状态（get_direction()
            # 不含俯仰/侧倾，见 `m5_auto_truth_probe` 的记录）：两个量各取权威来源。
            try:
                _pos_read = [float(v) for v in cam.get_position()]
            except Exception:                                 # noqa: BLE001
                from beamng_autopilot_tech.providers import CAMERA_POS as _CPOS
                _pos_read = [float(v) for v in (
                    np.asarray(st.pos, dtype=float) + float(_CPOS[0]) * left_w
                    + (-float(_CPOS[1])) * fwd_w + float(_CPOS[2]) * up_w)]
            cam_pos = _pos_read
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
            electrics = None
            if _el is not None:
                try:
                    conn.vehicle.sensors.poll()
                    d = _el.data
                    electrics = {k: d.get(k) for k in (
                        "gear", "gear_m", "reverse", "throttle", "brake",
                        "parkingbrake", "wheelspeed")}
                except Exception:                            # noqa: BLE001
                    electrics = None
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
                           "electrics": electrics,
                           "pose_drift": {"m": (None if drift_m is None
                                                else round(drift_m, 4)),
                                          "deg": (None if drift_deg is None
                                                  else round(drift_deg, 3))},
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
              n_frames: int, width: int, height: int, out_dir: Path,
              step_m: float = 0.0) -> dict:
    """一个站点的完整流程：teleport → 采帧 → 两种证据校验 → 记录。"""
    # 站点对齐的朝向（与生成时的 ego 朝向同一约定）
    heading = math.atan2(site["dir"][1], site["dir"][0])
    mid = site["mid"]
    conn.safe_teleport(float(mid[0]), float(mid[1]),
                       heading_deg=math.degrees(heading))
    # 驻车 + **显式挂前进挡**：实测（2026-09-27）场景里的玩家车 spawn 后挡位
    # 停在 R（NPC 是 N），teleport 与驻车指令都不改挡位——不显式选前进挡，
    # 留给人的就是一台挂着倒挡的车（用户实测反馈"卡在倒车档"）。
    # `control(gear=1)` 一发就从 R 变 P/N（同一次实测）。
    try:
        conn.vehicle.control(throttle=0.0, brake=1.0, parkingbrake=1.0, gear=1)
    except Exception:                                        # noqa: BLE001
        pass
    conn.step(6)
    _st = conn.get_state()
    _delta = float(np.hypot(np.asarray(_st.pos, dtype=float)[0] - float(mid[0]),
                            np.asarray(_st.pos, dtype=float)[1] - float(mid[1])))
    frames, palette, cam_meta = capture_site(
        conn, probe, site=site, n_frames=n_frames, width=width, height=height,
        out_dir=out_dir, tag=name, frame_truth=truth_points,
        step_m=float(step_m or 0.0))
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
    # 静态投影标定（渲染回读，落盘）：相机模型与渲染光轴之间的固定小角度差是
    # **标定常数**，不是真值放宽——拟合只在标定帧上做一次，参数与"校准前后
    # 残差"一起落盘；之后用它验证，残差才是几何一致性的度量。单看角度不可
    # 唯一辨识（沿路长直线对 yaw 退化，yaw 的残差会被 pitch 吸收），见
    # `auto_truth.ALIGN_YAW_RANGE_DEG` 注释。
    # 证据源按**站点实测**选择（方案 §4.3 逐通道资格）：引擎 annotation 的线类
    # 覆盖随路段变化（实测：有的路段上万像素，有的只有个位数）——覆盖够就用
    # annotation，不够就改用外观掩码（仍经深度遮挡校验），并把用的是哪种记进
    # 报告。固定用 annotation 会让"标注没覆盖的路段"全成假失败。
    _ann_px = sum(int(np.sum(np.asarray(fr.get("label")) == 2))
                  for fr in frames if fr.get("label") is not None)
    evidence = "annotation" if _ann_px >= 30 else "appearance"
    # 资格门（方案 §4.3）：证据**覆盖不足**的站点要隔离并写明原因，不能拿
    # "到最近证据像素的距离"当残差（那种站点上最近的像素根本不是这条线，
    # 实测会算出 10–14 px 的假残差）。
    _has_lines = bool(rec.get("line_generated"))
    if not _has_lines:
        # 无线场景（生成器声明未生成漆线）：线通道**没有主体** -> not_applicable，
        # 既不算合格也不算隔离（方案 §3.4/§3.5 的适用性语义）。
        eligibility = {"coverage": None, "threshold": 0.6, "eligible": None,
                       "status": "not_applicable", "evidence": evidence,
                       "why": ("generator declares line_generated=False: the "
                               "line channel has no subject here (judged by "
                               "the negative-line metrics instead)")}
    else:
        covs = [line_evidence_coverage(fr, radius_px=6, evidence=evidence)
                .get("coverage") for fr in frames]
        covs = [c for c in covs if c is not None]
        coverage = float(np.mean(covs)) if covs else None
        eligible = bool(coverage is not None and coverage >= 0.6)
        eligibility = {
            "coverage": (None if coverage is None else round(coverage, 4)),
            "threshold": 0.6, "eligible": eligible, "evidence": evidence,
            "status": "eligible" if eligible else "isolated",
            "why": ("" if eligible else
                    "line evidence does not cover the declared chain "
                    "(coverage below threshold): isolate this scene instead of "
                    "quoting a residual computed against unrelated pixels")}
    per_frame: list = []
    aligned: list = []
    for fr in frames:
        f1 = fit_projection_alignment([fr], evidence=evidence)
        per_frame.append({"frame_id": fr.get("frame_id"),
                          **{k: f1.get(k) for k in ("yaw_deg", "pitch_deg",
                                                    "before_px", "after_px",
                                                    "n", "status")}})
        if f1.get("status") == "measured":
            f2 = dict(fr)
            f2["camera"] = _rotate_camera_basis(fr["camera"], f1["yaw_deg"],
                                                f1["pitch_deg"])
            aligned.append(f2)
        else:
            aligned.append(dict(fr))
    # 交叉验证：用**其它帧**的标定套到本帧——检验标定不是对本帧过拟合。
    # 帧间位姿漂移大时这里会变差（异步旧帧），如实报告，不当通过。
    cross_px: list = []
    for i, fr in enumerate(frames):
        others = [g for j, g in enumerate(frames) if j != i]
        if not others:
            continue
        fo = fit_projection_alignment(others, evidence=evidence)
        if fo.get("status") != "measured":
            continue
        st = line_distance_stats(dict(
            fr, camera=_rotate_camera_basis(fr["camera"], fo["yaw_deg"],
                                            fo["pitch_deg"])),
            evidence=evidence)
        if st.get("mean_px") is not None:
            cross_px.append(round(float(st["mean_px"]), 3))
    fit = {"line_evidence_mode": evidence, "annotation_line_px": _ann_px,
           "per_frame": per_frame, "cross_validated_px": cross_px,
           "cross_validated_mean_px": (None if not cross_px
                                       else round(float(np.mean(cross_px)), 3)),
           "n_frames": len(frames),
           "note": ("静态投影标定（渲染回读）：逐帧拟合，角度不可唯一辨识；"
                    "验收看 after_px（逐帧）与 cross_validated_px（泛化）")}
    if cross_px:
        fit["before_px"] = round(float(np.mean([
            p["before_px"] for p in per_frame
            if p.get("before_px") is not None] or [float("nan")])), 3)
        fit["after_px"] = round(float(np.mean([
            p["after_px"] for p in per_frame
            if p.get("after_px") is not None] or [float("nan")])), 3)
        fit["status"] = "measured"
    else:
        fit["status"] = "unknown"
    batch_raw = dict(batch, frames=frames)
    batch_al = dict(batch, frames=aligned)
    rep_app = verify_batch(batch_al, line_evidence="appearance")
    rep_ann = verify_batch(batch_al, line_evidence="annotation")
    rep_ann_raw = verify_batch(batch_raw, line_evidence="annotation")
    dist_after = [line_distance_stats(f, evidence=evidence).get("mean_px")
                  for f in aligned]
    return {
        "camera_alignment": fit,
        "eligibility": eligibility,
        "line_evidence_mode": evidence,
        "line_distance_after_px": [d for d in dist_after if d is not None],
        "verification_annotation_raw": rep_ann_raw,
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
    ap.add_argument("--map", default="italy",
                    help=f"地图名；已建材质档案的：{sorted(MAP_PROFILES)}")
    ap.add_argument("--mat-white", default=None, help="覆盖白线材质名（探查新地图用）")
    ap.add_argument("--mat-yellow", default=None, help="覆盖黄线材质名")
    ap.add_argument("--mat-blue", default=None, help="覆盖蓝线材质名")
    ap.add_argument("--mat-pave", default=None, help="覆盖铺装材质名")
    ap.add_argument("--mat-gravel", default=None, help="覆盖碎石材质名")
    ap.add_argument("--scenes", nargs="*", default=list(SCENES),
                    help="要跑的场景（默认全部五类）")
    ap.add_argument("--frames", type=int, default=2,
                    help="每站帧数；配 --step-m 采**序列**（不同位置），否则是原地多帧")
    ap.add_argument("--step-m", type=float, default=0.0,
                    help="逐帧沿站点方向前进的米数（0 = 原地）")
    ap.add_argument("--line-convention",
                    choices=("symmetric", "measured", "relative", "tiers",
                             "devdist", "pairfar"),
                    default="symmetric",
                    help="线位约定：symmetric=±LAT_LANE_HALF（旧）；"
                         "measured=开发集实测绝对线位；relative=按铺装半宽"
                         "的相对线位；tiers=多档线位（1.2/1.8/2.4 m 轮换，"
                         "覆盖更宽的线位先验）")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sites", type=int, default=5,
                    help="每个锚点（路段）上的站点数；五类按顺序轮转")
    ap.add_argument("--anchors", type=int, default=1,
                    help="从 road network 里挑几条互不相邻的路段（§4.4：先加路段）")
    ap.add_argument("--anchor-min-sep-m", type=float, default=150.0)
    ap.add_argument("--anchor-min-half-width", type=float, default=0.0,
                    help="只取铺装半宽 >= 该值的锚点（devdist 的远线需要宽路；"
                         "0 = 不限）")
    ap.add_argument("--anchor-offset", type=int, default=0,
                    help="跳过前 K 个合格锚点（R3 要用**没进过训练/开发集**的"
                         "新路段；pick_anchors 是确定性的，不跳过就会重复采同一批）")
    ap.add_argument("--spacing-m", type=float, default=SITE_SPACING_M)
    args = ap.parse_args()

    global LINE_CONVENTION
    LINE_CONVENTION = str(args.line_convention)
    print(f"[scenes] 线位约定 = {LINE_CONVENTION}"
          + (f"（近线 {LINE_LATERAL_M['near']:+.1f} / 远线 "
             f"{LINE_LATERAL_M['far']:+.1f} m）"
             if LINE_CONVENTION == "measured" else ""), flush=True)
    profile = apply_map_profile(str(args.map), overrides={
        "white": args.mat_white, "yellow": args.mat_yellow, "blue": args.mat_blue,
        "pave": args.mat_pave, "gravel": args.mat_gravel})
    print(f"[scenes] 地图档案 {args.map}: 铺装={profile['pave']} 白={profile['white']} "
          f"黄={profile['yellow']} 蓝={profile['blue']} 碎石={profile['gravel']}",
          flush=True)
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
                    "map_profile": dict(profile),
                    "scenes": {}, "errors": []}
    # 收尾要用的"游戏进程基线"（见 finally）：只关**本次新起**的实例，
    # 用户自己的会话永远不碰（close_started_game 的所有权校验）
    from beamng_autopilot.experiments.collection import (  # noqa: E402
        close_started_game, game_pids)
    _pids_before = game_pids()
    _launched_after = time.time()
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
        # 站点：先挑锚点（互不相邻的路段），每个锚点沿其道路链布站；
        # 五类场景按顺序轮转，所以每个锚点都覆盖全部五类（§4.4：先加路段，
        # 再加近邻帧；帧数只是预算规划，不是质量门）。
        texp = _load_truth_export()
        roads = read_road_network(conn)
        types_order = list(args.scenes)
        _all_anchors = pick_anchors(
            roads, n_anchors=10_000,
            min_sep_m=float(args.anchor_min_sep_m),
            min_half_width_m=float(args.anchor_min_half_width))
        _off = max(0, int(args.anchor_offset))
        anchors = _all_anchors[_off:_off + int(args.anchors)]
        if _off or len(_all_anchors) > int(args.anchors):
            print(f"[scenes] 锚点：合格 {len(_all_anchors)} 个，跳过前 {_off} 个，"
                  f"本轮取 {len(anchors)} 个（offset 用于避开已进训练/开发集的"
                  f"路段）", flush=True)
        if not anchors:
            raise RuntimeError(f"跳过 {_off} 个后没有锚点可用"
                               f"（合格 {len(_all_anchors)} 个）")
        if not anchors:
            raise RuntimeError("road network 里挑不出锚点")
        report["anchors"] = anchors
        sites: list[dict] = []
        scene_types: list[str] = []
        scene_names: list[str] = []
        for ai, anchor in enumerate(anchors):
            try:
                a_sites = plan_sites_at(texp, roads, anchor["pos"],
                                        anchor["dir"],
                                        n_sites=int(args.sites),
                                        spacing_m=float(args.spacing_m))
            except Exception as exc:                          # noqa: BLE001
                report["errors"].append({"anchor": ai,
                                         "error": f"{type(exc).__name__}: {exc}"})
                print(f"[scenes] 锚点 {ai}（{anchor['road_id']}）布站失败：{exc}",
                      flush=True)
                continue
            # 场景类型分配（五类轮转；slope_curve 用该锚点弯/坡最大的站）
            a_types = assign_scene_types(a_sites, types_order)
            for si, st_ in enumerate(a_sites):
                sites.append(st_)
                scene_types.append(a_types[si])
                scene_names.append(f"{a_types[si]}_a{ai}s{si}")
        if not sites:
            raise RuntimeError("没有布出任何站点")
        report["sites"] = sites
        report["scene_types"] = scene_types
        scen, rec = build_scenario(conn, sites, scene_types=scene_types,
                                   scene_names=scene_names)
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
        for k, name in enumerate(scene_names):
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
                                out_dir=out / "frames",
                                step_m=float(args.step_m or 0.0))
                report["scenes"][name] = res
                ap_ = res["verification_appearance"]["stats"]["appearance"]
                el = res.get("eligibility") or {}
                print(f"[scenes] {name} [{scene_types[k]}]"
                      f"{'' if el.get('eligible') else ' 隔离'} "
                      f"覆盖={el.get('coverage')} | "
                      f"线点 {res['n_line_points']} | "
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
        # 批次验收汇总：**只在合格站点上**算（覆盖不足的已隔离并写明原因）
        elig = [r for r in report["scenes"].values()
                if (r.get("eligibility") or {}).get("eligible") is True]
        inelig = [r for r in report["scenes"].values()
                  if (r.get("eligibility") or {}).get("eligible") is False]
        na = [r for r in report["scenes"].values()
              if (r.get("eligibility") or {}).get("eligible") is None]
        aft = [ (r.get("camera_alignment") or {}).get("after_px")
                for r in elig]
        aft = [x for x in aft if x is not None]
        bef = [ (r.get("camera_alignment") or {}).get("before_px")
                for r in elig]
        bef = [x for x in bef if x is not None]
        cross = [ (r.get("camera_alignment") or {}).get("cross_validated_mean_px")
                  for r in elig]
        cross = [x for x in cross if x is not None]
        report["acceptance"] = {
            "sites_total": len(report["scenes"]),
            "sites_eligible": len(elig), "sites_isolated": len(inelig),
            "sites_not_applicable": len(na),
            "isolated": [{"scene": r.get("scene"),
                          "coverage": (r.get("eligibility") or {}).get("coverage"),
                          "why": (r.get("eligibility") or {}).get("why")}
                         for r in inelig],
            "before_px": {"n": len(bef),
                          "mean": (None if not bef
                                   else round(float(np.mean(bef)), 3))},
            "after_px": {"n": len(aft),
                         "mean": (None if not aft
                                  else round(float(np.mean(aft)), 3)),
                         "median": (None if not aft else round(
                             float(np.median(aft)), 3)),
                         "p95": (None if not aft else round(
                             float(np.percentile(aft, 95)), 3)),
                         "max": (None if not aft else round(float(max(aft)), 3)),
                         "within_2px": int(sum(1 for x in aft if x <= 2.0))},
            "cross_validated_mean_px": (None if not cross
                                        else round(float(np.mean(cross)), 3)),
            "criterion": ("eligible sites: calibrated mean projection "
                          "distance to the line evidence; isolated sites are "
                          "reported, not scored")}
        print(f"[scenes] 批次验收：合格 {len(elig)} / 隔离 {len(inelig)} / "
              f"N-A {len(na)}（共 {len(report['scenes'])} 站）| "
              f"校准后 mean={report['acceptance']['after_px']['mean']} "
              f"median={report['acceptance']['after_px']['median']} "
              f"≤2px={report['acceptance']['after_px']['within_2px']}/"
              f"{report['acceptance']['after_px']['n']} | 隔离 {len(inelig)}",
              flush=True)
        calib = {name: res.get("camera_alignment") for name, res in
                 report["scenes"].items()}
        (out / "line_projection_calibration.json").write_text(
            json.dumps({"generator": report["generator"],
                        "map": str(args.map),
                        "note": ("静态投影标定（渲染回读）：yaw/pitch 只在对齐"
                                 "意义下可辨识，验收看 after_px；"
                                 "复用时按 scene 名取"),
                        "scenes": calib}, indent=1, ensure_ascii=False,
                       default=str), encoding="utf-8")
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
        # **进程收尾**（2026-10-05 实测）：conn.close() 只关连接，游戏进程会留着
        # ——本轮连续四轮采集后机器上积了 9 个 BeamNG 实例，直接把
        # tests/test_seg_collect_unattended.py 的资源门打红（"采集前没有游戏进程"
        # 断言失败），也会像 2026-09-25 那次一样拖慢后续实验。复用采集侧同一
        # 助手：只杀"本次新起 + 创建时间可验证"的 pid，不碰别人的会话。
        _ga = close_started_game(_pids_before, launched_after=_launched_after)
        print(f"[scenes] 收尾关游戏：closed={_ga.get('closed')} "
              f"killed={_ga.get('killed')} "
              f"{_ga.get('reason') or _ga.get('note') or ''}")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
