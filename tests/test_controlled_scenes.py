"""`m5_controlled_scenes.py` 的纯逻辑测试（T16 §4.1/§4.2 路线 B）。

不连游戏：用假 conn 喂合成道路，验证
1. 五类场景规格齐备（有线/无线/遮挡/坡面曲线/材质混合）；
2. 线节点与真值点的几何（横向偏移方向、role、沿链等距）；
3. 站点规划沿道路前进且间隔正确；
4. `known_no_line` 不生成线、`occluded_line` 记录遮挡车、`material_mix` 有干扰带。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_controlled_scenes", ROOT / "scripts" / "m5_controlled_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_controlled_scenes"] = mod
    spec.loader.exec_module(mod)
    return mod


class _BootstrapConn:
    def __init__(self, *, attached=False, attach_error=False, empty_load=False):
        self.calls = []
        self.vehicle = object() if attached else None
        self.attach_error = attach_error
        self.empty_load = empty_load

    def open(self, *, launch):
        self.calls.append(("open", launch))

    def attach_vehicle(self, *, already_open):
        self.calls.append(("attach", already_open))
        if self.attach_error:
            raise RuntimeError("no active scenario")

    def load_scenario(self):
        self.calls.append(("load",))
        if not self.empty_load:
            self.vehicle = object()


def test_fresh_collection_loads_before_any_current_vehicle_query():
    m = _load()
    conn = _BootstrapConn(attach_error=True)
    m.prepare_initial_scene(conn, attach=False)
    assert conn.calls == [("open", True), ("load",)]
    assert conn.vehicle is not None


def test_explicit_attach_reuses_existing_vehicle():
    m = _load()
    conn = _BootstrapConn(attached=True)
    original = conn.vehicle
    m.prepare_initial_scene(conn, attach=True)
    assert conn.calls == [("open", False), ("attach", True)]
    assert conn.vehicle is original


def test_unavailable_explicit_attach_keeps_existing_load_fallback():
    m = _load()
    conn = _BootstrapConn(attach_error=True)
    m.prepare_initial_scene(conn, attach=True)
    assert conn.calls == [("open", False), ("attach", True), ("load",)]


def test_missing_vehicle_after_load_stops_before_scene_generation():
    import pytest
    m = _load()
    conn = _BootstrapConn(empty_load=True)
    with pytest.raises(RuntimeError, match="no connected vehicle"):
        m.prepare_initial_scene(conn, attach=False)


class _State:
    def __init__(self, pos, heading):
        self.pos = np.asarray(pos, dtype=float)
        self.heading = float(heading)
        self.dir = np.array([np.cos(heading), np.sin(heading), 0.0])


class _Scenario:
    def __init__(self, roads):
        self._roads = roads

    def get_road_network(self, **_kw):
        return self._roads


class _Bng:
    def __init__(self, roads):
        self.scenario = _Scenario(roads)


class _Conn:
    def __init__(self, roads, state):
        self.bng = _Bng(roads)
        self._state = state
        self.io_lock = threading.Lock()

    def get_state(self):
        return self._state


def _road(n_rows=60, step=10.0, width=4.0):
    rows = [{"middle": [k * step, 0.0, 1.0], "left": [k * step, width],
             "right": [k * step, -width]} for k in range(n_rows)]
    return {"roadA": {"edges": rows, "lanesLeft": 1, "lanesRight": 1}}


def test_five_scene_classes_are_specified():
    m = _load()
    assert set(m.SCENES) == {"known_line", "known_no_line", "occluded_line",
                             "slope_curve", "material_mix",
                             "structure_negative"}
    assert m.SCENES["known_no_line"]["lines"] == []
    assert m.SCENES["occluded_line"].get("occluder") is True
    assert m.SCENES["material_mix"].get("gravel") is True
    assert len(m.SCENES["known_line"]["lines"]) == 2
    # 线材质必须是地图自己的贴花材质（路线 B：真实可见表面，不是自造方块）
    for spec in m.SCENES.values():
        for kind, _sign, _role in spec["lines"]:
            assert kind in m._MAT
            assert "italy_road_markings" in m._MAT[kind]


def test_structure_negative_is_line_free_with_map_props():
    """结构负例：不生成漆线 + 放地图自带静态物件（石墙/护栏）。

    机制（T16 §16.1）：身份率的损失是模型在"像线的非漆结构"上多画线。
    这类场景必须**没有漆线**（否则不是负例），结构必须是**地图自己的资产**
    （shapeName 取自 levels/italy，不引入域外外观），且摆放参数要落在路侧
    而不是铺装上（压到铺装会挡住自车/真值）。
    """
    m = _load()
    spec = m.SCENES["structure_negative"]
    assert spec["lines"] == [], "结构负例不得生成漆线"
    assert spec["structures"] == ["wall_stone", "guardrail"]
    for kind in spec["structures"]:
        shape, sign, off = m.STRUCT_SHAPES[kind]
        assert "art/shapes" in shape or shape.endswith(".dae"), shape
        assert abs(float(sign)) == 1.0 and float(off) > 0.0
    # 跨度与线链一致（6–30 m），数量/步长自洽
    assert m.STRUCT_FIRST_M == 6.0 and m.STRUCT_STEP_M == 4.0
    assert m.STRUCT_COUNT == 7
    assert (m.STRUCT_FIRST_M + m.STRUCT_STEP_M * (m.STRUCT_COUNT - 1)) <= 34.0


def test_line_nodes_follow_the_site_direction_and_lateral_sign():
    m = _load()
    site = {"mid": [0.0, 0.0, 1.0], "dir": [1.0, 0.0, 0.0],
            "half_width_m": 4.0}
    left = m._line_nodes(site, +m.LAT_LANE_HALF)      # 正 = 车体左
    assert left and all(n[0] > 1.0 for n in left), left   # 都在车前
    # dir=+x 时左向是 +y：正偏移 -> y 为正
    assert all(n[1] > 0 for n in left), left
    right = m._line_nodes(site, -m.LAT_LANE_HALF)
    assert all(n[1] < 0 for n in right), right
    # 沿链等距（1.0 m 步长：节点密到"节点连线 = 渲染贴花"）
    d = [abs(left[i + 1][0] - left[i][0]) for i in range(len(left) - 1)]
    assert all(abs(x - 1.0) < 1e-6 for x in d), d


def test_scene_roles_follow_the_lateral_sign_convention():
    """role 的横向符号：**正 = 左**。实测踩到反号 -> 翻转审计报 CAMERA_FLIP。"""
    m = _load()
    for name, spec in m.SCENES.items():
        for _kind, sign, role in spec["lines"]:
            assert (sign > 0) == (role == "left"), (name, sign, role)


def test_line_truth_points_are_class2_with_role():
    m = _load()
    nodes = [[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]
    pts = m._line_truth(nodes, "left", per_m=2.0)
    assert len(pts) == 5
    assert all(p["class"] == 2 and p["role"] == "left" for p in pts)
    assert abs(pts[0]["world"][0] - 0.0) < 1e-9
    assert abs(pts[-1]["world"][0] - 8.0) < 1e-9


def test_plan_sites_walks_along_the_road_with_spacing():
    m = _load()
    conn = _Conn(_road(), _State([0.0, 0.0, 1.0], 0.0))
    sites = m.plan_sites(conn, n_sites=5, spacing_m=120.0)
    assert len(sites) == 5
    xs = [s["mid"][0] for s in sites]
    assert xs == sorted(xs), xs
    assert abs(xs[0] - 0.0) < 1e-6
    gaps = [abs(xs[i + 1] - xs[i]) for i in range(4)]
    assert all(abs(g - 120.0) < 25.0 for g in gaps), gaps
    assert all(s["half_width_m"] > 0 and s["road_id"] == "roadA" for s in sites)


def test_plan_sites_raises_when_the_chain_is_too_short():
    m = _load()
    conn = _Conn(_road(n_rows=3, step=10.0), _State([0.0, 0.0, 1.0], 0.0))
    try:
        m.plan_sites(conn, n_sites=5, spacing_m=120.0)
    except RuntimeError as exc:
        assert "站点" in str(exc)
    else:                                                  # pragma: no cover
        raise AssertionError("道路链太短时必须报错，不能返回不足的站点")


def test_pick_anchors_respects_min_separation_and_prefers_long_roads():
    """锚点 = 互不相邻的路段（§4.4：扩量先加路段）。"""
    m = _load()
    roads = {}
    for k in range(6):
        n = 12 if k % 2 == 0 else 4          # 长路优先
        rows = [{"middle": [k * 400.0 + i * 10.0, 0.0, 1.0],
                 "left": [k * 400.0 + i * 10.0, 4.0],
                 "right": [k * 400.0 + i * 10.0, -4.0]} for i in range(n)]
        roads[f"r{k}"] = {"edges": rows, "lanesLeft": 1, "lanesRight": 1}
    a = m.pick_anchors(roads, n_anchors=3, min_sep_m=150.0)
    assert len(a) == 3
    xs = [c["pos"][0] for c in a]
    assert all(abs(xs[i + 1] - xs[i]) >= 150.0 for i in range(len(xs) - 1)), xs
    # 长路（12 行）优先
    assert all(c["n_rows"] == 12 for c in a), a
    tight = m.pick_anchors(roads, n_anchors=6, min_sep_m=500.0)
    assert len(tight) == 3, tight             # 500 m 间距下只剩 3 条


def test_plan_sites_at_uses_the_given_anchor_and_records_it():
    m = _load()
    texp = m._load_truth_export()
    roads = _road(n_rows=40, step=10.0)
    sites = m.plan_sites_at(texp, roads, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                            n_sites=4, spacing_m=60.0)
    assert len(sites) == 4
    assert all(abs(s["anchor_pos"][0] - 0.0) < 1e-6 for s in sites)
    xs = [s["mid"][0] for s in sites]
    assert xs == sorted(xs) and len(set(xs)) == 4


def test_the_generator_selects_drive_gear_explicitly():
    """挡位接线（实测：玩家车 spawn 后停在 R，teleport/驻车指令都不改挡位）。

    这是源码级接线检查：漏了 `gear=1` 的后果只有在游戏里由人发现（用户实测
    反馈"卡在倒车档"），所以用测试卡住。
    """
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(
        encoding="utf-8")
    assert src.count("gear=1") >= 2, "run_scene 与采帧循环都要显式挂前进挡"
    assert '"electrics"' in src and "reverse" in src, "挡位/踏板要进证据"


def test_sequence_capture_is_wired():
    """`--step-m`：逐帧沿线前进采**序列**（静止多帧是重复画面）。

    源码级接线检查：漏了它，多帧站点会产出近似重复的样本（方案 §4.4 要求
    "先加路段、再加近邻帧"），而这一点在产物里不容易看出来。
    """
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(
        encoding="utf-8")
    assert "--step-m" in src and "step_m" in src
    assert "safe_teleport" in src, "序列帧要按站点方向前移（teleport）"


def test_no_undefined_args_reference_in_helpers():
    """辅助函数里不许引用 `args`——只有 main/cmd_* 有它。

    实测踩到（2026-09-28）：`capture_site` 里写了
    ``getattr(args, "step_m", 0.0)``，而该函数没有 `args` 参数，于是
    大批次 125 个站点全部 `NameError: name 'args' is not defined`、0 帧落盘；
    更糟的是它只在 ``--step-m > 0`` 时才触发，小批次（默认 0）完全看不出来。
    用 AST 静态卡住：任何引用 `args` 的函数必须有同名参数。
    """
    import ast
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        params = {a.arg for a in fn.args.args}
        params |= {a.arg for a in fn.args.kwonlyargs}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        # 函数内自己赋值也算合法（main 里就是 args = ap.parse_args()）
        assigned = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        uses = any(isinstance(n, ast.Name) and n.id == "args"
                   for n in ast.walk(fn))
        assert not uses or "args" in params or "args" in assigned, (
            f"{fn.name}() 引用了 args，但它既没有这个参数也没在函数内赋值"
            "（会在运行时 NameError）")


def test_assign_scene_types_rotates_and_does_not_index_over():
    """站点数 < 类型数时只轮转（实测踩到 --sites 3 时交换越界 IndexError）。"""
    m = _load()
    order = ["known_line", "known_no_line", "occluded_line", "slope_curve",
             "material_mix"]
    # 3 个站点：只有前 3 类，不越界
    sites3 = [{"curve_deg": 0.0, "slope_pct": 0.1},
              {"curve_deg": 2.0, "slope_pct": 9.0},
              {"curve_deg": 1.0, "slope_pct": 2.0}]
    types = m.assign_scene_types(sites3, order)
    assert types == order[:3], types
    # 5 个站点：slope_curve 换到弯/坡最大的那个站（交换站点，类型按位置）
    sites5 = [{"curve_deg": 0.0, "slope_pct": 0.0},
              {"curve_deg": 1.0, "slope_pct": 1.0},
              {"curve_deg": 1.0, "slope_pct": 1.0},
              {"curve_deg": 5.0, "slope_pct": 12.0},
              {"curve_deg": 1.0, "slope_pct": 1.0}]
    types5 = m.assign_scene_types(sites5, order)
    assert types5 == order, types5
    # 最弯/最陡的站现在在 slope_curve 的位置（index 3）
    assert sites5[3]["slope_pct"] == 12.0
    # 1 个站点：仍不崩
    assert m.assign_scene_types([{"curve_deg": 0.0, "slope_pct": 0.0}],
                                order) == [order[0]]


def test_measured_line_convention_matches_the_dev_set():
    """实测线位约定（开发集人工标签反投影，2026-09-28）：近线在自车右侧
    （-0.4 m，role right）、远线在左侧（+2.1 m，role left）。

    上一轮用对称 ±1.8 m 生成，身份率 +0.028 但角色一致率掉到 0.609——
    约定不一致是主嫌疑，所以把这个约定钉在测试里。
    """
    m = _load()
    assert m.LINE_LATERAL_M["near"] < 0 < m.LINE_LATERAL_M["far"]
    assert abs(m.LINE_LATERAL_M["near"]) < abs(m.LINE_LATERAL_M["far"])
    # 约定开关存在（symmetric = 旧行为，默认不变）
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(
        encoding="utf-8")
    assert "--line-convention" in src and "measured" in src
    assert m.LINE_CONVENTION in ("symmetric", "measured")


def test_measured_role_targets_cover_the_reference_vocabulary():
    """measured 约定要覆盖开发集的**参考角色词表**（实测 174 个实例：
    straddled 13.8% / near_right 28.7% / near_left 32.2% / far_left 12.1% /
    far_right 13.2%）。对称 ±1.8 m 只覆盖 near_left+near_right，缺 straddled
    与 far_*——这是上一轮角色一致率 0.609 的机械解释。
    """
    m = _load()
    roles = [r for spec in m.MIXED_DENSITY_CYCLE for r in spec]
    assert "straddled" in roles, "必须能生成 straddled（自车骑线）"
    assert "near_left" in roles and "near_right" in roles
    # far_* 是**有意不生成**的：实测 4 条线（含 far_left +4.2 m）让单相机标定
    # 对不齐所有线（逐线覆盖 ~0.5 -> 资格门全灭）且网格搜索代价 ×12。
    assert not any(r.startswith("far_") for r in roles)
    assert max(abs(v) for v in m.LINE_ROLE_LATERAL_M.values()) <= 3.0,         "横向跨度要收在标定能对齐的范围内"
    # straddled 的位置必须在探针的 STRADDLE_M(0.5) 内（否则判不成 straddled）
    lat = float(m.LINE_ROLE_LATERAL_M["straddled"])
    assert abs(lat) <= 0.5, lat
    # 每个位置都在一条合理车道范围内（生成时还会按铺装宽度夹紧）
    for _r, lat0 in m.LINE_ROLE_LATERAL_M.items():
        assert abs(float(lat0)) <= 6.0, (_r, lat0)


def test_measured_convention_keeps_the_line_free_control_empty():
    """measured 模式**不得**给 known_no_line 加线（它是无线对照）。

    实测踩到：词表展开写成了无条件覆盖，于是无线对照场景也画了 4 条线
    （线点 90、线类像素 5385）——控制组被破坏，负例包也就无从谈起。
    """
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(
        encoding="utf-8")
    # 词表展开只对**有线**场景生效（measured/relative/tiers 三套约定都带这条守卫）
    assert 'in ("measured", "relative", "tiers") and spec["lines"]' in src,         "词表展开必须只对有线场景生效"




def test_mixed_density_cycle_matches_the_dev_set():
    """混合密度循环：平均 1.75 条/站（开发集 2.3）、含 straddled 站、每站 ≤2 条。

    依据（2026-09-28 实测）：单线批次密度 1.0 -> 身份率崩到 0.206；对称 ±1.8 m
    只覆盖词表 61%（缺 straddled）；3-4 线站点覆盖 0.46-0.59 过不了门。
    """
    m = _load()
    cyc = m.MIXED_DENSITY_CYCLE
    lens = [len(x) for x in cyc]
    assert max(lens) <= 2, "每站最多 2 条线（多线过不了覆盖门）"
    assert 1.5 <= sum(lens) / len(lens) <= 2.5, lens
    roles = [r for spec in cyc for r in spec]
    assert "straddled" in roles, "必须含 straddled 站（补词表缺失的 13.8%）"
    assert "near_left" in roles and "near_right" in roles
    lat = m.LINE_ROLE_LATERAL_M
    assert abs(lat["straddled"]) <= 0.5, "straddled 位置要在 STRADDLE_M(0.5) 内"
    assert abs(lat["near_left"]) <= 1.8 and abs(lat["near_right"]) <= 1.8,         "两侧线夹在已验证覆盖的 ±1.8 m 带内"
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(encoding="utf-8")
    assert "MIXED_DENSITY_CYCLE[k % len(MIXED_DENSITY_CYCLE)]" in src


def test_relative_convention_is_scale_invariant():
    """relative 约定：线位 = frac × 半宽（宽路上与绝对 ±1.8 m 等价，窄路按比例收），
    straddled 仍 |lat|<=0.45（保证判 straddled，STRADDLE_M=0.5）。

    依据：候选比参考线系统外偏 1.0-1.3 m（far_*）、假线路外占比 0.67 —— 怀疑模型
    学成"绝对 ±1.8 m"先验；相对比例让"线在铺装内的相对位置"可迁移。
    """
    m = _load()
    assert set(m.LINE_RELATIVE_FRAC) == {"straddled", "near_left", "near_right"}
    assert abs(m.LINE_RELATIVE_FRAC["straddled"]) <= 0.45 / 2.0  # 4m 半宽下 <=0.45
    for half in (4.0, 3.0, 2.5):
        lat = m.LINE_RELATIVE_FRAC["near_left"] * half
        assert 0.6 <= lat <= half - 0.3 + 1e-9, (half, lat)
    assert abs(m.LINE_RELATIVE_FRAC["near_left"] * 4.0 - 1.8) < 1e-9,         "宽路上应与已验证覆盖的 ±1.8 m 等价"
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(encoding="utf-8")
    assert '"relative"' in src and "LINE_RELATIVE_FRAC" in src


def test_tier_convention_covers_wider_lateral_prior():
    """多档线位（1.2/1.8/2.4 m 轮换）：依据是固定 ±1.8 m 的候选-参考偏差 sd 1.19 m、
    far_* 候选系统外偏 1.0-1.3 m（线位先验太窄）。档位要夹在铺装内、角色循环不变。
    """
    m = _load()
    assert m.LINE_TIER_M == (1.2, 1.8, 2.4)
    assert min(m.LINE_TIER_M) < 1.8 < max(m.LINE_TIER_M), "要跨过旧的 ±1.8 m"
    src = (ROOT / "scripts" / "m5_controlled_scenes.py").read_text(encoding="utf-8")
    assert '"tiers"' in src and "LINE_TIER_M[k % len(LINE_TIER_M)]" in src
    assert 'if role == "straddled":' in src, "straddled 档要固定 -0.4（保角色可判）"


def test_devdist_cycle_matches_measured_vocabulary_and_probe_roles():
    """devdist 循环：实例配比贴近实测词表，且角色与**探针自己的**规则一致。

    实测（174 实例，docs §2）：straddled 13.8% / near_left 32.2% / near_right
    28.7% / far_left 12.1% / far_right 13.2%；横向主峰 −0.75…0.00（§1）。
    探针 `assign_roles` 的规则是"同侧由近到远定 near_/far_"——生成侧声明的角色
    必须满足它，否则生成词表与评价词表会漂移（角色一致率会被系统性扣分）。
    """
    m = _load()
    cyc = m.LINE_DEVDIST_CYCLE
    roles = [r for pair in cyc for r, _lat in pair]
    from collections import Counter
    c = Counter(roles)
    n = len(roles)
    assert n == 12, n
    share = {k: v / n for k, v in c.items()}
    measured = {"straddled": 0.138, "near_left": 0.322, "near_right": 0.287,
                "far_left": 0.121, "far_right": 0.132}
    for k, want in measured.items():
        assert abs(share.get(k, 0.0) - want) <= 0.06, (k, share, measured)
    assert set(share) == set(measured), share
    # 密度：每站 2 条（开发集 2.3 条/帧）
    assert all(len(p) == 2 for p in cyc)
    # 主峰：straddled 的位置必须落在实测主峰 −0.75…0.00 且 |lat|<=STRADDLE_M
    for pair in cyc:
        for r, lat in pair:
            if r == "straddled":
                assert -0.75 <= lat <= 0.0 and abs(lat) <= 0.5, (r, lat)
            if r.startswith("far_"):
                assert abs(lat) >= 3.0, (r, lat)
    # **用探针自己的 assign_roles 校验声明角色**
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "m5_marking_identity_probe",
        ROOT / "scripts" / "m5_marking_identity_probe.py")
    ip = _iu.module_from_spec(spec)
    sys.modules["m5_marking_identity_probe"] = ip
    spec.loader.exec_module(ip)
    for pair in cyc:
        # straddled 由 role_of(|lat|<=STRADDLE_M) 决定，near_/far_ 由
        # assign_roles（同侧由近到远）决定——两步都要对得上
        got = ip.assign_roles([{"role": ip.role_of(float(lat)),
                                "lat_m": float(lat)} for _r, lat in pair])
        for (r, lat), g in zip(pair, got):
            assert g["role"] == r, (pair, [x["role"] for x in got])


def test_pick_anchors_filters_by_half_width():
    """宽路过滤：devdist 的远线需要半宽够大的锚点；不够宽就不返回（不偷放）。"""
    m = _load()
    narrow = {"roadN": {"lanesLeft": 1, "lanesRight": 1, "edges": [
        {"middle": [k * 10.0, 0.0, 1.0], "left": [k * 10.0, 2.5],
         "right": [k * 10.0, -2.5]} for k in range(20)]}}
    wide = {"roadW": {"lanesLeft": 2, "lanesRight": 2, "edges": [
        {"middle": [500.0 + k * 10.0, 0.0, 1.0], "left": [500.0 + k * 10.0, 5.0],
         "right": [500.0 + k * 10.0, -5.0]} for k in range(20)]}}
    roads = {**narrow, **wide}
    a0 = m.pick_anchors(roads, n_anchors=5, min_sep_m=50.0)
    assert {a["road_id"] for a in a0} == {"roadN", "roadW"}
    assert all(a["half_width_m"] is not None for a in a0)
    a1 = m.pick_anchors(roads, n_anchors=5, min_sep_m=50.0,
                        min_half_width_m=4.6)
    assert [a["road_id"] for a in a1] == ["roadW"], a1
    assert a1[0]["half_width_m"] >= 4.6


def test_pairfar_cycle_is_all_same_side_pairs():
    """pairfar 扩量循环：每站都是**同侧近+远成对**，且角色与探针规则一致。

    依据（T16 §21）：近/远档配比臂（只有 16 帧成对数据）在 6 seed 下确认改善
    身份率 +0.0214、精度 +0.1147、IoU +0.0828、路外假线 −32%；扩量就是让每个
    有线站点都放一对。角色必须满足 `assign_roles` 的"同侧由近到远"，
    否则生成词表与评价词表漂移。
    """
    m = _load()
    cyc = m.LINE_PAIRFAR_CYCLE
    assert len(cyc) == 6 and all(len(p) == 2 for p in cyc)
    sides = set()
    for pair in cyc:
        roles = [r for r, _lat in pair]
        # 同一侧（角色后缀相同）、一近一远
        assert roles[0].split("_", 1)[1] == roles[1].split("_", 1)[1], pair
        assert {roles[0].split("_", 1)[0], roles[1].split("_", 1)[0]} == {
            "near", "far"}, pair
        lats = [abs(float(lat)) for _r, lat in pair]
        assert lats[0] < lats[1], pair          # 内近外远
        assert lats[1] >= 3.0, pair             # 远线档
        sides.add(roles[0].split("_", 1)[1])
    assert sides == {"left", "right"}           # 两侧都要有
    assert m.LINE_PAIRFAR_MIN_HALF_M >= max(
        abs(float(lat)) for _p in cyc for _r, lat in _p) + 0.3
    # 用探针自己的 role_of + assign_roles 校验
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "m5_marking_identity_probe",
        ROOT / "scripts" / "m5_marking_identity_probe.py")
    ip = _iu.module_from_spec(spec)
    sys.modules["m5_marking_identity_probe"] = ip
    spec.loader.exec_module(ip)
    for pair in cyc:
        got = ip.assign_roles([{"role": ip.role_of(float(lat)),
                                "lat_m": float(lat)} for _r, lat in pair])
        assert [g["role"] for g in got] == [r for r, _l in pair], pair


def test_anchor_offset_skips_used_routes():
    """`--anchor-offset` 跳过已用路段（R3 要用没进过训练/开发集的新路段）。

    `pick_anchors` 是确定性的：不跳过就会把 devdist/pairfar 批次采过的同一批
    宽路再采一遍（那批已进训练/开发集）。这里验证"取全部 -> 跳过 K -> 取 N"
    的切片语义，以及越界时返回空表（调用方如实报错，不静默重采）。
    """
    m = _load()
    roads = {}
    for i in range(6):
        x = 100.0 * i
        roads[f"road{i}"] = {"lanesLeft": 1, "lanesRight": 1, "edges": [
            {"middle": [x + k * 10.0, 0.0, 1.0], "left": [x + k * 10.0, 5.0],
             "right": [x + k * 10.0, -5.0]} for k in range(20)]}
    all_a = m.pick_anchors(roads, n_anchors=10_000, min_sep_m=50.0)
    assert len(all_a) == 6
    got = all_a[2:2 + 3]
    assert [a["road_id"] for a in got] == [a["road_id"] for a in all_a[2:5]]
    assert all_a[2:2 + 3] and not all_a[7:7 + 3]     # 越界 = 空表（调用方报错）


def test_ndc_depth_conversion_endpoints_and_monotonicity():
    """浮点深度标定帧的反解：0→near、1→far、单调递增（§9 仪表化的度量基础）。"""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_controlled_scenes", root / "scripts" / "m5_controlled_scenes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np

    near, far = 0.05, 150.0
    d = np.linspace(0.0, 1.0, 101)
    m = mod.ndc_depth_to_meters(d, near=near, far=far)
    assert m[0] == pytest.approx(near, rel=1e-6)
    assert m[-1] == pytest.approx(far, rel=1e-6)
    assert np.all(np.diff(m) > 0)
    # 中点不在几何中点（NDC 非线性），但必须是有限正数
    assert 0 < m[50] < far
    # 超界输入不产生负深度（饱和行为由调用方把关，这里只锁公式）
    assert mod.ndc_depth_to_meters(np.array([1.0]), near=near, far=far)[0] > 0


def test_depth_calibration_shot_is_wired_into_capture():
    """接线证明：capture_site 请求非共享内存标定帧并落盘 depthcal npz。"""
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text("utf-8")
    assert "is_using_shared_memory=False" in src
    assert "_depthcal.npz" in src
    assert "depth_ndc=" in src
    assert "ndc_depth_unavailable" in src   # 失败/跳过原样记录，不编造


def test_camera_local_arg_follows_vehicle_pitch_not_world_road_direction():
    """回归根因：挂载相机的局部光轴随车辆姿态旋转，不被每帧重瞄到 site.dir。"""
    m = _load()
    import math
    from types import SimpleNamespace

    cam_arg = np.array([0.0, -1.0, 0.0])  # Camera(dir=) 的车头前向
    flat = SimpleNamespace(dir=np.array([1.0, 0.0, 0.0]),
                           up=np.array([0.0, 0.0, 1.0]), heading=0.0)
    pitch = math.radians(10.0)
    tilted = SimpleNamespace(
        dir=np.array([math.cos(pitch), 0.0, math.sin(pitch)]),
        up=np.array([-math.sin(pitch), 0.0, math.cos(pitch)]),
        heading=0.0)
    f0 = m._camera_arg_to_world(cam_arg, flat)
    f1 = m._camera_arg_to_world(cam_arg, tilted)
    assert f0 == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-6)
    assert f1 == pytest.approx(tilted.dir, abs=1e-6)
    # 旧实现会返回平面 site.dir=(1,0,0)，把这 10° 车辆俯仰吞掉。
    assert abs(float(f1[2])) > 0.1


def test_capture_records_local_mount_direction_for_replay():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text("utf-8")
    assert "_camera_arg_to_world(cam_dir, st)" in src
    assert "mount_dir_local" in src


def test_ground_alignment_applies_to_measured_z_defect_maps_only():
    m = _load()
    # 集合只收有实测 z 偏差证据的地图（west_coast ~69m、east_coast ~28m）
    assert m.GROUND_ALIGN_MAPS == {"west_coast_usa", "east_coast_usa"}

    class Conn:
        def ground_z_at(self, x, y):
            return 213.5
    sites = [{"mid": [1.0, 2.0, 144.5],
              "z_profile": [[0.0, 144.5], [10.0, 145.5]]}]
    m._align_sites_to_ground(Conn(), sites, "west_coast_usa")
    assert sites[0]["mid"][2] == pytest.approx(213.5)
    assert sites[0]["z_profile"] == [[0.0, 213.5], [10.0, 214.5]]
    assert sites[0]["ground_alignment"]["status"] == "measured"
    eca = [{"mid": [1.0, 2.0, 26.07], "z_profile": [[0.0, 26.07]]}]
    m._align_sites_to_ground(Conn(), eca, "east_coast_usa")
    assert eca[0]["mid"][2] == pytest.approx(213.5)
    assert eca[0]["ground_alignment"]["status"] == "measured"
    other = [{"mid": [1.0, 2.0, 144.5], "z_profile": [[0.0, 144.5]]}]
    m._align_sites_to_ground(Conn(), other, "italy")
    assert "ground_alignment" not in other[0]


def test_run_scene_metadata_preserves_requested_map_and_palette_source():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text("utf-8")
    assert '"map": str(map_name)' in src
    # asset 块也要跟请求地图走：这里曾硬编码 "italy"，west_coast 批次的
    # verification.stats.asset.map 因此误记为 italy（凭证里靠导出侧
    # --map 才没被污染）。
    assert '"asset": {"map": str(map_name)' in src
    assert '"annotation_palette": palette' in src


def test_build_scenario_uses_requested_map_not_italy_default():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text("utf-8")
    assert 'Scenario(str(map_name), "m5_controlled_scenes")' in src
    # 2026-11-02 起每站 Scenario：参数表单从 (conn, sites, ...) 变为
    # (conn, [site], ...)，但地图仍必须显式穿传 args.map（不得回落 italy）。
    assert "map_name=str(args.map)" in src
    assert "build_scenario(" in src


def test_west_coast_rechecks_ground_after_scenario_load_and_can_reload():
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_controlled_scenes.py").read_text("utf-8")
    assert "sites_after_ground_alignment_reload" in src
    assert "post_load_ground_recheck" in src
    # 每站 Scenario 重载的变量名是 scen_k（重检下沉到每站循环后）。
    assert "conn.bng.scenario.load(scen_k)" in src


def test_east_coast_profile_uses_annotation_writable_material_and_width():
    m = _load()
    prof = m.MAP_PROFILES["east_coast_usa"]
    # 2026-10-18 探针定因后换材质/线宽：east 的 line_white/line_yellow 家族
    # opacityFactor 0.7 × bcf alpha 0.8–0.9，annotation pass 的 alphaRef=255
    # alpha 测试全像素剔除（ECA1/ECA2 0/5 根因）；唯一"annotation 写得进 +
    # 真值点居中"的替代是 line_yellow_double（alphaTest=True/alphaRef=0）。
    assert prof["white"] == "line_yellow_double"
    assert prof["yellow"] == "line_yellow_double"
    assert prof["line_width"] == 0.40  # 0.15 m 双条纹亚像素 appearance 0.43–0.67 挂
    assert prof["pave"] == "AsphaltRoad_variation_two_way"
    assert prof["gravel"] == "DirtRoad_variation_01"
    assert prof["blue"] is None  # 该地图无蓝色虚线材质，material_mix 不可用
    assert prof["anchor"] is None
    eff = m.apply_map_profile("east_coast_usa")
    assert eff["white"] == "line_yellow_double"
    assert eff["line_width"] == 0.40
    assert m.MAT_PAVE == "AsphaltRoad_variation_two_way"
    assert m.LINE_WIDTH_M == 0.40
    assert m._MAT == {"white": "line_yellow_double",
                      "yellow": "line_yellow_double", "blue": None}


def test_utah_profile_uses_annotation_writable_material_and_width():
    m = _load()
    prof = m.MAP_PROFILES["utah"]
    # 2026-10-18 Utah.zip 复扫 + 同款探针实测：line_white/line_yellow 同 east
    # alpha 缺陷（opacityFactor 0.7 × bcf alpha 0.90/1.0，alphaRef=255 全剔）；
    # line_yellow_double（alphaTest=True/alphaRef=0）实测唯一可写实线（近距
    # bin solid 4/4 vs 0/3、0/4，2 运行一致）。线宽 0.40m 与 east 同款纹理：
    # 0.20–0.30m 双条纹亚像素无漆、0.50m 远距回落，0.40m 探针 annotation 全写、
    # 导出门 paint_like 30/30×2。
    assert prof["white"] == "line_yellow_double"
    assert prof["yellow"] == "line_yellow_double"
    assert prof["line_width"] == 0.40
    assert prof["pave"] == "AsphaltRoad_variation_01"
    assert prof["gravel"] == "dirt_road_gravels"
    assert prof["blue"] is None  # 该地图无蓝色虚线材质，material_mix 不可用
    assert prof["anchor"] is None
    eff = m.apply_map_profile("utah")
    assert eff["white"] == "line_yellow_double"
    assert eff["line_width"] == 0.40
    assert m.MAT_PAVE == "AsphaltRoad_variation_01"
    assert m.MAT_GRAVEL == "dirt_road_gravels"
    assert m.LINE_WIDTH_M == 0.40
    assert m._MAT == {"white": "line_yellow_double",
                      "yellow": "line_yellow_double", "blue": None}
    # road z 与地形实测一致（探针站点 delta 0.02m，teleport_alignment 记录），
    # 无 z 埋地证据 → 不进 GROUND_ALIGN_MAPS（与实测 z 缺陷地图区分）
    assert "utah" not in m.GROUND_ALIGN_MAPS


def test_map_profile_line_width_defaults_to_contract_and_overrides():
    m = _load()
    # west/italy 档案没声明 line_width → 合同宽度 0.15 不变
    eff = m.apply_map_profile("west_coast_usa")
    assert eff["line_width"] == 0.15
    assert m.LINE_WIDTH_M == 0.15
    # CLI 覆盖（--line-width）压过档案
    eff = m.apply_map_profile("east_coast_usa", overrides={"line_width": 0.25})
    assert eff["line_width"] == 0.25
    assert m.LINE_WIDTH_M == 0.25
    # 无档案地图 + 显式材质时也回落到 0.15（utah 2026-10-18 已建档案 0.40，
    # 换一个仍未建档案的地图名验证回落语义）
    eff = m.apply_map_profile("smallgrid", overrides={
        "white": "line_white", "yellow": "line_yellow",
        "pave": "road_asphalt", "gravel": "m_dirt_road_gravels"})
    assert eff["line_width"] == 0.15


def test_east_line_material_choice_matches_map_alpha_admission():
    """档案选材必须满足 annotation 可写性（对真实资产回归）。

    annotation pass 的 alpha 测试语义：alphaTest 缺席时按 alphaRef 阈值
    剔像素（255 = 最终 alpha<255 全剔）。east 线材质必须 alphaTest=True
    或 alphaRef=0，否则 annotation 写不进（ECA1/ECA2 0/5 根因）。
    """
    zip_path = Path(r"G:\BeamNG.tech.v0.38.5.0\content\levels\east_coast_usa.zip")
    if not zip_path.exists():
        pytest.skip("BeamNG 地图 zip 不在本机（G: 盘）")
    import json as _json
    import zipfile as _zip
    with _zip.ZipFile(zip_path) as zf:
        mats = _json.loads(
            zf.read("levels/east_coast_usa/art/road/main.materials.json"))
    m = _load()
    for kind in ("white", "yellow"):
        name = m.MAP_PROFILES["east_coast_usa"][kind]
        d = mats[name]
        writable = bool(d.get("alphaTest")) or int(d.get("alphaRef", 255)) < 255
        assert writable, f"{name} annotation 不可写（alphaRef={d.get('alphaRef')})"
    # 被换下的材质正是反例：alphaRef=255 且 opacity 缩放 → annotation 全剔
    for name in ("line_white", "line_yellow"):
        d = mats[name]
        stage = (d.get("Stages") or [{}])[0]
        assert int(d.get("alphaRef", 255)) == 255 and not d.get("alphaTest")
        assert (stage.get("opacityFactor") or 1.0) < 1.0


def test_utah_line_material_choice_matches_map_alpha_admission():
    """utah 档案选材同款可写性回归（对真实 Utah.zip 资产）。

    Utah.zip 的 line_white/line_yellow 与 east 同缺陷（opacityFactor 0.7 ×
    bcf alpha 0.90/1.0，alphaRef=255 → annotation 全剔，探针近距 bin
    solid 0/3、0/4 实测印证）；line_yellow_double alphaTest=True/alphaRef=0
    是唯一可写实线（探针近距 bin solid 4/4 ×2 运行）。
    """
    zip_path = Path(r"G:\BeamNG.tech.v0.38.5.0\content\levels\Utah.zip")
    if not zip_path.exists():
        pytest.skip("BeamNG 地图 zip 不在本机（G: 盘）")
    import json as _json
    import zipfile as _zip
    with _zip.ZipFile(zip_path) as zf:
        mats = _json.loads(
            zf.read("levels/Utah/art/road/main.materials.json"))
    m = _load()
    for kind in ("white", "yellow"):
        name = m.MAP_PROFILES["utah"][kind]
        d = mats[name]
        if isinstance(d, list):
            d = d[0]
        writable = bool(d.get("alphaTest")) or int(d.get("alphaRef", 255)) < 255
        assert writable, f"{name} annotation 不可写（alphaRef={d.get('alphaRef')})"
    # 反例同 east：alphaRef=255 且 opacity 缩放 → annotation 全剔
    for name in ("line_white", "line_yellow"):
        d = mats[name]
        if isinstance(d, list):
            d = d[0]
        stage = (d.get("Stages") or [{}])[0]
        assert int(d.get("alphaRef", 255)) == 255 and not d.get("alphaTest")
        assert (stage.get("opacityFactor") or 1.0) < 1.0


def test_per_site_scenario_mode_covers_all_maps_after_east_migration():
    m = _load()
    # 2026-10-29 utah 探针判决：单 Scenario 10 条贴花路会让后创建的路渲染
    # 截断（idx9 的 22-34m bin annotation 全灭）；单站 2 路全跨渲染。
    # 2026-11-02 起全图统一每站 Scenario（east 的 a2 right 0.4216 同签名），
    # GROUND_ALIGN_MAPS 的加载后 ground 重检下沉到每站循环。
    for name in ("utah", "italy", "west_coast_usa", "east_coast_usa"):
        assert m.per_site_scenario_mode(name) is True
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "post_load_ground_recheck" in src
    assert "sites_after_ground_alignment_reload" in src


def test_weak_line_oversample_unpacks_frame_by_index():
    # 2026-11-02 实测：3 元帧（colour,label,annotation_raw）在
    # --weak-line-oversample 下触发 "too many values to unpack"（定长解包）。
    # 修复=按索引取 label（frame[1]），2/3 元帧都要能跑。
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_train_seg.py").read_text("utf-8")
    assert "for i, (_img, lb) in enumerate(train_frames)" not in src
    assert "for i, frame in enumerate(train_frames)" in src


def test_aug_scale_jitter_preserves_shape_and_label_domain():
    # 2026-11-02 多尺度增广代码臂（--aug-scale）：colour 线性/label 最近邻缩放，
    # 随机裁或边缘复制回原尺寸；label 值域必须保持 {0,1,2}（不能引入 255），
    # 形状、dtype 与三通道透传都必须稳定。
    import importlib.util
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_train_seg_aug", root / "scripts" / "m5_train_seg.py")
    m = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules["m5_train_seg_aug"] = m
    spec.loader.exec_module(m)

    rng = np.random.default_rng(0)
    colour = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    label = np.zeros((32, 48), np.uint8)
    label[14:18, :] = 2          # 一条水平线
    label[18:, :] = 1            # 下半路面
    annot = np.zeros((32, 48, 3), np.uint8)

    for _ in range(20):
        c2, l2, a2 = m._augment((colour, label, annot),
                                np.random.default_rng(1), scale_jitter=0.3)
        assert c2.shape == colour.shape and c2.dtype == np.uint8
        assert l2.shape == label.shape and l2.dtype == np.uint8
        assert set(np.unique(l2)).issubset({0, 1, 2})
        assert a2 is annot

    # 抖动确实改变内容（与关闭时相比）；关闭时保持原样（rng=0 仍会走其它增广，
    # 但 scale_jitter=0 不得进入缩放分支）。
    c_off, l_off = m._augment((colour, label), np.random.default_rng(7),
                              scale_jitter=0.0)[:2]
    assert c_off.shape == colour.shape and l_off.shape == label.shape


def test_ssl_pretrain_width_threads_into_encoder_keys():
    # 2026-11-02 容量臂：SSL 预训练必须支持 --width 且编码器键与
    # SegUNet(width) 严格一致（否则 --init 加载会 size mismatch）。
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "m5_ssl_pretrain.py").read_text("utf-8")
    assert '_AutoEncoder(width=float(args.width), arch=str(args.arch))' in src
    assert 'SegUNet(width=float(args.width), arch=str(args.arch))' in src
    assert 'ap.add_argument("--width"' in src
    # 通道数按 SegUNet 取整规则推导
    assert 'max(8, int(round(128 * float(width))))' in src


def test_segunet_strip_arch_forward_and_segmenter_reads_arch():
    # 2026-11-02 条带架构臂：strip 变体前向形状正确、参数键与 plain 不同、
    # Segmenter 必须从 checkpoint 的 arch_args.arch 建模型（否则 size mismatch）。
    import importlib.util
    import sys as _sys
    import numpy as np
    import torch

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "seg_mod_arch", root / "beamng_autopilot" / "vision" / "segmentation.py")
    m = importlib.util.module_from_spec(spec)
    _sys.modules["seg_mod_arch"] = m
    spec.loader.exec_module(m)

    x = torch.zeros(2, 3, 32, 48)
    plain = m.SegUNet(width=1.0, arch="plain")
    strip = m.SegUNet(width=1.0, arch="strip")
    assert plain(x).shape == (2, m.N_CLASSES, 32, 48)
    assert strip(x).shape == (2, m.N_CLASSES, 32, 48)
    assert set(plain.state_dict()) != set(strip.state_dict())
    # strip 块含 1×5 与 5×1 非对称核
    shapes = {tuple(v.shape[-2:]) for k, v in strip.state_dict().items()
              if k.startswith("e1.") and v.dim() == 4}
    assert (1, 5) in shapes and (5, 1) in shapes
    # 架构字段从 checkpoint 读出（源码断言：Segmenter 用 arch_args.arch 建模型）
    src = (root / "beamng_autopilot" / "vision" / "segmentation.py").read_text("utf-8")
    assert 'self.arch = str(arch_args.get("arch") or "plain")' in src
    train_src = (root / "scripts" / "m5_train_seg.py").read_text("utf-8")
    assert '"arch": str(getattr(model, "arch", "plain"))' in train_src
    assert 'ap.add_argument("--arch"' in train_src
    ssl_src = (root / "scripts" / "m5_ssl_pretrain.py").read_text("utf-8")
    assert '--arch' in ssl_src and 'arch=str(args.arch)' in ssl_src


def test_fsd_drive_tod_weather_wiring():
    # 2026-11-02 条件矩阵接线：入口 --tod/--weather-preset 必须到会话层，
    # 经 EnvironmentApi 应用并写入运行记录（失败也不静默）。
    root = Path(__file__).resolve().parents[1]
    entry = (root / "scripts" / "m5_fsd_drive.py").read_text("utf-8")
    assert 'ap.add_argument("--tod"' in entry
    assert 'ap.add_argument("--weather-preset"' in entry
    sess = (root / "beamng_autopilot" / "fsd_drive.py").read_text("utf-8")
    assert "conn.bng.env.set_tod(" in sess
    assert "conn.bng.env.set_weather_preset(" in sess
    assert '_run_record["environment"]' in sess
    # 顺序契约（2026-11-02 修订）：weather preset 的 diff 表可携带 timeOfDay，
    # 先设时刻再上天气会被预设覆盖——必须 weather -> tod，且应用后读回
    # get_tod 存证（条件格=引擎侧可验证）。
    wx_i = sess.index("conn.bng.env.set_weather_preset(")
    tod_i = sess.index("conn.bng.env.set_tod(")
    assert wx_i < tod_i, "必须先应用天气预设再设显式 TOD"
    # 预设过渡是动画（time 秒内逐帧插值回写）：立刻设 TOD 会被动画下一帧
    # 覆盖（矩阵 dawn_rain 实测请求 06:30 读回 09:56）。两者同时请求时必须
    # 等过渡结束再设时刻；get_tod 读回仍留作最终存证。
    assert "time.sleep(" in sess[wx_i:tod_i], \
        "天气->时刻之间必须等待过渡动画结束"
    assert '"order": "weather_then_tod"' in sess
    assert '"tod_after"' in sess and "get_tod()" in sess
    # 环境应用失败必须不静默且不崩溃：捕获要放宽到 Exception（预设名不
    # 存在时 beamngpy 等 ack 抛 TimeoutError，窄化 except 会崩掉整轮）。
    env_i = sess.index('_run_record["environment"] = {"tod": _tod_s')
    fail_i = sess.index('_run_record["environment"] = {"tod": _tod_s', env_i + 1)
    assert 'applied": False' in sess[fail_i:fail_i + 300]


# ---------------------------------------------------------------------------
# 2026-11-02 条件矩阵（③ 整车主线）：m5_fsd_conditions.py 的纯逻辑测试
# ---------------------------------------------------------------------------

def _load_conditions():
    spec = importlib.util.spec_from_file_location(
        "m5_fsd_conditions", ROOT / "scripts" / "m5_fsd_conditions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_fsd_conditions"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fsd_conditions_matrix_plan_shape():
    mod = _load_conditions()
    cells = mod.plan_cells(None, None)
    # 12 格 + 8 个严酷格重复 = 20 轮（预注册口径）
    assert len(cells) == 20
    # 预设名缺失时 rain/fog 全部不可跑且仍出现在计划里（不静默消失）
    runnable = [c for c in cells if c["runnable"]]
    assert {c["weather"] for c in runnable} == {"clear"}
    assert len(runnable) == 5          # 4 个 clear r0 + night_clear r1
    missing = [c for c in cells if not c["runnable"]]
    assert {c["weather"] for c in missing} == {"rain", "fog"}
    # 预设名解析后，rain/fog 格变为可跑
    cells2 = mod.plan_cells("rainy", "foggy")
    assert all(c["runnable"] for c in cells2)
    assert len(cells2) == 20
    # 严酷格 2 轮、其余 1 轮（预注册口径）
    from collections import Counter
    cnt = Counter((c["tod_name"], c["weather"]) for c in cells2)
    for key, n in cnt.items():
        assert n == (2 if key in mod.REPEAT_CELLS else 1)
    assert all(c["repeat"] in (0, 1) for c in cells2)


def test_fsd_conditions_build_run_args_threads_condition():
    mod = _load_conditions()
    cell = {"cell": "night_rain", "tod_name": "night", "tod": "21:30",
            "weather": "rain", "weather_preset": "rainy", "repeat": 0,
            "runnable": True}
    args = mod.build_run_args(cell, ROOT / "logs" / "x.json")
    assert args.tod == "21:30" and args.weather_preset == "rainy"
    assert args.map == mod.SCENARIO["map"]
    assert args.teleport == list(mod.SCENARIO["teleport"])
    assert args.goal == list(mod.SCENARIO["goal"])
    assert args.seconds == mod.SCENARIO["seconds"]
    assert args.seg_model == mod.SCENARIO["seg_model"]
    # FSD 合规路径由默认 namespace 保证（不许手抄漂移）
    assert args.lane_mode == "sensor" and args.strict is True
    assert not args.allow_legacy_map_lane


def _synth_hist(n=12, t_step=1.0):
    """合成遥测：感知线常在、无越线/冲出/倒车/急停，edge_over=0。"""
    hist = []
    for i in range(n):
        hist.append({
            "t": i * t_step, "speed": 4.0, "lane_sel": "sensor",
            "lane_src": "sensor_lane", "lane_paired": 1,
            "lat_left": 0.6, "lat_right": 1.2,
            "heading": 45.0, "route_bear": 45.0,
            "edge_over": 0.0, "reversing": 0, "stuck": 0,
            "emergency": 0, "level": "safe", "pos": [0.0, 0.0, 0.0],
        })
    return hist


_GOOD_SESSION = {
    "placed": True, "placement_rc": 0,
    "environment": {"tod": "21:30", "weather_preset": "rainy",
                    "applied": True, "order": "weather_then_tod",
                    "tod_after": "21:30:00"},
    "config": {"tod": "21:30", "weather_preset": "rainy"},
    "exit_code": 0,
}


def test_fsd_conditions_score_one_failclosed():
    from beamng_autopilot.eval import STATUS_UNKNOWN, assess_run, score_run
    mod = _load_conditions()
    hist = _synth_hist()
    row = mod.score_one(hist, dict(_GOOD_SESSION))
    # 无 fail-closed 触发时，状态必须等于直接评测的判定（不篡改）
    verdict = score_run(assess_run(hist, goal=tuple(mod.SCENARIO["goal"]),
                                   settle_s=8.0), require_goal=False)
    assert row["status"] == verdict["status"]
    assert row["placed"] and row["env_applied"]
    assert row["tod_after"] == "21:30:00"
    # 没放上车 -> UNKNOWN（fail-closed）
    s2 = dict(_GOOD_SESSION, placed=False)
    assert mod.score_one(hist, s2)["status"] == STATUS_UNKNOWN
    # 环境没应用上 -> UNKNOWN
    s3 = {"placed": True, "placement_rc": 0,
          "environment": {"applied": False, "error": "TimeoutError"},
          "config": dict(_GOOD_SESSION["config"])}
    r3 = mod.score_one(hist, s3)
    assert r3["status"] == STATUS_UNKNOWN and r3["env_error"] == "TimeoutError"
    # 引擎读回的时刻与请求不一致 -> UNKNOWN
    s4 = {"placed": True, "placement_rc": 0,
          "environment": {"applied": True, "tod_after": "10:00:00"},
          "config": dict(_GOOD_SESSION["config"])}
    assert mod.score_one(hist, s4)["status"] == STATUS_UNKNOWN
    # 空遥测 -> UNKNOWN
    assert mod.score_one([], dict(_GOOD_SESSION))["status"] == STATUS_UNKNOWN


def test_fsd_conditions_aggregate_failclosed():
    from beamng_autopilot.eval import STATUS_UNKNOWN
    mod = _load_conditions()

    def row(cell, status, sensor=0.8):
        return {"cell": cell, "status": status, "lane_sensor_rate": sensor,
                "lane_paired_rate": sensor - 0.1, "cross_centre_frames": 0,
                "off_road_frames": 0, "stuck_frames": 0}

    # 非重复格 1 轮 PASS -> PASS；最差口径取 min
    table = mod.aggregate([row("noon_clear", "PASS", 0.9)])
    assert table["noon_clear"]["status"] == "PASS"
    assert table["noon_clear"]["expected_runs"] == 1
    assert table["noon_clear"]["lane_sensor_rate_min"] == 0.9
    # 重复格只跑 1 轮 -> 轮次不全 -> UNKNOWN（不降门）
    table = mod.aggregate([row("night_clear", "PASS")])
    assert table["night_clear"]["status"] == STATUS_UNKNOWN
    assert table["night_clear"]["expected_runs"] == 2
    # 重复格 2 轮 PASS -> PASS
    table = mod.aggregate([row("night_clear", "PASS", 0.7),
                           row("night_clear", "PASS", 0.5)])
    assert table["night_clear"]["status"] == "PASS"
    assert table["night_clear"]["lane_sensor_rate_min"] == 0.5
    # 任一轮 UNKNOWN -> 整格 UNKNOWN
    table = mod.aggregate([row("night_rain", "PASS"),
                           row("night_rain", STATUS_UNKNOWN)])
    assert table["night_rain"]["status"] == STATUS_UNKNOWN
    assert table["night_rain"]["unknown_runs"] == 1
    # 任一轮 FAIL（其余 PASS）-> FAIL
    table = mod.aggregate([row("dusk_fog", "PASS"),
                           row("dusk_fog", "FAIL")])
    assert table["dusk_fog"]["status"] == "FAIL"


def test_fsd_conditions_expected_cells_from_manifest():
    mod = _load_conditions()
    cells = mod.plan_cells("rainy", "foggy_morning")
    manifest = {"cells": cells}
    exp = mod.expected_cells(manifest)
    assert len(exp) == 20
    assert ("night_fog", 1) in exp and ("noon_clear", 0) in exp
    # 损坏的 manifest 条目跳过，不崩
    assert mod.expected_cells({"cells": [{"cell": "x"}, {}, None]}) == []
    assert mod.expected_cells(None) == []
    # score_one 对「无遥测」的格给 UNKNOWN（fail-closed），且带 env 证据
    from beamng_autopilot.eval import STATUS_UNKNOWN
    r = mod.score_one([], {"placed": False, "placement_rc": 3,
                           "environment": {"applied": True,
                                           "tod_after": "06:29:59"},
                           "config": {"tod": "06:30"}})
    assert r["status"] == STATUS_UNKNOWN
    assert r["frames"] == 0 and r["placement_rc"] == 3


# ---------------------------------------------------------------------------
# 2026-11-04 推理侧杠杆：水平翻转 TTA（概率平均）
# ---------------------------------------------------------------------------

def test_tta_flip_average_and_wiring():
    from beamng_autopilot.vision.segmentation import tta_flip_average
    import numpy as np

    a = np.zeros((3, 4, 5), dtype=np.float32)
    b = np.ones((3, 4, 5), dtype=np.float32)
    m = tta_flip_average(a, b)
    assert m.shape == a.shape and np.allclose(m, 0.5)
    # 形状不一致必须报错，不做静默广播
    try:
        tta_flip_average(a, np.zeros((3, 4, 6), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("形状不一致必须 ValueError")

    # Segmenter 接线：predict_tta 存在、概率平均后走与 predict 相同的后处理
    root = Path(__file__).resolve().parents[1]
    src = (root / "beamng_autopilot" / "vision" / "segmentation.py").read_text("utf-8")
    assert "def predict_tta(" in src
    body = src[src.index("def predict_tta("):]
    body = body[:body.index("\n    def ")]
    assert "tta_flip_average(" in body
    assert "self._postprocess(frame_rgb, road, line)" in body
    assert "self._infer_logits(" in body

    # 评测入口：--tta + 调用 predict_tta
    ev = (root / "scripts" / "m5_eval_seg.py").read_text("utf-8")
    assert 'ap.add_argument("--tta"' in ev
    assert "seg.predict_tta(colour)" in ev


def test_ensemble_wiring_two_models():
    # 同折两种子集成：eval 入口支持 --model2 + --tta ens；Segmenter 提供
    # 概率图与"平均概率 -> 同一后处理"入口。
    root = Path(__file__).resolve().parents[1]
    ev = (root / "scripts" / "m5_eval_seg.py").read_text("utf-8")
    assert 'ap.add_argument("--model2"' in ev
    assert '"ens"' in ev and "prob_average(p1, p2)" in ev
    src = (root / "beamng_autopilot" / "vision" / "segmentation.py").read_text("utf-8")
    assert "def probability_maps_small(" in src
    assert "def masks_from_model_probs(" in src
    assert "def prob_average(" in src
    from beamng_autopilot.vision.segmentation import prob_average, tta_flip_average
    import numpy as np
    a = np.zeros((3, 2, 2), dtype=np.float32)
    b = np.ones((3, 2, 2), dtype=np.float32)
    assert np.allclose(prob_average(a, b), 0.5)
    assert np.allclose(tta_flip_average(a, b), 0.5)


def test_conditions_skip_cells_with_only_session_evidence():
    # 放置失败的格只有 session.json；断点续跑必须把它当"已完成证据"跳过，
    # 否则会反复重跑（违反预注册"至多一次协议重试"）。
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "m5_fsd_conditions.py").read_text("utf-8")
    assert 'Path(str(out_path) + ".session.json").exists()' in src


def test_conditions_score_ignores_own_output_and_non_list():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "m5_fsd_conditions.py").read_text("utf-8")
    assert '"conditions_table.json"' in src
    assert "if not isinstance(hist, list):" in src


# ---------------------------------------------------------------------------
# 2026-11-04 配对鲁棒性研究臂：探针工具（真值插值/真幻影判据/开关默认）
# ---------------------------------------------------------------------------

def _load_pairing_probe():
    spec = importlib.util.spec_from_file_location(
        "m5_pairing_probe", ROOT / "scripts" / "m5_pairing_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_pairing_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _probe_scene():
    return {"scene": {
        "mid": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0],
        "lines": [
            {"role": "left", "nodes": [[-10, 1.8, 0], [10, 1.8, 0]]},
            {"role": "right", "nodes": [[-10, -1.8, 0], [10, -1.8, 0]]},
        ],
    }}


def test_pairing_probe_truth_and_classification():
    import numpy as np
    m = _load_pairing_probe()
    scene = _probe_scene()
    x, y, yaw = m.scene_pose(scene)
    assert (x, y) == (0.0, 0.0) and abs(yaw) < 1e-9
    gx, gy = m.scene_goal(scene, ahead_m=60)
    assert abs(gx - 60.0) < 1e-9 and abs(gy) < 1e-9
    # 左正约定：left 线在 +1.8，right 线在 -1.8 -> 中心 0、宽 3.6
    gt = m.gt_lane(scene, [0.0, 0.0], 0.0)
    assert gt is not None
    assert abs(gt[0]) < 1e-9 and abs(gt[1] - 3.6) < 1e-9
    assert abs(m.lateral_of([0.0, 1.8], [0.0, 0.0], 0.0) - 1.8) < 1e-9
    # 判据：真 / 幻影 / 模糊
    assert m.classify_pair(0.0, 3.6, gt[0], gt[1])["label"] == "true"
    assert m.classify_pair(2.0, 3.6, gt[0], gt[1])["label"] == "phantom"
    assert m.classify_pair(0.0, 4.6, gt[0], gt[1])["label"] == "ambiguous"
    # 换 ego 位姿后真值仍正确（中心线在 y=0 时，ego 偏向 +0.5 -> 中心 -0.5）
    gt2 = m.gt_lane(scene, [0.0, 0.5], 0.0)
    assert abs(gt2[0] + 0.5) < 1e-9 and abs(gt2[1] - 3.6) < 1e-9


def test_pairing_probe_summary_and_defaults():
    m = _load_pairing_probe()
    scene = _probe_scene()
    recs = [
        {"t": 0.0, "pos": [0, 0, 0], "heading": 0.0,
         "pair": {"marking_count": 2, "mode": "paired", "rejects": {},
                  "result": {"center_xy": [0.0, 0.0], "width_m": 3.6}}},
        {"t": 0.5, "pos": [0, 0, 0], "heading": 0.0,
         "pair": {"marking_count": 2, "mode": "paired", "rejects": {},
                  "result": {"center_xy": [0.0, 2.0], "width_m": 3.6}}},
        {"t": 1.0, "pos": [0, 0, 0], "heading": 0.0,
         "pair": {"marking_count": 1, "mode": "none",
                  "rejects": {"far_kind": 3}}},
    ]
    s = m.summarize(recs, scene)
    assert s["ticks"] == 3 and s["ticks_with_candidates"] == 3
    assert s["accepted_pairs"] == 2
    assert s["true_pairs"] == 1 and s["phantom_pairs"] == 1
    assert abs(s["pair_rate_of_candidates"] - 2 / 3) < 1e-4
    assert s["rejects"] == {"far_kind": 3}

    # 研究开关默认等于现行行为：全部 False
    from beamng_autopilot.lane.pairing import (
        DEFAULT_PAIRING_RESEARCH, PairingResearchPolicy)
    assert DEFAULT_PAIRING_RESEARCH.is_default()
    assert DEFAULT_PAIRING_RESEARCH.as_dict() == {
        "far_thin_allow": False, "thin_near_relax": False,
        "mirror_road_center_ok": False}
    assert PairingResearchPolicy().is_default()


def test_pairing_probe_wiring():
    root = Path(__file__).resolve().parents[1]
    drv = (root / "scripts" / "m5_fsd_drive.py").read_text("utf-8")
    assert '"--pairing-probe-out"' in drv
    sess = (root / "beamng_autopilot" / "fsd_drive.py").read_text("utf-8")
    assert "pairing_probe_out" in sess and '"pair"' in sess
    stack = (root / "beamng_autopilot" / "fsd_stack.py").read_text("utf-8")
    assert '"width_m"' in stack and '"center_xy"' in stack
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert "--analyze" in probe and "DEFAULT_PAIRING_RESEARCH" in probe


def test_pairing_probe_collect_drops_and_cert_recall():
    m = _load_pairing_probe()
    scene = _probe_scene()
    recs = [
        {"t": 0.0, "pos": [0, 0, 0], "heading": 0.0,
         "pair": {"marking_count": 2, "collect_drops": {"thin:side": 1},
                  "cands": [{"med_lat": 1.75}, {"med_lat": -1.85}],
                  "rejects": {}, "result": None}},
        {"t": 0.5, "pos": [0, 0, 0], "heading": 0.0,
         "pair": {"marking_count": 2, "collect_drops": {},
                  "cands": [{"med_lat": 4.2}, {"med_lat": -4.3}],
                  "rejects": {"far_kind": 2}, "result": None}},
    ]
    s = m.summarize(recs, scene)
    assert s["thin_side_drops"] == 1
    assert s["collect_drops"] == {"thin:side": 1}
    rec = s["cert_line_recall"]
    assert rec["ticks_with_gt"] == 2
    assert rec["left_hit_rate"] == 0.5 and rec["right_hit_rate"] == 0.5
    assert rec["both_hit_rate"] == 0.5


def test_pairing_probe_no_off_reference_guard():
    # off_reference 守卫已移除：right_lane 场景里自车可能被 nav 路由拉回
    # 道路中线，认证车道合法偏移；此时按中心/宽度误差直接判。
    m = _load_pairing_probe()
    scene = _probe_scene()
    recs = [{"t": 0.0, "pos": [0.0, 2.0, 0.0], "heading": 0.0,
             "pair": {"marking_count": 2, "collect_drops": {},
                      "cands": [{"med_lat": -0.25}, {"med_lat": -3.7}],
                      "rejects": {},
                      "result": {"center_xy": [0.0, 2.0], "width_m": 3.6}}}]
    s = m.summarize(recs, scene)
    assert s["accepted_pairs"] == 1
    assert s["off_reference_pairs"] == 0
    # 配对中心在 ego 处，GT 车道中心偏 -2m -> 判幻影（参考系差 2m）
    assert s["phantom_pairs"] == 1 and s["true_pairs"] == 0


# ---------------------------------------------------------------------------
# 2026-11-04 配对专用场景：right_lane 约定（贴右侧铺装边）+ 探针场景模式
# ---------------------------------------------------------------------------

def test_right_lane_convention_geometry():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "m5_controlled_scenes.py").read_text("utf-8")
    assert "def right_lane_lats(" in src
    spec = importlib.util.spec_from_file_location(
        "m5_controlled_scenes_rl", root / "scripts" / "m5_controlled_scenes.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5_controlled_scenes_rl"] = m
    spec.loader.exec_module(m)
    left, right = m.right_lane_lats(3.483)   # 默认边缘余量 0.6m
    assert abs(right - (-2.883)) < 1e-6
    assert abs(left - 0.717) < 1e-6
    # 车道中心在右侧车道内、且在铺装内（±half）
    center = 0.5 * (left + right)
    assert abs(center - (-1.083)) < 1e-6
    assert abs(left) <= 3.483 and abs(right) <= 3.483
    # 窄路（半宽 2.5）：车道仍完全在铺装内
    l2, r2 = m.right_lane_lats(2.5)
    assert abs(r2) <= 2.5 and abs(l2) <= 2.5


def test_pairing_probe_scenario_mode_wiring():
    root = Path(__file__).resolve().parents[1]
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert "build_scenario(" in probe
    assert '"--line-convention"' in probe and "right_lane" in probe
    assert "scenario_mode" in probe
    assert "conn.open(launch=True)" in probe


def test_pairing_probe_no_goal_default():
    # 配对探针默认不建导航路线：route snap 会把自车从车道中心拉回道路中线
    root = Path(__file__).resolve().parents[1]
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert '"--no-goal"' in probe
    assert 'default=True' in probe
    assert 'if not args.no_goal:' in probe


# ---------------------------------------------------------------------------
# 2026-11-04 配对研究臂 F1/F2：几何判据 + 策略贯通（默认关）
# ---------------------------------------------------------------------------

def test_pairing_arm_geometry_helpers():
    from types import SimpleNamespace
    from beamng_autopilot.lane.pairing import (
        _far_thin_pair_ok, _thin_near_ok)

    def cand(lat, s0=0.0, s1=10.0, step=1.0):
        proj = np.array([[s, lat] for s in np.arange(s0, s1 + 1e-9, step)],
                        dtype=float)
        return SimpleNamespace(proj=proj, kind="thin")

    # 宽 3.6m、间距恒定、重叠 10m -> 允许
    assert _far_thin_pair_ok(cand(1.8), cand(-1.8))
    # 宽 1.0m（过窄）-> 拒绝
    assert not _far_thin_pair_ok(cand(0.5), cand(-0.5))
    # 重叠只有 2m -> 拒绝
    assert not _far_thin_pair_ok(cand(1.8, 0, 2), cand(-1.8, 0, 2))
    # 间距漂移 >0.35m -> 拒绝
    drift = SimpleNamespace(
        proj=np.array([[0, 1.8], [5, 2.9], [10, 4.0]], dtype=float), kind="thin")
    assert not _far_thin_pair_ok(drift, cand(-1.8))
    # 近场判据：0–8m 有 ≥1 站点
    assert _thin_near_ok(cand(1.8, 4, 20))
    assert not _thin_near_ok(cand(1.8, 10, 20))


def test_pairing_arm_policy_plumbing():
    root = Path(__file__).resolve().parents[1]
    pair = (root / "beamng_autopilot" / "lane" / "pairing.py").read_text("utf-8")
    assert "research: \"PairingResearchPolicy | None\" = None" in pair
    assert "pol.far_thin_allow and _far_thin_pair_ok(l, r)" in pair
    assert "pol.thin_near_relax" in pair
    stack = (root / "beamng_autopilot" / "fsd_stack.py").read_text("utf-8")
    assert "research=getattr(self, \"pairing_research\", None)" in stack
    drv = (root / "beamng_autopilot" / "fsd_drive.py").read_text("utf-8")
    assert "stack.pairing_research = PairingResearchPolicy(" in drv
    cli = (root / "scripts" / "m5_fsd_drive.py").read_text("utf-8")
    assert '"--pairing-research"' in cli and '"far_thin"' in cli
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert '"--arm"' in probe and "_arm_policy(" in probe
    # 默认全 False
    from beamng_autopilot.lane.pairing import PairingResearchPolicy
    assert PairingResearchPolicy().is_default()


def test_edge_margin_plumbing():
    root = Path(__file__).resolve().parents[1]
    coll = (root / "scripts" / "m5_controlled_scenes.py").read_text("utf-8")
    assert "edge_margin_m: float = 0.6" in coll
    assert "edge_margin_m=edge_margin_m" in coll
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert '"--edge-margin-m"' in probe
    assert "edge_margin_m=float(args.edge_margin_m)" in probe
    # 余量 1.0m 的几何：hw3.483 -> 右 -2.483、左 1.117、中心 -0.683
    spec = importlib.util.spec_from_file_location(
        "mcs_margin", root / "scripts" / "m5_controlled_scenes.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["mcs_margin"] = m
    spec.loader.exec_module(m)
    l, r = m.right_lane_lats(3.483, edge_margin_m=1.0)
    assert abs(r - (-2.483)) < 1e-6 and abs(l - 1.117) < 1e-6


# ---------------------------------------------------------------------------
# 2026-11-04 黄线/span/CLAHE 候选层研究臂（Y1/S1/I1，默认关）+ 仪器
# ---------------------------------------------------------------------------

def test_yellow_span_clahe_switches_and_instrumentation():
    root = Path(__file__).resolve().parents[1]
    seg = (root / "beamng_autopilot" / "vision" / "segmentation.py").read_text("utf-8")
    assert 'BEAMNG_YELLOW_GATE", "abs"' in seg
    assert 'BEAMNG_CAND_CLAHE", "0"' in seg
    assert '"mask_counts"' in seg and '"cv_yellow_px"' in seg
    # 默认分支保持原阈值（sat 30 / val 100 / 亮 +12）
    assert "sat >= 30" in seg and "val >= 100" in seg
    pair = (root / "beamng_autopilot" / "lane" / "pairing.py").read_text("utf-8")
    assert 'BEAMNG_THIN_SPAN", "abs"' in pair
    probe = (root / "scripts" / "m5_pairing_probe.py").read_text("utf-8")
    assert "def _arm_env(" in probe
    assert '"yellow"' in probe and '"span"' in probe and '"clahe"' in probe
    assert '"thin_span_drops"' in probe and '"collect_by_kind"' in probe


def test_probe_arm_env_mapping_and_by_kind():
    m = _load_pairing_probe()
    env = m._arm_env("yellow")
    assert env.get("BEAMNG_YELLOW_GATE") == "rel"
    assert "BEAMNG_THIN_SPAN" not in env or env.get("BEAMNG_THIN_SPAN") != "relax"
    env2 = m._arm_env("span")
    assert env2.get("BEAMNG_THIN_SPAN") == "relax"
    env3 = m._arm_env("clahe")
    assert env3.get("BEAMNG_CAND_CLAHE") == "1"
    base = m._arm_env("baseline")
    for k in ("BEAMNG_YELLOW_GATE", "BEAMNG_THIN_SPAN", "BEAMNG_CAND_CLAHE"):
        assert k not in base
    assert m._by_kind({"thin:span": 3, "thin:side": 1, "solid:align": 2}) == {
        "thin": {"span": 3, "side": 1}, "solid": {"align": 2}}
