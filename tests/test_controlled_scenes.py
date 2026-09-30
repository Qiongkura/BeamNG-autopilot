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
                             "slope_curve", "material_mix"}
    assert m.SCENES["known_no_line"]["lines"] == []
    assert m.SCENES["occluded_line"].get("occluder") is True
    assert m.SCENES["material_mix"].get("gravel") is True
    assert len(m.SCENES["known_line"]["lines"]) == 2
    # 线材质必须是地图自己的贴花材质（路线 B：真实可见表面，不是自造方块）
    for spec in m.SCENES.values():
        for kind, _sign, _role in spec["lines"]:
            assert kind in m._MAT
            assert "italy_road_markings" in m._MAT[kind]


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
