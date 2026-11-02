"""DecalRoad 漆线 annotation 归因探针（诊断用，**不产训练数据**）。

为什么存在：ECA1/ECA2（east_coast_usa，2026-10-18）两批 0/5 站点不合格——
逐帧 RGB 里漆线可见（paint-like 沿真值曲线），但 annotation 同像素是
ASPHALT，导出冻结门（coverage/appearance/投影）自然全挂。离线证据已排除：
z 埋地（ECA2 ground_alignment delta≈0、遮挡深度比 1.000）、镜像
（direct==mirrored coverage）、材质档案（两地图 line_white 定义逐字段一致、
annotation=SOLID_LINE）、材质遮蔽（mapTo 唯一）。

探针 1（同站点 7 条线，逐条改宽度/z-lift/over_objects/插入序）结论：
**近链段（4–12 m）7 条线全部 annotation=ASPHALT**，远链段只有正侧线拿到
SOLID_LINE（11/21、5/21），负侧 0/21。不是宽度/flag/插入序/小幅 z-lift 的事。

叠置分析（地图 zip DecalRoad 项 × 我们的链）指出机制：east 的链上压着
多条 4 m 宽 ``m_asphalt_damaged_*`` 修补带（node z 比地形高 0.06–0.38 m），
**annotation pass 的共面竞争按贴花带 z 高低判**——我们低就判 ASPHALT，
我们高就判 SOLID_LINE；west_coast 的覆盖项 z 全在我们之下，所以处处赢。
但探针 1 的 lift 变体 paint 同步变差（lift_015 近链段 paint=1/8），说明
抬升后渲染位置与节点 z 可能不一致（draping 语义），必须实测判决。

探针 2 两个场景、同站点、一次游戏会话：

场景 A（z 轴，只改 z/线位）：
* ``pos_ctrl``    +1.8 m / 0.15 m 宽 / z 不抬 —— 复现探针 1 repro_p0（对照）
* ``pos_noLift``  +3.3 m / 0.40 m 宽 / z 不抬（对照）
* ``pos_lift50``  +2.6 m / z +0.50 m（抬到所有修补带之上）
* ``neg_ctrl``    −3.4 m / 0.40 m 宽 / z 不抬 —— 失败侧基线
* ``neg_lift08``  −2.7 m / z +0.08 m
* ``neg_lift25``  −2.0 m / z +0.25 m
* ``neg_lift50``  −1.3 m / z +0.50 m（决定性 z 变体）

场景 B（renderPriority 轴；monkey-patch 还原 beamngpy ``_get_roads_list``
的 ``render_priority = idx`` 改写，见 beamngpy/scenario/scenario.py）：
* ``prio_hi_*``  −3.4/−2.0 m，render_priority=50（高于地图所有项的 17）
* ``prio_lo_*``  −2.7/−1.3 m，render_priority=0

判读（近/远分段 + 6px 邻接）：
* A 的 neg_lift 变体出现 SOLID_LINE 且 neg_ctrl 没有 → 机制 = z 竞争，
  修复走"线节点 z 取覆盖带上沿 + 余量"（生成器内做，真值同源）。
* B 的 prio_hi 赢而 prio_lo 输 → renderPriority 是杠杆，修复走
  beamngpy 补丁 + 高优先级。
* 两者都无效 → 另寻机制（贴花 draping / annotation pass 语义）。

用法（诊断目录，不进训练池）：
    python scripts/m5_decal_annotation_probe.py --map east_coast_usa \\
        --site-report logs/m5_seg/collect_e5line_eca2_20261018/scene_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m5_controlled_scenes as m  # noqa: E402
from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402
from beamng_autopilot.experiments.auto_truth import project_point  # noqa: E402
from beamng_autopilot.vision.paint_appearance import paint_like_mask  # noqa: E402

SOLID_RGB = (255, 196, 128)      # beamng_get_annotations_v1: SOLID_LINE
ASPHALT_RGB = (128, 128, 128)    # ASPHALT

#: (tag, lat_m, width_m, z_lift_m) —— 场景 A：z 轴变量分离
TEST_Z: tuple[tuple[str, float, float, float], ...] = (
    ("pos_ctrl",   +1.8, 0.15, 0.00),
    ("pos_noLift", +3.3, 0.40, 0.00),
    ("pos_lift50", +2.6, 0.40, 0.50),
    ("neg_ctrl",   -3.4, 0.40, 0.00),
    ("neg_lift08", -2.7, 0.40, 0.08),
    ("neg_lift25", -2.0, 0.40, 0.25),
    ("neg_lift50", -1.3, 0.40, 0.50),
)

#: (tag, lat_m, render_priority) —— 场景 B：优先级变量分离（z 全部不抬）
TEST_PRIO: tuple[tuple[str, float, int], ...] = (
    ("prio_hi_a", -3.4, 50),
    ("prio_lo_a", -2.7, 0),
    ("prio_hi_b", -2.0, 50),
    ("prio_lo_b", -1.3, 0),
)

#: (tag, lat_m, width_m, material) —— 场景 C：材质轴（探针 2 后追加）
#:
#: 探针 2 判决：z 抬到 +0.5 m 与 renderPriority 0/50 全部无效；east 的
#: 835 个 SOLID 像素 99% 属于地图黄带。材质档案比对给出根因候选：
#: east 的 line_white/line_yellow 家族 ``baseColorFactor`` alpha 0.8–0.9 ×
#: ``opacityFactor`` 0.7 → 像素最终 alpha ≤ 0.56，配 ``alphaRef=255`` 的
#: alpha 测试在 annotation pass 里**全像素被剔**（RGB pass 走 translucent
#: 混合，漆仍可见）；west 的 line_white 无 opacity 缩放（核心 alpha=1.0）
#: 所以写得进；east 地图自己的车道线用 ``line_yellow_double``
#: （alphaTest=True/alphaRef=0）所以黄带写得进。
TEST_MAT: tuple[tuple[str, float, float, str], ...] = (
    ("mat_dbl",      -2.6, 0.40, "line_yellow_double"),   # 预期 SOLID
    ("mat_dbl_narrow", -1.3, 0.15, "line_yellow_double"), # 导出宽度 0.15
    ("mat_white",    -3.4, 0.40, "line_white"),           # 对照：预期 ASPHALT
    ("mat_yellow",   -2.0, 0.40, "line_yellow"),          # 同族对照：预期 ASPHALT
)

#: (tag, lat_m, width_m, material) —— 场景 D：合同宽度 0.15 下的候选材质
#:
#: 探针 3 结论：line_yellow_double 在 0.4 m 宽度下冻结门三门全过
#: （appearance 0.90 / coverage 1.0 / 投影 0.55 px），但 0.15 m 合同宽度下
#: 双黄线条纹纹理被压到亚像素，appearance 只有 0.67。这里在 0.15 m 下测
#: 三个 annotation 可写的候选（alphaTest/alphaRef<255），实测真实门指标。
TEST_MAT2: tuple[tuple[str, float, float, str], ...] = (
    ("m2_parking1", -3.0, 0.15, "line_parking1"),
    ("m2_parking2", -2.0, 0.15, "line_parking2"),
    ("m2_dbl",      -1.0, 0.15, "line_yellow_double"),
    ("m2_dbl_dis",  +1.8, 0.15, "line_yellow_double_discontinue"),
)

#: (tag, lat_m, width_m, material) —— 场景 E：line_yellow_double 宽度扫描
#:
#: 探针 4 结论（0.15 m 合同宽度实测）：line_parking1/2 的条纹在纹理里
#: 偏心，annotation 落在真值外 5–8 px（投影门挂）；line_yellow_double_discontinue
#: 是虚线，沿链 appearance 必然缺口；只有 line_yellow_double 居中且
#: annotation 全写，但它 0.15 m 下双条纹各条纹被压成亚像素、真值点落在
#: 中心 gap（asphalt）上 → appearance 0.43–0.67 < 0.9。这里扫描宽度找
#: appearance≥0.9 的最小值（0.4 m 单点已知 0.90 擦线）。
TEST_WIDE: tuple[tuple[str, float, float, str], ...] = (
    ("w020", -3.3, 0.20, "line_yellow_double"),
    ("w025", -2.2, 0.25, "line_yellow_double"),
    ("w030", -1.1, 0.30, "line_yellow_double"),
    ("w040", +1.1, 0.40, "line_yellow_double"),
    ("w050", +2.2, 0.50, "line_yellow_double"),
)

#: 里程分段（米）：近链段像素分离好，远链段看 z 竞争翻转
S_BINS: tuple[tuple[float, float], ...] = ((4.0, 8.0), (8.0, 14.0),
                                           (14.0, 22.0), (22.0, 34.0))


def _restore_render_priority() -> None:
    """还原 beamngpy 对 render_priority 的插入序改写（只补丁内存对象）。

    beamngpy/scenario/scenario.py ``_get_roads_list`` 会把每个 road 的
    ``render_priority`` 强制写成插入序号 idx——我们显式传的值被丢弃。
    地图自带贴花的 renderPriority 是 2–17，用 idx(0..n) 根本压不过。
    这里包装该方法，把 Road 对象上存的原值写回去。诊断用；生成器要不要
    同步修，由探针 2 的判决决定。
    """
    from beamngpy.scenario import Scenario
    if getattr(Scenario._get_roads_list, "_prio_restored", False):
        return
    orig = Scenario._get_roads_list

    def _patched(self):
        roads = orig(self)
        for rd, road in zip(roads, self.roads):
            rd["render_priority"] = int(getattr(road, "render_priority", 10))
        return roads

    _patched._prio_restored = True            # type: ignore[attr-defined]
    Scenario._get_roads_list = _patched       # type: ignore[method-assign]


def _truth_bins(site: dict, truth: list[dict]) -> dict[str, list[dict]]:
    mid = np.asarray(site["mid"], dtype=float)
    d = m._unit(site["dir"])
    out = {f"{a:.0f}-{b:.0f}": [] for a, b in S_BINS}
    out["all"] = []
    for p in truth:
        pw = np.asarray(p["world"], dtype=float)
        s = float((pw - mid)[:2] @ d[:2])
        out["all"].append(p)
        for a, b in S_BINS:
            if a <= s <= b:
                out[f"{a:.0f}-{b:.0f}"].append(p)
                break
    return out


def _analyze(frame: dict, site: dict, truth: list[dict]) -> dict:
    ann = np.asarray(frame["annotation"])
    lab = np.asarray(frame["label"])
    rgb = np.asarray(frame["rgb"])
    cam = frame["camera"]
    h, w = lab.shape[:2]
    pm = paint_like_mask(rgb)
    solid = (ann == np.asarray(SOLID_RGB, dtype=np.uint8)).all(axis=2)
    asphalt = (ann == np.asarray(ASPHALT_RGB, dtype=np.uint8)).all(axis=2)
    rows: dict = {}
    for bname, pts in _truth_bins(site, truth).items():
        n = c_solid = c_solid6 = c_asphalt = c_lab2 = c_paint = 0
        for p in pts:
            u, v = project_point(p.get("world"), cam)
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < w and 0 <= y < h):
                continue
            n += 1
            sl = (slice(max(0, y - 2), y + 3), slice(max(0, x - 2), x + 3))
            sl6 = (slice(max(0, y - 6), y + 7), slice(max(0, x - 6), x + 7))
            c_solid += int(solid[sl].any())
            c_solid6 += int(solid[sl6].any())
            c_asphalt += int(asphalt[sl].any())
            c_lab2 += int((lab[sl] == 2).any())
            c_paint += int(pm[sl].any())
        rows[bname] = {"n": n, "solid2px": c_solid, "solid6px": c_solid6,
                       "asphalt2px": c_asphalt, "label2px": c_lab2,
                       "paint2px": c_paint}
    return rows


def _build_scenario(conn, *, map_name: str, site: dict,
                    z_specs: list, prio_specs: list,
                    mat_specs: list | None = None,
                    material: str = "line_white") -> tuple:
    from beamngpy import Scenario, Vehicle
    from beamngpy.misc.quat import angle_to_quat
    from beamngpy.scenario.road import Road

    scen = Scenario(str(map_name), "m5_decal_probe2")
    d = m._unit(np.asarray(site["dir"], dtype=float))
    half = float(site["half_width_m"])
    specs: list[dict] = []
    for tag, lat, width, lift in z_specs:
        lat_c = max(-(half - 0.3), min(half - 0.3, float(lat)))
        nodes = m._line_nodes(site, lat_c)
        if lift:
            nodes = [[p[0], p[1], float(p[2]) + float(lift)] for p in nodes]
        truth = m._line_truth(nodes, tag)
        road = Road(material=str(material), rid=f"probe_{tag}", interpolate=False,
                    default_width=float(width), over_objects=True,
                    render_priority=1000)   # 场景 A 不看优先级：保持中性
        road.add_nodes(*[tuple(nd) for nd in nodes])
        scen.add_road(road)
        specs.append({"tag": tag, "lat": lat_c, "width": width,
                      "z_lift_m": lift, "render_priority": None,
                      "material": str(material), "truth": truth})
    for tag, lat, prio in prio_specs:
        lat_c = max(-(half - 0.3), min(half - 0.3, float(lat)))
        nodes = m._line_nodes(site, lat_c)
        truth = m._line_truth(nodes, tag)
        road = Road(material=str(material), rid=f"probe_{tag}", interpolate=False,
                    default_width=0.40, over_objects=True,
                    render_priority=int(prio))
        road.add_nodes(*[tuple(nd) for nd in nodes])
        scen.add_road(road)
        specs.append({"tag": tag, "lat": lat_c, "width": 0.40, "z_lift_m": 0.0,
                      "render_priority": int(prio),
                      "material": str(material), "truth": truth})
    for tag, lat, width, material in (mat_specs or []):
        lat_c = max(-(half - 0.3), min(half - 0.3, float(lat)))
        nodes = m._line_nodes(site, lat_c)
        truth = m._line_truth(nodes, tag)
        road = Road(material=str(material), rid=f"probe_{tag}", interpolate=False,
                    default_width=float(width), over_objects=True,
                    render_priority=1000)
        road.add_nodes(*[tuple(nd) for nd in nodes])
        scen.add_road(road)
        specs.append({"tag": tag, "lat": lat_c, "width": width, "z_lift_m": 0.0,
                      "render_priority": None, "material": str(material),
                      "truth": truth})
    yaw_deg = -math.degrees(math.atan2(d[1], d[0])) - 90.0
    ego = Vehicle("ego", model="etk800", color="Red")
    scen.add_vehicle(ego, pos=tuple(site["mid"]),
                     rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)), cling=True)
    return scen, specs


def _build_full_scenario(conn, *, map_name: str, sites: list,
                         target_index: int, lat_l: float, lat_r: float,
                         material: str) -> tuple:
    """按批次结构重建全部站点 × 2 线（横向夹到 half−0.3），目标站线落在
    idx 2k/2k+1（a4s0 → 8/9），忠实复现批次的多对象/插入序上下文。"""
    from beamngpy import Scenario, Vehicle
    from beamngpy.misc.quat import angle_to_quat
    from beamngpy.scenario.road import Road

    scen = Scenario(str(map_name), "m5_decal_probe_full")
    tgt = sites[int(target_index)]
    d = m._unit(np.asarray(tgt["dir"], dtype=float))
    specs: list[dict] = []

    def _add(site_i: int, site: dict, lat: float, role: str) -> None:
        half = float(site["half_width_m"])
        lat_c = max(-(half - 0.3), min(half - 0.3, float(lat)))
        nodes = m._line_nodes(site, lat_c)
        truth = m._line_truth(nodes, role)
        road = Road(material=str(material), rid=f"m5_line_s{site_i}_{role}",
                    interpolate=False, default_width=0.40)
        road.add_nodes(*[tuple(nd) for nd in nodes])
        scen.add_road(road)
        if site_i == int(target_index):
            specs.append({"tag": f"full_{role}", "lat": lat_c, "width": 0.40,
                          "z_lift_m": 0.0, "render_priority": None,
                          "material": str(material), "truth": truth})

    for i, s_i in enumerate(sites):
        _add(i, s_i, float(lat_l), "left")
        _add(i, s_i, float(lat_r), "right")
    yaw_deg = -math.degrees(math.atan2(d[1], d[0])) - 90.0
    ego = Vehicle("ego", model="etk800", color="Red")
    scen.add_vehicle(ego, pos=tuple(tgt["mid"]),
                     rot_quat=angle_to_quat((0.0, 0.0, yaw_deg)), cling=True)
    return scen, specs


def _run_scene(conn, probe, *, scen, specs, site, out_dir: Path, tag: str,
               width: int, height: int) -> dict:
    with conn.io_lock:
        scen.make(conn.bng)
        conn.bng.scenario.load(scen)
        conn.bng.scenario.start()
    time.sleep(2.0)
    try:
        conn.attach_vehicle(already_open=True)
    except Exception:                                # noqa: BLE001
        pass
    all_truth = [dict(p, probe_tag=s["tag"]) for s in specs for p in s["truth"]]
    frames, _palette, meta = m.capture_site(
        conn, probe, site=site, n_frames=1, width=width, height=height,
        out_dir=out_dir / "frames", tag=tag, frame_truth=all_truth)
    result = {"capture_meta": {k: v for k, v in (meta or {}).items()
                               if k != "frames"}, "lines": {}}
    for s in specs:
        result["lines"][s["tag"]] = {
            "lat_m": s["lat"], "width_m": s["width"], "z_lift_m": s["z_lift_m"],
            "render_priority": s["render_priority"],
            "material": s.get("material", "line_white"),
            **_analyze(frames[0], site, s["truth"])}
    return result


def _print_scene(name: str, res: dict) -> None:
    for bname in [f"{a:.0f}-{b:.0f}" for a, b in S_BINS] + ["all"]:
        print(f"[decal-probe2] {name} bin={bname}:")
        for tag, r in res["lines"].items():
            b = r.get(bname) or {}
            print(f"    {tag:>12}: n={b.get('n', 0):3d} "
                  f"solid2px={b.get('solid2px', 0):3d} "
                  f"solid6px={b.get('solid6px', 0):3d} "
                  f"asph2px={b.get('asphalt2px', 0):3d} "
                  f"paint2px={b.get('paint2px', 0):3d} "
                  f"| lat={r['lat_m']:+.2f} lift={r['z_lift_m']:.2f} "
                  f"prio={r['render_priority']} mat={r.get('material')}",
                  flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="east_coast_usa")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--site-report", required=True)
    ap.add_argument("--site-index", type=int, default=0,
                    help="site-report.sites 的站点下标（utah a3s0=3 / a4s0=4）")
    ap.add_argument("--material", default="line_white",
                    help="z/prio 变体路的材质（utah/east 用 line_yellow_double）")
    ap.add_argument("--line-laterals", default=None,
                    help="批次线位（米，逗号分隔 left,right；utah 批次 ±1.8）。"
                         "默认用站点 lat_left/lat_right（路缘夹紧位）")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenes", nargs="*", default=["z", "mat", "mat2", "wide", "prio"],
                    choices=("z", "prio", "mat", "mat2", "wide", "utah", "utah2"),
                    help="跑哪些场景（默认全部）")
    args = ap.parse_args()

    rep = json.loads(Path(args.site_report).read_text(encoding="utf-8"))
    sites = rep.get("sites_after_ground_alignment") or rep.get("sites") or []
    if not sites:
        raise SystemExit(f"site-report 里没有站点：{args.site_report}")
    if not (0 <= int(args.site_index) < len(sites)):
        raise SystemExit(f"--site-index {args.site_index} 越界（0..{len(sites) - 1}）")
    site = dict(sites[int(args.site_index)])
    print(f"[decal-probe2] 地图={args.map} 站点[{args.site_index}] mid={site['mid']} "
          f"half_width={site['half_width_m']} lat_l={site.get('lat_left')} "
          f"lat_r={site.get('lat_right')}", flush=True)

    out = Path(args.out) if args.out else (
        config.LOGS_DIR / "m5_seg"
        / f"diag_decal_probe2_{args.map}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    probe = m._load_probe()
    conn = BeamNGConnector(str(args.map), "etk800",
                           port=probe._runtime_port(args),
                           home=probe._runtime_home(args))
    from beamng_autopilot.experiments.collection import (
        close_started_game, game_pids)
    _pids_before = game_pids()
    _launched_after = time.time()
    result: dict = {"probe": "m5_decal_annotation_probe/2", "map": str(args.map),
                    "site": {"mid": site["mid"], "dir": site["dir"],
                             "half_width_m": site["half_width_m"]},
                    "site_index": int(args.site_index),
                    "material": str(args.material),
                    "line_laterals": str(args.line_laterals),
                    "site_report": str(args.site_report),
                    "bins": [list(b) for b in S_BINS], "scenes": {},
                    "errors": []}
    try:
        m.prepare_initial_scene(conn, attach=bool(args.attach))
        if "z" in args.scenes:
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=list(TEST_Z), prio_specs=[])
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenA_z",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["z"] = res
            _print_scene("z", res)
        if "mat" in args.scenes:
            # 材质场景在 prio 场景之前跑：避免 _restore_render_priority 影响
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[], prio_specs=[], mat_specs=list(TEST_MAT))
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenC_mat",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["mat"] = res
            _print_scene("mat", res)
        if "mat2" in args.scenes:
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[], prio_specs=[], mat_specs=list(TEST_MAT2))
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenD_mat2",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["mat2"] = res
            _print_scene("mat2", res)
        if "wide" in args.scenes:
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[], prio_specs=[], mat_specs=list(TEST_WIDE))
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenE_wide",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["wide"] = res
            _print_scene("wide", res)
        if "utah" in args.scenes:
            # utah 截断定因（T16_UTAH_DECAL_EXTENT_PROBE_PREREG_20261029）：
            # lift000 基线 = 批次的 render_priority=idx 行为（先跑，不补丁）；
            # lift 序列看 z 竞争；补丁后 prio 序列看优先级战争。
            # 材质必须用该图 annotation 可写的 line_yellow_double。
            _mat = str(args.material)
            _lat_r = float(site.get("lat_right") or -1.8)
            _lat_l = float(site.get("lat_left") or 1.8)
            _z_specs = [("ut_r_lift000", _lat_r, 0.40, 0.0),
                        ("ut_r_lift010", _lat_r, 0.40, 0.10),
                        ("ut_r_lift020", _lat_r, 0.40, 0.20),
                        ("ut_r_lift035", _lat_r, 0.40, 0.35),
                        ("ut_l_lift000", _lat_l, 0.40, 0.0),
                        ("ut_l_lift020", _lat_l, 0.40, 0.20)]
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=_z_specs, prio_specs=[], material=_mat)
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenU_utah_z",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["utah_z"] = res
            _print_scene("utah_z", res)
            _restore_render_priority()
            _prio_specs = [("ut_r_p030", _lat_r, 30), ("ut_r_p060", _lat_r, 60),
                           ("ut_l_p060", _lat_l, 60)]
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[], prio_specs=_prio_specs, material=_mat)
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenU_utah_prio",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["utah_prio"] = res
            _print_scene("utah_prio", res)
        if "utah2" in args.scenes:
            # 第二轮（2026-10-29 首轮判读后）：首轮 lat_left/right 是路缘夹紧位
            # (±(half−0.3))，不是批次线位 ±1.8 —— 本组补批次忠实变体：
            #   ut18  = 单站场景、批次线位（idx 0/1）——分离"线位"与"对象数/序"
            #   utfull= 全 5 站 × 2 线批次结构（目标站线 idx 8/9）——忠实复现
            _mat = str(args.material)
            if args.line_laterals:
                _ll, _lr = (float(v) for v in str(args.line_laterals).split(","))
            else:
                _ll = float(site.get("lat_left") or 1.8)
                _lr = float(site.get("lat_right") or -1.8)
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[("ut18_l", _ll, 0.40, 0.0),
                         ("ut18_r", _lr, 0.40, 0.0)],
                prio_specs=[], material=_mat)
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenU_utah_18",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["utah_18"] = res
            _print_scene("utah_18", res)
            scen, specs = _build_full_scenario(
                conn, map_name=args.map, sites=list(sites),
                target_index=int(args.site_index),
                lat_l=_ll, lat_r=_lr, material=_mat)
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenU_utah_full",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["utah_full"] = res
            _print_scene("utah_full", res)
        if "prio" in args.scenes:
            _restore_render_priority()
            scen, specs = _build_scenario(
                conn, map_name=args.map, site=site,
                z_specs=[], prio_specs=list(TEST_PRIO))
            res = _run_scene(conn, probe, scen=scen, specs=specs, site=site,
                             out_dir=out, tag="scenB_prio",
                             width=int(args.width), height=int(args.height))
            result["scenes"]["prio"] = res
            _print_scene("prio", res)
        (out / "probe_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"[decal-probe2] 产物 -> {out}")
    finally:
        try:
            conn.close()
        except Exception:                                # noqa: BLE001
            pass
        _ga = close_started_game(_pids_before, launched_after=_launched_after,
                                 verify_timeout_s=10.0)
        print(f"[decal-probe2] 收尾关游戏：closed={_ga.get('closed')} "
              f"killed={_ga.get('killed')} "
              f"{_ga.get('reason') or _ga.get('note') or ''}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
