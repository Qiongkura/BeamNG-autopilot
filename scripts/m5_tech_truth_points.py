"""从 road network 导出已知 3D 真值点（自动真值探针的输入；T16 §4.1/§4.3）。

为什么需要它：探针要求"已知线段的可见投影与标签一致"，就必须有**生成侧**的
世界坐标真值点，而不是拿模型输出或渲染结果当真值。本脚本在车辆当前所在路段上，
用 BeamNG 的 road network（`scenario.get_road_network(include_edges=True,
drivable_only=True)`）按车道边界采样：

* **线点**（``class=2``）：车道分界线的横向位置，带 ``role``（相对行驶车道的
  左/右）。横向按 ``lanesLeft``/``lanesRight`` 等分路面（与
  ``m5_lane_center_capture._road_lane_geometry`` 同一约定）；**不宣称虚实**——
  网络给几何，虚实由渲染决定，探针只校验"线在那里"；
* **路面点**（``class=1``）：行驶车道中心；
* z 取 ``middle`` 行在该处的路面高度（路面横向视为同高：路拱/超高忽略，写进
  ``limits``）；``left``/``right`` 行没有 z 时不再另算。

输出（``--out``）是可复现入口的产物：同一地图/出生点/参数重跑得到同一批点
（点坐标由网络几何决定，不依赖模型）。给
``m5_auto_truth_probe.py --runtime tech --truth-json`` 使用。

**发现即结论**：投影对不上说明该地图/路段的可见漆线与网络几何不一致——探针
必须报失败并隔离该资产，不放宽判据（方案 §4.3）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_tech_truth_points.py \\
        --map italy --out logs\\experiments\\t16_tech_probe_20260927\\truth_italy.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402


def _unit_fwd(state) -> np.ndarray:
    d = getattr(state, "dir", None)
    if d is not None and len(d) >= 2:
        v = np.asarray(d, dtype=float)[:2]
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            return v / n
    h = float(getattr(state, "heading", 0.0))
    return np.array([math.cos(h), math.sin(h)])


def _interp(a, b, t) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a + t * (b - a)


def _nearest_edge(roads, pos):
    """最近的可行驶路段行：返回 ``(rid, meta, i, t, edges)``（找不到 None）。

    与 ``m5_lane_center_capture._road_lane_geometry`` 同一搜索口径（同一份
    ``include_edges`` 数据、同一"点到折线距离"定义），避免"哪个路段算最近"
    随入口而变。
    """
    pos = np.asarray(pos, dtype=float)[:2]
    best = None
    for rid, meta in (roads or {}).items():
        if not isinstance(meta, dict):
            continue
        edges = meta.get("edges")
        if not isinstance(edges, list) or len(edges) < 2:
            continue
        mids = [None if row.get("middle") is None
                else np.asarray(row["middle"], dtype=float) for row in edges]
        for i in range(len(mids) - 1):
            a, b = mids[i], mids[i + 1]
            if a is None or b is None:
                continue
            ab = b[:2] - a[:2]
            d2 = float(ab @ ab)
            if d2 < 1e-9:
                continue
            t = float(np.clip((pos - a[:2]) @ ab / d2, 0.0, 1.0))
            p = a[:2] + t * ab
            dist = float(np.hypot(*(p - pos)))
            if best is None or dist < best[0]:
                best = (dist, rid, meta, i, t, edges)
    return None if best is None else best[1:]


def _lane_boundaries(meta, lat_left, lat_right, car_rel):
    """行驶车道的两条边界横向值（车体坐标，左为正）；等分路面。

    返回 ``(lane_left_lat, lane_right_lat, note)``；车道数缺失时按整幅路面
    一条车道处理（与采集侧一致，note 里说明）。
    """
    width = float(lat_left - lat_right)
    half = width / 2.0
    n_left = int(meta.get("lanesLeft") or 0)
    n_right = int(meta.get("lanesRight") or 0)
    mid_lat = 0.5 * (lat_left + lat_right)
    if n_left <= 0 and n_right <= 0:
        return lat_left, lat_right, "lanesLeft/Right 缺失：整幅路面按一条车道"
    if n_left <= 0 or n_right <= 0:
        total = max(1, n_left + n_right)
        lane_w = width / total
        d_from_left = float(lat_left)
        k = min(max(0, int(math.floor(max(0.0, d_from_left) / lane_w))),
                total - 1)
        return lat_left - k * lane_w, lat_left - (k + 1) * lane_w, ""
    if car_rel < 0.0:                       # 车在中线右侧
        lane_w = half / n_right
        k = min(max(0, int(math.floor(-car_rel / lane_w)) if lane_w > 0 else 0),
                n_right - 1)
        return (mid_lat - half + (k + 1) * lane_w,
                mid_lat - half + k * lane_w, "")
    lane_w = half / n_left
    k = min(max(0, int(math.floor(car_rel / lane_w)) if lane_w > 0 else 0),
            n_left - 1)
    return (mid_lat + half - k * lane_w,
            mid_lat + half - (k + 1) * lane_w, "")


def _world_at(edges, i, t, lat_off, *, lat_left, lat_right) -> list[float]:
    """在第 ``i`` 段上按参数 ``t`` 与横向值 ``lat_off`` 取世界点。

    横向插值用该处的 ``left``/``right`` 行；z 用 ``middle`` 的路面高度
    （``left``/``right`` 无 z 时不另算）。
    """
    row_a, row_b = edges[i], edges[i + 1]
    mid = _interp(row_a["middle"], row_b["middle"], t)
    lp = row_a.get("left"), row_b.get("left")
    rp = row_a.get("right"), row_b.get("right")
    if lp[0] is not None and lp[1] is not None and \
            rp[0] is not None and rp[1] is not None:
        L = _interp(lp[0], lp[1], t)
        R = _interp(rp[0], rp[1], t)
        span = float(lat_left - lat_right)
        f = 0.5 if abs(span) < 1e-9 else float(
            (lat_off - lat_right) / span)
        xy = R[:2] + f * (L[:2] - R[:2])
    else:
        # 没有左右边界行：用中线 + 法向偏移（车体左向的逆插值不可得，
        # 用该段方向构造法向）
        ab = np.asarray(row_b["middle"], dtype=float)[:2] - \
            np.asarray(row_a["middle"], dtype=float)[:2]
        n = float(np.linalg.norm(ab))
        if n < 1e-9:
            xy = mid[:2]
        else:
            nrm = np.array([-ab[1], ab[0]]) / n     # 段方向的左法向
            xy = mid[:2] + float(lat_off) * nrm
    z = float(mid[2]) if np.asarray(mid).size >= 3 else 0.0
    return [float(xy[0]), float(xy[1]), z]


def build_truth_points(conn, *, span_m: float, n_samples: int,
                       forward_skip_m: float) -> dict:
    """在车辆前方路段上采一批真值点；返回带 ``truth_points`` 的 dict。"""
    st = conn.get_state()
    pos = np.asarray(st.pos, dtype=float)
    fwd = _unit_fwd(st)
    left2d = np.array([-fwd[1], fwd[0]])          # 车体左向（左为正）

    with conn.io_lock:
        roads = conn.bng.scenario.get_road_network(include_edges=True,
                                                  drivable_only=True)
    found = _nearest_edge(roads, pos)
    if found is None:
        return {"truth_points": [], "error": "no drivable road edge near ego"}
    rid, meta, i0, t0, edges = found

    def lat_of(p_xy) -> float:
        return float((np.asarray(p_xy, dtype=float)[:2] - pos[:2]) @ left2d)

    row_a, row_b = edges[i0], edges[i0 + 1]
    L0 = _interp(row_a["left"], row_b["left"], t0) if (
        row_a.get("left") is not None and row_b.get("left") is not None) else None
    R0 = _interp(row_a["right"], row_b["right"], t0) if (
        row_a.get("right") is not None and row_b.get("right") is not None) else None
    if L0 is None or R0 is None:
        return {"truth_points": [],
                "error": "nearest edge rows have no left/right bounds"}
    lat_left, lat_right = lat_of(L0), lat_of(R0)
    if lat_left < lat_right:                       # 网络可能逆行存储
        lat_left, lat_right = lat_right, lat_left
    mid0 = _interp(row_a["middle"], row_b["middle"], t0)
    car_rel = -lat_of(mid0)
    lane_left_lat, lane_right_lat, note = _lane_boundaries(
        meta, lat_left, lat_right, car_rel)

    # 沿前进方向按弧长采样；先跳过 forward_skip_m（相机看得到的地方才有点）。
    # **方向必须按车头判定**：road network 的 edge 顺序可能与行驶方向相反
    # （实测踩到：按数组顺序走会把点采到车后，18 个真值点全部投影到相机后方，
    # 看起来像"标签错"，其实是采样方向错）。
    _ab = (np.asarray(edges[i0 + 1]["middle"], dtype=float)[:2]
           - np.asarray(edges[i0]["middle"], dtype=float)[:2])
    step_dir = 1 if float(_ab @ fwd) >= 0 else -1
    pts: list[dict] = []
    spacing = max(1.0, float(span_m) / max(1, int(n_samples) - 1))
    seg_i, seg_t = i0, t0
    walked = -float(forward_skip_m)
    hops = 0

    def _seg_len(i: int) -> float:
        if i < 0 or i + 1 >= len(edges):
            return 0.0
        a = np.asarray(edges[i]["middle"], dtype=float)
        b = np.asarray(edges[i + 1]["middle"], dtype=float)
        return float(np.linalg.norm(b[:2] - a[:2]))

    while len(pts) < int(n_samples) and hops < 60:
        L = _seg_len(seg_i)
        if L < 1e-6:
            seg_i, seg_t, hops = seg_i + step_dir, (0.0 if step_dir > 0 else 1.0), hops + 1
            if not (0 <= seg_i < len(edges) - 1):
                break
            continue
        if walked < 0.0:                       # 还没走到跳过距离
            remaining = -walked
            if remaining >= L * (1.0 - seg_t if step_dir > 0 else seg_t):
                # 整段都在跳过区内 -> 走到段末端再进下一段
                walked += L * (1.0 - seg_t if step_dir > 0 else seg_t)
                seg_i += step_dir
                seg_t = 0.0 if step_dir > 0 else 1.0
                hops += 1
                if not (0 <= seg_i < len(edges) - 1):
                    break
                continue
            seg_t += step_dir * remaining / L
            walked = 0.0
        for lat_off, cls, role in (
                (lane_left_lat, 2, "left"),
                (lane_right_lat, 2, "right"),
                (0.5 * (lane_left_lat + lane_right_lat), 1, None)):
            w = _world_at(edges, seg_i, seg_t, lat_off,
                          lat_left=lat_left, lat_right=lat_right)
            pts.append({"world": w, "class": cls, "role": role,
                        "source": "roadnet_lane_boundary" if cls == 2
                                  else "roadnet_lane_center",
                        "road_id": str(rid), "edge_index": int(seg_i),
                        # 前向距离：负值=在车后（采样方向错时必须能看出来）
                        "forward_m": round(float(
                            (np.asarray(w[:2], dtype=float) - pos[:2]) @ fwd), 2)})
        seg_t += step_dir * spacing / L
        walked += spacing
        while (step_dir > 0 and seg_t >= 1.0) or (step_dir < 0 and seg_t <= 0.0):
            seg_t -= step_dir * 1.0
            seg_i += step_dir
            hops += 1
            if not (0 <= seg_i < len(edges) - 1):
                break
        if not (0 <= seg_i < len(edges) - 1):
            break
    if not pts:
        return {"truth_points": [], "error": "sampling produced no points "
                                             "(edge too short?)"}
    behind = [p for p in pts if float(p.get("forward_m") or 0.0) < 0.0]
    if behind:
        return {"truth_points": [], "ego_pose": None,
                "error": f"{len(behind)}/{len(pts)} 个真值点落在车后：采样方向"
                         f"判定错误（step_dir={step_dir}）——不把无效几何当输入"}
    heading = float(getattr(st, "heading", 0.0))
    # 道路方向（走过的方向，3D）：探针用它对齐测试台架相机。实测踩到：车停的
    # 朝向可能与道路差 45°，按车头定向时所有真值点投影到画面外——那是台架
    # 朝向问题，不是标签问题。
    a_i = np.asarray(edges[max(0, min(seg_i, len(edges) - 2))]["middle"],
                     dtype=float)
    b_i = np.asarray(edges[max(1, min(seg_i + 1, len(edges) - 1))]["middle"],
                     dtype=float)
    d3 = (b_i - a_i) * float(step_dir)
    n3 = float(np.linalg.norm(d3))
    road_dir = [float(v) for v in (d3 / n3 if n3 > 1e-9 else d3)]
    return {"truth_points": pts,
            "road_dir": road_dir,
            "road_dir_note": ("walked road direction (world, unit); the probe "
                              "aligns its test camera to it"),
            # 生成时的自车位姿：探针必须核对"采帧时的位姿 = 生成真值时的位姿"，
            # 否则几何对不上（车被挪过/换了 spawn）会伪装成"标签错误"。
            "ego_pose": {"pos": [float(v) for v in pos[:3]],
                         "heading_rad": heading,
                         "heading_deg": math.degrees(heading)},
            "road_id": str(rid), "edge_index": int(i0),
            "lanes_left": int(meta.get("lanesLeft") or 0),
            "lanes_right": int(meta.get("lanesRight") or 0),
            "lane_width_m": round(abs(lane_left_lat - lane_right_lat), 3),
            "road_width_m": round(abs(lat_left - lat_right), 3),
            "lane_note": note,
            "generated_by": "scripts/m5_tech_truth_points.py",
            "method": ("roadnet lane boundaries (lanesLeft/Right equal split) "
                       "projected to world; z from the middle row at the same "
                       "station"),
            "limits": [
                "网络给几何，不给虚实/是否被渲染：投影对不上 = 发现，不放宽判据",
                "路拱/超高忽略（z 取中线高度）",
                "车道数缺失时整幅路面按一条车道（lane_note 里说明）",
            ]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"), default="tech")
    ap.add_argument("--attach", action="store_true",
                    help="只连已在跑的实例（默认自己启动）")
    ap.add_argument("--map", default="italy")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--span-m", type=float, default=30.0, help="采样前向跨度")
    ap.add_argument("--n-samples", type=int, default=7)
    ap.add_argument("--forward-skip-m", type=float, default=4.0,
                    help="跳过车头这段距离再开始采（相机最近可见范围）")
    ap.add_argument("--step", type=int, default=5, help="采样前推进的仿真步")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conn = BeamNGConnector(str(args.map), "etk800",
                           port=config.runtime_port(args.runtime),
                           home=config.runtime_home(args.runtime))
    out = Path(args.out)
    conn.open(launch=not args.attach)
    try:
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:                                  # noqa: BLE001
            conn.load_scenario()
        if args.teleport:
            x, y, yaw = args.teleport
            conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
        if int(args.step) > 0:
            conn.step(int(args.step))
        blob = build_truth_points(conn, span_m=float(args.span_m),
                                  n_samples=int(args.n_samples),
                                  forward_skip_m=float(args.forward_skip_m))
    finally:
        try:
            conn.close()
        except Exception:                                  # noqa: BLE001
            pass
    blob.update({"map": str(args.map), "segment": str(args.map),
                 "scene_seed": 0, "generated_at": time.strftime(
                     "%Y-%m-%dT%H:%M:%S")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    n_line = sum(1 for p in blob.get("truth_points") or []
                 if int(p.get("class") or 0) == 2)
    n_road = sum(1 for p in blob.get("truth_points") or []
                 if int(p.get("class") or 0) == 1)
    print(f"[truth] {out}: 线点 {n_line} / 路面点 {n_road}"
          f"（road_id={blob.get('road_id')}, "
          f"车道 {blob.get('lanes_left')}+{blob.get('lanes_right')}, "
          f"宽 {blob.get('road_width_m')} m）")
    if blob.get("error"):
        print(f"[truth] 失败：{blob['error']}")
        return 1
    return 0 if n_line and n_road else 1


if __name__ == "__main__":
    raise SystemExit(main())
