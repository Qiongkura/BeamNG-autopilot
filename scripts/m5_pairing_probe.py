"""配对鲁棒性研究臂探针（预注册 T16_LANE_PAIRING_ROBUSTNESS_PREREG_20261104）。

两种用法：

1) 采集（需要 BeamNG.tech；只放置、不驾驶）：
   .venv\\Scripts\\python.exe scripts\\m5_pairing_probe.py \\
       --scene logs\\experiments\\t16_scenes_20261102_181708\\scene_known_line_a1s0.json \\
       --map italy --model logs\\m5_seg\\seg_model_v28_sslA\\best.pt \\
       --tod 12:00 --frames 40 --out logs\\m5_seg\\pairing_probe_italy_a1s0.json

   工具按场景的站心/方向生成 teleport 位姿，复用 `m5_fsd_drive.py` 的
   严格感知放置回路（`--pairing-probe-out` 只读 JSONL），逐 tick 记录
   配对决策与拒绝计数；放置不会成功也没关系——本工具只看配对，不驾驶。

2) 离线重算（无引擎）：
   ... --analyze logs\\m5_seg\\pairing_probe_italy_a1s0.jsonl \\
       --scene <scene.json> --out <report.json>

真值：认证场景 `scene.lines` 的左右线节点（engine_verified 几何），按每条
记录的 ego 位姿投影到车体系插值出车道中心与宽度；配对结果用
`pair.result.center_xy` + `width_m` 做同样的投影。判据（预注册）：
中心/宽度误差 ≤0.75m = true；中心 >1.5m 或宽度 >1.5m = phantom；其余
ambiguous。研究开关默认全部 False（= 现行行为），只记录不生效。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.lane.pairing import DEFAULT_PAIRING_RESEARCH  # noqa: E402

CENTER_TOL_M = 0.75
PHANTOM_TOL_M = 1.5
REF_OFFSET_TOL_M = 1.0  # ego 距认证车道中心超过此值 -> 参考系无效


# ---------------------------------------------------------------------------
# 纯函数（单测覆盖）
# ---------------------------------------------------------------------------
def load_scene(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = d.get("scene") or {}
    if "mid" not in scene or "dir" not in scene:
        raise ValueError(f"场景缺少 mid/dir: {path}")
    return d


def scene_pose(scene_doc: dict) -> tuple[float, float, float]:
    """场景站心 + 方向 -> (x, y, yaw_deg)。yaw 供 teleport 使用；真值
    计算只依赖实际位姿（JSONL 里的 pos/heading），不依赖该 yaw。"""
    s = scene_doc["scene"]
    x, y = float(s["mid"][0]), float(s["mid"][1])
    dx, dy = float(s["dir"][0]), float(s["dir"][1])
    return x, y, math.degrees(math.atan2(dy, dx))


def scene_goal(scene_doc: dict, ahead_m: float = 60.0) -> tuple[float, float]:
    s = scene_doc["scene"]
    return (float(s["mid"][0]) + float(s["dir"][0]) * ahead_m,
            float(s["mid"][1]) + float(s["dir"][1]) * ahead_m)


def scene_lines(scene_doc: dict) -> dict[str, np.ndarray]:
    """从场景规格取左右线节点（世界 2D）。"""
    out: dict[str, np.ndarray] = {}
    for ln in (scene_doc.get("scene") or {}).get("lines") or []:
        role = str(ln.get("role") or "")
        nodes = ln.get("nodes") or []
        if role in ("left", "right") and len(nodes) >= 2:
            out[role] = np.asarray(nodes, dtype=float)[:, :2]
    return out


def _side_lat(nodes: np.ndarray, ego: np.ndarray, fwd: np.ndarray,
              left: np.ndarray) -> float | None:
    """线节点在 ego 体系里 s=0 处的横向坐标（左正）。"""
    if nodes is None or len(nodes) == 0:
        return None
    rel = nodes - ego
    s = rel @ fwd
    lat = rel @ left
    order = np.argsort(s)
    s, lat = s[order], lat[order]
    if len(s) >= 2 and s[0] <= 0.0 <= s[-1]:
        return float(np.interp(0.0, s, lat))
    return float(lat[int(np.argmin(np.abs(s)))])


def gt_lane(scene_doc: dict, ego_xy, heading: float):
    """-> (center_lat, width) 或 None（缺左右真值）。"""
    lines = scene_lines(scene_doc)
    if "left" not in lines or "right" not in lines:
        return None
    ego = np.asarray(ego_xy, dtype=float)[:2]
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-fwd[1], fwd[0]])
    ll = _side_lat(lines["left"], ego, fwd, left)
    rl = _side_lat(lines["right"], ego, fwd, left)
    if ll is None or rl is None:
        return None
    return (0.5 * (ll + rl), abs(ll - rl))


def lateral_of(xy, ego_xy, heading: float) -> float:
    ego = np.asarray(ego_xy, dtype=float)[:2]
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-fwd[1], fwd[0]])
    return float((np.asarray(xy, dtype=float)[:2] - ego) @ left)


def classify_pair(center_lat: float, width: float,
                  gt_center: float, gt_width: float) -> dict:
    ce = abs(float(center_lat) - float(gt_center))
    we = abs(float(width) - float(gt_width))
    if ce <= CENTER_TOL_M and we <= CENTER_TOL_M:
        label = "true"
    elif ce > PHANTOM_TOL_M or we > PHANTOM_TOL_M:
        label = "phantom"
    else:
        label = "ambiguous"
    return {"center_err_m": round(ce, 3), "width_err_m": round(we, 3),
            "label": label}


def gt_side_lats(scene_doc: dict, ego_xy, heading: float):
    """-> (left_lat, right_lat) 或 None（左正约定）。"""
    lines = scene_lines(scene_doc)
    if "left" not in lines or "right" not in lines:
        return None
    ego = np.asarray(ego_xy, dtype=float)[:2]
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-fwd[1], fwd[0]])
    ll = _side_lat(lines["left"], ego, fwd, left)
    rl = _side_lat(lines["right"], ego, fwd, left)
    if ll is None or rl is None:
        return None
    return (ll, rl)


def cert_line_recall(records: list[dict], scene_doc: dict,
                     tol_m: float = 0.6) -> dict:
    """认证线召回：候选横向坐标是否落在 GT 左右线 ±tol 内。

    这是比配对门更上游的指标：薄线若根本没成为候选，F1/F2/M1 都无从生效。
    """
    ticks = 0
    left_ok = right_ok = both_ok = 0
    for rec in records:
        pair = rec.get("pair") or {}
        cands = pair.get("cands") or []
        gt = gt_side_lats(scene_doc, rec.get("pos"), float(rec.get("heading")))
        if gt is None:
            continue
        ticks += 1
        lats = [float(c["med_lat"]) for c in cands
                if c.get("med_lat") is not None]
        l = any(abs(x - gt[0]) <= tol_m for x in lats)
        r = any(abs(x - gt[1]) <= tol_m for x in lats)
        left_ok += int(l)
        right_ok += int(r)
        both_ok += int(l and r)
    return {
        "ticks_with_gt": ticks,
        "left_hit_rate": round(left_ok / max(1, ticks), 4),
        "right_hit_rate": round(right_ok / max(1, ticks), 4),
        "both_hit_rate": round(both_ok / max(1, ticks), 4),
    }


def summarize(records: list[dict], scene_doc: dict) -> dict:
    """逐 tick 记录 -> 配对/真配对/幻影率与拒绝直方图。"""
    n = len(records)
    n_mark = 0
    n_pairs = 0
    n_true = 0
    n_phantom = 0
    n_amb = 0
    n_off = 0
    rejects: dict[str, int] = {}
    modes: dict[str, int] = {}
    collect_drops: dict[str, int] = {}
    thin_side = 0
    per_frame = []
    for rec in records:
        pair = rec.get("pair") or {}
        if int(pair.get("marking_count") or 0) > 0:
            n_mark += 1
        mode = pair.get("mode") or "none"
        modes[mode] = modes.get(mode, 0) + 1
        for k, v in (pair.get("rejects") or {}).items():
            rejects[k] = rejects.get(k, 0) + int(v)
        for k, v in (pair.get("collect_drops") or {}).items():
            collect_drops[k] = collect_drops.get(k, 0) + int(v)
        thin_side += int((pair.get("collect_drops") or {}).get("thin:side", 0))
        res = pair.get("result")
        if not res or res.get("center_xy") is None or res.get("width_m") is None:
            per_frame.append({"t": rec.get("t"), "label": "unpaired"})
            continue
        n_pairs += 1
        gt = gt_lane(scene_doc, rec["pos"], float(rec["heading"]))
        if gt is None:
            per_frame.append({"t": rec.get("t"), "label": "no_gt"})
            continue
        cl = lateral_of(res["center_xy"], rec["pos"], float(rec["heading"]))
        # 直接按配对中心 vs 认证车道中心判：right_lane 场景里自车可能被
        # nav 路由拉回道路中线（认证车道合法偏 1.7m），不应因此拒判。
        cls = classify_pair(cl, float(res["width_m"]), gt[0], gt[1])
        per_frame.append({"t": rec.get("t"), **cls})
        if cls["label"] == "true":
            n_true += 1
        elif cls["label"] == "phantom":
            n_phantom += 1
        else:
            n_amb += 1
    return {
        "ticks": n,
        "ticks_with_candidates": n_mark,
        "accepted_pairs": n_pairs,
        "pair_rate_of_candidates": round(n_pairs / max(1, n_mark), 4),
        "true_pairs": n_true,
        "phantom_pairs": n_phantom,
        "ambiguous_pairs": n_amb,
        "off_reference_pairs": n_off,
        "off_reference_rate_of_pairs": round(n_off / max(1, n_pairs), 4),
        "true_pair_rate_of_pairs": round(n_true / max(1, n_pairs), 4),
        "phantom_rate_of_pairs": round(n_phantom / max(1, n_pairs), 4),
        "true_pair_rate_of_candidates": round(n_true / max(1, n_mark), 4),
        "phantom_rate_of_candidates": round(n_phantom / max(1, n_mark), 4),
        "rejects": dict(sorted(rejects.items())),
        "modes": dict(sorted(modes.items())),
        "collect_drops": dict(sorted(collect_drops.items())),
        "collect_by_kind": _by_kind(collect_drops),
        "thin_span_drops": int(collect_drops.get("thin:span", 0)),
        "thin_span_drop_rate_of_ticks": round(
            collect_drops.get("thin:span", 0) / max(1, n), 4),
        "thin_side_drops": thin_side,
        "thin_side_drop_rate_of_ticks": round(thin_side / max(1, n), 4),
        "cert_line_recall": cert_line_recall(records, scene_doc),
        "per_frame": per_frame,
    }


def _arm_env(arm: str) -> dict:
    """研究臂 -> 只读环境开关（默认不改任何生产阈值）。"""
    import os
    env = os.environ.copy()
    if arm == "yellow":
        env["BEAMNG_YELLOW_GATE"] = "rel"
    elif arm == "span":
        env["BEAMNG_THIN_SPAN"] = "relax"
    elif arm == "clahe":
        env["BEAMNG_CAND_CLAHE"] = "1"
    return env


def _arm_policy(arm: str):
    from beamng_autopilot.lane.pairing import PairingResearchPolicy
    return PairingResearchPolicy(
        far_thin_allow=(str(arm) == "far_thin"),
        thin_near_relax=(str(arm) == "thin_near"))


def _by_kind(drops: dict[str, int]) -> dict[str, dict[str, int]]:
    """{"thin:span": 3, ...} -> {"thin": {"span": 3}, ...}（只读归因）。"""
    out: dict[str, dict[str, int]] = {}
    for key, val in (drops or {}).items():
        kind, _, reason = str(key).partition(":")
        out.setdefault(kind or "?", {})[reason or "?"] = int(val)
    return out


def read_jsonl(path: Path) -> list[dict]:
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            continue
    return recs


# ---------------------------------------------------------------------------
# 采集（engine）：自带引擎 -> 构建并加载贴花 Scenario -> drive --attach
# ---------------------------------------------------------------------------
def _load_collector():
    """复用采集器的场景构建器（同一脚本模块，避免复制场景定义）。"""
    import importlib.util
    entry = ROOT / "scripts" / "m5_controlled_scenes.py"
    spec = importlib.util.spec_from_file_location("m5_controlled_scenes", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_controlled_scenes"] = mod
    spec.loader.exec_module(mod)
    return mod


def _site_from_scene(scene_doc: dict) -> dict:
    s = scene_doc["scene"]
    return {
        "station_m": float(s.get("station_m", 0.0)),
        "road_id": str(s.get("road_id", "")),
        "mid": [float(x) for x in s["mid"]],
        "dir": [float(x) for x in s["dir"]],
        "half_width_m": float(s.get("half_width_m", 2.5)),
        "slope_pct": float(s.get("slope_pct", 0.0)),
        "curve_deg": float(s.get("curve_deg", 0.0)),
    }


def collect(args) -> int:
    import time

    from beamng_autopilot import config
    from beamng_autopilot.connector import BeamNGConnector
    from beamng_autopilot.experiments.collection import (
        close_started_game, game_pids)

    scene_doc = load_scene(Path(args.scene))
    scene_name = Path(args.scene).stem
    if scene_name.startswith("scene_"):
        scene_name = scene_name[len("scene_"):]
    gx, gy = scene_goal(scene_doc, ahead_m=float(args.goal_ahead_m))
    if args.goal:
        gx, gy = args.goal

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl) if args.jsonl else out.with_suffix(".jsonl")
    if jsonl.exists():
        jsonl.unlink()

    wall_s = 90.0 + float(args.frames) * 0.8 + 30.0
    pids_before = game_pids()
    conn = BeamNGConnector(args.map, "etk800",
                           port=config.runtime_port(args.runtime),
                           home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=True)
        conn.load_scenario()
        if not args.no_scenario:
            collector = _load_collector()
            site = _site_from_scene(scene_doc)
            scen, _rec = collector.build_scenario(
                conn, [site], map_name=args.map,
                scene_types=["known_line"], scene_names=[scene_name],
                line_convention=str(args.line_convention),
                edge_margin_m=float(args.edge_margin_m))
            _built_lines = (_rec.get("sites", {}).get(scene_name, {})
                            .get("lines") or [])
            if _built_lines:
                # GT 用构建记录里的真实节点（right_lane 约定下与规格不同）
                scene_doc = dict(scene_doc)
                scene_doc["scene"] = dict(scene_doc["scene"])
                scene_doc["scene"]["lines"] = [
                    {"role": ln.get("role"), "nodes": ln.get("nodes")}
                    for ln in _built_lines]
            with conn.io_lock:
                scen.make(conn.bng)
                conn.bng.scenario.load(scen)
                conn.bng.scenario.start()
            time.sleep(2.0)
            conn.attach_vehicle(already_open=True)
            print(f"[pairing-probe] scenario loaded: {scen.name} "
                  f"(lines={len(site and scene_doc['scene'].get('lines') or [])})",
                  flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "m5_fsd_drive.py"),
               "--runtime", args.runtime, "--attach",
               "--seconds", "1", "--max-wall-s", f"{wall_s:.1f}",
               "--seg-model", args.model,
               "--pairing-probe-out", str(jsonl),
               "--out", str(out.with_suffix(".telemetry.json"))]
        if not args.no_goal:
            cmd += ["--goal", f"{gx:.3f}", f"{gy:.3f}"]
        if getattr(args, "post_norm_width", None):
            cmd += ["--seg-post-norm-width", str(int(args.post_norm_width))]
        _frames_dir = None
        if getattr(args, "save_frames", False):
            _frames_dir = str(out.with_suffix(".frames"))
            cmd += ["--pairing-probe-frames-dir", _frames_dir]
        if str(args.arm) in ("far_thin", "thin_near"):
            cmd += ["--pairing-research", str(args.arm)]
        env = _arm_env(str(args.arm))
        if args.tod:
            cmd += ["--tod", str(args.tod)]
        if args.weather_preset:
            cmd += ["--weather-preset", str(args.weather_preset)]
        print(f"[pairing-probe] scene={Path(args.scene).name} "
              f"scenario_mode={not args.no_scenario} frames<={args.frames}",
              flush=True)
        log = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=wall_s + 180, env=env)
        (out.with_suffix(".run.log")).write_text(
            (log.stdout or "") + "\n--- stderr ---\n" + (log.stderr or ""),
            encoding="utf-8")
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            close_started_game(pids_before)
        except Exception:  # noqa: BLE001
            pass

    if not jsonl.exists():
        print("[pairing-probe] 探针 JSONL 未生成；见 "
              f"{out.with_suffix('.run.log')}", flush=True)
        return 2
    records = read_jsonl(jsonl)[: int(args.frames)]
    report = {
        "schema": "pairing-probe-v2",
        "scene": str(args.scene),
        "map": args.map,
        "scenario_mode": not args.no_scenario,
        "model": args.model,
        "tod": args.tod,
        "weather_preset": args.weather_preset,
        "goal": [gx, gy],
        "post_norm_width": getattr(args, "post_norm_width", None),
        "frames_dir": (_frames_dir if getattr(args, "save_frames", False)
                       else None),
        "policy": _arm_policy(args.arm).as_dict(),
        "policy_is_default": _arm_policy(args.arm).is_default(),
        "arm": str(args.arm),
        "summary": summarize(records, scene_doc),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    s = report["summary"]
    print(f"[pairing-probe] ticks={s['ticks']} cand={s['ticks_with_candidates']} "
          f"pairs={s['accepted_pairs']} true={s['true_pairs']} "
          f"phantom={s['phantom_pairs']} amb={s['ambiguous_pairs']} "
          f"cert_both={s['cert_line_recall']['both_hit_rate']} "
          f"rejects={s['rejects']}", flush=True)
    print(f"[pairing-probe] -> {out}", flush=True)
    return 0


def analyze(args) -> int:
    scene_doc = load_scene(Path(args.scene))
    # 与采集口径对齐：只分析前 --frames 个 tick
    records = read_jsonl(Path(args.analyze))[: int(args.frames)]
    report = {
        "schema": "pairing-probe-v1",
        "scene": str(args.scene),
        "jsonl": str(args.analyze),
        "policy": DEFAULT_PAIRING_RESEARCH.as_dict(),
        "policy_is_default": DEFAULT_PAIRING_RESEARCH.is_default(),
        "summary": summarize(records, scene_doc),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="认证场景 scene_*.json")
    ap.add_argument("--analyze", default=None,
                    help="离线重算已采集的 JSONL（不需要引擎）")
    ap.add_argument("--jsonl", default=None,
                    help="采集时的 JSONL 路径（默认与 --out 同名 .jsonl）")
    ap.add_argument("--out", required=True, help="报告 JSON 路径")
    ap.add_argument("--map", default="italy")
    ap.add_argument("--runtime", default="tech", choices=("tech", "steam", "auto"))
    ap.add_argument("--model", default="logs/m5_seg/seg_model_v28_sslA/best.pt")
    ap.add_argument("--tod", default=None)
    ap.add_argument("--weather-preset", default=None)
    ap.add_argument("--frames", type=int, default=40,
                    help="最多分析的 tick 数（采集时按此估算墙钟）")
    ap.add_argument("--goal-ahead-m", type=float, default=60.0)
    ap.add_argument("--no-goal", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="不建导航路线（避免 route snap 把自车从车道中心拉回"
                         "道路中线；配对探针默认用它）")
    ap.add_argument("--arm",
                    choices=("baseline", "far_thin", "thin_near",
                             "yellow", "span", "clahe"),
                    default="baseline",
                    help="研究臂：far_thin/thin_near 走 --pairing-research；"
                         "yellow/span/clahe 走环境变量（Y1/S1/I1）")
    ap.add_argument("--save-frames", action="store_true",
                    help="回放诊断：逐 tick 保存 live 帧+位姿+相机模型")
    ap.add_argument("--post-norm-width", type=int, default=None,
                    help="研究开关：透传 drive 的 --seg-post-norm-width")
    ap.add_argument("--edge-margin-m", type=float, default=0.6,
                    help="right_lane 约定下右线距铺装边的余量（米）")
    ap.add_argument("--line-convention", default="right_lane",
                    help="贴花横向约定：right_lane=贴右侧铺装边（配对专用）；"
                         "symmetric=原认证约定")
    ap.add_argument("--no-scenario", action="store_true",
                    help="不构建贴花 Scenario（仅调试；认证线召回必然为 0）")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    args = ap.parse_args()
    if args.analyze:
        return analyze(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
