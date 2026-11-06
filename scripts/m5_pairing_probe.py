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


def _apply_environment(conn, tod: str | None, weather: str | None) -> dict:
    """应用并读回条件；失败/读回缺失都返回 invalid，不伪造成功。"""
    rec = {"tod_requested": tod, "weather_requested": weather,
           "tod_after": None, "weather_after": None,
           "weather_readback": "UNKNOWN", "valid": False}
    try:
        if weather:
            conn.bng.env.set_weather_preset(weather)
        if tod:
            parts = str(tod).split(":")
            hms = f"{int(parts[0]):02d}:{int(parts[1]) if len(parts) > 1 else 0:02d}:00"
            conn.bng.env.set_tod(tod=hms)
        read = conn.bng.env.get_tod() or {}
        rec["tod_after"] = str(read.get("timeStr") or "") or None
        # 无论是否指定天气预设，缺少独立天气读回都不能声称 clear/rain/fog。
        # 时刻一致性单独记录；正式环境资格必须在天气也可验证后重定义。
        rec["tod_matches"] = bool(tod and rec["tod_after"] == f"{tod}:00")
        rec["weather_after"] = None
        rec["valid"] = False
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def collect(args) -> int:
    if getattr(args, "kinematic_capture", False):
        # Block before connecting: a free port / existing process is not proof
        # of ownership, and environment and pose provenance are not verified.
        print("[pairing-probe] BLOCKED: 运动学采集尚未验证环境和进程归属", file=sys.stderr)
        return 3
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
            # 关键：按地图切换材质档案（采集器在入口做；探针此前漏了，
            # east/west 的贴花因此用 italy 材质 → 引擎找不到 → 贴花不可见）。
            _ov = {}
            for k in ("white", "yellow", "pave", "gravel"):
                v = getattr(args, f"mat_{k}", None)
                if v:
                    _ov[k] = v
            _profile = collector.apply_map_profile(str(args.map), overrides=_ov or None)
            print(f"[pairing-probe] material profile: "
                  f"white={_profile.get('white')} "
                  f"yellow={_profile.get('yellow')} "
                  f"width={_profile.get('line_width')}", flush=True)
            site = _site_from_scene(scene_doc)
            scen, _rec = collector.build_scenario(
                conn, [site], map_name=args.map,
                scene_types=[str(args.scene_type)], scene_names=[scene_name],
                line_convention=str(args.line_convention),
                edge_margin_m=float(args.edge_margin_m))
            _built_lines = (_rec.get("sites", {}).get(scene_name, {})
                            .get("lines") or [])
            try:
                # 构建记录落盘：导出侧必须用**实际生成的节点**做真值，不能
                # 重新推算（否则 right_lane 的边距/材质差异不可追）。
                (out.parent / (out.stem + ".built.json")).write_text(
                    json.dumps({"scene": scene_name, "lines": _built_lines},
                               ensure_ascii=False, indent=1),
                    encoding="utf-8")
            except (OSError, ValueError, TypeError):
                pass
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
            # GROUND_ALIGN（与采集器一致）：east/west 的地图路面 z 与 roadnet
            # 的中间 z 可差数十米；不对齐会让贴花埋在真实地面下（live 帧在
            # GT 位置只有沥青灰），ego 却被 ensure_above_ground 抬起。
            _ga = {"applied": False}
            if str(args.map) in getattr(collector, "GROUND_ALIGN_MAPS", set()):
                try:
                    _gz = conn.ground_z_at(float(site["mid"][0]),
                                           float(site["mid"][1]))
                except Exception:  # noqa: BLE001
                    _gz = None
                _delta = (None if _gz is None
                          else float(_gz) - float(site["mid"][2]))
                _ga = {"applied": False, "ground_z": _gz,
                       "road_z": float(site["mid"][2]), "delta_m": _delta}
                if _delta is not None and abs(_delta) > 2.0:
                    collector._align_sites_to_ground(conn, [site], str(args.map))
                    scen, _rec = collector.build_scenario(
                        conn, [site], map_name=args.map,
                        scene_types=[str(args.scene_type)],
                        scene_names=[scene_name],
                        line_convention=str(args.line_convention),
                        edge_margin_m=float(args.edge_margin_m))
                    with conn.io_lock:
                        scen.make(conn.bng)
                        conn.bng.scenario.load(scen)
                        conn.bng.scenario.start()
                    time.sleep(2.0)
                    conn.attach_vehicle(already_open=True)
                    _ga["applied"] = True
            print(f"[pairing-probe] scenario loaded: {scen.name} "
                  f"(lines={len(site and scene_doc['scene'].get('lines') or [])}) "
                  f"ground_align={_ga}", flush=True)
        if getattr(args, "kinematic_capture", False):
            # 运动学步进采集：不启动 fsd_drive，与模型能力完全解耦。
            _env = _apply_environment(conn, args.tod, args.weather_preset)
            if not _env["valid"]:
                print(f"[pairing-probe] BLOCKED: environment readback invalid {_env}",
                      file=sys.stderr)
                out.write_text(json.dumps({"schema": "pairing-probe-v2",
                                           "environment": _env}, ensure_ascii=False),
                               encoding="utf-8")
                return 3
            _kdir = (out.with_suffix(".frames")
                     if getattr(args, "save_frames", False) else out.parent)
            _k = kinematic_capture(
                conn, scene_doc, _kdir,
                step_m=float(args.kinematic_step_m),
                span_m=float(args.kinematic_span_m),
                lateral_jitter=float(args.kinematic_lateral_jitter),
                yaw_jitter_deg=float(args.kinematic_yaw_jitter),
                settle_s=float(args.kinematic_settle_s))
            print(f"[pairing-probe] kinematic capture: {_k}", flush=True)
            report = {
                "schema": "pairing-probe-v2",
                "scene": str(args.scene),
                "map": args.map,
                "scenario_mode": True,
                "kinematic": _k,
                "tod": args.tod,
                "weather_preset": args.weather_preset,
                "frames": int(_k.get("saved", 0)),
            }
            out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            return 0
        cmd = [sys.executable, str(ROOT / "scripts" / "m5_fsd_drive.py"),
               "--runtime", args.runtime, "--attach",
               "--seconds", f"{float(args.drive_seconds):.1f}",
               "--max-wall-s", f"{wall_s:.1f}",
               "--seg-model", args.model,
               "--pairing-probe-out", str(jsonl),
               "--out", str(out.with_suffix(".telemetry.json"))]
        if not args.no_goal:
            cmd += ["--goal", f"{gx:.3f}", f"{gy:.3f}"]
        if getattr(args, "post_norm_width", None):
            cmd += ["--seg-post-norm-width", str(int(args.post_norm_width))]
        if getattr(args, "model2", None):
            cmd += ["--seg-model2", str(args.model2)]
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
        "material_profile": (_profile if not args.no_scenario else None),
        "model": args.model,
        "model2": getattr(args, "model2", None),
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


def kinematic_capture(conn, scene_doc: dict, out_dir: Path, *,
                      step_m: float = 1.0, span_m: float = 30.0,
                      lateral_jitter: float = 0.25,
                      yaw_jitter_deg: float = 2.0,
                      settle_s: float = 0.6,
                      frames_per_station: int = 1) -> dict:
    """沿站点中心线逐站传送取帧（不跑 FSD 驾驶）。

    与 FSD 采集的区别：这里的取帧**不依赖模型**——哪怕全黑画面、模型 0 检出，
    也能稳定采满整条里程。位姿多样性由 step_m 与两个 jitter 保证，导出侧
    再用"相邻入选帧位姿距离 >= 0.5 m"筛掉重复。
    """
    import time as _time
    import numpy as _np

    sc = scene_doc["scene"]
    mid = _np.asarray(sc["mid"], dtype=float)
    d = _np.asarray(sc["dir"], dtype=float)[:2]
    d = d / max(1e-9, float(_np.linalg.norm(d)))
    left = _np.array([-d[1], d[0]])
    yaw0 = float(_np.degrees(_np.arctan2(d[1], d[0])))

    try:
        from beamng_autopilot_tech.providers import TechCameraRingProvider
        ring = TechCameraRingProvider(conn, width=536, height=403,
                                      roles=("front_main",))
    except Exception:  # noqa: BLE001
        ring = None

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(20261106)
    saved = 0
    stations = []
    s_m = 4.0
    while s_m <= float(span_m):
        stations.append(s_m)
        s_m += float(step_m)

    for s_m in stations:
        lat = float(rng.uniform(-lateral_jitter, lateral_jitter))
        dyaw = float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        p = mid.copy()
        p[:2] = mid[:2] + d * s_m + left * lat
        heading_deg = float(yaw0 + dyaw)
        ok = conn.safe_teleport(float(p[0]), float(p[1]),
                                heading_deg=heading_deg, settle=6)
        if not ok:
            continue
        _time.sleep(float(settle_s))
        for _k in range(int(frames_per_station)):
            st = conn.get_state()
            frame = None
            cam = None
            if ring is not None:
                try:
                    snap = ring.grab_ring()
                    if snap:
                        role = ("front_main" if "front_main" in snap
                                else next(iter(snap)))
                        frame, cam = snap[role]
                except Exception:  # noqa: BLE001
                    frame = None
            if frame is None or cam is None:
                continue
            path = out_dir / f"frame_{saved:05d}.npz"
            _np.savez_compressed(
                path, frame=_np.asarray(frame, dtype=_np.uint8),
                pos=_np.asarray(st.pos, dtype=float),
                heading=float(st.heading),
                cam=_np.array([cam], dtype=object),
                t=float(_time.time()))
            saved += 1
    if ring is not None:
        try:
            ring.close()
        except Exception:  # noqa: BLE001
            pass
    return {"saved": saved, "stations": len(stations),
            "step_m": float(step_m), "span_m": float(span_m),
            "lateral_jitter_m": float(lateral_jitter),
            "yaw_jitter_deg": float(yaw_jitter_deg)}


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
    ap.add_argument("--model2", default=None,
                    help="部署集成的第二模型（与 --model 组成 EnsembleSegmenter）")
    ap.add_argument("--tod", default=None)
    ap.add_argument("--weather-preset", default=None)
    ap.add_argument("--frames", type=int, default=40,
                    help="最多分析的 tick 数（采集时按此估算墙钟）")
    ap.add_argument("--drive-seconds", type=float, default=1.0,
                    help="场景模式下 drive 的持续秒数（默认 1s=只覆盖放置；"
                         "采集扩池用 20-60s 以获得更多保存帧）")
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
    ap.add_argument("--mat-white", default=None)
    ap.add_argument("--mat-yellow", default=None)
    ap.add_argument("--mat-pave", default=None)
    ap.add_argument("--mat-gravel", default=None)
    ap.add_argument("--scene-type", default="known_line",
                    help="场景类型（known_line / known_no_line）——A/B 渲染校验用")
    ap.add_argument("--edge-margin-m", type=float, default=0.6,
                    help="right_lane 约定下右线距铺装边的余量（米）")
    ap.add_argument("--line-convention", default="right_lane",
                    help="贴花横向约定：right_lane=贴右侧铺装边（配对专用）；"
                         "symmetric=原认证约定")
    ap.add_argument("--no-scenario", action="store_true",
                    help="不构建贴花 Scenario（仅调试；认证线召回必然为 0）")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--kinematic-capture", action="store_true",
                    help="运动学步进采集：沿站点中心线逐站传送取帧，"
                         "不跑 FSD 驾驶（与模型能否看见线完全解耦）")
    ap.add_argument("--kinematic-step-m", type=float, default=1.0,
                    help="运动学采集的纵向步长（米）")
    ap.add_argument("--kinematic-span-m", type=float, default=30.0,
                    help="运动学采集的纵向跨度（米）")
    ap.add_argument("--kinematic-lateral-jitter", type=float, default=0.25,
                    help="每站横向抖动幅度（米，正负对称）")
    ap.add_argument("--kinematic-yaw-jitter", type=float, default=2.0,
                    help="每站偏航抖动幅度（度，正负对称）")
    ap.add_argument("--kinematic-settle-s", type=float, default=0.6,
                    help="每站传送后静置时间（秒）")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"))
    args = ap.parse_args()
    if args.analyze:
        return analyze(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
