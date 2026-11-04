"""FSD 条件矩阵（预注册 T16_FSD_CONDITIONS_MATRIX_PREREG_20261102）。

整车主线 ③ 的第一场战役：固定 mountain 场景（italy，README 实车记录的
hairpin 起点），只在环境条件上变化——4 时刻 × 3 天气，共 12 格 + 8 个
严酷格重复 = 20 轮驾驶。回答"真车能力"最基本的问题：**栈在什么条件下
还能开、什么条件下 fail-close**。

三种模式（与 m5_fsd_benchmark 同构）::

    # 计划打印（无引擎）
    .venv\\Scripts\\python.exe scripts\\m5_fsd_conditions.py --plan

    # 跑矩阵（每格一个独立引擎会话；已存在的输出跳过，可断点续跑）
    .venv\\Scripts\\python.exe scripts\\m5_fsd_conditions.py --run \\
        --preset-rain <名字> --preset-fog <名字>

    # 只聚合已有遥测（无引擎）
    .venv\\Scripts\\python.exe scripts\\m5_fsd_conditions.py --score \\
        logs\\fsd_conditions

纪束：产物只落 logs/；探针/诊断不进训练池；不降门、不晋级。环境应用
失败（applied=False）的格聚合为 UNKNOWN，不静默丢弃。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config
from beamng_autopilot.eval import STATUS_UNKNOWN, assess_run, score_run

# 固定场景（= m5_fsd_benchmark.SCENARIOS["mountain"]，保证与既有验收
# 记录可比）：italy hairpin 起点 + 沿路图 ~200 m 的 goal；分割模型沿用
# benchmark 的 v8 pin（山道域客观指标最好的模型）。
SCENARIO = {
    # 2026-11-04 预注册修订案 A3：mountain 无标线、town 单侧（含集成），
    # 改用 west a3s0 站点（地图自带标线、live 配对率 1.00、放置 1/2）。
    "map": "west_coast_usa",
    "teleport": (-488.26, -258.82, -2.28),
    "goal": (-485.87, -318.77),
    "seconds": 60.0,
    "speed": 6.0,
    # 2026-11-02 预注册修订案：原 pin（benchmark 的 seg_model_v8_backup）
    # 在 2026-10-29~11-01 的 logs/m5_seg 外部删除事件中灭失，部署默认
    # seg_model/best.pt 同灭。改 pin seg_model_v28_sslA（SSL +7.5% 水位
    # 模型、种子 42；fold A 留出=早期 utah，训练集含 italy——矩阵场景在
    # italy）。20 格全程单一固定 pin，不做格间换模型。
    "seg_model": "logs/m5_seg/seg_model_v28_sslA/best.pt",
}

# 时刻维度：黎明/正午/黄昏/夜间（EnvironmentApi.set_tod，"HH:MM"）。
TODS: dict[str, str] = {
    "dawn": "06:30",
    "noon": "12:00",
    "dusk": "18:30",
    "night": "21:30",
}

# 天气维度：clear = 引擎默认（不调用预设，避免依赖预设库存）；rain/fog =
# 预设名，由引擎探测（logs/probe_weather_presets.py）在跑批前解析后经 CLI
# 传入。名字缺失时对应格不跑，聚合标 UNKNOWN（不静默降级为"晴天"）。
WEATHER_DIMS = ("clear", "rain", "fog")

# 严酷格重复（r1）：夜间全部 + 黄昏/正午/黎明的雨雾格——夜间是感知最差
# 条件，雨雾是视觉退化主轴。clear 白天格不重复（信息量最低）。
REPEAT_CELLS = {
    ("night", "clear"), ("night", "rain"), ("night", "fog"),
    ("dusk", "rain"), ("dusk", "fog"),
    ("noon", "rain"), ("noon", "fog"),
    ("dawn", "fog"),
}


def plan_cells(preset_rain: str | None, preset_fog: str | None) -> list[dict]:
    """返回矩阵的全部 (tod, weather, repeat) 格及其实参。

    rain/fog 预设名缺失的格**仍会出现在计划里**，标 ``runnable=False``——
    计划必须完整呈现预注册的矩阵，跑不了的格在聚合时标 UNKNOWN。
    """
    presets = {"clear": "", "rain": preset_rain or "", "fog": preset_fog or ""}
    cells: list[dict] = []
    for tod_name, tod in TODS.items():
        for wx_name in WEATHER_DIMS:
            repeats = 2 if (tod_name, wx_name) in REPEAT_CELLS else 1
            for r in range(repeats):
                cells.append({
                    "cell": f"{tod_name}_{wx_name}",
                    "tod_name": tod_name, "tod": tod,
                    "weather": wx_name, "weather_preset": presets[wx_name],
                    "repeat": r,
                    "runnable": bool(wx_name == "clear" or presets[wx_name]),
                })
    return cells


def cell_out_path(out_dir: Path, cell: dict) -> Path:
    return out_dir / f"{cell['cell']}_r{cell['repeat']}.json"


def _default_run_args() -> SimpleNamespace:
    """从 scripts/m5_fsd_drive.py 的 build_parser() 取完整默认实参。

    手抄字段清单会随入口演化漂移（漏一个字段就是 AttributeError 崩一轮）；
    直接用 CLI 解析器产默认 namespace，再覆盖场景/条件字段，保证与入口
    永远同构。
    """
    import importlib.util

    entry = ROOT / "scripts" / "m5_fsd_drive.py"
    spec = importlib.util.spec_from_file_location("m5_fsd_drive_cli", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_parser().parse_args([])


def build_run_args(cell: dict, out_path: Path) -> SimpleNamespace:
    """一格的 fsd_drive.run 实参：CLI 默认（sensor+strict=唯一 FSD 合规
    路径）+ benchmark mountain 场景 + 该格的环境条件。"""
    args = _default_run_args()
    args.runtime = "tech"
    args.map = SCENARIO["map"]
    args.tod = cell["tod"]
    args.weather_preset = cell["weather_preset"] or None
    args.seconds = SCENARIO["seconds"]
    args.max_wall_s = 300.0
    args.speed = SCENARIO["speed"]
    args.seg_model = SCENARIO["seg_model"]
    args.teleport = list(SCENARIO["teleport"])
    args.goal = list(SCENARIO["goal"])
    args.out = str(out_path)
    return args


def run_matrix(out_dir: Path, preset_rain: str | None,
               preset_fog: str | None) -> int:
    from beamng_autopilot.fsd_drive import run as fsd_run

    cells = plan_cells(preset_rain, preset_fog)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = build_run_args(cells[0], out_dir / "_template.json")
    manifest = {
        "schema": "fsd-conditions-manifest-v1",
        "written_at": time.time(),
        "scenarios": SCENARIO,
        "tods": TODS, "weather_dims": WEATHER_DIMS,
        "preset_rain": preset_rain or "", "preset_fog": preset_fog or "",
        "cells": [{k: v for k, v in c.items()} for c in cells],
        "config_hash": hashlib.sha256(json.dumps(
            sorted(vars(template).items(), key=lambda kv: kv[0]),
            sort_keys=True, default=str).encode()).hexdigest(),
        "cells_runnable": sum(1 for c in cells if c["runnable"]),
    }
    (out_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[conditions] plan cells={len(cells)} "
          f"runnable={manifest['cells_runnable']} -> {out_dir}", flush=True)
    bad = 0
    for c in cells:
        out_path = cell_out_path(out_dir, c)
        if not c["runnable"]:
            print(f"[conditions] SKIP {c['cell']}_r{c['repeat']} "
                  f"(weather preset 名未解析)", flush=True)
            continue
        if out_path.exists() or Path(str(out_path) + ".session.json").exists():
            # 有 session 审计即视为该格已完成（放置失败格没有遥测文件，
            # 但 session 里的 placement_rc/环境读回就是它的证据；重跑等于
            # 为了出表反复尝试，违反预注册的"至多一次协议重试"）。
            print(f"[conditions] SKIP {out_path.name} (已存在证据，断点续跑)",
                  flush=True)
            continue
        print(f"[conditions] RUN {c['cell']}_r{c['repeat']} "
              f"tod={c['tod']} wx={c['weather'] or '-'}", flush=True)
        rc = fsd_run(build_run_args(c, out_path))
        print(f"[conditions] DONE {c['cell']}_r{c['repeat']} rc={rc}", flush=True)
        if rc != 0:
            bad += 1
    print(f"[conditions] 批次完成：非零退出 {bad} 格", flush=True)
    return 0


def score_one(hist: list[dict], session: dict | None) -> dict:
    """一轮遥测 -> 该格的指标行（含 UNKNOWN 传播，绝不静默丢证据）。"""
    if not isinstance(hist, list):
        # 误把聚合表/审计当遥测传入时按空遥测处理，绝不崩
        hist = []
    assessed = assess_run(hist, goal=tuple(SCENARIO["goal"]), settle_s=8.0)
    verdict = score_run(assessed, require_goal=False)
    env = (session or {}).get("environment") or {}
    row = {
        "frames": assessed.get("frames", 0),
        "settled_frames": assessed.get("settled_frames", 0),
        "duration_s": assessed.get("duration_s", 0.0),
        "lane_sensor_rate": assessed.get("lane_sensor_rate"),
        "lane_paired_rate": assessed.get("lane_paired_rate"),
        "level_hist": assessed.get("level"),
        "cross_centre_frames": assessed.get("cross_centre_frames"),
        "cross_right_frames": assessed.get("cross_right_frames"),
        "off_road_frames": assessed.get("off_road_frames"),
        "reversing_frames": assessed.get("reversing_frames"),
        "stuck_frames": assessed.get("stuck_frames"),
        "emergency_frames": assessed.get("emergency_frames"),
        "status": verdict.get("status"),
        "unknown_checks": verdict.get("unknown", []),
        "placed": bool((session or {}).get("placed")),
        "placement_rc": (session or {}).get("placement_rc"),
        "env_applied": bool(env.get("applied")),
        "env_error": env.get("error"),
        "tod_after": env.get("tod_after"),
    }
    # UNKNOWN 判据（fail-closed）：没放上车 / 环境没应用上 / 引擎侧读回的
    # 时刻与请求不一致 / 无帧。任何一项成立，这格就不能当 PASS/FAIL 读。
    tod_req = str((session or {}).get("config", {}).get("tod") or "")
    tod_ok = (not tod_req) or (row["tod_after"] or "").startswith(tod_req)
    if (not row["placed"] or not row["env_applied"] or not tod_ok
            or int(row["frames"] or 0) == 0):
        row["status"] = STATUS_UNKNOWN
    return row


def aggregate(rows: list[dict]) -> dict:
    """格级聚合：重复轮取最差（严酷条件的可用性以最差轮为准），任何
    UNKNOWN/缺轮都把整格拉回 UNKNOWN——不降门。"""
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(str(r.get("cell")), []).append(r)
    table: dict[str, dict] = {}
    repeat_cells = {f"{t}_{w}" for t in TODS for w in WEATHER_DIMS
                    if (t, w) in REPEAT_CELLS}
    for cell, rs in sorted(by_cell.items()):
        runs = [r for r in rs if r.get("status") != STATUS_UNKNOWN]
        unknowns = [r for r in rs if r.get("status") == STATUS_UNKNOWN]
        exp = 2 if cell in repeat_cells else 1
        if unknowns or len(rs) < exp:
            # fail-closed：有 UNKNOWN 轮或轮次不全，整格只能是 UNKNOWN
            status = STATUS_UNKNOWN
        else:
            status = "PASS" if all(r.get("status") == "PASS" for r in runs) \
                else "FAIL"

        def worst(key: str):
            # 重复轮取最差：严酷条件的可用性以最差轮为准
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            if not vals:
                return None
            return min(float(v) for v in vals)

        table[cell] = {
            "runs": len(rs), "unknown_runs": len(unknowns),
            "expected_runs": exp,
            "status": status,
            "lane_sensor_rate_min": worst("lane_sensor_rate"),
            "lane_paired_rate_min": worst("lane_paired_rate"),
            "cross_centre_frames_max": max(
                (int(r.get("cross_centre_frames") or 0) for r in runs),
                default=None),
            "off_road_frames_max": max(
                (int(r.get("off_road_frames") or 0) for r in runs),
                default=None),
            "stuck_frames_max": max(
                (int(r.get("stuck_frames") or 0) for r in runs), default=None),
        }
    return table


def expected_cells(manifest: dict | None) -> list[tuple[str, int]]:
    """从 batch_manifest 播种期望格（cell, repeat）。

    放置失败的格只有 session 审计没有遥测文件——如果只按遥测 glob 聚合，
    这些格会从表里静默消失（违反不丢证据）。以预注册的 manifest 为准。
    """
    out: list[tuple[str, int]] = []
    for c in (manifest or {}).get("cells", []):
        try:
            out.append((str(c["cell"]), int(c["repeat"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def score_dir(out_dir: Path) -> int:
    manifest = None
    mf = out_dir / "batch_manifest.json"
    if mf.exists():
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
    rows: list[dict] = []
    seen: set[str] = set()
    for path in sorted(out_dir.glob("*.json")):
        name = path.name
        if name.endswith(".session.json") or name.endswith(".vis_index.json") \
                or name in ("batch_manifest.json", "conditions_table.json"):
            continue
        try:
            hist = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[score] 跳过损坏遥测 {name}: {exc}", flush=True)
            continue
        session_p = Path(str(path) + ".session.json")
        session = None
        if session_p.exists():
            try:
                session = json.loads(session_p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                session = None
        stem = path.stem  # <cell>_r<repeat>
        cell, _, rep = stem.rpartition("_r")
        row = {"file": name, "cell": cell or stem, "repeat": rep,
               **score_one(hist, session)}
        rows.append(row)
        seen.add(name)
        print(f"[score] {name}: status={row['status']} "
              f"sensor={row['lane_sensor_rate']} placed={row['placed']} "
              f"env={row['env_applied']}", flush=True)
    # 放置失败/缺文件的格：session 审计存在 -> 用空遥测补 UNKNOWN 行；
    # 连 session 都没有 -> 也补一行 missing（表里绝不静默缺格）。
    for cell, rep in expected_cells(manifest):
        stem = f"{cell}_r{rep}"
        if f"{stem}.json" in seen:
            continue
        session_p = out_dir / f"{stem}.json.session.json"
        session = None
        note = "missing_session"
        if session_p.exists():
            try:
                session = json.loads(session_p.read_text(encoding="utf-8"))
                note = "no_telemetry_placement_failure"
            except (OSError, ValueError):
                session = None
        row = {"file": f"{stem}.json", "cell": cell, "repeat": str(rep),
               "note": note, **score_one([], session)}
        rows.append(row)
        print(f"[score] {stem}: status={row['status']} ({note}) "
              f"placement_rc={row['placement_rc']}", flush=True)
    table = aggregate(rows)
    result = {
        "schema": "fsd-conditions-table-v1",
        "written_at": time.time(),
        "scenario": SCENARIO,
        "rows": rows, "cells": table,
    }
    out = out_dir / "conditions_table.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n[score] 格级表 -> {out}", flush=True)
    hdr = ("| 格 | 轮次(UNK) | 状态 | sensor率(最差) | paired率(最差) | "
           "crossC | offRoad | stuck |")
    print(hdr, flush=True)
    print("| --- | --- | --- | --- | --- | --- | --- | --- |", flush=True)
    for cell, c in table.items():
        print(f"| {cell} | {c['runs']}({c['unknown_runs']}) | {c['status']} "
              f"| {c['lane_sensor_rate_min']} | {c['lane_paired_rate_min']} "
              f"| {c['cross_centre_frames_max']} "
              f"| {c['off_road_frames_max']} | {c['stuck_frames_max']} |",
              flush=True)
    n_unknown = sum(1 for c in table.values()
                    if c["status"] == STATUS_UNKNOWN)
    print(f"\n[score] cells={len(table)} UNKNOWN={n_unknown}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="只打印矩阵计划")
    mode.add_argument("--run", action="store_true",
                      help="跑矩阵（需要 BeamNG.tech）")
    mode.add_argument("--score", metavar="DIR", help="聚合已有遥测目录")
    ap.add_argument("--out-dir", default="logs/fsd_conditions")
    ap.add_argument("--preset-rain", default=None,
                    help="rain 天气预设名（引擎探测解析后传入）")
    ap.add_argument("--preset-fog", default=None,
                    help="fog 天气预设名（引擎探测解析后传入）")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() \
        else Path(args.out_dir)
    if args.plan:
        cells = plan_cells(args.preset_rain, args.preset_fog)
        for c in cells:
            print(f"{c['cell']}_r{c['repeat']}\t tod={c['tod']} "
                  f"wx={c['weather']}\t runnable={c['runnable']}")
        print(f"total={len(cells)} runnable="
              f"{sum(1 for c in cells if c['runnable'])}")
        return 0
    if args.run:
        return run_matrix(out_dir, args.preset_rain, args.preset_fog)
    return score_dir(Path(args.score))


if __name__ == "__main__":
    raise SystemExit(main())
