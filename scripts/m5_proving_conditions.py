# -*- coding: utf-8 -*-
"""试车场静止条件实验（T16_PROVING_GROUND_CONDITIONS_PREREG_20261105）。

固定 GridMap 无标线试车场（`proving_lane`、decal 双线、无铺装），只在
TOD×天气上变化。**静止采集**：不要求驾驶（试车场目前无可行驶参考），
只测放置阶段感知链路能否在恶劣环境下稳定给出双侧配对。

模式::

    .venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --plan
    .venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --run
    .venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --score

纪束：产物只落 logs/；环境应用读回失败 → UNKNOWN（fail-closed）；
不降门、不改默认、不晋级模型。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "logs" / "m5_seg" / "proving_conditions"
SCENE = ROOT / "logs" / "m5_seg" / "gridmap_scene.json"
MODEL = "logs/m5_seg/seg_model_v28_sslA/best.pt"

TODS = {"dawn": "06:30", "noon": "12:00", "dusk": "18:30", "night": "21:30"}
# clear = 引擎默认（不调预设）；rain/fog 用引擎库存名（见条件矩阵预注册）
WEATHERS = {"clear": None, "rain": "rainy", "fog": "foggy_morning"}

# 试车场诊断配置（预注册 §2）：非生产默认，必须逐格记录
PROVING_ENV = {
    "BEAMNG_CV_WHITE": "0",
    "BEAMNG_LINE_MERGE": "0",
    "BEAMNG_PAIR_WIDTH_NEAR": "1",
}

PASS_TRUE_RATE = 0.80       # 双侧配对真值率
PASS_CERT_BOTH = 0.80       # GT 左右线都命中
MIN_TICKS = 8               # 少于此 tick 数判 UNKNOWN
DEFAULT_CELL_TIMEOUT_S = 480.0  # 单格硬超时（防止引擎退出后探针挂死阻塞整批）


def _cells():
    for tod_name, tod in TODS.items():
        for wx_name, preset in WEATHERS.items():
            yield tod_name, tod, wx_name, preset


def _cell_name(tod: str, wx: str) -> str:
    return f"gridmap_{tod}_{wx}"


def plan() -> int:
    print(f"{'cell':28s} {'tod':6s} {'weather':8s} out")
    for tod_name, tod, wx_name, preset in _cells():
        name = _cell_name(tod_name, wx_name)
        print(f"{name:28s} {tod:6s} {wx_name:8s} {OUT_DIR / (name + '.json')}")
    print(f"共 {len(list(_cells()))} 格；配置 {PROVING_ENV}")
    return 0


def _run_env() -> dict:
    env = dict(os.environ)
    env.update(PROVING_ENV)
    return env


def _cell_cmd(tod: str, preset: str | None, out: Path) -> list[str]:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "m5_pairing_probe.py"),
        "--map", "gridmap_v2", "--scene", str(SCENE),
        "--scene-type", "proving_lane",
        "--line-convention", "right_lane", "--edge-margin-m", "0.6",
        "--mat-white", "line_yellow_double",
        "--mat-yellow", "line_yellow_double",
        "--mat-pave", "m_road_asphalt_rough_01",
        "--mat-gravel", "m_dirt_road_gravels",
        "--model", MODEL,
        "--frames", "12", "--save-frames",
        "--tod", tod,
        "--out", str(out),
    ]
    if preset:
        cmd += ["--weather-preset", preset]
    return cmd


def _game_pids() -> set[int]:
    """引擎 PID 快照（用于超时后清理探针直接子进程之外的残留引擎）。"""
    try:
        from beamng_autopilot.experiments.collection import game_pids
        return {int(p) for p in game_pids()}
    except Exception:  # noqa: BLE001 — 清理是 best-effort，不能反过来阻断批跑
        return set()


def _kill_tree(pid: int) -> None:
    """杀一个进程及其全部子进程（Windows: taskkill /T；POSIX: killpg）。"""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_new_game_pids(before: set[int]) -> None:
    for pid in _game_pids() - before:
        _kill_tree(pid)


def _run_cell_process(cmd: list[str], *, timeout_s: float):
    """跑一格；超时杀掉**整棵进程树**和本次新起的引擎。

    `subprocess.run(timeout=...)` 只杀直接子进程：探针拉起的 drive 与
    BeamNG 会继续占用 RPC 端口，下一格连接直接失败（2026-11-05 实测）。
    返回 ``(rc, stdout, stderr, timed_out)``。
    """
    pids_before = _game_pids()
    extra: dict = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env=_run_env(), **extra)
    try:
        stdout, stderr = proc.communicate(timeout=float(timeout_s))
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        _kill_new_game_pids(pids_before)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
        return -9, stdout or "", stderr or "", True


def run(preset_filter: str | None = None, *,
        cell_timeout_s: float = DEFAULT_CELL_TIMEOUT_S) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tod_name, tod, wx_name, preset in _cells():
        if preset_filter and wx_name != preset_filter:
            continue
        out = OUT_DIR / (_cell_name(tod_name, wx_name) + ".json")
        if out.exists():
            print(f"[skip] {out.name} 已存在")
            continue
        print(f"[run] {out.name} tod={tod} weather={preset or '-'}", flush=True)
        rc, stdout, stderr, timed_out = _run_cell_process(
            _cell_cmd(tod, preset, out), timeout_s=float(cell_timeout_s))
        # 注意：探针自己会写 <out>.run.log（内含 drive 的 environment 读回）；
        # runner 的 stdout 必须另存，否则会把环境证据覆盖掉（2026-11-05 踩过）。
        (OUT_DIR / (out.stem + ".probe.log")).write_text(
            stdout + "\n--- stderr ---\n" + stderr
            + (f"\n--- timeout ({float(cell_timeout_s):.1f}s) ---\n"
               if timed_out else ""),
            encoding="utf-8")
        row = {"cell": out.stem, "rc": rc, "out": str(out)}
        if timed_out:
            row["timeout"] = True
        rows.append(row)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


def _env_applied(log_path: Path) -> dict:
    """从 run.log 读环境应用读回（fail-closed 依据）。"""
    info = {"tod_after": None, "weather": None, "applied_line": None}
    if not log_path.exists():
        return info
    for line in log_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        if "environment ->" in line:
            info["applied_line"] = line.strip()
            for part in line.split():
                if part.startswith("applied_tod="):
                    val = part.split("=", 1)[1]
                    info["tod_after"] = None if val in ("", "?") else val
                if part.startswith("weather="):
                    info["weather"] = part.split("=", 1)[1] or None
    return info


def _score_cell(report: dict, log_path: Path) -> dict:
    s = report.get("summary") or {}
    ticks = int(s.get("ticks") or 0)
    true_n = int(s.get("true_pairs") or 0)
    phantom = int(s.get("phantom_pairs") or 0)
    amb = int(s.get("ambiguous_pairs") or 0)
    cert = (s.get("cert_line_recall") or {})
    cert_both = float(cert.get("both_hit_rate") or 0.0)
    true_rate = (true_n / ticks) if ticks else 0.0
    env = _env_applied(log_path)
    reasons = []
    if ticks < MIN_TICKS:
        status = "UNKNOWN"
        reasons.append(f"ticks={ticks}<{MIN_TICKS}")
    elif not env["tod_after"]:
        status = "UNKNOWN"
        reasons.append("environment_readback_missing")
    elif true_rate < PASS_TRUE_RATE:
        status = "FAIL"
        reasons.append(f"true_rate={true_rate:.3f}<{PASS_TRUE_RATE}")
    elif cert_both < PASS_CERT_BOTH:
        status = "FAIL"
        reasons.append(f"cert_both={cert_both:.3f}<{PASS_CERT_BOTH}")
    else:
        status = "PASS"
    return {
        "ticks": ticks, "true_pairs": true_n, "phantom_pairs": phantom,
        "ambiguous_pairs": amb, "true_rate": round(true_rate, 4),
        "cert_both": round(cert_both, 4),
        "modes": s.get("modes"), "cert": cert,
        "tod_after": env["tod_after"], "weather_applied": env["weather"],
        "status": status, "reasons": reasons,
    }


def score() -> int:
    table = {}
    for tod_name, _tod, wx_name, _preset in _cells():
        name = _cell_name(tod_name, wx_name)
        rep_path = OUT_DIR / (name + ".json")
        if not rep_path.exists():
            table[name] = {"status": "UNKNOWN", "reasons": ["no_report"]}
            continue
        try:
            report = json.loads(rep_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            table[name] = {"status": "UNKNOWN",
                           "reasons": [f"bad_report:{exc}"]}
            continue
        table[name] = _score_cell(report, OUT_DIR / (name + ".run.log"))
    counts: dict[str, int] = {}
    for v in table.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    summary = {"schema": "proving_conditions_v1", "cells": table,
               "counts": counts, "env": PROVING_ENV,
               "gates": {"true_rate": PASS_TRUE_RATE,
                         "cert_both": PASS_CERT_BOTH,
                         "min_ticks": MIN_TICKS}}
    (OUT_DIR / "conditions_table.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, row in table.items():
        print(f"{name:32s} {row['status']:8s} {row.get('reasons')}")
    print("counts", counts, "->", OUT_DIR / "conditions_table.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--weather", choices=tuple(WEATHERS), default=None,
                    help="只跑某一档天气（省略=全部 12 格）")
    ap.add_argument("--cell-timeout-s", type=float,
                    default=DEFAULT_CELL_TIMEOUT_S,
                    help="单格硬超时秒数（默认 480s）")
    args = ap.parse_args()
    if args.plan:
        return plan()
    if args.run:
        return run(args.weather, cell_timeout_s=args.cell_timeout_s)
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
