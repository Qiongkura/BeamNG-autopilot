# -*- coding: utf-8 -*-
"""试车场低照度/雨天采集扩池（T16_PROVING_CAPTURE_PREREG_20261105）。

模式::

    python scripts/m5_proving_capture.py --plan
    python scripts/m5_proving_capture.py --build-sites
    python scripts/m5_proving_capture.py --capture --site S1 --cell noon_rain

采集通过既有 ``m5_pairing_probe.py``（场景模式 + ``--save-frames``）完成：
RGB 帧、位姿、相机模型逐 tick 落盘；环境读回与探针汇总一起写进
``capture.json``。本脚本不生成标签（导出/门禁见预注册 §3.3，pilot 下一步）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SITES_DIR = ROOT / "logs" / "m5_seg" / "proving_sites"
CAP_DIR = ROOT / "logs" / "m5_seg" / "proving_capture"
MODEL = "logs/m5_seg/seg_model_v28_sslA/best.pt"

#: 评测试车场 (500,500) 不参与采集；S1-S3 训练，P 留出（预注册 §2.1）
SITES = {
    "S1": {"mid": [300.0, 700.0, 100.0], "dir": [0.0, 1.0, 0.0],
           "edge_margin_m": 0.8, "split": "train"},
    "S2": {"mid": [700.0, 300.0, 100.0], "dir": [1.0, 0.0, 0.0],
           "edge_margin_m": 1.0, "split": "train"},
    "S3": {"mid": [500.0, 900.0, 100.0], "dir": [0.0, -1.0, 0.0],
           "edge_margin_m": 1.2, "split": "train"},
    "P": {"mid": [600.0, 800.0, 100.0], "dir": [0.0, 1.0, 0.0],
          "edge_margin_m": 0.6, "split": "holdout"},
}

TODS = {"dawn": "06:30", "noon": "12:00", "dusk": "18:30", "night": "21:30"}
WEATHERS = {"clear": None, "rain": "rainy", "fog": "foggy_morning"}

# Formal capture is disabled until ownership/environment/freshness contracts are
# repaired. Tests may explicitly enable orchestration with monkeypatch; production
# CLI cannot bypass this switch.
_CAPTURE_ENABLED = False

PROVING_ENV = {
    "BEAMNG_CV_WHITE": "0",
    "BEAMNG_LINE_MERGE": "0",
    "BEAMNG_PAIR_WIDTH_NEAR": "1",
}


def cells() -> list[str]:
    return [f"{t}_{w}" for t in TODS for w in WEATHERS]

#: P 留出站只采预注册 §2.1 的 4 格
P_CELLS = ("noon_clear", "noon_rain", "dusk_clear", "night_clear")

CELLS_PER_SITE = 12


def site_doc(site: str) -> dict:
    """站点 -> 探针可读的场景 JSON（right_lane 由 CLI 约定给定）。"""
    s = SITES[site]
    return {
        "asset": {"map": "gridmap_v2"},
        "scene_type": "proving_lane",
        "scene": {
            "station_m": 0.0,
            "road_id": f"proving_{site}",
            "mid": [float(x) for x in s["mid"]],
            "dir": [float(x) for x in s["dir"]],
            "half_width_m": 3.5,
            "slope_pct": 0.0,
            "curve_deg": 0.0,
            "lines": [{"role": "left", "nodes": []},
                      {"role": "right", "nodes": []}],
        },
        "capture": {"site": site, "split": s["split"],
                    "edge_margin_m": float(s["edge_margin_m"])},
    }


def build_sites() -> int:
    SITES_DIR.mkdir(parents=True, exist_ok=True)
    for name in SITES:
        p = SITES_DIR / f"{name}.json"
        p.write_text(json.dumps(site_doc(name), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"[site] {p}")
    return 0


def cell_args(cell: str) -> tuple[str, str | None]:
    """``dawn_rain`` -> ("06:30", "rainy")；未知格 ValueError。"""
    tod, _, wx = str(cell).partition("_")
    if tod not in TODS or wx not in WEATHERS:
        raise ValueError(f"未知条件格 {cell!r}（可选 {cells()}）")
    return TODS[tod], WEATHERS[wx]


def capture_cmd(site: str, cell: str, out: Path, *, frames: int) -> list[str]:
    tod, preset = cell_args(cell)
    scene = SITES_DIR / f"{site}.json"
    if not scene.is_file():
        raise FileNotFoundError(f"缺站点场景 {scene}（先跑 --build-sites）")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "m5_pairing_probe.py"),
        "--map", "gridmap_v2", "--scene", str(scene),
        "--scene-type", "proving_lane",
        "--line-convention", "right_lane",
        "--edge-margin-m", str(float(SITES[site]["edge_margin_m"])),
        "--mat-white", "line_yellow_double",
        "--mat-yellow", "line_yellow_double",
        "--mat-pave", "m_road_asphalt_rough_01",
        "--mat-gravel", "m_dirt_road_gravels",
        "--model", MODEL,
        "--frames", str(int(frames)), "--save-frames",
        # 运动学步进采集：沿站点中心线逐站传送取帧，不跑 FSD 驾驶。
        # 原 FSD 采集在低照度下永远出不了放置阶段（2026-11-06 实测：
        # S1 四格 1848 帧、位移 <=0.0007 m、独立位姿 1 个），已废弃。
        "--kinematic-capture",
        "--kinematic-step-m", "1.0",
        "--kinematic-span-m", "30.0",
        "--kinematic-lateral-jitter", "0.25",
        "--kinematic-yaw-jitter", "2.0",
        "--kinematic-settle-s", "0.6",
        "--tod", tod,
        "--out", str(out),
    ]
    if preset:
        cmd += ["--weather-preset", preset]
    return cmd


def _run_env() -> dict:
    env = dict(os.environ)
    env.update(PROVING_ENV)
    return env


def _env_evidence(run_log: Path) -> dict:
    """采集的 drive 日志里的环境读回（applied_tod / weather）。"""
    info = {"tod_after": None, "weather": None, "line": None}
    if not run_log.exists():
        return info
    for line in run_log.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        if "environment ->" in line:
            info["line"] = line.strip()
            for part in line.split():
                if part.startswith("applied_tod="):
                    v = part.split("=", 1)[1]
                    info["tod_after"] = None if v in ("", "?") else v
                elif part.startswith("weather="):
                    info["weather"] = part.split("=", 1)[1] or None
    return info


def _game_pids() -> set[int]:
    try:
        from beamng_autopilot.experiments.collection import game_pids
        return {int(p) for p in game_pids()}
    except Exception:  # noqa: BLE001 — 清理是 best-effort
        return set()


def _kill_games() -> None:
    """Never kill by image name: ownership of those processes is unknown."""
    raise RuntimeError("BLOCKED: 无法证明游戏进程归属，拒绝清理")


def wait_engine_gone(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if not _game_pids():
            return True
        time.sleep(1.0)
    return not _game_pids()


def capture(site: str, cell: str, *, frames: int = 60) -> int:
    # Also guard programmatic callers; the CLI block alone is not a boundary.
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 环境与进程归属尚未验证，禁止采集")
    if site not in SITES:
        raise SystemExit(f"未知站点 {site!r}（可选 {sorted(SITES)}）")
    tod, preset = cell_args(cell)
    out_dir = CAP_DIR / site / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "probe.json"
    if out.exists():
        print(f"[skip] {out} 已存在")
        return 0
    print(f"[capture] site={site} cell={cell} tod={tod} "
          f"weather={preset or '-'} frames={frames}", flush=True)
    timed_out = False
    stdout = stderr = ""
    rc = -1
    try:
        proc = subprocess.run(capture_cmd(site, cell, out, frames=frames),
                              cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env=_run_env(), timeout=1200)
        stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        # 探针超时只杀直接子进程；引擎/采集进程树要显式清理，否则下一格
        # 会撞上残留 RPC 端口（2026-11-05 条件批跑教训）。
        timed_out = True
        rc = -9
        _kill_games()
        wait_engine_gone()
    (out_dir / "probe.log").write_text(
        stdout + "\n--- stderr ---\n" + stderr
        + ("\n--- timeout (1200s) ---\n" if timed_out else ""),
        encoding="utf-8")
    report = {}
    if out.exists():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
    env = _env_evidence(out_dir / "probe.run.log")
    n_frames = len(list((out_dir / "probe.frames").glob("frame_*.npz")))
    manifest = {
        "schema": "proving_capture_v1",
        "site": site,
        "cell": cell,
        "tod": tod,
        "weather_preset": preset,
        "split": SITES[site]["split"],
        "edge_margin_m": float(SITES[site]["edge_margin_m"]),
        "env": env,
        "probe_rc": rc,
        "timeout": timed_out,
        "probe_summary": report.get("summary"),
        "frames_saved": n_frames,
        "status": ("ok" if rc == 0 and n_frames >= 20
                   and env.get("tod_after") else "INCOMPLETE"),
    }
    (out_dir / "capture.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("site", "cell", "status", "frames_saved", "env")},
                     ensure_ascii=False))
    return 0


def run_site(site: str, only: tuple[str, ...] | None = None) -> int:
    """顺序采一个站点的全部条件格（P 站默认只采 4 个留出格）。"""
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 环境与进程归属尚未验证，禁止批量采集")
    if site not in SITES:
        raise SystemExit(f"未知站点 {site!r}（可选 {sorted(SITES)}）")
    todo = only or (P_CELLS if site == "P" else tuple(cells()))
    for cell in todo:
        man = CAP_DIR / site / cell / "capture.json"
        if man.exists():
            try:
                if json.loads(man.read_text(encoding="utf-8")).get(
                        "status") == "ok":
                    print(f"[skip] {site}/{cell} already ok")
                    continue
            except (OSError, ValueError):
                pass
        wait_engine_gone()
        try:
            capture(site, cell)
        except Exception as exc:  # noqa: BLE001 — 单格失败不阻断整站
            d = CAP_DIR / site / cell
            d.mkdir(parents=True, exist_ok=True)
            (d / "capture.json").write_text(json.dumps({
                "schema": "proving_capture_v1", "site": site, "cell": cell,
                "status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[FAIL] {site}/{cell}: {type(exc).__name__}: {exc}",
                  flush=True)
        time.sleep(1.0)
    return 0


def run_all() -> int:
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 环境与进程归属尚未验证，禁止批量采集")
    for site in ("S1", "S2", "S3", "P"):
        run_site(site)
    return 0


def plan() -> int:
    print(f"{'site':5s} {'split':8s} {'cell':12s} frames_dir")
    for name in SITES:
        for cell in cells():
            print(f"{name:5s} {SITES[name]['split']:8s} {cell:12s} "
                  f"{CAP_DIR / name / cell}")
    print(f"{len(SITES) * len(cells())} cells; env={PROVING_ENV}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--build-sites", action="store_true")
    g.add_argument("--capture", action="store_true")
    g.add_argument("--run-site", action="store_true",
                   help="顺序采一个站点全部条件格（P 站只采 4 个留出格）")
    g.add_argument("--run-all", action="store_true",
                   help="S1/S2/S3 各 12 格 + P 4 格，共 36 格顺序采集")
    ap.add_argument("--site", choices=tuple(SITES), default=None)
    ap.add_argument("--cell", default=None, help="例如 dawn_rain")
    ap.add_argument("--frames", type=int, default=60,
                    help="探针分析 tick 数（约等于保存帧数；30 帧/格用 60）")
    args = ap.parse_args()
    if args.plan:
        return plan()
    if args.build_sites:
        return build_sites()
    # The current runner cannot prove environment readback, process ownership
    # or capture freshness.  Do not launch/attach or touch existing evidence
    # until those contracts have been repaired and independently validated.
    if args.run_all or args.run_site or args.capture:
        print("[capture] BLOCKED: 环境读回、进程归属与帧来源尚未验证；"
              "禁止批量采集，保留既有证据。", file=sys.stderr)
        return 3
    if args.run_all:
        return run_all()
    if args.run_site:
        if not args.site:
            raise SystemExit("--run-site 需要 --site")
        return run_site(args.site)
    if not args.site or not args.cell:
        raise SystemExit("--capture 需要 --site 和 --cell")
    return capture(args.site, args.cell, frames=args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
