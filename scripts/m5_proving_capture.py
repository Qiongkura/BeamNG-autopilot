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
import hashlib
import json
import os
import subprocess
import tempfile
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

# 采集总开关（kill-switch）：门禁闭环已由 attempt 编排强制（环境读回核对、
# 帧清单核对、按记录 PID 清理引擎、独占清单与隔离目录）。出问题时拨回
# False 即整体停批，不影响既有证据。
_CAPTURE_ENABLED = True

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


def assess_capture_result(*, returncode, timed_out, frame_count, environment,
                          ownership_verified, minimum_frames=20):
    """Classify one attempt without performing IO or authorising capture.

    Missing, malformed or non-boolean evidence is UNKNOWN, not success.
    ``weather_matches`` must come from an independent readback, never from
    echoing the requested weather. This helper does not establish ownership
    or certify frame provenance; those require separate evidence.
    """
    reasons = []
    if type(minimum_frames) is not int or minimum_frames < 20:
        # Preregistered minimum is 20: invalid or lowered thresholds cannot
        # make an incomplete attempt appear evidence-ready.
        reasons.append("invalid_minimum_frames")
        minimum_frames = 20
    if timed_out is not False:
        reasons.append("timeout_or_unknown")
    if type(returncode) is not int or returncode != 0:
        reasons.append("subprocess_failed_or_unknown")
    if type(frame_count) is not int or frame_count < minimum_frames:
        reasons.append("insufficient_or_unknown_frames")
    if ownership_verified is not True:
        reasons.append("ownership_unverified")
    env = environment if isinstance(environment, dict) else {}
    if env.get("tod_matches") is not True:
        reasons.append("tod_mismatch_or_unknown")
    if env.get("weather_matches") is not True:
        reasons.append("weather_mismatch_or_unknown")
    # Even matching environment and a successful process do not by themselves
    # prove actual frame identity, sensor freshness or training qualification.
    return {"status": "INCOMPLETE" if reasons else "EVIDENCE_READY",
            "reasons": reasons, "training_qualified": False}


def create_attempt_dir(parent: Path) -> Path:
    """Create a unique attempt directory under an existing parent.

    Never reuses, moves or deletes prior attempts. This only allocates a
    directory; it does not establish process ownership, run identity or
    permission to start a game. Caller must separately persist a manifest.
    """
    parent = Path(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("attempt parent must be an existing non-symlink directory")
    return Path(tempfile.mkdtemp(prefix="attempt_", dir=str(parent)))


def write_attempt_manifest(attempt_dir: Path, manifest: dict) -> Path:
    """Exclusively write one manifest, never replace existing evidence.

    Exclusive create prevents accidental overwrite; it does not promise an
    atomic publish under crash or disk failure. A partial file is an invalid
    attempt and requires explicit review, never automatic reuse.
    """
    directory = Path(attempt_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("attempt directory must be an existing non-symlink directory")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    path = directory / "attempt_manifest.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def verify_frame_manifest(frame_dir: Path, entries: list[dict]) -> dict:
    """只读验证声明的帧文件；这不是新鲜度或采集所有权证明。

    清单必须非空、文件名唯一且是目录内的普通 npz 文件，所有 SHA-256
    必须与文件内容一致。额外或缺失文件均失败，避免旧帧混入本次统计。
    不解压或加载不可信的 pickle，也不写任何产物。
    """
    root = Path(frame_dir).resolve()
    errors = []
    declared = set()
    declared_folded = set()
    if not root.is_dir() or not isinstance(entries, list) or not entries:
        return {"valid": False, "errors": ["missing_directory_or_entries"]}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("invalid_entry")
            continue
        name, digest = entry.get("file"), entry.get("sha256")
        if (not isinstance(name, str) or not name or Path(name).name != name
                or "/" in name or "\\" in name or ":" in name
                or not name.lower().endswith(".npz")):
            errors.append("invalid_filename")
            continue
        if name.casefold() in declared_folded:
            errors.append(f"duplicate:{name}")
            continue
        declared.add(name)
        declared_folded.add(name.casefold())
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            errors.append(f"invalid_digest:{name}")
            continue
        path = root / name
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                errors.append(f"missing_or_unsafe:{name}")
                continue
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                errors.append(f"digest_mismatch:{name}")
        except OSError:
            errors.append(f"unreadable:{name}")
    actual = {p.name for p in root.glob("*.npz")}
    for name in sorted(actual - declared):
        errors.append(f"unlisted:{name}")
    for name in sorted(declared - actual):
        errors.append(f"missing:{name}")
    return {"valid": not errors, "errors": errors,
            "declared_count": len(declared), "actual_count": len(actual)}


def _game_pids() -> set[int] | None:
    """游戏进程集合；探测失败返回 None，绝不伪装成空集合。"""
    try:
        from beamng_autopilot.experiments.collection import game_pids
        result = game_pids()
        return None if result is None else {int(p) for p in result}
    except (OSError, RuntimeError, ValueError, TypeError, ImportError):
        return None


def _kill_games() -> None:
    """Never kill by image name: ownership of those processes is unknown."""
    raise RuntimeError("BLOCKED: 无法证明游戏进程归属，拒绝清理")


def _kill_pid_tree(pid: int) -> bool:
    """按**记录归属**的 PID 清理进程树（探针报告的引擎 PID）。

    与按映像名清理的本质区别：该 PID 是我们自己的子进程（探针写入
    probe.json 的 engine_pid）——归属链条完整。GBK 输出必须
    errors="replace"（§7）。清理失败返回 False，调用方据此判
    ownership_unverified。
    """
    try:
        proc = subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60)
        return proc.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[capture] pid-tree cleanup failed: {type(exc).__name__}: {exc}",
              flush=True)
        return False


def _write_attempt_result(attempt_dir: Path, result: dict) -> Path:
    """独占写尝试结果（与启动清单不同名，互不覆盖）。"""
    directory = Path(attempt_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("attempt directory must be an existing non-symlink directory")
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    path = directory / "attempt_result.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def wait_engine_gone(timeout_s: float = 30.0) -> bool:
    """只有明确读到空集合才返回 True；未知或超时返回 False。"""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        pids = _game_pids()
        if pids is None:
            return False
        if not pids:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def capture(site: str, cell: str, *, frames: int = 60,
            reclaim: bool = False) -> int:
    """按 attempt 流程采集一格（§6.2 门 1/2/4 闭环）。

    每次尝试独立目录（隔离残留帧）、独占清单（不覆盖任何历史证据）、
    引擎按记录 PID 清理（归属链完整）、帧清单与磁盘核对、环境读回核对
    （tod + 引擎侧天气）。EVIDENCE_READY 才写格级 summary。
    """
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 采集门禁未启用")
    if site not in SITES:
        raise SystemExit(f"未知站点 {site!r}（可选 {sorted(SITES)}）")
    tod, preset = cell_args(cell)
    cell_dir = CAP_DIR / site / cell
    legacy_summary = cell_dir / "capture.json"
    if legacy_summary.exists() and not reclaim:
        print(f"[audit] {site}/{cell}: 存在历史 capture.json，"
              "默认跳过（--reclaim 显式重采，历史证据保留）", flush=True)
        return 1
    cell_dir.mkdir(parents=True, exist_ok=True)
    attempt = create_attempt_dir(cell_dir)
    write_attempt_manifest(attempt, {
        "schema": "proving_attempt_v1",
        "site": site, "cell": cell, "tod": tod,
        "weather_preset": preset,
        "split": SITES[site]["split"],
        "edge_margin_m": float(SITES[site]["edge_margin_m"]),
        "requested_frames": int(frames),
        "reclaim": bool(reclaim),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime()),
    })
    out = attempt / "probe.json"
    print(f"[capture] site={site} cell={cell} tod={tod} "
          f"weather={preset or '-'} frames={frames} attempt={attempt.name}",
          flush=True)
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
        timed_out = True
        rc = -9
    (attempt / "probe.log").write_text(
        stdout + "\n--- stderr ---\n" + stderr
        + ("\n--- timeout (1200s) ---\n" if timed_out else ""),
        encoding="utf-8")

    report = {}
    if out.exists():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
    engine_pid = report.get("engine_pid")

    # §6.2-1/3：帧清单与磁盘核对（声明缺失/多余/哈希不符都算 INVALID）
    kin = report.get("kinematic") or {}
    declared = kin.get("files") or []
    frames_dir = out.with_suffix(".frames")
    fver = verify_frame_manifest(frames_dir, declared)
    frame_count = int(fver.get("declared_count") or 0) if fver["valid"] \
        else int(fver.get("actual_count") or 0)

    # 归属链：探针记录的引擎 PID → 清理 → 引擎消失
    cleanup_ok = None
    if engine_pid:
        cleanup_ok = _kill_pid_tree(int(engine_pid))
    engine_gone = wait_engine_gone()
    ownership_verified = bool(engine_pid) and cleanup_ok is True \
        and engine_gone is True

    env = report.get("environment") or {}
    result = assess_capture_result(
        returncode=rc, timed_out=timed_out, frame_count=frame_count,
        environment=env, ownership_verified=ownership_verified,
        minimum_frames=20)
    if not fver["valid"]:
        result["reasons"].extend(f"frame_manifest:{e}"
                                 for e in fver["errors"][:8])
        result["status"] = "INCOMPLETE"

    _write_attempt_result(attempt, {
        "schema": "proving_attempt_result_v1",
        "status": result["status"],
        "reasons": result["reasons"],
        "probe": {"returncode": rc, "timed_out": timed_out},
        "environment": env,
        "frame_verification": {"valid": fver["valid"],
                               "declared": fver.get("declared_count") or 0,
                               "actual": fver.get("actual_count") or 0,
                               "errors": fver["errors"][:8]},
        "frame_count": frame_count,
        "engine_pid": engine_pid,
        "engine_cleanup_ok": cleanup_ok,
        "engine_gone": engine_gone,
        "ownership_verified": ownership_verified,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
    })

    # 格级 summary：EVIDENCE_READY 才更新（legacy 内容完整并入）；
    # INCOMPLETE 且无 legacy 时写指针 summary，绝不覆盖历史 ok 证据。
    new_summary = {
        "schema": "proving_capture_v2",
        "site": site, "cell": cell,
        "status": ("ok" if result["status"] == "EVIDENCE_READY"
                   else "INCOMPLETE"),
        "attempt": attempt.name,
        "attempt_result": result,
        "tod": tod, "weather_preset": preset,
        "frames_saved": frame_count,
        "env": env,
        "probe_rc": rc,
    }
    if legacy_summary.exists():
        if result["status"] != "EVIDENCE_READY":
            print(f"[keep] {site}/{cell}: 尝试未过门，历史 capture.json 保留",
                  flush=True)
        else:
            new_summary["legacy"] = json.loads(
                legacy_summary.read_text(encoding="utf-8"))
            new_summary["superseded_legacy"] = True
            legacy_summary.write_text(
                json.dumps(new_summary, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(json.dumps({k: new_summary[k] for k in
                              ("site", "cell", "status", "frames_saved")},
                             ensure_ascii=False), flush=True)
    else:
        legacy_summary.write_text(
            json.dumps(new_summary, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(json.dumps({k: new_summary[k] for k in
                          ("site", "cell", "status", "frames_saved")},
                         ensure_ascii=False), flush=True)
    return 0 if result["status"] == "EVIDENCE_READY" else 1


def run_site(site: str, only: tuple[str, ...] | None = None,
             *, reclaim: bool = False) -> int:
    """顺序采一个站点的全部条件格（P 站默认只采 4 个留出格）。

    status=ok 的格默认跳过（历史证据不信任、不覆盖）；INCOMPLETE 或
    无 summary 的格自动重试（新 attempt 目录，§6.2-4 可恢复性）；
    ``reclaim=True`` 时 ok 格也重采（旧证据并入新 summary）。
    """
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 采集门禁未启用")
    if site not in SITES:
        raise SystemExit(f"未知站点 {site!r}（可选 {sorted(SITES)}）")
    todo = only or (P_CELLS if site == "P" else tuple(cells()))
    failed = False
    for cell in todo:
        summary = CAP_DIR / site / cell / "capture.json"
        if summary.exists() and not reclaim:
            status = None
            try:
                status = json.loads(summary.read_text(
                    encoding="utf-8")).get("status")
            except (OSError, ValueError):
                status = None
            if status == "ok":
                print(f"[audit] {site}/{cell}: status=ok 且未开 --reclaim，"
                      "跳过", flush=True)
                failed = True
                continue
            print(f"[retry] {site}/{cell}: status={status!r}，"
                  "开新 attempt 重试", flush=True)
        if not wait_engine_gone():
            # An unknown process state is not permission to attach or kill.
            print(f"[FAIL] {site}/{cell}: engine state unknown or busy", flush=True)
            failed = True
            break
        try:
            rc = capture(site, cell, reclaim=reclaim)
            if rc != 0:
                failed = True
                print(f"[FAIL] {site}/{cell}: capture returned {rc!r}", flush=True)
        except Exception as exc:  # noqa: BLE001 — preserve evidence and report
            failed = True
            print(f"[FAIL] {site}/{cell}: {type(exc).__name__}: {exc}",
                  flush=True)
        time.sleep(1.0)
    return 1 if failed else 0


def run_all(*, reclaim: bool = False) -> int:
    if not _CAPTURE_ENABLED:
        raise RuntimeError("BLOCKED: 采集门禁未启用")
    failed = False
    for site in ("S1", "S2", "S3", "P"):
        if run_site(site, reclaim=reclaim) != 0:
            failed = True
    return 1 if failed else 0


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
    ap.add_argument("--reclaim", action="store_true",
                    help="允许重采 status=ok 的格（旧证据并入新 summary）")
    args = ap.parse_args()
    if args.plan:
        return plan()
    if args.build_sites:
        return build_sites()
    if args.run_all:
        return run_all(reclaim=args.reclaim)
    if args.run_site:
        if not args.site:
            raise SystemExit("--run-site 需要 --site")
        return run_site(args.site, reclaim=args.reclaim)
    if not args.site or not args.cell:
        raise SystemExit("--capture 需要 --site 和 --cell")
    return capture(args.site, args.cell, frames=args.frames,
                   reclaim=args.reclaim)


if __name__ == "__main__":
    raise SystemExit(main())
