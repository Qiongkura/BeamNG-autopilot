"""W0：冻结本轮基线（git/环境/模型指纹/数据清单/协议），写成一个可复现快照。

方案 W0 的要求与这里的对应关系：

* 实际 git commit、未提交差异清单、各版本号、GPU、相机分辨率、模型与后处理哈希
  → ``git`` / ``env`` / ``files`` 三段；
* 已登记 run/checkpoint/数据来源用**显式目录清单**（禁止递归扫仓库根或整个 logs）
  → ``--run-dir`` / ``--model`` 只读你点名的那几项；
* 生产模型与研究基线分别命名并各存只读指纹 → ``--model production=... --model research=...``；
* 冻结任务指标定义与覆盖要求 → ``beamng_autopilot.experiments.protocol`` 的
  ``protocol_blob`` + ``protocol_hash``；
* 没有的东西记 UNKNOWN（不写 0、不写空）→ 每个读取失败都带 ``error`` 字段。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_freeze_baseline.py \\
        --run-id t14_w0_20260925 --model production=<路径> --model research=<路径> \\
        --run-dir logs/m5_seg/diverse_town_20260924/front_main=agent_revision \\
        --thresholds docs/t14_thresholds_v2.json --out <快照.json>
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.experiments.checkpoint import file_sha16  # noqa: E402
from beamng_autopilot.experiments.labels import PAINT_SOURCE_RANK  # noqa: E402
from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402
from beamng_autopilot.experiments.protocol import (  # noqa: E402
    PROTOCOL_VERSION, eligibility, protocol_blob, protocol_hash,
)

#: 训练/评价的相机分辨率契约（m5_train_seg.py 的 input_size）
CAMERA_RESOLUTION = (536, 403)


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=60)
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception as exc:                                  # noqa: BLE001
        return f"<git {args[0]} failed: {type(exc).__name__}>"


def git_block() -> dict:
    """commit / 分支 / 未提交差异清单（**标记归属**，不还原不覆盖）。"""
    porcelain = _git("status", "--porcelain")
    dirty = [ln for ln in porcelain.splitlines() if ln.strip()]
    tracked = sorted(ln[3:].strip() for ln in dirty
                     if not ln.startswith("??"))
    untracked = sorted(ln[3:].strip() for ln in dirty if ln.startswith("??"))
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_dirty": tracked,
        "untracked": untracked,
        "dirty_note": ("未提交改动属于工作区既有内容（用户工作/本轮开发），"
                       "本工具只记录清单，不还原、不覆盖"),
    }


def env_block() -> dict:
    """解释器/库/CUDA/GPU/相机契机的只读指纹。"""
    blob: dict = {"python": sys.version.split()[0],
                  "platform": platform.platform(),
                  "executable": str(Path(sys.executable))}
    for mod in ("torch", "cv2", "numpy"):
        try:
            m = __import__(mod)
            blob[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception as exc:                              # noqa: BLE001
            blob[mod] = f"UNKNOWN ({type(exc).__name__})"
    try:
        import torch
        blob["cuda_available"] = bool(torch.cuda.is_available())
        blob["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
        blob["cuda"] = str(getattr(torch.version, "cuda", None))
    except Exception as exc:                                  # noqa: BLE001
        blob["gpu"] = f"UNKNOWN ({type(exc).__name__})"
    blob["camera_resolution_px"] = list(CAMERA_RESOLUTION)
    blob["logs_dir"] = str(config.LOGS_DIR)
    return blob


def file_block(path: Path, *, kind: str) -> dict:
    """单个文件/目录的指纹；不存在就记 UNKNOWN，不猜。"""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "kind": kind, "status": "UNKNOWN",
                "error": "path does not exist"}
    if p.is_dir():
        files = sorted(x for x in p.rglob("*") if x.is_file())
        return {"path": str(p), "kind": kind, "status": "ok",
                "n_files": len(files),
                "sha16_of_names": file_sha16_of(str(sorted(
                    x.name for x in files)))}
    return {"path": str(p), "kind": kind, "status": "ok",
            "sha16": file_sha16(str(p)), "bytes": int(p.stat().st_size)}


def file_sha16_of(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_block(spec: str) -> dict:
    """``PATH[=SOURCE]`` → 显式清单里那一项的覆盖与身份（用仓库同一套审计）。"""
    path, _, source = spec.partition("=")
    d = Path(path)
    if not d.exists():
        return {"dir": str(d), "status": "UNKNOWN", "error": "dir does not exist"}
    src = source.strip() or "engine_annotation"
    rank = PAINT_SOURCE_RANK.get(src, "absent")
    mf = DatasetManifest.build([d], root=ROOT, paint_sources={str(d): src})
    cov = mf.coverage().get("train", {})
    groups = sorted({r.group for r in mf.records if not r.reject_reason})
    frames = [r for r in mf.records if not r.reject_reason]
    return {
        "dir": str(d), "status": "ok", "paint_source": src, "rank": rank,
        "eligibility": eligibility(rank),
        "groups": groups,
        "dataset_id": mf.dataset_id,
        "n_frames": len(mf.records),
        "n_usable": len(frames),
        "n_rejected": len(mf.records) - len(frames),
        "reject_reasons": sorted({r.reject_reason for r in mf.records
                                  if r.reject_reason}),
        "coverage": {k: cov.get(k) for k in (
            "n_frames", "trainable_frames", "paint_valid_frames")},
        "map_name": (frames[0].map_name if frames else None),
        "source_id": (frames[0].source_id if frames else None),
        "notes": list(mf.notes),
    }


def build_snapshot(*, run_id: str, models: list, run_dirs: list,
                   thresholds: Path | None) -> dict:
    t_blob, t_hash = None, None
    if thresholds is not None:
        p = Path(thresholds)
        if p.exists():
            blob = json.loads(p.read_text(encoding="utf-8"))
            t_blob = {"path": str(p), "config_hash": blob.get("config_hash"),
                      "thresholds": blob.get("thresholds")}
            t_hash = file_sha16(str(p))
        else:
            t_blob = {"path": str(p), "status": "UNKNOWN",
                      "error": "thresholds file does not exist"}
    proto = protocol_blob(thresholds=(t_blob or {}).get("thresholds") or {})
    snap = {
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(
            thresholds=(t_blob or {}).get("thresholds") or {}),
        "protocol": proto,
        "thresholds_file": t_blob, "thresholds_sha16": t_hash,
        "git": git_block(),
        "env": env_block(),
        "models": [file_block(Path(spec.split("=", 1)[1]),
                              kind=f"model:{spec.split('=', 1)[0]}")
                   for spec in models if "=" in spec],
        "runs": [run_block(s) for s in run_dirs],
        "control_ownership": {
            "game_session_started": False,
            "vehicle_control": "none",
            "note": ("本工具只读文件与版本信息：不启动游戏、不连接车辆、"
                     "不持任何控制权（方案 §3：研究/影子候选不自动获得控制权）"),
        },
        "will_read": sorted(set(
            [str(Path(s.split("=", 1)[1])) for s in models if "=" in s]
            + [str(Path(s.split("=", 1)[0])) for s in run_dirs]
            + ([str(thresholds)] if thresholds else []))),
        "will_update": [],       # W0 只读；更新候选目录发生在后续工作包
    }
    return snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=PATH", help="可重复；生产模型与研究基线分别命名")
    ap.add_argument("--run-dir", action="append", default=[],
                    metavar="PATH[=SOURCE]", help="可重复；显式清单，不做递归扫描")
    ap.add_argument("--thresholds", default="docs/t14_thresholds_v2.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-protocol", default=None,
                    help="把协议快照（定义+覆盖+资格+阈值）写成 JSON，"
                         "供 docs/ 版本化引用")
    args = ap.parse_args(argv)

    snap = build_snapshot(run_id=args.run_id, models=list(args.model),
                          run_dirs=list(args.run_dir),
                          thresholds=Path(args.thresholds) if args.thresholds
                          else None)
    out = Path(args.out) if args.out else (config.LOGS_DIR / "experiments"
                                           / args.run_id / "baseline_freeze.json")
    if args.dump_protocol:
        dp = Path(args.dump_protocol)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps(
            {**snap["protocol"], "hash": snap["protocol_hash"]},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[freeze] 协议快照 -> {dp}（hash {snap['protocol_hash']}）")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[freeze] commit={snap['git']['commit'][:12]} "
          f"dirty={len(snap['git']['tracked_dirty'])} "
          f"protocol={snap['protocol_hash']}")
    for m in snap["models"]:
        print(f"[freeze] {m['kind']}: {m['status']} "
              f"{m.get('sha16', m.get('error', ''))}")
    for r in snap["runs"]:
        print(f"[freeze] run {r['dir']}: {r['status']} "
              f"{r.get('n_usable', r.get('error', ''))} 帧可用"
              f"（标签来源 {r.get('paint_source')} / rank {r.get('rank')}）")
    print(f"[freeze] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
