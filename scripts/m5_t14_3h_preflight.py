"""T14 3 小时计划 · 步骤 1+2：冻结输入 + 可信数据门盘点（preflight）。

用法::

    .venv\Scripts\python.exe scripts\m5_t14_3h_preflight.py

只读：写 `logs/experiments/<run_id>/00_freeze.json` 与 `01_dir_probe.json`，
不训练、不改权重、不碰用户未提交目录。

冻结：HEAD/工作树/生产与初值 checkpoint 哈希/阈值文件哈希/环境/GPU，
写入 `logs/experiments/<run_id>/00_freeze.json`。
盘点：哪些目录有**逐帧可追溯的漆线真值**（人工修订 line-only 集），
哪些目录只能当开发集（引擎标注，paint 不可信）；并列出每类的帧数、
标线像素占比分布、`255`（未知）占比——用于判断能不能过准入门。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beamng_autopilot import config  # noqa: E402

def _run_id() -> str:
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-id", default=None)
    known, _ = ap.parse_known_args()
    return known.run_id or f"t14_3h_{time.strftime('%Y%m%d_%H%M')}"


RUN_ID = _run_id()
OUT = Path(config.LOGS_DIR) / "experiments" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)


def sha16(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sh(*cmd) -> str:
    try:
        return subprocess.run(list(cmd), cwd=str(ROOT), capture_output=True,
                              text=True, timeout=120).stdout.strip()
    except Exception as exc:                 # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def freeze() -> dict:
    import torch
    rec = {
        "run_id": RUN_ID,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head": sh("git", "rev-parse", "HEAD"),
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": sh("git", "status", "--short").splitlines(),
        "hashes": {
            "thresholds_v2": sha16(ROOT / "docs" / "t14_thresholds_v2.json"),
            "prod_weights": sha16(ROOT / "logs" / "m5_seg" / "seg_model"
                                  / "best.pt"),
            "init_v13b": sha16(ROOT / "logs" / "m5_seg" / "seg_model_v13b"
                               / "best.pt"),
        },
        "env": {"torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "CPU"),
                "python": ".".join(str(v) for v in sys.version_info[:3])},
        "gpu_state": sh("nvidia-smi",
                        "--query-gpu=memory.total,memory.used,memory.free,"
                        "utilization.gpu", "--format=csv,noheader"),
    }
    (OUT / "00_freeze.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def probe_dir(d: Path, limit: int = 40) -> dict:
    fs = sorted(d.glob("frame_*.npz"))
    if not fs:
        return {"dir": str(d), "n": 0}
    step = max(1, len(fs) // limit)
    line, unk, shapes, classes = [], [], set(), set()
    no_label = 0
    for f in fs[::step]:
        z = np.load(f)
        if "label" not in z.files:
            # 纯采集目录（只有 colour）：不是训练候选，如实计数而不是崩掉
            no_label += 1
            continue
        lab = np.asarray(z["label"])
        shapes.add(lab.shape)
        classes.update(int(v) for v in np.unique(lab))
        line.append(float((lab == 2).mean()))
        unk.append(float((lab == 255).mean()))
    meta = d / "meta.json"
    if not meta.exists() and (d.parent / "meta.json").exists():
        meta = d.parent / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    if not line:
        return {"dir": str(d.relative_to(ROOT)), "n": len(fs),
                "no_label_frames": no_label,
                "note": "frames without a label array: not a training candidate"}
    a = np.asarray(line)
    u = np.asarray(unk)
    return {
        "dir": str(d.relative_to(ROOT)),
        "n": len(fs), "no_label_frames": no_label,
        "classes": sorted(classes),
        "shapes": sorted(str(s) for s in shapes),
        "line_med": round(float(np.median(a)), 6),
        "line_max": round(float(a.max()), 6),
        "unknown_med": round(float(np.median(u)), 4),
        "zero_line_frames": int((a == 0).sum()),
        "map": m.get("map_name"), "source_id": m.get("source_id"),
        "label_source": (m.get("label_source") or "")[:50],
    }


def main() -> int:
    rec = freeze()
    print(f"=== 冻结（{RUN_ID}）===")
    print(f"  HEAD {rec['head'][:10]} | 工作树未跟踪/改动 "
          f"{len(rec['status_short'])} 行")
    for k, v in rec["hashes"].items():
        print(f"  {k:14s} {v}")
    print(f"  env {rec['env']}")
    print(f"  gpu {rec['gpu_state']}")
    print(f"  -> {OUT / '00_freeze.json'}")
    print()

    seg = ROOT / "logs" / "m5_seg"
    manual, dev = [], []
    for d in sorted({p.parent for p in seg.rglob("frame_*.npz")}):
        info = probe_dir(d)
        if not info.get("n"):
            continue
        name = d.name
        if name.startswith("manual") or "gold" in name or "pkg" in name:
            manual.append(info)
        elif name in ("front_main", "front_narrow") and info.get("map"):
            dev.append(info)
    print("=== 候选『人工修订 line-only』目录（可作可信 paint 真值）===")
    print(f"  {'dir':52s} {'n':>4s} {'classes':16s} {'line_med':>9s} "
          f"{'line_max':>9s} {'unknown_med':>11s} {'zero_line':>9s}")
    for i in manual:
        if "classes" not in i:
            print(f"  {i['dir'][-52:]:52s} {i['n']:4d} 无 label 数组"
                  f"（纯采集目录，不作候选）")
            continue
        print(f"  {i['dir'][-52:]:52s} {i['n']:4d} "
              f"{str(i['classes']):16s} {i['line_med']:9.5f} "
              f"{i['line_max']:9.5f} {i['unknown_med']:11.4f} "
              f"{i['zero_line_frames']:9d}")
    print()
    print("=== 候选开发路段（引擎标注；paint 只作引擎参考）===")
    for i in dev[:16]:
        if "classes" not in i:
            continue
        print(f"  {i['dir'][-52:]:52s} {i['n']:4d} {str(i['map']):10s} "
              f"line_med {i['line_med']:.5f} zero_line {i['zero_line_frames']}")
    (OUT / "01_dir_probe.json").write_text(
        json.dumps({"manual": manual, "dev": dev}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT / '01_dir_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
