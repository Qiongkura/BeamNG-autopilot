"""采集数据的空间隔离核对：候选采集 vs 开发/评价帧的最近距离（方案 §S1/E1）。

E1 的解除规格要求"新训练负例不得取自当前固定开发/校准/最终评价帧**或其空间
邻近段**"，并写明"距 dev >50 m（脚本自动核对）"。本脚本就是那份核对：

* 开发集默认取：agent 真值三组（`line_truth_agent_full_20260925/{town,wide,plain}`）
  + 人工复核包（`review_pack_20260926/reviewed_full`）——与 `_rounds_audit` 的
  dev 集合一致；
* 候选：一个或多个采集目录（自动识别 `meta.json` 在目录内还是上一级）；
* 判据：**不同地图 = 天然隔离**（地图不同就没有空间邻近问题）；同地图时取
  "候选任一帧到开发集任一帧"的最近距离，与 `--buffer-m`（默认 50 m，与
  `SPATIAL_BUFFER_M` 同值）比较；
* 缺位姿的帧记 UNKNOWN 并计数——**不能**把"没位姿"当成"离得远"。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_collect_isolation_check.py \\
        --dir logs/m5_seg/collect_collect_e1_jv_20260927_072234 \\
        --dir logs/m5_seg/collect_collect_e2_it3_20260927_073951 \\
        --out logs/experiments/collect_isolation_20260927.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV_DIRS = (
    "logs/m5_seg/line_truth_agent_full_20260925/town/front_main",
    "logs/m5_seg/line_truth_agent_full_20260925/wide/front_main",
    "logs/m5_seg/line_truth_agent_full_20260925/plain/front_main",
)
PACK_DIR = "logs/experiments/review_pack_20260926/reviewed_full"


def _meta_of(d: Path) -> dict:
    for cand in (d / "meta.json", d.parent / "meta.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def positions(d: Path) -> tuple[str, list, int]:
    """``(map_name, [pos...], n_missing)``；缺位姿的帧单独计数。"""
    m = _meta_of(d)
    pos, missing = [], 0
    for fr in (m.get("frames") or []):
        p = fr.get("pos")
        if p:
            pos.append(tuple(float(x) for x in p))
        else:
            missing += 1
    return str(m.get("map_name") or ""), pos, missing


def dev_positions() -> dict:
    by_map: dict = {}
    for d in DEV_DIRS:
        name, pos, _ = positions(Path(d))
        if pos:
            by_map.setdefault(name, []).extend(pos)
    for mp in sorted(Path(PACK_DIR).glob("*/front_main")):
        name, pos, _ = positions(mp)
        if pos:
            by_map.setdefault(name, []).extend(pos)
    return by_map


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True,
                    help="候选采集目录（可多次）")
    ap.add_argument("--buffer-m", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    dev = dev_positions()
    print(f"[iso] 开发集：{sum(len(v) for v in dev.values())} 帧（有位姿），"
          f"地图 {sorted(dev)}")
    report = {"buffer_m": args.buffer_m,
              "dev_frames_with_pose": sum(len(v) for v in dev.values()),
              "dev_maps": sorted(dev), "candidates": {}}
    ok_all = True
    for d_str in args.dir:
        d = Path(d_str)
        name, pos, missing = positions(d)
        same_map = dev.get(name, [])
        entry = {"map_name": name, "frames_with_pose": len(pos),
                 "frames_missing_pose": missing}
        if not same_map:
            entry["verdict"] = "isolated_other_map"
            entry["why"] = "开发集没有该地图的帧：不同地图不构成空间邻近"
        else:
            worst = None
            for p in pos:
                for q in same_map:
                    dist = sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5
                    worst = dist if worst is None or dist < worst else worst
            entry["min_distance_m"] = None if worst is None else round(worst, 2)
            if worst is None:
                entry["verdict"] = "unknown_no_pose"
                entry["why"] = "候选帧没有位姿：无法核对（缺位姿 ≠ 离得远）"
            elif worst > args.buffer_m:
                entry["verdict"] = "isolated"
            else:
                entry["verdict"] = "too_close"
                entry["why"] = f"最近 {worst:.1f} m <= 缓冲 {args.buffer_m:.0f} m"
        if entry["verdict"] not in ("isolated", "isolated_other_map"):
            ok_all = False
        report["candidates"][str(d)] = entry
        print(f"[iso] {d.name}: map={name} 有位姿 {len(pos)} 帧"
              f"（缺位姿 {missing}）-> {entry['verdict']}"
              + (f"，最近 {entry.get('min_distance_m')} m" if "min_distance_m" in entry else ""))
    report["all_isolated"] = ok_all
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[iso] -> {args.out}")
    print(f"[iso] 结论：{'全部通过（>缓冲 或 异图）' if ok_all else '有候选不满足隔离'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
