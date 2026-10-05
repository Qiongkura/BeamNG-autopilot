"""最终集组成与封存：**路段不重叠 + 已认证 + 内容零交集**，一条命令。

为什么要有它（2026-10-05 两次错误封存的教训）：最终集的判据是"评估帧不与训练
同路段（相邻站点）"，而这件事**必须可机读核对**——

* 导出报告的 ``dir`` 指向 ``front_main`` 子目录，用目录名里的锚点号做过滤会
  **静默不命中**（我第一次就是这么错的，v1/v2 两个封存的声明因此不可核对，
  已标 INVALID）；
* 各生成批次的锚点编号**互不可比**（每批自己的 offset），跨批次比锚点号是错的；
* 正确判据是生成侧 ``scene_<name>.json`` 里的 **``scene.road_id``**。

本脚本因此只做三件事，且每一步都打印证据：
1. **路段集**：从 ``--train-batch`` 的 ``scene_*.json`` 取训练路段；
2. **候选筛选**：``--candidate OUT_ROOT=GEN_BATCH``（可重复）里已认证
   （``export_report.json``）的包，按 ``scene.road_id ∉ 训练路段`` 保留；
3. **封存**：调 ``experiments.final_set.seal``，并报线帧数是否达到
   ``--min-line-frames``（预注册下限；不足时**不阻止封存**但明确打印"不足"，
   避免把"部分可用"冒充成"达标"）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_final_set_compose.py `
        --train-batch logs/experiments/t16_scenes_devdist_20260930 `
        --train-batch logs/experiments/t16_scenes_struct_20260930 `
        --train-batch logs/experiments/t16_scenes_structneg_20261001 `
        --candidate logs/experiments/t16_autotruth_big_20260928=logs/experiments/t16_scenes_big_20260928 `
        --candidate logs/experiments/t16_final_pool_cert_20261005=logs/experiments/t16_final_pool_20261005 `
        --name v4-20261005 --out logs/experiments/final_set_v4_20261005/seal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def road_map(batch) -> dict:
    """``{场景名: road_id}``——从生成批次的逐场景 JSON 取（不猜、不从目录名推）。"""
    out = {}
    for f in sorted(Path(batch).glob("scene_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sc = d.get("scene") or {}
        if sc.get("road_id") is not None:
            out[f.name[len("scene_"):-len(".json")]] = str(sc["road_id"])
    return out


def compose(train_roads: set, candidates: list) -> dict:
    """纯函数：给定训练路段集合与 ``[(type, scene, road_id, frames)]``，分组。

    返回 ``{"keep": [...], "drop_same_road": [...], "drop_unknown_road": [...]}``。
    路段未知（``road_id is None``）**不保留**——判不了就不收（不猜）。
    """
    keep, same, unknown = [], [], []
    for typ, scene, rid, frames in candidates:
        if rid is None:
            unknown.append((scene, rid))
        elif str(rid) in train_roads:
            same.append((scene, rid))
        else:
            keep.append((typ, scene, rid, frames))
    return {"keep": keep, "drop_same_road": same, "drop_unknown_road": unknown}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-batch", action="append", required=True,
                    help="训练侧生成批次目录（可重复）")
    ap.add_argument("--candidate", action="append", required=True,
                    metavar="OUT_ROOT=GEN_BATCH",
                    help="候选导出根目录=其生成批次（可重复）")
    ap.add_argument("--min-line-frames", type=int, default=64,
                    help="带真值线帧的预注册下限（默认 64）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sealed-by", default="agent: m5_final_set_compose.py")
    args = ap.parse_args()

    from beamng_autopilot.experiments.final_set import seal
    from beamng_autopilot.experiments.manifest import dir_group
    from beamng_autopilot.experiments.protocol import protocol_hash

    train_roads = set()
    for b in args.train_batch:
        m = road_map(b)
        train_roads |= set(m.values())
        print(f"[compose] 训练批次 {b}: {len(m)} 场景 / "
              f"{len(set(m.values()))} 条道路")
    print(f"[compose] 训练路段合计 {len(train_roads)} 条")

    cands = []
    for spec in args.candidate:
        out_root, _, gen = str(spec).partition("=")
        if not gen:
            raise SystemExit(f"--candidate 需要 OUT_ROOT=GEN_BATCH，收到 {spec!r}")
        roads = road_map(gen)
        rep = Path(out_root) / "export_report.json"
        if not rep.is_file():
            raise SystemExit(f"{out_root} 没有 export_report.json（未认证？）")
        n = 0
        for e in json.loads(rep.read_text(encoding="utf-8"))["exported"]:
            scene = e.get("scene")
            frames = sorted(Path(e["dir"]).glob("*.npz"))
            if not frames:
                continue
            cands.append((e.get("type"), scene, roads.get(scene), frames))
            n += 1
        print(f"[compose] 候选 {out_root}: {n} 个已认证包（生成批次 {gen}）")

    res = compose(train_roads, cands)
    keep = res["keep"]
    line = [p for p in keep if p[0] != "known_no_line"]
    neg = [p for p in keep if p[0] == "known_no_line"]
    n_line = sum(len(f) for *_, f in line)
    n_neg = sum(len(f) for *_, f in neg)
    print(f"[compose] 保留 {len(keep)} 包：带线 {len(line)}/{n_line} 帧、"
          f"无线 {len(neg)}/{n_neg} 帧")
    print(f"[compose] 同路段排除 {len(res['drop_same_road'])} 包："
          f"{res['drop_same_road'][:8]}")
    if res["drop_unknown_road"]:
        print(f"[compose] 路段未知排除 {len(res['drop_unknown_road'])} 包（判不了不收）："
              f"{res['drop_unknown_road'][:8]}")
    for typ, scene, rid, fr in line:
        print(f"[compose]   带线 [{rid}] {typ:16s} {scene:26s} {len(fr)} 帧")
    if not keep:
        raise SystemExit("[compose] 没有可封存的包")
    if n_line < int(args.min_line_frames):
        print(f"[compose] ⚠ 线帧 {n_line} < 预注册下限 {args.min_line_frames}："
              f"**不达标**（可封存但只能算部分可用，不得冒充达标）")
    frames = [p for *_, fr in keep for p in fr]
    rec = seal(frames, name=args.name, dataset_id=f"t16-final-{args.name}",
               protocol_hash=protocol_hash(), out_dir=args.out,
               sealed_by=args.sealed_by,
               groups=sorted({dir_group(p.parent) for p in frames}))
    print(f"[compose] 封存 {rec['n_frames']} 帧 digest={rec['digest']} "
          f"协议={rec['protocol_hash']} 组 {len(rec['groups'])}")
    print(f"[compose] -> {Path(args.out) / 'final_set_seal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
