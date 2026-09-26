"""修复已标注目录的 meta：补回打包 meta 里被丢掉的键（一次性数据修复）。

背景：`m5_annotate_manual.py` 的 sidecar 种子只看输出目录，`--out` 与
`--frames-dir` 不同时种子为空 → 已标注的 `meta.json` 只剩
map/source/annotation/frames，**丢了 `cameras`**（以及 width/height/classes/
palette_source）。身份探针的投影依赖 `cameras`，所以 136 帧权威真值集上
**候选身份/覆盖率指标测不出来**（R2 要的正是这些）。

本脚本只**补缺失键**：不动标注器写下的 `label_source`/`annotation`/`frames`，
并把补了什么记进 `meta_repair.json`。可重复运行（已补齐的不再改）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 不覆盖标注器写入的键
PROTECTED = ("label_source", "annotation", "frames")


def repair_dir(reviewed: Path, package: Path) -> dict:
    rp = reviewed / "meta.json"
    pp = package / "meta.json"
    if not rp.is_file() or not pp.is_file():
        return {"dir": str(reviewed), "status": "skipped",
                "why": "meta missing (reviewed or package)"}
    rblob = json.loads(rp.read_text(encoding="utf-8"))
    pblob = json.loads(pp.read_text(encoding="utf-8"))
    added = {}
    for k, v in pblob.items():
        if k in PROTECTED or k == "frames":
            continue
        if k not in rblob:
            rblob[k] = v
            added[k] = v if not isinstance(v, (dict, list)) else type(v).__name__
    if added:
        rp.write_text(json.dumps(rblob, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    return {"dir": str(reviewed), "package": str(package),
            "status": "repaired" if added else "already-complete",
            "added": sorted(added), "has_cameras": bool(rblob.get("cameras"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-dir",
                    default="logs/experiments/review_pack_20260926")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    pack_dir = Path(args.pack_dir)
    out: list = []
    for tree, pkg_tree in (("reviewed", "packages"),
                           ("reviewed_full", "packages_full")):
        for d in sorted((pack_dir / tree).glob("*/front_main")):
            pkg = pack_dir / pkg_tree / d.parent.name / "front_main"
            rec = repair_dir(d, pkg)
            # `pkg_pkg_X` 这种包是从"已复核目录"再打的包，它自己的 meta 当时也没有
            # cameras -> 用对应的原始包（去掉一层 pkg_ 前缀）当供体再补一次
            name = d.parent.name
            if not rec.get("has_cameras") and name.startswith("pkg_pkg_"):
                donor = pack_dir / pkg_tree / ("pkg_" + name[len("pkg_pkg_"):])                     / "front_main"
                rec2 = repair_dir(d, donor)
                rec = {**rec2, "donor": str(donor),
                       "status": "repaired-from-donor" if rec2.get("added")
                                 else rec.get("status")}
            out.append(rec)
            print(f"[repair] {rec['status']:16s} {d.parent.name:52s} "
                  f"cameras={rec.get('has_cameras')} "
                  f"added={','.join(rec.get('added') or [])}")
    outp = Path(args.out) if args.out else (pack_dir / "meta_repair.json")
    outp.write_text(json.dumps({"repaired": out}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    n_ok = sum(1 for r in out if r.get("has_cameras"))
    print(f"[repair] {n_ok}/{len(out)} 个目录现在带 cameras -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
