"""修复"标注文件与帧名错位"（就绪核对会先把它拦下来）。

实测背景（2026-09-27）：`m5_annotate_manual.py` 的保存名原来用"先自增再命名"，
于是 15 帧标注完成后目录里出现**整批错位 +1**：`frame_00001.npz` 里装的其实是
`frame_00000` 的图像与标签，同时多出一个未标注的原帧和一个越界名。就绪核对
（`m5_annotation_readiness.py`）会因此拒绝（"内容与文件名对不上"）。

好消息是导出帧**自带 `identity_json.identity_provenance`**，写明它来自哪个源文件
——所以这类错位可以**逐帧自证并机械修复**，不需要重标。本脚本：

1. 逐帧读 `identity_provenance`（`pos`/`heading`/`exposure` 任一）得到源名；
2. 校验映射**一一对应**且目标名都在目录里，否则**拒绝修复**（交回重标）；
3. **两阶段改名**（先全部 `.tmp` 再落到目标名），帧与预览图一起改；
4. 同步 `meta.json` 的 `frames[]` 路径，并丢掉没有 `classes_painted` 的种子条目
   （那是未标注的原帧副本）；
5. 默认 **dry-run**，`--apply` 才动手；修复后写 `repair_names_*.json` 留痕。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_repair_annotated_names.py \\
        --package logs/experiments/annotate_pkg_e1_jv_20260927
    # 确认无误后
    .venv\\Scripts\\python.exe scripts\\m5_repair_annotated_names.py \\
        --package logs/experiments/annotate_pkg_e1_jv_20260927 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PROV_KEYS = ("pos", "heading", "exposure")


def source_name(npz_path: Path) -> str:
    """这一帧自报的源文件名（来自 ``identity_provenance``）；读不到返回空串。"""
    z = np.load(npz_path)
    if "identity_json" not in z.files:
        return ""
    try:
        ij = json.loads(str(z["identity_json"]))
    except Exception:                                     # noqa: BLE001
        return ""
    prov = ij.get("identity_provenance") or {}
    for k in _PROV_KEYS:
        raw = str(prov.get(k) or "")
        if raw.startswith("npz:"):
            return raw.split("npz:", 1)[1]
    return ""


def plan_view(view: Path) -> dict:
    """一个视角的改名计划；不写任何文件。

    也吃**半完成状态**：两阶段改名如果中断，会留下 `frame_XXXXX.npz.tmp`
    （Windows 上 `Path.rename` 遇到已存在的目标会报 WinError 183，实测踩到）。
    这些 tmp 仍在计划里，按它们自己的 provenance 落到目标名即可续跑。
    """
    frames = sorted(view.glob("frame_*.npz")) + sorted(view.glob("*.npz.tmp"))
    mapping: dict = {}
    unknown: list = []
    for f in frames:
        # 逻辑名 = 去掉半完成状态后缀：映射与校验都按逻辑名做；`apply_view`
        # 两种状态都能接（源还在就挪 tmp，已在 tmp 就直接落到目标名）
        logical = f.name[:-4] if f.name.endswith(".tmp") else f.name
        src = source_name(f)
        if not src:
            unknown.append(logical)
            continue
        if src != logical:
            mapping[logical] = src
    out = {"view": str(view), "n_frames": len(frames), "mapping": mapping,
           "no_provenance": unknown, "refused": []}
    if not mapping:
        return out
    names = {f.name[:-4] if f.name.endswith(".tmp") else f.name for f in frames}
    if len(set(mapping.values())) != len(mapping):
        out["refused"].append("映射不是一一对应（两个文件指向同一源名）")
    missing = sorted(set(mapping.values()) - names)
    if missing:
        out["refused"].append(f"目标名不在目录里：{missing[:3]}")
    return out


def apply_view(view: Path, mapping: dict) -> dict:
    """两阶段改名 + meta 路径改写；返回留痕记录。

    用 ``os.replace``：Windows 上 ``Path.rename`` 遇到已存在的目标会报
    WinError 183（实测踩到），而这里的语义正是"用标注版**替换**未标注的原帧"。
    也接受**已是 tmp** 的源（续跑半完成状态）。
    """
    moved: list = []
    for a in mapping:
        src = view / a
        if src.is_file():
            os.replace(src, view / (a + ".tmp"))
        pv = view / a.replace("frame_", "preview_").replace(".npz", ".png")
        if pv.is_file():
            os.replace(pv, view / (pv.name + ".tmp"))
    for a, b in mapping.items():
        tmp = view / (a + ".tmp")
        if tmp.is_file():
            os.replace(tmp, view / b)
        tmpv = (view / (a.replace("frame_", "preview_").replace(".npz", ".png")
                        + ".tmp"))
        if tmpv.is_file():
            os.replace(tmpv, view / b.replace("frame_", "preview_")
                       .replace(".npz", ".png"))
        moved.append({"from": a, "to": b})
    meta_p = view / "meta.json"
    dropped = 0
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        recs = [dict(fr) for fr in (meta.get("frames") or [])]
        for r in recs:
            p = Path(str(r.get("path") or "")).name
            if p in mapping:
                r["path"] = mapping[p]
        keep = [r for r in recs if r.get("classes_painted")]
        dropped = len(recs) - len(keep)
        meta["frames"] = keep
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    return {"moved": moved, "meta_dropped_seed_entries": dropped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default=None, help="任务包目录（自动找视角子目录）")
    ap.add_argument("--dir", action="append", default=[],
                    help="直接指定视角目录（可多次）")
    ap.add_argument("--apply", action="store_true", help="真正改名（默认 dry-run）")
    args = ap.parse_args(argv)

    views: list = [Path(d) for d in args.dir]
    if args.package:
        pkg = Path(args.package)
        views += [d for d in sorted(pkg.iterdir())
                  if d.is_dir() and list(d.glob("frame_*.npz"))]
    if not views:
        print("[repair] 没有可检查的视角目录")
        return 2

    records = []
    bad = 0
    for v in views:
        rep = plan_view(v)
        records.append(rep)
        if rep["refused"]:
            bad += 1
            print(f"[repair] 拒绝修复 {v}：{'; '.join(rep['refused'])}")
            continue
        if not rep["mapping"]:
            print(f"[repair] {v.name}: 无需改名（{rep['n_frames']} 帧）"
                  + (f"，{len(rep['no_provenance'])} 帧无 provenance"
                     if rep["no_provenance"] else ""))
            continue
        print(f"[repair] {v.name}: 需改名 {len(rep['mapping'])}/{rep['n_frames']} 帧"
              + ("（dry-run）" if not args.apply else ""))
        for a, b in list(sorted(rep["mapping"].items()))[:3]:
            print(f"[repair]   {a} -> {b}")
        if len(rep["mapping"]) > 3:
            print(f"[repair]   …共 {len(rep['mapping'])} 个")
        if args.apply:
            rep["applied"] = apply_view(v, rep["mapping"])
            print(f"[repair]   已改名并同步 meta（丢掉未标注种子条目 "
                  f"{rep['applied']['meta_dropped_seed_entries']} 条）")
    if args.apply:
        out = Path(args.package or views[0].parent) / (
            f"repair_names_{time.strftime('%Y%m%d_%H%M%S')}.json")
        out.write_text(json.dumps(records, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print(f"[repair] 记录 -> {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
