"""标注就绪核对：人工标注落地后，先验证"能进审计"，再打印后续两条命令。

为什么需要它：项目里已经发生过"48 帧人工修订因为**丢了身份**被审计拒收且不可
恢复"（见 `m5_annotate_package.py` 的说明）。标注是人的时间，不能在跑训练那一刻
才发现包不合法。本脚本逐目录核对：

1. **凭据**：`meta.json` 存在且 `label_source == human_revision`（这是审计把目录
   认成 verified 的唯一依据）；
2. **标注真的落了盘**：每个 `frame_*.npz` 都有 `label` 数组（只有 `colour` 的是
   还没标的包副本）；
3. **身份完整**：`annotation.identity_missing` 为空、每帧有 `pos`/`heading`
   （缺位姿 = 空间隔离核对不了 = 审计可能拒收）；
4. **标的是什么**：逐帧 `classes_painted.line` 的分布——负例包应几乎全 0
   （"确认无线"），正例包应大多数 >0（画了漆线）；
5. **空间隔离**：复用 `m5_collect_isolation_check` 的判据（异图 / >50 m / 缺位姿
   记 UNKNOWN）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_annotation_readiness.py \\
        --dir logs/experiments/annotate_pkg_e1_jv_20260927/front_main \\
        --dir logs/experiments/annotate_pkg_e2_it3_20260927/front_main \\
        --out logs/experiments/annotation_readiness_20260927.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def check_dir(d: Path) -> dict:
    """单个视角目录的就绪检查；返回结构化结论（不抛异常）。"""
    from m5_collect_isolation_check import positions

    out: dict = {"dir": str(d), "ok": False, "reasons": [], "notes": []}
    meta_p = d / "meta.json"
    if not meta_p.is_file():
        out["reasons"].append("没有 meta.json：标注器的凭据与身份都写在这里")
        return out
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    out["label_source"] = meta.get("label_source")
    if str(meta.get("label_source")) != "human_revision":
        out["reasons"].append(
            f"label_source={meta.get('label_source')!r}，不是 human_revision："
            "审计不会把它当 verified（负例资格/可晋级训练都会被拒）")
    ann = meta.get("annotation") or {}
    out["reviewer"] = ann.get("reviewer")
    out["annotated_at"] = ann.get("annotated_at")
    missing = list(ann.get("identity_missing") or [])
    if missing:
        out["reasons"].append(f"身份缺字段 {missing}：空间隔离无法核对")
    if not ann:
        out["notes"].append("没有 annotation 段：可能还没用标注器打开过这个目录")
    frames = sorted(d.glob("frame_*.npz"))
    out["n_frames"] = len(frames)
    if not frames:
        out["reasons"].append("目录里没有 frame_*.npz")
        return out
    unlabeled, line_px = [], []
    # 帧名 ↔ 帧内身份**对齐检查**（实测踩到：保存名先自增 -> 整批错位 +1，
    # 目录里同时多出未标注原帧与越界名）。导出帧自带 identity_provenance，
    # 它写明这一帧来自哪个源文件——与自己的文件名不符就是错位。
    misaligned: list = []
    expected_names = {Path(str(fr.get("path") or "")).name
                      for fr in (meta.get("frames") or []) if fr.get("path")}
    by_name = {str(fr.get("path", "")).split("/")[-1]: fr
               for fr in (meta.get("frames") or [])}
    for f in frames:
        try:
            z = np.load(f)
        except Exception as exc:                      # noqa: BLE001
            out["reasons"].append(f"{f.name}: 读不了（{type(exc).__name__}）")
            continue
        if "label" not in z.files:
            unlabeled.append(f.name)
            continue
        cp = (by_name.get(f.name) or {}).get("classes_painted") or {}
        line_px.append(int(cp.get("line", int((z["label"] == 2).sum()))))
        _prov = ""
        if "identity_json" in z.files:
            try:
                _ij = json.loads(str(z["identity_json"].item()
                                     if getattr(z["identity_json"], "size", 1) == 1
                                     else z["identity_json"]))
                _prov = str((_ij.get("identity_provenance") or {}).get("pos")
                            or (_ij.get("identity_provenance") or {}).get("exposure")
                            or "")
            except Exception:                          # noqa: BLE001
                _prov = ""
        if _prov.startswith("npz:"):
            _src = Path(_prov.split("npz:", 1)[1]).name
            if _src and _src != f.name:
                misaligned.append((f.name, _src))
        elif "exposure" in z.files:
            _exp_meta = (by_name.get(f.name) or {}).get("exposure")
            if _exp_meta is not None:
                try:
                    if int(z["exposure"]) != int(_exp_meta):
                        misaligned.append((f.name, f"exposure={int(z['exposure'])}"
                                                   f" vs meta {int(_exp_meta)}"))
                except Exception:                      # noqa: BLE001
                    pass
    if misaligned:
        out["misaligned_frames"] = misaligned[:5]
        out["reasons"].append(
            f"{len(misaligned)} 帧的**内容与文件名对不上**（例如 "
            f"{misaligned[0][0]} 实际来自 {misaligned[0][1]}）：标注会贴到错的帧上，"
            "必须重标或按源名改正后再用")
    if expected_names:
        _extra = sorted({f.name for f in frames} - expected_names)
        if _extra:
            out["reasons"].append(
                f"目录里有不属于本批的帧（多余文件）：{_extra[:3]}——"
                "通常说明保存命名与源帧名不一致")
    out["unlabeled_frames"] = len(unlabeled)
    if unlabeled:
        out["reasons"].append(
            f"{len(unlabeled)} 帧没有 label（例如 {unlabeled[0]}）：还没标注")
    if line_px:
        n_pos = sum(1 for v in line_px if v > 0)
        out["line_stats"] = {"n": len(line_px), "n_with_line": n_pos,
                             "min": min(line_px), "max": max(line_px),
                             "median": sorted(line_px)[len(line_px) // 2]}
        out["kind"] = ("negative_pack" if n_pos == 0 else
                       "positive_pack" if n_pos > len(line_px) / 2 else "mixed")
        if out["kind"] == "mixed":
            out["notes"].append(
                f"有线帧 {n_pos}/{len(line_px)}：既不是纯负例也不是纯正例，"
                "按用途分别使用（负例只用 line=0 的帧）")
    # 空间隔离（异图 / >50 m / 缺位姿 UNKNOWN）
    try:
        map_name, pos, n_missing = positions(d)
        out["map_name"] = map_name
        out["frames_missing_pose"] = n_missing
        if n_missing:
            out["reasons"].append(f"{n_missing} 帧缺位姿")
    except Exception as exc:                          # noqa: BLE001
        out["notes"].append(f"隔离检查未跑：{type(exc).__name__}: {exc}")
    out["ok"] = not out["reasons"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    reports = [check_dir(Path(d)) for d in args.dir]
    ready = [r for r in reports if r["ok"]]
    for r in reports:
        tag = "就绪" if r["ok"] else "未就绪"
        print(f"[ready] {tag} {r['dir']}（{r.get('n_frames', 0)} 帧"
              + (f"，{r.get('kind')}" if r.get("kind") else "") + "）")
        for x in r["reasons"]:
            print(f"[ready]   阻止：{x}")
        for x in r["notes"]:
            print(f"[ready]   提示：{x}")
        if r.get("line_stats"):
            s = r["line_stats"]
            print(f"[ready]   漆线像素：{s['n_with_line']}/{s['n']} 帧 >0，"
                  f"中位 {s['median']}，范围 {s['min']}~{s['max']}")
    neg = [r for r in ready if r.get("kind") == "negative_pack"]
    pos = [r for r in ready if r.get("kind") in ("positive_pack", "mixed")]
    print(f"[ready] 结论：{len(ready)}/{len(reports)} 就绪"
          f"（可用负例包 {len(neg)}，含正例包 {len(pos)}）")
    if neg:
        print("[ready] 合格版 E1 的训练负例目录：")
        for r in neg:
            print(f"[ready]   {r['dir']}")
    if pos:
        print("[ready] 可晋级线通道训练的正例目录：")
        for r in pos:
            print(f"[ready]   {r['dir']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"dirs": reports, "n_ready": len(ready),
             "n_total": len(reports)}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(f"[ready] -> {args.out}")
    return 0 if len(ready) == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
