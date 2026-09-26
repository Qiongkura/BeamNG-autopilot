"""E1 研究臂的负例池筛选：用**与模型无关**的漆料判据挑"证据支持的负例"。

背景：引擎标注（`beamng_annotation`）**不提供 line 类**，所以"标签全零"是**缺失**
而不是"确认无线"（T10）。研究臂要用这类帧当负例，就必须给出**可测的证据**而不是
假设。本脚本用与分割模型无关的经典漆料判据逐帧筛查：

* 黄色漆料：`beamng_autopilot.vision.yellow_line_mask.yellow_line_mask`（HSV 先验）；
* 白色漆料：V ≥ `--white-v-min` 且 S ≤ `--white-s-max` 的像素数 ≥ `--min-paint-px`。

判据在跑之前冻结（见 `docs/T14_S6_EXPERIMENT_DESIGN_20260926.md` §4.1）：
一帧只有在**两种漆料都低于阈值**时才进负例池（保守：宁可少收，也不教模型
"这里没有线"而那里可能真有漆）。输出候选清单 JSON + 可选的派生负例目录
（复制帧、带 lineage 的 meta，**不改原目录**）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_negative_pool_audit.py \\
        --dir logs/m5_seg/collect_t14_collect_it2_20260925_20260925_235858/front_main \\
        --dir logs/m5_seg/collect_t14_collect_wc_20260925_20260925_234107/front_main \\
        --out logs/experiments/e1_negative_pool_20260927.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _read_meta(d: Path) -> dict:
    for cand in (d / "meta.json", d.parent / "meta.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def paint_pixels(colour: np.ndarray, *, white_v_min: int,
                 white_s_max: int) -> dict:
    """与模型无关的漆料像素计数（黄 = HSV 先验，白 = 亮且低饱和）。"""
    from beamng_autopilot.vision.yellow_line_mask import yellow_line_mask
    yellow = int(np.count_nonzero(yellow_line_mask(colour)))
    arr = colour.astype(np.int16)
    v = arr.max(axis=2)
    s = np.where(v > 0, (v - arr.min(axis=2)) * 255 // np.maximum(v, 1), 0)
    white = int(np.count_nonzero((v >= white_v_min) & (s <= white_s_max)))
    return {"yellow_px": yellow, "white_px": white,
            "paint_px": yellow + white}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True,
                    help="候选负例目录（可多次）")
    ap.add_argument("--min-paint-px", type=int, default=64,
                    help="一帧超过这个漆料像素数就算'疑似有漆'，不进负例池")
    ap.add_argument("--white-v-min", type=int, default=200)
    ap.add_argument("--white-s-max", type=int, default=60)
    ap.add_argument("--max-per-dir", type=int, default=None,
                    help="每个目录最多取多少帧（缺省全取）")
    ap.add_argument("--derive-dir", default=None,
                    help="把合格帧复制到这个目录（生成带 lineage 的派生负例目录）")
    ap.add_argument("--screen-model", default=None,
                    help="用这个权重做**模型筛查**：只保留'模型一个线像素都没预测'"
                         "的帧（经典漆料判据实测不可用：在人工确认无漆的帧上也会"
                         "命中 4 万+ 像素，见 docs 的校准记录）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = {"criteria": {"min_paint_px": args.min_paint_px,
                        "white_v_min": args.white_v_min,
                        "white_s_max": args.white_s_max,
                        "classic_paint_rules_usable": False,
                        "screen_model": args.screen_model},
           "dirs": {}, "eligible": [], "excluded_suspect": [],
           "excluded_unknown_or_line": [], "excluded_model_paint": []}
    _seg = None
    if args.screen_model:
        from beamng_autopilot.vision.segmentation import Segmenter
        _seg = Segmenter(model_path=args.screen_model, device=args.device)
        out["criteria"]["screen_model_device"] = str(
            getattr(_seg, "device", "unknown"))
    for d_str in args.dir:
        d = Path(d_str)
        meta = _read_meta(d)
        frames = sorted(d.glob("frame_*.npz")) or sorted(d.glob("**/frame_*.npz"))
        info = {"n_frames": len(frames), "n_eligible": 0, "n_suspect": 0,
                "map_name": meta.get("map_name"), "source_id": meta.get("source_id"),
                "label_source": meta.get("label_source")}
        for f in frames:
            z = np.load(f)
            if "colour" not in z.files or "label" not in z.files:
                out["excluded_unknown_or_line"].append(
                    {"frame": str(f), "why": "npz 缺 colour/label"})
                continue
            lab = z["label"]
            if (lab == 255).any() or (lab == 2).any():
                out["excluded_unknown_or_line"].append(
                    {"frame": str(f),
                     "why": ("label has unknown pixels" if (lab == 255).any()
                             else "label has line pixels")})
                continue
            pp = paint_pixels(np.asarray(z["colour"]), white_v_min=args.white_v_min,
                              white_s_max=args.white_s_max)
            if _seg is not None:
                _road, _line = _seg.predict(np.asarray(z["colour"]))
                n_pred = int(np.count_nonzero(_line))
                if n_pred > 0:
                    info["n_model_paint"] = info.get("n_model_paint", 0) + 1
                    out["excluded_model_paint"].append(
                        {"frame": str(f), "pred_line_px": n_pred, **pp})
                    continue
            else:
                n_pred = None
            if _seg is None and pp["paint_px"] > args.min_paint_px:
                info["n_suspect"] += 1
                out["excluded_suspect"].append({"frame": str(f), **pp})
                continue
            info["n_eligible"] += 1
            out["eligible"].append({"frame": str(f), "dir": str(d),
                                    "pred_line_px": n_pred, **pp})
        out["dirs"][str(d)] = info
        if args.max_per_dir:
            per = [e for e in out["eligible"] if e["dir"] == str(d)]
            keep = per[:int(args.max_per_dir)]
            drop = per[int(args.max_per_dir):]
            for e in drop:
                out["eligible"].remove(e)
            info["n_eligible"] = len(keep)
            info["n_dropped_by_max"] = len(drop)

    if args.derive_dir:
        dst_root = Path(args.derive_dir)
        made: dict = {}
        for e in out["eligible"]:
            src = Path(e["frame"])
            d = Path(e["dir"])
            meta = _read_meta(d)
            key = str(d)
            if key not in made:
                sub = dst_root / f"neg_{len(made)}" / "front_main"
                sub.mkdir(parents=True, exist_ok=True)
                frames_meta = []
                m = dict(meta)
                m["frames"] = frames_meta
                m["note"] = ("E1 研究臂派生负例：源目录 "
                             f"{d.as_posix()}；筛选判据（跑前冻结）：标签无未知像素、"
                             "无 line 类，且经典漆料像素（黄 HSV 先验 + 亮低饱和白）"
                             f"≤ {args.min_paint_px}。标签档位仍为引擎标注 -> "
                             "只能作研究臂负例，不构成 T10 的'已确认负例'。")
                m["lineage"] = {"derived_from": d.as_posix(),
                                "criteria": out["criteria"],
                                "created": "2026-09-27",
                                "purpose": "S6 E1 research-arm negatives (evidence-screened)"}
                made[key] = {"sub": sub, "meta": m, "frames": frames_meta}
            # 逐帧 meta 从**源 meta** 复制（保留 exposure/位姿等），只改 path：
            # 丢掉位姿会让空间隔离审计查不了（缺位姿 = UNKNOWN，不是"干净"）
            _src_meta = _read_meta(Path(e["dir"]))
            _by_name = {str(fr.get("path", "")).split("/")[-1]: fr
                        for fr in (_src_meta.get("frames") or [])}
            _fr = dict(_by_name.get(src.name) or {})
            _fr["path"] = f"front_main/{src.name}"
            _fr.setdefault("view", "front_main")
            made[key]["frames"].append(_fr)
            shutil.copy2(src, made[key]["sub"] / src.name)
        for key, v in made.items():
            (v["sub"].parent / "meta.json").write_text(
                json.dumps(v["meta"], indent=1, ensure_ascii=False),
                encoding="utf-8")
        out["derived"] = {k: {"dir": str(v["sub"]), "n": len(v["frames"])}
                          for k, v in made.items()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"[negpool] 判据 {out['criteria']}")
    for d_str, info in out["dirs"].items():
        print(f"[negpool] {d_str}: 帧 {info['n_frames']} -> 合格 {info['n_eligible']}"
              f"（疑似有漆 {info['n_suspect']}）map={info['map_name']} "
              f"source={info['source_id']}")
    print(f"[negpool] 合计合格负例 {len(out['eligible'])} 帧")
    if out.get("derived"):
        for k, v in out["derived"].items():
            print(f"[negpool] 派生目录 {v['dir']}（{v['n']} 帧）")
    if args.out:
        print(f"[negpool] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
