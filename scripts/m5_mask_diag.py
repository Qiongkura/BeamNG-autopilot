"""语义掩码侧右线诊断（预注册 T16_MASK_SIDE_DIAG_PREREG_20261104）。

离线、只读：在认证池帧上比较
  1) 引擎真值线像素（label==2）按漆色拆分的 yellow/white GT；
  2) 模型 raw argmax 线掩码（未后处理）与 post 后处理线掩码；
计算黄线召回、后处理删除比例、预测碎段跨度，判定 HA/HB/HC。

用法：
  .venv\\Scripts\\python.exe scripts\\m5_mask_diag.py \\
      --runs <front_main 目录...> --model logs/m5_seg/seg_model_v28_sslA/best.pt \\
      --out logs/m5_seg/mask_diag.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与经典 CV 黄线先验同阈值（只用于把 GT 线像素按漆色拆分）
YELLOW_HUE = (12, 45)
YELLOW_SAT_MIN = 30
YELLOW_VAL_MIN = 100


def split_line_gt(colour: np.ndarray, label: np.ndarray):
    """引擎真值线像素 -> (yellow_gt, white_gt) bool 掩码。"""
    lab = np.asarray(label)
    line = lab == 2
    hsv = cv2.cvtColor(np.asarray(colour, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (line & (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1])
              & (sat >= YELLOW_SAT_MIN) & (val >= YELLOW_VAL_MIN))
    return yellow, line & ~yellow


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    g = int(np.asarray(gt, dtype=bool).sum())
    if g == 0:
        return float("nan")
    p = int((np.asarray(pred, dtype=bool) & np.asarray(gt, dtype=bool)).sum())
    return p / g


def component_stats(mask: np.ndarray) -> dict:
    """连通段数量与最大跨度（像素，bbox 长边）。"""
    m = np.asarray(mask, dtype=np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    spans = [max(int(stats[i, cv2.CC_STAT_WIDTH]),
                 int(stats[i, cv2.CC_STAT_HEIGHT]))
             for i in range(1, n)]
    return {"count": int(n - 1),
            "max_span_px": int(max(spans)) if spans else 0,
            "mean_span_px": round(float(np.mean(spans)), 2) if spans else 0.0}


def _frames(runs: list[str]) -> list[Path]:
    out: list[Path] = []
    for r in runs:
        out.extend(sorted(Path(p) for p in glob.glob(
            str(Path(r) / "**" / "frame_*.npz"), recursive=True)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="含 frame_*.npz 的目录（可递归）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from beamng_autopilot.vision.segmentation import Segmenter

    seg = Segmenter(model_path=args.model, device=args.device)
    rows = []
    for f in _frames(args.runs):
        d = np.load(f)
        colour = d["colour"]
        label = d["label"]
        yellow_gt, white_gt = split_line_gt(colour, label)
        _maps, _raw_road, raw_line = seg.predict_proba(colour)
        _road, post_line = seg.predict(colour)
        rec = {
            "file": str(f),
            "gt_yellow_px": int(yellow_gt.sum()),
            "gt_white_px": int(white_gt.sum()),
            "raw_yellow_recall": recall(raw_line, yellow_gt),
            "post_yellow_recall": recall(post_line, yellow_gt),
            "raw_white_recall": recall(raw_line, white_gt),
            "post_white_recall": recall(post_line, white_gt),
            "raw_line_px": int(np.asarray(raw_line).sum()),
            "post_line_px": int(np.asarray(post_line).sum()),
        }
        raw_r = rec["raw_yellow_recall"]
        if np.isfinite(raw_r) and raw_r > 0:
            rec["drop_frac_yellow"] = round(
                1.0 - (rec["post_yellow_recall"] / raw_r), 3)
        else:
            rec["drop_frac_yellow"] = None
        rec["yellow_tp_components"] = component_stats(
            np.asarray(raw_line) & yellow_gt)
        rows.append(rec)

    def mean(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {k: mean(k) for k in (
        "gt_yellow_px", "gt_white_px",
        "raw_yellow_recall", "post_yellow_recall",
        "raw_white_recall", "post_white_recall",
        "raw_line_px", "post_line_px", "drop_frac_yellow")}
    summary["frames"] = len(rows)
    summary["yellow_mean_span_px"] = round(float(np.mean(
        [r["yellow_tp_components"]["mean_span_px"] for r in rows])), 2) \
        if rows else None
    out = {"schema": "mask-side-diag-v1", "model": str(args.model),
           "runs": [str(x) for x in args.runs], "summary": summary,
           "frames": rows}
    text = json.dumps(out, ensure_ascii=False, indent=1,
                      default=lambda o: None)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[mask-diag] -> {args.out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
