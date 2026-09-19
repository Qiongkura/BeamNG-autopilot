"""把「沿路面边缘画的那一笔」从 line 类里拆出来（标注修正）。

背景：`paved_shoulder` 标注包里，路面与土肩/草地交界的边缘被用「标线」笔
（键 1，标签值 2）描了边——这是标注时很自然的动作（“路到这里为止”），但它
把「漆画标线」这个类污染了。用这批标签微调出来的模型于是学会「铺装边缘 =
标线」，实测后果（2026-09-18 实车）：

* 车道配对在车底下抓到一条「线」（pair_debug 里 near_med ≈ −0.5~−1.3 m 的
  候选），配出的车道右边界落在车身里 → 车体越界 → 安全监视器 minimal_risk
  → 车在原地冻住；
* 真正的中心线（画面里那条）反而和边缘笔画混在一起。

修正规则（几何，可复核）：**真标线在路面内部**（到背景区的距离大），
**描边的笔画贴着背景**。所以：

1. 取 `solid = road ∪ line`（所有非背景=用户认为的"铺装面"）；
2. 对每个 line 像素算它到背景区（label==0）的距离；
3. 距离 < `--margin-px`（默认 8 px）的 line 像素判为"描边"，按"离路面内部更近
   就归路面(1)，离背景更近就归背景(0)"的方式消解——消解后的边界落在笔画
   中线上，和用户原来画的位置一致；
4. 距离 ≥ margin 的 line 像素保留为标线(2)（真正的中心线）。

用法：

    .venv\\Scripts\\python.exe scripts/m5_fix_boundary_strokes.py \\
        --src logs/m5_seg/paved_shoulder_labeled \\
        --out logs/m5_seg/paved_shoulder_labeled_fixed [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fix_label(label: np.ndarray, margin_px: int = 8) -> tuple[np.ndarray, dict]:
    """Split pavement-edge strokes out of the line class (see module doc).

    A stroke is found by CONNECTIVITY, not by a width threshold: the
    annotator's edge stroke touches the background by construction, while
    a real painted line is surrounded by road on both sides.  Labeling
    the line mask's connected components and dissolving the ones that
    reach the background therefore removes a stroke of ANY width (a
    distance threshold alone left the inner half of a wide stroke in the
    line class - measured 46% of the line pixels still "line" at
    margin 8 px, which is how the model learned "pavement edge = line").
    ``margin_px`` only decides how much of the dissolved stroke's pixels
    fall to the road side vs the background side, via the distance to
    each region: the resulting boundary lands on the stroke's own centre.
    """
    lab = np.asarray(label, dtype=np.uint8).copy()
    road = lab == 1
    line = lab == 2
    bg = lab == 0
    if not line.any() or not bg.any():
        return lab, {"line_px": int(line.sum()), "dropped": 0, "kept": 0}
    n_lbl, comp = cv2.connectedComponents(line.astype(np.uint8), 8)
    # A foreground component "reaches" the background when it contains a
    # pixel within ``margin_px`` of a background pixel (the stroke does
    # not have to touch it exactly - a 1-2 px gap between the brush and
    # the region edge is still the same stroke).
    k = 2 * int(max(1, margin_px)) + 1
    bg_neigh = cv2.dilate(bg.astype(np.uint8),
                          np.ones((k, k), np.uint8)).astype(bool)
    stroke = np.zeros_like(line)
    for k in range(1, n_lbl):
        sel = comp == k
        if bool((sel & bg_neigh).any()):
            stroke |= sel
    keep = line & ~stroke if line.any() else line
    if stroke.any():
        dt_bg = cv2.distanceTransform((~bg).astype(np.uint8),
                                      cv2.DIST_L2, 3)
        dt_road = cv2.distanceTransform((~road).astype(np.uint8),
                                        cv2.DIST_L2, 3)
        lab[stroke & (dt_road <= dt_bg)] = 1   # asphalt side of the stroke
        lab[stroke & (dt_road > dt_bg)] = 0    # gravel / grass side
    lab[keep] = 2
    return lab, {
        "line_px": int(line.sum()),
        "dropped": int(stroke.sum()),
        "kept": int(keep.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="标注修正：路面边缘笔画 -> 路面/背景")
    ap.add_argument("--src", required=True, help="原标注目录（*_labeled）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--margin-px", type=int, default=8,
                    help="line 像素离背景多近就判为描边（默认 8 px）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(src.glob("frame_*.npz"))
    if not files:
        print(f"[fix] {src} 里没有 frame_*.npz")
        return 2
    out = Path(args.out)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    tot = {"line_px": 0, "dropped": 0, "kept": 0}
    for fp in files:
        d = np.load(fp)
        colour = d["colour"]
        lab, st = fix_label(d["label"], margin_px=args.margin_px)
        for k in tot:
            tot[k] += st[k]
        if not args.dry_run:
            np.savez_compressed(out / fp.name, colour=colour, label=lab)
    n = len(files)
    print(f"[fix] {n} 帧, margin={args.margin_px}px")
    print(f"      line 像素 合计 {tot['line_px']}, 其中描边判定 "
          f"{tot['dropped']} ({100.0 * tot['dropped'] / max(1, tot['line_px']):.1f}%), "
          f"保留为标线 {tot['kept']}")
    print(f"      每帧平均：描边 {tot['dropped'] / n:.0f} px, "
          f"真标线 {tot['kept'] / n:.0f} px")
    if args.dry_run:
        print("[fix] dry-run，未写文件")
    else:
        print(f"[fix] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
