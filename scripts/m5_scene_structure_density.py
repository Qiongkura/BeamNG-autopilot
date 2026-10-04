"""受限场景 Tech R3 的**场景筛选口径**：近场背景类结构密度（只读人工标签）。

为什么需要它（T16 §10.3 裁决 B 的下一步）：身份率在现口径下被"像线的非漆结构"
占住分母（§18.5），而 R3 要选的正是**少这类结构**的路线。筛选判据必须
* 与模型无关（不能用"模型在哪些场景分高"来选场景——那是挑结果，不是挑场景）；
* 只读**人工标签**与真值线（标签里线类的像素位置），不依赖任何预测。

判据（逐帧，近场）：

```
band(row)  = [u_min - M, u_max + M]      # 该行线类像素的左右极值 ± M（M = 0.10W）
structure  = band 内 label == 0 的像素    # 标签说"背景"（墙/护栏/路缘/植被/阴影）
density    = Σ structure_px / Σ band_px   # 场景级池化（近场 rows >= 0.55H）
```

干净路段（线之间只有路面）density ≈ 0；城镇/路口/路侧有墙或护栏的帧会明显更高。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_scene_structure_density.py `
        --dir logs/experiments/review_pack_20260926/reviewed_full/pkg_town/front_main `
        --dir logs/experiments/t16_scenes_pairfar_20260930/frames `
        --out logs/experiments/t16_r3_structure_density.json

只读 npz；不训练、不启动游戏。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 近场行比例（从帧底往上算）：远处线细、结构也小，判据只看近场
NEAR_ROW_FRAC = 0.55
#: 线类像素左右极值之外还要算进来的余量（占帧宽比例）：路侧结构常紧贴车道外沿
BAND_MARGIN_FRAC = 0.10


def frame_structure_density(label, *, near_row_frac: float = NEAR_ROW_FRAC,
                            margin_frac: float = BAND_MARGIN_FRAC) -> dict:
    """单帧的近场背景类结构密度（纯函数，可单测）。

    返回 ``{band_px, structure_px, density, rows_with_line}``；没有线类像素的行
    不进判据（无线帧由负例通道管，不是本判据的对象），全帧无线时
    ``density=None``（不猜、不写 0）。
    """
    lab = np.asarray(label)
    if lab.ndim != 2 or lab.size == 0:
        raise ValueError("label must be a nonempty 2D array")
    h, w = lab.shape
    line = (lab == 2)
    row0 = int(h * (1.0 - float(near_row_frac)))
    m = int(round(float(margin_frac) * w))
    band_px = structure_px = rows = 0
    for y in range(row0, h):
        cols = np.nonzero(line[y])[0]
        if len(cols) == 0:
            continue
        u0 = max(0, int(cols.min()) - m)
        u1 = min(w, int(cols.max()) + m + 1)
        seg = lab[y, u0:u1]
        band_px += int(seg.size)
        structure_px += int(np.count_nonzero(seg == 0))
        rows += 1
    return {"band_px": band_px, "structure_px": structure_px,
            "rows_with_line": rows,
            "density": (None if band_px == 0
                        else round(structure_px / band_px, 4))}


def dir_structure_density(pattern: str) -> dict:
    """一个目录/通配下的场景级密度（逐帧池化）。"""
    files = sorted(glob.glob(str(pattern)))
    band = struct = frames = 0
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            lab = np.asarray(z["label"])
        except Exception:                                    # noqa: BLE001
            continue
        s = frame_structure_density(lab)
        band += s["band_px"]
        struct += s["structure_px"]
        frames += 1
    return {"pattern": str(pattern), "frames": frames, "band_px": band,
            "structure_px": struct,
            "density": (None if not band else round(struct / band, 4))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True,
                    help="帧目录或 glob（如 .../front_main 或 .../frames/*.npz）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for pat in args.dir:
        p = Path(pat)
        if p.is_dir():
            # 目录：先按本层 *.npz 找，找不到再按下一层 */**.npz 找（采集包是
            # <pack>/<view>/frame_*.npz，受控场景是 <batch>/frames/*.npz）
            subs = sorted({str(Path(f).parent) for f in
                           glob.glob(str(p / "**" / "*.npz"),
                                     recursive=True)})
            if subs:
                for sub in subs:
                    rows.append(dir_structure_density(
                        str(Path(sub) / "*.npz")))
            else:
                rows.append(dir_structure_density(str(p / "*.npz")))
        else:
            rows.append(dir_structure_density(pat))
    rows.sort(key=lambda r: (r["density"] is None, r["density"] or 0))
    print(f"{'density':>8} {'frames':>6} {'band_px':>9} {'struct_px':>9}  pattern")
    for r in rows:
        print(f"{str(r['density']):>8} {r['frames']:6d} {r['band_px']:9d} "
              f"{r['structure_px']:9d}  {r['pattern'][-70:]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[density] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
