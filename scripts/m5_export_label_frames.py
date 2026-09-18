"""从 shadow episode 挑出**最需要人工标注**的帧，打包成标注任务。

对应这条要求：*有铺装路面时，土肩不得算作道路*。随机抽帧会浪费标注人力，
所以按下面三类挑选（每类都打印命中理由）：

1. **掩码里含土/砾石**（`soil & road` 像素多）—— 模型把路肩当路的地方，
   标注时把路肩涂成背景；
2. **几乎看不到标线**（line 掩码很小）—— 右侧边界最吃紧、最需要铺装真值的
   路段；
3. **正常铺装帧**（无土、标线充足）—— 负样本，避免模型学会"把地面全判成
   背景"。

输出 ``logs/m5_seg/<name>_pkg/frame_XXXXX.npz``（``colour`` + 空 ``label``）
与同名 PNG 预览，另写一份 ``RULES.txt``（标注规则随包走）。

    .venv\\Scripts\\python.exe scripts\\m5_export_label_frames.py \\
        --episode logs/m5_e2e/shadow_fsd_...npz --name paved_shoulder
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from beamng_autopilot import config
from beamng_autopilot.vision.segmentation import (
    Segmenter, snow_or_soil_mask,
)

RULES = """\
标注规则（对应 AGENTS.md「驾驶约束」第 2/3 条）
------------------------------------------------
1  = line   漆画线（白色/黄色实虚线本体）
2  = road   可行驶的**铺装**路面（沥青 / 水泥）
3  = erase  背景：**土肩 / 砾石路肩 / 路缘外的地面**、植被、天空、车体

要点
* 铺装路两侧的土肩、砂石带、草地 → 一律 3（这正是本次要修的错误）。
* 如果这一帧**整条路都是土路/砂石路**（画面里没有铺装）→ 那条土路本身涂 2。
* 路缘石（kerb）本身属于路的一部分，涂 2；它外侧的地面涂 3。
* 不确定的窄带（1~2 像素）可以留空，不要纠结。

工具
* 画笔左键涂；右键/`f` 油漆桶填充；`1/2/3` 切换类别；`u` 撤销；`s` 保存并下一张。
"""


def _soil_road_px(rgb: np.ndarray, road: np.ndarray) -> int:
    soil = snow_or_soil_mask(rgb)
    return int((soil & road).sum())


def pick_frames(ep: Path, want_soil: int, want_noline: int,
                want_plain: int, min_gap: int = 3):
    """Return [(index, rgb, reason)] with the most useful frames first."""
    d = np.load(ep, allow_pickle=True)
    rgbs, ts = d["rgb"], d["t"]
    seg = Segmenter()
    soil_rows, line_rows, plain_rows = [], [], []
    for i in range(len(ts)):
        rgb = np.asarray(rgbs[i], dtype=np.uint8)
        road, line = seg.predict(rgb)
        s = _soil_road_px(rgb, road)
        lp = int(line.sum())
        soil_rows.append((s, i))
        line_rows.append((lp, i))
        if s == 0 and lp > 500:
            plain_rows.append(i)
    picked: list[tuple[int, str]] = []
    for rows, want, why, reverse in (
            (soil_rows, want_soil, "掩码含土肩", True),
            (line_rows, want_noline, "几乎无标线", False)):
        rows = sorted(rows, key=lambda r: r[0], reverse=reverse)
        kept = 0
        for _score, i in rows:
            if kept >= want:
                break
            if any(abs(i - j) < min_gap for j, _w in picked):
                continue
            picked.append((i, why))
            kept += 1
    if plain_rows:
        idxs = np.linspace(0, len(plain_rows) - 1, want_plain).astype(int)
        for k in idxs:
            i = plain_rows[k]
            if any(abs(i - j) < min_gap for j, _w in picked):
                continue
            picked.append((i, "正常铺装(负样本)"))
    picked.sort()
    return [(i, np.asarray(rgbs[i], dtype=np.uint8), w) for i, w in picked]


def main() -> int:
    ap = argparse.ArgumentParser(description="导出待标注帧（土肩/无标线优先）")
    ap.add_argument("--episode", required=True,
                    help="shadow episode npz，或 latest")
    ap.add_argument("--name", default="paved_shoulder",
                    help="输出包名 logs/m5_seg/<name>_pkg")
    ap.add_argument("--soil", type=int, default=18,
                    help="掩码含土肩的帧数上限")
    ap.add_argument("--no-line", dest="noline", type=int, default=12,
                    help="几乎无标线的帧数上限")
    ap.add_argument("--plain", type=int, default=8,
                    help="正常铺装（负样本）帧数上限")
    args = ap.parse_args()

    ep = Path(args.episode)
    if str(args.episode) == "latest":
        eps = sorted(glob.glob(str(config.LOGS_DIR / "m5_e2e"
                                   / "shadow_fsd_*.npz")),
                     key=os.path.getmtime)
        ep = Path(eps[-1])
    out = config.LOGS_DIR / "m5_seg" / f"{args.name}_pkg"
    out.mkdir(parents=True, exist_ok=True)

    frames = pick_frames(ep, args.soil, args.noline, args.plain)
    print(f"episode: {ep.name}")
    print(f"选中 {len(frames)} 帧 -> {out}")
    for k, (i, rgb, why) in enumerate(frames, start=1):
        fp = out / f"frame_{k:05d}.npz"
        np.savez_compressed(str(fp), colour=rgb,
                            label=np.zeros(rgb.shape[:2], np.uint8))
        cv2.imwrite(str(out / f"preview_{k:05d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  frame_{k:05d}  ep#{i:03d}  {why}")
    (out / "RULES.txt").write_text(RULES, encoding="utf-8")
    print("\n" + RULES)
    print("标注命令：\n  .venv\\Scripts\\python.exe "
          "scripts\\m5_annotate_manual.py "
          f"--frames-dir {out.as_posix()} "
          f"--out {out.as_posix().replace('_pkg', '_labeled')} "
          "--prefill-model logs/m5_seg/seg_model/best.pt")
    print("\n标注完成后训练：\n  .venv\\Scripts\\python.exe "
          "scripts\\m5_train_seg.py --runs "
          f"{out.as_posix().replace('_pkg', '_labeled')} "
          "--init logs/m5_seg/seg_model/best.pt --epochs 12 --balance-runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
