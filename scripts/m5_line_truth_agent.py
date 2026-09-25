"""逐帧核对式标线标注（agent_revision）：机器提议 + 逐帧目视复核。

为什么不是"人工修订"：这些帧是**我看图画的**，不是人手画的。所以标注来源记
``agent_revision``（新的 rank：可训练、可测，但**晋级仍需人点头**），每个文件都
带逐帧的来历（建了多少、改了多少、留了多少"无法判断"）。不冒充 `human_revision`。

为什么纯规则不行（实测 2026-09-25）：RGB 里"亮 + 低饱和"既抓到白漆也抓到**混凝土
路缘石**——亮度 213 vs 200–208、饱和度 1.2 vs 1.3–3.4，颜色分不开。所以规则只做
**提议**，判据收窄到可以辩护的三条，剩下交给我逐帧看：

1. **引擎标线整块保留**（它与画面里最亮的那条线 100% 重合，逐帧目视确认过）；
2. **细长且不与"宽亮带"相交的亮成分**加进来（虚线中线就是这样被找回来的：
   短边 ≤ 24 px、细长比 ≥ 2.5）；
3. **宽亮带**（短边 > 24 px，实测是路缘石，不是漆线）判背景——这是**看出来的**判断，
   不是算法猜的；其余的碎亮斑**标"无法判断"**，不当背景（UNKNOWN ≠ 0）。

输出与 `m5_annotate_manual.py` 的训练契约一致：``colour`` / ``label`` /
``unknown_kind`` / 身份字段 + ``meta.json``（身份跟着走，否则审计会拒收——3h 轮
48 帧的教训）。每帧另有 ``provenance``：引擎像素数、新增像素数、被否掉的宽亮带像素数、
无法判断像素数、以及人工覆盖（``overrides``）用在哪一帧。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from beamng_autopilot.vision.segmentation import (  # noqa: E402
    fill_interior_holes, filter_line_shape,
)
from scripts.m5_annotate_manual import identity_npz_extras  # noqa: E402

#: 亮度/饱和度阈值（"亮 + 低饱和"= 可能是漆或混凝土）
BRIGHT = 185
SAT_MAX = 20
#: ^ 实测（2026-09-25）：饱和度才是"真漆线 vs 天际线雾"的判别特征——
#: 真漆线（白/灰）mx-mn = 1.2–12，远处置信雾/云边 = 30–32。原阈值 45 会把雾
#: 当成漆线（pillar_left 视角逐帧复核抓到：地平线上多了几条品红）。亮度分不开
#: （雾 188–190 vs 漆线 199–213 有重叠），形状也分不开（雾同样是细长条）。
#: "细"的判据：短边 ≤ 这个值（实测路缘石短边 84–95 px，漆线 22–65 px）
THIN_SHORT_SIDE = 24
#: 细长比下限（漆线实测 ≥ 3.0，路缘石 1.8–2.6）
ELONG_MIN = 2.5
#: 无法判断的碎亮斑：面积在 [MIN, MAX) 之间且不细长
UNKNOWN_SPECKS = (30, 400)
#: 路面上"亮"的容差：路面掩码外扩多少像素（漆线紧贴路面边缘）
ROAD_DILATE = 15


def propose(colour: np.ndarray, engine_line: np.ndarray, road: np.ndarray,
            *, add_rgb: bool = True) -> tuple:
    """``(line, unknown, stats)``：机器提议 + 每类的像素统计（可复核）。"""
    c = np.asarray(colour, np.int16)
    mx = c.max(-1)
    mn = c.min(-1)
    cand = (mx > BRIGHT) & ((mx - mn) < SAT_MAX)
    rd = np.asarray(road, bool)
    filled = fill_interior_holes(rd).astype(np.uint8)
    import cv2
    near_road = cv2.dilate(
        filled, cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (ROAD_DILATE, ROAD_DILATE))
    ).astype(bool)
    cand &= near_road

    eng = np.asarray(engine_line, bool)
    line = eng.copy()                       # 1) 引擎标线整块保留
    unknown = np.zeros_like(line)
    n_add = n_wide = n_speck = 0

    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), 8)
    for i in range(1, n_comp):
        _x, _y, w, h, area = stats[i]
        comp = labels == i
        short, long_ = min(w, h), max(w, h)
        elong = long_ / max(1, short)
        if short > THIN_SHORT_SIDE:
            # 2) 宽亮带 = 路缘石一类：判背景，但**记数**，让人能复核
            n_wide += int(area)
            continue
        if add_rgb and elong >= ELONG_MIN and comp.sum() >= 30:
            # 3) 细长亮条：虚线中线就是这样被找回来的
            add = comp & ~line
            n_add += int(add.sum())
            line |= comp
            continue
        if UNKNOWN_SPECKS[0] <= area < UNKNOWN_SPECKS[1]:
            unknown |= comp & ~line
            n_speck += int(area)
        else:
            n_wide += int(area)             # 太小或太大的非细长亮斑都判背景
    return line, unknown, {"engine_px": int(eng.sum()), "added_px": n_add,
                           "rejected_wide_px": n_wide,
                           "unknown_px": int(unknown.sum()),
                           "line_px": int(line.sum()),
                           "specks_unknown_px": n_speck}


def apply_overrides(line: np.ndarray, unknown: np.ndarray,
                    overrides: dict | None) -> tuple:
    """逐帧人工覆盖（我复核时改的地方）：``{"keep_rect": [...],
    "drop_rect": [...], "unknown_rect": [...]}``，矩形 ``(x0, y0, x1, y1)``。"""
    ov = overrides or {}
    for x0, y0, x1, y1 in ov.get("drop_rect", []):
        line[y0:y1, x0:x1] = False
    for x0, y0, x1, y1 in ov.get("keep_rect", []):
        line[y0:y1, x0:x1] = True
    for x0, y0, x1, y1 in ov.get("unknown_rect", []):
        unknown[y0:y1, x0:x1] = True
    return line, unknown


def build(cfg: dict, *, out: Path) -> dict:
    """把一批"采集帧 + 引擎标签"变成 agent_revision 标注目录。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames_meta = {}
    records = []
    total = {"engine_px": 0, "added_px": 0, "rejected_wide_px": 0,
             "unknown_px": 0, "line_px": 0, "specks_unknown_px": 0}
    add_views = set(cfg.get("views_add_rgb") or [])
    for src_dir, view in cfg["sources"]:
        src = Path(src_dir)
        meta_src = src.parent / "meta.json"
        meta = json.loads(meta_src.read_text(encoding="utf-8"))
        d = out / view
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                               indent=1), encoding="utf-8")
        frames_meta[view] = meta
        view_records: list = []
        for p in sorted(src.glob("frame_*.npz")):
            with np.load(p) as z:
                colour = np.asarray(z["colour"], np.uint8)
                lab = np.asarray(z["label"], np.uint8)
            line, unknown, st = propose(colour, lab == 2, lab == 1,
                                        add_rgb=(not add_views or view in add_views))
            ov = (cfg.get("overrides") or {}).get(f"{view}/{p.name}")
            line, unknown = apply_overrides(line, unknown, ov)
            label = np.zeros_like(lab)
            label[np.asarray(lab == 1, bool)] = 1        # 路面取引擎（密集可靠）
            label[line] = 2                              # 标线取"引擎∪细长亮条"
            uk = np.zeros_like(lab)
            uk[unknown] = 3                              # 3 = 无法判断
            # 未知区在 **label 里写 255(ignore)**：评估指标 (`known = lab != 255`)
            # 与训练损失都尊重它——"不知道"既不算假阳也不算漏检（UNKNOWN≠0）。
            label[unknown] = 255
            for k in total:
                total[k] += int(st.get(k, 0))
            rec_path = d / p.name
            payload = {"colour": colour, "label": label}
            if uk.any():
                payload["unknown_kind"] = uk
            # 身份：从采集 meta 的帧记录里按 basename 取
            pos = next((f.get("pos") for f in (meta.get("frames") or [])
                        if Path(str(f.get("path") or "")).name == p.name), None)
            hdg = next((f.get("heading") for f in (meta.get("frames") or [])
                        if Path(str(f.get("path") or "")).name == p.name), None)
            ident = {"map_name": meta.get("map_name"),
                     "source_id": meta.get("source_id"),
                     "pos": pos, "heading": hdg}
            np.savez_compressed(rec_path, **payload,
                                **identity_npz_extras(ident))
            records.append({"path": f"{view}/{p.name}", "view": view,
                            "identity": ident, "stats": st,
                            "overrides": bool(ov)})
            view_records.append(records[-1])
        # 凭证跟着帧走：每个视角目录写一份（方案 §6.1：来源资格要能逐目录查证；
        # 只写在根目录时，读取方按 self->parent 回退才能找到，容易读到采集 meta）
        (d / "annotation.json").write_text(json.dumps({
            "label_source": "agent_revision",
            "view": view,
            "generator": "scripts/m5_line_truth_agent.py",
            "source_dir": str(src),
            "frames": view_records}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    sidecar = {"label_source": "agent_revision",
               "note": ("机器提议 + 逐帧目视复核：引擎标线整块保留；细长亮条"
                        "（短边≤24、细长比≥2.5）补进来；宽亮带（路缘石一类）判背景；"
                        "其余碎亮斑标无法判断（UNKNOWN≠0）。这是**机器画的**，"
                        "不是人工修订——晋级仍需人确认。"),
               "views_add_rgb": sorted(add_views) or "全部",
               "config": {k: v for k, v in cfg.items() if k != "sources"},
               "totals": total, "frames": records}
    (out / "annotation.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
    return sidecar


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON：sources/overrides")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    side = build(cfg, out=Path(args.out))
    t = side["totals"]
    print(f"[agent-anno] {len(side['frames'])} 帧 -> {args.out}")
    print(f"[agent-anno] 引擎 {t['engine_px']} px | 新增细长条 {t['added_px']} px | "
          f"否掉宽亮带 {t['rejected_wide_px']} px | 无法判断 {t['unknown_px']} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
