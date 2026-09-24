"""固定探针图：同一组版本锁定的帧上，把真值/预测/误差画成可复核的图。

方案要求（T14 §"学习过程可视化"与"事件与图像协议"）：

* 每 N epoch 或每个 checkpoint 生成一次，**图像与标签的哈希固定**，
  所以两次结果可比、能回查；
* 一张图里同时给出 RGB、可信标签、预测掩码、FP/FN 误差图；
* **同一帧的四阶段后处理对照**（raw → morph_close → road_constraint →
  full_predict），阶段函数与 `scripts/m5_seg_stage_eval.py` **共用同一套实现**
  （`Segmenter._morph_close_line`、`constrain_line_to_road`、`filter_line_shape`、
  `Segmenter.predict`），并逐帧交叉检查 `full_predict` 与 `after_shape_filter`
  是否一致（不一致要报出来，而不是当作同一个东西）；
* **无真值区域灰显 UNKNOWN**，不能画成"预测正确"；
* 预测**不覆盖原图细节**（只描边界，不铺色块）；
* 产物放在 ``logs/experiments/<run_id>/probes/<checkpoint_id>/``，每个
  checkpoint 有清单记录：帧内容哈希、标签来源、权重哈希、后处理配置、
  生成时间；**每轮保存数量有上限**，且不复制原始数据进看板目录。

场景分类用**可测量的判据**（标线像素占比）而不是"看起来像"，并把判据与
数值写进清单，便于别人复核这一帧为什么被归到"退化线/无线"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.vision.segmentation import (  # noqa: E402
    constrain_line_to_road, filter_line_shape, strip_soil_from_road,
)

#: 后处理阶段顺序（与 m5_seg_stage_eval.STAGES 对齐；这里少一个"旧像素约束"
#: 的对照臂，那是调试臂不是生产路径）
POSTPROCESS_STAGES = ("raw_argmax", "after_morph_close",
                      "after_road_constraint", "after_shape_filter",
                      "full_predict")

CLS_LINE = 2
IGNORE = 255

#: 面板字体是否支持中文（写进清单：图能生成 ≠ 字能看见）
_FONT_OK = {"cjk": True}

#: 场景分档（按标线像素占比）：阈值写在这里并被清单引用，避免"凭感觉"
SCENE_BANDS = (
    ("none", 0.0, 0.0005, "无标线（标线像素占比 < 0.05%）"),
    ("faint", 0.0005, 0.01, "退化/稀疏标线（0.05%–1%）"),
    ("dense", 0.01, 1.0, "密集标线（>1%）"),
)


def scene_of(line_frac: float) -> tuple:
    """``(场景名, 判据说明)``：可测量、可复核。"""
    for name, lo, hi, why in SCENE_BANDS:
        if lo <= float(line_frac) < hi:
            return name, f"{why}，实测 line_frac={float(line_frac):.5f}"
    return "dense", f"密集标线，实测 line_frac={float(line_frac):.5f}"


def frame_sha16(colour: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(colour, np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def stage_masks(seg, colour: np.ndarray) -> tuple:
    """四阶段（含 full_predict）的掩码，与 `m5_seg_stage_eval` 同一条路径。

    返回 ``(stages, full_matches_shape_filter)``：后者是逐帧交叉检查——
    `Segmenter.predict()` 是部署链，`after_shape_filter` 是手工搭的同一条链；
    两者不一致说明有一条已经漂移，必须报出来而不是当成同一个东西。
    """
    road, line = seg._argmax_masks(seg._infer_logits(colour), colour)
    road = strip_soil_from_road(road, colour,
                               route_is_dirt=getattr(seg, "route_is_dirt",
                                                     False))
    stages = {"raw_argmax": np.asarray(line, dtype=bool)}
    stages["after_morph_close"] = np.asarray(
        seg._morph_close_line(stages["raw_argmax"]), dtype=bool)
    c = constrain_line_to_road(stages["after_morph_close"], road)
    stages["after_road_constraint"] = np.asarray(c, dtype=bool)
    stages["after_shape_filter"] = np.asarray(filter_line_shape(c), bool)
    _prod_road, prod_line = seg.predict(colour)
    stages["full_predict"] = np.asarray(prod_line, dtype=bool)
    ok = bool(np.array_equal(stages["full_predict"],
                             stages["after_shape_filter"]))
    return stages, ok


def stage_diffs(stages: dict) -> dict:
    """相邻阶段的像素差异，以及 raw→full 的总差异（都是像素计数）。"""
    order = list(POSTPROCESS_STAGES)
    out = {}
    for a, b in zip(order, order[1:]):
        if a in stages and b in stages:
            out[f"{a}->{b}"] = int(np.count_nonzero(stages[a] != stages[b]))
    if "raw_argmax" in stages and "full_predict" in stages:
        out["raw_argmax->full_predict"] = int(np.count_nonzero(
            stages["raw_argmax"] != stages["full_predict"]))
    return out


def frame_metrics(pred_line: np.ndarray, label: np.ndarray) -> dict:
    """单帧 line 像素统计（与评估矩阵同一套口径）。"""
    pred = np.asarray(pred_line).astype(bool)
    lab = np.asarray(label)
    known = lab != IGNORE
    gt = lab == CLS_LINE
    pr = pred & known
    tp = int((gt & pr).sum())
    fp = int((pr & ~gt).sum())
    fn = int((gt & ~pr & known).sum())
    return {
        "tp_px": tp, "fp_px": fp, "fn_px": fn,
        "offroad_false_line_px": int((pr & (lab == 0)).sum()),
        "unknown_px": int((~known).sum()),
        "pred_line_px": int(pred.sum()), "gt_line_px": int(gt.sum()),
        "precision": None if (tp + fp) == 0 else round(tp / (tp + fp), 4),
        "recall": None if (tp + fn) == 0 else round(tp / (tp + fn), 4),
        "iou": (None if (tp + fp + fn) == 0
                else round(tp / (tp + fp + fn), 4)),
    }


def panel(rgb: np.ndarray, label: np.ndarray, pred_line: np.ndarray,
          *, title: str, stages: dict | None = None, max_side: int = 420):
    """四联图：RGB+预测描边 / 误差图 / 可信标签 / 预测掩码。

    预测在 RGB 上**只描边界**（sobel 式梯度），不铺半透明色块——否则细节
    被盖住，图上看着"漂亮"但没法复核细线。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _FONT_OK["cjk"] = bool(_use_cjk_font(plt))

    rgb = np.asarray(rgb, np.uint8)
    lab = np.asarray(label)
    pred = np.asarray(pred_line).astype(bool)
    known = lab != IGNORE
    gt = lab == CLS_LINE
    tp = pred & gt
    fp = pred & ~gt & known
    fn = gt & ~pred & known
    unk = ~known

    edge = _edges(pred)
    overlay = rgb.copy()
    overlay[edge] = [255, 0, 0]                     # 预测：只描边
    overlay[tp & ~edge] = np.minimum(overlay[tp & ~edge], [0, 200, 0])

    err = np.zeros_like(rgb)
    err[...] = 255
    err[gt] = [170, 170, 170]                       # 真值底
    err[tp] = [40, 170, 70]
    err[fp] = [220, 40, 40]
    err[fn] = [240, 160, 20]
    err[unk] = [205, 205, 205]                      # UNKNOWN 灰显

    lab_vis = np.full((*lab.shape, 3), 255, np.uint8)
    lab_vis[gt] = [0, 90, 200]
    lab_vis[unk] = [205, 205, 205]

    pred_vis = np.full((*pred.shape, 3), 255, np.uint8)
    pred_vis[pred] = [200, 0, 0]

    nrow = 2 if stages else 1
    fig, grid = plt.subplots(nrow, 4, figsize=(13, 4.0 * nrow), dpi=100,
                             squeeze=False)
    # squeeze=False 永远给二维数组：单行时也必须摊平，否则把整行当成一个轴
    axes = [ax for row in grid for ax in row]
    zh = ((overlay, "RGB + 预测描边（红）"),
          (err, "误差图：绿=命中 红=路外假线 橙=漏检 灰=UNKNOWN"),
          (lab_vis, "可信标签（灰=UNKNOWN，不计入指标）"),
          (pred_vis, "预测掩码"))
    panels = zh if _FONT_OK["cjk"] else tuple(
        (img, t) for (img, _z), t in zip(zh, ASCII_TITLES))
    for ax, (img, sub) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(sub, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if stages:
        # 第二行：同一帧的四个后处理阶段，一眼看出哪一步吃掉了什么
        labels = (("raw_argmax", "raw argmax"),
                  ("after_morph_close", "after morph close"),
                  ("after_road_constraint", "after road constraint"),
                  ("after_shape_filter", "after shape filter (= full predict)"))
        for ax, (key, sub) in zip(axes[4:], labels):
            m = stages.get(key)
            if m is None:
                ax.set_title(f"{sub}（缺）", fontsize=8)
            else:
                vis = np.full((*np.asarray(m).shape, 3), 255, np.uint8)
                vis[np.asarray(m, dtype=bool)] = [200, 0, 0]
                ax.imshow(vis)
                ax.set_title(f"{sub}  px={int(np.count_nonzero(m))}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _use_cjk_font(plt) -> str:
    """标题里的中文必须能显示：本机实测 DejaVu Sans 缺 CJK 字形，会画成方框。

    选到就返回字体名；一个都没有时退回英文副标题（调用方用 ASCII_TITLES），
    并把这件事写进清单——不能让"图能生成"掩盖"字看不见"。
    """
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return ""


ASCII_TITLES = ("RGB + predicted outline (red)",
                "errors: green=TP red=offroad-FP orange=missed grey=UNKNOWN",
                "trusted label (grey=UNKNOWN, excluded from metrics)",
                "predicted mask")


def _edges(mask: np.ndarray) -> np.ndarray:
    """4 邻域梯度：True 表示该像素处于掩码边界。"""
    m = np.asarray(mask).astype(bool)
    e = np.zeros_like(m)
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    e[1:, :] |= m[:-1, :] != m[1:, :]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:, 1:] |= m[:, :-1] != m[:, 1:]
    return e & m


def select_frames(dirs: list, *, max_frames: int, per_scene: bool = True
                  ) -> list:
    """挑帧：按场景分档各取若干，保证探针集覆盖"无线/退化/密集"。"""
    pool = []
    for d in [Path(x) for x in dirs]:
        for f in sorted(d.glob("frame_*.npz")):
            z = np.load(f)
            colour = np.asarray(z["colour"], np.uint8)
            label = np.asarray(z["label"], np.uint8)
            frac = float((label == CLS_LINE).mean())
            name, why = scene_of(frac)
            pool.append({"path": str(f), "scene": name, "scene_why": why,
                         "line_frac": round(frac, 6), "colour": colour,
                         "label": label,
                         "dir": d.name,
                         "label_source": _label_source(Path(f).parent)})
    if not per_scene:
        return pool[:int(max_frames)]
    by_scene: dict = {}
    for item in pool:
        by_scene.setdefault(item["scene"], []).append(item)
    picked, seen = [], set()
    # 轮转各档：没有某一档就跳过（如实反映数据里没有这类场景）
    while len(picked) < int(max_frames):
        added = False
        for scene in [s[0] for s in SCENE_BANDS]:
            for item in by_scene.get(scene, []):
                if item["path"] in seen:
                    continue
                seen.add(item["path"])
                picked.append(item)
                added = True
                break
            if len(picked) >= int(max_frames):
                break
        if not added:
            break
    return picked[:int(max_frames)]


def _label_source(d: Path) -> str:
    meta = d / "meta.json"
    if not meta.exists() and (d.parent / "meta.json").exists():
        meta = d.parent / "meta.json"
    if meta.exists():
        try:
            return str(json.loads(meta.read_text(encoding="utf-8")).get(
                "label_source") or "")[:60]
        except Exception:                    # noqa: BLE001
            return ""
    return ""


def checkpoint_id(model_path: Path) -> str:
    h = hashlib.sha256()
    with open(model_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{model_path.stem}_{h.hexdigest()[:12]}"


def _predict(seg, colour: np.ndarray) -> np.ndarray:
    _road, line, _probs = seg.predict_with_probs(colour)
    return np.asarray(line) > 0


def build(run_dir: Path, model_path: Path, *, runs: list, max_frames: int = 8,
          seg=None, stages: bool = True) -> dict:
    """生成一个 checkpoint 的探针目录 + 清单；返回清单 dict。"""
    out = Path(run_dir) / "probes" / checkpoint_id(model_path)
    out.mkdir(parents=True, exist_ok=True)
    # 同一 checkpoint 目录可能被再次写入（换了 --runs/--max-frames）。旧图
    # 若不清理，目录里就会出现"清单没记录的图"，复核时会误以为它是本轮的
    # 证据；这里只删本目录的探针图，不动别的东西。
    stale = sorted(out.glob("probe_*.png"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"[probes] 清理上一轮遗留的 {len(stale)} 张旧探针图")
    if seg is None:
        from beamng_autopilot.vision.segmentation import Segmenter
        seg = Segmenter(model_path=str(model_path))
    frames = select_frames(runs, max_frames=max_frames)
    entries = []
    for i, item in enumerate(frames):
        colour, label = item["colour"], item["label"]
        stage_masks_d = None
        full_ok = None
        if stages:
            stage_masks_d, full_ok = stage_masks(seg, colour)
        pred = (stage_masks_d["full_predict"] if stage_masks_d
                else _predict(seg, colour))
        metres = frame_metrics(pred, label)
        stage_metres = ({name: frame_metrics(mask, label)
                         for name, mask in stage_masks_d.items()}
                        if stage_masks_d else {})
        diffs = stage_diffs(stage_masks_d) if stage_masks_d else {}
        fig = panel(colour, label, pred, stages=stage_masks_d,
                    title=(f"{Path(item['path']).parent.name}/"
                           f"{Path(item['path']).name} · 场景={item['scene']}"
                           f" · IoU={metres['iou']}"
                           + (f" · 阶段 IoU: "
                              f"raw={stage_metres['raw_argmax']['iou']}"
                              f" → full={stage_metres['full_predict']['iou']}"
                              if stage_metres else "")))
        png = out / f"probe_{i:02d}_{item['scene']}.png"
        fig.savefig(str(png), bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        entries.append({
            "probe": png.name,
            "scene": item["scene"], "scene_why": item["scene_why"],
            "frame": Path(item["path"]).name,
            "frame_sha16": frame_sha16(colour),
            "label_sha16": hashlib.sha256(
                np.ascontiguousarray(label).tobytes()).hexdigest()[:16],
            "label_source": item["label_source"],
            "line_frac": item["line_frac"],
            "metrics": metres,
            "stages": {name: m for name, m in stage_metres.items()},
            "stage_diffs": diffs,
            "full_predict_matches_shape_filter": full_ok,
            **({} if stages else {"stages_skipped": "build(stages=False)"}),
        })
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(Path(run_dir)),
        "checkpoint_id": checkpoint_id(model_path),
        "model": str(model_path),
        "model_sha16": checkpoint_id(model_path).split("_")[-1],
        "postprocess": {"line_source": "Segmenter.predict (deployed path)",
                        "resolution": "as stored in the frames",
                        "smoothing": "temporal_smooth disabled (single frame)",
                        "stages_compared": list(POSTPROCESS_STAGES),
                        "shared_with": "scripts/m5_seg_stage_eval.py "
                                       "(same stage functions)"},
        "scene_bands": [{"name": n, "lo": lo, "hi": hi, "why": w}
                        for n, lo, hi, w in SCENE_BANDS],
        "max_frames": int(max_frames),
        "cjk_font": _FONT_OK["cjk"],
        "n_frames": len(entries),
        "frames": entries,
        "note": ("探针帧由内容哈希锁定：换帧或换权重都会产生新的 checkpoint_id 目录，"
                 "两次结果因此可比。原始数据不复制到这里。"
                 "同一帧额外给出四阶段后处理对照（与 m5_seg_stage_eval 共用实现），"
                 "每帧带 full_predict 与 after_shape_filter 的一致性交叉检查。"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="固定探针图：同帧的 RGB/真值/预测/FP-FN，带哈希清单")
    ap.add_argument("--model", required=True, help="checkpoint（best.pt）")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="帧目录（每场景各取若干帧）")
    ap.add_argument("--run-id", default="probes",
                    help="日志目录名（logs/experiments/<run_id>/probes/...）")
    ap.add_argument("--max-frames", type=int, default=8,
                    help="每个 checkpoint 最多保存多少帧（方案要求有上限）")
    ap.add_argument("--no-stages", action="store_true",
                    help="跳过同帧四阶段后处理对照（只出 RGB/误差/标签/预测四联）")
    ap.add_argument("--json", default=None, help="清单输出路径（默认在探针目录内）")
    args = ap.parse_args(argv)

    run_dir = Path(config.LOGS_DIR) / "experiments" / args.run_id
    manifest = build(run_dir, Path(args.model), runs=args.runs,
                     max_frames=args.max_frames, stages=not args.no_stages)
    print(f"[probes] {manifest['n_frames']} 帧 -> "
          f"{run_dir / 'probes' / manifest['checkpoint_id']}")
    for e in manifest["frames"]:
        print(f"  {e['scene']:5s} {e['frame']:22s} "
              f"IoU={e['metrics']['iou']} 路外假线={e['metrics']['offroad_false_line_px']} "
              f"({e['scene_why']})")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        print(f"[probes] 清单 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
