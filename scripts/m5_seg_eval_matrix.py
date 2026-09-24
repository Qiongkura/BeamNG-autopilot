"""像素层 + 性能层的评估矩阵：同一批冻结输入上对比多个 checkpoint。

T14 阶段 4 的评估层入口。方案要求"每个 checkpoint 都在同一批冻结输入上与
当前生产模型比较"，并**分开报**：line precision / recall / IoU、漏真线、
路外假线（预测标线像素落在 ``label==0``）与推理延迟 p50/p95。

为什么不是"看 IoU 就够了"：
* 只有 IoU 会把"画得多但错得多"和"画得少但准"混成一个数；
* 路外假线是计划里单列的一项（方向判据/线通道的假阳性会直接体现成它）；
* 延迟必须报 p50/p95，而不是平均——deadline 看的是尾部。

指标数学放在 :func:`line_pixel_metrics` 里，是不依赖模型的纯函数，测试直接
拿合成掩码验证；CLI 只负责遍历 checkpoint 与汇总。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.segmentation import Segmenter  # noqa: E402

#: 与 vision.segmentation 的训练契约一致
CLS_LINE = 2
IGNORE = 255


def line_pixel_metrics(pred_line: np.ndarray, label: np.ndarray) -> dict:
    """一帧的 line 像素统计。纯函数，无模型、无 I/O。

    ``offroad_false_line_px`` = 预测为标线但真值类别为背景(0) 的像素数，
    即"画到路面外/非铺装上的线"；``missed_true_line_px`` = 真值标线但没预测到的像素
    （假负例）。两者是方案在像素层点名要分开报的量。
    """
    pred = np.asarray(pred_line).astype(bool)
    lab = np.asarray(label)
    if pred.shape != lab.shape:
        raise ValueError(f"mask/label shape mismatch: {pred.shape} vs "
                         f"{lab.shape}")
    known = lab != IGNORE
    gt = lab == CLS_LINE
    pr = pred & known
    tp = int((gt & pr).sum())
    fp = int((pr & ~gt).sum())
    fn = int((gt & ~pr & known).sum())
    return {
        "tp_px": tp, "fp_px": fp, "fn_px": fn,
        "missed_true_line_px": fn,
        "offroad_false_line_px": int((pr & (lab == 0)).sum()),
        "pred_line_px": int(pred.sum()),
        "gt_line_px": int(gt.sum()),
        "known_px": int(known.sum()),
    }


def road_pixel_metrics(pred_road: np.ndarray, label: np.ndarray) -> dict:
    """一帧的**路面类**像素统计（纯函数）。

    为什么要有它：line 通道被整通道屏蔽的实验（缺可信漆线真值）里，标线指标
    没有任何真值，唯一可测的通道是路面。没有这个数，那种实验就只剩"未测"，
    无法回答"扩充场景数据后路面是否变好"。
    """
    pred = np.asarray(pred_road).astype(bool)
    lab = np.asarray(label)
    if pred.shape != lab.shape:
        raise ValueError(f"mask/label shape mismatch: {pred.shape} vs "
                         f"{lab.shape}")
    known = lab != IGNORE
    gt = lab == 1                       # 1 = road（见 curve_schema）
    pr = pred & known
    tp = int((gt & pr).sum())
    fp = int((pr & ~gt).sum())
    fn = int((gt & ~pr & known).sum())
    return {"r_tp_px": tp, "r_fp_px": fp, "r_fn_px": fn,
            "pred_road_px": int(pred.sum()), "gt_road_px": int(gt.sum()),
            # 名字必须与 line 通道的 known_px 区分：accumulate 是按 key 相加的，
            # 同名会把两个通道的"已知像素"叠加（实测把 0.4025 算成 0.2012）
            "r_known_px": int(known.sum())}


def worst_frames(per_frame: list, *, k: int = 5) -> dict:
    """按 road_iou 排序给出最差/最好的若干帧（纯函数，供"点进具体帧"用）。

    分母为 0 的帧（该帧没有路面真值）单独计数，不参与排序——它们不是"差"，
    是"没得比"。
    """
    got = [f for f in (per_frame or []) if f.get("iou") is not None]
    none_n = len([f for f in (per_frame or []) if f.get("iou") is None])
    got.sort(key=lambda f: float(f["iou"]))
    return {"n": len(got), "n_no_denominator": none_n,
            "worst": [{"frame": str(f.get("frame")), "iou": float(f["iou"]),
                       "gt_px": int(f.get("gt_px") or 0)}
                      for f in got[:int(k)]],
            "best": [{"frame": str(f.get("frame")), "iou": float(f["iou"]),
                      "gt_px": int(f.get("gt_px") or 0)}
                     for f in got[-int(k):][::-1]]}


def accumulate(acc: dict, frame: dict) -> None:
    """把一帧的统计累加进总计（全局累加，不做逐帧平均）。"""
    for k, v in frame.items():
        acc[k] = acc.get(k, 0) + int(v)


def totals_to_metrics(acc: dict, *, n_frames: int, ms: list) -> dict:
    """总计 -> 指标。分母为 0 时返回 None，绝不返回 0 冒充"测过了"。"""
    tp, fp, fn = acc.get("tp_px", 0), acc.get("fp_px", 0), acc.get("fn_px", 0)
    pred, gt = acc.get("pred_line_px", 0), acc.get("gt_line_px", 0)
    rtp = acc.get("r_tp_px", 0)
    rfp = acc.get("r_fp_px", 0)
    rfn = acc.get("r_fn_px", 0)
    a = np.asarray(ms, dtype=float) if ms else np.asarray([], dtype=float)
    out = {
        "n_frames": int(n_frames),
        "tp_px": tp, "fp_px": fp, "fn_px": fn,
        "missed_true_line_px": acc.get("missed_true_line_px", 0),
        "offroad_false_line_px": acc.get("offroad_false_line_px", 0),
        "pred_line_px": pred, "gt_line_px": gt,
        "known_px": acc.get("known_px", 0),
        "line_precision": (None if (tp + fp) == 0
                           else round(tp / (tp + fp), 4)),
        "line_recall": None if (tp + fn) == 0 else round(tp / (tp + fn), 4),
        "line_iou": (None if (tp + fp + fn) == 0
                     else round(tp / (tp + fp + fn), 4)),
        # 路面类（road-only 实验的唯一可测通道；分母为 0 时仍返回 None）
        "road_precision": (None if (rtp + rfp) == 0
                           else round(rtp / (rtp + rfp), 4)),
        "road_recall": (None if (rtp + rfn) == 0
                        else round(rtp / (rtp + rfn), 4)),
        "road_iou": (None if (rtp + rfp + rfn) == 0
                     else round(rtp / (rtp + rfp + rfn), 4)),
        "road_tp_px": rtp, "road_fp_px": rfp, "road_fn_px": rfn,
        "gt_road_px": acc.get("gt_road_px", 0),
        # 平凡基线参照：这一类指标在类别不平衡下会"白送"一个高分——
        # 把**全部**像素预测成路面就能拿到 路面像素/已知像素 的 IoU
        # （实测：开发集 0.4025，而"看起来稳定"的模型读数是 0.4239）。
        # 不给出这个参照，0.42 会被读成"模型学会了路面"。
        "road_iou_trivial_all_road": (None if not acc.get("r_known_px")
                                      else round(acc.get("gt_road_px", 0)
                                                 / acc["r_known_px"], 5)),
        "road_iou_trivial_all_background": (None if not acc.get("gt_road_px")
                                            else 0.0),
        "offroad_false_frac_of_pred": (None if pred == 0
                                       else round(acc.get(
                                           "offroad_false_line_px", 0) / pred,
                                           4)),
        "inference_ms_p50": (None if a.size == 0
                             else round(float(np.percentile(a, 50)), 2)),
        "inference_ms_p95": (None if a.size == 0
                             else round(float(np.percentile(a, 95)), 2)),
        "inference_ms_mean": (None if a.size == 0
                              else round(float(a.mean()), 2)),
    }
    if out["line_precision"] is None:
        out["line_precision_missing"] = "no predicted line pixels at all"
    if out["line_recall"] is None:
        out["line_recall_missing"] = "no true line pixels in this set"
    return out


def load_frames(dirs: list) -> list:
    """读 npz 帧：``[(repo 相对路径, colour, label)]``，按文件名排序。"""
    frames = []
    for d in dirs:
        for f in sorted(Path(d).glob("frame_*.npz")):
            z = np.load(f)
            try:
                rel = str(Path(f).resolve().relative_to(ROOT))
            except ValueError:
                rel = str(f)
            frames.append((rel, np.asarray(z["colour"], np.uint8),
                           np.asarray(z["label"], np.uint8)))
    return frames


def evaluate_model(model_path: Path, frames: list, *, device: str = "cuda"
                   ) -> dict:
    """在一个 checkpoint 上跑完整评估（学习掩码 + 后处理，走 Segmenter）。"""
    from beamng_autopilot.experiments.checkpoint import file_sha16
    seg = Segmenter(model_path=str(model_path))
    acc: dict = {}
    ms: list = []
    per_frame: list = []
    for _name, colour, label in frames:
        road, line, _probs = seg.predict_with_probs(colour)
        ms.append(float((seg.last_timing_ms or {}).get("total") or 0.0))
        accumulate(acc, line_pixel_metrics(np.asarray(line) > 0, label))
        rm = road_pixel_metrics(np.asarray(road) > 0, label)
        accumulate(acc, rm)
        denom = rm["r_tp_px"] + rm["r_fp_px"] + rm["r_fn_px"]
        per_frame.append({"frame": str(_name),
                          "iou": (None if denom == 0
                                  else rm["r_tp_px"] / denom),
                          "gt_px": rm["gt_road_px"]})
    out = totals_to_metrics(acc, n_frames=len(frames), ms=ms)
    out["mask_compare"] = worst_frames(per_frame)
    out["model"] = str(model_path)
    out["sha256_16"] = file_sha16(model_path)
    return out


def parse_model_arg(spec: str) -> tuple:
    """``name=path`` 或裸路径；名字缺省用文件名。"""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    p = Path(spec)
    return p.stem, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="多 checkpoint 的像素/性能评估矩阵")
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH",
                    help="可重复；每个 checkpoint 一行（带 sha256 前缀）")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="主评估目录（冻结集；方案要求只用一次）")
    ap.add_argument("--dev-runs", nargs="*", default=[],
                    help="开发诊断集目录（可选，与主集分开报告）")
    ap.add_argument("--device", default=None, choices=("cuda", "cpu"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                    # noqa: BLE001
            device = "cpu"

    frozen = load_frames(args.runs)
    dev = load_frames(args.dev_runs) if args.dev_runs else []
    print(f"[eval-matrix] frozen {len(frozen)} frames, dev {len(dev)} frames, "
          f"device={device}", flush=True)
    out = {"frozen": {}, "dev": {},
           "frozen_runs": [str(r) for r in args.runs],
           "dev_runs": [str(r) for r in args.dev_runs]}
    for spec in args.model:
        name, path = parse_model_arg(spec)
        if not path.exists():
            print(f"[eval-matrix] MISSING {path}", flush=True)
            out["frozen"][name] = {"error": f"missing checkpoint: {path}"}
            continue
        try:
            out["frozen"][name] = evaluate_model(path, frozen, device=device)
            if dev:
                out["dev"][name] = evaluate_model(path, dev, device=device)
        except Exception as exc:             # noqa: BLE001
            out["frozen"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        f = out["frozen"][name]
        d = out["dev"].get(name) or {}
        print(f"[eval-matrix] {name:26s} sha={f.get('sha256_16')} "
              f"P={f.get('line_precision')} R={f.get('line_recall')} "
              f"IoU={f.get('line_iou')} "
              f"offroadFP={f.get('offroad_false_line_px')} "
              f"p50={f.get('inference_ms_p50')}ms p95={f.get('inference_ms_p95')}"
              f"ms | dev IoU={d.get('line_iou')}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[eval-matrix] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
