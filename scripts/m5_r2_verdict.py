"""R2 判定文件生成器：可靠参考上的验收（方案 Order 5 签收行）。

为什么要有它：R2 的判定必须**可复现、带证据、按协议版本分开存**（方案纪律：
判定文件按版本分开、不互相覆盖；旧判定不可比）。此前 R2 的数字散在各次 arm gate /
eval-matrix 的运行里，靠人手汇总——本脚本把"集合清单 + 逐 seed 五门 + 协议版本 +
定义 + 分歧率"一次产出，写成一个 JSON。

口径（预注册，`docs/T16_R2_PATH_AND_BRANCH_PLAN_20261005.md`）：

* 门（**不放宽**）：覆盖率 ≥0.80、身份率 ≥0.60、精度 ≥0.40、召回 ≥0.70、
  角色 ≥0.70；缺测一律 UNKNOWN（不写 0）；
* 集合：dev 用**去重后的 9 个场景**（重复目录会双计难场景、足以翻转召回门）；
* **定义**：v7 = 线是人工标签（召回按标签）；v8 = 线是"铺装上的漆"，
  除标签范围外**另报漆范围召回**（真值 = 标签 ∩ 像漆）与**分歧率**
  （标签线像素里不像漆的占比）——两边都不假装另一边的标签是对的。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_r2_verdict.py `
        --arm s42=logs/experiments/t16_negdose6x_20261001/round0/seed42/checkpoint_last.pt `
        --arm s43=... --dev-runs <去重 9 场景> `
        --out logs/experiments/r2_verdict_v7_20261005.json

不训练、不启动游戏；只读 checkpoint 与帧。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

#: 冻结门（**不得放宽**；改动即协议新版本）
GATES = {"candidate_reference_coverage": 0.80,
         "candidate_identity_rate": 0.60,
         "line_precision": 0.40,
         "line_recall": 0.70,
         "left_right_role_agreement": 0.70}
#: 逐场景候选数下限（在 R 上）
PER_SCENE_MIN_CANDIDATES = 30


def evaluate_gates(by_seed: dict, *, mode: str = "strict",
                   recall_scope: str = "label") -> dict:
    """逐 seed 判定 + 均值判定（纯函数，可单测）。

    ``by_seed``：``{seed: {metric: value|None}}``。缺测（None）记 UNKNOWN，
    **不算通过也不算失败**（方案纪律：缺测不当 PASS）。返回
    ``{"per_seed": {...}, "means": {...}, "gates": {...}, "verdict": ...}``。

    ``recall_scope``（召回门的真值范围，与"线"的定义配套）：
    * ``"label"``（默认）：召回按**人工标签**（v7 定义）；
    * ``"paint"``：召回按**标签 ∩ 像漆**（v8 定义——它只承诺召回漆线；
      用标签范围去判 v8 是把另一套定义的分母塞进来）。
    两种范围都在返回里可见（``line_recall`` 与 ``line_recall_paint_scope``），
    判门只看选中的那个。

    ``mode``（判定语义，方案层待裁决项②）：
    * ``"strict"``（默认）：**全 seed 通过**才算该门通过（任一 seed 跌破即 fail）；
    * ``"mean"``：按**池化均值**判门（方案原文"逐 seed 判定 + 配对/均值双报"的
      "均值"读法）；逐 seed 分布仍完整报出（``n_pass/n_fail``），未过的 seed 需要
      另附归因（人工/文档层，脚本不代替）。
    两种模式下缺测都记 UNKNOWN。
    """
    if mode not in ("strict", "mean"):
        raise ValueError(f"未知判定语义 {mode!r}（strict|mean）")
    if recall_scope not in ("label", "paint"):
        raise ValueError(f"未知召回范围 {recall_scope!r}（label|paint）")
    # 召回门按选定范围取值（其余门不变）；原始两列都保留在 per_seed 里
    gate_field = {"label": "line_recall",
                  "paint": "line_recall_paint_scope"}[recall_scope]
    # 判门用 judge（召回列按选定范围替换），per_seed 仍保留**原始两列**
    judge = {s_: dict(v) for s_, v in by_seed.items()}
    if recall_scope == "paint":
        for v in judge.values():
            v["line_recall"] = v.get("line_recall_paint_scope")
    seeds = sorted(by_seed)
    out_gates = {}
    means = {}
    for metric, thr in GATES.items():
        vals = [judge[s].get(metric) for s in seeds]
        measured = [float(v) for v in vals if v is not None]
        means[metric] = (round(float(np.mean(measured)), 4)
                         if measured else None)
        per = {}
        for s, v in zip(seeds, vals):
            if v is None:
                per[str(s)] = "UNKNOWN"
            else:
                per[str(s)] = "pass" if float(v) >= thr else "fail"
        n_pass = sum(1 for x in per.values() if x == "pass")
        n_fail = sum(1 for x in per.values() if x == "fail")
        n_unk = sum(1 for x in per.values() if x == "UNKNOWN")
        if mode == "strict":
            verdict = ("pass" if n_fail == 0 and n_unk == 0 and n_pass
                       else "fail" if n_fail
                       else "unknown" if n_unk and not n_pass
                       else "partial")
        else:                      # mean：按池化均值判门（逐 seed 分布仍报出）
            verdict = ("pass" if (means[metric] is not None
                                  and means[metric] >= thr)
                       else "fail" if means[metric] is not None
                       else "unknown")
        out_gates[metric] = {
            "threshold": thr, "mean": means[metric], "per_seed": per,
            "n_pass": n_pass, "n_fail": n_fail, "n_unknown": n_unk,
            "mode": mode, "verdict": verdict}
    failing = [m for m, g in out_gates.items() if g["verdict"] == "fail"]
    unknown = [m for m, g in out_gates.items() if g["verdict"] == "unknown"]
    partial = [m for m, g in out_gates.items() if g["verdict"] == "partial"]
    return {"mode": mode, "recall_scope": recall_scope,
            "recall_gate_field": gate_field,
            "per_seed": {str(s): dict(by_seed[s]) for s in seeds},
            "means": means, "gates": out_gates,
            "failing_gates": failing, "unknown_gates": unknown,
            "partial_gates": partial,
            "verdict": ("pass" if not failing and not unknown and not partial
                        else "fail" if failing else "incomplete")}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=CHECKPOINT")
    ap.add_argument("--dev-runs", nargs="+", required=True)
    ap.add_argument("--r3-runs", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recall-scope", choices=("label", "paint"),
                    default="label",
                    help="召回门的真值范围：label=人工标签（v7 定义，默认）；"
                         "paint=标签∩像漆（v8 定义）")
    ap.add_argument("--verdict-mode", choices=("strict", "mean"),
                    default="strict",
                    help="判定语义：strict=全 seed 必须过（默认）；"
                         "mean=按池化均值判门（逐 seed 分布仍完整报出）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from beamng_autopilot.experiments.protocol import active_protocol_version
    from beamng_autopilot.vision import line_scope

    gate_mod = _load(Path("m5_arm_gate_measure.py"), "m5_arm_gate_measure_r2")
    em = _load(Path("m5_seg_eval_matrix.py"), "m5_seg_eval_matrix_r2")
    ip = gate_mod._probe()

    dev_dirs = [Path(p) for p in args.dev_runs if Path(p).is_dir()]
    if not dev_dirs:
        raise SystemExit(f"dev 集合为空：{args.dev_runs}")

    def run_set(dirs):
        cand, pixel, pr = {}, {}, {}
        for spec in args.arm:
            name, _, ck = str(spec).partition("=")
            rows = []
            for d in dirs:
                mp = d / "meta.json"
                if not mp.is_file() and (d.parent / "meta.json").is_file():
                    mp = d.parent / "meta.json"
                meta = (json.loads(mp.read_text(encoding="utf-8"))
                        if mp.is_file() else None)
                res = ip.probe(d, meta, view=d.name,
                               model_path=str(ROOT / ck), device=args.device)
                rows.extend(res.get("rows") or [])
            cand[name] = gate_mod.summarize_rows(rows)
            # 分组装载复用评估矩阵自己的实现（map/source_id 分组，同组多目录合并）
            by = em.frames_by_group([str(d) for d in dirs])
            m = em.evaluate_model_per_group(Path(ROOT / ck), by,
                                            device=args.device)
            pixel[name] = {k: m.get(k) for k in
                           ("line_precision", "line_recall", "line_iou",
                            "n_frames")}
            pixel[name]["per_group"] = m.get("per_group")
            pr[name] = paint_scope_recall(Path(ROOT / ck), dirs, ip, args.device)
        return cand, pixel, pr

    dev_cand, dev_pixel, dev_pr = run_set(dev_dirs)
    # R3（受限认证类）：方案 §5 要求 R3 结论与 R2 并列记录；--r3-runs 给了就必须
    # 真跑（不给就明说没测，绝不静默忽略——本项目已被静默 no-op 坑过）
    r3_dirs = [Path(p) for p in (args.r3_runs or []) if Path(p).is_dir()]
    r3 = None
    if r3_dirs:
        rc, rp, rr = run_set(r3_dirs)
        r3 = {"sets": {"dirs": [str(d) for d in r3_dirs]},
              "candidate": rc, "pixel": rp, "paint_recall": rr}
    elif args.r3_runs:
        raise SystemExit(f"--r3-runs 给了但没有有效目录：{args.r3_runs}")
    out = {"protocol": active_protocol_version(),
           "definition": ("v8: line = paint on pavement within lat scope"
                          if line_scope.protocol() == "v8"
                          else "v7: line = human label"),
           "lat_max_m": line_scope.lat_max_m(),
           "merge_final": line_scope.merge_final_enabled(),
           "appearance_gate": line_scope.appearance_gate_enabled(),
           "sets": {"dev": {"dirs": [str(d) for d in dev_dirs],
                            "frames": dev_pixel[next(iter(dev_pixel))]["n_frames"]
                            if dev_pixel else None}},
           "gates_thresholds": GATES,
           "verdict_mode": args.verdict_mode,
           "recall_scope": args.recall_scope,
           "per_scene_min_candidates": PER_SCENE_MIN_CANDIDATES,
           "dev": {"candidate": dev_cand, "pixel": dev_pixel,
                   "paint_recall": dev_pr},
           "r3": r3}
    by_seed = {}
    for name in dev_cand:
        c = dev_cand[name]
        p = dev_pixel[name]
        pr = dev_pr.get(name) or {}
        by_seed[name] = {
            "candidate_reference_coverage": c["candidate_reference_coverage"],
            "candidate_identity_rate": c["candidate_identity_rate"],
            "left_right_role_agreement": c["left_right_role_agreement"],
            "line_precision": p.get("line_precision"),
            "line_recall": p.get("line_recall"),
            "line_recall_paint_scope": pr.get("recall_paint"),
            "label_nonpaint_frac": pr.get("nonpaint_frac"),
        }
    out["dev"]["gates"] = evaluate_gates(by_seed, mode=args.verdict_mode,
                                         recall_scope=args.recall_scope)
    if r3 is not None:
        r3_seed = {}
        for name in r3["candidate"]:
            c = r3["candidate"][name]
            pp = r3["pixel"][name]
            prr = (r3["paint_recall"].get(name) or {})
            r3_seed[name] = {
                "candidate_reference_coverage":
                    c["candidate_reference_coverage"],
                "candidate_identity_rate": c["candidate_identity_rate"],
                "left_right_role_agreement":
                    c["left_right_role_agreement"],
                "line_precision": pp.get("line_precision"),
                "line_recall": pp.get("line_recall"),
                "line_recall_paint_scope": prr.get("recall_paint"),
                "label_nonpaint_frac": prr.get("nonpaint_frac")}
        out["r3"]["gates"] = evaluate_gates(r3_seed,
                                            mode=args.verdict_mode,
                                            recall_scope=args.recall_scope)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                             encoding="utf-8")
    g = out["dev"]["gates"]
    print(f"[r2] protocol={out['protocol']} definition={out['definition']}")
    print(f"[r2] dev 门（逐 seed）：")
    for metric, gg in g["gates"].items():
        print(f"      {metric:32s} thr={gg['threshold']:.2f} "
              f"mean={gg['mean']} pass={gg['n_pass']} fail={gg['n_fail']} "
              f"unknown={gg['n_unknown']} -> {gg['verdict']}")
    print(f"[r2] verdict={g['verdict']} failing={g['failing_gates']} "
          f"partial={g['partial_gates']}")
    if r3 is not None:
        g3 = out["r3"]["gates"]
        print("[r2] R3 受限认证类（逐 seed）：")
        for metric, gg in g3["gates"].items():
            print(f"      {metric:32s} thr={gg['threshold']:.2f} "
                  f"mean={gg['mean']} pass={gg['n_pass']} fail={gg['n_fail']} "
                  f"-> {gg['verdict']}")
        print(f"[r2] R3 verdict={g3['verdict']} failing={g3['failing_gates']}")
    print(f"[r2] -> {args.out}")
    return 0


def paint_scope_recall(ckpt: Path, dirs: list, ip, device: str) -> dict:
    """漆范围召回 + 分歧率（v8 定义用；真值 = 标签 ∩ 像漆）。

    返回 ``{recall_label, recall_paint, nonpaint_frac, truth_px, truth_paint_px}``；
    与像素门同一预测掩码（Segmenter 的 line mask），只换真值范围。
    """
    from beamng_autopilot.vision.hydra import FrameContext, HydraNet
    from beamng_autopilot.vision.heads.semantic import SemanticHead
    from beamng_autopilot.vision.paint_appearance import paint_like_mask
    from beamng_autopilot.vision.segmentation import Segmenter

    net = HydraNet()
    net.add(SemanticHead(segmenter=Segmenter(model_path=str(ckpt),
                                            device=device)))
    tp = fn = tp_p = fn_p = truth_px = truth_paint_px = 0
    for d in dirs:
        mp = d / "meta.json"
        if not mp.is_file() and (d.parent / "meta.json").is_file():
            mp = d.parent / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        cam = ip.camera_from_meta(meta, view="front_main")
        for fp in sorted(d.glob("*.npz")):
            z = np.load(fp, allow_pickle=True)
            colour = np.asarray(z["colour"])
            label = np.asarray(z["label"])
            truth = label == 2
            if not truth.any():
                continue
            ctx = FrameContext(frame_rgb=colour, cam=cam, pos=np.zeros(3),
                               heading=0.0, ground_z=0.0, role="front_main")
            out = net.run(ctx).get("semantic")
            line = np.asarray(out.masks["line"], dtype=bool)
            paint = paint_like_mask(colour)
            truth_p = truth & paint
            tp += int((line & truth).sum())
            fn += int((~line & truth).sum())
            tp_p += int((line & truth_p).sum())
            fn_p += int((~line & truth_p).sum())
            truth_px += int(truth.sum())
            truth_paint_px += int(truth_p.sum())
    return {"recall_label": round(tp / max(tp + fn, 1), 4),
            "recall_paint": round(tp_p / max(tp_p + fn_p, 1), 4),
            "nonpaint_frac": (round(1.0 - truth_paint_px / truth_px, 4)
                              if truth_px else None),
            "truth_px": truth_px, "truth_paint_px": truth_paint_px}


if __name__ == "__main__":
    raise SystemExit(main())
