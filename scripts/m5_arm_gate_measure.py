"""臂级门表：同一套提取/匹配口径下，逐 arm（checkpoint）量全部任务门。

为什么要有它：`rounds` 的判定只对**当轮两臂**出数，而"提取器/协议换版后旧判定
不可比"（协议 v7 起）需要**把已有 checkpoint 在同一口径下重测**——本脚本就是
那个入口：给若干个 `名字=checkpoint`，在**同一开发集、同一实现**上跑探针，
输出覆盖率 / 身份率 / 角色一致率 / 掩码 recall-precision-IoU / 路外候选。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_arm_gate_measure.py `
        --arm devdist-s42=logs/experiments/<run>/round0/seed42/checkpoint_last.pt `
        --arm baseline-s42=logs/experiments/<run>/baseline/seed42/checkpoint_last.pt `
        --out logs/experiments/<run>/gate_v7.json

不训练、不启动游戏；只读 checkpoint 与开发帧。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_DEV = "logs/experiments/review_pack_20260926/reviewed_full"


def summarize_rows(rows: list) -> dict:
    """逐帧探针行 -> 臂级计数与比率（纯函数，可单测）。

    只累加**整数计数**再算一次比率（与判定同口径）；掩码指标取逐帧均值并
    同时给出样本数（缺测不静默当 0）。

    另外给一份**表面口径**（§10.3 变更提案的双口径对照）：把"像素主要落在
    **人工标签的背景类**（``off_road_frac >= 0.5``）"的候选从分母里排除，
    即只统计"标签自己说这是路面或漆线"的候选。判据用的是**标签自己的类**，
    不是阈值旋钮；``matched`` 一条都不应被排除（匹配上的候选必然落在漆线上），
    这个不变式在返回值里可见（``surface_scope.matched_lost``）。
    """
    from beamng_autopilot.experiments import candidate_metrics as cm
    acc = cm.empty()
    mask = {"recall": [], "precision": [], "iou": []}
    off_road = n_cand = frames = 0
    by_role: dict = {}
    # 匹配对上的角色混淆（model_role × engine_role）与逐参考实例被匹配数：
    # 角色一致率 A/L 只有 0.70 上下时，必须能回答"谁被叫成了谁"
    conf: dict = {}
    inst_matched: dict = {}
    # 表面口径（双口径对照）：候选是否"标签说在路面/漆线上"
    surf = {"C": 0, "R": 0, "M": 0, "L": 0, "A": 0, "excluded_C": 0,
            "excluded_R": 0, "matched_lost": 0}
    merges = merged = 0
    for r in rows or []:
        c = r.get("counts") or {}
        cm.accumulate(acc, c)
        frames += 1
        for k in mask:
            if r.get(k) is not None:
                mask[k].append(float(r[k]))
        off_road += int(r.get("candidates_off_road") or 0)
        n_cand += int(r.get("n_candidates") or 0)
        for c in (r.get("candidates") or []):
            # 逐角色/逐参考实例的滚存：身份率 ~0.5 时最需要回答"是弥散噪声还是
            # 某个具体实例（某条线/某个角色）系统性不匹配"（R3 归因要用的分解）
            rk = str(c.get("role") or "unknown")
            b = by_role.setdefault(rk, {"C": 0, "R": 0, "M": 0})
            b["C"] += 1
            if c.get("reference_available"):
                b["R"] += 1
            if c.get("matched"):
                b["M"] += 1
            er = c.get("engine_role")
            if er and c.get("matched"):
                conf.setdefault(rk, {})
                conf[rk][str(er)] = conf[rk].get(str(er), 0) + 1
                inst_matched[str(er)] = inst_matched.get(str(er), 0) + 1
            on_surface = float(c.get("off_road_frac") or 0.0) < 0.5
            matched = bool(c.get("matched"))
            if on_surface:
                surf["C"] += 1
                if c.get("reference_available"):
                    surf["R"] += 1
                if matched:
                    surf["M"] += 1
                    if c.get("role_agrees") is not None:
                        surf["L"] += 1
                        if c.get("role_agrees"):
                            surf["A"] += 1
            else:
                surf["excluded_C"] += 1
                if c.get("reference_available"):
                    surf["excluded_R"] += 1
                # **表面口径的召回代价**：现口径按"横向 ≤0.8 m"判匹配（不看像素），
                # 所以会有"横向近、像素却在背景上"的匹配；表面口径把它们排除，
                # 这个数就是代价（提案里必须与收益一起报，不能只报收益）。
                if matched:
                    surf["matched_lost"] += 1
        g = r.get("line_candidate_gate") or {}
        mg = (g.get("line_candidate_merge") or {}) if isinstance(g, dict) else {}
        merges += int(mg.get("groups") or 0)
        merged += int(mg.get("merged") or 0)
    rt = cm.ratios(acc)
    out = {"frames": frames, **{k: int(acc[k]) for k in ("P_frames", "C", "R",
                                                         "M", "L", "A",
                                                         "C_outside_P")},
           "candidate_reference_coverage": rt["candidate_reference_coverage"],
           "candidate_identity_rate": rt["candidate_identity_rate"],
           "left_right_role_agreement": rt["left_right_role_agreement"],
           "candidates_off_road": off_road, "n_candidates": n_cand,
           "off_road_frac": (round(off_road / n_cand, 4) if n_cand else None),
           "merge_groups": merges, "merge_merged": merged,
           "surface_scope": {
               **surf,
               "candidate_reference_coverage": (
                   round(surf["R"] / surf["C"], 4) if surf["C"] else None),
               "candidate_identity_rate": (
                   round(surf["M"] / surf["R"], 4) if surf["R"] else None),
               "left_right_role_agreement": (
                   round(surf["A"] / surf["L"], 4) if surf["L"] else None)}}
    for k, v in mask.items():
        out[f"mask_{k}_mean"] = (round(sum(v) / len(v), 4) if v else None)
        out[f"mask_{k}_n"] = len(v)
    # 身份率按模型自报角色分解（哪个角色在拉低身份率）
    for rk, b in by_role.items():
        b["identity"] = (round(b["M"] / b["R"], 4) if b["R"] else None)
    out["by_role"] = dict(sorted(by_role.items()))
    out["role_confusion"] = {k: dict(sorted(v.items()))
                             for k, v in sorted(conf.items())}
    out["reference_instances_matched"] = dict(sorted(inst_matched.items()))
    return out


def _probe():
    spec = importlib.util.spec_from_file_location(
        "m5_marking_identity_probe", ROOT / "scripts"
        / "m5_marking_identity_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_marking_identity_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=CHECKPOINT")
    ap.add_argument("--dev-runs", nargs="*", default=None,
                    help=f"开发集目录（默认 {DEFAULT_DEV}/*/front_main）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = ([Path(p) for p in args.dev_runs] if args.dev_runs
            else sorted((ROOT / DEFAULT_DEV).glob("*/front_main")))
    if not dirs:
        raise SystemExit(f"开发集为空：{args.dev_runs or DEFAULT_DEV}")
    ip = _probe()
    out = {"dev_runs": [str(d) for d in dirs], "arms": {}}
    for spec in args.arm:
        name, _, ck = str(spec).partition("=")
        if not ck:
            raise SystemExit(f"--arm 需要 NAME=CHECKPOINT，收到 {spec!r}")
        rows = []
        per_scene: dict = {}
        for d in dirs:
            # meta 可能在目录里，也可能在**父目录**（导出包是
            # <pkg>/meta.json + <pkg>/front_main/frame_*.npz）——不找父目录时
            # 探针会拒测、计数全 0（实测踩到：R3 首轮验收 C=0/R=0）
            meta_p = d / "meta.json"
            if not meta_p.is_file() and (d.parent / "meta.json").is_file():
                meta_p = d.parent / "meta.json"
            meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta_p.is_file() else None)
            res = ip.probe(d, meta, view=d.name, model_path=str(ROOT / ck),
                           device=args.device)
            drows = res.get("rows") or []
            # 逐场景明细：冻结口径有 per_scene_min_candidates=30（在 R 上），
            # 池化均值会掩盖单场景样本不足（R3 设计 §7 教训 1）
            per_scene[d.parent.name + "/" + d.name] = summarize_rows(drows)
            rows.extend(drows)
        s = summarize_rows(rows)
        out["arms"][name] = {"checkpoint": str(ck), **s,
                             "per_scene": per_scene}
        print(f"[gate] {name:16s} C={s['C']:4d} R={s['R']:4d} M={s['M']:4d} "
              f"L={s['L']:4d} A={s['A']:4d} | 覆盖={s['candidate_reference_coverage']} "
              f"身份={s['candidate_identity_rate']} 角色={s['left_right_role_agreement']} "
              f"| 掩码 r/p/iou={s['mask_recall_mean']}/{s['mask_precision_mean']}/"
              f"{s['mask_iou_mean']} | 路外候选 {s['candidates_off_road']}"
              f"（{s['off_road_frac']}）| 合并 {s['merge_merged']}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[gate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
