"""候选/身份失败归因：把 C→R→M→L→A 的损失定位到具体层（方案 v2 §S2/§S3.3）。

方案点名："失败样本能定位'掩码漏检 / 候选提取 / 投影 / 角色分配'哪一层"，并且
"不得把所有身份率低的问题直接归因于神经网络"。本脚本在**审计后的唯一清单**上
跑一次探针（与标定/评价同一实现），然后按层归因：

* 掩码层：帧级 `compare_masks` 的 recall/IoU（模型没画出真值线 = 掩码漏检）；
* 候选层：真值有链成线但该帧候选为 0（候选提取漏了）；
* 参考层（C→R）：候选 `reference_available=False` —— 该侧真值没有足够参考像素
  （`side_ref_px` 分布；含 MIN_REF_PX 阈值的影响）；
* 匹配层（R→M）：有参考但 `matched=False` —— 按"最近引擎线的横向距离"分桶，
  区分"几何/容差"与"该侧确实没有线"；
* 角色层（M→L→A）：`role_agrees=False`。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_candidate_failure_attribution.py \\
        --pack-dir logs/experiments/review_pack_20260926 \\
        --model logs/m5_seg/seg_model/best.pt --device cuda \\
        --out logs/experiments/review_pack_20260926/failure_attribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _bucket_dist(values: list[float], edges: list[float]) -> dict:
    out = {f"<{edges[0]}": 0}
    for a, b in zip(edges, edges[1:]):
        out[f"{a}~{b}"] = 0
    out[f">={edges[-1]}"] = 0
    for v in values:
        if v < edges[0]:
            out[f"<{edges[0]}"] += 1
            continue
        placed = False
        for a, b in zip(edges, edges[1:]):
            if a <= v < b:
                out[f"{a}~{b}"] += 1
                placed = True
                break
        if not placed:
            out[f">={edges[-1]}"] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir",
                    default="logs/experiments/review_pack_20260926")
    ap.add_argument("--trees", nargs="+", default=["reviewed_full"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--match-tolerance-m", type=float, default=None,
                    help="仅用于归因分桶（不改变探针的冻结匹配口径）")
    ap.add_argument("--line-road-keep-frac", type=float, default=None,
                    help="候选层旋钮：线上连通域必须在路面内的比例（默认 0.5）")
    ap.add_argument("--line-road-elongated-frac", type=float, default=None,
                    help="候选层第二档：细长笔画在路面内的下限（默认 0.25）")
    ap.add_argument("--label", default=None,
                    help="本次配置的名字（写进 JSON，便于对照表引用）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import m5_marking_identity_probe as ip
    from beamng_autopilot.experiments.manifest import dir_group
    from m5_identity_calibration import audited_inventory, _group_by_dir

    inv = audited_inventory(Path(args.pack_dir), trees=tuple(args.trees))
    per_scene: dict = {}
    totals = {"frames": 0, "frames_with_line_truth": 0,
              "frames_mask_recall_zero": 0, "frames_no_candidate": 0,
              "candidates": 0, "ref_available": 0, "matched": 0,
              "role_comparable": 0, "role_agree": 0}
    dist_ref_px: list[float] = []
    dist_unmatched_gap_m: list[float] = []
    unmatched_kinds: dict = {}
    unmatched_offroad: list[float] = []
    ref_missing_sides: dict = {}
    mask_recall: list[float] = []
    mask_precision: list[float] = []
    mask_iou: list[float] = []
    spurious_by_arm: dict = {}
    spurious_by_colour: dict = {}
    unmatched_by_arm: dict = {}
    unmatched_by_colour: dict = {}
    matched_by_arm: dict = {}
    for g, paths in sorted(inv["by_scene"].items()):
        agg = {"frames": 0, "frames_with_line_truth": 0,
               "frames_mask_recall_zero": 0, "frames_no_candidate": 0,
               "candidates": 0, "ref_available": 0, "matched": 0,
               "role_comparable": 0, "role_agree": 0}
        for d_str, dir_paths in sorted(_group_by_dir(paths).items()):
            d = Path(d_str)
            meta_p = d / "meta.json"
            meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta_p.is_file() else None)
            res = ip.probe(d, meta, view=d.name, model_path=str(args.model),
                           frames=dir_paths, device=args.device,
                           line_road_keep_frac=args.line_road_keep_frac,
                           line_road_elongated_frac=args.line_road_elongated_frac)
            for r in (res.get("rows") or []):
                agg["frames"] += 1
                totals["frames"] += 1
                has_truth = bool(r.get("counts", {}).get("P_frames"))
                if has_truth:
                    agg["frames_with_line_truth"] += 1
                    totals["frames_with_line_truth"] += 1
                rec = (r.get("recall") if r.get("recall") is not None
                       else r.get("line_recall"))
                if has_truth and rec is not None:
                    mask_recall.append(float(rec))
                    if float(rec) <= 0.0:
                        agg["frames_mask_recall_zero"] += 1
                        totals["frames_mask_recall_zero"] += 1
                if r.get("precision") is not None:
                    mask_precision.append(float(r["precision"]))
                if r.get("iou") is not None:
                    mask_iou.append(float(r["iou"]))
                cands = r.get("candidates") or []
                eng_lines = r.get("engine_lines") or []
                if has_truth and not cands:
                    agg["frames_no_candidate"] += 1
                    totals["frames_no_candidate"] += 1
                for c in cands:
                    agg["candidates"] += 1
                    totals["candidates"] += 1
                    arm = ip.candidate_source_of(c)
                    colour = str(c.get("colour") or "unknown")
                    if c.get("reference_available"):
                        agg["ref_available"] += 1
                        totals["ref_available"] += 1
                    else:
                        dist_ref_px.append(float(c.get("side_ref_px") or 0))
                        ref_missing_sides[str(c.get("side"))] = \
                            ref_missing_sides.get(str(c.get("side")), 0) + 1
                        spurious_by_arm[arm] = spurious_by_arm.get(arm, 0) + 1
                        spurious_by_colour[colour] = \
                            spurious_by_colour.get(colour, 0) + 1
                    if c.get("matched"):
                        agg["matched"] += 1
                        totals["matched"] += 1
                        matched_by_arm[arm] = matched_by_arm.get(arm, 0) + 1
                        if c.get("role_agrees") is not None:
                            agg["role_comparable"] += 1
                            totals["role_comparable"] += 1
                            if c.get("role_agrees"):
                                agg["role_agree"] += 1
                                totals["role_agree"] += 1
                    elif c.get("reference_available"):
                        # 有参考但没匹配上：离**最近**引擎线的横向距离（不管容差），
                        # 用来区分"几何/容差"与"该侧根本没有线"
                        lat = float(c.get("lat_m") or 0.0)
                        if eng_lines:
                            gaps = [abs(lat - float(ln.get("lat_m") or 0.0))
                                    for ln in eng_lines]
                            dist_unmatched_gap_m.append(min(gaps))
                        unmatched_kinds[str(c.get("kind"))] = \
                            unmatched_kinds.get(str(c.get("kind")), 0) + 1
                        unmatched_by_arm[arm] = unmatched_by_arm.get(arm, 0) + 1
                        unmatched_by_colour[colour] = \
                            unmatched_by_colour.get(colour, 0) + 1
                        if c.get("off_road_frac") is not None:
                            unmatched_offroad.append(float(c["off_road_frac"]))
        per_scene[g] = agg
    tot = totals
    out = {
        "pack_dir": str(args.pack_dir), "trees": list(args.trees),
        "model": str(args.model), "device": args.device,
        "config_label": args.label,
        "line_road_keep_frac": args.line_road_keep_frac,
        "line_road_elongated_frac": args.line_road_elongated_frac,
        "inventory": {k: inv[k] for k in
                      ("raw_inputs", "n_unique", "n_rejected", "n_conflicts")},
        "totals": tot,
        "ratios": {
            "reference_coverage_R_over_C": (None if not tot["candidates"]
                                            else round(tot["ref_available"]
                                                       / tot["candidates"], 4)),
            "identity_M_over_R": (None if not tot["ref_available"]
                                  else round(tot["matched"]
                                             / tot["ref_available"], 4)),
            "role_A_over_L": (None if not tot["role_comparable"]
                              else round(tot["role_agree"]
                                         / tot["role_comparable"], 4)),
        },
        "attribution": {
            "frames_with_line_truth": tot["frames_with_line_truth"],
            "frames_mask_recall_zero": tot["frames_mask_recall_zero"],
            "frames_no_candidate": tot["frames_no_candidate"],
            "mask_recall_mean": (None if not mask_recall else
                                 round(sum(mask_recall) / len(mask_recall), 4)),
            "mask_precision_mean": (None if not mask_precision else
                                    round(sum(mask_precision)
                                          / len(mask_precision), 4)),
            "mask_iou_mean": (None if not mask_iou else
                              round(sum(mask_iou) / len(mask_iou), 4)),
            "ref_missing_side_px_hist": _bucket_dist(dist_ref_px,
                                                     [1, 5, 20, 50, 200]),
            "ref_missing_by_side": ref_missing_sides,
            "unmatched_has_engine_line_gap_m_hist": _bucket_dist(
                dist_unmatched_gap_m, [0.5, 1.0, 2.0, 4.0, 8.0]),
            "unmatched_by_kind": unmatched_kinds,
            # 来源臂归因（决定"算法级改动该动哪条臂"）：假线/未匹配/已匹配
            # 各自落在 learned / cv_only / unattributed 上
            "spurious_by_arm": spurious_by_arm,
            "spurious_by_colour": spurious_by_colour,
            "unmatched_by_arm": unmatched_by_arm,
            "unmatched_by_colour": unmatched_by_colour,
            "matched_by_arm": matched_by_arm,
            "unmatched_off_road_frac_mean": (
                None if not unmatched_offroad
                else round(sum(unmatched_offroad) / len(unmatched_offroad), 4)),
        },
        "per_scene": per_scene,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    t, a = out["totals"], out["attribution"]
    print(f"[attr] 帧 {t['frames']}（有标线真值 {t['frames_with_line_truth']}）："
          f"掩码 recall=0 {t['frames_mask_recall_zero']}、"
          f"有真值但无候选 {t['frames_no_candidate']}")
    print(f"[attr] 候选 {t['candidates']}：有参考 {t['ref_available']}"
          f"（覆盖 {out['ratios']['reference_coverage_R_over_C']}）→ "
          f"匹配 {t['matched']}（身份 {out['ratios']['identity_M_over_R']}）→ "
          f"角色可比 {t['role_comparable']}、一致 {t['role_agree']}"
          f"（角色 {out['ratios']['role_A_over_L']}）")
    print(f"[attr] 无参考候选的该侧参考像素直方图："
          f"{a['ref_missing_side_px_hist']}")
    print(f"[attr] 未匹配候选（有参考）离最近引擎线的横向距离："
          f"{a['unmatched_has_engine_line_gap_m_hist']}")
    print(f"[attr] 未匹配候选来源：{a['unmatched_by_kind']}；"
          f"平均路外占比 {a['unmatched_off_road_frac_mean']}")
    print(f"[attr] 来源臂：假线（无参考）{a['spurious_by_arm']}；"
          f"未匹配 {a['unmatched_by_arm']}；已匹配 {a['matched_by_arm']}")
    print(f"[attr] 来源颜色：假线 {a['spurious_by_colour']}；"
          f"未匹配 {a['unmatched_by_colour']}")
    if args.out:
        print(f"[attr] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
