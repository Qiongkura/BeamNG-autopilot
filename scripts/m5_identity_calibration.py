"""身份/覆盖门槛的标定测量（方案 §10.2：在独立校准数据上定值，冻结后再搜索）。

用途：在**权威人工真值集**上测出候选匹配口径的实际分布，作为门槛标定的证据：

* `candidate_reference_coverage`：产生的候选里有多少条有参考可判（分母=全部候选）；
* `candidate_identity_rate`：有参考候选的匹配率；
* `left_right_role_agreement`：左右角色一致率；
* 每个场景的**候选条数**（用于定"一个场景至少多少条候选才允许单独判身份"）。

参考来自数据集自己的 `label == 2`（人工真值），所以本测量与训练标签无关。
**本脚本只测量，不改阈值**：阈值由人依据这些分布与"要求驱动"的原则写进协议。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_identity_calibration.py \\
        --pack-dir logs/experiments/review_pack_20260926 \\
        --model production=logs/m5_seg/seg_model/best.pt \\
        --model cand=logs/experiments/t14_e0_20260925/seed43/checkpoint_last.pt \\
        --out logs/experiments/review_pack_20260926/identity_calibration.json
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

from beamng_autopilot.experiments.manifest import dir_group  # noqa: E402


def _reviewed_dirs(pack_dir: Path) -> dict:
    """复核目录 -> 场景键（两棵树按目录去重：reviewed 与 reviewed_full 可能重叠）。"""
    out: dict = {}
    for tree in ("reviewed", "reviewed_full"):
        for d in sorted((pack_dir / tree).glob("*/front_main")):
            if not list(d.glob("frame_*.npz")):
                continue
            g = dir_group(d)
            out.setdefault(g, []).append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir",
                    default="logs/experiments/review_pack_20260926")
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个目录最多用多少帧（调试用；缺省全用）")
    args = ap.parse_args(argv)

    import m5_marking_identity_probe as probe_mod
    pack_dir = Path(args.pack_dir)
    by_scene = _reviewed_dirs(pack_dir)
    print(f"[calib] 场景 {len(by_scene)} 个，"
          f"帧 {sum(len(list(d.glob('frame_*.npz'))) for ds in by_scene.values() for d in ds)}")

    models = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        models.append((name.strip(), Path(path.strip())))

    out = {"pack_dir": str(pack_dir), "models": {}, "per_scene_frames": {
        g: sum(len(list(d.glob("frame_*.npz"))) for d in ds)
        for g, ds in sorted(by_scene.items())}}
    for name, path in models:
        if not path.exists():
            print(f"[calib] 缺权重 {path}")
            continue
        per_scene = {}
        for g, dirs in sorted(by_scene.items()):
            # 分母口径（探针 summary）：role_agreement_rate = roles_agreeing /
            # **candidates_matched**（已匹配候选数），不是帧数——实测读错键会
            # 把角色一致率当成 None。
            agg = {"n_candidates": 0, "n_with_reference": 0,
                   "matched": 0, "role_agree": 0, "role_total": 0,
                   "off_road": 0, "frames": 0,
                   "frames_with_line_ref": 0}
            for d in dirs:
                meta_p = d / "meta.json"
                meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                        if meta_p.is_file() else None)
                res = probe_mod.probe(d, meta, view=d.name,
                                      model_path=str(path),
                                      limit=args.limit)
                s = res.get("summary") or {}
                if not s:
                    print(f"[calib]   {g} / {d.name}: 无 summary（{res.get('reason')}）")
                    continue
                agg["frames"] += int(s.get("n_frames") or 0)
                agg["n_candidates"] += int(s.get("n_candidates") or 0)
                agg["n_with_reference"] += int(
                    s.get("n_candidates_with_reference") or 0)
                agg["matched"] += int(s.get("candidates_matched") or 0)
                agg["role_agree"] += int(s.get("roles_agreeing") or 0)
                agg["role_total"] += int(s.get("candidates_matched") or 0)
                agg["off_road"] += int(s.get("candidates_off_road") or 0)
                # 有标线参考的帧数：无标线场景（人确认无线）没有参考，
                # 覆盖率在那里天然为 0——门槛标定必须把它们分开算。
                agg["frames_with_line_ref"] += int(
                    s.get("frames_with_engine_line") or 0)
            n_cand = agg["n_candidates"]
            n_ref = agg["n_with_reference"]
            per_scene[g] = {
                **agg,
                "has_line_reference": bool(agg["frames_with_line_ref"]),
                "off_road_frac": (None if not n_cand
                                  else round(agg["off_road"] / n_cand, 4)),
                "candidate_reference_coverage": (None if not n_cand
                                                 else round(n_ref / n_cand, 4)),
                "candidate_identity_rate": (None if not n_ref
                                            else round(agg["matched"] / n_ref, 4)),
                "left_right_role_agreement": (
                    None if not agg["role_total"]
                    else round(agg["role_agree"] / agg["role_total"], 4)),
            }
        tot_c = sum(v["n_candidates"] for v in per_scene.values())
        tot_r = sum(v["n_with_reference"] for v in per_scene.values())
        tot_m = sum(v["matched"] for v in per_scene.values())
        tot_ra = sum(v["role_agree"] for v in per_scene.values())
        tot_rt = sum(v["role_total"] for v in per_scene.values())
        ref_scenes = [v for v in per_scene.values() if v["has_line_reference"]]
        rc = sum(v["n_candidates"] for v in ref_scenes)
        rr = sum(v["n_with_reference"] for v in ref_scenes)
        out["models"][name] = {
            "path": str(path), "per_scene": per_scene,
            "reference_scenes_only": {
                "n_scenes": len(ref_scenes), "n_candidates": rc,
                "n_with_reference": rr,
                "candidate_reference_coverage": (None if not rc
                                                 else round(rr / rc, 4)),
                "candidate_identity_rate": (
                    None if not rr else round(
                        sum(v["matched"] for v in ref_scenes) / rr, 4)),
                "left_right_role_agreement": (
                    None if not sum(v["role_total"] for v in ref_scenes)
                    else round(sum(v["role_agree"] for v in ref_scenes)
                               / sum(v["role_total"] for v in ref_scenes), 4)),
            },
            "pooled": {
                "n_candidates": tot_c, "n_with_reference": tot_r,
                "candidate_reference_coverage": (None if not tot_c
                                                 else round(tot_r / tot_c, 4)),
                "candidate_identity_rate": (None if not tot_r
                                            else round(tot_m / tot_r, 4)),
                "left_right_role_agreement": (
                    None if not tot_rt else round(tot_ra / tot_rt, 4)),
            },
        }
        p = out["models"][name]["pooled"]
        ro = out["models"][name]["reference_scenes_only"]
        print(f"[calib] {name}: 全部场景 候选 {p['n_candidates']} 有参考 "
              f"{p['n_with_reference']} 覆盖 {p['candidate_reference_coverage']} "
              f"身份率 {p['candidate_identity_rate']} "
              f"角色一致 {p['left_right_role_agreement']}")
        print(f"[calib] {name}: **仅有标线参考的场景**（{ro['n_scenes']} 个）候选 "
              f"{ro['n_candidates']} 覆盖 {ro['candidate_reference_coverage']} "
              f"身份率 {ro['candidate_identity_rate']} "
              f"角色一致 {ro['left_right_role_agreement']}")
        for g, v in sorted(per_scene.items()):
            print(f"[calib]   {g:28s} 候选 {v['n_candidates']:4d} "
                  f"覆盖 {v['candidate_reference_coverage']} "
                  f"身份 {v['candidate_identity_rate']} "
                  f"角色 {v['left_right_role_agreement']}")
    outp = Path(args.out) if args.out else (pack_dir / "identity_calibration.json")
    outp.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[calib] -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
