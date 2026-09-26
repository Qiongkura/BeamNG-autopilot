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

from beamng_autopilot.experiments import candidate_metrics as cm  # noqa: E402
from beamng_autopilot.experiments.checkpoint import file_sha16  # noqa: E402
from beamng_autopilot.experiments.manifest import dir_group  # noqa: E402


def audited_inventory(pack_dir: Path,
                      *, trees: tuple = ("reviewed_full",)) -> dict:
    """**唯一**评价清单：内容去重 + 冲突隔离 + 拒绝理由（方案 v2 §3.1）。

    标定与评价都消费这一份，禁止各自 glob（实测：两棵树重叠 -> 159 输入里
    只有 136 张唯一图、22 个重复内容组）。返回
    ``{records, rejected, by_scene, by_dir, raw_inputs, conflicts, aliases}``；
    ``records`` 里每条带 ``path/group/view/label_sha16``。

    ``trees``：参与测量的目录树，**必须显式给出**（默认只取完成版
    ``reviewed_full``）。实测（2026-09-26）：把早先的预览包 ``reviewed`` 一起
    混进来会产生 24 个内容冲突组——同一张图在预览包里被复制进两个槽位、且两次
    人工标注不同（打包缺陷，已在 ``m5_review_pack.py`` 的池内去重修掉）。
    预览包是**被取代的**输入：把它计入测量等于用同一张图的两份矛盾标签；
    排除它不等于"放行冲突"——冲突仍在 union 审计里逐条记录（见
    ``audit_union``），只是不进入测量分母。完成版单独审计为 136/136 无冲突。
    """
    from beamng_autopilot.experiments.manifest import DatasetManifest
    dirs = []
    raw = 0
    for tree in trees:
        for d in sorted((pack_dir / tree).glob("*/front_main")):
            n = len(list(d.glob("frame_*.npz")))
            if n:
                dirs.append(d)
                raw += n
    mf = DatasetManifest.build(dirs, root=Path("."))
    records = [{"path": r.path, "group": r.group, "view": r.view,
                "label_sha16": r.label_sha16, "run": r.run}
               for r in mf.records if not r.reject_reason]
    rejected = [{"path": r.path, "run": r.run, "reason": r.reject_reason,
                 "code": getattr(r, "reject_code", "")}
                for r in mf.records if r.reject_reason]
    by_scene: dict = {}
    by_dir: dict = {}
    for r in records:
        by_scene.setdefault(r["group"], []).append(r["path"])
        by_dir.setdefault(str(Path(r["path"]).parent), []).append(r["path"])
    # 冲突与别名取 **manifest 的同一份口径**（方案 v2 §3.1）：此前这里另抄一套
    # 按 label_sha16 的分组，算出的别名组恒为 0，而 manifest 里同一内容多份
    # 拷贝是真实存在的（实测 24 个冲突组）。
    conflicts = [dict(c) for c in (getattr(mf, "conflicts", None) or [])]
    _aliases = dict(getattr(mf, "aliases", None) or {})
    alias_groups = {k: list((v or {}).get("paths") or [])
                    for k, v in _aliases.items()}
    return {"records": records, "rejected": rejected, "by_scene": by_scene,
            "by_dir": by_dir, "raw_inputs": raw,
            "n_unique": len(records), "n_rejected": len(rejected),
            "conflicts": conflicts, "n_conflicts": len(conflicts),
            "content_alias_groups": len(alias_groups),
            "content_aliases": {k: v for k, v in
                                list(alias_groups.items())[:10]},
            "dirs": [str(d) for d in dirs]}


def _reviewed_dirs(pack_dir: Path) -> dict:
    """兼容旧调用：场景 -> 目录列表（**仅供展示**；测量请用 audited_inventory）。"""
    out: dict = {}
    for tree in ("reviewed", "reviewed_full"):
        for d in sorted((pack_dir / tree).glob("*/front_main")):
            if not list(d.glob("frame_*.npz")):
                continue
            out.setdefault(dir_group(d), []).append(d)
    return out


def _group_by_dir(paths: list) -> dict:
    """按所在目录分组（探针需要"目录 + 该目录的帧清单"，meta 才对得上）。"""
    out: dict = {}
    for p in paths:
        out.setdefault(str(Path(p).parent), []).append(p)
    return out


def audit_union(pack_dir: Path,
                *, trees: tuple = ("reviewed", "reviewed_full")) -> dict:
    """把**所有**目录树一起审计：冲突/别名逐条留档（测量分母不取它）。

    用途：证明"同一内容多份拷贝"确实被发现了，而不是靠"不扫它"来假装没有
    冲突。测量清单（:func:`audited_inventory`）默认只取完成版，两者的差额
    必须能逐条对上。
    """
    inv = audited_inventory(pack_dir, trees=trees)
    return {"trees": list(trees), "raw_inputs": inv["raw_inputs"],
            "n_unique": inv["n_unique"], "n_rejected": inv["n_rejected"],
            "n_conflicts": inv["n_conflicts"], "conflicts": inv["conflicts"],
            "rejected": inv["rejected"],
            "content_alias_groups": inv["content_alias_groups"]}


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
    ap.add_argument("--trees", nargs="+", default=["reviewed_full"],
                    help="参与**测量**的目录树（默认只取完成版 reviewed_full；"
                         "把预览包 reviewed 混进来会产生 24 个内容冲突组，见 "
                         "audited_inventory 的说明）")
    ap.add_argument("--device", default=None,
                    help="推理设备（cpu/cuda）。缺省交给 Segmenter 默认；实际"
                         "设备由探针读回并写进报告的 device 字段")
    args = ap.parse_args(argv)

    import m5_marking_identity_probe as probe_mod
    pack_dir = Path(args.pack_dir)
    inv = audited_inventory(pack_dir, trees=tuple(args.trees))
    union = audit_union(pack_dir)
    by_scene = inv["by_scene"]
    print(f"[calib] 测量清单（trees={list(args.trees)}）：原始输入 "
          f"{inv['raw_inputs']} 帧 -> 唯一 {inv['n_unique']} 帧，拒绝 "
          f"{inv['n_rejected']}，内容冲突 {inv['n_conflicts']} 组，重复别名组 "
          f"{inv['content_alias_groups']}；场景 {len(by_scene)} 个")
    print(f"[calib] 全部目录树审计（trees={union['trees']}）：原始输入 "
          f"{union['raw_inputs']} -> 唯一 {union['n_unique']}，拒绝 "
          f"{union['n_rejected']}，内容冲突 {union['n_conflicts']} 组"
          + ("（测量清单不含被取代的预览包；冲突逐条留在 union_audit）"
             if union["n_conflicts"] else ""))
    for r in inv["rejected"][:3]:
        print(f"[calib]   拒绝：{Path(r['path']).name} @ {r['run']} - {r['reason'][:80]}")

    models = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        models.append((name.strip(), Path(path.strip())))

    out = {"pack_dir": str(pack_dir), "models": {},
           "measure_trees": list(args.trees),
           "inventory": {k: inv[k] for k in
                         ("raw_inputs", "n_unique", "n_rejected",
                          "content_alias_groups", "n_conflicts")},
           "union_audit": union,
           "rejected": inv["rejected"],
           "conflicts": inv["conflicts"],
           "content_aliases": inv["content_aliases"],
           "per_scene_frames": {g: len(ps)
                                for g, ps in sorted(by_scene.items())}}
    for name, path in models:
        if not path.exists():
            print(f"[calib] 缺权重 {path}")
            continue
        per_scene = {}
        devices: set = set()
        for g, paths in sorted(by_scene.items()):
            # 计数**先累加整数、最后算一次比率**（方案 v2 §3.3）：本脚本不再另抄
            # 一套公式，直接调用 `candidate_metrics` 的同一实现（S2 验收要求
            # 标定/循环/评价逐项相同）。
            acc = cm.empty()
            extra = {"off_road": 0, "frames": 0, "frames_unknown": 0,
                     "frames_skipped": 0, "n_errors": 0,
                     "incomplete_runs": [], "frames_with_line_ref": 0}
            # 按目录分组、只喂**该目录已接受的帧**（显式清单，不再 glob）
            for d_str, dir_paths in sorted(
                    _group_by_dir(paths).items()):
                d = Path(d_str)
                meta_p = d / "meta.json"
                meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                        if meta_p.is_file() else None)
                res = probe_mod.probe(d, meta, view=d.name,
                                      model_path=str(path),
                                      frames=dir_paths,
                                      device=args.device,
                                      limit=args.limit)
                s = res.get("summary") or {}
                if s.get("device"):
                    devices.add(str(s["device"]))
                _c = s.get("counts") or {}
                if not s or not _c:
                    # 没有 summary/计数 = **缺测**，不是"测了 0 帧"：不往计数里
                    # 加 0（那会让缺测与实测 0 无法区分），单独记账并标明原因。
                    extra["frames_unknown"] += len(dir_paths)
                    extra["incomplete_runs"].append(str(d))
                    print(f"[calib]   {g} / {d.name}: 缺测（"
                          f"{res.get('reason') or 'no counts in summary'}）")
                    continue
                cm.accumulate(acc, _c)
                # 按探针**实际** schema 读（frames_processed，不是 n_frames）；
                # 字段缺失记 unknown，不用 `or 0` 静默当已处理。
                if s.get("frames_processed") is None:
                    extra["frames_unknown"] += len(dir_paths)
                    extra["incomplete_runs"].append(str(d))
                else:
                    extra["frames"] += int(s["frames_processed"])
                _sk = int(s.get("frames_skipped") or 0)
                extra["frames_skipped"] += _sk
                extra["n_errors"] += int(s.get("n_errors") or 0)
                if _sk or int(s.get("n_errors") or 0):
                    extra["incomplete_runs"].append(str(d))
                extra["off_road"] += int(s.get("candidates_off_road") or 0)
                # 有标线参考的帧数：无标线场景（人确认无线）没有参考，
                # 覆盖率在那里天然为 0——门槛标定必须把它们分开算。
                extra["frames_with_line_ref"] += int(
                    s.get("frames_with_engine_line")
                    or _c.get("P_frames") or 0)
            r = cm.ratios(acc)
            n_cand, n_ref = int(acc["C"]), int(acc["R"])
            per_scene[g] = {
                **extra, **{k: int(acc[k]) for k in cm.COUNTERS}, **r,
                # 兼容旧键名（下游/报告沿用）：同一批整数计数的别名
                "n_candidates": n_cand, "n_with_reference": n_ref,
                "matched": int(acc["M"]), "role_agree": int(acc["A"]),
                "role_total": int(acc["L"]),
                "counts_complete": not extra["incomplete_runs"],
                "has_line_reference": bool(extra["frames_with_line_ref"]),
                "off_road_frac": (None if not n_cand
                                  else round(extra["off_road"] / n_cand, 4)),
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
            "path": str(path),
            # 模型身份与真实设备（方案 §8.1：路径/hash/device 都要留档）
            "sha256_16": file_sha16(path),
            "device": (sorted(devices)[0] if len(devices) == 1
                       else (sorted(devices) if devices else "unknown")),
            "device_requested": args.device or "default(Segmenter)",
            "per_scene": per_scene,
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
        _inc = sorted({r for v in per_scene.values()
                       for r in v["incomplete_runs"]})
        if _inc:
            print(f"[calib] 注意：{len(_inc)} 个目录计数不完整（有帧被跳过/报错）："
                  f"{[Path(x).name for x in _inc][:3]}")
        out["models"][name]["incomplete_runs"] = _inc
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
