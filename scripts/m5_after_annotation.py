"""标注完成后的下一步：就绪核对 -> 生成合格版 E1 与可晋级线训练的命令。

背景（方案 v2 §S6）：合格版 E1 需要**评价集之外**的 verified 负例；可晋级的线通道
训练需要评价集之外的 verified **正例**。两者都只能来自人工标注（`m5_annotate_pkg`
包 + `scripts/annotate_pkg.ps1`）。本脚本把"标注完了接下来怎么跑"固化成一条命令：

1. **先核对就绪**（复用 `m5_annotation_readiness.check_dir`）：任一目录未就绪就
   **拒绝出命令**（rc=3），逐条打印原因——不带着未标注的包去跑训练；
2. 就绪后按目录的**实际标的类别**分工：`negative_pack` 进 E1 的负例臂，
   `positive_pack`/`mixed` 进线通道训练；
3. 生成两份冻结配置 + 提议文件（写到 `--out-dir`），并打印可直接粘贴的
   `rounds` 命令；**默认只打印不执行**（`--execute` 才跑 E1）。

冻结口径（写死在生成的配置里，跑前可审）：两臂共享起始权重/划分/优化器/步数，
只改一个因子；E1 用 `run_weights`（负例臂 20%），线通道训练用 `add_runs`；
两臂都**不加** `--research-arm`（因为标签是 human_revision → 可晋级）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_after_annotation.py \\
        --negative-dir logs/experiments/annotate_pkg_e1_jv_20260927/front_main \\
        --positive-dir logs/experiments/annotate_pkg_e2_it3_20260927/front_main \\
        --out-dir logs/experiments/t14_after_annotation_20260927
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

#: 训练侧的既有配方（与研究臂 E2/E1 完全一致：同样的 town 组、同样的步数预算）
TOWN = "logs/m5_seg/line_truth_agent_full_20260925/town/front_main"
DEV = ["logs/m5_seg/line_truth_agent_full_20260925/wide/front_main",
       "logs/m5_seg/line_truth_agent_full_20260925/plain/front_main"]


def _norm(p: str) -> str:
    return str(p).replace("\\", "/")


def build(out_dir: Path, *, negative_dirs: list, positive_dirs: list) -> dict:
    """生成两份配置 + 提议；返回 ``{configs, proposals, commands, notes}``。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    neg = [_norm(d) for d in negative_dirs]
    pos = [_norm(d) for d in positive_dirs]
    notes: list = []
    configs: dict = {}
    proposals: dict = {}
    commands: dict = {}

    paint_sources = [f"{TOWN}=agent_revision"] + [f"{d}=human_revision"
                                                  for d in neg + pos]
    for d in DEV:
        paint_sources.append(f"{d}=agent_revision")

    if neg:
        cfg = {
            "_note": ("合格版 E1（方案 v2 §S6）：verified 负例替换 20% 采样额度。"
                      "两臂共享起始权重/划分/优化器/步数，只改 run_weights；"
                      "不加 --research-arm（负例是 human_revision -> 可晋级）。"
                      "判定规则见 docs/T14_S6_EXPERIMENT_DESIGN_20260926.md §3/§8。"),
            "collect": "off", "daily_gpu_minutes": 0.0, "max_wall_minutes": 120.0,
            "window_start_hour": 0, "window_end_hour": 24,
            "pause_while_user_active": False,
            "min_free_vram_mb": 2048.0, "min_free_disk_gb": 10.0,
            "max_candidates": 3, "max_rounds_without_gain": 2,
            "seeds": [42, 43, 44], "dry_run": False, "python": "",
            "runs": [TOWN] + neg, "baseline_runs": [TOWN], "eval_runs": DEV,
            "proposals": str(out_dir / "proposal_e1_promotable.json"),
            "rounds": 1, "epochs": 24, "batch": 4, "lr": 0.001,
            "allow_road_only": False, "equal_steps": True,
            "trainer_script": "m5_train_seg.py", "paint_sources": paint_sources,
        }
        weights = {"0": round(1.0 - 0.2, 4)}
        per = round(0.2 / len(neg), 4)
        for i in range(len(neg)):
            weights[str(i + 1)] = per
        prop = {"_note": ("E1 因子：数据组成（verified 负例占 20% 采样额度）。"
                          "baseline 臂 = 固定配方（只含 town）。"),
                "proposals": [{"candidate_id": "e1-neg20-promotable",
                               "family": "scene_mix",
                               "factor": {"run_weights": weights}}]}
        configs["e1"] = out_dir / "loop_config_e1_promotable.json"
        proposals["e1"] = out_dir / "proposal_e1_promotable.json"
        configs["e1"].write_text(json.dumps(cfg, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
        proposals["e1"].write_text(json.dumps(prop, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
        commands["e1"] = (
            f".venv/Scripts/python.exe scripts/m5_seg_autoloop.py run "
            f"--run-id t14_e1_promotable_20260927 "
            f"--config {_norm(configs['e1'])} --no-dry-run")
    else:
        notes.append("没有负例包：跳过合格版 E1（需要 verified 负例）")

    if pos:
        cfg = {
            "_note": ("可晋级线通道训练（方案 v2 §S6/S7 前置）：把评价集之外的"
                      "verified 正例加进训练输入。两臂共享步数（equal_steps），"
                      "只改训练数据组成；不加 --research-arm。"),
            "collect": "off", "daily_gpu_minutes": 0.0, "max_wall_minutes": 120.0,
            "window_start_hour": 0, "window_end_hour": 24,
            "pause_while_user_active": False,
            "min_free_vram_mb": 2048.0, "min_free_disk_gb": 10.0,
            "max_candidates": 3, "max_rounds_without_gain": 2,
            "seeds": [42, 43, 44], "dry_run": False, "python": "",
            "runs": [TOWN] + pos, "baseline_runs": [TOWN], "eval_runs": DEV,
            "proposals": str(out_dir / "proposal_line_promotable.json"),
            "rounds": 1, "epochs": 24, "batch": 4, "lr": 0.001,
            "allow_road_only": False, "equal_steps": True,
            "trainer_script": "m5_train_seg.py", "paint_sources": paint_sources,
        }
        prop = {"_note": "线通道因子：把 verified 正例组加进训练输入（数据组成）。",
                "proposals": [{"candidate_id": "line-add-verified",
                               "family": "scene_mix",
                               "factor": {"add_runs": pos}}]}
        configs["line"] = out_dir / "loop_config_line_promotable.json"
        proposals["line"] = out_dir / "proposal_line_promotable.json"
        configs["line"].write_text(json.dumps(cfg, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
        proposals["line"].write_text(json.dumps(prop, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
        commands["line"] = (
            f".venv/Scripts/python.exe scripts/m5_seg_autoloop.py run "
            f"--run-id t14_line_promotable_20260927 "
            f"--config {_norm(configs['line'])} --no-dry-run")
    else:
        notes.append("没有正例包：跳过可晋级线通道训练（需要 verified 正例）")

    return {"configs": {k: str(v) for k, v in configs.items()},
            "proposals": {k: str(v) for k, v in proposals.items()},
            "commands": commands, "notes": notes,
            "negative_dirs": neg, "positive_dirs": pos}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--negative-dir", action="append", default=[],
                    help="标注完成的**负例**视角目录（可多次）")
    ap.add_argument("--positive-dir", action="append", default=[],
                    help="标注完成的**正例**视角目录（可多次）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--execute", action="store_true",
                    help="生成后直接执行 E1（默认只打印命令）")
    args = ap.parse_args(argv)

    from m5_annotation_readiness import check_dir

    reports = [check_dir(Path(d)) for d in args.negative_dir + args.positive_dir]
    not_ready = [r for r in reports if not r["ok"]]
    for r in reports:
        print(f"[after] {'就绪' if r['ok'] else '未就绪'} {r['dir']}"
              + (f"（{r.get('kind')}）" if r.get("kind") else ""))
        for x in r["reasons"]:
            print(f"[after]   阻止：{x}")
    if not_ready:
        print(f"[after] {len(not_ready)} 个目录未就绪：拒绝生成训练命令"
              f"（不带未标注的包去跑训练）")
        return 3
    # 路径比较统一成 posix 形式：输入常用 `/`，而 `str(Path)` 在 Windows 上给
    # `\`——直接 `in` 会判不相等（实测踩到：明明 negative_pack 却被判"含线帧"）
    _want_neg = {_norm(d) for d in args.negative_dir}
    _want_pos = {_norm(d) for d in args.positive_dir}
    neg = [r["dir"] for r in reports
           if _norm(r["dir"]) in _want_neg and r.get("kind") == "negative_pack"]
    pos = [r["dir"] for r in reports
           if _norm(r["dir"]) in _want_pos
           and r.get("kind") in ("positive_pack", "mixed")]
    if len(neg) != len(_want_neg):
        print("[after] 有 --negative-dir 判成非纯负例（含线帧）：请按实际用途决定，"
              "脚本不替你改口径")
        return 3
    plan = build(Path(args.out_dir), negative_dirs=neg, positive_dirs=pos)
    for n in plan["notes"]:
        print(f"[after] 提示：{n}")
    for k, c in plan["configs"].items():
        print(f"[after] 配置({k}) -> {c}")
    for k, cmd in plan["commands"].items():
        print(f"[after] 命令({k})：\n  {cmd}")
    out = Path(args.out_dir) / "after_annotation_plan.json"
    out.write_text(json.dumps(plan, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[after] -> {out}")
    if args.execute and "e1" in plan["commands"]:
        import subprocess
        print("[after] 执行 E1：")
        r = subprocess.run(plan["commands"]["e1"].split(), cwd=str(ROOT))
        return int(r.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
