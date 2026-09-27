"""训练真伪证据：权重差异 / 步数 / loss 曲线 / 逐 seed 独立性（方案 §S6 要求）。

方案原文："看板必须能证明真实反向传播、optimizer step 和权重更新，**不能以生成了
目录证明学习发生**"。本脚本对一次 `rounds` 运行的产物做四项核对，并落 JSON：

1. **两臂权重确实不同**：逐 seed 比 `baseline/seedN/checkpoint_last.pt` 与
   `round0/seedN/checkpoint_last.pt` 的平均绝对差与"变化张量占比"——两臂若只差
   一个开关，权重仍必须真的不同（复制/未训练会露馅）；
2. **步数可追溯**：从 checkpoint 的 `train_args` 读 `n_train/epochs/batch`，
   与判定文件里的 `steps_by_arm` 对照（等步数设计的证据）；
3. **loss 真的在降**：`train_hist.json`（列式或行式两种写法都吃）首末 epoch；
4. **逐 seed 独立**：seed42 vs seed43 的基线权重必须不同（同 seed 复跑才该相同）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_training_authenticity.py \\
        --run-id t14_e2_direction_v3_20260926 --run-id t14_e1_research_20260927 \\
        --out logs/experiments/train_authenticity_20260927.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_state(path: Path) -> tuple[dict, dict]:
    """``(state_dict, whole_ckpt)``；兼容 ``model``/``state_dict``/``net`` 三种包装。"""
    import torch
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "net"):
            if isinstance(ck.get(k), dict):
                return ck[k], ck
    return ck if isinstance(ck, dict) else {}, ck if isinstance(ck, dict) else {}


def weight_diff(a: Path, b: Path) -> dict:
    """两套权重的差异：平均绝对差 + 变化张量占比（阈值 1e-9）。"""
    import torch
    sa, _ = load_state(a)
    sb, _ = load_state(b)
    total, changed, n = 0.0, 0, 0
    for k in sa:
        if k not in sb:
            continue
        ta, tb = sa[k].float(), sb[k].float()
        if ta.shape != tb.shape:
            continue
        d = (ta - tb).abs()
        total += float(d.mean())
        changed += int(float(d.max()) > 1e-9)
        n += 1
    return {"mean_abs_diff": (total / n if n else None),
            "changed_tensors": changed, "n_tensors": n}


def loss_curve(path: Path) -> dict:
    """``train_hist.json`` 的首末 train_loss（列式 dict 与行式 list 都支持）。"""
    if not path.is_file():
        return {"status": "missing"}
    h = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(h, dict) and isinstance(h.get("train_loss"), list):
        tl = [v for v in h["train_loss"] if v is not None]
    else:
        rows = h if isinstance(h, list) else (h.get("epochs") or h.get("history") or [])
        tl = [e.get("train_loss") for e in rows
              if isinstance(e, dict) and e.get("train_loss") is not None]
    if len(tl) < 2:
        return {"status": "too_short", "n": len(tl)}
    drop = (None if not tl[0] else round(100.0 * (1 - tl[-1] / tl[0]), 2))
    return {"status": "ok", "n": len(tl), "first": tl[0], "last": tl[-1],
            "drop_pct": drop}


def audit_run(run_dir: Path, seeds: tuple = (42, 43, 44)) -> dict:
    out: dict = {"run_id": run_dir.name, "seeds": {}}
    for seed in seeds:
        b = run_dir / "baseline" / f"seed{seed}" / "checkpoint_last.pt"
        c = run_dir / "round0" / f"seed{seed}" / "checkpoint_last.pt"
        entry: dict = {"baseline_ckpt": b.is_file(), "candidate_ckpt": c.is_file()}
        if b.is_file() and c.is_file():
            entry["arms_weight_diff"] = weight_diff(b, c)
        if b.is_file():
            _, ck = load_state(b)
            ta = (ck or {}).get("train_args") or {}
            entry["train_args"] = {k: ta.get(k) for k in
                                   ("n_train", "epochs", "batch", "lr", "seed")}
            if ta.get("n_train") and ta.get("batch") and ta.get("epochs"):
                entry["steps_expected"] = (int(ta["n_train"]) // max(1, int(ta["batch"]))
                                           * int(ta["epochs"]))
        entry["baseline_loss"] = loss_curve(
            run_dir / "baseline" / f"seed{seed}" / "train_hist.json")
        out["seeds"][str(seed)] = entry
    b42 = run_dir / "baseline" / "seed42" / "checkpoint_last.pt"
    b43 = run_dir / "baseline" / "seed43" / "checkpoint_last.pt"
    if b42.is_file() and b43.is_file():
        out["seed_independence"] = weight_diff(b42, b43)
    decs = sorted(run_dir.glob("decision_*.json"))
    if decs:
        blob = json.loads(decs[0].read_text(encoding="utf-8"))
        out["steps_by_arm"] = blob.get("steps_by_arm")
        out["decision"] = (blob.get("decision") or {}).get("decision")
        out["round_outcome"] = blob.get("round_outcome")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", action="append", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    reports = []
    for rid in args.run_id:
        run_dir = ROOT / "logs" / "experiments" / rid
        if not run_dir.is_dir():
            print(f"[auth] 缺 run 目录：{run_dir}")
            continue
        rep = audit_run(run_dir)
        reports.append(rep)
        print(f"=== {rep['run_id']} ===")
        for seed, e in sorted(rep["seeds"].items()):
            d = e.get("arms_weight_diff") or {}
            loss = e.get("baseline_loss") or {}
            print(f"  seed{seed}: 两臂权重差 {d.get('mean_abs_diff')}"
                  f"（变化 {d.get('changed_tensors')}/{d.get('n_tensors')}）"
                  f" | steps≈{e.get('steps_expected')}"
                  f" | loss {loss.get('first')}->{loss.get('last')}"
                  f"（降 {loss.get('drop_pct')}%）")
        si = rep.get("seed_independence") or {}
        print(f"  逐 seed 独立：seed42 vs seed43 权重差 {si.get('mean_abs_diff')}")
        print(f"  判定 {rep.get('decision')} / 归因 {rep.get('round_outcome')}"
              f" | steps_by_arm={rep.get('steps_by_arm')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(reports, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[auth] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
