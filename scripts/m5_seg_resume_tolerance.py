"""量化"GPU 上续训 ≈ 未中断"的容差：给冻结协议提供有出处的数字。

背景（T14 §4 与 `docs/T14_PROGRESS_20260924.md` 结论 3）：CUDA 上**没有**
`nll_loss2d` 的确定性实现（本项目交叉熵就用它），所以逐位相等在 GPU 上做不到；
能做的是量出"同一配置两次运行"的固有噪声，和"中断续训 vs 未中断"的差异，
用**同一条噪声基线**判断续训是否落在容差内。

本工具跑三组：

* ``control``  —— 同配置（同 seed、同 epoch 数）跑两次未中断 → 进程级噪声基线；
* ``resume``   —— 同配置跑到 ``--stop-after`` 再续训到同样的总轮数 → 待检差异；
* 两者都用**同一条** ``--epochs``（改 ``--epochs`` 是换 LR 计划，不是中断）。

输出：每对的 ``max_abs_diff`` / ``n_diff`` / 相对尺度（max|w|），以及分布摘要。
判据（写进 ``docs/t14_thresholds*.json`` 的那条）：续训差异的**最大值**不得超过
控制组噪声最大值的 ``--factor`` 倍（默认 3）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.experiments.checkpoint import (  # noqa: E402
    load_full, weights_equal,
)


def weight_scale(state: dict) -> float:
    """权重尺度：用来把绝对差换算成相对差（不同量纲的层可以一起比）。"""
    worst = 0.0
    for t in state.values():
        try:
            worst = max(worst, float(t.detach().abs().max()))
        except Exception:                    # noqa: BLE001
            continue
    return worst or 1.0


def pair_diff(a_path: Path, b_path: Path) -> dict:
    """一对 checkpoint 的差异（含相对尺度）。"""
    ca, cb = load_full(a_path), load_full(b_path)
    eq = weights_equal(ca["state_dict"], cb["state_dict"], atol=0.0)
    scale = max(weight_scale(ca["state_dict"]), weight_scale(cb["state_dict"]))
    return {"a": a_path.name, "b": b_path.name,
            "n_diff": eq["n_diff"], "n_compared": eq["n_compared"],
            "max_abs_diff": eq["max_abs_diff"], "worst_key": eq["max_key"],
            "weight_scale": scale,
            "max_rel_diff": (eq["max_abs_diff"] / scale) if scale else None}


def summarize(pairs: dict) -> dict:
    """把 ``{组名: [pair_diff, ...]}`` 汇总成可写进协议的容差。

    纯函数（不读盘、不训练），所以单测可以直接喂构造的差异。判据是
    "续训组的最大值不超过控制组最大值的 ``factor`` 倍"——两组用同一批
    seed、同一批超参，差异才可比。
    """
    out: dict = {}
    for name, rows in (pairs or {}).items():
        diffs = [float(r["max_abs_diff"]) for r in rows if r]
        rels = [float(r["max_rel_diff"]) for r in rows
                if r.get("max_rel_diff") is not None]
        if not diffs:
            out[name] = {"n": 0, "missing": "no pairs measured"}
            continue
        out[name] = {
            "n": len(diffs),
            "max_abs_diff_max": max(diffs),
            "max_abs_diff_median": statistics.median(diffs),
            "max_abs_diff_min": min(diffs),
            "max_rel_diff_max": (max(rels) if rels else None),
            "per_pair": [round(d, 6) for d in diffs],
            "n_diff_frac_max": max(
                float(r["n_diff"]) / max(1, int(r["n_compared"]))
                for r in rows if r),
        }
    if "control" in out and "resume" in out and out["control"].get("n") \
            and out["resume"].get("n"):
        out["tolerance"] = {
            "basis": "control noise (same config, two runs) vs resume vs "
                     "uninterrupted, same seeds and epochs",
            "control_max_abs_diff": out["control"]["max_abs_diff_max"],
            "resume_max_abs_diff": out["resume"]["max_abs_diff_max"],
            "ratio": (out["resume"]["max_abs_diff_max"]
                      / max(1e-12, out["control"]["max_abs_diff_max"])),
        }
    return out


def _train(runs: Path, out: Path, args, seed: int, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_train_seg.py"),
           "--runs", str(runs), "--split", "tail", "--val-frac", "0.2",
           "--batch", str(args.batch), "--lr", str(args.lr), "--seed", str(seed),
           "--epochs", str(args.epochs), "--device", args.device,
           "--save-every-epoch",
           "--out", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=max(600, args.timeout_s))
    if r.returncode != 0:
        raise RuntimeError(f"训练失败 rc={r.returncode}\n{r.stdout[-1200:]}\n"
                           f"{r.stderr[-800:]}")


def measure(runs: list, *, seeds: list, epochs: int, stop_after: int,
            device: str, batch: int, lr: float, work: Path,
            timeout_s: int = 3600, log=print) -> dict:
    """跑控制组与续训组，返回 ``{"pairs": {...}, "summary": {...}}``。"""
    pairs: dict = {"control": [], "resume": []}
    work.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(batch=batch, lr=lr, epochs=epochs, device=device,
                            timeout_s=timeout_s)
    for seed in seeds:
        a = work / f"s{seed}_A"
        a2 = work / f"s{seed}_A2"
        bh = work / f"s{seed}_Bhalf"
        br = work / f"s{seed}_B"
        log(f"[tol] seed {seed}: 控制组 A / A2 各 {epochs} 轮（{device}）")
        _train(Path(runs[0]) if len(runs) == 1 else runs, a, ns, seed)
        _train(Path(runs[0]) if len(runs) == 1 else runs, a2, ns, seed)
        log(f"[tol] seed {seed}: 续训组 {stop_after} 轮后中断 + 续训到 {epochs} 轮")
        _train(Path(runs[0]) if len(runs) == 1 else runs, bh, ns, seed,
               "--stop-after", str(stop_after))
        _train(Path(runs[0]) if len(runs) == 1 else runs, br, ns, seed,
               "--resume", str(bh / "checkpoint_last.pt"))
        pairs["control"].append(pair_diff(a / "checkpoint_last.pt",
                                          a2 / "checkpoint_last.pt"))
        pairs["resume"].append(pair_diff(a / "checkpoint_last.pt",
                                         br / "checkpoint_last.pt"))
        c, r = pairs["control"][-1], pairs["resume"][-1]
        log(f"[tol] seed {seed}: control max={c['max_abs_diff']:.3e} "
            f"| resume max={r['max_abs_diff']:.3e}")
    return {"pairs": pairs, "summary": summarize(pairs),
            "config": {"seeds": list(seeds), "epochs": epochs,
                       "stop_after": stop_after, "device": device,
                       "batch": batch, "lr": lr, "runs": [str(r) for r in runs]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU 续训容差量化")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="默认 epochs-1")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--work", default=None,
                    help="工作目录（默认 logs/experiments/t14_gpu_tol）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    stop = args.stop_after if args.stop_after is not None else max(1,
                                                                  args.epochs - 1)
    work = Path(args.work) if args.work else (
        Path(config.LOGS_DIR) / "experiments" / "t14_gpu_tol")
    rep = measure(args.runs, seeds=args.seeds, epochs=args.epochs,
                  stop_after=stop, device=args.device, batch=args.batch,
                  lr=args.lr, work=work)
    s = rep["summary"]
    print(f"[tol] control: max={s['control']['max_abs_diff_max']:.3e} "
          f"median={s['control']['max_abs_diff_median']:.3e} "
          f"n={s['control']['n']}")
    print(f"[tol] resume : max={s['resume']['max_abs_diff_max']:.3e} "
          f"median={s['resume']['max_abs_diff_median']:.3e} "
          f"n={s['resume']['n']}")
    print(f"[tol] 比值 resume/control = {s['tolerance']['ratio']:.2f}")
    out = Path(args.json) if args.json else work / "tolerance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[tol] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
