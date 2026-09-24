"""逐位比较两个 checkpoint：T14 阶段 B"续训≈未中断"的验收入口。

用来分清两类差异：
* **续训漏状态**（要修）：未中断与"中途续训"的权重差，与同配置两次未中断
  的差异不同量级；
* **GPU 进程级非确定性**（要如实报告）：同一配置两次运行本来就不逐位相同。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_seg_checkpoint_diff.py `
            --a logs\\experiments\\run\\resumeA\\checkpoint_last.pt `
            --b logs\\experiments\\run\\resumeB\\checkpoint_last.pt `
            --atol 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.checkpoint import (  # noqa: E402
    load_full, missing_extras, weights_equal,
)


def compare(a: Path, b: Path, *, atol: float = 0.0) -> dict:
    ca, cb = load_full(a), load_full(b)
    eq = weights_equal(ca["state_dict"], cb["state_dict"], atol=atol)
    return {
        "a": str(a), "b": str(b), "atol": atol,
        "a_missing_extras": missing_extras(ca),
        "b_missing_extras": missing_extras(cb),
        "a_dataset_id": ca.get("dataset_id"),
        "b_dataset_id": cb.get("dataset_id"),
        "a_next_epoch": ca.get("next_epoch"),
        "b_next_epoch": cb.get("next_epoch"),
        "weights": eq,
        "equal": eq["n_diff"] == 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="两个 checkpoint 的逐位比较")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--atol", type=float, default=0.0,
                    help="允许的最大绝对差；0 = 要求逐位相同")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rep = compare(Path(args.a), Path(args.b), atol=args.atol)
    print(f"[ckpt-diff] A next_epoch={rep['a_next_epoch']} "
          f"dataset_id={rep['a_dataset_id']} missing={rep['a_missing_extras']}")
    print(f"[ckpt-diff] B next_epoch={rep['b_next_epoch']} "
          f"dataset_id={rep['b_dataset_id']} missing={rep['b_missing_extras']}")
    w = rep["weights"]
    print(f"[ckpt-diff] compared {w['n_compared']} tensors, "
          f"n_diff={w['n_diff']} max_abs_diff={w['max_abs_diff']:.3e} "
          f"worst={w['max_key']}")
    print(f"[ckpt-diff] verdict: "
          f"{'EQUAL' if rep['equal'] else 'DIFFERENT'} (atol={args.atol:g})")
    if args.json:
        Path(args.json).write_text(
            json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if rep["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
