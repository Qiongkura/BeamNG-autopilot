"""最终确认程序：封存最终集、登记访问（方案 §7/§10.3）。

只有本程序（`purpose=final_confirm`）被允许读最终集；搜索/训练入口读到带封存文件的
目录会**拒训**（见 `m5_seg_autoloop._rounds_audit` 的 final-set 检查）。

用法::

    # 封存（冻结候选前做一次；此后内容不可变）
    .venv\\Scripts\\python.exe scripts\\m5_final_set.py seal --name <名字> \\
        --dataset-id <数据集哈希> --protocol-hash <协议哈希> --out <封存目录> \\
        --frame <帧路径> [--frame ...] [--group <组键> ...]

    # 状态（封存内容是否仍然一致 + 消费账）
    .venv\\Scripts\\python.exe scripts\\m5_final_set.py status --out <封存目录>

    # 确认访问（一次即消费；被拒也留痕）
    .venv\\Scripts\\python.exe scripts\\m5_final_set.py access --out <封存目录> \\
        --protocol-hash <协议哈希> --candidate-id <候选> --caller <调用方> \\
        [--result <结果>]

    # 最终确认（**唯一**被允许读最终集的路径）：访问 -> 在最终集上评估 -> 写确认记录
    .venv\Scripts\python.exe scripts\m5_final_set.py confirm --out <封存目录> --dataset <最终集帧目录> --candidate-id <候选> --model <checkpoint> [--record <确认记录.json>] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.checkpoint import file_sha16  # noqa: E402
from beamng_autopilot.experiments.final_set import (  # noqa: E402
    access, confirmation_record, consumption, read_seal, seal, verify_seal,
)
from beamng_autopilot.experiments.protocol import protocol_hash  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal", help="封存最终集（内容此后不可变）")
    s.add_argument("--name", required=True)
    s.add_argument("--dataset-id", required=True)
    s.add_argument("--protocol-hash", default=None,
                   help="缺省用当前协议的哈希（冻结口径）")
    s.add_argument("--out", required=True)
    s.add_argument("--frame", action="append", default=[])
    s.add_argument("--group", action="append", default=[])
    s.add_argument("--sealed-by", default="")

    s = sub.add_parser("status", help="封存一致性与消费账")
    s.add_argument("--out", required=True)

    s = sub.add_parser("confirm", help="最终确认：访问最终集并评估候选（会消费）")
    s.add_argument("--out", required=True, help="封存目录（seal 的输出）")
    s.add_argument("--dataset", nargs="+", required=True,
                   help="最终集帧目录（必须与封存时的帧一致）")
    s.add_argument("--candidate-id", required=True)
    s.add_argument("--model", required=True, help="候选 checkpoint")
    s.add_argument("--caller", default="m5_final_set.py confirm")
    s.add_argument("--protocol-hash", default=None)
    s.add_argument("--record", default=None,
                   help="确认记录输出路径（缺省 <out>/confirmation_<候选>.json）")
    s.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    s.add_argument("--purpose", default="final_confirm")

    s = sub.add_parser("access", help="登记一次确认访问")
    s.add_argument("--out", required=True)
    s.add_argument("--protocol-hash", default=None)
    s.add_argument("--candidate-id", required=True)
    s.add_argument("--caller", required=True)
    s.add_argument("--purpose", default="final_confirm")
    s.add_argument("--result", default="")
    args = ap.parse_args(argv)

    if args.cmd == "seal":
        rec = seal(args.frame, name=args.name, dataset_id=args.dataset_id,
                   protocol_hash=(args.protocol_hash or protocol_hash()),
                   out_dir=args.out, sealed_by=args.sealed_by,
                   groups=args.group)
        print(f"[final-set] 已封存 {rec['n_frames']} 帧 -> {args.out} "
              f"(digest {rec['digest']}, 协议 {rec['protocol_hash']})")
        return 0
    if args.cmd == "status":
        ver = verify_seal(args.out)
        cons = consumption(args.out)
        seal_rec = read_seal(args.out)
        if seal_rec is None:
            print(f"[final-set] {args.out} 未封存：这不是最终集")
            return 3
        print(f"[final-set] {seal_rec['name']} 帧={seal_rec['n_frames']} "
              f"digest={seal_rec['digest']} 内容一致={ver['ok']} "
              f"协议={seal_rec['protocol_hash']}")
        if not ver["ok"]:
            print(f"[final-set] 不一致：{ver['reasons']} 变更={ver['changed']}")
        print(f"[final-set] 消费账：确认 {cons['confirmed']} / 被拒 "
              f"{cons['refused']} / 已消费 {cons['consumed']}")
        return 0 if ver["ok"] else 1
    if args.cmd == "confirm":
        return _cmd_confirm(args)
    got = access(args.out, protocol_hash=(args.protocol_hash or protocol_hash()),
                 candidate_id=args.candidate_id, caller=args.caller,
                 purpose=args.purpose, result=args.result)
    print(json.dumps(got, ensure_ascii=False, indent=1))
    return 0 if got["allowed"] else 3


def _cmd_confirm(args) -> int:
    """访问（一次即消费）-> 在最终集上评估 -> 写确认记录。

    访问被拒时**不评估**（没被放行就不该读数据），rc=3；被拒也会留痕。
    记录里写清"测了哪些口径"，没测的（候选身份/左右角色探针）留 UNKNOWN
    并写进 notes——最终确认不允许拿一部分口径冒充整套（方案 §10.1）。
    """
    ph = args.protocol_hash or protocol_hash()
    got = access(args.out, protocol_hash=ph, candidate_id=args.candidate_id,
                 caller=args.caller, purpose=args.purpose,
                 result=f"model={Path(args.model).name}")
    if not got["allowed"]:
        print(json.dumps(got, ensure_ascii=False, indent=1))
        print("[final-set] 拒绝访问：不评估最终集（被拒也留痕）")
        return 3
    import m5_seg_eval_matrix as em
    from beamng_autopilot.experiments.manifest import dir_group
    by: dict = {}
    for r in args.dataset:
        by.setdefault(dir_group(r), []).extend(em.load_frames([Path(r)]))
    metrics = em.evaluate_model_per_group(Path(args.model), by,
                                          device=args.device)
    keep = ("line_recall", "line_precision", "offroad_false_frac_of_pred",
            "line_iou", "road_iou", "n_frames", "inference_ms_p95")
    per_group = {g: {k: m.get(k) for k in keep}
                 for g, m in (metrics.get("per_group") or {}).items()}
    results = {
        "per_group": per_group,
        "overall": {k: metrics.get(k) for k in keep},
        "model": str(args.model),
        "model_sha16": file_sha16(Path(args.model)),
    }
    rec = confirmation_record(
        seal_dir=args.out, protocol_hash=ph,
        candidate_id=args.candidate_id, caller=args.caller,
        model_sha16=results["model_sha16"], results=results,
        notes=["候选身份/左右角色/参考覆盖探针未在最终集上运行：这些口径在本次"
               "确认里是 UNKNOWN，不能声称已确认（方案 §10.1）"])
    out_rec = Path(args.record) if args.record else (
        Path(args.out) / f"confirmation_{args.candidate_id}.json")
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    out_rec.write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[final-set] 已确认 {args.candidate_id}（协议 {ph}，权重 "
          f"{results['model_sha16']}）-> {out_rec}")
    for g, m in per_group.items():
        print(f"[final-set]   {g}: line_recall={m.get('line_recall')} "
              f"line_precision={m.get('line_precision')} n={m.get('n_frames')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
