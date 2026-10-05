"""候选集口径扫描（离线，不训练）：横向门 / 并行重复合并的收益-代价表。

为什么要有它（R3 第二轮的归因，`docs/T16_R3_ACCEPTANCE_20261004.md` §8.4）：
v7 身份率 0.60 门不达标的剩余项在**候选集本身**——远场路外尾带（未匹配 64%
在 |lat| ≥ 5 m）与路面幻影（|lat| < 1.2 m 的 solid 候选，像素在标签的路面类）。
按纪律，协议/候选集变更**先离线扫描收益与代价，再决定是否升版本**（同 v7 合并
的做法）。本脚本就是那个扫描入口：

* ``--scan lateral``：模拟"经典候选 |lat| > L 丢弃"（L 扫描）。代价 = 丢掉的
  匹配候选（宽路真线可能在 ≥5 m，dev 实测有 40 个）——收益/代价必须一起报。
* ``--scan parallel``：模拟"同侧 |Δlat| ≤ d 的候选合并"（d 扫描）。优先保留
  matched 的那个；都未匹配时保留 |lat| 小者（离自车近）。角色一致率应回升。

实现只做**候选集变换 + 计数重算**：变换后调探针自己的 ``frame_counts``
（契约只有一份），再用 ``candidate_metrics`` 累加——比率口径与判定完全一致。
不改任何运行时行为、不训练、不启动游戏。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_candidate_scope_scan.py `
        --arm base6x-s42=logs/experiments/t16_negdose6x_20261001/round0/seed42/checkpoint_last.pt `
        --dev-runs logs/experiments/t16_autotruth_r3_b_20261004/*/front_main `
        --scan lateral --lmax 3.0 4.0 4.5 5.0 6.0 `
        --out logs/experiments/r3b_scan_lateral.json
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


def apply_scope(cands: list, *, mode: str, param: float,
                only_classic: bool = False) -> list:
    """候选集变换（纯函数，可单测）。

    ``mode="lateral"``：丢弃 ``|lat_m| > param`` 的候选；``only_classic=True``
    时只丢**经典通道**（``learned_frac is None``）的，学习通道的不动。
    ``mode="parallel"``：同侧、``|Δlat| ≤ param`` 的候选并为一组，每组保留
    **matched 优先**、否则 |lat| 最小者；被并掉的记进 ``merged_from`` 计数。
    返回新的候选列表（不改原 dict，浅拷贝后加 ``_merged`` 计数）。
    """
    out = []
    if mode == "lateral":
        lim = float(param)
        for c in cands:
            if only_classic and c.get("learned_frac") is not None:
                out.append(c)
                continue
            if abs(float(c.get("lat_m") or 0.0)) > lim:
                continue
            out.append(c)
        return out
    if mode == "parallel":
        d = float(param)
        # 按侧分组：同侧才可能是同一条线的重复
        used = [False] * len(cands)
        order = sorted(range(len(cands)),
                       key=lambda i: (float(cands[i].get("lat_m") or 0.0) < 0,
                                      abs(float(cands[i].get("lat_m") or 0.0))))
        for i in order:
            if used[i]:
                continue
            ci = cands[i]
            li = float(ci.get("lat_m") or 0.0)
            group = [i]
            for j in order:
                if used[j] or j == i:
                    continue
                lj = float(cands[j].get("lat_m") or 0.0)
                if (li > 0) != (lj > 0):
                    continue
                if abs(li - lj) <= d:
                    group.append(j)
            keep = max(group, key=lambda k: (
                bool(cands[k].get("matched")),
                -abs(float(cands[k].get("lat_m") or 0.0))))
            for k in group:
                used[k] = True
            c = dict(cands[keep])
            if len(group) > 1:
                c["_merged"] = len(group) - 1
            out.append(c)
        return out
    raise ValueError(f"未知 scan mode {mode!r}")


def _probe():
    spec = importlib.util.spec_from_file_location(
        "m5_marking_identity_probe", ROOT / "scripts"
        / "m5_marking_identity_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_marking_identity_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_rows(ip, arm_spec: str, dirs: list, device: str) -> list:
    name, _, ck = str(arm_spec).partition("=")
    if not ck:
        raise SystemExit(f"--arm 需要 NAME=CHECKPOINT，收到 {arm_spec!r}")
    rows = []
    for d in dirs:
        meta_p = d / "meta.json"
        if not meta_p.is_file() and (d.parent / "meta.json").is_file():
            meta_p = d.parent / "meta.json"
        meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                if meta_p.is_file() else None)
        res = ip.probe(d, meta, view=d.name, model_path=str(ROOT / ck),
                       device=device)
        for r in (res.get("rows") or []):
            r["_scene"] = d.parent.name
            rows.append(r)
    return rows


def summarize(ip, cm, rows: list, *, mode: str, param: float,
              only_classic: bool = False) -> dict:
    """变换后重算契约计数与比率（与判定同口径：只累加整数再算比率）。"""
    acc = cm.empty()
    merged = kept = 0
    for r in rows:
        cands = r.get("candidates") or []
        in_p = bool((r.get("counts") or {}).get("P_frames"))
        c2 = apply_scope(cands, mode=mode, param=param,
                         only_classic=only_classic)
        merged += sum(int(c.get("_merged") or 0) for c in c2)
        kept += len(c2)
        cm.accumulate(acc, ip.frame_counts(c2, in_p=in_p))
    rt = cm.ratios(acc)
    return {"mode": mode, "param": param, "only_classic": only_classic,
            "C": int(acc["C"]), "R": int(acc["R"]), "M": int(acc["M"]),
            "L": int(acc["L"]), "A": int(acc["A"]),
            "candidate_reference_coverage": rt["candidate_reference_coverage"],
            "candidate_identity_rate": rt["candidate_identity_rate"],
            "left_right_role_agreement": rt["left_right_role_agreement"],
            "kept_candidates": kept, "merged_away": merged}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, metavar="NAME=CHECKPOINT")
    ap.add_argument("--dev-runs", nargs="+", required=True)
    ap.add_argument("--scan", choices=("lateral", "parallel"), required=True)
    ap.add_argument("--lmax", nargs="+", type=float, default=None,
                    help="lateral 扫描的 |lat| 上限列表")
    ap.add_argument("--dmax", nargs="+", type=float, default=None,
                    help="parallel 扫描的 |Δlat| 合并半径列表")
    ap.add_argument("--only-classic", action="store_true",
                    help="lateral 只丢经典通道候选（学习通道不动）")
    ap.add_argument("--baseline", action="store_true", default=True,
                    help="先报 baseline（不变换）行")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = [Path(p) for p in args.dev_runs]
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        raise SystemExit(f"开发集为空：{args.dev_runs}")
    ip = _probe()
    from beamng_autopilot.experiments import candidate_metrics as cm
    rows = _load_rows(ip, args.arm, dirs, args.device)
    if not rows:
        raise SystemExit("没有探针行（帧目录/相机模型有问题？）")
    name = str(args.arm).partition("=")[0]
    results = []
    if args.baseline:
        # baseline 用"不变换"的恒等参数：lateral 取一个不会丢任何候选的极限
        base = summarize(ip, cm, rows, mode="lateral", param=1e9)
        base["mode"] = "baseline"
        results.append(base)
    params = (args.lmax if args.scan == "lateral" else args.dmax) or []
    if not params:
        raise SystemExit(f"--scan {args.scan} 需要 "
                         + ("--lmax" if args.scan == "lateral" else "--dmax"))
    for p_ in params:
        results.append(summarize(ip, cm, rows, mode=args.scan, param=float(p_),
                                 only_classic=bool(args.only_classic)))
    hdr = (f"{'scan':>10} {'param':>7} {'C':>6} {'R':>6} {'M':>5} "
           f"{'覆盖':>7} {'身份':>7} {'角色':>7} {'并掉':>5}")
    print(f"[scan] arm={name} frames={len(rows)} dirs={len(dirs)}")
    print(hdr)
    for r in results:
        print(f"{r['mode']:>10} {r['param']:>7} {r['C']:6d} {r['R']:6d} "
              f"{r['M']:5d} {str(r['candidate_reference_coverage']):>7} "
              f"{str(r['candidate_identity_rate']):>7} "
              f"{str(r['left_right_role_agreement']):>7} {r['merged_away']:5d}")
    out = {"arm": str(args.arm), "dev_runs": [str(d) for d in dirs],
           "frames": len(rows), "scan": args.scan,
           "only_classic": bool(args.only_classic), "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[scan] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
