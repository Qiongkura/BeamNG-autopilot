"""T14 阶段 D 的离线入口：审计 → 提议 → 成对评估 → 硬门槛判定 → 重放。

薄入口：只解析参数、调用 ``beamng_autopilot.experiments`` 里的库件，并把每一步
写进事件流（``logs/experiments/<run_id>/events.jsonl``）。

子命令：

* ``audit``    —— 建不可变数据版本 + 逐通道覆盖；**没有可靠漆线真值时进复核队列**
  （``needs_review``）而不是硬着头皮训练；
* ``propose``  —— 从评估产物分桶错误，给出**一次只改一个因子**的有限提议；
* ``evaluate`` —— 成对 seed 比较 + 事前冻结的硬门槛 → ``rejected`` /
  ``needs_evidence`` / ``shadow_candidate``（绝不覆盖生产模型）；
* ``replay``   —— 用同一批输入重算决策，验证"相同输入 → 相同决策与理由"；
* ``run --once --dry-run`` —— 资源门 + 单实例锁 + 命令清单，**不执行**。

为什么 dry-run 是默认：方案规定"先 ``--once --dry-run`` 验证将执行的命令与
数据流，再允许常驻"，且在两个运行参数（是否允许无人值守启动游戏、空闲窗口/
每日 GPU 上限）确定前不起常驻任务。
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import asdict  # noqa: E402

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot.experiments.controller import (  # noqa: E402
    InstanceLock, LoopConfig, RoundRecord, plan_once, probe_resources,
    resource_gate, should_stop,
)
from beamng_autopilot.experiments.events import Event, EventLog, metric  # noqa: E402
from beamng_autopilot.experiments.gates import (  # noqa: E402
    Thresholds, decide, paired_compare, threshold_violations,
)
from beamng_autopilot.experiments.manifest import DatasetManifest  # noqa: E402
from beamng_autopilot.experiments.proposer import (  # noqa: E402
    bucket_errors, load_eval_artifacts, needs_review, propose,
)

THRESHOLDS_DIR = ROOT / "docs"


def newest_thresholds_file() -> Path | None:
    """取版本号最高的冻结阈值文件（`t14_thresholds[_vN].json`）。"""
    import re
    best, best_v = None, -1
    for f in THRESHOLDS_DIR.glob("t14_thresholds*.json"):
        m = re.search(r"_v(\d+)", f.name)
        v = int(m.group(1)) if m else 1
        if v > best_v:
            best, best_v = f, v
    return best


def exp_dir(run_id: str) -> Path:
    return Path(config.LOGS_DIR) / "experiments" / run_id


def _log(run_id: str) -> EventLog:
    return EventLog(exp_dir(run_id))


def _ev(run_id: str, phase: str, status: str, *, candidate: str = "loop",
        seed: int = 0, dataset: str = "", note: str = "", **kw) -> Event:
    return Event(run_id=run_id, candidate_id=candidate, dataset_id=dataset,
                 config_hash=Thresholds().config_hash, seed=seed, phase=phase,
                 status=status, note=note, **kw)


def thresholds(path: Path | None = None) -> Thresholds:
    """读冻结阈值（默认取版本最高的那份），并**校验文件里的 config_hash**。

    冻结的意义在于"搜到一半不能改阈值"；如果只读值不校验，手改一个数字
    也不会被发现。哈希不符时直接拒绝，让人去开新版本。
    """
    p = Path(path) if path else newest_thresholds_file()
    if p is not None and p.exists():
        blob = json.loads(p.read_text(encoding="utf-8"))
        t = Thresholds(**{k: v for k, v in blob.get("thresholds", {}).items()
                          if k in Thresholds.__dataclass_fields__})
        stored = blob.get("config_hash")
        if stored and stored != t.config_hash:
            raise ValueError(
                f"{p} 里的 config_hash={stored} 与内容算出的 "
                f"{t.config_hash} 不符：阈值文件被改过。冻结协议要求**开一个新"
                f"版本文件**（docs/t14_thresholds_vN.json，N 递增），不能原地改。")
        return t
    return Thresholds()


# ---------------------------------------------------------------------------
def cmd_audit(args) -> int:
    mf = DatasetManifest.build([Path(r) for r in args.runs], root=ROOT,
                               dev_groups=args.dev_group,
                               final_groups=args.final_group)
    audit, coverage = mf.audit(), mf.coverage()
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    mf.save(out / "dataset.json", force=True)
    (out / "coverage.json").write_text(
        json.dumps({"audit": audit, "coverage": coverage, "groups": mf.groups},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    log = _log(args.run_id)
    log.append(_ev(args.run_id, "queued", "start", dataset=mf.dataset_id,
                   note=f"runs={len(args.runs)}"))
    log.append(_ev(args.run_id, "auditing", "started", dataset=mf.dataset_id,
                   note="逐通道覆盖 + 泄漏 + 身份检查"))
    trainable = coverage.get("train", {}).get("trainable_frames", 0)
    paint_ok = coverage.get("train", {}).get("paint_valid_frames", 0)
    if not trainable:
        log.append(_ev(args.run_id, "auditing", "no_trainable_data",
                       dataset=mf.dataset_id,
                       note="没有任何通道有可用真值：停止，不训练"))
        print("[autoloop] 审计失败：没有可用真值，停止")
        return 2
    if not paint_ok and not args.allow_road_only:
        # 方案的准入门：有 Tech annotation 但无可靠标线真值 -> 进复核队列
        log.append(_ev(args.run_id, "needs_review", "paint_truth_missing",
                       dataset=mf.dataset_id,
                       note=("有 Tech annotation 但无可靠标线真值：标线通道被屏蔽，"
                             "需人工修订或单独验证的模拟器真值；"
                             "用 --allow-road-only 可只做路面通道实验")))
        print("[autoloop] 审计：标线真值不可用 -> needs_review（未训练）")
        return 3
    log.append(_ev(args.run_id, "training", "ready", dataset=mf.dataset_id,
                   note=f"trainable={trainable} paint_ok={paint_ok}"))
    print(f"[autoloop] 审计通过：dataset_id={mf.dataset_id[:16]} "
          f"trainable={trainable} paint_ok={paint_ok} "
          f"rejected={audit['n_rejected']}")
    return 0


def cmd_propose(args) -> int:
    artifacts = load_eval_artifacts(args.eval_dir)
    buckets = bucket_errors(pixel=artifacts.get("pixel"),
                            identity=artifacts.get("identity"))
    review = needs_review(buckets)
    history = []
    log = _log(args.run_id)
    for e in log.replay()["events"]:
        if e.phase == "training" and e.status == "ready":
            pass
    dataset = {"n_train_frames": args.n_train_frames}
    champ = {"steps": args.champion_steps, "epochs": args.champion_epochs}
    blocked: list = []
    # 提议里引用的数据组必须真的存在：不存在就忽略并说明，不能让循环拿着
    # 一个不存在的目录去训练（那会在训练器里以另一种方式失败）
    avail: list = []
    missing: list = []
    for r in (args.available_runs or []):
        (avail if Path(r).exists() else missing).append(str(r))
    if missing:
        print(f"[autoloop] --available-runs 里不存在的目录已忽略：{missing}")
        blocked.append({"family": "scene_mix",
                        "why": f"引用了不存在的训练目录：{missing}",
                        "top_bucket": ""})
    props = propose(buckets=buckets, dataset=dataset, champion=champ,
                    max_proposals=args.max_proposals, history=history,
                    available_runs=avail, blocked=blocked)
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    blob = {"buckets": [b.as_dict() for b in buckets],
            "needs_review": review,
            "proposals": [p.as_dict() for p in props],
            "blocked_families": blocked,
            "champion": champ, "dataset": dataset,
            "inputs": {"eval_dir": str(args.eval_dir),
                       "available_runs": avail,
                       "available_runs_missing": missing}}
    (out / "proposals.json").write_text(
        json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[autoloop] 错误桶 {len(buckets)} 个，其中 {len(review)} 类"
          f"缺可信真值 -> 复核队列")
    for b in buckets:
        print(f"  {b.name:26s} n={b.count:6d} rate={b.rate}")
    for p in props:
        print(f"  提议 {p.candidate_id} 族={p.family} 因子={p.factor}")
    for b in blocked:
        print(f"[autoloop] 族 {b['family']} 未产出提议：{b['why']}")
    if review:
        print("[autoloop] 复核队列（不得直接当负例训练）：")
        for r in review:
            print(f"  {r['name']}: {r['count']} -> {r['why'][:70]}")
    (exp_dir(args.run_id) / "proposals.json").write_text(
        json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


def cmd_evaluate(args) -> int:
    """成对 seed 比较 + 冻结硬门槛 -> 判定（不覆盖生产模型）。"""
    pairings = json.loads(Path(args.pairings).read_text(encoding="utf-8"))
    hard = (json.loads(Path(args.hard_gate).read_text(encoding="utf-8"))
            if args.hard_gate else {})
    t = thresholds(Path(args.thresholds) if args.thresholds else None)
    compared = {}
    for name, spec in pairings.items():
        compared[name] = paired_compare(
            name, spec.get("champion", []), spec.get("candidate", []),
            lower_is_better=bool(spec.get("lower_is_better", False)))
    violations = threshold_violations(hard, t)
    missing = [k for k, v in compared.items() if not v.get("n")]
    decision = decide(pairings=compared, thresholds=t,
                      missing_metrics=missing,
                      production_mismatch=bool(args.production_mismatch),
                      definition_drift=bool(args.definition_drift),
                      hard_gate_violations=violations)
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    blob = {"candidate_id": args.candidate_id, "thresholds": {
        "config_hash": t.config_hash,
        "source": str(Path(args.thresholds) if args.thresholds
                      else newest_thresholds_file() or "code defaults")},
        "pairings": compared, "hard_gate_violations": violations,
        "decision": decision}
    # 事件先落、判定后落：反过来一旦状态机拒绝迁移，就会出现"有判定文件
    # 却没有对应事件"的分叉（实测踩到过）。
    log = _log(args.run_id)
    last = log.last()
    try:
        if last is None:
            # 新 run：按状态机的合法路径走到 evaluating
            for phase, status, note in (
                    ("queued", "start", "候选评估"),
                    ("auditing", "started", "输入与硬门槛检查"),
                    ("training", "done", "候选权重已产出（由训练侧记录）"),
                    ("evaluating", "evaluated", "")):
                log.append(_ev(args.run_id, phase, status,
                               candidate=args.candidate_id, note=note))
        else:
            log.append(_ev(args.run_id, "evaluating", "evaluated",
                           candidate=args.candidate_id,
                           metrics={k: metric(v.get("mean_delta"), "delta",
                                              missing="" if v.get("n")
                                              else "未测")
                                    for k, v in compared.items()},
                           note="; ".join(decision["reasons"])[:200]))
        log.append(_ev(args.run_id, decision["decision"], "decided",
                       candidate=args.candidate_id,
                       note="; ".join(decision["reasons"])[:200]))
    except ValueError as exc:
        print(f"[autoloop] 状态机拒绝本次判定：{exc}")
        print("  该 run 处于待复核/终态时不能再评估：先补真值重新审计"
              "（needs_review -> auditing），或换一个 run-id。")
        return 2
    dec_path = out / f"decision_{args.candidate_id}.json"
    dec_path.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[autoloop] {args.candidate_id} -> {decision['decision']}")
    for r in decision["reasons"]:
        print(f"  - {r}")
    if violations:
        print(f"  硬门槛违反 {len(violations)} 项：{violations}")
    print(f"[autoloop] 判定 -> {dec_path}")
    # 退出码要让控制器能分辨："证据不足"不是成功也不是淘汰
    return {"shadow_candidate": 0, "rejected": 1,
            "needs_evidence": 3}.get(decision["decision"], 1)


def cmd_replay(args) -> int:
    """同一批输入重算决策：必须逐字相同，否则说明判定里混进了不确定量。"""
    log = _log(args.run_id)
    decs = sorted(exp_dir(args.run_id).glob("decision_*.json"))
    if not decs:
        print("[autoloop] 没有可重放的判定（先跑 evaluate）")
        return 2
    same, diff = 0, []
    for p in decs:
        blob = json.loads(p.read_text(encoding="utf-8"))
        t = thresholds(
            Path(blob["thresholds"]["source"])
            if Path(blob["thresholds"]["source"]).exists() else None)
        compared = {name: spec for name, spec in blob["pairings"].items()}
        again = decide(pairings=compared, thresholds=t,
                       missing_metrics=[k for k, v in compared.items()
                                        if not v.get("n")],
                       hard_gate_violations=blob["hard_gate_violations"])
        if again == blob["decision"]:
            same += 1
        else:
            diff.append({"file": p.name, "was": blob["decision"],
                         "now": again})
    print(f"[autoloop] 重放 {len(decs)} 个判定：相同 {same}，不同 {len(diff)}")
    for d in diff:
        print(f"  {d['file']}: 决策或理由不一致 -> 判定不可复现")
    return 0 if not diff else 1


def cmd_run(args) -> int:
    cfg_path = Path(args.config)
    cfg = (LoopConfig.load(cfg_path) if cfg_path.exists()
           else LoopConfig(dry_run=True))
    # 命令行优先：`--no-dry-run` 必须能覆盖配置里的 dry_run=true（实测踩到：
    # 配置样例写的 dry_run=true 会让入口一直只打印计划，什么都不做）
    cfg = LoopConfig(**{**cfg.__dict__, "dry_run": bool(args.dry_run)})
    # GPU 预算：以本 run 的账本为准（跨进程累计），命令行只作外部补充，
    # 取两者较大者——否则配置里的每日上限形同虚设（实测踩到：账本写了 999
    # 分钟，资源门还是按命令行默认的 0 放行）
    # 上次被杀留下的 pending 先结算（否则那段时间不进预算）
    _recovered = settle_gpu_minutes(args.run_id)
    if _recovered > 0:
        print(f"[autoloop] 补记上次未结算的 GPU 时间 {_recovered:.1f} min")
    used_today = max(float(args.gpu_minutes_today or 0.0),
                     gpu_minutes_today(args.run_id))
    st = probe_resources(disk_path=ROOT,
                         gpu_minutes_today=used_today,
                         user_active=args.user_active)
    gate = resource_gate(cfg, st)
    print(f"[autoloop] 资源门：allowed={gate['allowed']}")
    for r in gate["reasons"]:
        print(f"  - 阻止：{r}")
    for w in gate["warnings"]:
        print(f"  - 提示：{w}")
    log = _log(args.run_id)
    lock = InstanceLock(exp_dir(args.run_id) / "controller.lock")
    held = lock.acquire()
    if not held["acquired"]:
        print(f"[autoloop] 单实例锁：{held['reason']}")
        return 4
    try:
        plan = plan_once(cfg, run_id=args.run_id,
                         candidate_id=args.candidate_id or "candidate",
                         dataset_id=args.dataset_id or "unversioned",
                         python=sys.executable,
                         script_dir=ROOT / "scripts", seed=cfg.seeds[0],
                         extra_train_args=shlex.split(args.train_args or ""))
        print(f"[autoloop] dry_run={cfg.dry_run} collect={cfg.collect} "
              f"config_hash={plan.config_hash}")
        for a in plan.actions:
            tag = "（需授权）" if a.optional else ""
            print(f"  [{a.phase}] {a.name}{tag}: {' '.join(a.cmd)}")
            print(f"     理由：{a.reason}")
        hist = []
        for p in sorted(exp_dir(args.run_id).glob("decision_*.json")):
            blob = json.loads(p.read_text(encoding="utf-8"))
            hist.append(RoundRecord(
                round_index=len(hist), candidate_id=blob["candidate_id"],
                decision=blob["decision"]["decision"],
                reasons=blob["decision"]["reasons"]))
        stop = should_stop(cfg, hist, gpu_minutes_today=used_today,
                           candidates_used=len(hist))
        print(f"[autoloop] 停止条件：stop={stop['stop']} "
              f"连续无收益={stop['no_gain_streak']}")
        for r in stop["reasons"]:
            print(f"  - {r}")
        if not gate["allowed"]:
            print("[autoloop] 资源门未通过：本轮不执行")
            if not cfg.dry_run:
                log.append(_ev(args.run_id, "paused", "resource_blocked",
                               note="; ".join(gate["reasons"])[:200]))
                return 5
        if cfg.dry_run:
            print("[autoloop] dry-run：未执行任何训练/采集命令")
            return 0
        # ---- 非 dry-run：复用 rounds 同一条流程 -------------------------
        if cfg.collect != "off":
            print(f"[autoloop] collect={cfg.collect}：后台采集尚未授权，"
                  f"本轮仍只用现有数据（方案 §136）")
        if not cfg.runs or not cfg.eval_runs:
            print("[autoloop] 配置缺 runs/eval_runs：后台入口无事可做（不猜路径）")
            log.append(_ev(args.run_id, "paused", "no_data_configured",
                           note="配置里没有 runs/eval_runs"))
            return 5
        try:                      # torch 在本脚本里是惰性导入的
            import torch as _torch
            _dev = "cuda" if _torch.cuda.is_available() else "cpu"
        except Exception:                                 # noqa: BLE001
            _dev = "cpu"
        r_args = rounds_args_from(cfg, args.run_id, device=_dev)
        mark_gpu_start(args.run_id)
        _t0 = time.time()
        log.append(_ev(args.run_id, "training", "running",
                       note=f"后台入口启动 rounds={r_args.rounds} "
                            f"config_hash={cfg.hash()}"))
        rc = cmd_rounds(r_args)
        used = (time.time() - _t0) / 60.0
        settle_gpu_minutes(args.run_id)      # 结算（含被中断时的补记）
        total_today = gpu_minutes_today(args.run_id)
        if total_today <= 0:
            total_today = add_gpu_minutes(args.run_id, used)
        print(f"[autoloop] 本轮 GPU 墙钟 {used:.1f} min（今日累计 {total_today:.1f}"
              f" / 上限 {cfg.daily_gpu_minutes:.0f} min）")
        hist2 = []
        for p_ in sorted(exp_dir(args.run_id).glob("decision_*.json")):
            blob = json.loads(p_.read_text(encoding="utf-8"))
            hist2.append(RoundRecord(
                round_index=len(hist2), candidate_id=blob["candidate_id"],
                decision=blob["decision"]["decision"],
                reasons=blob["decision"]["reasons"]))
        stop2 = should_stop(cfg, hist2, gpu_minutes_today=total_today,
                            candidates_used=len(hist2))
        note = (f"rounds rc={rc}; 停因: " + "; ".join(stop2["reasons"])
                if stop2["stop"] else f"rounds rc={rc}; 未触发停止条件")
        log.append(_ev(args.run_id, "paused" if stop2["stop"] else "training",
                       "stopped" if stop2["stop"] else "idle", note=note[:200]))
        print(f"[autoloop] 后台入口结束：{note}")
        return int(rc)
    finally:
        lock.release()


def gpu_minutes_path(run_id: str) -> Path:
    return exp_dir(run_id) / "gpu_minutes.json"


def gpu_minutes_today(run_id: str, *, today: str | None = None) -> float:
    """今天这个 run 已用掉的 GPU 分钟（跨进程累计；缺文件按 0）。"""
    today = today or time.strftime("%Y-%m-%d")
    p = gpu_minutes_path(run_id)
    if not p.exists():
        return 0.0
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return 0.0
    return float(blob.get(today, {}).get("minutes") or 0.0)


def gpu_pending_path(run_id: str) -> Path:
    return exp_dir(run_id) / "gpu_minutes.pending.json"


def mark_gpu_start(run_id: str) -> None:
    """记下开始时间（崩溃/被杀时下次启动仍能把这段时间算进预算）。"""
    p = gpu_pending_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"started": time.time(),
                             "started_iso": time.strftime(
                                 "%Y-%m-%dT%H:%M:%S")}),
                 encoding="utf-8")


def settle_gpu_minutes(run_id: str) -> float:
    """把 pending 的时间结算进账本并清掉 pending；没有 pending 就返回 0。"""
    p = gpu_pending_path(run_id)
    if not p.exists():
        return 0.0
    try:
        started = float(json.loads(p.read_text(encoding="utf-8"))["started"])
    except Exception:                                     # noqa: BLE001
        p.unlink(missing_ok=True)
        return 0.0
    p.unlink(missing_ok=True)
    return add_gpu_minutes(run_id, (time.time() - started) / 60.0)


def add_gpu_minutes(run_id: str, minutes: float, *,
                    today: str | None = None) -> float:
    """累加并写回；返回今天的新累计值。"""
    today = today or time.strftime("%Y-%m-%d")
    p = gpu_minutes_path(run_id)
    blob = {}
    if p.exists():
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            blob = {}
    slot = dict(blob.get(today) or {})
    slot["minutes"] = round(float(slot.get("minutes") or 0.0)
                            + max(0.0, float(minutes)), 2)
    blob[today] = slot
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                 encoding="utf-8")
    return float(slot["minutes"])


def rounds_args_from(cfg, run_id: str, *, device: str = "cuda",
                     thresholds: str | None = None,
                     timeout_s: int = 3600) -> argparse.Namespace:
    """把 ``LoopConfig`` 变成 `rounds` 的参数——`run` 复用的就是同一条流程。

    这样"后台入口"与手工 `rounds` **共用**审计门/因子应用/基线臂/按 seed 配对/
    判定落盘，不存在第二套实现（方案要求"复用现有 rounds 流程"）。
    """
    # 关键：把这份配置写进 run 目录，并让 rounds 读它。否则 rounds 拿到一个
    # 不存在的路径会退回默认阈值（实测踩到：入口配 max_rounds_without_gain=2，
    # rounds 却按默认 3 在跑，第三轮才停）。
    cfg_path = exp_dir(run_id) / "loop_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(asdict(cfg), indent=1, ensure_ascii=False,
                                    default=str), encoding="utf-8")

    return argparse.Namespace(
        run_id=run_id, rounds=int(cfg.rounds),
        runs=list(cfg.runs), baseline_runs=list(cfg.baseline_runs) or None,
        eval_runs=list(cfg.eval_runs), proposals=(cfg.proposals or None),
        seeds=list(cfg.seeds), epochs=int(cfg.epochs), batch=int(cfg.batch),
        lr=float(cfg.lr), split="tail", device=device,
        thresholds=thresholds, config=str(cfg_path),
        hard_recall=0.70, hard_precision=0.40, hard_p95=45.0,
        timeout_s=int(timeout_s), trainer_script=str(cfg.trainer_script),
        plan_only=False, allow_road_only=bool(cfg.allow_road_only),
        equal_steps=bool(cfg.equal_steps), resume=True)


def _pixel_eval(model_path: Path, eval_runs: list, device: str) -> dict:
    """开发集上的像素层指标（复用评估矩阵的实现，不另写一套口径）。"""
    import m5_seg_eval_matrix as em
    frames = em.load_frames([Path(r) for r in eval_runs])
    return em.evaluate_model(model_path, frames, device=device)


def _identity_rate(model_path: Path, eval_runs: list) -> float | None:
    """候选身份确认率：identity probe 的匹配率（多段路取均值）；测不到返回 None。"""
    import m5_marking_identity_probe as ip
    rates = []
    for r in eval_runs:
        run = Path(r)
        meta = run / "meta.json"
        if not meta.exists() and (run.parent / "meta.json").exists():
            meta = run.parent / "meta.json"
        if not meta.exists():
            continue
        try:
            res = ip.probe(run, json.loads(meta.read_text(encoding="utf-8")),
                           view=run.name, model_path=str(model_path))
        except Exception:                    # noqa: BLE001
            continue
        summary = res.get("summary") or {}
        if summary.get("match_rate") is not None:
            rates.append(float(summary["match_rate"]))
    return (sum(rates) / len(rates)) if rates else None


#: 提议里的因子 -> 训练器开关的白名单。数据组成类键（add_runs/drop_runs/
#: group_weights）改的是 `--runs` 列表本身，不是训练器参数；把它们的字面值当
#: 开关传过去会被 argparse 拒（实测踩到），所以这里显式跳过并记账。
#: 训练器开关型因子（值是标量）。``run_weights`` 是字典型，单独格式化。
TRAINER_FLAG_FACTORS = ("epochs", "lr", "line_weight", "line_tversky_weight",
                        "line_cldice_weight", "line_tversky_beta",
                        "run_weights")


def factor_to_flags(factor: dict) -> tuple:
    """``(flags, skipped)``：只有白名单里的因子会变成命令行参数。"""
    flags: list = []
    skipped: dict = {}
    for key, value in (factor or {}).items():
        if key not in TRAINER_FLAG_FACTORS:
            skipped[key] = value
            continue
        if isinstance(value, dict):
            # 字典型开关（run_weights）：序列化成 "0=2.0,1=0.5"
            flags += ["--" + key.replace("_", "-"),
                      ",".join(f"{k}={v}" for k, v in sorted(value.items()))]
        else:
            flags += ["--" + key.replace("_", "-"), str(value)]
    return flags, skipped


#: 数据组成类因子：改的是训练输入（`--runs` 列表）本身，不是训练器开关。
DATA_FACTORS = ("add_runs", "drop_runs")


def _as_path_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(v) for v in value]


def arm_runs(base_runs, factor) -> tuple:
    """``(runs, note)``：把数据因子作用到训练输入上，并逐键记账。

    ``note["applied"]`` 是真被应用的键，``note["not_applied"]`` 是无效键或
    训练器没有实现的键。一个键都没生效时调用方必须**拒绝训练**：那种情况下
    候选臂与基线臂的训练输入逐字相同，跑出来的只是一场"看起来成功"的实验
    （方案步骤 3 点名过这个缺陷：只打印 skipped_factors 就照训）。
    """
    runs = [Path(r) for r in base_runs]
    note: dict = {"applied": [], "not_applied": []}
    fac = dict(factor or {})
    if "add_runs" in fac:
        added = _as_path_list(fac.get("add_runs"))
        if not added:
            note["not_applied"].append("add_runs 是空列表")
        for r in added:
            if r in runs:
                note["not_applied"].append(f"add_runs {r}: 已在训练列表里")
                continue
            runs.append(r)
            note["applied"].append(f"add_runs {r}")
    if "drop_runs" in fac:
        drops = _as_path_list(fac.get("drop_runs"))
        hit = [r for r in runs if r in drops]
        if not hit:
            note["not_applied"].append("drop_runs 没匹配到任何目录")
        elif len(hit) == len(runs):
            note["not_applied"].append("drop_runs 会清空训练列表（拒绝）")
        else:
            runs = [r for r in runs if r not in drops]
            note["applied"].append(f"drop_runs 去掉 {len(hit)} 个目录")
    for key in sorted(fac):
        if key in DATA_FACTORS or key in TRAINER_FLAG_FACTORS:
            continue
        note["not_applied"].append(f"{key}={fac[key]!r}: 训练器没有对应实现")
    return runs, note


def train_cmd(args, runs, out: Path, seed: int, extra: list) -> list:
    """两臂共用同一条训练命令构造。

    允许的差异只有数据列表、输出路径、seed（方案步骤 4 要求逐项 diff：
    除这三项外两臂命令必须一致）。
    """
    script = Path(args.trainer_script)
    if not script.is_absolute():
        script = ROOT / "scripts" / script
    # road-only：两臂**都**整通道屏蔽 line 类（共享配方，不是因子差异）。
    # 训练器有 class-masked CE 才能这么干；只把 line 权重置零是不够的
    # （未标注漆线仍会在 softmax 分母里当负样本）。
    road_only = (["--ignore-line-class", "--line-tversky-weight", "0",
                  "--line-cldice-weight", "0"]
                 if getattr(args, "allow_road_only", False) else [])
    return [sys.executable, str(script),
            "--runs", *[str(r) for r in runs],
            "--split", args.split, "--val-frac", "0.2",
            "--epochs", str(args.epochs), "--batch", str(args.batch),
            "--lr", str(args.lr), "--seed", str(seed),
            "--device", args.device, "--save-every-epoch",
            "--out", str(out), *road_only, *extra]


def steps_per_epoch(n_train: int, batch: int) -> int:
    """每个 epoch 的优化步数（与训练器一致：按批向上取整）。"""
    return max(1, -(-int(n_train) // max(1, int(batch))))


def equal_steps_epochs(*, target_steps: int, n_train: int, batch: int) -> int:
    """把某一臂的 epoch 数定成"总步数最接近 target_steps"（至少 1 轮）。

    方案步骤 3：数据臂增加帧数时必须做**等步数对照**，否则"训练更久"会被读成
    "数据更好"。帧数不是整数倍时无法精确相等，取最接近的整数轮，并以训练器
    落盘的实际步数为准（不精确就在判定里注明，不假装相等）。
    """
    spe = steps_per_epoch(n_train, batch)
    return max(1, int(round(float(target_steps) / spe)))


def count_frames(runs) -> int:
    """候选臂的帧数（用于**事前**估步数；真实步数仍以训练器落盘为准）。"""
    import glob as _glob
    return sum(len(_glob.glob(str(Path(r) / "frame_*.npz"))) for r in runs)


def _ckpt_train_args(path: Path) -> dict:
    """从 checkpoint 读训练参数（拿**实测**的 n_train/epochs 算真实步数）。"""
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        return dict(ck.get("train_args") or {})
    except Exception:                                     # noqa: BLE001
        return {}


#: p95 与同 seed p50 的比值超过它，就认为这次计时被机器负载污染
#: （实测：并发跑 pytest 时同一 checkpoint 测出 p95=158 ms，安静时 16.7 ms，
#:  比值 12.8；正常时 p95/p50 ≈ 1.3–1.6）。
TIMING_SUSPECT_RATIO = 4.0


def timing_suspect(p50: float | None, p95: float | None) -> bool:
    """这次 p95 是否"不可信"（同 seed 的 p50 与它差一个量级）。

    只做**内部一致性**检查，不需要知道机器上还有什么在跑：延迟的 p50 与 p95
    不可能差到几倍以上（除非负载在测量期间变化）。可疑时不把它当"模型很慢"，
    而是在判定里标出来，避免把并发负载记成模型缺陷。
    """
    try:
        if p50 is None or p95 is None:
            return False
        p50, p95 = float(p50), float(p95)
    except (TypeError, ValueError):
        return False
    if p50 <= 0:
        return False
    return (p95 / p50) > float(TIMING_SUSPECT_RATIO)


def _mean_or_none(values) -> float | None:
    """实测值求均值；**没有测量**时返回 None（UNKNOWN），绝不回落到命令行阈值。

    方案点名过这个缺陷：`--hard-recall/--hard-precision/--hard-p95` 是命令行输入，
    把它们当"测量结果"送进硬门，等于用阈值给自己打分。
    """
    vals = [float(v) for v in (values or []) if v is not None]
    return None if not vals else round(sum(vals) / len(vals), 4)


def _train_once(cmd: list, timeout_s: int):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=max(600, timeout_s))


def _log_give_up(log, run_id: str, cand_id: str, status: str, note: str,
                 *, phase: str = "needs_evidence") -> None:
    """记一次"不训练/不判定"的终止，走状态机允许的路径。

    ``auditing`` 只能到 ``needs_review|training|rejected|paused|failed``，
    到了 ``training`` 则必须经 ``evaluating`` 才能到 ``needs_evidence``
    （状态机规则，实测踩过两次）。
    """
    last = log.last()
    if last is not None and last.phase == "training" and \
            phase == "needs_evidence":
        log.append(_ev(run_id, "evaluating", "input_check", candidate=cand_id,
                       note=note[:120]))
    log.append(_ev(run_id, phase, status, candidate=cand_id, note=note[:200]))


def _rounds_audit(args, train_runs, log) -> tuple:
    """训练前的数据准入门（方案步骤 5 的"审计"那一步）。

    返回 ``(report, rc)``；``rc != 0`` 时调用方必须停下，不训练。检查项：
    训练侧逐通道可用真值、被拒帧（身份/格式）、训练组与开发组的**整组隔离**、
    以及"标线真值不可用"（除非显式 ``--allow-road-only``）。
    """
    mf_tr = DatasetManifest.build([Path(r) for r in train_runs], root=ROOT)
    mf_dev = DatasetManifest.build([Path(r) for r in args.eval_runs], root=ROOT)
    cov = mf_tr.coverage().get("train", {})
    trainable = int(cov.get("trainable_frames") or 0)
    paint_ok = int(cov.get("paint_valid_frames") or 0)
    train_groups = sorted({r.group for r in mf_tr.records if not r.reject_reason})
    dev_groups = sorted({r.group for r in mf_dev.records if not r.reject_reason})
    overlap = sorted(set(train_groups) & set(dev_groups))
    rejected = [{"path": r.path, "reason": r.reject_reason}
                for r in mf_tr.records if r.reject_reason]
    report = {"dataset_id": mf_tr.dataset_id,
              "dev_dataset_id": mf_dev.dataset_id,
              "train_groups": train_groups, "dev_groups": dev_groups,
              "group_overlap": overlap, "coverage": cov,
              "rejected": rejected, "notes": mf_tr.notes + mf_dev.notes}
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rounds_dataset.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[rounds] 审计：train groups={train_groups} dev groups={dev_groups} "
          f"trainable={trainable} paint_ok={paint_ok} rejected={len(rejected)}")
    for n in report["notes"]:
        print(f"[rounds]   审计提示：{n}")
    if rejected:
        _log_give_up(log, args.run_id, "audit", "data_rejected",
                     f"{len(rejected)} 帧被准入门拒绝（身份/格式）",
                     phase="needs_review")
        print("[rounds] 审计不通过：训练数据里有被拒帧，先修数据（不训练）")
        return report, 3
    if not trainable:
        log.append(_ev(args.run_id, "auditing", "no_trainable_data",
                       note="没有任何通道有可用真值：停止，不训练"))
        print("[rounds] 审计不通过：没有可用真值，停止")
        return report, 2
    if overlap:
        _log_give_up(log, args.run_id, "audit", "group_overlap",
                     f"训练/开发共用组 {overlap}：泄漏风险，先整组重划",
                     phase="needs_review")
        print(f"[rounds] 审计不通过：训练与开发共用组 {overlap}（泄漏），不训练")
        return report, 3
    if not paint_ok and not getattr(args, "allow_road_only", False):
        _log_give_up(log, args.run_id, "audit", "paint_truth_missing",
                     "有 Tech annotation 但无可靠标线真值：只做路面通道需 "
                     "--allow-road-only", phase="needs_review")
        print("[rounds] 审计：标线真值不可用 -> needs_review（未训练）；"
              "只做路面通道实验请加 --allow-road-only")
        return report, 3
    return report, 0


def cmd_rounds(args) -> int:
    """连续跑 N 轮离线循环：提议 → 训练（多 seed）→ 成对评估 → 判定 → 停止条件。

    方案阶段 D 的验收动作：多轮无人值守、至少一次净负收益淘汰、相同输入重放
    一致。它**不启动游戏、不常驻**：每轮训练都是前台子进程，跑完即退出，
    停止条件交给 ``should_stop`` 判定。评估只用开发集，最终集不参与搜索。

    三处纪律（方案步骤 3 的接口现状 1/3）：
    * 数据因子必须**真的改变训练输入**，一个键都没生效就拒绝训练；
    * champion 来自**基线臂**（第 0 轮用同一配方/seed/开发集训练一次并落盘），
      绝不用本轮候选自己的值当对照；
    * 成对比较**按 seed 配对**，seed 集合对不上就拒绝，不做截断错配。
    """
    cfg_path = Path(args.config)
    cfg = (LoopConfig.load(cfg_path) if cfg_path.exists()
           else LoopConfig(dry_run=True))
    log = _log(args.run_id)
    if log.last() is None:
        log.append(_ev(args.run_id, "queued", "start",
                       note=f"rounds={args.rounds} seeds={args.seeds}"))
        log.append(_ev(args.run_id, "auditing", "started",
                       note="训练前数据准入门（逐通道真值/身份/整组隔离）"))
    props = []
    if args.proposals:
        props = json.loads(Path(args.proposals).read_text(
            encoding="utf-8")).get("proposals", [])
    t = thresholds(Path(args.thresholds) if args.thresholds else None)
    base_runs = _as_path_list(args.baseline_runs or args.runs)
    # 训练前先审计：所有会被训练用到的目录（基线臂 + 各轮候选臂）+ 开发集
    all_train: list = [Path(r) for r in base_runs]
    for i in range(int(args.rounds)):
        factor_i = (props[i % len(props)] if props else {}).get("factor", {})
        for r in arm_runs(args.runs, factor_i)[0]:
            if Path(r) not in all_train:
                all_train.append(Path(r))
    for r in _as_path_list(args.runs):
        if Path(r) not in all_train:
            all_train.append(Path(r))
    if not args.plan_only:
        _report, rc = _rounds_audit(args, all_train, log)
        if rc:
            return rc
        log.append(_ev(args.run_id, "training", "running", note="候选逐轮训练"))
    champ_file = exp_dir(args.run_id) / "champion.json"
    history: list = []
    for rnd in range(int(args.rounds)):
        prop = props[rnd % len(props)] if props else {}
        factor = prop.get("factor", {})
        # 候选身份必须**逐轮唯一**：同一个因子在下一轮再跑，是新的一次候选
        # （方案：candidate_id 固定配置/数据/seed/命令）。复用旧 id 会撞上上一轮
        # 已终结的事件流（实测：rejected -> evaluating 被状态机拒绝）。
        cand_id = f"{prop.get('candidate_id', 'cand')}-r{rnd}"
        if getattr(args, "resume", False) and                 (exp_dir(args.run_id) / f"decision_{cand_id}.json").exists():
            print(f"[rounds] 第 {rnd + 1} 轮 {cand_id} 已有判定：resume 跳过")
            hist_done = json.loads(
                (exp_dir(args.run_id) / f"decision_{cand_id}.json")
                .read_text(encoding="utf-8"))["decision"]
            history.append(RoundRecord(round_index=rnd, candidate_id=cand_id,
                                       decision=hist_done["decision"],
                                       reasons=hist_done["reasons"]))
            continue
        extra, skipped = factor_to_flags(factor)
        cand_runs, data_note = arm_runs(args.runs, factor)
        # 因子未生效就拒绝训练：候选臂与基线臂输入相同，跑不出证据
        if not extra and not data_note["applied"]:
            note = ("factor_not_applied: " + "; ".join(
                data_note["not_applied"] or ["提议没有可用因子"]))
            _log_give_up(log, args.run_id, cand_id, "factor_not_applied", note)
            print(f"[rounds] 第 {rnd + 1} 轮拒绝训练：{note}")
            print("  候选与基线的训练输入会完全相同。要么让训练器支持该因子"
                  "（例如样本采样权重），要么换一个真能生效的因子。")
            return 3
        if data_note["not_applied"]:
            print(f"[rounds] 第 {rnd + 1} 轮部分因子未生效（已记账）："
                  f"{data_note['not_applied']}")
        # road-only：训练器已有 class-masked CE（line 类从 softmax 分母里
        # 去掉，既无正样本也无负样本），所以路面通道实验可以真跑；代价是
        # **标线指标没有任何真值**，必须记 UNKNOWN，不能拿 0 当"测过"。
        if getattr(args, "allow_road_only", False) and not args.plan_only:
            print("[rounds] road-only：两臂都带 --ignore-line-class（line 类整"
                  "通道屏蔽）；标线/身份指标将记为 UNKNOWN，主指标为 road_iou")
        if args.plan_only:
            print(f"[plan] 第 {rnd + 1} 轮 {cand_id} factor={factor}")
            for seed in args.seeds:
                print("[plan] baseline " + " ".join(train_cmd(
                    args, base_runs,
                    exp_dir(args.run_id) / "baseline" / f"seed{seed}",
                    seed, [])))
                print("[plan] candidate " + " ".join(train_cmd(
                    args, cand_runs,
                    exp_dir(args.run_id) / f"round{rnd}" / f"seed{seed}",
                    seed, extra)))
            continue
        # ---- 基线臂：第 0 轮训练一次并落盘，跨轮沿用 ----------------
        base_steps: dict = {}
        base_n_train: int = 0
        equal_cap: int = 0
        cand_epochs = int(args.epochs)
        if rnd == 0:
            champ_by_seed: dict = {}
            for seed in args.seeds:
                out = exp_dir(args.run_id) / "baseline" / f"seed{seed}"
                cmd = train_cmd(args, base_runs, out, seed, [])
                run = _train_once(cmd, args.timeout_s)
                if run.returncode != 0:
                    log.append(_ev(args.run_id, "failed", "baseline_train_error",
                                   candidate=cand_id, seed=seed,
                                   note=(run.stdout[-200:] + run.stderr[-200:])))
                    print(f"[rounds] 基线臂训练失败（seed {seed}），停止")
                    return 1
                m = _pixel_eval(out / "checkpoint_last.pt", args.eval_runs,
                                args.device)
                _key = "road_iou" if getattr(args, "allow_road_only", False)                     else "line_iou"
                champ_by_seed[str(seed)] = float(
                    m[_key] if m.get(_key) is not None
                    else m.get("line_iou") or 0.0)
                ta_b = _ckpt_train_args(out / "checkpoint_last.pt")
                if ta_b.get("n_train"):
                    base_n_train = max(base_n_train, int(ta_b["n_train"]))
                    base_steps[str(seed)] = steps_per_epoch(
                        ta_b["n_train"], ta_b.get("batch") or args.batch)                         * int(ta_b.get("epochs") or args.epochs)
            # 等步数对照（精确版）：两臂 **epochs 不变**，把帧多的一臂按
            # run 配额截到与基线相同的训练帧数 —— 步数因此逐位相等，
            # 不需要用"最接近的整数轮"去凑（那会留下 +33% 的残余差）。
            if args.equal_steps and base_steps and base_n_train:
                n_cand = count_frames(cand_runs)
                n_val_c = max(1, int(n_cand * 0.2))
                n_train_c = max(1, n_cand - n_val_c)
                if n_train_c > base_n_train:
                    equal_cap = base_n_train
                    print(f"[rounds] --equal-steps：候选训练帧 {n_train_c} "
                          f"→ 截到 {base_n_train}（按 run 配额，epochs 不变）")
                else:
                    print(f"[rounds] --equal-steps：候选训练帧 {n_train_c} "
                          f"≤ 基线 {base_n_train}，无需截帧")
            champ_file.write_text(json.dumps({
                "by_seed": champ_by_seed, "source": "baseline_arm",
                "arm_runs": [str(r) for r in base_runs],
                "epochs": args.epochs, "seeds": list(args.seeds),
                "steps_by_seed": base_steps,
                "candidate_epochs": int(cand_epochs),
                "base_n_train": int(base_n_train),
                "max_train_frames": int(equal_cap),
                "equal_steps": bool(args.equal_steps)},
                indent=1, ensure_ascii=False), encoding="utf-8")
        elif champ_file.exists():
            champ_by_seed = json.loads(champ_file.read_text(
                encoding="utf-8")).get("by_seed") or {}
            base_steps = json.loads(champ_file.read_text(
                encoding="utf-8")).get("steps_by_seed") or {}
            equal_cap = int(json.loads(champ_file.read_text(
                encoding="utf-8")).get("max_train_frames") or 0)
        else:
            print("[rounds] 没有基线记录（champion.json 缺失）：拒绝用本轮"
                  "候选自己的值当对照")
            _log_give_up(log, args.run_id, cand_id, "no_baseline",
                         "champion.json 缺失，无法做非自我比较")
            return 3
        # ---- 候选臂 ------------------------------------------------
        cand_by_seed: dict = {}
        worst_by_seed: dict = {}
        trivial: dict = {}
        timing_notes: list = []
        offroads: list = []
        idents: list = []
        hard_measured: dict = {}
        for seed in args.seeds:
            out = exp_dir(args.run_id) / f"round{rnd}" / f"seed{seed}"
            _cmd = train_cmd(args, cand_runs, out, seed, extra)
            if equal_cap:
                # 只加这一个量：epochs/其它开关与基线臂逐字相同
                _cmd += ["--max-train-frames", str(int(equal_cap))]
            run = _train_once(_cmd, args.timeout_s)
            if run.returncode != 0:
                log.append(_ev(args.run_id, "failed", "train_error",
                               candidate=cand_id, seed=seed,
                               note=(run.stdout[-200:] + run.stderr[-200:])))
                print(f"[rounds] 第 {rnd + 1} 轮训练失败（seed {seed}），停止")
                return 1
            metrics = _pixel_eval(out / "checkpoint_last.pt", args.eval_runs,
                                  args.device)
            _key = "road_iou" if getattr(args, "allow_road_only", False)                 else "line_iou"
            cand_by_seed[str(seed)] = float(
                metrics[_key] if metrics.get(_key) is not None
                else metrics.get("line_iou") or 0.0)
            # 硬门输入一律取自**本轮实测**（方案点名：--hard-* 是命令行输入，
            # 不能把阈值本身当测量结果；缺测就是 None=UNKNOWN）
            offroads.append(metrics.get("offroad_false_frac_of_pred"))
            if metrics.get("mask_compare"):
                # "点进具体帧"：判定文件带上该 seed 最差的几帧（路径+IoU+真值像素），
                # 人可以从一次失败判定直接看到是哪张图、差在哪
                worst_by_seed[str(seed)] = metrics["mask_compare"]["worst"]
            hard_measured.setdefault("line_recall", []).append(
                metrics.get("line_recall"))
            hard_measured.setdefault("line_precision", []).append(
                metrics.get("line_precision"))
            if metrics.get("inference_ms_p95") is not None:
                hard_measured.setdefault("inference_ms_p95", []).append(
                    float(metrics["inference_ms_p95"]))
            hard_measured.setdefault("inference_ms_p50", []).append(
                metrics.get("inference_ms_p50"))
            if metrics.get("road_iou_trivial_all_road") is not None:
                trivial = {"all_road": metrics["road_iou_trivial_all_road"],
                           "all_bg": metrics.get("road_iou_trivial_all_background")}
            if timing_suspect(metrics.get("inference_ms_p50"),
                              metrics.get("inference_ms_p95")):
                timing_notes.append(
                    {str(seed): {"p50": metrics.get("inference_ms_p50"),
                                 "p95": metrics.get("inference_ms_p95"),
                                 "why": "p95/p50 超过 "
                                        f"{TIMING_SUSPECT_RATIO:g}×：计时很可能"
                                        "被并发负载污染（实测 158ms vs 安静时 "
                                        "16.7ms）。该 p95 不作为模型属性引用"}})
            ident = _identity_rate(out / "checkpoint_last.pt", args.eval_runs)
            if ident is not None:
                idents.append(ident)
        # 按 **seed** 配对：顺序/数量不一致说明对照不完整，拒绝而不是截断
        missing = [str(s) for s in args.seeds if str(s) not in champ_by_seed]
        if missing:
            note = (f"seed 不匹配：基线缺 {missing}（基线 seeds="
                    f"{sorted(champ_by_seed)}）—— 错配的比较不是证据")
            _log_give_up(log, args.run_id, cand_id, "seed_mismatch", note)
            print(f"[rounds] 第 {rnd + 1} 轮拒绝判定：{note}")
            return 3
        champ = [champ_by_seed[str(s)] for s in args.seeds]
        cand = [cand_by_seed[str(s)] for s in args.seeds]
        road_only = bool(getattr(args, "allow_road_only", False))
        pair_metric = "road_iou" if road_only else "line_iou"
        paired = {name: paired_compare(name, champ, cand)
                  for name in (pair_metric,)}
        if road_only:
            # line 通道被整通道屏蔽：标线/身份指标**没有真值**，一律 UNKNOWN
            # （记 None 而不是 0，避免"没测"被读成"很差"或被硬门当违反）
            hard = {"line_recall": None, "line_precision": None,
                    "candidate_identity_rate": None,
                    "offroad_false_ratio": None,
                    "inference_ms_p95": _mean_or_none(
                        hard_measured.get("inference_ms_p95")),
                    "note": "road-only：标线通道整通道屏蔽，标线/身份指标未测"}
            idents = []
        else:
            hard = {"line_recall": _mean_or_none(
                        hard_measured.get("line_recall")),
                    "line_precision": _mean_or_none(
                        hard_measured.get("line_precision")),
                    "offroad_false_ratio": _mean_or_none(
                        [x for x in offroads if x is not None]),
                    "inference_ms_p95": _mean_or_none(
                        hard_measured.get("inference_ms_p95")),
                    "candidate_identity_rate": (round(sum(idents) / len(idents), 4)
                                                if idents else None)}
        _need = (paired.get(pair_metric) or {}).get("seeds_needed_for_effect")
        if args.equal_steps and _need:
            print(f"[rounds] 该效应按当前 sd 需要约 {_need} 个 seed 才判得出"
                  f"（现在 {len(args.seeds)} 个）——先算规模再决定加不加 GPU")
        if timing_notes:
            print(f"[rounds] 警告：{len(timing_notes)} 个 seed 的 p95/p50 超过 "
                  f"{TIMING_SUSPECT_RATIO:g}×，计时疑似受并发负载污染"
                  f"——已记录，不作为模型延迟引用")
            hard["inference_ms_p95"] = None
            hard["timing_suspect"] = timing_notes
        dec = decide(pairings=paired, thresholds=t,
                     missing_metrics=[k for k, v in paired.items()
                                      if not v.get("n")],
                     hard_gate_violations=threshold_violations(hard, t))
        blob = {"candidate_id": cand_id,
                "thresholds": {"config_hash": t.config_hash,
                               "source": str(Path(args.thresholds)
                                             if args.thresholds
                                             else newest_thresholds_file()
                                             or "code defaults")},
                "pairings": paired,
                "hard_gate_violations": threshold_violations(hard, t),
                "decision": dec,
                # 循环特有：证据与两臂清单（replay 只读上面四项）
                "factor": factor, "applied_flags": extra,
                "skipped_factors": skipped, "data_factor_note": data_note,
                "baseline_runs": [str(r) for r in base_runs],
                "candidate_runs": [str(r) for r in cand_runs],
                "champ_by_seed": champ_by_seed, "cand_by_seed": cand_by_seed,
                "hard_gate": hard,
                # 等步数对照的证据：两臂**实测**总步数（来自 checkpoint）
                "steps_by_arm": {
                    "baseline": base_steps or None,
                    "candidate": {
                        str(s): (steps_per_epoch(
                            (_ckpt_train_args(exp_dir(args.run_id) / f"round{rnd}"
                                              / f"seed{s}"
                                              / "checkpoint_last.pt")
                             .get("n_train") or 0),
                            args.batch) * int(cand_epochs))
                        for s in args.seeds}},
                "timing_suspect": timing_notes,
                # 评估的是哪个阶段的权重 + 平凡基线参照（见 eval matrix）：
                # 没有这两项，"0.42 的 road_iou"看不出是不是"全预测成路面"白送的
                "eval_checkpoint": "checkpoint_last.pt",
                "worst_frames_by_seed": worst_by_seed,
                "epochs": int(args.epochs),
                "road_iou_trivial_all_road": (trivial.get("all_road")
                                              if trivial else None),
                "road_iou_trivial_all_background": (trivial.get("all_bg")
                                                    if trivial else None),
                "equal_steps_requested": bool(args.equal_steps),
                "candidate_epochs": int(cand_epochs),
                "max_train_frames": int(equal_cap)}
        (exp_dir(args.run_id) / f"decision_{cand_id}.json").write_text(
            json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
        # 两臂命令落盘：逐项 diff 的证据（输入真的变了）
        (exp_dir(args.run_id) / f"round{rnd}_commands.json").write_text(
            json.dumps({"baseline": {str(s): train_cmd(
                args, base_runs,
                exp_dir(args.run_id) / "baseline" / f"seed{s}", s, [])
                for s in args.seeds},
                "candidate": {str(s): train_cmd(
                    args, cand_runs,
                    exp_dir(args.run_id) / f"round{rnd}" / f"seed{s}", s,
                    extra) for s in args.seeds},
                "data_factor_note": data_note}, indent=1,
                ensure_ascii=False), encoding="utf-8")
        # 每个候选一份**独立的事件流**：阶段机描述的是"一个候选的一生"，
        # 而"轮"是循环的迭代。共用一个流时，第一轮 rejected 是终态，第二轮
        # 就无法再进 evaluating（实测被状态机拦下）——分开才对。
        clog = EventLog(exp_dir(args.run_id) / "candidates" / cand_id)
        if clog.last() is None:
            clog.append(_ev(args.run_id, "queued", "start", candidate=cand_id,
                            note="离线循环候选"))
            clog.append(_ev(args.run_id, "auditing", "ok", candidate=cand_id,
                            note="数据与硬门槛输入检查"))
            clog.append(_ev(args.run_id, "training", "done", candidate=cand_id,
                            note=f"seeds={list(args.seeds)} epochs="
                                 f"{args.epochs}"))
        # 指标名必须跟着**实际配对的那个指标**：road-only 时它是 road_iou。
        # 写死 line_iou 会让看板/事件流显示一个并不存在的量（实测：road-only 运行的
        # 曲线被标成 line_iou，而模型那一路根本没有 line 监督）。
        clog.append(_ev(args.run_id, "evaluating", "evaluated",
                        candidate=cand_id,
                        metrics={pair_metric: metric(sum(cand) / len(cand),
                                                     "iou")},
                        note="; ".join(dec["reasons"])[:200]))
        clog.append(_ev(args.run_id, dec["decision"], "decided",
                        candidate=cand_id,
                        note="; ".join(dec["reasons"])[:200]))
        if trivial and road_only:
            _tr = trivial.get("all_road")
            _gap = (None if _tr is None or not cand
                    else round(sum(cand) / len(cand) - float(_tr), 4))
            print(f"[rounds] road_iou 平凡基线（全预测成路面）= {_tr}"
                  f"，候选领先 {_gap}"
                  + ("（差距很小：这不是『学会了路面』的证据）"
                     if (_gap is not None and abs(_gap) < 0.05) else ""))
        print(f"[rounds] 第 {rnd + 1}/{args.rounds} 轮 {cand_id} -> "
              f"{dec['decision']}（{pair_metric} "
              f"{[round(x, 4) for x in cand]} vs 基线 "
              f"{[round(x, 4) for x in champ]}，"
              f"identity {hard['candidate_identity_rate']}）")
        history.append(RoundRecord(round_index=rnd, candidate_id=cand_id,
                                   decision=dec["decision"],
                                   reasons=dec["reasons"]))
        stop = should_stop(cfg, history, gpu_minutes_today=0.0,
                           candidates_used=rnd + 1)
        if stop["stop"]:
            log.append(_ev(args.run_id, "paused", "stopped",
                           note="; ".join(stop["reasons"])[:200]))
            print(f"[rounds] 自动停止：{stop['reasons']}")
            break
    if args.plan_only:
        print("[plan] --plan-only：只列两臂命令，未训练、未写判定")
        return 0
    print(f"[rounds] 完成 {len(history)} 轮；判定文件在 {exp_dir(args.run_id)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="T14 阶段 D 离线循环：审计/提议/评估/重放/dry-run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("audit", help="建数据版本并过准入门")
    s.add_argument("--run-id", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--dev-group", action="append", default=[])
    s.add_argument("--final-group", action="append", default=[])
    s.add_argument("--allow-road-only", action="store_true",
                   help="允许只有路面真值的实验（标线通道屏蔽）")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("propose", help="错误分桶 + 有限提议")
    s.add_argument("--run-id", required=True)
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--n-train-frames", type=int, default=0)
    s.add_argument("--available-runs", nargs="*", default=None,
                   help="可用的、尚未入训的数据组（场景配比族的输入；为空时"
                        "该族不产出提议，而是报告需要新数据）")
    s.add_argument("--champion-steps", type=int, default=None)
    s.add_argument("--champion-epochs", type=int, default=3)
    s.add_argument("--max-proposals", type=int, default=3)
    s.set_defaults(func=cmd_propose)

    s = sub.add_parser("evaluate", help="成对 seed + 硬门槛 -> 判定")
    s.add_argument("--run-id", required=True)
    s.add_argument("--candidate-id", required=True)
    s.add_argument("--pairings", required=True,
                   help="JSON：{metric: {champion:[...], candidate:[...], "
                        "lower_is_better:bool}}")
    s.add_argument("--hard-gate", default=None,
                   help="JSON：{metric: value}，缺测项按 UNKNOWN 处理（不算通过）")
    s.add_argument("--thresholds", default=None)
    s.add_argument("--production-mismatch", action="store_true")
    s.add_argument("--definition-drift", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("replay", help="重放判定：相同输入必须相同决策")
    s.add_argument("--run-id", required=True)
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("rounds",
                       help="连续 N 轮离线循环（提议→训练→评估→判定→停止）")
    s.add_argument("--run-id", required=True)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--runs", nargs="+", required=True, help="训练数据目录")
    s.add_argument("--baseline-runs", nargs="+", default=None,
                   help="基线臂的数据列表（默认同 --runs）。第 0 轮用同一配方、"
                        "同一 seed、同一开发集训练基线臂并落盘 champion.json；"
                        "绝不用本轮候选自己的值当对照")
    s.add_argument("--trainer-script", default="m5_train_seg.py",
                   help="训练入口（默认 scripts/m5_train_seg.py；测试用桩）")
    s.add_argument("--plan-only", action="store_true",
                   help="只列出两臂命令并退出（不训练、不写判定）——用于逐项 diff")
    s.add_argument("--resume", action="store_true",
                   help="断点恢复：已有 decision_*.json 的轮次不重跑，"
                        "champion.json 直接复用（中断后接着跑）")
    s.add_argument("--equal-steps", action="store_true",
                   help="等步数对照：按基线臂的实测总步数缩放候选臂的 epochs"
                        "（数据臂加帧时避免把\"训练更久\"读成\"数据更好\"）；"
                        "真实步数仍以两臂 checkpoint 记录为准")
    s.add_argument("--allow-road-only", action="store_true",
                   help="标线真值不可用时只做路面通道实验（否则审计判 "
                        "needs_review 并停止）")
    s.add_argument("--eval-runs", nargs="+", required=True,
                   help="开发集目录（最终集不参与搜索）")
    s.add_argument("--proposals", default=None, help="propose 产出的 JSON")
    s.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--split", default="tail")
    s.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    s.add_argument("--thresholds", default=None)
    s.add_argument("--config", default=str(exp_dir("") / "loop_config.json"))
    s.add_argument("--hard-recall", type=float, default=0.70,
                   help="已废弃：硬门输入取自本轮实测，不再用命令行值当测量"
                        "（保留仅为兼容旧命令行）")
    s.add_argument("--hard-precision", type=float, default=0.40,
                   help="已废弃：同 --hard-recall")
    s.add_argument("--hard-p95", type=float, default=45.0,
                   help="已废弃：推理 p95 取自本轮实测（缺测记 UNKNOWN）")
    s.add_argument("--timeout-s", type=int, default=3600)
    s.set_defaults(func=cmd_rounds)

    s = sub.add_parser("run", help="一轮：资源门 + 锁 + 命令清单（默认 dry-run）")
    s.add_argument("--run-id", required=True)
    s.add_argument("--candidate-id", default="")
    s.add_argument("--dataset-id", default="")
    s.add_argument("--config", default=str(exp_dir("")/"loop_config.json"))
    s.add_argument("--dry-run", action="store_true", default=True)
    s.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    s.add_argument("--gpu-minutes-today", type=float, default=0.0)
    s.add_argument("--user-active", action="store_true")
    s.add_argument("--train-args", default="")
    s.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
