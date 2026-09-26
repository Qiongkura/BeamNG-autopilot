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
    InstanceLock, LoopConfig, MachineLease, RoundRecord, RunClock, plan_once,
    probe_resources, resource_gate, should_stop, user_activity_probe,
)
from beamng_autopilot.experiments.events import Event, EventLog, metric  # noqa: E402
from beamng_autopilot.experiments.checkpoint import file_sha16 as _ckpt_sha16
from beamng_autopilot.experiments.checkpoint import git_commit as _git_commit
from beamng_autopilot.experiments.final_set import (  # noqa: E402
    confirmation_check,
)
from beamng_autopilot.experiments.gates import (  # noqa: E402
    HARD_CHECKS, SCENE_HARD_FIELDS, Thresholds, decide, hard_split,
    legacy_replay_note, missing_metrics_for, paired_compare,
    per_seed_gate_violations, per_seed_missing, scene_count_violations,
    scene_report, threshold_violations,
)
from beamng_autopilot.experiments.manifest import (  # noqa: E402
    DatasetManifest, dir_group,
)
from beamng_autopilot.experiments.proposer import (  # noqa: E402
    bucket_errors, classify_round_outcome, factor_activity,
    load_eval_artifacts, needs_review, propose,
)
from beamng_autopilot.experiments.spatial import (  # noqa: E402
    SPATIAL_BUFFER_M, group_exposure_leak, spatial_conflicts,
)
from beamng_autopilot.experiments.credentials import (  # noqa: E402
    read_dir_credentials,
)
from beamng_autopilot.experiments.labels import (  # noqa: E402
    PAINT_SOURCE_RANK,
)
from beamng_autopilot.experiments.negative_scenes import (  # noqa: E402
    negative_training_eligibility,
)
from beamng_autopilot.experiments.protocol import (  # noqa: E402
    COVERAGE_GATE_FROZEN, effective_source, eligibility, protocol_blob,
    snapshot_hash, sources_can_promote, verify_snapshot,
)
from beamng_autopilot.experiments.collection import (  # noqa: E402
    CollectSpec, close_started_game, collect_command, collection_proposal,
    collector_python, game_pids, game_running_probe, merge_proposals,
    paint_frame_priority, preflight, python_can_import, verify_collection,
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


def machine_lease() -> MachineLease:
    """机器级学习资源租约（方案 G02）：同一 GPU 上只允许一个重型实验。

    单独一个函数是为了测试能指到 tmp；真实路径固定。
    """
    return MachineLease(Path(config.LOGS_DIR) / "experiments"
                        / "machine_lease.json")


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
    # 训练准入看"可用的漆线监督"（弱监督/agent 档 usable=True），晋级才看 valid。
    # 原来只查 valid -> agent 档研究训练被误挡（实测：agentline 池子被判
    # "标线真值不可用 -> needs_review"）。
    paint_usable = coverage.get("train", {}).get("paint_usable_frames", 0)
    if not trainable:
        log.append(_ev(args.run_id, "auditing", "no_trainable_data",
                       dataset=mf.dataset_id,
                       note="没有任何通道有可用真值：停止，不训练"))
        print("[autoloop] 审计失败：没有可用真值，停止")
        return 2
    _line_sup = bool(paint_sources_from(args))
    if not paint_usable and not args.allow_road_only and not _line_sup:
        # 准入门：**连可用的漆线监督都没有**才拒训（有 Tech annotation 但没标线类，
        # 或来源不可靠且未复核）-> 进复核队列。
        log.append(_ev(args.run_id, "needs_review", "paint_truth_missing",
                       dataset=mf.dataset_id,
                       note=("没有任何可用的标线监督（paint_usable=0）：标线通道"
                             "无可学内容，需人工修订或单独验证的模拟器真值；"
                             "用 --allow-road-only 可只做路面通道实验")))
        print("[autoloop] 审计：没有可用的标线监督 -> needs_review（未训练）")
        return 3
    if paint_usable and not paint_ok:
        # 弱/agent 档：**可以训练**（研究），但真值不是门槛真值 -> 晋级由来源资格挡。
        log.append(_ev(args.run_id, "auditing", "paint_truth_weak",
                       dataset=mf.dataset_id,
                       note=(f"标线监督是弱/agent 档（usable={paint_usable}、"
                             f"valid=0）：可训练，**不能**当门槛真值或晋级参考")))
        print(f"[autoloop] 审计：标线监督为弱/agent 档（usable={paint_usable}、"
              f"valid=0）——可训练，晋级由来源资格挡住")
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
    # 线通道被屏蔽（road-only）时线损失键不会生效：提议阶段就挡住，别把
    # "参数带着、实际不起作用"的因子送进轮次（方案 v2 §S5 / T14）。
    _line_sup = not getattr(args, "allow_road_only", False)
    props = propose(buckets=buckets, dataset=dataset, champion=champ,
                    max_proposals=args.max_proposals, history=history,
                    available_runs=avail, blocked=blocked,
                    line_supervision=_line_sup)
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    blob = {"buckets": [b.as_dict() for b in buckets],
            "needs_review": review,
            "line_supervision": _line_sup,
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
    # 缺测与违反分两路（与 rounds 同规则，方案 §10.3）：硬门里没测到的项是
    # "证据缺失" -> needs_evidence；原来合并成 violations 会判成 rejected。
    _hs = hard_split(hard, t)
    violations = _hs["violations"]
    missing = [k for k, v in compared.items() if not v.get("n")]
    missing += [f"{n}: UNKNOWN (hard gate needs a measurement)"
                for n in _hs["missing"]]
    decision = decide(pairings=compared, thresholds=t,
                      missing_metrics=missing,
                      production_mismatch=bool(args.production_mismatch),
                      definition_drift=bool(args.definition_drift),
                      hard_gate_violations=violations)
    # 与 rounds 共用同一条资格规则（方案 G03/G10）：evaluate 也不能被旁路
    _src_res = resolve_paint_sources(args)
    _research = bool(getattr(args, "research_arm", False)) or \
        bool(_src_res["research_only"])
    for _n in _src_res["notes"]:
        print(f"[evaluate] 来源凭证：{_n}")
    decision = _block_research_promotion(decision, _research)
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    blob = {"candidate_id": args.candidate_id,
        "research_only": bool(_research),
        "paint_source_resolution": _src_res,
        "thresholds": {
        "config_hash": t.config_hash,
        "source": str(Path(args.thresholds) if args.thresholds
                      else newest_thresholds_file() or "code defaults")},
        "protocol": _protocol_snapshot(t),
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
    legacy: list = []         # 早于 v5 计数契约的判定（不能按新分母重判）
    proto: list = []          # 每个判定文件的协议快照自查结果
    tampered: list = []       # 快照内容与记录哈希不符（被改过/截断）
    for p in decs:
        blob = json.loads(p.read_text(encoding="utf-8"))
        # 协议快照自查（方案 §10.3）：内容与记录的哈希是否自洽、是否就是
        # 当前口径。**旧协议留档**要明说，不能让人误以为还按新口径复现过。
        snap = blob.get("protocol")
        if snap:
            ver = verify_snapshot(snap)
            proto.append({"file": p.name, "recorded": ver["recorded_hash"],
                          "current": ver["current_hash"],
                          "ok": ver["ok"],
                          "matches_current": ver["matches_current_protocol"],
                          "reasons": ver["reasons"]})
            if not ver["ok"]:
                tampered.append(p.name)
        t = thresholds(
            Path(blob["thresholds"]["source"])
            if Path(blob["thresholds"]["source"]).exists() else None)
        # v5：早于计数契约的判定**不能**按新分母重判，也不许补 0——显式说明。
        _note = legacy_replay_note(blob)
        if _note:
            legacy.append({"file": p.name, "note": _note})
            continue
        compared = {name: spec for name, spec in blob["pairings"].items()}
        # 缺测清单**优先用判定文件里落盘的那份**：它可能含逐 seed/分场景缺测
        # 以及"新硬门未标定"这类不由 pairings 推导出来的条目；旧判定文件
        # 没有这个键时才按 pairings 重推（旧行为）。
        if blob.get("missing_metrics") is not None:
            _missing = list(blob["missing_metrics"])
        else:
            _missing = [k for k, v in compared.items() if not v.get("n")]
        again = decide(pairings=compared, thresholds=t,
                       missing_metrics=_missing,
                       hard_gate_violations=blob["hard_gate_violations"])
        # 重放必须包含"研究来源后置降级"（方案 G10）：判定文件里存了
        # research_only / paint_source_resolution，重放按同一结论走。
        _research = bool(blob.get("research_only"))
        if not _research and blob.get("paint_source_resolution"):
            _research = bool(
                (blob.get("paint_source_resolution") or {}).get("research_only"))
        again = _block_research_promotion(again, _research)
        if again == blob["decision"]:
            same += 1
        else:
            diff.append({"file": p.name, "was": blob["decision"],
                         "now": again})
    for pr in proto:
        if pr["ok"] and pr["matches_current"]:
            continue
        if not pr["ok"]:
            print(f"  {pr['file']}: 协议快照与记录的哈希不符（判定文件被改过？）："
                  f"{pr['reasons']}")
            continue
        print(f"  {pr['file']}: 按**旧协议** {pr['recorded']} 留档（当前 "
              f"{pr['current']}）——结论有效，但不能与当前口径混比")
    for lg in legacy:
        print(f"  {lg['file']}: {lg['note']}")
    print(f"[autoloop] 重放 {len(decs)} 个判定：相同 {same}，不同 {len(diff)}"
          + (f"，早于 v5 计数契约 {len(legacy)}" if legacy else "")
          + (f"，协议不一致 {len(tampered)}" if tampered else ""))
    for d in diff:
        print(f"  {d['file']}: 决策或理由不一致 -> 判定不可复现")
    return 0 if not diff and not tampered else 1


# ---------------------------------------------------------------------------
# 无人值守采集（用户 2026-09-25 授权"允许无人值守启动 Tech 采集"）。
#
# 三条纪律，缺一条都不算接通：
#   * 启动前过前置检查——游戏已在跑就不抢会话（可能有人在开），显存/磁盘
#     不够就不启动；
#   * 采集后过**身份审计**——缺 map_name/source_id 的采集一律拒收，绝不静默
#     进训练（3h 轮 48 帧因身份丢失被判死的教训：只止损，不补票）；
#   * 游戏时间记进每日 GPU 预算——游戏也是 GPU 负载，不记就是把上限写在纸上。
# ---------------------------------------------------------------------------
def collect_root() -> Path:
    """采集产物根目录（单独一个函数是为了测试能指到 tmp，不写进仓库 logs/）。"""
    return Path(config.LOGS_DIR) / "m5_seg"


def _pick(cli_value, cfg_value, cast):
    """命令行优先；``None`` 表示"没给"，才回落到配置（同 dry_run 的口径）。"""
    return cfg_value if cli_value is None else cast(cli_value)


def spec_from(cfg, args) -> CollectSpec:
    g = lambda n: getattr(args, n, None)                    # noqa: E731
    roles = _pick(g("collect_roles"), cfg.collect_roles, tuple)
    return CollectSpec(
        frames=int(_pick(g("collect_frames"), cfg.collect_frames, int)),
        step_m=float(_pick(g("collect_step_m"), cfg.collect_step_m, float)),
        roles=tuple(roles or ()),
        follow_road=bool(_pick(g("collect_follow_road"),
                               cfg.collect_follow_road, bool)),
        runtime=str(cfg.collect_runtime),
        save_annotation=bool(_pick(g("collect_save_annotation"),
                                   cfg.collect_save_annotation, bool)),
        step=int(_pick(g("collect_step"), cfg.collect_step, int)),
        map_name=str(_pick(g("collect_map"), cfg.collect_map, str)),
        attach=bool(_pick(g("collect_attach"), False, bool)),
        teleport=tuple(_pick(g("collect_teleport"), cfg.collect_teleport,
                             lambda v: tuple(v) if v else ()) or ()),
        timeout_s=int(_pick(g("collect_timeout_s"), cfg.collect_timeout_s,
                            int)),
    )


def probe_machine() -> tuple:
    """``(游戏是否在跑, 空闲显存 MB, 磁盘 GB)``；任何一项探测失败都不编数。"""
    free_vram = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / 2 ** 20
    except Exception:                                     # noqa: BLE001
        free_vram = 0.0
    try:
        import shutil as _sh
        free_disk = _sh.disk_usage(str(ROOT)).free / 2 ** 30
    except Exception:                                     # noqa: BLE001
        free_disk = 0.0
    return game_running_probe(), free_vram, free_disk


def collect_once(args, cfg, *, log) -> tuple:
    """无人值守采集一次。返回 ``(record, rc)``；``rc != 0`` 时调用方必须停下。

    rc 约定：0 通过；6 前置检查不通过（**没启动**游戏）；7 采集进程失败或
    身份审计拒收。三种情况分别写事件，日志里能分清"没跑"和"跑了但不要"。
    """
    run_id = args.run_id
    spec = spec_from(cfg, args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = collect_root() / f"collect_{run_id}_{stamp}"
    running, free_vram, free_disk = probe_machine()
    # 解释器：系统 Python 通常没有 beamngpy（实测踩到：白起一局，rc=1，然后被
    # 身份审计拒收）。启动前先问它能不能 import，问不过就不启动。
    py, py_src = collector_python(ROOT, configured=cfg.python)
    py_ok = python_can_import(py)
    pf = preflight(spec=spec, game_running=running, free_vram_mb=free_vram,
                   free_disk_gb=free_disk,
                   force=bool(getattr(args, "collect_force", False)),
                   collector_python_ok=py_ok, collector_python=py)
    print(f"[collect] 前置检查 ok={pf['ok']} 游戏在跑={running} "
          f"空闲显存={free_vram:.0f}MB 磁盘={free_disk:.1f}GB "
          f"解释器={py}（{py_src}）beamngpy={py_ok}")
    for r in pf["reasons"]:
        print(f"[collect]   阻止：{r}")
    for w in pf["warnings"]:
        print(f"[collect]   提示：{w}")
    if not pf["ok"]:
        log.append(_ev(run_id, "auditing", "collect_blocked",
                       note="; ".join(pf["reasons"])[:200]))
        return None, 6
    cmd = collect_command(python=py,
                          script=ROOT / "scripts" / "m5_collect_seg_ring.py",
                          spec=spec, out_dir=out_dir)
    # 采集会自己起一局游戏；记下"采集前有哪些游戏进程"，收尾只关
    # **本次新出现**的那些（用户自己开着的会话永远不在差集里）
    pids_before = game_pids()
    print(f"[collect] 采集前游戏进程：{pids_before}")
    print(f"[collect] 采集目录：{out_dir}")
    print("[collect] 命令：" + " ".join(str(c) for c in cmd))
    log.append(_ev(run_id, "auditing", "collecting",
                   note=(f"启动采集 {out_dir.name} frames={spec.frames} "
                         f"roles={list(spec.roles)}")[:200]))
    log_path = exp_dir(run_id) / f"collect_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mark_gpu_start(run_id)
    t0 = time.time()
    launched_after = float(t0)      # 所有权校验用：这个时刻之后创建的进程才归我们
    rc, tail = 0, ""
    try:
        # 输出**写文件**而不是管道：采集会拉起游戏，游戏会继承子进程的
        # stdout；用 capture_output 时那个管道的写端被游戏握着，采集进程
        # 已经退出、120 帧也落了盘，父进程却还在 communicate() 上等 EOF
        # （实测卡了 30 分钟直到超时）。文件句柄不会被孙子进程拖住。
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(cmd, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True,
                                  timeout=int(spec.timeout_s))
        rc = int(proc.returncode)
    except subprocess.TimeoutExpired:
        rc, tail = 124, f"采集超时 {spec.timeout_s}s"
    except Exception as exc:                              # noqa: BLE001
        rc, tail = 125, f"采集启动失败 {type(exc).__name__}: {exc}"
    try:                    # 采集自己的日志尾巴（完整日志留在 log_path）
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-400:]
    except Exception:                                     # noqa: BLE001
        pass
    used = (time.time() - t0) / 60.0
    settled = settle_gpu_minutes(run_id)
    total = settled if settled > 0 else add_gpu_minutes(run_id, used)
    print(f"[collect] rc={rc} 用时 {used:.1f} min（今日累计 {total:.1f} min）")
    if tail.strip():
        print("[collect] 采集输出尾部：" + tail.replace("\n", " | ")[-320:])
    # 收尾：关掉本次采集启动的游戏。实测缺陷（2026-09-25）：采集器只关连接、
    # 进程留着，一局游戏 4.4 GB 显存 + 持续吃 CPU/GPU 挂了 1.5 小时，把后面
    # 几轮实验的吞吐拖慢一个量级、推理 p95 漂到 7.6 倍。
    game_after = close_started_game(pids_before,
                                    launched_after=launched_after)
    _ga_closed = game_after.get("closed")
    _ga_killed = game_after.get("killed")
    _ga_note = game_after.get("reason") or game_after.get("note") or ""
    print(f"[collect] 收尾关游戏：closed={_ga_closed} "
          f"killed={_ga_killed} {_ga_note}")
    rep = verify_collection(
        out_dir, expected_roles=spec.roles,
        expected_frames=int(spec.frames) * max(1, len(spec.roles)))
    rep.update({"game_pids_before": (sorted(pids_before)
                                 if pids_before is not None else None),
                "game_after_collect": game_after,
                "rc": rc, "cmd": [str(c) for c in cmd], "stamp": stamp,
                "collector_python": py, "collector_python_source": py_src,
                "spec": spec.as_dict(), "gpu_minutes": round(used, 2),
                "gpu_minutes_today": round(float(total), 2),
                "collect_log": str(log_path),
                "preflight": pf, "output_tail": tail[-400:]})
    # 覆盖范围（方案 W2：帧数不等于独立样本）：车辆几乎没动的采集是近重复集，
    # 身份/计数审计看不出来，所以单独报出来。
    try:
        from beamng_autopilot.experiments.spatial import group_spread
        _mf = DatasetManifest.build([Path(rep["out_dir"]) / r
                                     for r in spec.roles], root=ROOT)
        _spread = group_spread(_mf.records, expect_step_m=spec.step_m)
        rep["group_spread"] = _spread
        for _sp in _spread:
            _ratio = _sp.get("step_ratio")
            print(f"[collect] 覆盖范围 {_sp['group']}: 跨度 "
                  f"{_sp.get('extent_m')} m / {_sp.get('n_positioned')} 帧，"
                  f"中位间距 {_sp.get('median_step_m')} m"
                  + (f"（期望 {_sp.get('expected_step_m')} m，比值 {_ratio}）"
                     if _ratio is not None else ""))
            if _ratio is not None and float(_ratio) < 0.5:
                print("[collect]   警告：实测间距远小于期望步长——这批帧是"
                      "近重复集（车辆几乎没动），不能当'新增了这么多独立样本'")
            _cov = _sp.get("coverage_ratio")
            if _cov is not None and float(_cov) < 0.3:
                print(f"[collect]   注意：路径在小范围折返（覆盖比 {_cov}）——"
                      "帧与帧不重复，但整组只覆盖很小一片区域，"
                      "不能当'新场景覆盖'")
    except Exception as exc:                              # noqa: BLE001
        print(f"[collect] 覆盖范围未计算（不掩盖）：{type(exc).__name__}: {exc}")
    print(f"[collect] 身份审计 ok={rep['ok']} map={rep['map_name']!r} "
          f"source={rep['source_id']!r} 帧={rep['frames_total']} "
          f"各视角={rep['roles']}")
    for r in rep["reasons"]:
        print(f"[collect]   拒收：{r}")
    for w in rep["warnings"]:
        print(f"[collect]   提示：{w}")
    meta_p = Path(rep["meta_path"])
    if meta_p.exists():
        try:
            # 每个视角各取漆线像素最多的 15 帧：只看总榜会被前向视角占满，
            # 而左右侧视角是"标线两侧"证据的来源（横向研究要用）。
            _meta = json.loads(meta_p.read_text(encoding="utf-8"))
            prio: list = []
            for _v in spec.roles:
                prio += paint_frame_priority(_meta, top=15, view=_v)
            rq = exp_dir(run_id) / f"review_queue_collect_{stamp}.json"
            rq.write_text(json.dumps(
                {"why": "各视角按漆线像素降序取前 15 帧：人工修订从这里开始。"
                        "引擎不给 line 类，但「哪里看得见漆线」是实测的",
                 "out_dir": str(out_dir), "per_view": 15, "frames": prio},
                indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[collect] 复核队列（漆线像素最多 30 帧）-> {rq}")
            rep["review_queue"] = str(rq)
        except Exception as exc:                          # noqa: BLE001
            print(f"[collect] 复核队列生成失败（不掩盖）："
                  f"{type(exc).__name__}: {exc}")
        # 选样（方案 §7.5–7.7）：给本次采集的帧打分并挑一小批。规则在
        # experiments/selection.py 里（纯函数、有反例测试）：
        #   * 未复核的帧只能进复核队列，**不能**当 hard negative；
        #   * 同组帧数上限（相邻帧不是独立样本）、同相机占比、覆盖兜底。
        try:
            import numpy as _np
            from beamng_autopilot.experiments.selection import (
                frame_score, pick_batch, pool_from_dirs)

            def _kinds_of(npz_path):
                """只看**正例**证据：标了漆线像素 -> line。

                绝不产出 ``no_line``：弱标签没画线不等于"确认没有线"
                （方案 §6.2/A2：未标注区域不是负例）。
                """
                try:
                    z = _np.load(npz_path)
                    lab = _np.asarray(z["label"])
                    return ["line"] if bool((lab == 2).any()) else []
                except Exception:                        # noqa: BLE001
                    return []

            _pool = pool_from_dirs([out_dir / r for r in spec.roles],
                                   kinds_of=_kinds_of)
            _scored = [frame_score(r) for r in _pool]
            _batch = pick_batch(_scored)
            _sel_p = exp_dir(run_id) / f"selection_collect_{stamp}.json"
            _sel_p.write_text(json.dumps(
                {**_batch, "why": "选样评分：未知程度/模型分歧/已验证错误/场景缺口；"
                                  "未复核帧只进复核队列，不教负例",
                 "pool_from": [str(out_dir / r) for r in spec.roles]},
                indent=1, ensure_ascii=False), encoding="utf-8")
            rep["selection"] = str(_sel_p)
            rep["_selection_blob"] = {
                "n_items": _batch["n_items"], "n_pool": _batch["n_pool"],
                "n_trainable": _batch["n_trainable"],
                "n_review_only": _batch["n_review_only"],
                "reasons": _batch["reasons"],
                "items": [{"frame": it["frame"], "view": it["view"],
                           "group": it["group"], "score": it["score"],
                           "route": it["route"]} for it in _batch["items"]]}
            print(f"[collect] 选样：池 {_batch['n_pool']} 帧 -> 本批 "
                  f"{_batch['n_items']} 帧（可训练 {_batch['n_trainable']}，"
                  f"仅复核 {_batch['n_review_only']}）-> {_sel_p}")
            for _r in _batch["reasons"]:
                print(f"[collect]   选样：{_r}")
        except Exception as exc:                          # noqa: BLE001
            print(f"[collect] 选样失败（不掩盖，不阻塞采集）："
                  f"{type(exc).__name__}: {exc}")
    if rc != 0:
        log.append(_ev(run_id, "auditing", "collect_failed",
                       note=f"采集进程 rc={rc}：{tail[-120:]}"[:200]))
    elif not rep["ok"]:
        log.append(_ev(run_id, "needs_review", "collect_rejected",
                       note="; ".join(rep["reasons"])[:200]))
    else:
        prop = collection_proposal(
            role_dirs=[out_dir / r for r in spec.roles],
            candidate_id=f"collect-{stamp}", stamp=stamp,
            selection=(rep.get("_selection_blob") or None))
        pp = exp_dir(run_id) / f"proposals_collect_{stamp}.json"
        pp.write_text(json.dumps(prop, indent=1, ensure_ascii=False),
                      encoding="utf-8")
        rep["proposal_path"] = str(pp)
        log.append(_ev(run_id, "auditing", "collect_ok",
                       candidate=f"collect-{stamp}",
                       note=(f"{rep['frames_total']} 帧 {rep['map_name']}/"
                             f"{rep['source_id']}")[:200]))
    rp = exp_dir(run_id) / f"collect_{stamp}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                  encoding="utf-8")
    print(f"[collect] 记录 -> {rp}")
    print(f"[collect] 采集日志 -> {log_path}")
    return rep, (0 if (rc == 0 and rep["ok"]) else 7)


def cmd_collect(args) -> int:
    """单独跑一次无人值守采集（不起训练），供 runbook/验收单独调用。"""
    cfg_path = Path(args.config)
    cfg = (LoopConfig.load(cfg_path) if cfg_path.exists() else LoopConfig())
    log = _log(args.run_id)
    lock = InstanceLock(exp_dir(args.run_id) / "controller.lock")
    held = lock.acquire()
    if not held["acquired"]:
        print(f"[collect] 单实例锁：{held['reason']}")
        return 4
    try:
        rep, rc = collect_once(args, cfg, log=log)
    finally:
        lock.release()
    if rep is None:
        print("[collect] 未启动：前置检查不通过（游戏可能有人在开）")
        return rc
    print(f"[collect] 结果: ok={rep['ok']} rc={rep['rc']} "
          f"帧={rep['frames_total']} 目录={rep['out_dir']}")
    return rc



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
    _machine_today = machine_gpu_minutes_today()
    used_today = max(float(args.gpu_minutes_today or 0.0),
                     gpu_minutes_today(args.run_id), _machine_today)
    print(f"[autoloop] GPU 今日：本 run {gpu_minutes_today(args.run_id):.1f} min，机器合计 {_machine_today:.1f} min（上限取机器合计）")
    # 用户活动信号：默认**真的去探测**（方案 §8.2：不能用写死的 false 冒充
    # 检测）。`--user-active` 是显式声明（测试/人工指定），会标明"未测量"。
    if getattr(args, "user_active", None) is None:
        _act = user_activity_probe()
    else:
        _act = {"active": bool(args.user_active), "idle_s": None,
                "source": "caller-declared (--user-active)",
                "why": "declared on the command line, not measured"}
    st = probe_resources(disk_path=ROOT, gpu_minutes_today=used_today,
                         user_active=_act["active"])
    st.user_idle_s = _act.get("idle_s")
    st.user_activity_source = _act.get("source") or "unknown"
    print(f"[autoloop] 用户活动信号：active={st.user_active} "
          f"idle={st.user_idle_s}s 来源={st.user_activity_source}"
          f"（{_act.get('why')}）")
    gate = resource_gate(cfg, st)
    # 观测落盘（方案 W6"资源与恢复"视图 + §8.2"记录原始检测值和状态变更"）：
    # 每次启动覆盖写一份**最新**观测，看板据此显示活动信号与墙钟；
    # 未接入时写 None（页面显示"未接入"），不写 false。
    _clock_probe = RunClock(exp_dir(args.run_id) / "run_state.json")
    try:
        _wall_now = _clock_probe.minutes()
        (exp_dir(args.run_id) / "resource_state.json").write_text(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_active": st.user_active,
            "user_idle_s": st.user_idle_s,
            "user_activity_source": st.user_activity_source,
            "user_activity_why": _act.get("why"),
            "wall_minutes": round(_wall_now, 2),
            "max_wall_minutes": cfg.max_wall_minutes,
            "daily_gpu_minutes": cfg.daily_gpu_minutes,
            "gpu_minutes_today": used_today,
            "free_vram_mb": st.free_vram_mb, "free_disk_gb": st.free_disk_gb,
            "allowed": gate["allowed"], "reasons": gate["reasons"],
            "warnings": gate["warnings"],
            "note": "每次入口启动覆盖写：这是**最新观测**，不是历史",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as _e:                    # noqa: BLE001
        print(f"[autoloop] 资源观测未落盘（不影响运行）：{_e}")
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
    # 机器级租约（方案 G02）：不同 run 也不能同时做重型实验。
    # 同进程重复获取是安全的（already_held），run -> rounds 不会自锁。
    lease = machine_lease()
    _force = bool(getattr(args, "force_lease", False))
    held_lease = lease.acquire(force=_force)
    if not held_lease["acquired"]:
        print(f"[autoloop] 机器级租约：{held_lease['reason']}")
        log.append(_ev(args.run_id, "paused", "lease_blocked",
                       note=held_lease["reason"][:200]))
        lock.release()
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
                reasons=blob["decision"]["reasons"],
                # 归因字段随判定文件走：只有"有意义的无收益"计入停止连胜，
                # 无效因子/缺标注/资格失败各有原因（旧文件无此字段则留空）
                outcome=str(blob.get("round_outcome") or "")))
        # 单次连续运行的墙钟（方案 §8.1）：起点落在 run 目录，重启不重置
        clock = RunClock(exp_dir(args.run_id) / "run_state.json")
        wall = clock.minutes()
        stop = should_stop(cfg, hist, gpu_minutes_today=used_today,
                           candidates_used=len(hist), wall_minutes=wall)
        print(f"[autoloop] 停止条件：stop={stop['stop']} "
              f"连续无收益={stop['no_gain_streak']} "
              f"本次连续运行={wall:.1f} min"
              f"（上限 {cfg.max_wall_minutes:.0f} min，<=0 为不限）")
        for r in stop["reasons"]:
            print(f"  - {r}")
        if not gate["allowed"]:
            print("[autoloop] 资源门未通过：本轮不执行")
            if not cfg.dry_run:
                log.append(_ev(args.run_id, "paused", "resource_blocked",
                               note="; ".join(gate["reasons"])[:200]))
                return 5
        # 停止条件必须是**启动硬门**（方案 G01）：原来只打印 stop=true，
        # 随后照样去采集/训练。采集前、训练前都过这一关（本入口采集在前，
        # 所以挡住这里就挡住了两者；rounds 内部另有逐轮检查）。
        if stop["stop"]:
            print("[autoloop] 停止条件已满足：本轮不采集、不训练")
            if not cfg.dry_run:
                log.append(_ev(args.run_id, "paused",
                               "stopped_before_start",
                               note="; ".join(stop["reasons"])[:200]))
            # 停止后开新一轮：下次启动重新计墙钟（否则会永远停在这里）
            clock.reset()
            return 8
        if cfg.dry_run:
            print("[autoloop] dry-run：未执行任何训练/采集命令")
            return 0
        # ---- 无人值守采集（授权后才有这一支；先采集再训练）-------------
        collect_rec = None
        if cfg.collect == "tech":
            collect_rec, rc_c = collect_once(args, cfg, log=log)
            if rc_c:
                print(f"[autoloop] 采集未通过（rc={rc_c}）：本轮不训练")
                return rc_c
        elif cfg.collect not in ("off", ""):
            print(f"[autoloop] collect={cfg.collect} 不是 off/tech：不猜，按 off 走")
        collect_proposals = ([collect_rec["proposal_path"]]
                             if collect_rec and collect_rec.get("proposal_path")
                             else [])
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
        r_args = rounds_args_from(cfg, args.run_id, device=_dev,
                                  collect_proposals=collect_proposals)
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
                reasons=blob["decision"]["reasons"],
                outcome=str(blob.get("round_outcome") or "")))
        stop2 = should_stop(cfg, hist2, gpu_minutes_today=total_today,
                            candidates_used=len(hist2))
        note = (f"rounds rc={rc}; 停因: " + "; ".join(stop2["reasons"])
                if stop2["stop"] else f"rounds rc={rc}; 未触发停止条件")
        # 收尾事件也要守状态机：rounds 里若已经写过 failed，从 failed 只能到
        # queued/paused，写 "training" 会被事件日志拒绝（实测踩到：整轮跑完，
        # 最后一步抛 ValueError，rc 变成 1）。
        _phase = "paused" if stop2["stop"] else "training"
        try:
            log.append(_ev(args.run_id, _phase,
                           "stopped" if stop2["stop"] else "idle",
                           note=note[:200]))
        except ValueError:
            log.append(_ev(args.run_id, "paused", "stopped",
                           note=note[:200]))
        print(f"[autoloop] 后台入口结束：{note}")
        return int(rc)
    finally:
        lease.release()
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


def machine_gpu_path() -> Path:
    """机器级 GPU 账本（当天所有 run 合计）。"""
    return Path(config.LOGS_DIR) / "experiments" / "gpu_ledger_machine.json"


def machine_gpu_minutes_today(*, today: str | None = None) -> float:
    """这台机器今天一共跑了多少分钟（缺文件按 0）。"""
    today = today or time.strftime("%Y-%m-%d")
    p = machine_gpu_path()
    if not p.exists():
        return 0.0
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return 0.0
    return float((blob.get(today) or {}).get("minutes") or 0.0)


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
    # 机器级账本：每日上限是"这台机器今天跑了多少"，不是"这个 run 跑了多少"。
    # 实测缺口：只按 run 记账时，换一个 run-id 当天已用时长就归零，上限形同虚设。
    m = machine_gpu_path()
    mblob = {}
    if m.exists():
        try:
            mblob = json.loads(m.read_text(encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            mblob = {}
    mslot = dict(mblob.get(today) or {})
    mslot["minutes"] = round(float(mslot.get("minutes") or 0.0)
                             + max(0.0, float(minutes)), 2)
    runs = dict(mslot.get("runs") or {})
    runs[run_id] = round(float(runs.get(run_id) or 0.0)
                         + max(0.0, float(minutes)), 2)
    mslot["runs"] = runs
    mblob[today] = mslot
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(mblob, indent=1, ensure_ascii=False),
                 encoding="utf-8")
    return float(slot["minutes"])


def rounds_args_from(cfg, run_id: str, *, device: str = "cuda",
                     thresholds: str | None = None,
                     timeout_s: int = 3600,
                     collect_proposals: list | None = None) -> argparse.Namespace:
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

    # 本轮采集出来的数据组作为**第一个**候选：它是本轮唯一的新信息；配置里
    # 手写的提议按原顺序跟在后面。合并后的文件写进 run 目录，重放能对上。
    props = cfg.proposals or None
    extra = [str(p) for p in (collect_proposals or []) if p and Path(p).exists()]
    if extra:
        blobs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in extra]
        if props and Path(props).exists():
            blobs.append(json.loads(Path(props).read_text(encoding="utf-8")))
        merged = merge_proposals(*blobs)
        mp = exp_dir(run_id) / "proposals_merged.json"
        mp.write_text(json.dumps(merged, indent=1, ensure_ascii=False),
                      encoding="utf-8")
        props = str(mp)
        print(f"[autoloop] 提议合并：{len(merged['proposals'])} 条"
              f"（采集在前）-> {mp}")

    return argparse.Namespace(
        run_id=run_id, rounds=int(cfg.rounds),
        runs=list(cfg.runs), baseline_runs=list(cfg.baseline_runs) or None,
        eval_runs=list(cfg.eval_runs), proposals=props,
        seeds=list(cfg.seeds), epochs=int(cfg.epochs), batch=int(cfg.batch),
        lr=float(cfg.lr), split="tail", device=device,
        thresholds=thresholds, config=str(cfg_path),
        hard_recall=0.70, hard_precision=0.40, hard_p95=45.0,
        timeout_s=int(timeout_s), trainer_script=str(cfg.trainer_script),
        plan_only=False, allow_road_only=bool(cfg.allow_road_only),
        paint_source=list(cfg.paint_sources) or None,
        research_arm=bool(cfg.research_arm),
        equal_steps=bool(cfg.equal_steps), resume=True)


#: 任务主指标（与 gates.PRIMARY_ORDER 同定义）：晋级判定必须看到它们。
TASK_METRICS = ("candidate_identity_rate", "line_recall", "line_precision",
                "offroad_false_line_px", "inference_ms_p95",
                # G09：候选匹配的新口径（分母不同，必须分开报）
                "candidate_reference_coverage",
                "left_right_role_agreement")
def _match_confirmed_seed(rec_sha, sha_by_seed: dict) -> tuple:
    """确认记录里的权重哈希落在哪个 seed 的候选 checkpoint 上。

    返回 ``(seed|None, issues)``。方案 §10.3 要求最终候选绑定**实际 checkpoint**，
    所以"确认的权重不在本轮候选里"必须当输入不一致处理，而不是放行。
    """
    want = str(rec_sha or "")
    if not want:
        return None, ["confirmation record has no model_sha16: R2 must name the "
                      "exact weights that were confirmed"]
    matched = sorted(sd for sd, sha in sha_by_seed.items() if sha == want)
    if not matched:
        return None, [
            f"confirmation weights {want!r} are not among this round's candidate "
            "checkpoints: R2 must name the exact deployed weights"]
    return matched[0], []


def _protocol_snapshot(t: Thresholds) -> dict:
    """判定文件里的协议快照 = protocol_blob(本轮阈值) + 哈希。

    阈值取**本轮实际解析出来的那份**（而不是"此刻磁盘上最新的文件"）：
    两者不一致时，判定必须按它自己用过的口径留档。
    """
    from dataclasses import asdict
    thr = {k: v for k, v in asdict(t).items()}
    blob = protocol_blob(thresholds=thr)
    return {**blob, "hash": snapshot_hash(blob)}


#: 越小越好的任务指标（路外假线计数、推理延迟）
LOWER_IS_BETTER = ("offroad_false_line_px", "inference_ms_p95")


def _seed_hard(metrics: dict, ident: dict | None, p95: float | None) -> dict:
    """一个 seed（或一个场景）自己的硬门输入（方案 §10.2：均值不能替它过门）。

    键与 ``gates.HARD_CHECKS`` 同名——两边漂移就会出现"判定要的指标没接线"
    （G05 的原始缺陷），所以有测试卡住这个对齐。测不到一律 ``None``
    （= UNKNOWN，不当通过）。
    """
    ident = ident or {}
    return {
        "candidate_identity_rate": ident.get("candidate_identity_rate"),
        # v4 新增的两道候选门：必须逐 seed 可测（否则记缺测阻止晋级）
        "candidate_reference_coverage": ident.get("candidate_reference_coverage"),
        "left_right_role_agreement": ident.get("left_right_role_agreement"),
        "line_recall": (metrics or {}).get("line_recall"),
        "line_precision": (metrics or {}).get("line_precision"),
        "offroad_false_ratio": (metrics or {}).get("offroad_false_frac_of_pred"),
        "inference_ms_p95": p95,
    }


def task_metric_value(name: str, metrics: dict, identity: float | None):
    """从一次 ``_pixel_eval`` 的结果里取任务指标；缺测返回 ``None``（UNKNOWN）。

    实测口径（方案 G05）：原来只把 IoU 送进判定器，而判定要求**任务主指标**
    有可信改善 —— 于是任何候选都不可能晋级（"IoU 改善"在判定里是辅助指标）。
    """
    if name in IDENTITY_FIELDS:
        # G09：身份类指标全部来自 identity 口径（分母各自不同）
        return None if not isinstance(identity, dict) else identity.get(name)
    if name == "offroad_false_line_px":
        return metrics.get("offroad_false_line_px")
    return metrics.get(name)


def task_pairings(champ_task: dict, cand_task: dict, seeds, *, extra: dict | None = None) -> dict:
    """按 seed 组装任务主指标的成对比较（缺测的 seed 直接不进配对，不补 0）。

    ``champ_task`` / ``cand_task``：``{seed: {metric: value}}``。
    ``extra``：像素代理指标（IoU）等附加成对结果。
    """
    out = dict(extra or {})
    for name in TASK_METRICS:
        va, vb = [], []
        for s in seeds:
            a = (champ_task.get(str(s)) or {}).get(name)
            b = (cand_task.get(str(s)) or {}).get(name)
            if a is None or b is None:
                continue
            va.append(float(a))
            vb.append(float(b))
        if not va:
            # 一个 seed 都测不到：记 UNKNOWN（不进分母、不写成 0）
            out[name] = {"metric": name, "n": 0, "champion": [], "candidate": [],
                         "verdict": "needs_evidence",
                         "why": "no paired measurement for this metric"}
            continue
        out[name] = paired_compare(name, va, vb,
                                   lower_is_better=name in LOWER_IS_BETTER)
    return out


def _pixel_eval(model_path: Path, eval_runs: list, device: str) -> dict:
    """开发集上的像素层指标（复用评估矩阵的实现，不另写一套口径）。

    按 ``map/source_id`` 分组评估（方案 §10.2/A7）：返回**同形状**的总体指标，
    另有 ``per_group`` 明细——每个场景自己过不过门，不能只看池化均值。
    同组的多个目录（例如同一次采集的不同视角）合并成一个场景。
    """
    import m5_seg_eval_matrix as em
    by: dict = {}
    for r in eval_runs:
        by.setdefault(dir_group(r), []).extend(em.load_frames([Path(r)]))
    return em.evaluate_model_per_group(model_path, by, device=device)


#: 候选匹配的新口径（方案 G09）：冻结口径之外，还要报
#: 可测候选覆盖率与左右角色一致率（分母不同不能混比）。
IDENTITY_FIELDS = ("candidate_identity_rate",
                   "candidate_identity_rate_with_reference",
                   "candidate_reference_coverage",
                   "left_right_role_agreement",
                   "candidate_paint_recall",
                   "n_candidates", "n_candidates_with_reference")
def _canon_dir(p) -> str:
    """目录的规范化键：解析成绝对路径 + posix 形式 + 小写（Windows 不区分大小写）。

    清单侧与调用侧的路径写法不同（绝对 vs 相对、`\\` vs `/`），键必须统一，
    否则唯一清单会"查不到"而被当成空清单（实测：身份率静默 UNKNOWN）。
    """
    try:
        return Path(p).resolve().as_posix().lower()
    except OSError:
        return Path(p).as_posix().lower()


def dev_frames_by_dir(records) -> dict:
    """评价集里**被接受**的帧按目录分组（键 = ``_canon_dir(目录)``）。

    身份/候选评价消费这份唯一清单；键必须与 ``identity_metrics`` 的查法一致
    （实测两次踩到：把帧文件路径或未规范化的相对路径当键 -> 查不到 -> 空清单
    -> 探针拒测 -> 身份率静默 UNKNOWN）。
    """
    out: dict = {}
    for r in records:
        if getattr(r, "reject_reason", ""):
            continue
        out.setdefault(_canon_dir(Path(r.path).parent), []).append(r.path)
    return out


def identity_metrics(model_path: Path, eval_runs: list, *,
                     probe_fn=None, frames_by_dir: dict | None = None) -> dict:
    """候选匹配的全口径：冻结匹配率 + 覆盖率 + 角色一致率。

    方案 G09 点名：原来只读原始 ``match_rate``，而它的分母里混着
    "该侧没有参考"的候选（实测约 38%）——那些候选应记 UNKNOWN，
    不能当"未确认/假线"。本函数把全部口径一起取回来（多段路取均值），
    缺测的字段一律 None。

    ``frames_by_dir``：审计后的**唯一**逐帧清单（``{目录: [接受的帧路径]}``，
    来自 ``_rounds_audit`` 的 ``dev_frames_by_dir``）。给了就按它调探针，
    而不是让探针各自 glob —— 两个内容相同的目录会让计数翻倍（独立复核实测：
    字节相同的两个目录把 C 从 10 变成 20，与标定的 10 不一致，方案 §S2 验收
    要求标定/循环/评价逐项相同）。目录不在清单里 = 该目录没有可接受的帧，
    按空清单处理（贡献 0，不退回 glob）。不传则维持旧行为（未去重口径）。
    """
    # 探针在 scripts/ 下：测试或别的入口直接调本函数时，sys.path 里可能没有它
    import sys as _sys
    _sd = str(Path(__file__).resolve().parent)
    if _sd not in _sys.path:
        _sys.path.insert(0, _sd)
    import m5_marking_identity_probe as ip
    probe_fn = probe_fn or ip.probe
    # 新口径：**累加整数计数**，最后算一次比率（方案 v2 §3.3）。
    # 旧实现把各目录的比率取平均、并用"全部候选"当分母，确定性输入会算出
    # 覆盖率 0.40（应 0.80）——所以这里只加 counts。
    from beamng_autopilot.experiments import candidate_metrics as _cm
    counts_acc = _cm.empty()
    counts_by_group: dict = {}
    acc: dict = {k: [] for k in IDENTITY_FIELDS}
    gacc: dict = {}          # 逐场景（map/source_id 组）明细
    run_errors: list = []    # 被跳过的评价 run（T11：缺测必须可见，不静默丢）
    for r in eval_runs:
        run = Path(r)
        meta = run / "meta.json"
        if not meta.exists() and (run.parent / "meta.json").exists():
            meta = run.parent / "meta.json"
        if not meta.exists():
            # 旧实现直接 continue：整个 run 消失、计数为 0，看起来像"没有候选"。
            # 缺 meta 是**缺测**，必须能定位（方案 v2 §S2 验收）。
            run_errors.append({"run": str(run), "why": "no meta.json"})
            continue
        try:
            # 只在给了唯一清单时传 frames：注入的假探针（测试）可能没有该参数，
            # 未去重口径的行为因此与旧版逐字一致
            _kw = {}
            if frames_by_dir is not None:
                _frames = frames_by_dir.get(_canon_dir(run))
                if _frames is None:
                    # 兜底：按后缀匹配（键可能是绝对路径、调用方给相对路径）
                    _suf = Path(run).as_posix().lower()
                    for _k, _v in frames_by_dir.items():
                        if Path(_k).as_posix().lower().endswith(_suf):
                            _frames = _v
                            break
                if _frames is None:
                    # 查不到 = 缺测，必须可见：空清单会让探针拒测，静默变 UNKNOWN
                    run_errors.append({
                        "run": str(run),
                        "why": ("no accepted frames in the audited inventory "
                                "for this eval run (frames_by_dir key "
                                "mismatch, or every frame was rejected)")})
                _kw["frames"] = list(_frames or [])
            res = probe_fn(run, json.loads(meta.read_text(encoding="utf-8")),
                           view=run.name, model_path=str(model_path), **_kw)
        except Exception as exc:                      # noqa: BLE001
            # 探针异常同样不许静默丢 run：结构化记账，调用方（判定文件/看板）
            # 能看出"这个 run 没测到"，而不是把它当成 0 候选。
            run_errors.append({"run": str(run),
                               "why": f"{type(exc).__name__}: {exc}"})
            continue
        # 探针把 match_rate_with_reference 的字段**内联在 summary 里**
        # （`**match_rate_with_reference(rows)`），所以这里直接读 summary。
        summary = res.get("summary") or {}
        # 整数计数（新口径的唯一来源）
        _c = summary.get("counts") or {}
        if _c:
            _cm.accumulate(counts_acc, _c)
            _cm.accumulate(counts_by_group.setdefault(dir_group(r), _cm.empty()),
                           _c)
        n_cand = summary.get("n_candidates")
        n_ref = summary.get("n_candidates_with_reference")
        vals = {
            "candidate_identity_rate": summary.get("match_rate"),
            "candidate_identity_rate_with_reference": summary.get(
                "match_rate_with_reference"),
            "candidate_reference_coverage": (
                None if not n_cand else round(float(n_ref or 0) / n_cand, 4)),
            "left_right_role_agreement": summary.get("role_agreement_rate"),
            "candidate_paint_recall": summary.get("candidate_paint_recall"),
            "n_candidates": n_cand,
            "n_candidates_with_reference": n_ref,
        }
        for k, v in vals.items():
            if v is not None:
                acc[k].append(float(v))
                gacc.setdefault(dir_group(r), {}).setdefault(k, []).append(
                    float(v))
    out = {k: (round(sum(v) / len(v), 4) if v else None)
           for k, v in acc.items()}
    # 旧原始匹配率**显式改名**（S2：旧指标不得继续被硬门消费）
    if out.get("candidate_identity_rate") is not None:
        out["candidate_identity_rate_legacy_match_rate"] = out[
            "candidate_identity_rate"]
    # 新口径：计数汇总 -> 比率；同时给出整数（n_candidates=C、
    # n_candidates_with_reference=R、candidates_matched=M、role_compared=L）
    _r = _cm.ratios(counts_acc)
    out.update({
        "candidate_reference_coverage": _r["candidate_reference_coverage"],
        "candidate_identity_rate": _r["candidate_identity_rate"],
        "left_right_role_agreement": _r["left_right_role_agreement"],
        "counts": dict(counts_acc),
        "counts_by_group": {g: dict(v) for g, v in counts_by_group.items()},
        "ratios_by_group": {g: _cm.ratios(v) for g, v in counts_by_group.items()},
        "n_candidates": float(counts_acc.get("C", 0)),
        "n_candidates_with_reference": float(counts_acc.get("R", 0)),
        "candidates_matched": float(counts_acc.get("M", 0)),
        "role_compared": float(counts_acc.get("L", 0)),
    })
    # 逐场景明细（方案 §10.2：分场景候选口径只上报，不进硬门）。没有它，
    # 调用方读 `per_group` 会**静默拿到空字典**——看起来像"没有候选"，
    # 实际是没接线（实测踩到）。
    out["per_group"] = {
        g: {k: round(sum(v) / len(v), 4) for k, v in d.items()}
        for g, d in gacc.items()}
    # 缺测的 run 逐条可见（T11）：空列表 = 每个 run 都测到了
    out["eval_run_errors"] = run_errors
    out["n_eval_run_errors"] = len(run_errors)
    return out


def _identity_rate(model_path: Path, eval_runs: list) -> float | None:
    """候选身份确认率：identity probe 的匹配率（多段路取均值）；测不到返回 None。"""
    # 探针在 scripts/ 下：测试或别的入口直接调本函数时，sys.path 里可能没有它
    import sys as _sys
    _sd = str(Path(__file__).resolve().parent)
    if _sd not in _sys.path:
        _sys.path.insert(0, _sd)
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
                        # FP/FN 方向旋钮（T15）：训练器一直有这个参数，但没进
                        # 白名单 -> 提议器发不出来（S6 E2 需要它做方向实验）
                        "line_tversky_alpha",
                        "run_weights",
                        # 容量族：数据与步数不动，只改模型宽度（参数约按平方增长）。
                        # 加这一族是因为前两条杠杆都测到了边界：单段新数据 <1 点且
                        # 跨 0，训练预算 120→240 步 +0.0092、240→480 步没有可判定增益。
                        "width")


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


def factor_epochs(base_epochs: int, extra: list) -> int:
    """记录里该写的训练轮数：因子用 ``--epochs`` 覆盖时以因子的为准。

    因子族里有 ``epochs``（TRAINER_FLAG_FACTORS），它的旗标附在训练命令末尾、
    覆盖命令行的 ``--epochs``。判定文件里若还写基线轮的数值，就成了"记录与实际
    不一致"——实测步数虽然在 ``steps_by_arm`` 里，但那种不一致本身就是本项目
    要杜绝的东西。命令行最后出现的那个 ``--epochs`` 胜出（与 argparse 一致）。
    """
    out = int(base_epochs)
    toks = list(extra or ())
    for i, tok in enumerate(toks):
        if tok == "--epochs" and i + 1 < len(toks):
            out = int(toks[i + 1])
    return out


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
    # 逐 step 指标：每臂每 seed 一个 run id（监控器/看板按"最新有指标的 run"服务）。
    # 不传的话训练不写 metrics.jsonl，看板就没有实时曲线（实测踩到）。
    metrics_run = f"{args.run_id}-{Path(out).parent.name}-s{seed}"
    # 研究臂（弱监督）：指定漆线真值来源，训练器会按 usable
    # 判据对 line 通道做监督（默认 engine_annotation 仍然屏蔽）。
    # 这是**两臂共享的配方**，不是因子差异。
    _srcs = sorted(set(paint_sources_from(args).values()))
    paint_flags = (["--paint-source", _srcs[0]] if len(_srcs) == 1 else [])
    return [sys.executable, str(script),
            "--runs", *[str(r) for r in runs],
            "--split", args.split, "--val-frac", "0.2",
            "--epochs", str(args.epochs), "--batch", str(args.batch),
            "--lr", str(args.lr), "--seed", str(seed),
            "--device", args.device, "--save-every-epoch",
            "--metrics-run", metrics_run,
            "--out", str(out), *road_only, *paint_flags, *extra]


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


def min_positive(*vals):
    """重复测量取较小值（None 忽略）；全为 None 时返回 None。

    本机实测时延有**间歇性**负载污染（同一 checkpoint：p50 21.7 vs 9.95 ms、
    p95 63.8 vs 11.4 ms），单次测量会把"机器在忙"记成模型很慢。取重复测量的
    较小值是最少受污染的一方，两次都记进判定供人复核。
    """
    got = [float(v) for v in vals if v is not None]
    return None if not got else min(got)


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


#: 平台期判据：最后 k 轮验证指标的最大变化不超过它就认为"已到平台期"。
#: 实测依据：3 epoch 时同一配方的读数摆 ±0.26，12 epoch 时收敛到 ±0.03 并单调上升。
PLATEAU_TOL = 0.02


def timing_precondition(game_state: bool | None) -> str | None:
    """独占计时的前提：``None`` = 可以计时；否则返回不可引用的原因（方案 G06）。

    * ``True``：有游戏在跑（GPU 被占用）→ 延迟不可引用；
    * ``None``：进程探测失败 → **不确定**，同样不可引用（不猜）。
    """
    if game_state is None:
        return ("game state UNKNOWN (process probe failed): exclusive timing "
                "not established")
    if game_state:
        return ("a game process is running: exclusive timing not established")
    return None


def _both_at_plateau(cand: dict, base: dict):
    """两臂都到平台期才返回 True；任一臂缺测返回 None（UNKNOWN，不当通过）。"""
    if not cand and not base:
        return None
    rows = [v for v in list(cand.values()) + list(base.values()) if v]
    if not rows or any(v.get("at_plateau") is None for v in rows):
        return None
    return all(bool(v.get("at_plateau")) for v in rows)


def plateau_from_hist(path: Path, *, k: int = 3,
                      tol: float = PLATEAU_TOL) -> dict:
    """读一份 ``train_hist.json`` 判平台期；读不到就是 **UNKNOWN**，不猜。"""
    p = Path(path)
    if not p.exists():
        return {"at_plateau": None, "missing": f"没有 {p.name}"}
    try:
        return plateau_check(json.loads(p.read_text(encoding="utf-8")),
                             k=k, tol=tol)
    except Exception as exc:                          # noqa: BLE001
        return {"at_plateau": None, "missing": f"hist 读取失败：{exc}"}

def plateau_check(hist: dict, *, k: int = 3, tol: float = PLATEAU_TOL) -> dict:
    """用训练历史判断"是否到平台期"（缺列时如实报 UNKNOWN，不猜）。

    ``hist`` 是训练器落的 `train_hist.json`。看最后 k 个 epoch 的验证指标
    （`val_miou` 优先，其次 `val_line_iou`）：变化幅度 > tol 就是**还没稳**，
    此时的判定只能算"暂行"，不能当结论引用。
    """
    for key in ("val_miou", "val_line_iou"):
        vals = [v for v in (hist.get(key) or []) if v is not None]
        if len(vals) >= 2:
            tail = vals[-int(k):]
            spread = max(tail) - min(tail)
            return {"key": key, "n_epochs": len(vals), "tail": tail,
                    "spread": round(spread, 5),
                    "at_plateau": bool(spread <= float(tol)),
                    "tol": float(tol)}
    return {"key": "", "n_epochs": 0, "tail": [], "spread": None,
            "at_plateau": None, "tol": float(tol),
            "missing": "训练历史里没有可用的验证指标列"}


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
    try:
        log.append(_ev(run_id, phase, status, candidate=cand_id,
                       note=note[:200]))
        return
    except ValueError:
        pass
    # 阶段不合法时**不能**让整轮以 ValueError 结束（实测踩到：auditing ->
    # needs_evidence 被拒，"因子没生效"这种拒训反而把入口炸掉）。退到当前阶段
    # 允许的"需要人看"，再退"暂停"；状态名（status）保持原样，原因照写。
    for _fb in ("needs_review", "paused"):
        if _fb == phase:
            continue
        try:
            log.append(_ev(run_id, _fb, status, candidate=cand_id,
                           note=note[:200]))
            return
        except ValueError:
            continue
    raise


def _git_dirty_paths(limit: int = 20) -> list:
    """未提交路径（判定文件要能说明"当时工作区不干净"）。

    读不到（不是 git 仓库/git 不在）返回 ``[]``——空列表在这里只表示"没读到"，
    看板另有 git_commit 为空串可区分；不猜、不伪造干净状态。
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=30)
    except Exception:                          # noqa: BLE001
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()][:limit]


def sum_negative_summaries(per_seed: dict) -> dict:
    """把逐 seed 的 ``negative_line`` 汇总成一份：**整数相加后再算比率**。

    与候选计数同一契约（方案 v2 §3.3）：先把各 seed 的帧数/像素数相加，再算
    假线帧率与像素占比；缺计数的 seed 不参与（不按 0 相加）。
    """
    rows = [v for v in (per_seed or {}).values() if v]
    if not rows:
        return {}
    keys = ("frames", "eligible_frames", "clean_frames",
            "false_positive_frames", "false_positive_px", "eligible_px",
            "positive_frames", "unknown_frames", "empty_frames",
            "unverified_frames", "unverified_pred_line_px")
    out = {k: sum(int((r or {}).get(k) or 0) for r in rows) for k in keys}
    out["n_seeds"] = len(rows)
    out["status"] = ("measured" if out["eligible_frames"]
                     else "no_eligible_frames")
    out["false_positive_frame_rate"] = (
        None if not out["eligible_frames"]
        else out["false_positive_frames"] / out["eligible_frames"])
    out["false_positive_pixel_fraction"] = (
        None if not out["eligible_px"]
        else out["false_positive_px"] / out["eligible_px"])
    out["excluded_frames"] = (out["unverified_frames"] + out["unknown_frames"]
                              + out["empty_frames"])
    return out


def paint_sources_from(args) -> dict:
    """``--paint-source RUN=SOURCE`` -> ``{run: source}``。

    为什么要逐 run 指定：引擎的漆线类**存在但不完整**
    （2026-09-25 实测：覆盖 RGB 漆线候选的 ~0.61），所以它不能当门槛真值，
    但可以当弱监督让模型先学会"产出标线"。字典键支持目录名
    或完整路径（与 manifest 的查找顺序一致）。
    """
    out: dict = {}
    for item in (getattr(args, "paint_source", None) or []):
        if "=" not in item:
            continue
        run, src = item.split("=", 1)
        run, src = run.strip(), src.strip()
        out[run] = src
        # manifest 按 ``str(rd)`` 查（rd 由 runs/eval_runs 的字符串构造，Windows
        # 下是反斜杠），所以把规范化后的路径也登记一份——实测踩到：只用正斜杠的键
        # 会静默回落 engine_annotation（训练侧仍按 --paint-source 走，但审计报告里
        # 的 paint_ok 与原因就是错的）。不登记 basename：不同采集的 front_main 同名。
        out[str(Path(run))] = src
    return out


def _block_research_promotion(dec: dict, research: bool) -> dict:
    """研究臂不允许晋级：弱监督真值不完整时，

    「精度达标」可能只是「只预测了被标注的那部分」。把
    ``shadow_candidate`` / ``approved_for_review`` 降级为 ``needs_evidence``
    并写明原因，不让它混进晋级流程；非研究臂原样返回。
    """
    if not research:
        return dec
    if dec.get("decision") not in ("shadow_candidate", "approved_for_review"):
        return dec
    reason = ("research arm: paint truth is weak/pseudo (engine "
              "annotation is incomplete) - promotion requires "
              "human-revised truth")
    return {**dec, "decision": "needs_evidence",
            "reasons": list(dec.get("reasons") or []) + [reason]}


def resolve_paint_sources(args) -> dict:
    """把 ``--paint-source RUN=SOURCE`` 解析成**以凭证为准**的资格结论。

    方案 §6.1 / G03：命令行字符串不能决定资格。对每个 run：

    * 读该目录自己的凭证（``annotation.json`` / ``meta.json``）；
    * ``effective_source(declared, credential)`` —— 凭证优先，命令行不能抬高质量；
    * rank 不能晋级的（agent / pseudo / 未知）→ 整轮记 ``research_only``。

    返回 ``{runs, research_only, reasons, notes, missing_credentials}``；
    调用方把整份结论写进判定文件，**replay 才能重放同一结论**。
    """
    out = {"runs": {}, "research_only": False, "reasons": [], "notes": [],
           "missing_credentials": []}
    for item in (getattr(args, "paint_source", None) or []):
        if "=" not in item:
            continue
        run, declared = item.split("=", 1)
        run, declared = run.strip(), declared.strip()
        cred = read_dir_credentials(run)
        cred_src = None if cred is None else str(cred.get("label_source") or "")
        eff, notes = effective_source(declared, cred_src)
        rank = PAINT_SOURCE_RANK.get(eff, "absent")
        ok, why = sources_can_promote([eff])
        if cred is None:
            out["missing_credentials"].append(run)
        if not ok:
            out["research_only"] = True
            out["reasons"].append(
                f"{run}: effective source {eff!r} (rank {rank}) cannot be used "
                f"as a promotion reference")
        for n in notes:
            out["notes"].append(f"{run}: {n}")
        out["runs"][run] = {"declared": declared, "credential": cred_src,
                            "credential_path": (cred or {}).get("path"),
                            "credential_frames": (cred or {}).get("frames"),
                            "effective": eff, "rank": rank,
                            "eligibility": eligibility(rank),
                            "can_promote": bool(ok), "notes": notes}
    # 评价侧的资格**必须读 eval 目录自己的凭证**（方案 §6.1 / A1：agent 评价
    # 标签**漏传** research 标志仍不能晋级）。原来只遍历 `--paint-source`：
    # 不带这个参数时，agent 起草的评价真值会被当成可晋级参考——实测踩到
    # （E2 那轮 research_only=False，而 wide/plain 的评价标签是 agent 档）。
    for run in list(getattr(args, "eval_runs", None) or []):
        key = str(run)
        cred = read_dir_credentials(run)
        cred_src = None if cred is None else str(cred.get("label_source") or "")
        eff, notes = effective_source("", cred_src)
        rank = PAINT_SOURCE_RANK.get(eff, "absent")
        ok, why = sources_can_promote([eff])
        if cred is None:
            out["missing_credentials"].append(key)
        if not ok:
            out["research_only"] = True
            out["reasons"].append(
                f"{key}: evaluation reference source {eff!r} (rank {rank}) "
                f"cannot be used as a promotion reference")
        for n in notes:
            out["notes"].append(f"{key}: {n}")
        out["runs"].setdefault(key, {
            "declared": "", "credential": cred_src,
            "credential_path": (cred or {}).get("path"),
            "credential_frames": (cred or {}).get("frames"),
            "effective": eff, "rank": rank,
            "eligibility": eligibility(rank),
            "can_promote": bool(ok), "notes": notes,
            "role": "evaluation_reference"})
    return out


def _rounds_audit(args, train_runs, log) -> tuple:
    """训练前的数据准入门（方案步骤 5 的"审计"那一步）。

    返回 ``(report, rc)``；``rc != 0`` 时调用方必须停下，不训练。检查项：
    训练侧逐通道可用真值、被拒帧（身份/格式）、训练组与开发组的**整组隔离**、
    以及"标线真值不可用"（除非显式 ``--allow-road-only``）。
    """
    # 漆线真值来源：默认引擎标注（不可当门槛真值）；研究臂可以显式
    # 指定 `engine_annotation_partial`（弱监督）——此时 paint 仍记 valid=False，
    # 但训练不再屏蔽 line 通道，判定也带 research_only 标记。
    # 最终集不得被搜索/训练读取（方案 §7/§10.3）：带封存文件的目录一律拒训。
    _sealed = []
    for _r in list(train_runs) + list(args.eval_runs):
        _d = Path(_r)
        for _cand in (_d, _d.parent):
            if (_cand / "final_set_seal.json").is_file():
                _sealed.append(str(_cand))
    if _sealed:
        _log_give_up(log, args.run_id, "audit", "final_set_read",
                     f"搜索/训练不得读最终集（封存目录 {sorted(set(_sealed))[:2]}）",
                     phase="needs_review")
        print(f"[rounds] 审计不通过：输入里含已封存的最终集 {sorted(set(_sealed))[:2]}——最终集只允许确认程序读，不训练")
        return {"final_set_read": sorted(set(_sealed))}, 3
    _ps = paint_sources_from(args)
    mf_tr = DatasetManifest.build([Path(r) for r in train_runs], root=ROOT,
                                  paint_sources=_ps)
    mf_dev = DatasetManifest.build([Path(r) for r in args.eval_runs], root=ROOT,
                                   paint_sources=_ps)
    cov = mf_tr.coverage().get("train", {})
    trainable = int(cov.get("trainable_frames") or 0)
    paint_ok = int(cov.get("paint_valid_frames") or 0)
    train_groups = sorted({r.group for r in mf_tr.records if not r.reject_reason})
    dev_groups = sorted({r.group for r in mf_dev.records if not r.reject_reason})
    overlap = sorted(set(train_groups) & set(dev_groups))
    rejected = [{"path": r.path, "reason": r.reject_reason}
                for r in mf_tr.records if r.reject_reason]
    # 空间隔离（方案 W2 §7.3）：不同 source_id 但**同一地点**的采集也会泄漏，
    # 整组隔离抓不到（实测：diverse_straightstreet 与 ident_probe_straight 相距 0 m）。
    spatial = spatial_conflicts(mf_tr.records, mf_dev.records,
                               buffer_m=SPATIAL_BUFFER_M)
    exp_leak = group_exposure_leak(list(mf_tr.records)
                                   + list(mf_dev.records))
    # 评价集里**被接受**的帧（按目录分组，规范化键）：身份/候选评价必须消费
    # 这份唯一清单，而不是让探针各自 glob —— 目录复制会带回同一张图的多份
    # 拷贝（实测 159 次输入里只有 136 张唯一图）。键的构造与查法都在
    # `dev_frames_by_dir`/`_canon_dir` 里，避免"两边各写一套"（实测踩过两次）。
    _dev_frames = dev_frames_by_dir(mf_dev.records)
    # E1 数据纪律（方案 §S6/E1 + §3.5/T10）：全零标线且非 verified 的目录只能作
    # **弱**负例，且必须在报告里可见——"游戏不提供 line 类"导致的全零是**缺失**，
    # 不是"确认无线"；可晋级运行里出现这种目录直接拒训。
    _neg_by_dir: dict = {}
    for _r in mf_tr.records:
        if _r.reject_reason:
            continue
        _k = _canon_dir(Path(_r.path).parent)
        _e = _neg_by_dir.setdefault(_k, {
            "dir": str(Path(_r.path).parent), "n_frames": 0,
            "n_line_frames": 0, "rank": ""})
        _e["n_frames"] += 1
        _e["n_line_frames"] += int(int(_r.line_px or 0) > 0)
        _e["rank"] = ((_r.quality or {}).get("paint") or {}).get("rank") or _e["rank"]
    _neg_gate = negative_training_eligibility(
        list(_neg_by_dir.values()),
        # 只有"来源全部可晋级"的运行才算 promotion-eligible：其余（含默认引擎
        # 标签、road-only、显式研究臂）都按研究处理，弱负例允许但必须可见。
        research=bool(getattr(args, "research_arm", False)) or bool(
            resolve_paint_sources(args)["research_only"]))
    report = {"dataset_id": mf_tr.dataset_id,
              "dev_dataset_id": mf_dev.dataset_id,
              "train_groups": train_groups, "dev_groups": dev_groups,
              "group_overlap": overlap, "coverage": cov,
              # 数据入口四计数（看板"数据入口"面板）需要的原始量：生成=清单全部
              # 记录（含被拒），评价=开发集记录数，复核=开发集 paint 档位有效帧。
              "n_records_train": len(mf_tr.records),
              "n_records_dev": len(mf_dev.records),
              "coverage_dev": mf_dev.coverage().get("dev", {}),
              "dev_frames_by_dir": _dev_frames,
              "negative_training": _neg_gate,
              "spatial": spatial, "exposure_leak": exp_leak,
              "rejected": rejected, "notes": mf_tr.notes + mf_dev.notes}
    out = exp_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rounds_dataset.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[rounds] 审计：train groups={train_groups} dev groups={dev_groups} "
          f"trainable={trainable} paint_ok={paint_ok} rejected={len(rejected)}")
    if _neg_gate["weak"] or _neg_gate["confirmed"] or _neg_gate["rejected"]:
        print(f"[rounds] 负例目录资格：{_neg_gate['note']}")
        for _w in _neg_gate["weak"]:
            print(f"[rounds]   弱负例（研究臂，不得当已确认负例）："
                  f"{_w['dir']} rank={_w['rank']} 帧={_w['n_frames']}")
    if _neg_gate["rejected"]:
        _note = ("weak_negative_training: " + "; ".join(
            f"{r['dir']}（rank={r['rank']}）" for r in _neg_gate["rejected"]))
        _log_give_up(log, args.run_id, "audit", "weak_negative_training",
                     _note, phase="needs_review")
        print("[rounds] 审计不通过：可晋级运行里出现'全零标线且非 verified'的"
              "训练目录——它不构成已确认负例（T10），拒绝训练")
        return report, 3
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

    if spatial["violations"]:
        _why = spatial["violations"][0]
        _log_give_up(log, args.run_id, "audit", "spatial_leak",
                     f"训练/开发存在同地点或同曝光泄漏（{len(spatial['violations'])} 条，缓冲 {SPATIAL_BUFFER_M:.0f} m）：{_why}",
                     phase="needs_review")
        print(f"[rounds] 审计不通过：空间隔离违规 {len(spatial['violations'])} 条（首条={_why['why']}，最小距离 {spatial['min_distance_m']} m，"
              f"缓冲 {SPATIAL_BUFFER_M:.0f} m）——不训练")
        return report, 3
    if exp_leak:
        _log_give_up(log, args.run_id, "audit", "exposure_split",
                     f"同一曝光的多视角被拆到不同组：{exp_leak[:3]}",
                     phase="needs_review")
        print(f"[rounds] 审计不通过：同曝光多视角跨组 {len(exp_leak)} 条（必须同组）——不训练")
        return report, 3
    _line_sup = bool(_ps)
    if _line_sup:
        print(f"[rounds] 研究臂：训练开启 line 通道（来源 {_ps}，弱监督：可学不可判）；"
              f"判定记 research_only，不允许晋级")
    if not paint_ok and not getattr(args, "allow_road_only", False) \
            and not _line_sup:
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
    # 直接调 rounds 也是重型实验：同样要持机器级租约（方案 G02；
    # run -> rounds 是同进程，acquire 会返回 already_held，不会自锁）。
    lease = None
    if not args.plan_only:
        lease = machine_lease()
        got_lease = lease.acquire(force=bool(getattr(args, "force_lease", False)))
        if not got_lease["acquired"]:
            print(f"[rounds] 机器级租约：{got_lease['reason']}")
            log.append(_ev(args.run_id, "paused", "lease_blocked",
                           note=got_lease["reason"][:200]))
            return 4
        # 只有**本次调用新拿到**的租约才由本次释放；run 持有的那份由 run 释放
        created_here = not got_lease.get("already_held")
    else:
        created_here = False
    try:
        return _cmd_rounds_inner(args, cfg, log)
    finally:
        if lease is not None and created_here:
            lease.release()


def _cmd_rounds_inner(args, cfg, log) -> int:
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
    _report: dict = {}      # 审计的逐通道覆盖（判定归因要看本轮依赖的标签）
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
            _blob_done = json.loads(
                (exp_dir(args.run_id) / f"decision_{cand_id}.json")
                .read_text(encoding="utf-8"))
            hist_done = _blob_done["decision"]
            history.append(RoundRecord(round_index=rnd, candidate_id=cand_id,
                                       decision=hist_done["decision"],
                                       reasons=hist_done["reasons"],
                                       # 旧判定文件没有归因字段：留空串，
                                       # should_stop 回退旧口径（不猜）
                                       outcome=str(_blob_done.get(
                                           "round_outcome") or "")))
            continue
        # 因子是否**真会生效**（方案 v2 §S5 / T14）：road-only 把 line 类整通道
        # 屏蔽时，线损失键会照样变成训练器旗标（extra 非空），但没有任何监督
        # ——历史上有 5 轮就这么白跑了。这一步比"至少有一个键变成旗标"更严：
        # 先判监督模式，再让 factor_to_flags/arm_runs 判实现是否存在。
        _road_only = bool(getattr(args, "allow_road_only", False))
        _fact_act = factor_activity(factor, line_supervision=not _road_only)
        if not _fact_act["active"] and not args.plan_only:
            _note = "factor_not_applied: " + _fact_act["why"]
            _log_give_up(log, args.run_id, cand_id, "factor_not_applied",
                         _note, phase="needs_review")
            print(f"[rounds] 第 {rnd + 1} 轮拒绝训练：{_note}")
            print("  无效因子会让候选臂与基线臂在**实际监督**上完全相同，"
                  "跑出来的只是白跑一轮（无收益也不构成平台期证据）。")
            return 3
        if not _fact_act["active"]:
            print(f"[plan] 第 {rnd + 1} 轮因子被判 inactive：{_fact_act['why']}"
                  "（真实运行时这一轮会被拒绝）")
        extra, skipped = factor_to_flags(factor)
        cand_runs, data_note = arm_runs(args.runs, factor)
        # 因子未生效就拒绝训练：候选臂与基线臂输入相同，跑不出证据
        if not extra and not data_note["applied"]:
            note = ("factor_not_applied: " + "; ".join(
                data_note["not_applied"] or ["提议没有可用因子"]))
            # "因子没生效"是提议/配置问题（不是数据问题）：写 needs_review，
            # 让下一轮拿一条可修的提议来（从 auditing 到 needs_evidence 非法）
            _log_give_up(log, args.run_id, cand_id, "factor_not_applied",
                         note, phase="needs_review")
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
        champ_task: dict = {}     # 逐 seed 任务主指标（G05）：{seed: {metric: value}}
        base_n_train: int = 0
        equal_cap: int = 0
        plateau_base_by_seed: dict = {}
        cand_epochs = factor_epochs(args.epochs, extra)
        if rnd == 0:
            champ_by_seed: dict = {}
            base_neg: dict = {}      # 逐 seed 负例诊断（E1 主指标）
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
                # 负例诊断（T10 资格在评价矩阵里判）：两臂都要留档，否则
                # "加困难负例有没有减少假线"这个 E1 问题在判定文件里看不到
                base_neg[str(seed)] = m.get("negative_line") or {}
                # 任务主指标也逐 seed 收（方案 G05）：判定要求主指标有可信改善，
                # 只送 IoU 等于让任何候选都晋不了级。
                _idb = identity_metrics(out / "checkpoint_last.pt",
                                        args.eval_runs,
                                        frames_by_dir=_report.get(
                                            "dev_frames_by_dir"))
                champ_task[str(seed)] = {
                    name: task_metric_value(name, m, _idb)
                    for name in TASK_METRICS}
                # 基线臂的平台期也要判：只判候选臂时，"候选更好"可能只是
                # 候选训得更久（第 5 轮 seed 46 的末段大跳就是这么暴露的）
                plateau_base_by_seed[str(seed)] = plateau_from_hist(
                    out / "train_hist.json")
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
                         "champion.json 缺失，无法做非自我比较",
                         phase="needs_review")
            return 3
        # ---- 候选臂 ------------------------------------------------
        cand_by_seed: dict = {}
        cand_neg: dict = {}      # 逐 seed 负例诊断（E1 主指标）
        cand_task: dict = {}      # 逐 seed 任务主指标（G05）
        timing_repeats: list = []
        plateau_by_seed: dict = {}
        worst_by_seed: dict = {}
        trivial: dict = {}
        timing_notes: list = []
        offroads: list = []
        idents: list = []
        hard_measured: dict = {}
        hard_by_seed: dict = {}          # 每个 seed 自己的硬门度量
        scene_by_seed: dict = {}         # 每个 seed 的分场景度量
        scene_ident_by_seed: dict = {}   # 分场景**整数计数**（判定用，下限按 R）
        scene_ratios_by_seed: dict = {}  # 分场景比率（只上报，不参与判定）
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
            cand_neg[str(seed)] = metrics.get("negative_line") or {}
            _idc = identity_metrics(out / "checkpoint_last.pt",
                                    args.eval_runs,
                                    frames_by_dir=_report.get(
                                        "dev_frames_by_dir"))
            if _idc.get("eval_run_errors"):
                print(f"[rounds] 第 {rnd + 1} 轮 seed {seed}："
                      f"{_idc['n_eval_run_errors']} 个评价 run 没测到"
                      f"（判定记 UNKNOWN，不当 0 候选）："
                      f"{_idc['eval_run_errors'][:2]}")
            cand_task[str(seed)] = {
                name: task_metric_value(name, metrics, _idc)
                for name in TASK_METRICS}
            # 硬门输入一律取自**本轮实测**（方案点名：--hard-* 是命令行输入，
            # 不能把阈值本身当测量结果；缺测就是 None=UNKNOWN）
            offroads.append(metrics.get("offroad_false_frac_of_pred"))
            _hist_p = out / "train_hist.json"
            if _hist_p.exists():
                try:
                    plateau_by_seed[str(seed)] = plateau_check(json.loads(
                        _hist_p.read_text(encoding="utf-8")))
                except Exception:                     # noqa: BLE001
                    plateau_by_seed[str(seed)] = {"at_plateau": None,
                                                  "missing": "hist 读取失败"}
            if metrics.get("mask_compare"):
                # "点进具体帧"：判定文件带上该 seed 最差的几帧（路径+IoU+真值像素），
                # 人可以从一次失败判定直接看到是哪张图、差在哪
                worst_by_seed[str(seed)] = metrics["mask_compare"]["worst"]
            # 候选口径（覆盖率/左右角色）也进池化硬门：v4 起它们是硬门输入
            hard_measured.setdefault("candidate_reference_coverage",
                                     []).append(
                _idc.get("candidate_reference_coverage"))
            hard_measured.setdefault("left_right_role_agreement", []).append(
                _idc.get("left_right_role_agreement"))
            hard_measured.setdefault("line_recall", []).append(
                metrics.get("line_recall"))
            hard_measured.setdefault("line_precision", []).append(
                metrics.get("line_precision"))
            # 计时按方案做"受控复测"：同 seed 立即再测一次，取较小值进硬门，
            # 两次都留下来（本机时延有间歇性负载，单次不可信）
            _m2 = _pixel_eval(out / "checkpoint_last.pt", args.eval_runs,
                              args.device)
            _p95 = min_positive(metrics.get("inference_ms_p95"),
                                _m2.get("inference_ms_p95"))
            timing_repeats.append({
                "seed": int(seed),
                "p95": [metrics.get("inference_ms_p95"),
                        _m2.get("inference_ms_p95")],
                "p50": [metrics.get("inference_ms_p50"),
                        _m2.get("inference_ms_p50")],
                "used": _p95})
            if _p95 is not None:
                hard_measured.setdefault("inference_ms_p95", []).append(
                    float(_p95))
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
            # 与配对用的任务指标**同一次**探针结果：两处各探一次会让
            # 判定文件里出现两个身份数字（也白跑一遍推理）。
            _ident = _idc.get("candidate_identity_rate")
            if _ident is not None:
                idents.append(_ident)
            # 逐 seed 硬门（方案 §10.2：一个坏 seed 不能被跨 seed 均值藏掉）
            hard_by_seed[str(seed)] = _seed_hard(metrics, _idc, _p95)
            # 逐场景明细：每个 map/source_id 组自己那一份（像素层可当硬门，
            # 身份类只上报——单场景候选数可能只有几个，门槛尚未标定）
            _pg = metrics.get("per_group") or {}
            if _pg:
                scene_by_seed[str(seed)] = {
                    str(g): _seed_hard(gm, None, gm.get("inference_ms_p95"))
                    for g, gm in _pg.items()}
            # 逐场景**整数计数**（判定用）与**比率**（只上报）必须分开取：
            # 旧实现把 `_idc["per_group"]`（键是比率名）当计数读
            # （键 P_frames/C/R/M/L/A），于是每个场景都被读成 P_frames=0，
            # 真实有线场景被判 not_applicable，R<30 的下限在 rounds 路径
            # 永不触发（独立复核实测：P=3,C=100,R=62,M=9 的输入判定全 0）。
            if _idc.get("counts_by_group"):
                scene_ident_by_seed[str(seed)] = {
                    str(g): dict(cv or {})
                    for g, cv in _idc["counts_by_group"].items()}
            if _idc.get("ratios_by_group"):
                scene_ratios_by_seed[str(seed)] = {
                    str(g): dict(rv or {})
                    for g, rv in _idc["ratios_by_group"].items()}
        # 按 **seed** 配对：顺序/数量不一致说明对照不完整，拒绝而不是截断
        missing = [str(s) for s in args.seeds if str(s) not in champ_by_seed]
        if missing:
            note = (f"seed 不匹配：基线缺 {missing}（基线 seeds="
                    f"{sorted(champ_by_seed)}）—— 错配的比较不是证据")
            _log_give_up(log, args.run_id, cand_id, "seed_mismatch", note)
            print(f"[rounds] 第 {rnd + 1} 轮拒绝判定：{note}")
            return 3
        # 两臂平台期一起判（在写判定**之前**算好：实测踩到——把它放在判定字典
        # 之后会在写盘那一行 UnboundLocalError，训练全跑完却拿不到判定）
        all_at_plateau = _both_at_plateau(plateau_by_seed,
                                         plateau_base_by_seed)
        champ = [champ_by_seed[str(s)] for s in args.seeds]
        cand = [cand_by_seed[str(s)] for s in args.seeds]
        road_only = bool(getattr(args, "allow_road_only", False))
        pair_metric = "road_iou" if road_only else "line_iou"
        # 像素代理（IoU）照旧成对比较；任务主指标另按 seed 组装（G05）
        _proxy = {pair_metric: paired_compare(pair_metric, champ, cand)}
        paired = task_pairings(champ_task, cand_task, args.seeds,
                               extra=_proxy)
        if road_only:
            # line 通道被整通道屏蔽：标线/身份指标**没有真值**，一律 UNKNOWN
            # （记 None 而不是 0，避免"没测"被读成"很差"或被硬门当违反）
            hard = {"line_recall": None, "line_precision": None,
                    "candidate_identity_rate": None,
                    "candidate_reference_coverage": None,
                    "left_right_role_agreement": None,
                    "offroad_false_ratio": None,
                    "inference_ms_p95": _mean_or_none(
                        hard_measured.get("inference_ms_p95")),
                    "note": "road-only：标线通道整通道屏蔽，标线/身份指标未测"}
            idents = []
        else:
            hard = {"line_recall": _mean_or_none(
                        hard_measured.get("line_recall")),
                    "candidate_reference_coverage": _mean_or_none(
                        hard_measured.get("candidate_reference_coverage")),
                    "left_right_role_agreement": _mean_or_none(
                        hard_measured.get("left_right_role_agreement")),
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
        # 独占计时的前提（方案 G06）：没有游戏在跑。跑着、或探测不确定 ->
        # 本轮延迟记 UNKNOWN（不是"测出来很快"），理由进判定文件。
        _pre = timing_precondition(game_running_probe())
        if _pre:
            timing_notes.append(_pre)
        if timing_notes:
            print(f"[rounds] 警告：本轮计时不可引用（{len(timing_notes)} 条原因）：")
            for _tn in timing_notes:
                print(f"  - {_tn}")
            hard["inference_ms_p95"] = None
            hard["timing_suspect"] = timing_notes
            # 逐 seed / 分场景耗时要跟着作废：本轮计时不可引用时
            # 任何一个"很漂亮"的分场景 p95 也不许进硬门。
            for _hs in list(hard_by_seed.values()):
                _hs["inference_ms_p95"] = None
            for _ss in scene_by_seed.values():
                for _sv in _ss.values():
                    _sv["inference_ms_p95"] = None
        # 研究臂（弱监督标签）：判定照算但**不允许晋级**——真值不完整时
        # "line_precision 达标"可能只是"只预测了被标注的那部分"。把 shadow_candidate
        # 降级为 needs_evidence 并写明原因，不让它混进晋级流程。
        # 来源资格：**以凭证为准**（G03）。原来用"字符串以 _partial 结尾/等于
        # pseudo"判断，于是 --paint-source X=agent_revision 不带 --research-arm
        # 就能绕过晋级限制；在命令行写 human_revision 也能把 agent 数据升格。
        _src_res = resolve_paint_sources(args)
        _research = bool(getattr(args, "research_arm", False)) or \
            bool(_src_res["research_only"])
        for _n in _src_res["notes"]:
            print(f"[rounds] 来源凭证：{_n}")
        for _r in _src_res["reasons"]:
            print(f"[rounds] 来源资格：{_r}")
        # 分场景：跨 seed 汇总**每个场景自己的最差**——同一场景在不同 seed 的
        # 表现不能被平均掉（方案 §10.2：关键场景不允许被总体均值抵消）。
        # road-only：标线/身份整通道屏蔽，逐 seed 与分场景只查可测口径
        _pf = ("inference_ms_p95",) if road_only else None
        _sf = (("inference_ms_p95",) if road_only else SCENE_HARD_FIELDS)
        per_scene: dict = {}
        for _ss in scene_by_seed.values():
            for _g, _sv in _ss.items():
                for _k, _v in _sv.items():
                    cur = per_scene.setdefault(_g, {}).get(_k)
                    if _v is None:
                        continue
                    if cur is None or float(_v) < float(cur):
                        per_scene[_g][_k] = _v
        _scene = scene_report(per_scene, t, fields=_sf)
        # 逐场景**样本量下限（按 R）**与适用性（方案 v2 §S3.2/§3.4）：
        # 下限对象是身份率的实际分母，不能用总候选数冒充；无标线场景
        # not_applicable（既不通过也不算缺测）。
        _scene_counts: dict = {}
        for _ss in scene_ident_by_seed.values():
            for _g, _cv in (_ss or {}).items():
                _acc = _scene_counts.setdefault(_g, {})
                for _k in ("P_frames", "C", "R", "M", "L", "A",
                           "C_outside_P"):
                    _acc[_k] = int(_acc.get(_k, 0)) + int(
                        (_cv or {}).get(_k, 0) or 0)
        _sc = scene_count_violations(_scene_counts, t)
        if _sc["low_sample"]:
            print(f"[rounds] 逐场景样本不足 {len(_sc['low_sample'])} 条"
                  f"（下限按身份率分母 R={t.per_scene_min_candidates} 计）")
        for _m in _sc["missing"]:
            print(f"[rounds] 场景 UNKNOWN：{_m}")
        # 分场景候选口径（匹配率/覆盖/左右角色）：**只上报不进硬门**——
        # 单场景候选数可能只有几个，逐场景身份门槛尚未标定（方案 §10.2）。
        # 取的是**比率**字典（`ratios_by_group`），不是判定用的整数计数：
        # 两者键空间不同，混用会把比率读成 None（旧实现正是这样读错的）。
        _scene_keys = sorted({g for d in scene_ratios_by_seed.values()
                              for g in d})
        _cand_fields = ("candidate_identity_rate",
                        "candidate_reference_coverage",
                        "left_right_role_agreement", "n_candidates")
        scene_candidates: dict = {}
        for _g in _scene_keys:
            scene_candidates[_g] = {
                f: _mean_or_none([(d.get(_g) or {}).get(f)
                                  for d in scene_ratios_by_seed.values()])
                for f in _cand_fields}
        # 缺测与硬门：总体、逐 seed、分场景三路都要进来（方案 §10.2/A7）
        # 缺测与违反**分两路**（方案 §10.3 的判定顺序）：池化的硬门输入里
        # 没测到的项原来被 threshold_violations 写成"违反"，于是 road-only
        # 实验（标线整通道屏蔽、指标本就未测）被判成 rejected，读起来像"候选
        # 不合格"，实际是"没测"——代码注释本来就写着"记 None 避免被硬门当违反"，
        # 这里把注释兑现：缺测走 missing_metrics -> needs_evidence。
        _hs = hard_split(hard, t)
        _missing = missing_metrics_for(
            paired, coverage_gate_frozen=COVERAGE_GATE_FROZEN)
        _missing += [f"{n}: UNKNOWN (hard gate needs a measurement)"
                     for n in _hs["missing"]]
        _missing += per_seed_missing(hard_by_seed, t, fields=_pf)
        _missing += _scene["missing"] + _sc["missing"] + _sc["low_sample"]
        _violations = (_hs["violations"]
                       + per_seed_gate_violations(hard_by_seed, t, fields=_pf)
                       + _scene["violations"])
        if _scene["violations"]:
            print(f"[rounds] 逐场景硬门不通过：{len(_scene['violations'])} 条"
                  f"（整体均值可能仍然是好的——这就是要分场景的原因）")
        if _violations:
            print(f"[rounds] 硬门违反 {len(_violations)} 条（含逐 seed/分场景）")
        # 最终确认（方案 §7）：只允许最终确认程序读最终集；这里是它产物的
        # **消费方**。对不上（换协议/换权重/换最终集）就是输入不一致 -> rejected；
        # 没有记录只说明 R2 未确认，不推翻研究结论（两者不能混成一个通道）。
        _conf = {"status": "missing", "issues": [],
                 "why": "no --final-confirm record given"}
        _conf_rec = None
        if getattr(args, "final_confirm", None):
            _p = Path(args.final_confirm)
            if not _p.is_file():
                _conf = {"status": "mismatch",
                         "issues": [f"confirmation record {_p} does not exist"],
                         "why": "missing file"}
            else:
                _conf_rec = json.loads(_p.read_text(encoding="utf-8"))
                # 绑定**具体 checkpoint**（方案 §10.3）：确认记录里的权重哈希
                # 必须就是本轮某个 seed 的候选权重——"横跨多个 seed 的平均成绩"
                # 不能被登记成已确认的模型。
                _sha_by_seed = {}
                for seed in args.seeds:
                    _cp = (exp_dir(args.run_id) / f"round{rnd}" / f"seed{seed}"
                           / "checkpoint_last.pt")
                    try:
                        _sha_by_seed[str(seed)] = _ckpt_sha16(_cp)
                    except OSError:
                        _sha_by_seed[str(seed)] = "UNREADABLE"
                # 权重一致性在这里单独查（用上面的逐 seed 哈希），所以不给
                # confirmation_check 传 model_sha16——拿记录里的哈希比它自己
                # 等于没查。
                _ck = confirmation_check(
                    _conf_rec, protocol_hash=_protocol_snapshot(t)["hash"],
                    candidate_id=cand_id)
                _matched, _w_issues = _match_confirmed_seed(
                    (_conf_rec or {}).get("model_sha16"), _sha_by_seed)
                if _w_issues:
                    _ck = {**_ck, "status": "mismatch",
                           "issues": list(_ck.get("issues") or []) + _w_issues}
                _conf = {**_ck, "record": str(_p), "matched_seed": _matched,
                         "candidate_sha_by_seed": _sha_by_seed}
        _conf_issues = list(_conf.get("issues") or [])
        if _conf.get("status") == "mismatch" and not _conf_issues:
            _conf_issues = [_conf.get("why") or "confirmation mismatch"]
        dec = decide(pairings=paired, thresholds=t,
                     missing_metrics=_missing,
                     confirmation_issues=_conf_issues,
                     hard_gate_violations=_violations)
        dec = _block_research_promotion(dec, _research)
        # 单因归因（方案 v2 §S5）：停止条件只统计"有意义的无收益"。资格失败/
        # 缺标注/无效因子各自另有原因，不能当"模型没有提升空间"的证据。
        # labels_ok 看**本轮判定依赖的通道**：road-only 按 road_iou 晋级，
        # 公路标签可用即可；线通道模式由审计的 paint 门保证。
        # 审计报告里的 ``coverage`` 本身就是 train split 的逐通道覆盖
        # （`mf_tr.coverage()["train"]`），不要再 .get("train") 一层
        _cov_tr = (_report or {}).get("coverage") or {}
        _labels_ok = bool(int(_cov_tr.get(
            "road_valid_frames" if road_only else "paint_valid_frames") or 0))
        _outcome = classify_round_outcome(
            decision=dec["decision"], factor_active=bool(_fact_act["active"]),
            labels_ok=_labels_ok, eligibility_ok=not _research)
        blob = {"candidate_id": cand_id,
                "research_only": bool(_research),
                "factor_activity": _fact_act,
                "round_outcome": _outcome,
                # 本轮身份（看板第一面板）：commit/dirty、实际设备、数据入口四计数。
                # 定义写死在这里，避免看板各算一套：
                #   generated=清单全部记录（含被拒）；reviewed=评价集 paint 档有效帧；
                #   trained=训练集 trainable 帧；evaluated=评价集记录数。
                "git_commit": _git_commit(ROOT),
                "git_dirty": _git_dirty_paths(),
                "device": str(getattr(args, "device", "") or ""),
                "data_counts": {
                    "generated": int(_report.get("n_records_train") or 0)
                                 + int(_report.get("n_records_dev") or 0),
                    "reviewed": int(((_report.get("coverage_dev") or {})
                                     .get("paint_valid_frames")) or 0),
                    "trained": int(_cov_tr.get("trainable_frames") or 0),
                    "evaluated": int(_report.get("n_records_dev") or 0)},
                "paint_sources": paint_sources_from(args),
                "paint_source_resolution": _src_res,
                "thresholds": {"config_hash": t.config_hash,
                               "source": str(Path(args.thresholds)
                                             if args.thresholds
                                             else newest_thresholds_file()
                                             or "code defaults")},
                # 完整协议快照 + 哈希（方案 §10.3）：定义/覆盖/资格/统计/聚合/
                # 空间缓冲/阈值一起留档。没有它，判定的口径只能靠"当时大概用
                # 了哪份 docs"来回忆，旧协议路径消失时更会被悄悄换成新口径。
                "protocol": _protocol_snapshot(t),
                "pairings": paired,
                # 三条通道都要落盘，replay 才能逐字重放（方案 G10/A8）：
                # 硬门违反、缺测清单、逐 seed 与分场景的**原始测量**。
                "hard_gate_violations": _violations,
                "missing_metrics": _missing,
                "hard_by_seed": hard_by_seed,
                # v5 计数契约（方案 v2 §S3.7）：判定必须自带**整数计数**，
                # 否则 replay 无法按新分母重判（legacy_replay_note 会明确说明）。
                "counts": _idc.get("counts") or {},
                "counts_by_group": _idc.get("counts_by_group") or {},
                # 评价 run 的缺测（T11）：空列表才是"每个 run 都测到了"，
                # 有内容时必须能在判定文件/看板上看到，不许当成 0 候选
                "eval_run_errors": _idc.get("eval_run_errors") or [],
                "scene_counts": _scene_counts,
                # 逐场景适用性（measured/not_applicable/unknown）：只写
                # scene_counts 的话，看板只能显示"无数据"，看不到
                # "确认真无线"与"有线但 R=0（UNKNOWN）"的分别（独立复核指出）。
                "scene_applicability": _sc["scenes"],
                "per_scene": per_scene,
                "scene_candidates": scene_candidates,
                "final_confirmation": _conf,
                "r2_confirmed": bool(_conf.get("status") == "verified"
                                     and dec["decision"] == "shadow_candidate"),
                "decision": dec,
                # 循环特有：证据与两臂清单（replay 只读上面四项）
                "factor": factor, "applied_flags": extra,
                "skipped_factors": skipped, "data_factor_note": data_note,
                "baseline_runs": [str(r) for r in base_runs],
                "candidate_runs": [str(r) for r in cand_runs],
                "champ_by_seed": champ_by_seed, "cand_by_seed": cand_by_seed,
                # 负例诊断逐 seed + 汇总（整数相加后再算比率）：E1 的
                # "加困难负例有没有减少假线"必须能从判定文件直接读
                "negative_line_by_seed": {"baseline": base_neg,
                                          "candidate": cand_neg},
                "negative_line": {"baseline": sum_negative_summaries(base_neg),
                                  "candidate": sum_negative_summaries(cand_neg)},
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
                # 实际入训**样本数**（来自 checkpoint 的 train_args，不是配置声称）：
                # 等步数对照的另一半证据，也是"四个计数"里"实际入训"的来源。
                "n_train_frames_by_arm": {
                    "baseline": {
                        str(s): (_ckpt_train_args(
                            exp_dir(args.run_id) / "baseline" / f"seed{s}"
                            / "checkpoint_last.pt").get("n_train"))
                        for s in args.seeds},
                    "candidate": {
                        str(s): (_ckpt_train_args(
                            exp_dir(args.run_id) / f"round{rnd}" / f"seed{s}"
                            / "checkpoint_last.pt").get("n_train"))
                        for s in args.seeds}},
                "timing_suspect": timing_notes,
                "timing_repeats": timing_repeats,
                # 评估的是哪个阶段的权重 + 平凡基线参照（见 eval matrix）：
                # 没有这两项，"0.42 的 road_iou"看不出是不是"全预测成路面"白送的
                "eval_checkpoint": "checkpoint_last.pt",
                "worst_frames_by_seed": worst_by_seed,
                # 到没到平台期：没到就是"暂行判定"，防止把瞬态读数当结论
                # plateau_by_seed = 候选臂；plateau_baseline_by_seed = 基线臂。
                # 两臂都到平台期，这一轮的判定才算"可引用"。
                "plateau_by_seed": plateau_by_seed,
                "plateau_baseline_by_seed": plateau_base_by_seed,
                "all_at_plateau": all_at_plateau,
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
        if all_at_plateau is False:
            _bad_c = {s_: v.get("spread") for s_, v in plateau_by_seed.items()
                      if not v.get("at_plateau")}
            _bad_b = {s_: v.get("spread")
                      for s_, v in plateau_base_by_seed.items()
                      if not v.get("at_plateau")}
            print(f"[rounds] 警告：未到平台期——候选臂仍动: {_bad_c}；"
                  f"基线臂仍动: {_bad_b}——本轮判定按方案只能算暂行，"
                  f"不能当结论引用（两臂都要到平台期才算可引用）")
        print(f"[rounds] 第 {rnd + 1}/{args.rounds} 轮 {cand_id} -> "
              f"{dec['decision']}（{pair_metric} "
              f"{[round(x, 4) for x in cand]} vs 基线 "
              f"{[round(x, 4) for x in champ]}，"
              f"identity {hard['candidate_identity_rate']}）")
        history.append(RoundRecord(round_index=rnd, candidate_id=cand_id,
                                   decision=dec["decision"],
                                   reasons=dec["reasons"],
                                   outcome=_outcome))
        # 真实累计（方案 G08）：原来传 0，预算停止条件在 rounds 里永远不触发
        stop = should_stop(cfg, history,
                           gpu_minutes_today=machine_gpu_minutes_today(),
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


def _add_collect_flags(ap) -> None:
    """采集参数：`collect` 与 `run` 两个入口共用同一套开关。"""
    ap.add_argument("--collect-frames", type=int, default=None,
                    help="无人值守采集的帧数（缺省用配置）")
    ap.add_argument("--collect-step-m", type=float, default=None,
                    help="每帧沿路推进的米数；0=静止取帧会得到重复样本")
    ap.add_argument("--collect-roles", nargs="+", default=None,
                    help="采集视角（缺省用配置里的四个：前向两档+左右柱）")
    ap.add_argument("--collect-map", default=None, help="连接器加载的地图名")
    ap.add_argument("--collect-timeout-s", type=int, default=None,
                    help="采集子进程超时（秒）")
    ap.add_argument("--collect-no-follow-road", dest="collect_follow_road",
                    action="store_false", default=None,
                    help="不沿路网走（默认沿路网，避免走到图外）")
    ap.add_argument("--collect-no-annotation", dest="collect_save_annotation",
                    action="store_false", default=None,
                    help="不额外保存引擎调色板帧（默认保存，材料级诊断要用）")
    ap.add_argument("--collect-teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="采集起点：换一个没采过的路段（同一 spawn 连采两次"
                         "会得到重复样本）")
    ap.add_argument("--collect-attach", dest="collect_attach",
                    action="store_true", default=None,
                    help="进入已有会话（无人值守不该用：那个会话可能有人在开）")
    ap.add_argument("--collect-force", action="store_true",
                    help="把「游戏可能有人在开」从阻止降级为警告；资源门不降级，"
                         "用了会在产物里记 forced")



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
    s.add_argument("--allow-road-only", action="store_true",
                   help="road-only 配方：line 类整通道屏蔽，线损失键判 inactive，"
                        "不产出这类提议（方案 v2 §S5/T14）")
    s.set_defaults(func=cmd_propose)

    s = sub.add_parser("evaluate",
                       help="成对 seed + 硬门槛 -> 判定（研究来源自动降级）")
    s.add_argument("--run-id", required=True)
    s.add_argument("--candidate-id", required=True)
    s.add_argument("--pairings", required=True,
                   help="JSON：{metric: {champion:[...], candidate:[...], "
                        "lower_is_better:bool}}")
    s.add_argument("--hard-gate", default=None,
                   help="JSON：{metric: value}，缺测项按 UNKNOWN 处理（不算通过）")
    s.add_argument("--thresholds", default=None)
    s.add_argument("--paint-source", action="append", default=None,
                   metavar="RUN=SOURCE", help="与 rounds 同一口径：以凭证为准")
    s.add_argument("--research-arm", action="store_true",
                   help="显式声明研究臂（来源资格不足时同样会降级）")
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
    s.add_argument("--paint-source", action="append", default=None,
                   metavar="RUN=SOURCE",
                   help="逐 run 指定漆线真值来源；`engine_annotation_partial` "
                        "= 弱监督研究臂（line 通道不屏蔽，但不允许晋级）")
    s.add_argument("--research-arm", action="store_true",
                   help="明标研究臂：判定照算，但不允许晋级（真值不完整）")
    s.add_argument("--eval-runs", nargs="+", required=True,
                   help="开发集目录（最终集不参与搜索）")
    s.add_argument("--final-confirm", default=None, metavar="RECORD.json",
                   help="最终确认记录（m5_final_set.py confirm 的产物）。"
                        "给了就按它校验协议/候选/权重：对不上 -> rejected；"
                        "没给 = 只记 R2 未确认，不推翻研究结论")
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

    s = sub.add_parser("run",
                       help="一轮：资源门 + 锁 + 命令清单（默认 dry-run）；"
                            "停止条件满足时 rc=8，资源门未过 rc=5，"
                            "采集被拦 rc=6/失败 rc=7")
    s.add_argument("--run-id", required=True)
    s.add_argument("--candidate-id", default="")
    s.add_argument("--dataset-id", default="")
    s.add_argument("--config", default=str(exp_dir("")/"loop_config.json"))
    s.add_argument("--dry-run", action="store_true", default=True)
    s.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    s.add_argument("--gpu-minutes-today", type=float, default=0.0)
    # 用户活动信号（方案 §8.2）：**缺省是去探测**（None），不是写死的 false。
    # 这两个开关是"显式声明"，用于测试或人工指定，会标明来源为未测量。
    s.add_argument("--user-active", dest="user_active", action="store_const",
                   const=True, default=None,
                   help="声明用户正在用机器（不探测）；缺省=真的探测")
    s.add_argument("--user-inactive", dest="user_active", action="store_const",
                   const=False, help="声明用户没在用机器（不探测）")
    s.add_argument("--train-args", default="")
    s.add_argument("--force-lease", action="store_true",
                    help="显式抢走仍存活持有者的机器级租约（长任务"
                         "心跳过期时才需要；会记 recovery_reason）")
    _add_collect_flags(s)

    s.set_defaults(func=cmd_run)

    s = sub.add_parser("collect",
                       help="无人值守采集一次（前置检查+身份审计，不训练）")
    s.add_argument("--run-id", required=True)
    s.add_argument("--config",
                   default=str(exp_dir("") / "loop_config.json"))
    _add_collect_flags(s)
    s.set_defaults(func=cmd_collect)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
