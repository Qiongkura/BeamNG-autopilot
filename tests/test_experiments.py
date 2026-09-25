"""T14 阶段 A/B 的门与协议：按方案点名的用例逐条固定。

方案 §130 列的测试覆盖：缺标签仍有可见漆线、部分标签损失屏蔽、内容重复、
跨视角/近邻泄漏、训练与最终集串用、并发锁、中断恢复、负收益淘汰、缺列门槛、
事件重放。这些都用合成小数据跑，不碰 GPU、不碰游戏。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "scripts"))

from beamng_autopilot.experiments import gates  # noqa: E402


# ---------------------------------------------------------------------------
# 缺标签仍有可见漆线 / 部分标签损失屏蔽
def _frame_with_visible_line(road_px: int = 800, line_px: int = 40):
    """RGB 里有一道亮线，但标注里只有路面 —— 这就是"未标出的可见漆线"。"""
    h, w = 30, 40
    colour = np.full((h, w, 3), 60, np.uint8)
    colour[10:20, 5:9] = 235                 # 可见漆线（很白）
    label = np.zeros((h, w), np.uint8)
    label[5:25, :] = 1                       # 只标了路面
    return colour, label


def test_visible_paint_without_truth_is_not_a_negative() -> None:
    from beamng_autopilot.experiments.labels import (
        audit_label, line_channel_mask, mask_line_for_loss)

    _colour, label = _frame_with_visible_line()
    audit = audit_label(label, paint_source="engine_annotation")
    assert audit.road.valid and not audit.paint.valid
    assert audit.unknown_reason.startswith("paint truth unusable")
    # 通道掩码整帧为 False：line 损失与 line 指标都不许看这一帧
    assert not line_channel_mask(label, audit).any()
    masked = mask_line_for_loss(label, audit)
    # 屏蔽后：有效区(label!=255)里不存在 line 类，所以"亮线"不可能被当负例
    effective = masked != 255
    assert not (masked[effective] == 2).any()
    assert int((masked == 1).sum()) == int((label == 1).sum()), \
        "road 标签必须原样保留（部分标签屏蔽不能连路面一起丢）"


def test_a_verified_paint_source_keeps_the_line_channel() -> None:
    from beamng_autopilot.experiments.labels import (
        audit_label, line_channel_mask, mask_line_for_loss)

    _colour, label = _frame_with_visible_line()
    label[10:20, 5:9] = 2                    # 人工修订/已验证真值
    audit = audit_label(label, paint_source="human_revision")
    assert audit.paint.valid and audit.trainable
    assert line_channel_mask(label, audit).all()
    masked = mask_line_for_loss(label, audit)
    assert int((masked == 2).sum()) == int((label == 2).sum())


def test_pseudo_labels_are_refused_as_truth() -> None:
    from beamng_autopilot.experiments.labels import audit_label

    _colour, label = _frame_with_visible_line()
    label[10:20, 5:9] = 2
    audit = audit_label(label, paint_source="pseudo")
    assert not audit.paint.valid
    assert "pseudo" in audit.paint.reason


# ---------------------------------------------------------------------------
# 内容重复 / 跨视角与相邻帧泄漏 / 训练与最终集串用
def _write_run(root: Path, name: str, *, frames: int, map_name: str,
               source_id: str, dup_of: Path | None = None,
               exposure_step: int = 1, base_gray: int = 60) -> Path:
    d = root / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    meta_frames = []
    for i in range(frames):
        colour = np.full((12, 16, 3), base_gray + i, np.uint8)
        if dup_of is not None:
            src = sorted(dup_of.glob("frame_*.npz"))[i]
            colour = np.asarray(np.load(src)["colour"], np.uint8)
        label = np.zeros((12, 16), np.uint8)
        label[4:8, :] = 1
        label[6, 2:6] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
        meta_frames.append({"i": i, "view": "front_main",
                            "exposure": i * exposure_step,
                            "t_wall": 1000.0 + i, "path":
                            f"front_main/frame_{i:05d}.npz",
                            "pixels": 12 * 16, "line_pixels": 0})
    (root / name / "meta.json").write_text(json.dumps({
        "stamp": "20260924_000000", "roles": {"front_main": frames},
        "width": 16, "height": 12, "classes": ["background", "road", "line"],
        "map_name": map_name, "map_name_source": "test",
        "source_id": source_id, "frames": meta_frames,
    }), encoding="utf-8")
    return d


def test_byte_identical_frames_are_rejected_not_split(tmp_path) -> None:
    from beamng_autopilot.experiments.manifest import DatasetManifest

    _write_run(tmp_path, "coll_a", frames=4, map_name="italy",
               source_id="ring_a")
    # 第二组把第一组的帧原样复制过来（内容哈希相同）
    _write_run(tmp_path, "coll_b", frames=4, map_name="italy",
               source_id="ring_b", dup_of=tmp_path / "coll_a" / "front_main")
    mf = DatasetManifest.build(
        [tmp_path / "coll_a" / "front_main", tmp_path / "coll_b" / "front_main"],
        root=tmp_path)
    rej = mf.rejected()
    assert len(rej) == 4, "复制的 4 帧必须被隔离，而不是分到另一侧"
    assert all("byte-identical" in r.reject_reason for r in rej)
    audit = mf.audit()
    assert audit["content_overlap"]["n"] == 0, "隔离后不存在跨集合同内容"
    assert audit["n_rejected"] == 4


def test_group_isolation_keeps_neighbours_and_exposures_together(tmp_path) -> None:
    from beamng_autopilot.experiments.manifest import DatasetManifest

    _write_run(tmp_path, "coll_a", frames=5, map_name="italy",
               source_id="ring_a", exposure_step=1, base_gray=40)
    _write_run(tmp_path, "coll_b", frames=5, map_name="italy",
               source_id="ring_b", exposure_step=1, base_gray=120)
    mf = DatasetManifest.build(
        [tmp_path / "coll_a" / "front_main", tmp_path / "coll_b" / "front_main"],
        root=tmp_path, dev_groups=["italy/ring_b"])
    audit = mf.audit()
    assert audit["n_rejected"] == 0,         "两组内容不同，不该有任何帧被隔离（否则本测试是假通过）"
    assert audit["group_overlap"]["n"] == 0
    assert audit["exposure_overlap"]["n"] == 0
    assert audit["exposure_overlap"]["checked"] is True, \
        "有曝光计数就必须真的检查过，不能报未检查"
    # 相邻帧（同组、t_wall 相差 1 s）必然同侧：整组划分的硬保证
    sides: dict = {}
    for r in mf.records:
        sides.setdefault(r.source_id, set()).add(r.split)
    assert all(len(s) == 1 for s in sides.values())


def test_a_consumed_set_may_not_pose_as_the_final_set(tmp_path) -> None:
    from beamng_autopilot.experiments.manifest import DatasetManifest

    _write_run(tmp_path, "holdout_wide2_20260924", frames=3,
               map_name="italy", source_id="ring_old", base_gray=40)
    _write_run(tmp_path, "coll_new", frames=3, map_name="italy",
               source_id="ring_new", base_gray=150)
    mf = DatasetManifest.build(
        [tmp_path / "holdout_wide2_20260924" / "front_main",
         tmp_path / "coll_new" / "front_main"],
        root=tmp_path, final_groups=["italy/ring_old"])
    assert mf.faces_known_to_contain(), \
        "T13 已查看过的集合不得充当最终集"
    # 最终集冻结后只能用于一次确认，不能被搜索反复使用
    with pytest.raises(ValueError):
        DatasetManifest.build([tmp_path / "coll_new" / "front_main"],
                              root=tmp_path).freeze_final()
    mf.groups = {"italy/ring_new": "final", "italy/ring_old": "train"}
    for r in mf.records:
        r.split = "final" if r.source_id == "ring_new" else "train"
    assert mf.freeze_final()["n_final"] == 3
    with pytest.raises(PermissionError):
        mf.assert_final_unused(purpose="search")
    mf.assert_final_unused(purpose="final_confirmation")   # 允许这一次


def test_a_dataset_id_changes_when_the_content_changes(tmp_path) -> None:
    from beamng_autopilot.experiments.manifest import DatasetManifest

    _write_run(tmp_path, "coll_a", frames=3, map_name="italy",
               source_id="ring_a")
    runs = [tmp_path / "coll_a" / "front_main"]
    first = DatasetManifest.build(runs, root=tmp_path)
    np.savez(sorted(runs[0].glob("frame_*.npz"))[0],
             colour=np.full((12, 16, 3), 200, np.uint8),
             label=np.zeros((12, 16), np.uint8))
    second = DatasetManifest.build(runs, root=tmp_path)
    assert first.dataset_id != second.dataset_id, \
        "内容变了就不是同一个数据版本"
    p = first.save(tmp_path / "ds.json")
    with pytest.raises(FileExistsError):
        second.save(p)                                    # 不可变：拒绝覆盖
    assert DatasetManifest.load(p).dataset_id == first.dataset_id


# ---------------------------------------------------------------------------
# 并发锁
def test_two_controllers_cannot_run_at_once(tmp_path) -> None:
    from beamng_autopilot.experiments.controller import InstanceLock

    holder = subprocess.Popen([sys.executable, "-c",
                               "import time; time.sleep(30)"])
    try:
        lock = InstanceLock(tmp_path / "ctrl.lock")
        assert lock.acquire(pid=holder.pid)["acquired"] is True
        res = lock.acquire(pid=holder.pid + 1)
        assert res["acquired"] is False and "second controller" in res["reason"]
    finally:
        holder.terminate()
        holder.wait(timeout=15)
    assert InstanceLock(tmp_path / "ctrl.lock").acquire(pid=holder.pid + 1)[
        "acquired"] is True, "持有者死后必须能接管（断电恢复）"


def test_the_resource_gate_blocks_on_budget_and_vram() -> None:
    from beamng_autopilot.experiments.controller import (
        LoopConfig, ResourceState, resource_gate)

    cfg = LoopConfig(dry_run=True, daily_gpu_minutes=100)
    ok = resource_gate(cfg, ResourceState(now_hour=3, free_vram_mb=8000,
                                          free_disk_gb=50,
                                          gpu_minutes_today=10))
    assert ok["allowed"] and any("dry_run" in w for w in ok["warnings"])
    bad = resource_gate(cfg, ResourceState(now_hour=3, free_vram_mb=100,
                                           free_disk_gb=50,
                                           gpu_minutes_today=150))
    assert bad["allowed"] is False
    assert any("GPU budget" in r for r in bad["reasons"])
    assert any("VRAM" in r for r in bad["reasons"])


def test_the_wall_clock_limit_is_consumed() -> None:
    """G08 反例：`max_wall_minutes` 必须真的能停，不能只是配置里的一个数字。

    实测缺口：`max_wall_minutes` 在核查的入口里**没有任何消费点**——配置写着
    180 分钟，跑 8 小时也不会停（方案 §8.1：停止条件要在下一轮前检查）。
    """
    from beamng_autopilot.experiments.controller import LoopConfig, should_stop

    cfg = LoopConfig(max_wall_minutes=60.0, daily_gpu_minutes=0.0,
                     max_candidates=99, max_rounds_without_gain=99)
    st = should_stop(cfg, [], gpu_minutes_today=0, candidates_used=0,
                     wall_minutes=61.0)
    assert st["stop"] is True, st
    assert any("wall" in r.lower() for r in st["reasons"]), st["reasons"]
    # 没到上限不停；上限 <= 0 = 不限制（与每日上限同一口径）
    assert should_stop(cfg, [], gpu_minutes_today=0, candidates_used=0,
                       wall_minutes=59.0)["stop"] is False
    unlimited = LoopConfig(max_wall_minutes=0.0, daily_gpu_minutes=0.0,
                           max_candidates=99, max_rounds_without_gain=99)
    assert should_stop(unlimited, [], gpu_minutes_today=0, candidates_used=0,
                       wall_minutes=9999.0)["stop"] is False


def test_an_absent_user_activity_signal_is_unknown_not_false() -> None:
    """G08/A6：用户活动信号**未接入**时记 UNKNOWN，不能写死 false 冒充检测。

    方案 §8.2：「不能用写死的 user_active=false 冒充检测」「记录原始检测值和
    状态变更」。所以三态：True（测到在用）/ False（测到没人）/ None（没接入）。
    """
    from beamng_autopilot.experiments.controller import (
        LoopConfig, ResourceState, resource_gate, user_activity_probe)

    cfg = LoopConfig(dry_run=False, pause_while_user_active=True,
                     daily_gpu_minutes=0.0, min_free_vram_mb=0.0,
                     min_free_disk_gb=0.0)
    # 测到用户在动 -> 暂停
    busy = resource_gate(cfg, ResourceState(now_hour=3, free_vram_mb=8000,
                                            free_disk_gb=50,
                                            gpu_minutes_today=0,
                                            user_active=True, user_idle_s=1.0))
    assert busy["allowed"] is False
    assert any("user" in r for r in busy["reasons"])
    # 没接入 -> 放行但**必须警告**"未接入"，且状态里保留 None（不是 False）
    unknown = ResourceState(now_hour=3, free_vram_mb=8000, free_disk_gb=50,
                            gpu_minutes_today=0, user_active=None,
                            user_activity_source="not connected")
    gate = resource_gate(cfg, unknown)
    assert gate["allowed"] is True
    assert any("未接入" in w or "not connected" in w for w in gate["warnings"]),         gate["warnings"]
    assert unknown.user_active is None
    # 探测函数在无法读取时必须返回 None（未接入），绝不回落到 False
    probe = user_activity_probe(idle_probe=lambda: None)
    assert probe["active"] is None and probe["source"] == "not connected", probe
    # 有信号时给原始值与阈值，便于事后核对状态变更
    probe2 = user_activity_probe(idle_probe=lambda: 5.0, idle_threshold_s=120.0)
    assert probe2["active"] is True and probe2["idle_s"] == 5.0
    probe3 = user_activity_probe(idle_probe=lambda: 600.0, idle_threshold_s=120.0)
    assert probe3["active"] is False and probe3["idle_s"] == 600.0


def test_the_run_clock_survives_a_restart_and_resets_after_a_stop() -> None:
    """单次连续运行的墙钟：起点落盘（重启不重置），停止后重开一轮。"""
    import time as _time

    from beamng_autopilot.experiments.controller import RunClock

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        clk = RunClock(Path(td) / "run_state.json")
        first = clk.start()
        assert clk.minutes() < 1.0
        # 第二次读取（模拟重启）不重置起点
        again = RunClock(Path(td) / "run_state.json").start()
        assert again["started_at"] == first["started_at"]
        # 手改起点 -> 墙钟反映真实经过时间
        clk.write_started_at(_time.time() - 3600)
        assert 59.0 < clk.minutes() < 61.0
        clk.reset()
        assert clk.minutes() < 1.0


def test_the_stop_rule_counts_consecutive_rounds_without_gain() -> None:
    from beamng_autopilot.experiments.controller import (
        LoopConfig, RoundRecord, should_stop)

    cfg = LoopConfig(max_rounds_without_gain=3)
    hist = [RoundRecord(i, f"c{i}", "rejected", ["no gain"], gain=0.0)
            for i in range(3)]
    st = should_stop(cfg, hist, gpu_minutes_today=10, candidates_used=3)
    assert st["stop"] and st["no_gain_streak"] == 3
    hist[-1] = RoundRecord(2, "c2", "shadow_candidate", ["improved"], gain=0.1)
    assert should_stop(cfg, hist, gpu_minutes_today=10,
                       candidates_used=3)["stop"] is False


# ---------------------------------------------------------------------------
# 中断恢复所需的 checkpoint 字段
def test_the_checkpoint_carries_everything_resume_needs() -> None:
    from beamng_autopilot.experiments.checkpoint import (
        checkpoint_extras, missing_extras, rng_state, restore_rng,
        weights_equal)

    import torch
    ckpt = {"state_dict": {"w": torch.zeros(2)},
            "optimizer": {}, "scheduler": {}, "next_epoch": 3,
            "dataset_id": "ds1", **checkpoint_extras(dataset_id="ds1")}
    assert missing_extras(ckpt) == [], "恢复字段齐了才允许声称可复现"
    old = {"state_dict": {"w": torch.zeros(2)}, "optimizer": {},
           "scheduler": {}, "next_epoch": 1}
    miss = missing_extras(old)
    assert "dataset_id" in miss and "torch_rng" in miss and \
        "numpy_rng" in miss, "旧 checkpoint 缺随机状态与数据版本，必须报出来"

    state = rng_state()
    assert {"torch_rng", "numpy_rng", "python_rng"} <= set(state)
    assert restore_rng(state) is True
    assert restore_rng({}) is False, "缺项要返回 False，不能假装恢复成功"
    # 落盘/搬设备的形态都要能恢复：CUDA 上的张量、list、bytes 表示
    assert restore_rng({"torch_rng": state["torch_rng"].cuda(),
                        "numpy_rng": state["numpy_rng"],
                        "python_rng": state["python_rng"]}) is         (True if torch.cuda.is_available() else True),         "map_location=cuda 读回来的 RNG 状态必须能恢复（实测这里出过 TypeError）"
    assert restore_rng({"torch_rng": list(state["torch_rng"].tolist()),
                        "numpy_rng": state["numpy_rng"],
                        "python_rng": state["python_rng"]}) is True

    a = {"w": torch.zeros(3)}
    b = {"w": torch.zeros(3)}
    assert weights_equal(a, b)["n_diff"] == 0
    b["w"][0] = 1e-7
    eq = weights_equal(a, b)
    assert eq["n_diff"] == 1 and eq["max_abs_diff"] > 0


# ---------------------------------------------------------------------------
# 负收益淘汰 / 缺列门槛 / 重放确定性
def _pair(metric, champ, cand, lower=False):
    from beamng_autopilot.experiments.gates import paired_compare
    return paired_compare(metric, champ, cand, lower_is_better=lower)


def test_a_net_negative_round_is_eliminated_with_reasons() -> None:
    from beamng_autopilot.experiments.gates import Thresholds, decide

    t = Thresholds()
    worse = decide(pairings={
        "line_recall": _pair("line_recall", [0.80, 0.81, 0.79],
                             [0.70, 0.71, 0.69]),
        "offroad_false_line_px": _pair("offroad_false_line_px",
                                       [10000, 11000, 9000],
                                       [20000, 21000, 19000], lower=True)},
        thresholds=t)
    assert worse["decision"] == "rejected"
    assert any("champion_better" in r for r in worse["reasons"])

    flat = decide(pairings={
        "line_recall": _pair("line_recall", [0.80, 0.79], [0.80, 0.79])},
        thresholds=t)
    assert flat["decision"] in ("rejected", "needs_evidence")

    one_seed = decide(pairings={
        "line_recall": _pair("line_recall", [0.8], [0.9])}, thresholds=t)
    assert one_seed["decision"] == "needs_evidence", \
        "单个最好 seed 不许晋级"

    good = decide(pairings={
        "line_recall": _pair("line_recall", [0.80, 0.79, 0.81],
                             [0.85, 0.84, 0.86]),
        "offroad_false_line_px": _pair("offroad_false_line_px",
                                       [10000, 11000, 9000],
                                       [8000, 9000, 7000], lower=True)},
        thresholds=t)
    assert good["decision"] == "shadow_candidate"
    assert "shadow only" in good["reasons"][-1]

    void = decide(pairings={}, thresholds=t, production_mismatch=True)
    assert void["decision"] == "rejected" and "void" in void["reasons"][0]


def test_a_missing_metric_is_not_a_pass() -> None:
    from beamng_autopilot.experiments.gates import (
        Thresholds, threshold_violations)

    t = Thresholds()
    v = threshold_violations({"line_recall": 0.9}, t)
    assert any("candidate_identity_rate: UNKNOWN" in x for x in v)
    assert any("inference_ms_p95: UNKNOWN" in x for x in v)
    full = threshold_violations({
        "candidate_identity_rate": 0.7, "line_recall": 0.8,
        "line_precision": 0.5, "offroad_false_ratio": 0.01,
        "inference_ms_p95": 20.0}, t)
    assert full == []


def test_the_same_inputs_give_the_same_decision() -> None:
    from beamng_autopilot.experiments.gates import Thresholds, decide

    t = Thresholds()
    pairings = {"line_recall": _pair("line_recall", [0.80, 0.79, 0.81],
                                     [0.85, 0.84, 0.86])}
    a = decide(pairings=pairings, thresholds=t)
    b = decide(pairings=dict(pairings), thresholds=t)
    assert a == b, "相同输入重放必须得到相同决策与理由"
    assert t.config_hash == Thresholds().config_hash, "阈值不许随成绩漂移"
    assert Thresholds(line_recall_min=0.75).config_hash != t.config_hash


# ---------------------------------------------------------------------------
# 事件重放（含半写行与重启重复点）
def _event(run_id: str, phase: str, status: str, **kw):
    from beamng_autopilot.experiments.events import Event
    base = dict(run_id=run_id, candidate_id="cand1", dataset_id="ds1",
                config_hash="cfg1", seed=42)
    base.update(kw)
    return Event(phase=phase, status=status, **base)


def test_event_log_replays_without_double_drawing(tmp_path) -> None:
    from beamng_autopilot.experiments.events import EventLog, metric

    log = EventLog(tmp_path / "run1")
    log.append(_event("run1", "queued", "start"))
    log.append(_event("run1", "auditing", "ok"))
    log.append(_event("run1", "training", "epoch_end", epoch=0,
                      metrics={"train_loss": metric(2.0, "loss"),
                               "val_line_iou": metric(0.1, "iou")}))
    # 断电：最后一行只有半条 JSON，且**没有换行**（最坏情况）
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": 1, "run_id": "run1", "phase": "trai')

    # 重启后重复写同一个 epoch（序号不同，点是一个）
    log.append(_event("run1", "training", "epoch_end", epoch=0,
                      metrics={"train_loss": metric(2.0, "loss")}))
    rep = log.replay()
    # 5 条写入（含半写行）里可读 4 条，其中 1 条是重启后的重复点 -> 保留 3 条
    assert rep["n_events"] == 3
    assert rep["dropped_duplicates"] == 1, "重复点必须被去掉并计数"
    assert any("unreadable" in p for p in rep["problems"]), \
        "半写行要报出来，不能静默跳过"
    epochs = log.epochs()
    assert [e["epoch"] for e in epochs] == [0]


def test_a_missing_measurement_is_not_zero() -> None:
    from beamng_autopilot.experiments.events import metric

    m = metric(None, "px", missing="boundary not published")
    assert m["value"] is None and m["missing"] == "boundary not published"
    assert metric(12.0, "ms", numerator=3, denominator=7)["denominator"] == 7


def test_phase_transitions_are_checked(tmp_path) -> None:
    from beamng_autopilot.experiments.events import EventLog

    log = EventLog(tmp_path / "run2")
    log.append(_event("run2", "queued", "start"))
    log.append(_event("run2", "auditing", "ok"))
    with pytest.raises(ValueError):
        log.append(_event("run2", "approved_for_review", "jump"))


# ---------------------------------------------------------------------------
# 提议器
def test_proposals_change_one_factor_and_queue_unverifiable_errors() -> None:
    from beamng_autopilot.experiments.proposer import (
        FAMILIES, bucket_errors, needs_review, propose)

    buckets = bucket_errors(pixel={
        "offroad_false_line_px": 9000, "pred_line_px": 30000,
        "missed_true_line_px": 1000, "gt_line_px": 20000, "model": "m"},
        identity={"candidates_total": 100, "candidates_off_road": 70,
                  "role_agreement_rate": 0.5,
                  "candidate_paint_recall_p50": 0.001, "run": "r"})
    q = needs_review(buckets)
    assert any("candidates_not_on_paint" in item["name"] for item in q), \
        "缺可信漆线真值的错误必须进复核队列"
    # 场景配比族的输入是"可用的未入训数据组"：给了才产出可执行因子
    props = propose(buckets=buckets,
                    dataset={"n_train_frames": 173},
                    champion={"steps": 129, "epochs": 3},
                    available_runs=["logs/m5_seg/diverse_curve_20260924/front_main"])
    assert props and all(p.family in FAMILIES for p in props)
    for p in props:
        assert len(p.factor) == 1, "一次实验只改一个因子"
    assert props[0].family == "scene_mix"
    assert list(props[0].factor) == ["add_runs"],         "场景配比族要改的是训练输入本身（可执行），不是没人实现的 group_weights"
    fams = [p.family for p in props]
    assert fams == sorted(fams, key=FAMILIES.index), "提议按方案给定优先级排序"
    # 已经试过的族不再重复提议
    again = propose(buckets=buckets, dataset={"n_train_frames": 173},
                    champion={"steps": 129, "epochs": 3},
                    available_runs=["x"], history=[{"family": props[0].family}])
    assert props[0].family not in [p.family for p in again]


def test_the_proposer_only_emits_keys_the_trainer_can_apply() -> None:
    """提议器与训练器不能"各说各话"：发出的键必须是 rounds 能应用的键。

    实测来源：旧 scene_mix 族发 `group_weights`、困难采样族发
    `hard_neg_manifest`，训练器都没有实现；`rounds` 现在的纪律是"因子未生效
    就拒绝训练"，于是循环会停在"因子未应用"上。这里把两张表钉在一起防漂移。
    """
    import importlib.util
    from pathlib import Path as _P
    from beamng_autopilot.experiments.proposer import (
        APPLICABLE_KEYS, ALLOWED_KEYS, FAMILIES, bucket_errors, propose)

    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop_for_keys", root / "scripts" / "m5_seg_autoloop.py")
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    assert set(APPLICABLE_KEYS) == (set(loop.DATA_FACTORS)
                                   | set(loop.TRAINER_FLAG_FACTORS)),         "提议器的可应用表和 rounds 的白名单漂移了"
    for fam, keys in ALLOWED_KEYS.items():
        assert fam in FAMILIES and keys, fam

    buckets = bucket_errors(pixel={
        "offroad_false_line_px": 9000, "pred_line_px": 30000,
        "missed_true_line_px": 1000, "gt_line_px": 20000, "model": "m"},
        identity={"candidates_total": 100, "candidates_off_road": 70,
                  "role_agreement_rate": 0.5,
                  "candidate_paint_recall_p50": 0.001, "run": "r"})
    blocked: list = []
    props = propose(buckets=buckets, dataset={"n_train_frames": 173},
                    champion={"steps": 129, "epochs": 3},
                    available_runs=[], blocked=blocked)
    assert blocked, "没有可用数据时必须报告原因，而不是静默不提议"
    assert all(k in APPLICABLE_KEYS for p in props for k in p.factor),         f"提议里出现了没有人实现的键：{[p.factor for p in props]}"
    assert not props, "数据/场景类错误 + 没有新数据 -> 不应退到'多训几轮'"

    # 旧调用方式（没有数据因子入口 available_runs=None）保留回退：发 epochs，
    # 并把"这一族想要数据"记进 blocked，供调用方看见
    legacy_blocked: list = []
    legacy = propose(buckets=buckets, dataset={"n_train_frames": 173},
                     champion={"steps": 129, "epochs": 3},
                     blocked=legacy_blocked)
    assert [p.family for p in legacy] == ["epochs"]
    assert any(b["family"] == "scene_mix" for b in legacy_blocked)
    assert all(k in APPLICABLE_KEYS for p in legacy for k in p.factor)


def test_inference_latency_regression_is_a_hard_gate() -> None:
    from beamng_autopilot.experiments.gates import (
        Thresholds, threshold_violations)

    v = threshold_violations({
        "candidate_identity_rate": 0.7, "line_recall": 0.8,
        "line_precision": 0.5, "offroad_false_ratio": 0.01,
        "inference_ms_p95": 90.0}, Thresholds())
    assert any("inference_ms_p95" in x for x in v)


def test_the_digest_index_finds_copies_by_content_not_by_path(tmp_path) -> None:
    """复制样本经常换名字/换采集，路径不可靠；内容哈希才是证据。"""
    from beamng_autopilot.experiments.manifest import content_digest_index

    a = tmp_path / "coll_a" / "front_main"
    b = tmp_path / "coll_b" / "front_main"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    for i in range(3):
        np.savez(a / f"frame_{i:05d}.npz",
                 colour=np.full((8, 8, 3), 40 + i, np.uint8),
                 label=np.zeros((8, 8), np.uint8))
    # 第二组：两张复制自 a（但文件名不同），一张新内容
    np.savez(b / "frame_00000.npz",
             colour=np.full((8, 8, 3), 40, np.uint8),
             label=np.zeros((8, 8), np.uint8))
    np.savez(b / "frame_00001.npz",
             colour=np.full((8, 8, 3), 42, np.uint8),
             label=np.zeros((8, 8), np.uint8))
    np.savez(b / "frame_00002.npz",
             colour=np.full((8, 8, 3), 200, np.uint8),
             label=np.zeros((8, 8), np.uint8))
    rep = content_digest_index([a, b])
    assert rep["n_frames"] == 6 and rep["n_unique_images"] == 4
    assert rep["n_duplicate_groups"] == 2
    assert rep["sampled"] is False
    # 抽样模式必须如实标注抽样过，否则"没有重复"会被误读成"全量检查过"
    rep_s = content_digest_index([a, b], sample=2)
    assert rep_s["sampled"] is True and rep_s["n_frames"] <= 6


def test_the_printed_plan_uses_flags_the_cli_actually_accepts() -> None:
    """计划里打印的命令必须能真的跑起来。

    实测漂移过一次：plan_once 给评估步骤写了 ``--checkpoint``，而
    ``m5_seg_autoloop.py evaluate`` 收的是 ``--pairings/--hard-gate``——打印
    出来的"可执行计划"其实跑不通。这里把两边钉在一起。
    """
    import subprocess
    from pathlib import Path as _P

    from beamng_autopilot.experiments.controller import LoopConfig, plan_once

    ROOT = _P(__file__).resolve().parents[1]
    cfg = LoopConfig(dry_run=True, collect="off")
    plan = plan_once(cfg, run_id="r1", candidate_id="c1", dataset_id="ds1",
                     python="py", script_dir=ROOT / "scripts", seed=42)
    root = _P(__file__).resolve().parents[1]
    helps = {}
    for action in plan.actions:
        cmd = [str(c) for c in action.cmd]
        idx = next((i for i, c in enumerate(cmd)
                    if c.endswith("m5_seg_autoloop.py")), None)
        if idx is None:
            continue
        sub = cmd[idx + 1]
        if sub not in helps:
            out = subprocess.run(
                [sys.executable, cmd[idx], sub, "--help"],
                capture_output=True, text=True, timeout=120)
            helps[sub] = out.stdout
        flags = [c for c in cmd[idx + 2:] if c.startswith("--")]
        for f in flags:
            assert f in helps[sub], (
                f"计划里的 {f} 不在 `{sub} --help` 里：计划与 CLI 契约漂移了")


def test_our_checkpoint_extras_stay_loadable_by_a_weights_only_loader(tmp_path):
    """部署链用 torch.load(weights_only=True)（torch 2.6+ 默认）读权重。

    实测缺陷：`torch.__version__` 是 ``TorchVersion``（str 子类）对象，pickle 会
    把它存成全局类，于是**新写出的 checkpoint 在部署链上加载失败**。这条测试把
    "我们的扩展字段必须能被 weights_only 加载器读"钉死。
    """
    import torch

    from beamng_autopilot.experiments.checkpoint import checkpoint_extras

    ckpt = {"state_dict": {"w": torch.zeros(3)}, "next_epoch": 1,
            **checkpoint_extras(dataset_id="ds1", candidate_id="c1",
                                run_id="r1")}
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["dataset_id"] == "ds1"
    assert isinstance(loaded["env"]["torch"], str)
    assert "TorchVersion" not in str(type(loaded["env"]["torch"]))


def test_seeds_needed_for_effect_tells_you_when_to_stop() -> None:
    """先算"要多少 seed 才判得出来"，再决定投不投 GPU。

    实测背景：3 个 seed 上 road_iou 的配对差 +0.039、sd 0.066 —— 盲目加 seed
    可能在可行规模内永远判不出，必须先估所需规模。
    """
    from beamng_autopilot.experiments.gates import (
        paired_compare, seeds_needed_for_effect)

    assert seeds_needed_for_effect(0.066, 0.039) == 12       # ceil((2*0.066/0.039)^2)
    assert seeds_needed_for_effect(0.05, 0.20) == 1          # 效应大于噪声
    assert seeds_needed_for_effect(0.0, 0.1) is None         # 零噪声：不可估
    assert seeds_needed_for_effect(0.1, 0.0) is None         # 零效应：没必要
    assert seeds_needed_for_effect(0.5, 0.001, max_seeds=200) is None

    # 成对比较的输出里带上它，判定文件因此自带"还需多少 seed"
    r = paired_compare("road_iou", [0.42, 0.42, 0.42], [0.50, 0.43, 0.55])
    assert r["seeds_needed_for_effect"] == 3      # 效应大于噪声：几个 seed 就够

    # 真实实测（t14_3h_roads3d，精确等步数）：+0.039 配 sd 0.066 → 判不出来，
    # 且所需规模 12 个 seed，比"再多跑两三个"大一个量级——先算规模再决定投不投。
    real = paired_compare("road_iou", [0.4239, 0.4241, 0.4231],
                          [0.4239, 0.5396, 0.4242])
    assert real["verdict"] == "inconclusive"
    assert real["seeds_needed_for_effect"] == 12

def test_a_non_positive_daily_cap_means_no_cap():
    """每日 GPU 上限：`<= 0` = 不设上限（用户 2026-09-25 的明确指示）。

    上限机制本身必须还在（填数字就生效），且"不设上限"要**说出来**——资源门
    打印警告说明依据，而不是静默放行；账本照记，随时能看到今天用了多少。
    """
    from beamng_autopilot.experiments.controller import (
        LoopConfig, ResourceState, resource_gate, should_stop,
    )

    st = ResourceState(now_hour=12, free_vram_mb=9000.0, free_disk_gb=50.0,
                       gpu_minutes_today=999.0)
    free = resource_gate(LoopConfig(dry_run=False, daily_gpu_minutes=0.0), st)
    assert free["allowed"] is True, free
    assert any("不设上限" in w for w in free["warnings"]), free["warnings"]
    assert should_stop(LoopConfig(daily_gpu_minutes=0.0), [],
                       gpu_minutes_today=999.0, candidates_used=0)["stop"] is False

    capped = resource_gate(LoopConfig(dry_run=False, daily_gpu_minutes=60.0), st)
    assert capped["allowed"] is False, capped
    assert any("exhausted" in r for r in capped["reasons"]), capped
    assert should_stop(LoopConfig(daily_gpu_minutes=60.0), [],
                       gpu_minutes_today=999.0, candidates_used=0)["stop"] is True
    # 负值同样按"不设上限"处理（配置写 -1 不是"立刻停"）
    neg = resource_gate(LoopConfig(dry_run=False, daily_gpu_minutes=-1.0), st)
    assert neg["allowed"] is True, neg


def test_paint_usable_is_separate_from_valid():
    """标线真值：能学（usable）与 能判（valid）必须分开。

    实测（2026-09-25）：引擎标注的漆线类**存在但不完整**
    （覆盖 RGB 漆线候选约 0.61）——不能当门槛真值，但可以当弱监督。
    只用一个 valid 会把"不能判"误当成"不能学"，白白丢掉唯一的
    标线监督。默认来源（engine_annotation）行为一字未变：usable=False。
    """
    import numpy as np

    from beamng_autopilot.experiments.labels import (
        ClassQuality, PAINT_SOURCE_RANK, audit_label,
    )

    lab = np.zeros((20, 30), np.uint8)
    lab[5:15, :] = 1
    lab[10, :8] = 2
    q = audit_label(lab, paint_source="engine_annotation").paint
    assert (q.valid, q.usable) == (False, False), q
    assert "不完整" in q.reason or "incomplete" in q.reason, q.reason
    q2 = audit_label(lab, paint_source="engine_annotation_partial").paint
    assert (q2.valid, q2.usable) == (False, True), q2
    q3 = audit_label(lab, paint_source="human_revision").paint
    assert (q3.valid, q3.usable) == (True, True), q3
    assert PAINT_SOURCE_RANK["engine_annotation_partial"] == "pseudo"

    # agent 逐帧核对式标注：可训练可测（usable），但晋级仍需人确认

    assert PAINT_SOURCE_RANK["agent_revision"] == "agent"

    q4 = audit_label(lab, paint_source="agent_revision").paint

    assert (q4.valid, q4.usable) == (False, True), q4
    # 老调用方语义不变：usable 缺省等于 valid
    assert ClassQuality(True, "x").usable is True
    assert ClassQuality(False, "x").usable is False
    assert set(ClassQuality(True, "x").as_dict()) == {"valid", "reason",
                                                      "pixels", "usable", "rank"}
    # 档位要逐帧记下来：看板/报告要按档位分别计数（生成帧 ≠ 人工真值帧）
    assert audit_label(lab, paint_source="human_revision").paint.rank == "verified"
    assert audit_label(lab, paint_source="agent_revision").paint.rank == "agent"
    assert audit_label(lab).paint.rank == "unreliable"


def test_the_paired_interval_uses_a_documented_t_interval():
    """不确定度用**明确记录的** t 区间（方案 §10.3）。

    旧口径 `2*sd/sqrt(n)` 只是「乘子取 2」的近似，小样本下会低估区间
    （n=3 时 t=4.303，差 2 倍以上）——而区间宽度直接决定 verdict。
    本测试同时把分位表与 scipy 对照（判定路径不依赖 scipy，但表值要对）。
    """
    from beamng_autopilot.experiments.gates import (
        T95_TABLE, paired_compare, t_critical,
    )

    assert t_critical(0) == float("inf")
    assert t_critical(3) == 3.182 and t_critical(31) == 1.96
    try:
        from scipy import stats
        for df, v in T95_TABLE.items():
            assert abs(v - float(stats.t.ppf(0.975, df))) < 0.0005, df
    except ImportError:
        pass

    r = paired_compare("m", [0.10, 0.12, 0.14], [0.20, 0.23, 0.25])
    assert r["df"] == 2 and "t interval" in r["ci_method"]
    assert r["ci95_halfwidth"] > 2.0 * r["ci95_halfwidth_approx_2sd"], (
        "小样本下 t 区间必须比旧口径宽（旧口径低估）",
        r["ci95_halfwidth"], r["ci95_halfwidth_approx_2sd"])
    assert r["ci95_halfwidth_approx_2sd"] is not None, "旧值要照实报告"
    assert r["verdict"] == "candidate_better"
    # 同一批数据，区间变宽后结论会变：这是口径变更的直接后果
    # 同一批数据在两种口径下结论不同（旧：candidate_better；新：inconclusive）——
    # 这正是"不确定度方法必须写清"的理由：差值的散布大时，旧口径会给出假阳性。
    r2 = paired_compare("m", [0.00, 0.00, 0.00], [0.10, 0.02, 0.20])
    assert r2["mean_delta"] > r2["ci95_halfwidth_approx_2sd"], r2
    assert r2["ci95_halfwidth"] > abs(r2["mean_delta"]), r2
    assert r2["verdict"] == "inconclusive", r2


def test_a_bad_seed_cannot_hide_behind_the_mean():
    """A7 反例：单个坏 seed 不能被跨 seed 均值藏掉（方案 §10.2）。

    实测口径：硬门原来只喂"跨 seed 均值"——4 个 seed 的 line_recall 都是 0.9、
    第 5 个是 0.3，均值 0.78 照样过 0.70。真正待部署的是**具体 checkpoint**，
    它必须自己满足门槛，所以逐 seed 检查必须进硬门。
    """
    t = gates.Thresholds()
    good = {"line_recall": 0.90, "line_precision": 0.60,
            "offroad_false_ratio": 0.02, "inference_ms_p95": 20.0,
            "candidate_identity_rate": 0.80}
    per_seed = {str(s): dict(good) for s in (42, 43, 44, 45)}
    per_seed["46"] = {**good, "line_recall": 0.30}       # 一个坏 seed
    mean = {k: sum(v[k] for v in per_seed.values()) / len(per_seed)
            for k in good}
    # 均值口径看不出问题（这就是缺口）
    assert gates.threshold_violations(mean, t) == [], mean
    v = gates.per_seed_gate_violations(per_seed, t)
    assert v and "46" in v[0] and "line_recall" in v[0], v
    # 逐 seed 违反进硬门 -> rejected（不能被均值抬成 shadow_candidate）
    dec = gates.decide(pairings=_all_better_pairings(gates, t),
                      thresholds=t, missing_metrics=[], hard_gate_violations=v)
    assert dec["decision"] == "rejected", dec
    assert any("line_recall" in r for r in dec["reasons"]), dec["reasons"]


def test_a_bad_scene_cannot_hide_behind_the_pooled_mean():
    """A7 反例：坏场景不能被合并均值抵消；缺测场景记 UNKNOWN（不是通过）。

    方案 §10.2：关键场景不允许被总体均值抵消，覆盖不足时给 needs_evidence。
    """
    t = gates.Thresholds()
    ok = {"line_recall": 0.90, "line_precision": 0.60,
          "offroad_false_ratio": 0.02, "inference_ms_p95": 20.0,
          "candidate_identity_rate": 0.80}
    per_scene = {"italy/ring_a": dict(ok),
                 "italy/ring_b": {**ok, "line_precision": 0.15},
                 "italy/ring_c": dict(ok)}
    rep = gates.scene_report(per_scene, t)
    assert rep["violations"] and "ring_b" in rep["violations"][0], rep
    assert rep["missing"] == [], rep
    # 只有"没测到"的场景才算缺测 -> needs_evidence（不当通过）
    per_scene["italy/ring_d"] = {}
    rep2 = gates.scene_report(per_scene, t)
    # 分场景只把**已标定**的口径当硬门（像素层 + 该场景耗时）；身份/候选类
    # 指标单场景候选数可能只有几个，门槛未标定，所以只上报不进硬门。
    assert len(rep2["missing"]) == len(gates.SCENE_HARD_FIELDS), rep2
    assert all("ring_d" in x and "UNKNOWN" in x for x in rep2["missing"]), rep2
    # 两个通道分开：坏场景 -> rejected（测了不达标）；缺测场景 -> needs_evidence
    # （证据缺失，方案 §10.3 的判定顺序）。混在一起时硬门优先，所以分开断言。
    dec_bad = gates.decide(pairings=_all_better_pairings(gates, t), thresholds=t,
                          missing_metrics=[], hard_gate_violations=rep["violations"])
    assert dec_bad["decision"] == "rejected", dec_bad
    dec_gap = gates.decide(pairings=_all_better_pairings(gates, t), thresholds=t,
                          missing_metrics=rep2["missing"],
                          hard_gate_violations=[])
    assert dec_gap["decision"] == "needs_evidence", dec_gap


def _all_better_pairings(gates, t):
    """一组"任务指标全面改善"的成对结果（用于把判定拉到只看硬门）。"""
    seeds = [42, 43, 44, 45, 46]
    base = {"candidate_identity_rate": 0.62, "line_recall": 0.72,
            "line_precision": 0.42, "offroad_false_line_px": 1000.0,
            "inference_ms_p95": 20.0}
    cand = {"candidate_identity_rate": 0.85, "line_recall": 0.88,
            "line_precision": 0.58, "offroad_false_line_px": 600.0,
            "inference_ms_p95": 16.0}
    a, b = {}, {}
    for i, s in enumerate(seeds):
        a[str(s)] = {k: v + (i - 2) * 0.001 for k, v in base.items()}
        b[str(s)] = {k: v + (i - 2) * 0.001 for k, v in cand.items()}
    lower = ("offroad_false_line_px", "inference_ms_p95")
    return {k: gates.paired_compare(k, [a[str(s)][k] for s in seeds],
                                    [b[str(s)][k] for s in seeds],
                                    lower_is_better=k in lower)
            for k in base}


def test_an_unfrozen_gate_blocks_promotion_even_when_it_is_measured(tmp_path):
    """G09/§10.2：门槛未标定时，**测到了也不许当通过**；replay 必须重放同一结论。

    实测缺口：判定文件只落盘 pairings 与硬门违反，replay 从 pairings 重推缺测
    ——于是"新硬门未标定"这条在线判定加进去、重放时消失，同输入不同决策。
    """
    t = gates.Thresholds()
    cov = gates.paired_compare("candidate_reference_coverage",
                              [0.80, 0.85], [0.86, 0.90])
    assert cov["n"] == 2
    # 未标定 -> 进缺测 -> needs_evidence
    m = gates.missing_metrics_for({"candidate_reference_coverage": cov},
                                coverage_gate_frozen=False)
    assert any("unfrozen" in x for x in m), m
    dec = gates.decide(pairings={"candidate_reference_coverage": cov}, thresholds=t,
                      missing_metrics=m, hard_gate_violations=[])
    assert dec["decision"] == "needs_evidence", dec
    # 标定后（冻结）同一批数字不再被拦——门本身该由阈值管，不是永远拦住
    assert gates.missing_metrics_for({"candidate_reference_coverage": cov},
                                    coverage_gate_frozen=True) == []
    # 没测到的口径任何时候都算缺测
    none_cov = {"candidate_reference_coverage": {"metric": "x", "n": 0}}
    assert gates.missing_metrics_for(none_cov, coverage_gate_frozen=True) ==         ["candidate_reference_coverage"]
