"""训练失败（含 CUDA OOM）必须**留痕**，且不许被读成"还在跑"或"测过了"。

计划 §7 的第 4 条点名："补 CUDA OOM 注入测试"（此前只有预算耗尽与数据缺失
路径有回归）。这条测试盯的是失败之后**看到什么**：

* `record_training_failure` 在异常退出时补一条 ``failed`` 任务记录（图表保留
  失败前的记录，错误摘要写进 ``error``）——不写的话看板会停在最后一条
  ``running`` 上，看起来像还在跑；
* ``failed`` 是状态机里合法的状态（写错名字会被 ``task_record`` 拒掉）；
* 失败记录里**没有**任何"指标=0"的字段：UNKNOWN 不是 0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.metrics import (  # noqa: E402
    MetricsStore, task_record,
)


def _store(tmp_path: Path) -> MetricsStore:
    return MetricsStore(tmp_path / "metrics.jsonl")


def test_a_failed_run_appends_a_failed_task_record(tmp_path, monkeypatch):
    from scripts import m5_train_seg as tr

    store = _store(tmp_path)
    store.append(task_record("r1", "训练任务", "running", total_steps=60,
                             current_step=12, epoch=1, started_at=1000.0))
    tr._MON.clear()
    tr._MON.update({"store": store, "step": 12, "total_steps": 60, "epoch": 1,
                    "args": type("A", (), {"metrics_run": "r1",
                                           "task_name": "训练任务"})()})
    tr.record_training_failure(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))

    recs = [json.loads(ln) for ln in
            store.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tasks = [r for r in recs if r.get("kind") == "task"]
    assert [t["status"] for t in tasks] == ["running", "failed"], tasks
    last = tasks[-1]
    assert "CUDA out of memory" in last["error"], last
    assert last["current_step"] == 12 and last["total_steps"] == 60
    tr._MON.clear()


def test_the_failure_record_does_not_invent_measurements(tmp_path, monkeypatch):
    """失败记录只写失败：不许顺手写 0 分（UNKNOWN 不是 0）。"""
    from scripts import m5_train_seg as tr

    store = _store(tmp_path)
    tr._MON.clear()
    tr._MON.update({"store": store, "step": 0,
                    "args": type("A", (), {"metrics_run": "r2",
                                           "task_name": "t"})()})
    tr.record_training_failure(RuntimeError("CUDA out of memory"))
    blob = json.loads(store.path.read_text(encoding="utf-8").splitlines()[-1])
    for k in ("val_miou", "train_loss", "road_iou", "val_line_iou"):
        assert k not in blob, f"失败记录里不该有 {k}"
    tr._MON.clear()


def test_failure_recording_never_masks_the_original_exception(tmp_path):
    """留痕本身失败（没有 store）不能把原始异常吃掉。"""
    from scripts import m5_train_seg as tr

    tr._MON.clear()                      # 没有 store：record_training_failure 直接返回
    tr.record_training_failure(RuntimeError("CUDA out of memory"))
    tr._MON.update({"store": object()})  # store 是坏对象：也必须吞掉自己的错
    tr.record_training_failure(RuntimeError("CUDA out of memory"))
    tr._MON.clear()


def test_an_unknown_status_is_rejected_instead_of_written(tmp_path):
    """状态名写错要当场炸，不能写进一条没人认识的状态。"""
    import pytest

    with pytest.raises(ValueError):
        task_record("r3", "t", "oom", total_steps=None)
