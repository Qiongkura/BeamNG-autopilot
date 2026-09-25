"""部分标签的训练语义（方案 W1 §6.2）："可以训练" ≠ "未标像素都是背景"。

方案点名五个确定性测试，其中"区域级监督屏蔽""全 ignore 不 NaN"是损失级测试，
放在 `tests/test_seg_losses.py`；这里钉**来源语义**这两条：

* 弱标签来源（`engine_annotation_partial` / `agent_revision`）下，**零标线帧不制造
  标线负例**（整通道屏蔽）——没标注不是"确实没有标线"；
* 已核验来源（`human_revision`）下，零标线帧**可以**贡献负例——人工确认过
  "这里没有线"，模型必须能学到这一点；
* 默认来源（`engine_annotation`）连有标线的帧都不监督（`usable=False`）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "m5_train_seg_partial", ROOT / "scripts" / "m5_train_seg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_train_seg_partial"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_weak_labels_do_not_turn_unannotated_frames_into_negatives():
    """弱来源下，零标线帧整通道屏蔽（不制造负例）；有标线的帧仍然监督。"""
    tr = _load_trainer()
    with_line = torch.zeros((1, 8, 8), dtype=torch.int64)
    with_line[0, 4, :4] = 2
    empty = torch.zeros((1, 8, 8), dtype=torch.int64)
    for src in ("engine_annotation_partial", "agent_revision"):
        flags = tr.line_supervision_flags(torch.cat([with_line, empty]), src)
        assert flags == [True, False], (src, flags)
    # 默认来源（engine_annotation）连有标线的帧都不监督（usable=False）
    assert tr.line_supervision_flags(torch.cat([with_line, empty]),
                                     "engine_annotation") == [False, False]


def test_verified_labels_allow_a_confirmed_empty_frame_as_a_negative():
    """已核验来源下，零标线帧是"确实没有标线"，可以贡献负例。"""
    tr = _load_trainer()
    empty = torch.zeros((1, 8, 8), dtype=torch.int64)
    assert tr.line_supervision_flags(empty, "human_revision") == [True], \
        "人工确认的无线帧必须能当负例（否则模型永远学不到'这里没有线'）"
    with_line = torch.zeros((1, 8, 8), dtype=torch.int64)
    with_line[0, 4, :4] = 2
    assert tr.line_supervision_flags(with_line, "human_revision") == [True]
