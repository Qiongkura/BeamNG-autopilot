"""线口径的**协议开关**（v7 默认 / v8 候选）：一处决定三项候选集/掩码策略。

为什么要有它：三项变更（合并补齐、横向范围、掩码外观门）各自是独立 env 开关，
散在三个模块里；采纳时"切默认"如果靠手改三处，容易只改一半（本项目已经踩过
"改了一半的口径"的坑：静默 no-op、debug 键互相覆盖）。这里把它们收成一个
**协议级开关**：

* ``BEAMNG_PROTOCOL`` 未设或 ``v7`` → 三项全关（**冻结默认，与历史判定可比**）；
* ``BEAMNG_PROTOCOL=v8`` → 三项全开，其中横向范围取
  ``LANE_PAIR_NEAR_MAX_M``（**规划配对的可达边界，5.5 m**）——L 的值有代码依据，
  不是调参；
* 单项 env（``BEAMNG_LINE_LAT_MAX_M`` / ``BEAMNG_LINE_MERGE_FINAL`` /
  ``BEAMNG_LINE_APPEARANCE_GATE``）**优先于**协议默认，供单因子测量与消融。

实测（seed 42–47，base6x）：v8 全开时 R3 认证集五门全过（覆盖 0.860、身份
0.716、精度 0.65/0.68、召回 0.746、角色 0.793），dev 三门全过（覆盖 0.96、
身份 0.671、角色 0.88–0.90）；代价见
``docs/T16_DECISION_SUMMARY_PROTOCOL_V8_20261005.md``（外观门把"线"收窄成"漆"，
dev 人工标签 29% 非漆；臂间灵敏度压缩）。
"""

from __future__ import annotations

import os

from beamng_autopilot.lane.constants import LANE_PAIR_NEAR_MAX_M

#: 默认协议（冻结）：三项全关，与 v7 历史判定可比
DEFAULT_PROTOCOL = "v7"
#: v8 的横向范围 = 规划配对的"近"候选可达上限（有代码依据，不是调参）
LAT_MAX_M_V8 = float(LANE_PAIR_NEAR_MAX_M)


def protocol() -> str:
    """当前协议：``BEAMNG_PROTOCOL`` 的规范化取值（未知值按默认处理）。"""
    val = str(os.environ.get("BEAMNG_PROTOCOL", "") or "").strip().lower()
    return val if val in ("v7", "v8") else DEFAULT_PROTOCOL


def _flag(name: str, v8_default: bool) -> bool:
    """单项 env > 协议默认。``0/off/false`` 关，``1/on/true`` 开。"""
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() not in ("0", "off", "false", "no")
    return bool(v8_default) and protocol() == "v8"


def lat_max_m() -> float:
    """横向范围上限（米）。0 = 不启用横向门。"""
    raw = os.environ.get("BEAMNG_LINE_LAT_MAX_M")
    if raw is not None and str(raw).strip() not in ("", "off"):
        return float(raw)
    return LAT_MAX_M_V8 if protocol() == "v8" else 0.0


def merge_final_enabled() -> bool:
    """v7 同侧近邻合并是否**补齐**到最终候选集。"""
    return _flag("BEAMNG_LINE_MERGE_FINAL", True)


def appearance_gate_enabled() -> bool:
    """掩码侧外观门（标线像素必须像漆）。"""
    return _flag("BEAMNG_LINE_APPEARANCE_GATE", True)
