"""engine_verified 反伪造：只有凭证支持才算 verified（T16 §4.3 / Order0 §3.4）。

实测风险：`PAINT_SOURCE_RANK["engine_verified"] = "verified"` 只看 sidecar 字符串，
谁把 `label_source` 改成 `engine_verified`（或命令行声明）就能把旧的不完整标签
抬成"已核验真值"。修复：解析必须经凭证里的真值证明（``truth_verified`` 或
``truth_contract=="v1"`` + ``report.verified`` + ``verifier_version``）；否则档位
降 ``absent`` 且理由可见。人工路径（human_revision）与既有 agent/pseudo 语义逐字
不变。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamng_autopilot.experiments.labels import (  # noqa: E402
    PAINT_SOURCE_RANK, audit_label, engine_verified_proof, line_channel_mask,
    mask_line_for_loss, resolve_paint_rank,
)

FORGED_NOTE = ("engine_verified declared but no verifier proof (truth_provenance "
               "missing/unverified): treated as absent")


def _lab() -> np.ndarray:
    lab = np.zeros((20, 30), np.uint8)
    lab[5:15, :] = 1          # road
    lab[10, :8] = 2           # line
    return lab


def _proven_credential() -> dict:
    """一个满足 Order0 §3.4 的凭证（模拟 read_dir_credentials 的返回）。"""
    return {
        "label_source": "engine_verified",
        "path": "runs/probe/annotation.json",
        "readable": True,
        "frames": 12,
        "truth_contract": "v1",
        "truth_provenance": {
            "generator": {"name": "m5_auto_truth_probe", "version": "0.1.0",
                          "sha": "abc123"},
            "asset": {"map": "italy", "segment": "ring_a", "sha": "d4e5"},
            "run": {"id": "probe_001", "scene_seed": 42,
                    "game_version": "0.36", "renderer": "vulkan"},
            "camera": {"name": "front_main", "calibration_sha": "c0ffee",
                       "frame_ids": [0, 1, 2]},
            "labels": {"source_image_sha": "f00d", "label_sha": "beef",
                       "channel_valid_area": 1234, "unknown_reason": ""},
            "report": {"test_report_sha": "5eed", "verifier_version": "0.1.0",
                       "verified": True},
        },
    }


# ---------------------------------------------------------------------------
# 1) engine_verified 的凭证证明
# ---------------------------------------------------------------------------
def test_engine_verified_without_any_proof_is_downgraded_to_absent():
    res = resolve_paint_rank("engine_verified", credential=None)
    assert res["rank"] == "absent", res
    assert res["declared_rank"] == "verified", res     # 声明档位只作对照
    assert res["proof"]["proven"] is False, res
    assert FORGED_NOTE in res["notes"][0], res["notes"]
    # 旧 sidecar（有 label_source，但没有 truth 字段）同样不认
    old_sidecar = {"label_source": "engine_verified",
                   "path": "runs/old/annotation.json", "readable": True,
                   "frames": 8,
                   "annotation": {"reviewer": "someone"}}
    res2 = resolve_paint_rank("engine_verified", credential=old_sidecar)
    assert res2["rank"] == "absent", res2
    assert res2["proof"]["basis"] == "no_truth_provenance", res2["proof"]
    assert FORGED_NOTE in res2["notes"][0], res2["notes"]


def test_contract_v1_with_verified_report_proves_engine_verified():
    res = resolve_paint_rank("engine_verified", credential=_proven_credential())
    assert res["rank"] == "verified", res
    assert res["proof"]["basis"] == "truth_contract_v1_report_verified", res
    assert "supported by credential" in res["notes"][0], res["notes"]
    # 派生键 truth_verified=True（子 agent A 的结论）单独也成立
    res2 = resolve_paint_rank(
        "engine_verified",
        credential={"label_source": "engine_verified", "truth_verified": True})
    assert res2["rank"] == "verified", res2
    assert res2["proof"]["basis"] == "truth_verified_flag", res2
    p = engine_verified_proof(_proven_credential())
    assert set(p) == {"proven", "basis", "why"}, p


def test_incomplete_or_failed_provenance_never_passes():
    base = _proven_credential()
    # 1) 契约版本不对（v0 不算）
    bad = {**base, "truth_contract": "v0"}
    assert resolve_paint_rank("engine_verified", credential=bad)["rank"] == \
        "absent"
    # 2) report.verified 不是 True
    bad = _proven_credential()
    bad["truth_provenance"] = {**bad["truth_provenance"],
                               "report": {"test_report_sha": "5eed",
                                          "verifier_version": "0.1.0",
                                          "verified": False}}
    assert resolve_paint_rank("engine_verified", credential=bad)["rank"] == \
        "absent"
    # 3) 没有 verifier_version（无法追责到具体校验器版本）
    bad = _proven_credential()
    bad["truth_provenance"] = {**bad["truth_provenance"],
                               "report": {"test_report_sha": "5eed",
                                          "verified": True}}
    res = resolve_paint_rank("engine_verified", credential=bad)
    assert res["rank"] == "absent" and \
        res["proof"]["basis"] == "provenance_incomplete", res
    # 4) truth_verified 显式为 False 且无 provenance
    res = resolve_paint_rank(
        "engine_verified",
        credential={"label_source": "engine_verified", "truth_verified": False})
    assert res["rank"] == "absent", res


def test_human_and_existing_ranks_are_unchanged():
    # 人工逐帧修订有自己的复核记录，不要求 truth_provenance（语义不变）
    res = resolve_paint_rank("human_revision")
    assert res["rank"] == "verified" and res["notes"] == [], res
    assert resolve_paint_rank("human_revision", credential=None)["rank"] == \
        "verified"
    for src, rank in (("engine_annotation", "unreliable"),
                      ("engine_annotation_partial", "pseudo"),
                      ("agent_revision", "agent"),
                      ("pseudo", "pseudo"), ("none", "absent"),
                      ("no-such-source", "absent")):
        r = resolve_paint_rank(src)
        assert r["rank"] == rank and r["notes"] == [], (src, r)
    # 声明档位表本身不变（其他模块按它取名义档位；解析必须走 resolve_paint_rank）
    assert PAINT_SOURCE_RANK["engine_verified"] == "verified"
    assert PAINT_SOURCE_RANK["human_revision"] == "verified"
    assert PAINT_SOURCE_RANK["agent_revision"] == "agent"


# ---------------------------------------------------------------------------
# 2) audit_label 的接线：伪造的 engine_verified 连弱监督都不给
# ---------------------------------------------------------------------------
def test_audit_label_masks_a_forged_engine_verified_channel():
    lab = _lab()
    audit = audit_label(lab, paint_source="engine_verified")
    assert audit.paint.rank == "absent", audit.paint
    assert audit.paint.valid is False and audit.paint.usable is False, audit
    assert FORGED_NOTE in audit.paint.reason, audit.paint.reason
    assert any(FORGED_NOTE in n for n in audit.notes), audit.notes
    # line 通道整通道屏蔽：不做正样本，也不把未标注区当负例
    assert not line_channel_mask(lab, audit).any(), "伪造来源不得进监督"
    masked = mask_line_for_loss(lab, audit)
    assert 2 not in np.unique(masked), np.unique(masked)


def test_audit_label_accepts_a_proven_engine_verified_credential():
    lab = _lab()
    audit = audit_label(lab, paint_source="engine_verified",
                        credential=_proven_credential())
    assert audit.paint.rank == "verified", audit.paint
    assert audit.paint.valid is True and audit.paint.usable is True, audit
    assert line_channel_mask(lab, audit).all(), "有证明的来源应保持可信"
    assert mask_line_for_loss(lab, audit).max() == 2


def test_audit_label_without_credential_matches_kwarg_default():
    """credential 缺省 = None = 旧调用；只有 engine_verified 受影响。"""
    lab = _lab()
    for src in ("human_revision", "agent_revision",
                "engine_annotation_partial", "engine_annotation"):
        a = audit_label(lab, paint_source=src)
        b = audit_label(lab, paint_source=src, credential=None)
        assert a.as_dict() == b.as_dict(), src
    # 人工/agent 档位与旧行为逐字一致（本次改动不碰它们）
    agent = audit_label(lab, paint_source="agent_revision").paint
    assert agent.as_dict() == {"valid": False, "reason": agent.reason,
                               "pixels": 8, "usable": True, "rank": "agent"}
    human = audit_label(lab, paint_source="human_revision").paint
    assert human.as_dict() == {"valid": True, "reason": human.reason,
                               "pixels": 8, "usable": True, "rank": "verified"}
    assert human.reason == "paint truth from human_revision", human.reason
