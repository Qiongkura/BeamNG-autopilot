"""最终集封存与访问账本（方案 §7/§10.3）：一次确认即消费，不可反复调参。

方案原文：「最终集必须有访问与消费记录，绑定协议、候选、调用方和结果；只允许最终
确认程序读取。失败后不得对着同一最终集调参再宣称独立验证；旧集降为已消费诊断集，
新一轮另冻结未使用场景。」

本文件钉住每一条：
* 未封存 → 不是最终集，拒读；
* 搜索/训练用途 → 拒读（只有 `final_confirm` 能读）；
* 封存后内容变了 → 拒（不可变）；
* 协议哈希不符 → 拒（换了口径就不算同一集）；
* 已经确认过 → 拒（**失败也消费**，否则"失败后再调参"就成立）；
* 每次访问（含被拒）都进只追加的账本。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.final_set import (  # noqa: E402
    ALLOWED_PURPOSES, SEAL_NAME, access, confirmation_check,
    confirmation_record, consumption, frame_digests, ledger_path, read_seal,
    seal, verify_seal,
)


def _frames(tmp_path: Path, n: int = 3) -> list:
    d = tmp_path / "final" / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"frame_{i:05d}.npz"
        p.write_bytes(b"frame-%d" % i)
        out.append(p)
    return out


def _seal(tmp_path: Path, *, protocol_hash: str = "ph-1", name: str = "final1"):
    return seal(_frames(tmp_path), name=name, dataset_id="ds-1",
                protocol_hash=protocol_hash, out_dir=tmp_path / "seal",
                sealed_by="pytest", groups=["italy/ring_f"])


def test_seal_records_hashes_and_verifies(tmp_path):
    rec = _seal(tmp_path)
    assert rec["n_frames"] == 3 and len(rec["digest"]) == 16
    assert all(len(v) == 16 for v in rec["digests"].values())
    assert (tmp_path / "seal" / SEAL_NAME).is_file()
    assert read_seal(tmp_path / "seal")["name"] == "final1"
    assert verify_seal(tmp_path / "seal")["ok"] is True
    # 空集 / 读不到的帧都不能封存
    with pytest.raises(ValueError):
        seal([], name="x", dataset_id="d", protocol_hash="p",
             out_dir=tmp_path / "s2")
    with pytest.raises(ValueError):
        seal([tmp_path / "nope.npz"], name="x", dataset_id="d",
             protocol_hash="p", out_dir=tmp_path / "s3")
    assert frame_digests([tmp_path / "nope.npz"]).popitem()[1] == "UNREADABLE"


def test_an_unsealed_directory_is_not_a_final_set(tmp_path):
    got = access(tmp_path / "not_sealed", protocol_hash="ph-1",
                 candidate_id="c1", caller="m5_final_confirm",
                 purpose="final_confirm")
    assert got["allowed"] is False
    assert any("not sealed" in r for r in got["reasons"]), got
    assert ledger_path(tmp_path / "not_sealed").is_file(), "被拒也要留痕"


def test_only_the_final_confirm_purpose_may_read(tmp_path):
    _seal(tmp_path)
    for purpose in ("search", "train", "dashboard", ""):
        got = access(tmp_path / "seal", protocol_hash="ph-1",
                     candidate_id="c1", caller="m5_seg_autoloop",
                     purpose=purpose)
        assert got["allowed"] is False, purpose
        assert any("purpose" in r for r in got["reasons"]), got
    assert ALLOWED_PURPOSES == ("final_confirm",)
    assert consumption(tmp_path / "seal")["confirmed"] == 0
    assert consumption(tmp_path / "seal")["refused"] == 4


def test_one_confirmation_consumes_the_set(tmp_path):
    """一次确认即消费：第二次（无论候选是否相同）都要被拒。"""
    _seal(tmp_path)
    first = access(tmp_path / "seal", protocol_hash="ph-1",
                   candidate_id="cand-A", caller="m5_final_confirm",
                   purpose="final_confirm", result="rejected")
    assert first["allowed"] is True, first
    second = access(tmp_path / "seal", protocol_hash="ph-1",
                    candidate_id="cand-A", caller="m5_final_confirm",
                    purpose="final_confirm")
    assert second["allowed"] is False
    assert any("already consumed" in r for r in second["reasons"]), second
    third = access(tmp_path / "seal", protocol_hash="ph-1",
                   candidate_id="cand-B", caller="m5_final_confirm",
                   purpose="final_confirm")
    assert third["allowed"] is False, "旧集已消费：新一轮必须另冻结未使用场景"
    cons = consumption(tmp_path / "seal")
    assert cons["consumed"] is True and cons["confirmed"] == 1
    assert cons["refused"] == 2 and cons["entries"] == 3
    # 账本绑定协议、候选、调用方与结果
    last = json.loads(ledger_path(tmp_path / "seal").read_text(
        encoding="utf-8").splitlines()[0])
    assert last["protocol_hash"] == "ph-1" and last["candidate_id"] == "cand-A"
    assert last["caller"] == "m5_final_confirm" and last["result"] == "rejected"


def test_a_changed_or_rehashed_set_is_refused(tmp_path):
    _seal(tmp_path)
    # 内容改了
    (tmp_path / "final" / "front_main" / "frame_00000.npz").write_bytes(b"tampered")
    ver = verify_seal(tmp_path / "seal")
    assert ver["ok"] is False and ver["changed"], ver
    got = access(tmp_path / "seal", protocol_hash="ph-1", candidate_id="c1",
                 caller="m5_final_confirm", purpose="final_confirm")
    assert got["allowed"] is False
    assert any("immutability" in r for r in got["reasons"]), got
    # 协议哈希不符
    _seal(tmp_path / "other", protocol_hash="ph-2", name="final2")
    got2 = access(tmp_path / "other" / "seal", protocol_hash="ph-1",
                  candidate_id="c1", caller="m5_final_confirm",
                  purpose="final_confirm")
    assert got2["allowed"] is False
    assert any("protocol mismatch" in r for r in got2["reasons"]), got2


def test_a_new_set_can_be_sealed_after_the_old_one_is_consumed(tmp_path):
    _seal(tmp_path)
    assert access(tmp_path / "seal", protocol_hash="ph-1", candidate_id="c1",
                  caller="m5_final_confirm",
                  purpose="final_confirm")["allowed"] is True
    # 新一轮：另冻结未使用场景（不同目录）
    new_frames = []
    d = tmp_path / "final2" / "front_main"
    d.mkdir(parents=True)
    for i in range(2):
        p = d / f"frame_{i:05d}.npz"
        p.write_bytes(b"new-%d" % i)
        new_frames.append(p)
    rec = seal(new_frames, name="final-round2", dataset_id="ds-2",
               protocol_hash="ph-1", out_dir=tmp_path / "seal2",
               groups=["italy/ring_g"])
    assert rec["n_frames"] == 2
    assert consumption(tmp_path / "seal2")["consumed"] is False
    assert access(tmp_path / "seal2", protocol_hash="ph-1",
                  candidate_id="cand-B", caller="m5_final_confirm",
                  purpose="final_confirm")["allowed"] is True


def test_a_confirmation_record_is_bound_to_protocol_candidate_and_weights():
    """确认记录必须绑定协议/候选/权重（方案 §7、A8）。

    实测缺口：原来只有账本（"谁在什么时候读了一次"），于是"换一个独立确认协议
    再宣称旧判定仍然成立"无从检查。这里把三种对不上都钉住。
    """
    rec = {"kind": "final_confirmation", "protocol_hash": "ph-2",
           "candidate_id": "cand-A", "model_sha16": "abc123",
           "seal_digest": "seal-9"}
    ok = confirmation_check(rec, protocol_hash="ph-2", candidate_id="cand-A",
                            model_sha16="abc123", seal_digest="seal-9")
    assert ok["status"] == "verified" and not ok["issues"], ok
    # 换了协议：旧确认不能复用
    bad = confirmation_check(rec, protocol_hash="ph-3", candidate_id="cand-A",
                             model_sha16="abc123", seal_digest="seal-9")
    assert bad["status"] == "mismatch" and "another protocol" in bad["issues"][0]
    # 换了权重：确认的不是这个 checkpoint
    w = confirmation_check(rec, protocol_hash="ph-2", candidate_id="cand-A",
                           model_sha16="other")
    assert w["status"] == "mismatch" and "weights" in w["issues"][0]
    # 换了候选 / 换了最终集
    c = confirmation_check(rec, protocol_hash="ph-2", candidate_id="cand-B")
    assert c["status"] == "mismatch" and "candidate" in c["issues"][0]
    h = confirmation_check(rec, protocol_hash="ph-2", candidate_id="cand-A",
                           seal_digest="seal-other")
    assert h["status"] == "mismatch" and "digest" in h["issues"][0]
    # 没有记录 = R2 未确认，**不是**"记录对不上"（不推翻研究结论）
    none = confirmation_check(None, protocol_hash="ph-2", candidate_id="cand-A")
    assert none["status"] == "missing" and none["issues"] == []


def test_confirmation_record_records_what_was_measured(tmp_path):
    """记录要能看出"测了哪些口径"（方案 §10.1：没测的记 UNKNOWN）。"""
    _seal(tmp_path)
    rec = confirmation_record(seal_dir=tmp_path / "seal", protocol_hash="ph-1",
                              candidate_id="cand-A", caller="test",
                              model_sha16="abc", results={"overall":
                                                          {"line_recall": 0.5}},
                              notes=["identity probe not run"])
    assert rec["seal_digest"] and rec["protocol_hash"] == "ph-1"
    assert rec["results"]["overall"]["line_recall"] == 0.5
    assert rec["notes"] == ["identity probe not run"]
