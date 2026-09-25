"""标注凭证与"以凭证为准"的来源资格（方案 §6.1、G03、验收 A1/A8）。

实测缺口：来源资格原来由**命令行字符串**决定（"以 _partial 结尾或等于 pseudo"
才算研究臂），于是 `--paint-source X=agent_revision` 不带 `--research-arm` 就能绕过
晋级限制；反过来写 `human_revision` 也能把 agent 数据升格。

本文件钉三件事：

1. 凭证读取：帧所在目录自己声明的来历优先（self → parent 回退，与 manifest 的
   meta 查找同序）；读不到就是 ``None``，不猜；
2. 规则：**凭证优先**，命令行只能降低不能抬高；无凭证时声明 verified 一律不认；
3. 资格：agent/pseudo 不能作为晋级参考，混用与未知 rank 按最保守处理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.credentials import (  # noqa: E402
    CREDENTIAL_FILES, read_dir_credentials,
)
from beamng_autopilot.experiments.protocol import (  # noqa: E402
    effective_source, sources_can_promote,
)


def _view_dir(tmp_path: Path, *, credential=None, parent_credential=None,
              frames=3) -> Path:
    d = tmp_path / "coll" / "front_main"
    d.mkdir(parents=True)
    for i in range(frames):
        np.savez_compressed(d / f"frame_{i:05d}.npz",
                            colour=np.zeros((4, 4, 3), np.uint8),
                            label=np.zeros((4, 4), np.uint8))
    if parent_credential is not None:
        (d.parent / "annotation.json").write_text(json.dumps(
            parent_credential, ensure_ascii=False), encoding="utf-8")
    if credential is not None:
        (d / "annotation.json").write_text(json.dumps(
            credential, ensure_ascii=False), encoding="utf-8")
    return d


def test_the_credential_lives_with_the_frames(tmp_path):
    d = _view_dir(tmp_path, credential={"label_source": "agent_revision",
                                        "frames": [{"path": "a"}, {"path": "b"}]})
    c = read_dir_credentials(d)
    assert c["label_source"] == "agent_revision" and c["where"] == "self"
    assert c["frames"] == 2
    assert Path(c["path"]) == d / "annotation.json"


def test_the_parent_credential_is_used_as_a_fallback(tmp_path):
    """帧目录没有凭证时看父目录（与 manifest 的 meta 查找同序），并标出来源。"""
    d = _view_dir(tmp_path, parent_credential={
        "label_source": "agent_revision", "frames": []})
    c = read_dir_credentials(d)
    assert c["label_source"] == "agent_revision" and c["where"] == "parent"
    # 子目录自己那份优先
    d2 = _view_dir(tmp_path / "other", credential={
        "label_source": "human_revision", "frames": []},
        parent_credential={"label_source": "agent_revision", "frames": []})
    c2 = read_dir_credentials(d2)
    assert c2["label_source"] == "human_revision" and c2["where"] == "self"


def test_unreadable_or_missing_credentials_are_not_guessed(tmp_path):
    d = _view_dir(tmp_path)
    assert read_dir_credentials(d) is None, "没有凭证文件 -> None，不猜"
    assert read_dir_credentials(tmp_path / "nope") is None
    bad = _view_dir(tmp_path / "bad", credential=None)
    (bad / "annotation.json").write_text("{not json", encoding="utf-8")
    c = read_dir_credentials(bad)
    assert c["readable"] is False and c["label_source"] == ""
    nolabel = _view_dir(tmp_path / "nolabel",
                        credential={"frames": [{"path": "a"}]})
    c2 = read_dir_credentials(nolabel)
    assert c2["readable"] is True and c2["label_source"] == ""
    assert "does not declare label_source" in c2["why"]
    assert CREDENTIAL_FILES[0] == "annotation.json"


def test_the_credential_wins_and_the_command_line_cannot_upgrade():
    # 没有凭证：声明 verified 不认（降级），其余照声明
    src, notes = effective_source("human_revision", None)
    assert src == "engine_annotation" and notes, notes
    assert effective_source("agent_revision", None) == ("agent_revision", [])
    # 有凭证：凭证优先，两边都要有说明
    src2, notes2 = effective_source("human_revision", "agent_revision")
    assert src2 == "agent_revision" and "using the credential" in notes2[0]
    src3, notes3 = effective_source("agent_revision", "human_revision")
    assert src3 == "human_revision" and notes3, "凭证说人工复核过，就该按 verified"
    # 凭证文件存在但没声明来源：同样不能靠命令行升格
    src4, notes4 = effective_source("human_revision", "")
    assert src4 == "engine_annotation" and notes4


def test_only_verified_can_be_a_promotion_reference():
    assert sources_can_promote(["human_revision"])[0] is True
    for s in ("agent_revision", "engine_annotation_partial", "pseudo",
              "engine_annotation"):
        ok, why = sources_can_promote([s])
        assert ok is False and why, s
    ok, why = sources_can_promote(["human_revision", "agent_revision"])
    assert ok is False and "agent" in why[0], why
