"""标注凭证：数据目录自己声明的来历（方案 §6.1、验收 A1/A8）。

为什么需要它：来源资格不能由**命令行字符串**决定。实测缺口（G03）：`rounds` 用
"字符串以 `_partial` 结尾或等于 `pseudo`"来判断研究臂，于是
`--paint-source X=agent_revision` **不带** `--research-arm` 就能绕过晋级限制；
反过来，在命令行写 `human_revision` 也能把 agent 数据"升格"。

本模块只做一件事：**读目录自己的凭证**（标注器/工具写下的 sidecar），
让判定去比对"声明"与"凭证"。规则本身（谁可以晋级）在
``experiments/protocol.py`` 的纯函数里。

凭证的形态（两种都存在，取先命中的）：

* ``annotation.json``：本轮 agent 核对式标注工具写的（``label_source`` + 逐帧记录）；
* ``meta.json``：人工标注器的 sidecar（``label_source: human_revision`` +
  ``annotation`` 块 + 逐帧记录）。

读不到凭证返回 ``None``（= 无凭证），由调用方按"不得升格"处理，**不猜**。

自动真值凭证（T16 冻结契约 §3.4，2026-09-27 增量）：sidecar 里可能有
``truth_contract: "v1"`` + ``truth_provenance``（生成器/资产/运行/相机/标签/
校验报告）。本模块额外暴露两个**只读**字段，不改变已有键与语义：

* ``truth_provenance``：dict 或 ``None``（旧 sidecar 没有 → None）；
* ``truth_verified``：严格口径 = ``truth_contract=="v1"`` 且
  ``report.verified is True`` 且 ``report.verifier_version`` 非空且
  ``generator.sha``/``labels.label_sha`` 非空。

为什么严格：仅改 sidecar 字符串（只写 ``label_source: engine_verified`` 或只写
一个空的 provenance）不得把旧的不完整标签提升为已验证来源。
"""

from __future__ import annotations

import json
from pathlib import Path

#: 凭证文件名（按优先级）
CREDENTIAL_FILES = ("annotation.json", "meta.json")

#: 自动真值凭证契约版本（``experiments/auto_truth.py`` 的写入方必须一致）。
TRUTH_CONTRACT = "v1"


def _truth_provenance(blob) -> dict | None:
    prov = blob.get("truth_provenance") if isinstance(blob, dict) else None
    return prov if isinstance(prov, dict) else None


def truth_verified(blob) -> bool:
    """严格判断这份 sidecar 是否可当"引擎已验证"来源（见模块 docstring）。

    任何一环缺失/类型不对都返回 ``False``（不猜、不把 UNKNOWN 当通过）：
    契约不是 v1、没有 ``truth_provenance``、``report.verified`` 不是 True、
    ``report.verifier_version`` 为空、``generator.sha`` 为空、
    ``labels.label_sha`` 为空。
    """
    if not isinstance(blob, dict):
        return False
    if str(blob.get("truth_contract") or "") != TRUTH_CONTRACT:
        return False
    prov = _truth_provenance(blob)
    if prov is None:
        return False
    report = prov.get("report")
    if not isinstance(report, dict) or report.get("verified") is not True:
        return False
    if not str(report.get("verifier_version") or "").strip():
        return False
    generator = prov.get("generator")
    if not isinstance(generator, dict) or not str(generator.get("sha") or "").strip():
        return False
    labels = prov.get("labels")
    if not isinstance(labels, dict) or not str(labels.get("label_sha") or "").strip():
        return False
    return True


def _credential_record(blob: dict, path: Path, where: str, n_frames: int) -> dict:
    """把 sidecar 变成一个统一记录；``truth_*`` 字段对每条返回路径都带上。"""
    return {"label_source": str(blob.get("label_source") or ""), "path": str(path),
            "where": where, "readable": True, "frames": n_frames,
            "truth_provenance": _truth_provenance(blob),
            "truth_verified": truth_verified(blob)}


def read_dir_credentials(d: str | Path) -> dict | None:
    """``None`` = 没有可用凭证；否则返回 ``{label_source, path, frames, ...}``。

    ``frames`` 是逐帧记录的条数（0 表示没有逐帧明细——凭证不完整，调用方要看清）。
    另含 ``truth_provenance``（dict|None）与 ``truth_verified``（bool）：旧 sidecar
    没有这些字段时分别是 ``None`` / ``False``，已有键语义不变。
    """
    d = Path(d)
    if not d.is_dir():
        return None
    cands = [(d / name, "self") for name in CREDENTIAL_FILES]
    cands += [(d.parent / name, "parent") for name in CREDENTIAL_FILES]
    for p, where in cands:
        if not p.is_file():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            return {"label_source": "", "path": str(p), "where": where,
                    "readable": False, "frames": 0,
                    "truth_provenance": None, "truth_verified": False,
                    "why": "credential file is not parseable"}
        if not isinstance(blob, dict):
            continue
        src = str(blob.get("label_source") or "")
        frames = blob.get("frames")
        n_frames = len(frames) if isinstance(frames, list) else 0
        if not src:
            # 有 sidecar 但没写来源：算"有文件、无来源声明"
            rec = _credential_record(blob, p, where, n_frames)
            rec["why"] = "credential file does not declare label_source"
            return rec
        rec = _credential_record(blob, p, where, n_frames)
        rec.update({
            "annotation": blob.get("annotation") or {},
            "reviewer": (blob.get("annotation") or {}).get("reviewer")
            if isinstance(blob.get("annotation"), dict) else None,
            "reviewed_at": (blob.get("annotation") or {}).get("reviewed_at")
            if isinstance(blob.get("annotation"), dict) else None})
        return rec
    return None

