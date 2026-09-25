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
"""

from __future__ import annotations

import json
from pathlib import Path

#: 凭证文件名（按优先级）
CREDENTIAL_FILES = ("annotation.json", "meta.json")


def read_dir_credentials(d: str | Path) -> dict | None:
    """``None`` = 没有可用凭证；否则返回 ``{label_source, path, frames, ...}``。

    ``frames`` 是逐帧记录的条数（0 表示没有逐帧明细——凭证不完整，调用方要看清）。
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
                    "why": "credential file is not parseable"}
        if not isinstance(blob, dict):
            continue
        src = str(blob.get("label_source") or "")
        frames = blob.get("frames")
        n_frames = len(frames) if isinstance(frames, list) else 0
        if not src:
            # 有 sidecar 但没写来源：算"有文件、无来源声明"
            return {"label_source": "", "path": str(p), "where": where,
                    "readable": True, "frames": n_frames,
                    "why": "credential file does not declare label_source"}
        return {"label_source": src, "path": str(p), "where": where,
                "readable": True, "frames": n_frames,
                "annotation": blob.get("annotation") or {},
                "reviewer": (blob.get("annotation") or {}).get("reviewer")
                if isinstance(blob.get("annotation"), dict) else None,
                "reviewed_at": (blob.get("annotation") or {}).get("reviewed_at")
                if isinstance(blob.get("annotation"), dict) else None}
    return None
