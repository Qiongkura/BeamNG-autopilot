"""最终集封存与访问账本（方案 §7、§10.3）。

方案原文：「新的最终集单独封存，搜索器无读取权限；冻结候选后才允许一次确认。」
「最终集必须有访问与消费记录，绑定协议、候选、调用方和结果；只允许最终确认程序读取。
失败后不得对着同一最终集调参再宣称独立验证；旧集降为已消费诊断集，新一轮另冻结
未使用场景。」

本模块实现三件事：

* ``seal()``：把最终集**封存**——记逐帧内容哈希、组、dataset_id、协议哈希、封存人与
  时间；此后内容一变就与封存不符（最终集不可变）；
* ``access()``：登记一次访问并给出是否放行。规则（每条都能给出理由）：
  未封存 → 拒；purpose 不是 ``final_confirm`` → 拒（搜索器/训练不得读）；
  内容与封存不符 → 拒；协议哈希不符 → 拒；**已经确认过 → 拒**（一次确认即消费，
  失败也不例外——否则"失败后对同一集调参再宣称独立验证"就成立了）；
* ``consumption()``：消费账（确认次数、被拒次数、是否已消费、最后一条）。

账本是**只追加**的 JSONL，与判定文件一样属于证据，不手工改写。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

SEAL_NAME = "final_set_seal.json"
LEDGER_NAME = "final_set_ledger.jsonl"
#: 只允许这一个用途读最终集（方案："只允许最终确认程序读取"）
ALLOWED_PURPOSES = ("final_confirm",)


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def frame_digests(paths) -> dict:
    """``{相对/绝对路径: sha16}``；读不到的文件记 ``"UNREADABLE"``（不猜）。"""
    out: dict = {}
    for p in paths:
        p = Path(p)
        try:
            out[str(p)] = _sha16(p)
        except Exception:                                  # noqa: BLE001
            out[str(p)] = "UNREADABLE"
    return out


def _digest_of(digests: dict) -> str:
    blob = json.dumps(digests, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def seal(paths, *, name: str, dataset_id: str, protocol_hash: str,
         out_dir, sealed_by: str = "", groups=None) -> dict:
    """封存最终集；返回封存记录（也写进 ``<out_dir>/final_set_seal.json``）。"""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("最终集不能为空：至少一帧才能封存")
    digests = frame_digests(paths)
    unreadable = sorted(k for k, v in digests.items() if v == "UNREADABLE")
    if unreadable:
        raise ValueError(f"封存前必须能读到每一帧；读不到的：{unreadable[:3]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"name": str(name), "dataset_id": str(dataset_id),
           "protocol_hash": str(protocol_hash),
           "n_frames": len(paths), "groups": sorted(set(groups or [])),
           "digests": digests, "digest": _digest_of(digests),
           "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "sealed_by": str(sealed_by or "")}
    (out / SEAL_NAME).write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    return rec


def read_seal(out_dir) -> dict | None:
    p = Path(out_dir) / SEAL_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return None


def verify_seal(out_dir) -> dict:
    """封存内容是否**仍然**与记录一致（最终集不可变）。"""
    seal_rec = read_seal(out_dir)
    if seal_rec is None:
        return {"sealed": False, "ok": False, "reasons": ["not sealed"]}
    now = frame_digests(list(seal_rec.get("digests") or {}))
    changed = sorted(k for k, v in now.items()
                     if v != (seal_rec["digests"] or {}).get(k))
    missing = sorted(k for k, v in now.items() if v == "UNREADABLE")
    return {"sealed": True, "ok": not changed and not missing,
            "digest_matches": _digest_of(now) == seal_rec.get("digest"),
            "changed": changed[:10], "missing": missing[:10],
            "reasons": ([f"{len(changed)} frame(s) changed after sealing"]
                        if changed else [])
            + ([f"{len(missing)} frame(s) unreadable"] if missing else [])}


def confirmation_record(*, seal_dir, protocol_hash: str, candidate_id: str,
                        caller: str, model_sha16: str, dataset_id: str = "",
                        results: dict | None = None,
                        notes: list | None = None) -> dict:
    """一次最终确认的**结论记录**：绑定协议、候选、权重与结果（方案 §7）。

    为什么账本不够：账本记的是"谁在什么时候读了一次"，它回答不了判定要问的
    "这个候选是不是在**同一协议**下、用**这个权重**、在**这份最终集**上做过
    确认"。两者缺一，"换独立确认协议不覆盖旧判定"（A8）就无从检查。
    """
    seal_rec = read_seal(seal_dir) or {}
    return {
        "schema": 1,
        "kind": "final_confirmation",
        "seal_dir": str(seal_dir),
        "seal_name": seal_rec.get("name"),
        "seal_digest": seal_rec.get("digest"),
        "dataset_id": str(dataset_id or seal_rec.get("dataset_id") or ""),
        "protocol_hash": str(protocol_hash),
        "candidate_id": str(candidate_id),
        "caller": str(caller),
        "model_sha16": str(model_sha16),
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": dict(results or {}),
        "notes": list(notes or []),
    }


def confirmation_check(rec: dict | None, *, protocol_hash: str,
                       candidate_id: str, model_sha16: str | None = None,
                       seal_digest: str | None = None) -> dict:
    """判定/报告侧校验：确认记录与**当前**协议、候选、权重是否一致。

    返回 ``{"status": "missing"|"verified"|"mismatch", "issues": [...]}``。
    区别对待两种"没有确认"：**没有记录**只说明 R2 未确认（不推翻研究结论），
    而**记录对不上**是输入不一致（方案 §10.3 第一条 → invalid/rejected）：
    换了协议/换了权重/换了最终集，旧确认一律作废。
    """
    if not rec:
        return {"status": "missing", "issues": [],
                "why": "no final confirmation record: R2 (independent final-set "
                       "confirmation) is not established"}
    issues: list = []
    if str(rec.get("protocol_hash")) != str(protocol_hash):
        issues.append(
            f"confirmation was made under protocol {rec.get('protocol_hash')!r}, "
            f"current protocol is {protocol_hash!r} - a confirmation under "
            f"another protocol cannot be reused")
    if str(rec.get("candidate_id")) != str(candidate_id):
        issues.append(f"confirmation is for candidate {rec.get('candidate_id')!r}, "
                      f"not {candidate_id!r}")
    if model_sha16 is not None and str(rec.get("model_sha16")) != str(model_sha16):
        issues.append(
            f"confirmation used weights {rec.get('model_sha16')!r}, current "
            f"checkpoint is {model_sha16!r} - a confirmation of other weights "
            f"does not transfer")
    if seal_digest is not None and str(rec.get("seal_digest")) != str(seal_digest):
        issues.append(f"confirmation is bound to final-set digest "
                      f"{rec.get('seal_digest')!r}, current seal is {seal_digest!r}")
    return {"status": "mismatch" if issues else "verified", "issues": issues,
            "why": ("" if not issues else
                    "confirmation record does not match the current inputs")}


def ledger_path(out_dir) -> Path:
    return Path(out_dir) / LEDGER_NAME


def consumption(out_dir) -> dict:
    """消费账：确认次数、被拒次数、是否已消费。"""
    p = ledger_path(out_dir)
    entries = []
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    entries.append(json.loads(ln))
                except Exception:                          # noqa: BLE001
                    entries.append({"unreadable": True})
    confirmed = [e for e in entries if e.get("allowed")]
    refused = [e for e in entries if e.get("allowed") is False]
    return {"confirmed": len(confirmed), "refused": len(refused),
            "consumed": bool(confirmed), "entries": len(entries),
            "last": entries[-1] if entries else None}


def access(out_dir, *, protocol_hash: str, candidate_id: str, caller: str,
           purpose: str, result: str = "") -> dict:
    """登记一次访问并判定是否放行；无论放行与否都写进账本（只追加）。"""
    out = Path(out_dir)
    reasons: list = []
    seal_rec = read_seal(out)
    ver = verify_seal(out)
    cons = consumption(out)
    if seal_rec is None:
        reasons.append("not sealed: a final set must be sealed before it can be "
                       "read (an unsealed directory is not a final set)")
    else:
        if str(protocol_hash) != str(seal_rec.get("protocol_hash")):
            reasons.append(f"protocol mismatch: caller {protocol_hash!r} vs "
                           f"sealed {seal_rec.get('protocol_hash')!r}")
        if not ver["ok"]:
            reasons.append("seal content changed after sealing (immutability "
                           f"violated): {ver['reasons']}")
        if cons["consumed"]:
            reasons.append("this final set is already consumed: a consumed set "
                           "is a diagnostic set now - freeze a new one")
        if purpose not in ALLOWED_PURPOSES:
            reasons.append(f"purpose {purpose!r} is not allowed to read the "
                           f"final set (allowed: {list(ALLOWED_PURPOSES)})")
    entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "protocol_hash": str(protocol_hash),
             "candidate_id": str(candidate_id), "caller": str(caller),
             "purpose": str(purpose), "result": str(result),
             "allowed": not reasons, "reasons": reasons,
             "seal_digest": (seal_rec or {}).get("digest")}
    out.mkdir(parents=True, exist_ok=True)
    with open(ledger_path(out), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"allowed": not reasons, "reasons": reasons, "entry": entry,
            "ledger": str(ledger_path(out)), "consumption": consumption(out)}
