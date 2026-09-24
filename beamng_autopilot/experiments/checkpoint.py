"""完整 checkpoint 与"中断续训 ≈ 未中断"的可验证实现。

方案 §4 的要求：`checkpoint_last.pt` 负责恢复，必须保存**恢复所需的全部
随机状态与数据版本**，并先验证等价性再谈可复现。现有训练入口的 checkpoint
只有 state_dict/optimizer/scheduler/scaler/epoch，缺随机状态与数据版本，本
模块补齐这两块（训练入口按同一键名写入，旧 checkpoint 仍可读）。
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

RNG_KEYS = ("torch_rng", "torch_cuda_rng", "numpy_rng", "python_rng")


def _numpy_state_plain() -> dict:
    """numpy 全局 RNG 状态的**纯数字**表示。

    ``np.random.get_state()`` 返回的元组里含 ndarray；在 ``torch.load(
    weights_only=True)``（torch 2.6+ 默认，部署链就是这么读的）下 numpy 全局不在
    白名单里，**含它的 checkpoint 会加载失败**（实测）。状态本身就是整数数组，
    拆成 list 后既小又能安全加载。
    """
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {"kind": str(kind), "keys": [int(k) for k in keys],
            "pos": int(pos), "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached)}


def _numpy_state_from(state) -> tuple:
    """把自己写的 dict 或 numpy 原生的元组还原成 ``set_state`` 能吃的形式。"""
    if isinstance(state, dict):
        return (state.get("kind", "MT19937"),
                np.array(state["keys"], dtype=np.uint32),
                int(state.get("pos", 0)), int(state.get("has_gauss", 0)),
                float(state.get("cached_gaussian", 0.0)))
    return state                            # 旧 checkpoint：原生元组


def rng_state() -> dict:
    """当前进程的全部随机状态（CPU/CUDA/numpy/python）。"""
    st = {"torch_rng": torch.get_rng_state(),
          "numpy_rng": _numpy_state_plain(),
          "python_rng": random.getstate()}
    if torch.cuda.is_available():
        st["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return st


def _as_cpu_byte_tensor(v):
    """把落盘后的 RNG 状态还原成 ``torch.set_rng_state`` 能吃的 CPU 张量。

    实测坑：训练入口用 ``torch.load(..., map_location=device)`` 读 checkpoint，
    会把 RNG 状态一起搬到 CUDA，而 ``set_rng_state`` 只接受 CPU 的 ByteTensor
    —— 于是续训在第一步就抛 ``TypeError: RNG state must be a torch.ByteTensor``
    （B2 验收因此第一次判 FAIL）。这里统一：张量先落 CPU，list/bytes 先转张量。
    """
    if torch.is_tensor(v):
        return v.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(v, (bytes, bytearray)):
        return torch.tensor(list(v), dtype=torch.uint8)
    return torch.tensor(list(v), dtype=torch.uint8)


def restore_rng(state: dict | None) -> bool:
    """恢复随机状态；缺项返回 False（而不是假装恢复了）。"""
    if not state:
        return False
    ok = True
    if state.get("torch_rng") is not None:
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch_rng"]))
    else:
        ok = False
    if state.get("numpy_rng") is not None:
        np.random.set_state(_numpy_state_from(state["numpy_rng"]))
    else:
        ok = False
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    else:
        ok = False
    if state.get("torch_cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_as_cpu_byte_tensor(s) for s in state["torch_cuda_rng"]])
    return ok


def env_fingerprint() -> dict:
    """环境指纹。**全部字段必须是纯字符串/数字**。

    实测缺陷：``torch.__version__`` 在新版 torch 里是 ``TorchVersion``（str 的
    子类）对象，pickle 会把它当全局类存下来；torch 2.6+ 的 ``torch.load`` 默认
    ``weights_only=True`` 于是拒绝加载——**新写出的 checkpoint 部署链读不了**。
    修在数据侧（存纯字符串），而不是去放宽加载器的安全默认。
    """
    return {"torch": str(torch.__version__),
            "cuda": (None if torch.version.cuda is None
                     else str(torch.version.cuda)),
            "cudnn": (int(torch.backends.cudnn.version())
                      if torch.backends.cudnn.is_available() else None),
            "device": str(torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu"),
            "python": _py_version()}


def _py_version() -> str:
    import sys
    return ".".join(str(v) for v in sys.version_info[:3])


def git_commit(cwd: Path | str | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd or "."),
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:                        # noqa: BLE001
        return ""


def checkpoint_extras(*, dataset_id: str, candidate_id: str = "",
                      run_id: str = "", cwd: Path | str | None = None
                      ) -> dict:
    """要并入 checkpoint 的收尾字段（数据版本、随机状态、环境、commit）。"""
    return {"dataset_id": str(dataset_id),
            "candidate_id": str(candidate_id),
            "run_id": str(run_id),
            "git_commit": git_commit(cwd),
            "env": env_fingerprint(),
            **rng_state()}


def missing_extras(ckpt: dict) -> list[str]:
    """恢复所需的字段里缺了哪些（旧 checkpoint 的体检结果）。"""
    need = ["state_dict", "optimizer", "scheduler", "next_epoch",
            "dataset_id", *RNG_KEYS]
    return [k for k in need if k not in ckpt or ckpt[k] is None]


def weights_equal(a: dict, b: dict, *, atol: float = 0.0) -> dict:
    """两个 state_dict 的逐张量比较：返回最大绝对差与不等的键。

    ``atol=0`` 表示要求**逐位相同**——"中断续训 ≈ 未中断"的默认判据。
    """
    keys = sorted(set(a) & set(b))
    worst, worst_key, diff_keys = 0.0, "", []
    for k in keys:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            diff_keys.append(k)
            continue
        d = float((ta.double() - tb.double()).abs().max().item())
        if d > atol:
            diff_keys.append(k)
        if d > worst:
            worst, worst_key = d, k
    return {"max_abs_diff": worst, "max_key": worst_key,
            "n_diff": len(diff_keys), "diff_keys": diff_keys[:8],
            "n_compared": len(keys),
            "only_in_a": sorted(set(a) - set(b))[:5],
            "only_in_b": sorted(set(b) - set(a))[:5]}


def load_full(path: Path | str) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def save_json(path: Path | str, blob: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, indent=1, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p


def file_sha16(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
