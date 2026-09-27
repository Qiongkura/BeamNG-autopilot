"""运行产物的**保留策略**：只删中间产物，恢复/选择 checkpoint 与证据一律保留。

方案 v2 §S5 原文：「复用已有逐 epoch 清理，不让产物保留破坏磁盘保护；可进一步
按需求实现保留策略，**至少保存恢复 checkpoint、选择 checkpoint、评价证据及所有者
信息**」「不动历史验收产物；清理只作用本任务拥有、已验证绝对路径且处于允许目录的
中间文件」。

本脚本把这条做成显式、可核对的动作（默认 **dry-run**，要加 `--apply` 才删）：

* **保留**：`checkpoint_last.pt`（恢复）、`best.pt`（选择）、`decision_*.json`、
  评估矩阵/曲线/事件/日志（`*.json`/`*.jsonl`/`*.log`）、`meta.json`、
  `controller.lock`/`run_state.json`（所有者与租约）；
* **删除**：`epoch_*.pt`（逐 epoch 中间产物；4 小时耐久实测它们吃掉 ~25 GB，
  把磁盘压到资源门以下、后 9 轮全被拦）、以及 `--keep-epochs N` 之外的最新 N 个；
* **安全**：目标必须在 `logs/experiments/` 下（解析后的绝对路径），**含
  `final_set_seal.json` 的目录一律拒绝**（最终集不是可清理的运行产物），
  删除前列出清单，删完写 `prune_report_*.json`（含所有者/commit/保留与删除清单/
  释放字节）。

用法::

    # 先看会删什么（默认 dry-run）
    .venv\\Scripts\\python.exe scripts\\m5_prune_run_artifacts.py \\
        --run-dir logs/experiments/<run-id>
    # 确认后执行
    .venv\\Scripts\\python.exe scripts\\m5_prune_run_artifacts.py \\
        --run-dir logs/experiments/<run-id> --apply
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_ROOT = ROOT / "logs" / "experiments"
#: 中间产物模式：逐 epoch 权重（保留策略里唯一允许删的东西）
PRUNABLE_GLOBS = ("epoch_*.pt", "checkpoint_epoch*.pt")
#: 必须保留的文件名（恢复/选择/证据/所有者）
KEEP_NAMES = ("checkpoint_last.pt", "best.pt", "controller.lock",
              "run_state.json", "meta.json", "final_set_seal.json")
KEEP_SUFFIXES = (".json", ".jsonl", ".log")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def plan(run_dir: Path, *, keep_epochs: int = 0) -> dict:
    """算出保留/删除清单（**不删任何东西**）；run_dir 必须在允许目录下。"""
    rd = run_dir.resolve()
    allowed = ALLOWED_ROOT.resolve()
    out: dict = {"run_dir": str(rd), "allowed_root": str(allowed),
                 "keep": [], "prune": [], "refused": []}
    if allowed not in rd.parents and rd != allowed:
        out["refused"].append(
            f"{rd} 不在 {allowed} 之下：只清理本任务拥有的运行目录")
        return out
    if not rd.is_dir():
        out["refused"].append(f"{rd} 不是目录")
        return out
    if (rd / "final_set_seal.json").is_file():
        out["refused"].append(
            "目录里有 final_set_seal.json：最终集不是可清理的运行产物")
        return out
    prunable: list[Path] = []
    for pat in PRUNABLE_GLOBS:
        prunable += sorted(rd.rglob(pat))
    # 只保留最新的 keep_epochs 个（其余进删除清单）
    prunable.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set = set(prunable[:max(0, int(keep_epochs))])
    for p in sorted(rd.rglob("*")):
        if not p.is_file():
            continue
        if p in keep_set:
            out["keep"].append(str(p.relative_to(rd)))
            continue
        name_ok = p.name in KEEP_NAMES or p.suffix in KEEP_SUFFIXES
        if p in prunable:
            out["prune"].append({"path": str(p.relative_to(rd)),
                                 "bytes": p.stat().st_size})
        elif name_ok:
            out["keep"].append(str(p.relative_to(rd)))
        else:
            # 既不是已知证据类型、也不是已知中间产物：**不删**（保守），
            # 但要列出来让人看到（"没识别"不等于"可删"）
            out["keep"].append(str(p.relative_to(rd)))
    out["prune_bytes"] = sum(x["bytes"] for x in out["prune"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--keep-epochs", type=int, default=0,
                    help="保留最新的 N 个逐 epoch 权重（默认 0：全删，"
                         "恢复/选择 checkpoint 另有 KEEP 规则）")
    ap.add_argument("--apply", action="store_true",
                    help="真正删除（默认 dry-run）")
    args = ap.parse_args(argv)

    rep = plan(Path(args.run_dir), keep_epochs=args.keep_epochs)
    if rep["refused"]:
        for r in rep["refused"]:
            print(f"[prune] 拒绝：{r}")
        return 3
    freed = 0
    deleted: list = []
    if args.apply:
        for item in rep["prune"]:
            p = Path(rep["run_dir"]) / item["path"]
            try:
                p.unlink()
                freed += int(item["bytes"])
                deleted.append(item["path"])
            except OSError as exc:                        # noqa: BLE001
                print(f"[prune] 删除失败 {item['path']}: {exc}")
    print(f"[prune] {'已删除' if args.apply else '将删除（dry-run）'} "
          f"{len(rep['prune']) if not args.apply else len(deleted)} 个中间产物，"
          f"释放 {rep['prune_bytes'] / 1e9:.2f} GB；保留 {len(rep['keep'])} 个文件"
          f"（含 checkpoint_last/best/判定/日志/所有者）")
    if rep["prune"][:5]:
        for item in rep["prune"][:5]:
            print(f"[prune]   - {item['path']}")
        if len(rep["prune"]) > 5:
            print(f"[prune]   …共 {len(rep['prune'])} 个")
    if args.apply:
        rec = {"run_dir": rep["run_dir"], "applied_at": time.strftime(
                   "%Y-%m-%dT%H:%M:%S"),
               "git_commit": _git("rev-parse", "HEAD"),
               "git_dirty": [ln for ln in _git("status", "--porcelain")
                             .splitlines() if ln.strip()][:20],
               "deleted": deleted, "freed_bytes": freed,
               "kept_n": len(rep["keep"]),
               "kept_examples": rep["keep"][:20]}
        out = Path(rep["run_dir"]) / f"prune_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print(f"[prune] 记录 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
