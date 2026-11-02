"""logs/ 权重保留策略：删训练中间权重、留证据，回收磁盘。

背景（2026-10-20 实测）：I: 盘 free 16.2 GB，低于实验资源门
``min_free_disk_gb = 20 GB``（``beamng_autopilot/experiments/collection.py``），
于是所有 Tech 采集在前置检查就被拒；而 ``logs/experiments`` 下 60+ 个实验
共约 33 GB 的 ``*.pt``，其中绝大多数是**已出报告、结论为负、未部署**的
逐种子 checkpoint。

这里的取舍是「证据 >> 权重」：报告、JSON、日志、PNG 一律保留；可复现性靠
报告里记录的 sha16 与协议，不靠把 9.7 MB/个的优化器状态永远留在盘上。

**与既有 ``m5_prune_run_artifacts.py`` 的分工**（先看那个，别重复造）：

* 那个是**单 run 目录**的逐 epoch 清理（只删 ``epoch_*.pt``，保留
  ``checkpoint_last.pt`` / ``best.pt``），语义是"训练过程中腾地方”；
* 这个是**跨实验的权重保留**（删已出报告的负结果臂的逐种子 ``*.pt``），
  语义是"报告写完之后，把不再需要复算的权重回收"，并带保留清单与封存集保护。
  两者互不替代；同一个目录先跑那个、再跑这个。

规则（不可协商）：

* 只删权重后缀（``.pt`` / ``.pth`` / ``.ckpt``），其它文件一律不碰；
* 部署权重（``logs/m5_seg/seg_model*/``）与 ``best*`` 权重默认永不删；
* **封存最终集目录**（含 ``final_set_seal.json`` 的整棵树）一律不碰；
* **文档点名当 ``--init`` / 交付候选 / 确认集权重的目录**按
  ``DEFAULT_KEEP_GLOBS`` 保留，可用 ``logs/retention_keep.txt`` 或
  ``--keep`` 追加——删掉它们，预注册协议就无法原样重跑；
* 默认 dry-run，必须 ``--apply`` 才真删；
* 删除前把每个文件的 path / size / mtime 写进 manifest（可审计、可追责）；
* 每个待删路径必须 realpath 落在 logs 目录之内，否则整次拒绝执行。

用法::

    pwsh> .venv\\Scripts\\python.exe scripts\\m5_logs_retention.py
    pwsh> .venv\\Scripts\\python.exe scripts\\m5_logs_retention.py --apply `
            --min-free-gb 30
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402

#: 只认这些后缀；其它文件（报告、帧、日志）永远不在删除范围。
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

#: 默认扫描的顶层目录（相对 logs 根）。``m5_seg`` 里既有部署权重又有采集帧，
#: 所以默认不整体扫它——部署权重由 ``PROTECTED_DIR_PREFIXES`` 兜底。
#: ``m3_bc`` / ``m4_dqn`` 是**被 README 点名的模型文件**（``bc_tech_smallgrid.pt``
#: 等），体积只有几十 MB，收益远小于「删掉后就地不一定能重建」的风险，默认不扫。
DEFAULT_ROOTS = ("experiments", "m5_e2e", "m5_rl")

#: 目录名以这些前缀开头的，任何权重都受保护（部署 / 交付候选）。
PROTECTED_DIR_PREFIXES = ("seg_model",)

#: 文件名的保护前缀（``best.pt`` / ``best_task.pt`` / ``best_pre_missfix2.pt``）。
PROTECTED_NAME_PREFIXES = ("best",)

#: 默认保留的路径模式（相对 logs，``*`` 也匹配分隔符，写法按 fnmatch）。
#: 每一条都是**文档点名**、被后续协议当 ``--init``/交付候选/确认集权重用的
#: 产物——删了它们，预注册协议就无法原样重跑：
#:   * ``t14_e1_promotable_20260927``：T16 跨池/剂量全套的共享初始化；
#:   * ``t16_negdose6x_20261001``：唯一在封存最终集上确认过的研究臂；
#:   * ``t14_e0_20260925``：身份标定与候选门校准点名的候选；
#:   * ``seg_model*``：部署/交付目录；
#:   * ``best*``：脚本按开发集选出的权重，体积小、语义独立；
#:   * ``t16_scenes_*`` / ``collect_e5line_*`` / ``t16_*_export_*``：
#:     engine_verified 采集批次与导出池（2026-11-01 外部删除事件后点名——
#:     历史池已不可再生，见 T16_LINE_MEASURED_UTAH2_RESULT_20261101.md）；
#:   * ``evidence_archives``：哈希清单证据档。
DEFAULT_KEEP_GLOBS = (
    "**/seg_model*/**",
    "**/best*.pt",
    "**/t14_e1_promotable_*/**",
    "**/t16_negdose6x_20261001/**",
    "**/t14_e0_20260925/**",
    "**/t16_scenes_*/**",
    "**/collect_e5line_*/**",
    "**/t16_*_export_*/**",
    "**/evidence_archives/**",
)

#: 可选的额外保留清单文件（相对 logs 根），一行一个 glob，``#`` 注释。
KEEP_FILE = "retention_keep.txt"

#: 资源门阈值，仅用于报告与提示。
RESOURCE_GATE_GB = 20.0


class RetentionError(RuntimeError):
    """策略违规（路径越界等）。**整批拒绝执行**，不静默跳过。"""


def free_gb(path: Path) -> float:
    """所在卷的空闲空间（GB）。取不到就让异常冒出去——不编数。"""
    return shutil.disk_usage(str(path)).free / 2 ** 30


def resolve_within(path: Path, root: Path) -> Path:
    """``path`` 的 realpath 必须落在 ``root`` 之内，否则抛错。

    删东西之前必须证明目标就是自己以为的那个（符号链接、``..``、盘符
    大小写都算）。只用 realpath 比较，不做任何猜测。
    """
    real = Path(path).resolve()
    base = Path(root).resolve()
    try:
        real.relative_to(base)
    except ValueError as exc:
        raise RetentionError(
            f"拒绝：{real} 不在 {base} 之内（越界路径不允许删除）") from exc
    return real


def load_keep_globs(logs_dir: Path, extra: list[str] | None = None) -> tuple[str, ...]:
    """默认保留模式 + ``logs/retention_keep.txt`` + 命令行 ``--keep``。"""
    globs = list(DEFAULT_KEEP_GLOBS)
    keep_file = Path(logs_dir) / KEEP_FILE
    if keep_file.is_file():
        for line in keep_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                globs.append(line)
    if extra:
        globs.extend(str(g) for g in extra if str(g).strip())
    return tuple(globs)


def is_protected(path: Path, logs_dir: Path,
                 keep_globs: tuple[str, ...] = DEFAULT_KEEP_GLOBS) -> str | None:
    """受保护则返回原因字符串，否则 None。

    ``keep_globs`` 按 ``relpath`` 的通配匹配；``*`` 也跨分隔符匹配，
    所以 ``**/foo/**`` 等价于「路径里出现 foo 目录」。
    """
    for part in path.parts:
        if part.lower().startswith(PROTECTED_DIR_PREFIXES):
            return f"部署/交付权重目录：{part}"
    name = path.name.lower()
    for pref in PROTECTED_NAME_PREFIXES:
        if name.startswith(pref):
            return f"best 权重默认保留：{path.name}"
    try:
        rel = Path(path).resolve().relative_to(Path(logs_dir).resolve())
    except ValueError:
        rel = Path(path)
    rel_posix = rel.as_posix()
    for pat in keep_globs:
        if fnmatch.fnmatch(rel_posix, pat):
            return f"保留清单命中：{pat}"
    return None


def is_weight(path: Path) -> bool:
    return path.suffix.lower() in WEIGHT_SUFFIXES


#: 封存最终集的标记文件；出现了它，整个目录树都不是可清理的运行产物。
SEAL_NAME = "final_set_seal.json"


def _sealed_ancestor(path: Path, base: Path, cache: dict) -> Path | None:
    """返回包住 ``path`` 的封存子树根（含 ``final_set_seal.json``），没有则 None。

    安全底线与 ``scripts/m5_prune_run_artifacts.py`` 一致：最终集不是运行产物，
    哪怕它旁边的权重再大也不动。

    实测两种布局都要覆盖：封存标记可能在**近邻子目录** ``<run>/seal/``，
    也可能在更深的 ``<run>/<phase>/seal/``；所以从待删文件往上走到扫描根，
    逐级看「本级有标记」或「本级有 seal/ 标记」，命中即整棵子树受保护。
    结果按目录缓存，几千个候选也只 stat 几百次。
    """
    parent = path.parent
    while True:
        if parent not in cache:
            cache[parent] = ((parent / SEAL_NAME).is_file()
                             or (parent / "seal" / SEAL_NAME).is_file())
        if cache[parent]:
            return parent
        if parent == base or parent == parent.parent:
            return None
        parent = parent.parent


def scan(logs_dir: Path, roots: tuple[str, ...] = DEFAULT_ROOTS,
         *, older_than_days: float | None = None,
         now: float | None = None,
         keep_globs: tuple[str, ...] = DEFAULT_KEEP_GLOBS,
         ) -> tuple[list[dict], list[dict]]:
    """返回 ``(candidates, protected)``；每项含 path/size/mtime。

    只读：不删任何东西。``older_than_days`` 按 mtime 过滤候选。
    """
    now = time.time() if now is None else float(now)
    candidates: list[dict] = []
    protected: list[dict] = []
    seal_cache: dict = {}
    for name in roots:
        base = logs_dir / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or not is_weight(path):
                continue
            resolve_within(path, logs_dir)
            stat = path.stat()
            item = {"path": str(path), "size": stat.st_size,
                    "mtime": stat.st_mtime}
            sealed = _sealed_ancestor(path, base, seal_cache)
            reason = (f"封存最终集目录（{SEAL_NAME}）：{sealed}" if sealed
                      else is_protected(path, logs_dir, keep_globs))
            if reason:
                protected.append({**item, "reason": reason})
                continue
            if older_than_days is not None:
                age_days = (now - item["mtime"]) / 86400.0
                if age_days < float(older_than_days):
                    continue
            candidates.append(item)
    candidates.sort(key=lambda it: (-it["size"], it["path"]))
    return candidates, protected


def apply_plan(logs_dir: Path, candidates: list[dict], *,
               min_free_gb: float, manifest_path: Path,
               dry_run: bool = True) -> dict:
    """按计划删除；manifest 先落盘。``dry_run`` 时一个字节都不删。"""
    logs_dir = Path(logs_dir).resolve()
    manifest_path = Path(manifest_path)
    before = free_gb(logs_dir)
    #: 目标 <= 0 表示「按计划删完」，不是「目标是 0 GB」——否则默认值会
    #: 在第一个候选上就判定达标，整个 --apply 变成空操作。
    target = float(min_free_gb) if float(min_free_gb) > 0 else None
    deleted: list[dict] = []
    skipped: list[dict] = []
    freed = 0
    for item in candidates:
        path = Path(item["path"])
        if dry_run:
            deleted.append(item)
            freed += int(item["size"])
            continue
        if target is not None and before + freed / 2 ** 30 >= target:
            skipped.append({**item, "reason": f"已达 {target:.1f} GB 目标"})
            continue
        real = resolve_within(path, logs_dir)
        try:
            size = real.stat().st_size
            real.unlink()
        except FileNotFoundError:
            skipped.append({**item, "reason": "已不存在"})
            continue
        except OSError as exc:                  # 占用/权限：如实记，不吞
            skipped.append({**item, "reason": f"unlink 失败：{exc}"})
            continue
        deleted.append({**item, "size": size})
        freed += int(size)
    after = free_gb(logs_dir)
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "logs_dir": str(logs_dir),
        "dry_run": bool(dry_run),
        "policy": {"weight_suffixes": sorted(WEIGHT_SUFFIXES),
                   "protected_dir_prefixes": list(PROTECTED_DIR_PREFIXES),
                   "protected_name_prefixes": list(PROTECTED_NAME_PREFIXES),
                   "min_free_gb": target},
        "free_before_gb": round(before, 2),
        "free_after_gb": round(after, 2),
        "planned_bytes": sum(int(it["size"]) for it in candidates),
        "freed_bytes": freed,
        "deleted": deleted,
        "skipped": skipped,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="logs 权重保留策略（默认 dry-run）")
    ap.add_argument("--logs-dir", default=str(config.LOGS_DIR),
                    help="logs 根目录（默认 config.LOGS_DIR）")
    ap.add_argument("--root", action="append", default=None,
                    help=f"相对 logs 的扫描根，可重复（默认 {list(DEFAULT_ROOTS)}）")
    ap.add_argument("--older-than-days", type=float, default=None,
                    help="只删 mtime 早于 N 天的权重")
    ap.add_argument("--keep", action="append", default=None,
                    help="额外保留的 glob（相对 logs，可重复）；"
                         f"也可写进 logs/{KEEP_FILE}")
    ap.add_argument("--min-free-gb", type=float, default=0.0,
                    help="达到该空闲空间即停（默认 0=按计划删完）")
    ap.add_argument("--manifest", default=None,
                    help="manifest 路径（默认 logs/logs_retention_<ts>.json）")
    ap.add_argument("--apply", action="store_true",
                    help="真的删；不加就是 dry-run")
    args = ap.parse_args(argv)

    logs_dir = Path(args.logs_dir).resolve()
    if not logs_dir.is_dir():
        print(f"[retention] logs 目录不存在：{logs_dir}", file=sys.stderr)
        return 2
    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    keep_globs = load_keep_globs(logs_dir, args.keep)
    candidates, protected = scan(logs_dir, roots,
                                 older_than_days=args.older_than_days,
                                 keep_globs=keep_globs)
    planned = sum(int(it["size"]) for it in candidates)
    print(f"[retention] logs={logs_dir} free={free_gb(logs_dir):.2f} GB "
          f"(资源门 {RESOURCE_GATE_GB:.0f} GB)")
    print(f"[retention] 候选 {len(candidates)} 个权重 / "
          f"{planned / 2 ** 30:.2f} GB；受保护 {len(protected)} 个"
          f"（保留模式 {len(keep_globs)} 条）")
    for it in protected[:5]:
        print(f"[retention]   跳过 {it['path']}（{it['reason']}）")
    manifest = Path(args.manifest) if args.manifest else (
        logs_dir / f"logs_retention_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        summary = apply_plan(logs_dir, candidates,
                             min_free_gb=args.min_free_gb,
                             manifest_path=manifest,
                             dry_run=not args.apply)
    except RetentionError as exc:
        print(f"[retention] {exc}", file=sys.stderr)
        return 2
    verb = "将删除" if not args.apply else "已删除"
    print(f"[retention] {verb} {len(summary['deleted'])} 个 / "
          f"{summary['freed_bytes'] / 2 ** 30:.2f} GB；"
          f"free {summary['free_before_gb']:.2f} → "
          f"{summary['free_after_gb']:.2f} GB")
    if summary["skipped"]:
        print(f"[retention] 跳过 {len(summary['skipped'])} 个（见 manifest）")
    print(f"[retention] manifest -> {summary['manifest']}")
    if not args.apply:
        print("[retention] dry-run：加 --apply 才真的删")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
