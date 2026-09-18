"""一条命令跑完标注流程：导出待标注帧 → 打开标注器 → 打印后续训练命令。

用户侧的完整入口。它只负责编排，真正的标注器是 `m5_annotate_manual.py`
（画笔 / 油漆桶 / 撤销 / 保存），训练是 `m5_train_seg.py`。

    :: 直接跑（交互标注）
    .venv\\Scripts\\python.exe scripts\\m5_run_label_task.py

    :: 只看会执行什么、校验文件是否齐全（不开窗口）
    .venv\\Scripts\\python.exe scripts\\m5_run_label_task.py --check-only

    :: 指定 episode / 包名
    .venv\\Scripts\\python.exe scripts\\m5_run_label_task.py \\
        --episode logs/m5_e2e/shadow_fsd_...npz --name paved_shoulder

Windows 上也可以直接双击 `scripts\\run_label_task.cmd`。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config

PY = ROOT / ".venv" / "Scripts" / "python.exe"
RULES_HINT = (
    "标注规则见包内 RULES.txt：铺装=road(2)、土肩/砂石/草地=erase(3)、"
    "漆线=line(1)；整帧无铺装的土路本身=road(2)"
)


def _run(cmd: list[str], **kw) -> int:
    print("[label-task] " + " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=str(ROOT), **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 + 标注 + 训练提示 一条龙")
    ap.add_argument("--name", default="paved_shoulder",
                    help="任务名，包目录 logs/m5_seg/<name>_pkg")
    ap.add_argument("--episode", default="latest",
                    help="导出用的 shadow episode（包已存在时不使用）")
    ap.add_argument("--soil", type=int, default=18)
    ap.add_argument("--no-line", dest="noline", type=int, default=12)
    ap.add_argument("--plain", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12,
                    help="打印的训练命令用多少轮")
    ap.add_argument("--check-only", action="store_true",
                    help="只校验并打印命令，不打开标注窗口")
    args = ap.parse_args()

    pkg = config.LOGS_DIR / "m5_seg" / f"{args.name}_pkg"
    out = config.LOGS_DIR / "m5_seg" / f"{args.name}_labeled"
    model = config.LOGS_DIR / "m5_seg" / "seg_model" / "best.pt"
    if not PY.is_file():
        print(f"[label-task] 找不到虚拟环境解释器：{PY}")
        print("             先建环境并把依赖装上，再运行本脚本。")
        return 2

    # 1) 待标注帧：不存在就导出
    if pkg.is_dir() and any(pkg.glob("frame_*.npz")):
        n = len(list(pkg.glob("frame_*.npz")))
        print(f"[label-task] 已有标注包 {pkg}（{n} 帧），跳过导出")
    else:
        rc = _run([PY, ROOT / "scripts" / "m5_export_label_frames.py",
                   "--episode", args.episode, "--name", args.name,
                   "--soil", args.soil, "--no-line", args.noline,
                   "--plain", args.plain])
        if rc != 0:
            return rc

    # 2) 标注器（交互；--check-only 时只打印）
    print(f"[label-task] {RULES_HINT}")
    annotate = [PY, ROOT / "scripts" / "m5_annotate_manual.py",
                "--frames-dir", pkg, "--out", out,
                "--prefill-model", model]
    if args.check_only:
        print("[label-task] check-only，将执行：")
        print("  " + " ".join(str(c) for c in annotate))
    else:
        out.mkdir(parents=True, exist_ok=True)
        rc = _run(annotate)
        if rc != 0:
            print(f"[label-task] 标注器退出码 {rc}（窗口关闭即算完成）")

    # 3) 后续：训练 + 验证
    done = len(list(out.glob("frame_*.npz"))) if out.is_dir() else 0
    trained = config.LOGS_DIR / "m5_seg" / f"{args.name}_model"
    print(f"\n[label-task] 已保存标注 {done} 帧 -> {out}")
    print("[label-task] 训练（微调现网模型，不要从零训练；"
          "输出到独立目录，不覆盖现网 best.pt）：")
    print("  " + " ".join(str(c) for c in [
        PY, ROOT / "scripts" / "m5_train_seg.py", "--runs", out,
        "--init", model, "--out", trained,
        "--epochs", args.epochs, "--balance-runs"]))
    print("[label-task] 训练后先离线看土肩是否被剔除（不用开游戏）：")
    print("  " + " ".join(str(c) for c in [
        PY, ROOT / "scripts" / "m5_eval_seg.py", "--model",
        trained / "best.pt", "--runs", out, "--save"]))
    print("[label-task] 确认变好再决定是否 deploy 到 "
          f"{model}（现网模型，需 A/B 后才替换）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
