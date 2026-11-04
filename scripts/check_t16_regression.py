"""T16 诊断链回归套件（Phase 0 固化，2026-11-04）。

把本会话建立的契约按组跑一遍，任何一组失败即整体 FAIL：

  deploy       集成/部署（EnsembleSegmenter、权重、model3、drive 接线）
  post_norm    后处理分辨率归（默认 None、接口）
  diagnostics  掩码诊断/回放（GT 拆分、重采样、H-M0..M3、渲染门）
  conditions   条件矩阵（计划、fail-closed 聚合、断点续跑、自输出排除）
  scene        场景几何（right_lane、贴边余量、场景模式、no-goal）
  probe        探针（真值插值、collect 归因、臂环境变量、默认行为）
  drive        drive 入口接线（TOD/天气、post_norm live、save-frames、集成）

用法：
  .venv\\Scripts\\python.exe scripts/check_t16_regression.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TESTS = str(ROOT / "tests" / "test_controlled_scenes.py")

GROUPS = {
    "deploy": "ensemble or model3",
    "post_norm": "post_norm",
    "diagnostics": "mask_diag or marking_replay",
    "conditions": "conditions",
    "scene": "right_lane or edge_margin or scenario_mode",
    "probe": "pairing_probe or tta or pairing_arm",
    "drive": "fsd_drive_tod_weather or seg_post_norm_live or ensemble_drive "
             "or save_frames",
}


def run_group(expr: str) -> tuple[bool, str]:
    proc = subprocess.run([PY, "-m", "pytest", TESTS, "-k", expr, "-q"],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    tail = (proc.stdout or "").strip().splitlines()
    last = tail[-1] if tail else ""
    # pytest -q 的末行可能是进度行（"[100%]"）；退出码才是权威判据
    return proc.returncode == 0, last


def main() -> int:
    failures = 0
    print(f"{'group':12s} {'result':6s} summary")
    for name, expr in GROUPS.items():
        ok, summary = run_group(expr)
        failures += 0 if ok else 1
        print(f"{name:12s} {'PASS' if ok else 'FAIL':6s} {summary}")
    print(f"\nT16 regression: {'PASS' if failures == 0 else f'FAIL ({failures} groups)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
