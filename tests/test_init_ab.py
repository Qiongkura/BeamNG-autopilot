"""`m5_init_ab.py` 的纯逻辑测试（T16 §3.3：初始化对照）。

不训练：只验证
1. 两臂命令构造**除 `--init` 与 run id 外逐字相同**（初始化是唯一因子）；
2. 成对汇总按 seed 配对、缺 seed 不补 0、方向（越低越好）正确。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_init_ab", ROOT / "scripts" / "m5_init_ab.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_init_ab"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**over):
    base = dict(runs=["r1", "r2"], total_steps=120, batch=4, lr=1e-3,
                device="cuda", run_id="x", paint_source="human_revision",
                init="w.pt")
    base.update(over)
    return argparse.Namespace(**base)


def _strip(cmd):
    """去掉 per-arm 的 run id 后比较（run id 里带 arm 名是设计如此）。"""
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok == "--metrics-run":
            skip = True
            continue
        out.append(tok)
    return out


def test_two_arms_differ_only_by_init():
    m = _load()
    a = _args()
    rnd = m.train_cmd(a, "random", 42, "out")
    ini = m.train_cmd(a, "init", 42, "out")
    assert "--init" not in rnd
    assert "--init" in ini and ini[ini.index("--init") + 1] == "w.pt"
    # 去掉 per-arm run id 后，两臂的差异**只允许**是 --init <ckpt>
    assert _strip(ini) == _strip(rnd) + ["--init", "w.pt"],         "除 --init 外两臂命令必须逐字相同"
    # 预算/seed/数据在命令里且一致
    for flag in ("--total-steps", "--seed", "--batch", "--lr", "--runs"):
        assert flag in rnd and rnd[rnd.index(flag) + 1] == ini[ini.index(flag) + 1]


def test_no_init_means_random_arm_only():
    m = _load()
    cmd = m.train_cmd(_args(init=None), "init", 42, "out")
    assert "--init" not in cmd          # init=None 时即使 arm=init 也不加


def test_summarize_pair_pairs_by_seed_and_keeps_direction():
    m = _load()
    ps = {"random": {"42": {"line_iou": 0.26}, "43": {"line_iou": 0.28},
                     "44": {"line_iou": 0.27}},
          "init": {"42": {"line_iou": 0.38}, "43": {"line_iou": 0.40},
                   "44": {"line_iou": 0.39}}}
    s = m.summarize_pair(ps, "line_iou", lower_is_better=False)
    assert s["n"] == 3 and abs(s["delta"] - 0.12) < 1e-9
    assert s["compare"]["n"] == 3
    # 缺一个 seed -> 该 seed 不进配对，不补 0
    ps2 = {"random": {"42": {"line_iou": 0.26}, "43": {"line_iou": 0.28}},
           "init": {"42": {"line_iou": 0.38}, "43": {"line_iou": 0.40},
                    "44": {"line_iou": 0.39}}}
    s2 = m.summarize_pair(ps2, "line_iou", lower_is_better=False)
    assert s2["n"] == 2, s2
    # 越低越好的指标：方向要传对（这里只验证字段透传）
    s3 = m.summarize_pair(ps, "offroad_false_line_px", lower_is_better=True)
    assert s3["lower_is_better"] is True
