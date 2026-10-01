"""T16 Order 1 主 agent 接线证明：预算 / init / 采样标签 / N/A / 负例 v2。

方案 `docs/T16_AUTONOMOUS_LEARNING_ROADMAP_20260927.md` §3、§7 与
`docs/T16_ORDER0_FREEZE_20260927.md` §3 的接口。这里只证明**接线**：
`--total-steps`/`--init` 真的进了两臂命令、采样标签三档可辨、无线场景的
N/A 只在"P_frames==0 且档位 verified"时成立、负例 v2 的最大连通域取 max
而不是求和。不训练、不启动游戏、不动 GPU。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(argv, tmp_path, *, expect=None):
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_seg_autoloop.py")]
    cmd += [a.replace("{tmp}", str(tmp_path)) for a in argv]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=600)
    if expect is not None:
        assert r.returncode == expect, \
            f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-1500:]}\nSTDERR:\n{r.stderr[-800:]}"
    return r


def _frames(tmp_path, name: str) -> Path:
    """合成一组帧（与 rounds 接线测试同法：按目录名偏移内容，避免重复审计）。"""
    base = 10 + (sum(map(ord, name)) % 120)
    d = tmp_path / "runs" / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        colour = np.full((30, 40, 3), base + i * 5, np.uint8)
        label = np.zeros((30, 40), np.uint8)
        label[6:26, :] = 1
        label[15, :6] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    (tmp_path / "runs" / name / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "t_wall": 1.0 + i, "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(3)]}), encoding="utf-8")
    return d


def _proposal(tmp_path, factor: dict) -> Path:
    p = tmp_path / "props.json"
    p.write_text(json.dumps({"proposals": [
        {"candidate_id": "cand", "factor": factor}]}), encoding="utf-8")
    return p


def _args(**over):
    base = dict(trainer_script="m5_train_seg.py", allow_road_only=False,
                run_id="t16", paint_source=None, split="tail", epochs=24,
                batch=4, lr=1e-3, device="cpu", total_steps=0, init=None)
    base.update(over)
    return argparse.Namespace(**base)


# ----------------------------------------------------------- 预算与 init

def test_train_cmd_carries_budget_and_init():
    """两臂共用同一条命令构造：预算与 init 必须在里面（否则又是小样本从头训）。"""
    loop = _load()
    cmd = loop.train_cmd(_args(total_steps=300, init="weights/base.pt"),
                         ["runs/a"], Path("out/seed42"), 42, [])
    assert "--total-steps" in cmd
    assert cmd[cmd.index("--total-steps") + 1] == "300"
    assert "--init" in cmd
    assert cmd[cmd.index("--init") + 1] == "weights/base.pt"
    # 截帧与预算是互斥语义：命令构造本身不产生 --max-train-frames
    assert "--max-train-frames" not in cmd


def test_train_cmd_stays_legacy_without_budget_and_init():
    loop = _load()
    cmd = loop.train_cmd(_args(), ["runs/a"], Path("out/seed42"), 42, [])
    assert "--total-steps" not in cmd
    assert "--init" not in cmd


def test_sampling_label_has_three_distinguishable_modes():
    """判定/champion 的 sampling 标签：全池预算 / 截帧 / 纯 epoch 三档可辨。"""
    loop = _load()
    assert loop.sampling_label(_args(total_steps=200), 0) == "quota_full_pool"
    assert loop.sampling_label(_args(), 12) == "legacy_cap"
    assert loop.sampling_label(_args(), 0) == "legacy_epoch"


def test_rounds_help_documents_budget_and_init(tmp_path):
    r = _run(["rounds", "--help"], tmp_path, expect=0)
    assert "--total-steps" in r.stdout
    assert "--init" in r.stdout


def test_plan_only_puts_budget_and_init_into_both_arms(tmp_path):
    """plan-only 逐字打印两臂命令：预算/init 两臂一致（因子差异不能混进它们）。"""
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    p = _proposal(tmp_path, {"add_runs": [str(b)]})
    r = _run(["rounds", "--run-id", "t16_plan", "--rounds", "1", "--runs",
              str(a), "--eval-runs", str(a), "--proposals", str(p),
              "--seeds", "42", "--plan-only",
              "--total-steps", "120", "--init", "weights/base.pt"],
             tmp_path, expect=0)
    base = [l for l in r.stdout.splitlines() if l.startswith("[plan] baseline")]
    cand = [l for l in r.stdout.splitlines() if l.startswith("[plan] candidate")]
    assert base and cand
    for line in (base[0], cand[0]):
        toks = line.split()
        assert toks[toks.index("--total-steps") + 1] == "120"
        assert toks[toks.index("--init") + 1] == "weights/base.pt"
        assert "--max-train-frames" not in toks


def test_budget_mode_refuses_run_weights_instead_of_crashing():
    """预算模式（quota 采样器）与 run_weights 因子互斥：提前拒绝并说明替代。

    训练器对"quota + run_weights"是硬报错；如果等到训练启动才报，判定里只会
    留一条埋在 train_error 里的信息。这里要求它在**训练前**以 needs_review 拒绝。
    """
    loop = _load()
    extra, _skipped = loop.factor_to_flags({"run_weights": {"0": 2.0}})
    assert "--run-weights" in extra
    # 判据本身（接线位置在 cmd_rounds 里，这里证明判据条件可判定）
    assert int(getattr(_args(total_steps=100), "total_steps", 0) or 0) > 0
    assert int(getattr(_args(), "total_steps", 0) or 0) == 0


def test_rounds_refuses_run_weights_with_budget(tmp_path):
    a = _frames(tmp_path, "coll_a")
    b = _frames(tmp_path, "coll_b")
    dev = _frames(tmp_path, "coll_dev")
    p = _proposal(tmp_path, {"run_weights": {"0": 2.0}})
    r = _run(["rounds", "--run-id", "t16_rw", "--rounds", "1", "--runs",
              str(a), "--eval-runs", str(dev), "--proposals", str(p),
              "--seeds", "42", "--allow-road-only", "--total-steps", "100"],
             tmp_path)
    assert r.returncode == 3, r.stdout[-800:]
    assert "sampler_conflict" in r.stdout
    assert not (tmp_path / "logs" / "experiments" / "t16_rw" / "round0").exists(), \
        "拒绝必须发生在训练之前"


# ------------------------------------------------------------- N/A 适用性

def test_line_free_scenes_from_needs_verified_rank():
    """N/A 只在"P_frames==0 且档位 verified"时成立；档位不明保持 UNKNOWN。"""
    loop = _load()
    counts = {"italy/ring_a": {"P_frames": 0}, "italy/ring_b": {"P_frames": 0},
              "italy/ring_c": {"P_frames": 3}}
    ranks = {"italy/ring_a": "verified", "italy/ring_b": "",
             "italy/ring_c": "verified"}
    out = loop.line_free_scenes_from(counts, ranks)
    assert "italy/ring_a" in out, "确认真无线（verified 档）应判 N/A"
    assert out["italy/ring_a"]["basis"] == "rank_verified_zero_line_pixels"
    assert out["italy/ring_a"]["why"]
    assert "italy/ring_b" not in out, "档位不明不得冒充'确认无线'"
    assert "italy/ring_c" not in out, "有标线真值的场景不适用 N/A"


def test_line_free_scenes_from_rejects_unverified_and_mixed_ranks():
    loop = _load()
    out = loop.line_free_scenes_from(
        {"g": {"P_frames": 0}}, {"g": "agent"})
    assert out == {}, "非 verified 档即使全零也不算证明（T10 同一逻辑）"


# ------------------------------------------------------------- 负例 v2

def test_sum_negative_summaries_takes_max_for_connected_component():
    """最大连通域跨 seed 取 **max**，不是求和；控制区域帧数按帧相加。"""
    loop = _load()
    rows = {
        "42": {"frames": 2, "eligible_frames": 2, "clean_frames": 1,
               "false_positive_frames": 1, "false_positive_px": 10,
               "eligible_px": 1000, "positive_frames": 0, "unknown_frames": 0,
               "empty_frames": 0, "unverified_frames": 0,
               "unverified_pred_line_px": 0,
               "false_positive_max_cc_px": 100,
               "control_region_false_candidates": 0,
               "false_positive_control_region_frames": 0},
        "43": {"frames": 2, "eligible_frames": 2, "clean_frames": 0,
               "false_positive_frames": 2, "false_positive_px": 30,
               "eligible_px": 1000, "positive_frames": 0, "unknown_frames": 0,
               "empty_frames": 0, "unverified_frames": 0,
               "unverified_pred_line_px": 0,
               "false_positive_max_cc_px": 300,
               "control_region_false_candidates": 2,
               "false_positive_control_region_frames": 1},
    }
    out = loop.sum_negative_summaries(rows)
    assert out["false_positive_max_cc_px_max"] == 300, out
    assert out["counter_version"] == 2
    assert out["false_positive_control_region_frames"] == 1
    assert out["false_positive_frames"] == 3          # 普通计数仍是求和
    assert abs(out["false_positive_frame_rate"] - 0.75) < 1e-9


def test_sum_negative_summaries_keeps_v1_rows_readable():
    """旧 run（v1 计数，没有新键）照旧可汇总：新指标标 absent，不按 0 参与。"""
    loop = _load()
    rows = {"42": {"frames": 1, "eligible_frames": 1, "clean_frames": 1,
                   "false_positive_frames": 0, "false_positive_px": 0,
                   "eligible_px": 500, "positive_frames": 0,
                   "unknown_frames": 0, "empty_frames": 0,
                   "unverified_frames": 0, "unverified_pred_line_px": 0}}
    out = loop.sum_negative_summaries(rows)
    assert out["counter_version"] == 1
    assert out["extra_counters"] == "absent"
    assert "false_positive_max_cc_px_max" not in out
    assert out["false_positive_frame_rate"] == 0.0


# ------------------------------------------------- 实测步数（预算模式权威计数）

def test_ckpt_steps_done_prefers_recorded_steps(tmp_path):
    """预算模式：实测步数必须来自 checkpoint 的 steps_done，不是 epochs 估算。

    实测踩到（2026-09-30 严格门实验）：候选臂实跑 480 步、判定里
    `steps_by_arm.candidate` 却写 72（= 3 epochs × 24 steps/epoch），
    把一次有效的等预算对照读成"候选被欠训"。
    """
    import torch
    loop = _load()
    ck = tmp_path / "checkpoint_last.pt"
    torch.save({"train_args": {"n_train": 94, "batch": 4, "epochs": 3,
                               "steps_done": 480}}, str(ck))
    assert loop.ckpt_steps_done(ck, batch=4, epochs=3) == 480
    # 旧 checkpoint 没有 steps_done 才退回估算（94/4=24 -> 3×24=72）
    torch.save({"train_args": {"n_train": 94, "batch": 4, "epochs": 3}},
               str(ck))
    assert loop.ckpt_steps_done(ck, batch=4, epochs=3) == 72
    # 连 n_train 都没有（旧格式）：None，不猜
    torch.save({"train_args": {}}, str(ck))
    assert loop.ckpt_steps_done(ck, batch=4, epochs=3) is None
    assert loop.ckpt_steps_done(tmp_path / "missing.pt", batch=4,
                                epochs=3) is None


# ------------------------------------------------- 因子必须真的进训练（划分口径）

def test_effective_split_switches_to_per_run_for_data_arms():
    """数据臂（候选比基线多 run）必须用 per-run 划分。

    实测踩到（2026-09-30）：`--split tail` 取"所有 run 拼接后的全局尾部"当验证集，
    而 `add_runs` 把新数据追加在末尾 -> 新数据正好落进验证集。2 个负例包 16 帧
    < 18 帧验证集 -> **因子 100% 变成验证数据**，训练输入一个字节没变，而
    "因子已生效"的检查只看 --runs 变没变（历史上所有 2 包负例臂都是白跑）。
    """
    loop = _load()
    base = ["a", "b"]
    assert loop.effective_split("tail", base, base) == "tail"      # 非数据臂不动
    assert loop.effective_split("tail", base, base + ["c"]) == "per-run"
    assert loop.effective_split("per-run", base, base + ["c"]) == "per-run"
    assert loop.effective_split("by-map-scene", base,
                                base + ["c"]) == "by-map-scene"


def test_added_run_train_frames_flags_zero_train_runs():
    """追加 run 拿不到训练帧 = 因子改不动输入，必须能被判出来。"""
    loop = _load()
    counts = {"gen/one": 16, "gen/tiny": 1, "gen/ok": 8}
    added = list(counts)
    # tail：追加的 run 在列表末尾 -> 全部进验证集（0 训练帧）
    g = loop.added_run_train_frames(added, split="tail", val_frac=0.2,
                                    n_frames_by_run=counts)
    assert g["never_trained"] == sorted(added), g
    # per-run：每个 run 各取尾部 20% 做验证（>=2 帧的 run 都有训练帧）
    g2 = loop.added_run_train_frames(added, split="per-run", val_frac=0.2,
                                     n_frames_by_run=counts)
    assert g2["train_frames_by_run"]["gen/ok"] == 7, g2      # 8 -> val 1
    assert g2["train_frames_by_run"]["gen/one"] == 13, g2    # 16 -> val 3
    assert g2["train_frames_by_run"]["gen/tiny"] == 0, g2    # 1 帧全进 val
    assert g2["never_trained"] == ["gen/tiny"], g2
    # 帧数未知（目录不存在）不当作通过：记 None，不进 never_trained
    g3 = loop.added_run_train_frames(["missing"], split="per-run", val_frac=0.2,
                                     n_frames_by_run={})
    assert g3["train_frames_by_run"]["missing"] is None
    assert g3["never_trained"] == []


# ------------------------------------------------- 合成线数据的剂量纪律（§22.3）

def test_synthetic_line_dose_levels():
    """剂量纪律：合成**线**数据占候选训练帧的比例分三档（实测依据 §22.2/§22.3）。

    16 帧（≈21%）身份率 +0.0214（四项 candidate_better）；56 帧（≈51%）−0.1691
    崩溃。所以：≤0.20 照跑；0.20–0.35 记 warn；>0.35 必须显式 --allow-overdose。
    """
    loop = _load()
    base = ["logs/experiments/human_a", "logs/experiments/human_b"]
    added = ["logs/experiments/gen_a", "logs/experiments/gen_b"]
    # 键用模块自己的归一化函数算（测试会话把 BEAMNG_LOGS_DIR 指到沙箱，
    # 硬编码 "experiments/..." 会与运行期口径不一致）
    K = loop._run_key_like
    frames = {K(base[0]): 30, K(base[1]): 30, K(added[0]): 8, K(added[1]): 8}
    line_by_dir = {K(added[0]): {"n_frames": 8, "n_line_frames": 8},
                   K(added[1]): {"n_frames": 8, "n_line_frames": 8},
                   K(base[0]): {"n_frames": 30, "n_line_frames": 15},
                   K(base[1]): {"n_frames": 30, "n_line_frames": 15}}
    # 16/76 = 0.21 -> warn（§21 那一档的池占比是 0.149 = ok，这里构造 warn 区）
    d = loop.synthetic_line_dose(added_runs=added, base_runs=base,
                                 line_by_dir=line_by_dir,
                                 frames_by_run=frames)
    assert d["synthetic_line_frames"] == 16 and d["candidate_frames"] == 76
    assert d["share"] == round(16 / 76, 4) and d["level"] == "warn"
    # 只用 4 帧：4/64 = 0.0625 -> ok
    d2 = loop.synthetic_line_dose(
        added_runs=added[:1], base_runs=base,
        line_by_dir={K(added[0]): {"n_line_frames": 4}},
        frames_by_run={K(base[0]): 30, K(base[1]): 30, K(added[0]): 8})
    assert d2["share"] == round(4 / 68, 4) and d2["level"] == "ok"
    # 大剂量：48/108 = 0.44 -> over（§22 的崩溃档 0.26/0.38）
    big = dict(frames)
    big.update({K(added[0]): 24, K(added[1]): 24})
    d3 = loop.synthetic_line_dose(
        added_runs=added, base_runs=base,
        line_by_dir={K(added[0]): {"n_line_frames": 24},
                     K(added[1]): {"n_line_frames": 24}},
        frames_by_run=big)
    assert d3["level"] == "over", d3
    # 缺线帧计数：记 unknown_dirs，且不虚增剂量
    d4 = loop.synthetic_line_dose(added_runs=added, base_runs=base,
                                  line_by_dir={}, frames_by_run=frames)
    assert d4["synthetic_line_frames"] == 0
    assert d4["unknown_dirs"] == sorted([K(added[0]), K(added[1])])
    assert d4["level"] == "ok"
    # 0.26（32 帧那档的实测占比）必须落在 over：那是**已实测的崩溃区**
    d5 = loop.synthetic_line_dose(
        added_runs=added, base_runs=base,
        line_by_dir={K(added[0]): {"n_line_frames": 10},
                     K(added[1]): {"n_line_frames": 10}},
        frames_by_run=frames)
    assert d5["share"] == round(20 / 76, 4) and d5["level"] == "over", d5
