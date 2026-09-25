"""评估口径复核工具（`m5_seg_eval_criterion.py`）的对照逻辑。

`rounds` 现在只用 `checkpoint_last.pt`。要不要换成 `best.pt` 不能用"感觉更合理"来定：
两种口径回答的问题不同（定轮 vs 按训练内验证选优），所以先用同一批 checkpoint 把
两种都算出来，看结论会不会翻转。这里钉的是**计算与记账**，不替人做决定：

* 两种口径各给一份成对（按 seed）结果，缺测的 seed 剔除而不是补 0；
* 逐 seed 的优劣变化要能看见（翻转的 seed 单独列出来）；
* 两个口径结论一致时也如实写"一致"，不一致时列出翻转的 seed。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_eval_criterion", ROOT / "scripts" / "m5_seg_eval_criterion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_eval_criterion"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_eval(table):
    """``{(ckpt_name, seed_int): iou}`` -> 假的 evaluate。"""
    def run(path, frames, device="cuda"):
        p = Path(path)
        seed = p.parent.name.replace("seed", "")
        return {"road_iou": table.get((p.name, seed)),
                "inference_ms_p50": 12.0, "inference_ms_p95": 15.0}
    return run


def _run_dir(tmp_path: Path, *, arms=("baseline", "round0"), n=5,
             with_best=True) -> Path:
    run = tmp_path / "run"
    for arm in arms:
        for k in range(n):
            d = run / arm / f"seed{42 + k}"
            d.mkdir(parents=True)
            (d / "checkpoint_last.pt").write_bytes(b"x")
            if with_best:
                (d / "best.pt").write_bytes(b"x")
    return run


def test_both_criteria_are_computed_and_agreeing_verdicts_are_reported(
        tmp_path, monkeypatch):
    t = _load()
    run = _run_dir(tmp_path)
    # 候选在 last 上更好、在 best 上也更好 -> 两口径一致
    table = {}
    for k in range(5):
        seed = str(42 + k)
        table[("checkpoint_last.pt", seed)] = 0.90
        table[("best.pt", seed)] = 0.91
        table[("best.pt", seed)] = 0.91
    cand = {k: v for k, v in table.items()}
    both = {}
    for k in range(5):
        seed = str(42 + k)
        both[("checkpoint_last.pt", seed)] = 0.92
        both[("best.pt", seed)] = 0.93

    def ev(path, frames, device="cuda"):
        p = Path(path)
        seed = p.parent.name.replace("seed", "")
        src = both if p.parent.parent.name == "round0" else table
        return {"road_iou": src[(p.name, seed)], "inference_ms_p50": 12.0,
                "inference_ms_p95": 15.0}

    rep = t.criterion_report(run, ["baseline", "round0"], ["dev"],
                             evaluate=ev, load_frames=lambda dirs: ["f"])
    c = rep["compare"]["road_iou"]
    assert c["checkpoint_last.pt"]["n"] == 5
    assert c["best.pt"]["n"] == 5
    assert c["checkpoint_last.pt"]["verdict"] == "candidate_better"
    assert c["best.pt"]["verdict"] == "candidate_better"
    assert rep["same_verdict"] is True and rep["flipped_seeds"] == []
    assert cand  # 表本身用到了，避免 lint 误判未使用


def test_a_flip_between_criteria_is_listed_per_seed(tmp_path):
    t = _load()
    run = _run_dir(tmp_path, n=5)

    def ev(path, frames, device="cuda"):
        p = Path(path)
        seed = int(p.parent.name.replace("seed", ""))
        base = p.parent.parent.name == "baseline"
        if p.name == "checkpoint_last.pt":
            # last 上：候选更好（+0.01）
            v = 0.90 + (0.01 if not base else 0.0)
        else:
            # best 上：baseline 更好（-0.01），只有 seed45 例外
            v = 0.90 + (0.0 if not base else 0.01)
            if seed == 45:
                v = 0.95 if not base else 0.90
        return {"road_iou": v, "inference_ms_p50": 12.0,
                "inference_ms_p95": 15.0}

    rep = t.criterion_report(run, ["baseline", "round0"], ["dev"],
                             evaluate=ev, load_frames=lambda dirs: ["f"])
    c = rep["compare"]["road_iou"]
    assert c["checkpoint_last.pt"]["verdict"] == "candidate_better"
    assert c["best.pt"]["verdict"] in ("champion_better", "inconclusive"), c["best.pt"]
    flips = [f["seed"] for f in rep["flipped_seeds"]]
    # 42/43/44/46 在两个口径下优劣相反 -> 必须列出来；45 两口径都偏向候选 -> 不算翻转
    assert sorted(flips) == ["42", "43", "44", "46"], rep["flipped_seeds"]
    assert "45" not in flips
    assert rep["same_verdict"] is False


def test_a_missing_best_checkpoint_is_excluded_not_zeroed(tmp_path):
    t = _load()
    run = _run_dir(tmp_path, n=3, with_best=False)

    def ev(path, frames, device="cuda"):
        p = Path(path)
        base = p.parent.parent.name == "baseline"
        return {"road_iou": 0.90 + (0.0 if base else 0.01),
                "inference_ms_p50": 12.0, "inference_ms_p95": 15.0}

    rep = t.criterion_report(run, ["baseline", "round0"], ["dev"],
                             evaluate=ev, load_frames=lambda dirs: ["f"])
    c = rep["compare"]["road_iou"]
    assert c["checkpoint_last.pt"]["n"] == 3
    assert (c.get("best.pt") or {}).get("verdict") == "needs_evidence", \
        "没有 best.pt 时要报缺证据，不能拿 last 顶替"
    assert rep["per_arm"]["baseline"]["42"]["best.pt"].get("missing")


def test_the_cli_writes_the_report(tmp_path, monkeypatch):
    """CLI 只负责把对照结论落盘；采集/评估被替换成假表（不碰 GPU）。"""
    t = _load()
    per_arm = {
        "baseline": {str(42 + k): {"checkpoint_last.pt": {"metric": 0.90},
                                   "best.pt": {"metric": 0.91}}
                     for k in range(3)},
        "round0": {str(42 + k): {"checkpoint_last.pt": {"metric": 0.92},
                                 "best.pt": {"metric": 0.93}}
                   for k in range(3)},
    }
    monkeypatch.setattr(t, "collect_arms", lambda *a, **k: per_arm)
    out = tmp_path / "crit.json"
    rc = t.main(["--run-dir", str(tmp_path / "run"), "--arm", "baseline",
                 "--arm", "round0", "--dev-runs", "dev", "--out", str(out)])
    assert rc == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["metric"] == "road_iou" and "compare" in blob
    assert blob["compare"]["road_iou"]["checkpoint_last.pt"]["n"] == 3
    assert blob["same_verdict"] is True
