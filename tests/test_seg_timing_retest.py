"""受控复测协议（`m5_seg_timing_retest.py`）的判据与记账。

为什么单独钉：第 3 轮 5 个 seed 的推理 p95 出现 10.6–80.2 ms 的 7.6 倍漂移，
按纪律那批数字**不可引用**，要在静默机器上重测。重测本身也会出两种错：

1. 把"进程名里有 python"当成"机器在忙"——看板服务、监控进程都是空闲 python，
   于是永远测不出数（假阳性），或者反过来只看进程名不看 CPU，把真正在训练的那次
   当成安静的（假阴性，比前者更糟：污染被记成"模型很快"）；
2. 把被污染的那次重复也算进结论——所以协议要求**只在没被污染的重复里取最小**，
   一次都不干净就报"未测"。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_timing_retest", ROOT / "scripts" / "m5_seg_timing_retest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_timing_retest"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_suspect_rule_matches_the_rounds_gate():
    t = _load()
    assert t.is_suspect(21.7, 30.0) is False          # 正常 p95/p50 ≈ 1.3–1.6
    assert t.is_suspect(21.7, 100.0) is True          # 比值 4.6 -> 被污染
    assert t.is_suspect(None, 5.0) is True, "缺测不能被当成很快"
    assert t.is_suspect(0.0, 5.0) is True
    assert t.is_suspect(10.0, None) is True


def _fake_eval(seq):
    it = iter(seq)

    def run(_model, _frames, device="cuda"):
        return next(it)

    return run


def test_measure_keeps_every_repeat_and_picks_the_min_clean_one():
    t = _load()
    seq = [{"inference_ms_p50": 21.0, "inference_ms_p95": 30.0, "road_iou": 0.9},
           {"inference_ms_p50": 10.0, "inference_ms_p95": 150.0},   # 被污染
           {"inference_ms_p50": 20.0, "inference_ms_p95": 25.0}]
    got = t.measure("m.pt", ["dev"], repeats=3, evaluate=_fake_eval(seq),
                    load_frames=lambda dirs: ["f"])
    assert [r["p95"] for r in got["repeats"]] == [30.0, 150.0, 25.0], "重复都要留下"
    assert got["chosen"]["p95"] == 25.0, "取干净重复里的最小值"
    assert got["clean_repeats"] == 2 and got["any_suspect"] is True
    assert got["p95_min"] == 25.0 and got["p95_max"] == 150.0


def test_all_polluted_repeats_report_untested_not_fast():
    t = _load()
    bad = {"inference_ms_p50": 5.0, "inference_ms_p95": 90.0}   # 比值 18
    got = t.measure("m.pt", ["dev"], repeats=2,
                    evaluate=_fake_eval([bad, dict(bad)]),
                    load_frames=lambda dirs: ["f"])
    assert got["chosen"] is None, "一次都不干净就不给结论值"
    assert got["clean_repeats"] == 0 and got["any_suspect"] is True


def test_busy_detection_uses_cpu_delta_not_process_names(monkeypatch):
    t = _load()
    monkeypatch.setattr(t, "_cpu_times", lambda *a, **k: {})
    assert t.busy_pythons(interval_s=0.2) == [], "空闲的 python 不算忙"

    calls = {"n": 0}

    def climbing(*a, **k):
        calls["n"] += 1
        # 第一次采样 1.0 s，第二次 5.0 s -> 涨了 4 s，判忙
        return {111: 1.0} if calls["n"] == 1 else {111: 5.0}

    monkeypatch.setattr(t, "_cpu_times", climbing)
    busy = t.busy_pythons(interval_s=0.2)
    assert busy and busy[0]["pid"] == 111 and busy[0]["cpu_delta_s"] >= 3.0

    monkeypatch.setattr(t, "_cpu_times", lambda *a, **k: None)
    assert t.busy_pythons(interval_s=0.2) is None, "探测不确定不能报'安静'"
    assert calls["n"] >= 2


def test_a_noisy_machine_refuses_to_report(tmp_path, monkeypatch):
    t = _load()
    monkeypatch.setattr(t, "busy_pythons", lambda **k: [{"pid": 1}])
    monkeypatch.setattr(t, "game_running", lambda: False)
    monkeypatch.setattr(t, "gpu_util", lambda: 3.0)
    out = tmp_path / "t.json"
    rc = t.main(["--model", "s=some.pt", "--dev-runs", "dev", "--out", str(out)])
    assert rc == 3, "机器不安静要拒绝出数"
    assert not out.exists(), "拒绝时不许留下结果文件让人误用"


def test_a_quiet_machine_runs_and_writes_the_evidence(tmp_path, monkeypatch):
    t = _load()
    monkeypatch.setattr(t, "busy_pythons", lambda **k: [])
    monkeypatch.setattr(t, "game_running", lambda: False)
    monkeypatch.setattr(t, "gpu_util", lambda: 1.0)
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"")
    monkeypatch.setattr(t, "measure", lambda *a, **k: {
        "model": str(ck), "repeats": [{"i": 0, "p50": 20.0, "p95": 26.0}],
        "chosen": {"p95": 26.0}, "p95_min": 26.0, "p95_median": 26.0,
        "p95_max": 26.0, "clean_repeats": 1, "any_suspect": False})
    out = tmp_path / "t.json"
    rc = t.main(["--model", f"s42={ck}", "--dev-runs", "dev", "--out", str(out)])
    assert rc == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["quiet"] is True and blob["gpu_util"] == 1.0
    assert blob["models"][0]["name"] == "s42"
    assert blob["models"][0]["chosen"]["p95"] == 26.0
    assert "started_iso" in blob
