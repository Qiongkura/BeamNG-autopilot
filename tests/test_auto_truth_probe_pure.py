"""自动真值探针（``--runtime pure``）的薄入口行为。

探针是"先证明能自动产/能自动拒，再决定扩量"的守门脚本（方案 §4.1/§4.3）：
五类已知场景全过、五个反例全被对应码拒绝才 ``exit 0``；任何一个没做到就非 0，
报告里看得到是哪一个。Tech 模式缺 BeamNGpy/未连接/没有真值点必须**大声失败**，
不把 UNKNOWN 当通过（本开发窗口不执行 Tech 实跑，只钉住失败路径与退出码）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_auto_truth_probe", ROOT / "scripts" / "m5_auto_truth_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_auto_truth_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_pure_probe_passes_five_scenes_and_rejects_five_counterexamples(tmp_path):
    mod = _load()
    out = tmp_path / "pure.json"
    rc = mod.main(["--runtime", "pure", "--frames", "2", "--out", str(out)])
    assert rc == 0, out.read_text(encoding="utf-8")
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["runtime"] == "pure" and report["ok"] is True
    assert report["truth_contract"] == "v1"
    assert report["verifier_version"]
    assert report["n_scenes"] == 5 and report["n_scenes_ok"] == 5
    names = [s["name"] for s in report["scenes"]]
    assert names == ["known_line", "known_no_line", "occluded_line",
                     "slope_curve", "material_mix"]
    for scene in report["scenes"]:
        assert scene["ok"] is True and scene["rejections"] == []
        assert scene["evidence"]["projection_checked"] > 0
        assert set(scene["channels"]) == {"LINE", "ROAD", "SHOULDER", "ROLE"}
    by_name = {s["name"]: s for s in report["scenes"]}
    assert by_name["known_no_line"]["channels"]["LINE"] == "not_applicable"
    assert by_name["known_no_line"]["channels"]["ROAD"] == "measured"
    assert by_name["material_mix"]["channels"]["SHOULDER"] == "measured"
    assert by_name["occluded_line"]["evidence"]["occlusion_occluded_px"] > 0

    assert report["n_counterexamples"] == 5
    assert report["n_counterexamples_rejected"] == 5
    kinds = {c["kind"]: c for c in report["counterexamples"]}
    assert set(kinds) == {"flip", "resize", "time_shift", "occlude", "tamper"}
    for kind, entry in kinds.items():
        assert entry["ok"] is True, (kind, entry)
        assert entry["expected_code"] in entry["rejected_codes"], (kind, entry)
    assert "不盲目扩量" in mod.__doc__


def test_the_pure_probe_exits_nonzero_when_a_counterexample_slips_through(
        tmp_path, monkeypatch):
    """退出码语义：反例没被拒 = 探针失败（不是"报告好看就过"）。"""
    mod = _load()
    monkeypatch.setattr(mod, "inject", lambda batch, kind, **kw: batch)
    out = tmp_path / "blind.json"
    rc = mod.main(["--runtime", "pure", "--frames", "1", "--out", str(out)])
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["n_counterexamples_rejected"] == 0
    assert report["n_scenes_ok"] == 5


def test_the_pure_probe_runs_as_a_subprocess_with_the_documented_exit_code(tmp_path):
    out = tmp_path / "sub.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "m5_auto_truth_probe.py"),
         "--runtime", "pure", "--frames", "1", "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True and report["n_scenes_ok"] == 5


def test_the_tech_probe_fails_loudly_without_truth_points_or_connection(tmp_path):
    mod = _load()
    out = tmp_path / "tech.json"
    rc = mod.main(["--runtime", "tech", "--out", str(out)])
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["runtime"] == "tech" and report["ok"] is False
    assert "truth-json" in report["error"]
    assert report.get("hint")
    # 文件不存在同样是大声失败（不是静默空真值）
    out2 = tmp_path / "tech2.json"
    rc2 = mod.main(["--runtime", "tech", "--truth-json",
                    str(tmp_path / "nope.json"), "--out", str(out2)])
    assert rc2 == 1
    report2 = json.loads(out2.read_text(encoding="utf-8"))
    assert report2["ok"] is False and "truth-json" in report2["error"]


def test_depth_conversion_and_sanity_guard_the_tech_path():
    mod = _load()
    np = importlib.import_module("numpy")
    # NDC 反解：d=1（远平面）-> far；d=0 -> near
    meters = mod._depth_to_meters(np.array([[0.0, 1.0]], dtype=np.float32),
                                  near=0.05, far=150.0, mode="ndc")
    assert abs(float(meters[0, 1]) - 150.0) < 1e-3
    assert abs(float(meters[0, 0]) - 0.05) < 1e-3
    linear = mod._depth_to_meters(np.array([[0.5]], dtype=np.float32),
                                  near=0.05, far=150.0, mode="linear")
    assert abs(float(linear[0, 0]) - (0.05 + 0.5 * 149.95)) < 1e-3
    # 已是米（>1.5）时原样返回，不重复换算
    passthrough = mod._depth_to_meters(np.array([[12.0, 3.0]], dtype=np.float32),
                                       near=0.05, far=150.0, mode="ndc")
    assert float(passthrough[0, 0]) == 12.0 and float(passthrough[0, 1]) == 3.0
    # 自检：空深度/离谱深度都要失败
    assert mod._tech_depth_sanity(None)[0] is False
    assert mod._tech_depth_sanity(np.zeros((8, 8), np.float32))[0] is False
    ok, why = mod._tech_depth_sanity(np.full((8, 8), 2.0, np.float32))
    assert ok is True and "2.00" in why
