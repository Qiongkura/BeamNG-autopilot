"""无人值守采集的**决策与校验**（`experiments/collection.py`）。

用户 2026-09-25 授权"允许无人值守启动 Tech 采集"。授权放开的是**自动启动游戏**，
不是"把没经过人工修订的采集结果当标线真值"，也不是"少一道检查"。本文件钉：

1. 前置检查：游戏已经在跑就不抢会话（有人可能在开），探测不确定**不算通过**；
   `--collect-force` 只能降级"人在开"这一类，资源门不降级；
2. 采集后身份审计：缺 `map_name`/`source_id` 一律拒收；身份来源弱（命令行回退）
   与"某视角没有漆线像素"要作为警告报出来，不能混成"通过"；
3. 起点：同一个 spawn 连采两次会得到同一段路，所以起点必须能换（`--teleport`）；
4. Tech 与 Steam 的进程名不同，探测必须都查——只查一个会在别人开着 Tech 时
   误判"没在跑"，然后去抢一个已经在用的端口。

脚本侧的接线（采集→训练）在 `tests/test_seg_collect_unattended.py`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.collection import (
    CollectSpec, collect_command, collection_proposal, collector_python,
    game_running_probe, merge_proposals, paint_frame_priority, preflight,
    python_can_import, verify_collection,
)


# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
def test_preflight_blocks_when_the_game_is_already_running():
    pf = preflight(spec=CollectSpec(), game_running=True,
                   free_vram_mb=8000.0, free_disk_gb=50.0)
    assert pf["ok"] is False
    assert any("游戏已在运行" in r for r in pf["reasons"]), pf


def test_preflight_force_downgrades_only_the_game_signal():
    pf = preflight(spec=CollectSpec(), game_running=True,
                   free_vram_mb=8000.0, free_disk_gb=50.0, force=True)
    assert pf["ok"] is True and pf["forced"] is True
    assert any("游戏已在运行" in w for w in pf["warnings"]), pf
    # 资源门不因为 force 降级
    hard = preflight(spec=CollectSpec(), game_running=True, free_vram_mb=10.0,
                     free_disk_gb=50.0, force=True, min_free_vram_mb=2048.0)
    assert hard["ok"] is False
    assert any("VRAM" in r for r in hard["reasons"]), hard


def test_an_unknown_probe_result_is_not_a_pass():
    """探测不确定（tasklist 读不出来）≠ 没在跑：无人值守不猜。"""
    pf = preflight(spec=CollectSpec(), game_running=None,
                   free_vram_mb=8000.0, free_disk_gb=50.0)
    assert pf["ok"] is False
    assert any("无法确认游戏是否在运行" in r for r in pf["reasons"]), pf


def test_preflight_flags_a_stationary_grab_as_duplicate_samples():
    pf = preflight(spec=CollectSpec(step_m=0.0), game_running=False,
                   free_vram_mb=8000.0, free_disk_gb=50.0)
    assert pf["ok"] is True
    assert any("重复样本" in w for w in pf["warnings"]), pf


def test_the_collector_uses_the_project_venv_not_the_system_python(tmp_path):
    """实测缺陷：系统 Python 没有 beamngpy，采集会白起一局再被审计拒收。"""
    v = tmp_path / ".venv" / "Scripts"
    v.mkdir(parents=True)
    (v / "python.exe").write_bytes(b"")
    got, src = collector_python(tmp_path)
    assert src == "project-venv" and Path(got) == v / "python.exe"
    # 配置里给了就用配置的（来源要标出来，别让人以为是自动挑的）
    got2, src2 = collector_python(tmp_path, configured="D:/other/python.exe")
    assert (got2, src2) == ("D:/other/python.exe", "config")
    # 没有 venv 时回落到 sys.executable，来源照写
    got3, src3 = collector_python(tmp_path / "nope")
    assert src3 == "sys.executable" and got3


def test_python_can_import_reports_a_missing_module(tmp_path):
    import sys as _sys
    assert python_can_import(_sys.executable, "definitely_not_a_module_xyz") \
        is False
    assert python_can_import(_sys.executable, "json") is True
    assert python_can_import(str(tmp_path / "no-such-python.exe")) is False


def test_preflight_refuses_an_interpreter_without_beamngpy():
    pf = preflight(spec=CollectSpec(), game_running=False,
                   free_vram_mb=8000.0, free_disk_gb=50.0,
                   collector_python_ok=False,
                   collector_python="I:/python/python.exe")
    assert pf["ok"] is False
    assert any("beamngpy" in r for r in pf["reasons"]), pf
    assert pf["collector_python"] == "I:/python/python.exe"
    # 没检查过不算通过（要报出来，不能默认当成好的）
    pf2 = preflight(spec=CollectSpec(), game_running=False,
                    free_vram_mb=8000.0, free_disk_gb=50.0)
    assert pf2["ok"] is True
    assert any("beamngpy" in w for w in pf2["warnings"]), pf2


def test_attach_is_not_the_unattended_default():
    cmd = collect_command(python="py", script="s.py", spec=CollectSpec(),
                          out_dir="o")
    assert "--attach" not in cmd, "无人值守默认不带 --attach（不抢别人的会话）"
    assert cmd[cmd.index("--runtime") + 1] == "tech"
    assert cmd[cmd.index("--out") + 1] == "o"
    assert "--follow-road" in cmd and "--save-annotation" in cmd
    roles = cmd[cmd.index("--roles") + 1:cmd.index("--out")]
    assert list(roles) == ["front_main", "front_fisheye", "pillar_left",
                           "pillar_right"]
    pf = preflight(spec=CollectSpec(attach=True), game_running=False,
                   free_vram_mb=8000.0, free_disk_gb=50.0)
    assert any("attach" in w for w in pf["warnings"]), pf


def test_a_fresh_start_point_can_be_given():
    """同一个 spawn 连采两次会得到同一段路：起点必须能换。"""
    spec = CollectSpec(teleport=(1312.2, 676.9, 34.8))
    cmd = collect_command(python="py", script="s.py", spec=spec, out_dir="o")
    i = cmd.index("--teleport")
    assert cmd[i + 1:i + 4] == ["1312.2", "676.9", "34.8"], cmd
    assert spec.as_dict()["teleport"] == [1312.2, 676.9, 34.8]
    # 没给起点就不带这个开关（默认按场景 spawn）
    assert "--teleport" not in collect_command(
        python="py", script="s.py", spec=CollectSpec(), out_dir="o")


# ---------------------------------------------------------------------------
# 采集后身份审计
# ---------------------------------------------------------------------------
def _write_collection(root: Path, *, map_name="italy",
                      source_id="ring_20260925_120000",
                      map_source="session.get_current().level",
                      roles=("front_main", "pillar_left"),
                      frames_per_role=3, paint_roles=("front_main",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    recs = []
    for role in roles:
        d = root / role
        d.mkdir(parents=True, exist_ok=True)
        for i in range(frames_per_role):
            (d / f"frame_{i:05d}.npz").write_bytes(b"not-a-real-npz")
            recs.append({"i": i, "view": role, "exposure": i,
                         "path": f"{role}/frame_{i:05d}.npz",
                         "line_pixels": 500 if role in paint_roles else 0,
                         "pos": [1.0, 2.0, 3.0], "heading": 0.5})
    meta = {"stamp": "20260925_120000", "roles": {r: frames_per_role
                                                 for r in roles},
            "width": 536, "height": 403,
            "classes": ["background", "road", "line"],
            "map_name": map_name, "map_name_source": map_source,
            "source_id": source_id, "frames": recs}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")
    return root


def test_verify_accepts_a_collection_with_identity(tmp_path):
    d = _write_collection(tmp_path / "c1")
    rep = verify_collection(d, expected_roles=("front_main", "pillar_left"),
                            expected_frames=6)
    assert rep["ok"] is True, rep
    assert rep["map_name"] == "italy"
    assert rep["source_id"] == "ring_20260925_120000"
    assert rep["frames_total"] == 6
    assert rep["paint_frames_by_role"] == {"front_main": 3}


def test_verify_rejects_a_collection_without_map_identity(tmp_path):
    d = _write_collection(tmp_path / "c2", map_name="", map_source=None)
    rep = verify_collection(d)
    assert rep["ok"] is False
    assert any("map_name 为空" in r for r in rep["reasons"]), rep
    # 身份缺失时不许伪造：审计结果里没有地图名
    assert rep["map_name"] == ""


def test_verify_rejects_an_empty_view(tmp_path):
    d = _write_collection(tmp_path / "c3")
    rep = verify_collection(d, expected_roles=("front_main", "front_fisheye"))
    assert rep["ok"] is False
    assert any("front_fisheye" in r for r in rep["reasons"]), rep


def test_verify_warns_on_weak_provenance_and_on_a_paintless_view(tmp_path):
    d = _write_collection(tmp_path / "c4", map_source="argument-fallback")
    rep = verify_collection(d, expected_roles=("front_main", "pillar_left"))
    assert rep["ok"] is True, rep
    assert any("argument-fallback" in w for w in rep["warnings"]), rep
    assert any("pillar_left" in w and "漆线" in w for w in rep["warnings"]), rep


def test_paint_priority_points_the_annotator_at_the_paint(tmp_path):
    d = _write_collection(tmp_path / "c5", roles=("front_main",),
                          frames_per_role=3)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["frames"][0]["line_pixels"] = 9000
    rows = paint_frame_priority(meta, top=2)
    assert [r["line_pixels"] for r in rows] == [9000, 500]
    assert rows[0]["view"] == "front_main"


# ---------------------------------------------------------------------------
# 提议合并 / 进程探测
# ---------------------------------------------------------------------------
def test_merge_proposals_dedupes_by_candidate_id_and_keeps_order():
    a = collection_proposal(role_dirs=["x/front_main"], candidate_id="collect-1")
    b = {"note": "手写", "proposals": [
        {"candidate_id": "collect-1", "factor": {"add_runs": ["dup"]}},
        {"candidate_id": "manual", "factor": {"add_runs": ["y"]}}]}
    merged = merge_proposals(a, b)
    ids = [p["candidate_id"] for p in merged["proposals"]]
    assert ids == ["collect-1", "manual"], ids
    # Windows 上 Path 用反斜杠；比较 Path 而不是字面串
    assert [Path(p) for p in merged["proposals"][0]["factor"]["add_runs"]] ==         [Path("x/front_main")]


def test_the_tech_process_name_is_probed_too():
    """Tech 的进程名与 Steam 版不同；只查 drive 会在别人开着 Tech 时误判。"""
    from beamng_autopilot.experiments.collection import GAME_IMAGES
    assert "BeamNG.tech.x64.exe" in GAME_IMAGES, GAME_IMAGES


def test_game_probe_treats_unreadable_output_as_not_running(monkeypatch):
    """中文 Windows 的 tasklist 输出不是 UTF-8；乱码里没有进程名就是没在跑。"""
    import subprocess as sp

    class R:
        returncode = 0
        stdout = ("\u4fe1\u606f: \u6ca1\u6709\u8fd0\u884c\u7684\u4efb\u52a1"
                  "\u5339\u914d\u6307\u5b9a\u7684\u6807\u51c6\u3002")

    monkeypatch.setattr(sp, "run", lambda *a, **k: R())
    assert game_running_probe() is False

    class R2:
        returncode = 0
        stdout = ("BeamNG.drive.x64.exe           1234 Console"
                  "                    1  1,000,000 K")

    monkeypatch.setattr(sp, "run", lambda *a, **k: R2())
    assert game_running_probe() is True

    class R3:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(sp, "run", lambda *a, **k: R3())
    assert game_running_probe() is None, "返回码非 0 时不敢说'没在跑'"


def test_game_pids_parses_tasklist_and_refuses_to_guess(monkeypatch):
    """进程列表：看清了才敢给集合，看不清就返 None。

    实测坑：中文 Windows 的 tasklist 输出不是 UTF-8，读空了当成"没进程"
    会把正在驾驶的会话当成可以杀的对象。
    """
    import subprocess as sp

    from beamng_autopilot.experiments import collection as col

    class R:
        returncode = 0
        stdout = ("BeamNG.tech.x64.exe        4242 Console"
                  "                    1  4,400,000 K")

    monkeypatch.setattr(sp, "run", lambda *a, **k: R())
    assert col.game_pids() == {4242}

    class RNone:
        returncode = 0
        stdout = "信息: 没有运行的任务匹配指定的标准。"

    monkeypatch.setattr(sp, "run", lambda *a, **k: RNone())
    assert col.game_pids() == set()

    class RBad:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(sp, "run", lambda *a, **k: RBad())
    assert col.game_pids() is None, "探测失败不能当成空集合"


def test_close_started_game_kills_only_the_new_pids(monkeypatch):
    """只杀本次采集新出现的游戏进程：用户自己开着的会话永远不在差集里。"""
    import subprocess as sp

    from beamng_autopilot.experiments import collection as col

    calls: list = []

    class R:
        returncode = 0
        stdout = ""

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return R()

    monkeypatch.setattr(sp, "run", fake_run)
    # 调用顺序：先看到 {7,8,9}，杀完再看到 {7}（实现里就是这么校验的）
    seq = [{7, 8, 9}, {7}]
    monkeypatch.setattr(col, "game_pids", lambda *a, **k: seq.pop(0))
    got = col.close_started_game({7})
    assert got["closed"] is True and got["killed"] == [8, 9], got
    killed = [c[c.index("/PID") + 1] for c in calls if "/PID" in c]
    assert killed == ["8", "9"], "不能动已存在的 pid 7"


def test_close_started_game_never_guesses(monkeypatch):
    from beamng_autopilot.experiments import collection as col

    called = []
    monkeypatch.setattr(col.subprocess, "run",
                        lambda *a, **k: called.append(a))
    got = col.close_started_game(None)
    assert got["closed"] is False and "reason" in got
    assert not called, "不确定时不许发任何杀进程命令"
    monkeypatch.setattr(col, "game_pids", lambda *a, **k: None)
    got2 = col.close_started_game({1, 2})
    assert got2["closed"] is False and not called


def test_close_started_game_reports_what_is_still_running(monkeypatch):
    from beamng_autopilot.experiments import collection as col

    class R:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(col.subprocess, "run", lambda *a, **k: R())
    seq = [{5, 6}, {6}]        # 杀前看到 5/6，杀后 6 还在
    monkeypatch.setattr(col, "game_pids", lambda *a, **k: seq.pop(0))
    got = col.close_started_game({4})
    assert got["closed"] is False and got["still_running"] == [6], got
    assert "GPU 仍在被占用" in got["reason"], got


def test_the_game_is_closed_only_with_proven_ownership(monkeypatch):
    """G06：新出现的 pid 还要能证明是这次任务起的才关。

    实测风险：只看 pid 差集会把"PID 复用"的旧进程（或拿不到
    创建时间的进程）当成自己起的而结束掉。
    """
    import time as _t

    from beamng_autopilot.experiments import collection as col

    now = _t.time()

    def ticks(unix):
        return str(int((unix + 62135596800) * 1e7))   # .NET ticks

    monkeypatch.setattr(col, "game_pids", lambda *a, **k: {7, 8, 9})
    monkeypatch.setattr(col, "game_procs", lambda *a, **k: {
        7: ticks(now - 3600),        # 早就在跑（不在差集里）
        8: ticks(now + 1),           # 刚刚由我们启动 -> 可关
        9: None})                    # 拿不到创建时间 -> 不能关

    class R:
        returncode = 0

    calls: list = []
    monkeypatch.setattr(col.subprocess, "run",
                        lambda cmd, **k: calls.append(list(cmd)) or R())
    got = col.close_started_game({7}, launched_after=now)
    assert got["killed"] == [8], got
    killed = [c[c.index("/PID") + 1] for c in calls if "/PID" in c]
    assert killed == ["8"], killed
    assert [u["pid"] for u in got["unverified"]] == [9], got
    # 创建时间早于启动时刻（PID 复用）也不关
    monkeypatch.setattr(col, "game_procs", lambda *a, **k: {
        7: ticks(now - 3600), 8: ticks(now - 60), 9: ticks(now + 1)})
    calls.clear()
    got2 = col.close_started_game({7}, launched_after=now)
    assert got2["killed"] == [9], got2
    assert any(u["pid"] == 8 and "pid reuse" in u["why"]
               for u in got2["unverified"]), got2
    # 不给 launched_after（旧调用）：保持原行为，不做所有权判定
    calls.clear()
    got3 = col.close_started_game({7})
    assert sorted(got3["killed"]) == [8, 9], got3
