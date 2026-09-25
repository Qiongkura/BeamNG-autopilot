"""无人值守采集的**接线**：采集 → 身份审计 → 数据因子 → 训练。

`experiments/collection.py` 里的纯逻辑测试在
`tests/test_experiments_collection.py`。本文件只钉 `scripts/m5_seg_autoloop.py`
怎么用它：前置检查不过就不启动游戏（rc=6），采集跑完但身份不合格就拒收且
**不产出提议**（rc=7），通过时把游戏时间记进每日 GPU 账本、产出
`add_runs` 因子并排在**第一个**候选，采集没过时**不训练**。
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

from beamng_autopilot.experiments.collection import (  # noqa: E402
    collection_proposal,
)


def _write_collection(root: Path, *, map_name="italy",
                      source_id="ring_20260925_120000",
                      map_source="session.get_current().level",
                      roles=("front_main", "pillar_left"),
                      frames_per_role=3, paint_roles=("front_main",)) -> Path:
    """假采集产物：`verify_collection` 只读 meta.json 的字段，这里照它写。"""
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
    meta = {"stamp": "20260925_120000",
            "roles": {r: frames_per_role for r in roles},
            "width": 536, "height": 403,
            "classes": ["background", "road", "line"],
            "map_name": map_name, "map_name_source": map_source,
            "source_id": source_id, "frames": recs}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                    encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# 脚本接线（把 rounds/采集子进程都换成记录器）
# ---------------------------------------------------------------------------
def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop_collect", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop_collect"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeSP:
    """假的 subprocess 模块：run / TimeoutExpired / STDOUT 三个成员。"""

    STDOUT = -2                  # 采集的 stderr 并进同一个日志文件

    class TimeoutExpired(Exception):
        pass

    def __init__(self, fn):
        self.run = fn


def _config(tmp_path: Path, **over) -> Path:
    blob = {"collect": "tech", "daily_gpu_minutes": 120.0,
            "max_wall_minutes": 180.0, "window_start_hour": 0,
            "window_end_hour": 24, "pause_while_user_active": True,
            "min_free_vram_mb": 0.0, "min_free_disk_gb": 0.0,
            "max_candidates": 6, "max_rounds_without_gain": 2,
            "seeds": [42], "dry_run": True,
            "runs": ["logs/m5_seg/diverse_town_20260924/front_main"],
            "baseline_runs": ["logs/m5_seg/diverse_town_20260924/front_main"],
            "eval_runs": ["logs/m5_seg/diverse_wide_20260924/front_main"],
            "proposals": "", "rounds": 1, "epochs": 1, "batch": 4,
            "lr": 0.001, "allow_road_only": True, "equal_steps": True,
            "trainer_script": "m5_train_seg.py"}
    blob.update(over)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


@pytest.fixture()
def loop(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "exp_dir", lambda rid: tmp_path / "exp" / rid)
    monkeypatch.setattr(mod, "probe_machine",
                        lambda: (False, 9000.0, 40.0))
    # 采集产物根目录也指到 tmp：测试不许往仓库 logs/ 里写采集结果
    monkeypatch.setattr(mod, "collect_root", lambda: tmp_path / "m5_seg")
    return mod


def _events(mod, run_id: str) -> list:
    p = mod.exp_dir(run_id) / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def test_collect_blocks_before_launching_when_the_game_is_running(
        loop, monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "probe_machine", lambda: (True, 9000.0, 40.0))
    called = []
    monkeypatch.setattr(loop, "subprocess", _FakeSP(
        lambda *a, **k: called.append(a) or None))
    rc = loop.main(["collect", "--run-id", "c1", "--config",
                    str(_config(tmp_path))])
    assert rc == 6, "前置检查不通过时不得启动游戏"
    assert not called, "被拦下就不该起子进程"
    st = [e["status"] for e in _events(loop, "c1")]
    assert "collect_blocked" in st, st


def test_collect_rejects_a_collection_without_identity(loop, monkeypatch,
                                                       tmp_path):
    """采集跑完了但没身份：拒收、记事件、**不产出提议**（不训练）。"""
    monkeypatch.setattr(loop, "game_running_probe", lambda: False)

    class R:
        returncode = 0
        stdout = "[ring-collect] wrote x"
        stderr = ""

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        _write_collection(out, map_name="", roles=("front_main",),
                          frames_per_role=2)
        # 采集的输出必须写文件，不能被父进程用管道收走：游戏会继承那个管道，
        # 采集进程退出后父进程会一直等 EOF（实测卡 30 分钟）
        assert kw.get("stdout") is not None, "采集输出要重定向到文件"
        kw["stdout"].write("来自采集进程的日志" + chr(10))
        return R()

    monkeypatch.setattr(loop, "subprocess", _FakeSP(fake_run))
    cfg = _config(tmp_path)
    rc = loop.main(["collect", "--run-id", "c2", "--config", str(cfg),
                    "--collect-roles", "front_main", "--collect-frames", "2"])
    assert rc == 7, "身份缺失必须拒收"
    d = loop.exp_dir("c2")
    assert not list(d.glob("proposals_collect_*.json")), "拒收的采集不许变成候选"
    rec = json.loads(next(d.glob("collect_*.json")).read_text(encoding="utf-8"))
    assert rec["ok"] is False and rec["rc"] == 0
    assert any("map_name" in r for r in rec["reasons"]), rec["reasons"]
    # 采集自己的日志要留在盘上（排障用），尾巴也进记录
    log_p = Path(rec["collect_log"])
    assert log_p.exists(), rec.get("collect_log")
    assert "来自采集进程的日志" in log_p.read_text(encoding="utf-8")
    assert "来自采集进程的日志" in rec["output_tail"]
    st = [e["status"] for e in _events(loop, "c2")]
    assert "collect_rejected" in st, st


def test_collect_happy_path_records_gpu_time_and_a_data_factor(loop,
                                                              monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr(loop, "game_running_probe", lambda: False)

    class R:
        returncode = 0
        stdout = "[ring-collect] wrote x"
        stderr = ""

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        time.sleep(0.7)          # 采集是要花时间的：账本必须真的动
        out = Path(cmd[cmd.index("--out") + 1])
        _write_collection(out, roles=("front_main", "pillar_left"),
                          frames_per_role=3)
        return R()

    monkeypatch.setattr(loop, "subprocess", _FakeSP(fake_run))
    cfg = _config(tmp_path)
    rc = loop.main(["collect", "--run-id", "c3", "--config", str(cfg),
                    "--collect-roles", "front_main", "pillar_left",
                    "--collect-frames", "3"])
    assert rc == 0, "全部通过时返回 0"
    d = loop.exp_dir("c3")
    rec = json.loads(next(d.glob("collect_*.json")).read_text(encoding="utf-8"))
    assert rec["frames_total"] == 6 and rec["map_name"] == "italy"
    assert rec["gpu_minutes_today"] > 0, "采集时间要记进每日 GPU 账本"
    assert "--attach" not in seen["cmd"]
    ledger = json.loads((d / "gpu_minutes.json").read_text(encoding="utf-8"))
    assert list(ledger.values())[0]["minutes"] > 0
    prop = json.loads(Path(rec["proposal_path"]).read_text(encoding="utf-8"))
    add = prop["proposals"][0]["factor"]["add_runs"]
    assert len(add) == 2 and all(Path(p).exists() for p in add), add
    # 选样（方案 §7.5–7.7）：本次采集要给出小批选择，并且**未复核的帧不许当负例**
    sel = json.loads(Path(rec["selection"]).read_text(encoding="utf-8"))
    assert sel["n_pool"] == 6 and sel["n_items"] == 6, sel
    assert sel["n_review_only"] == 6 and sel["n_trainable"] == 0, sel
    assert all(it["route"] == "review" for it in sel["items"]), sel["items"]
    assert prop["proposals"][0]["selection"]["n_review_only"] == 6, prop
    assert "复核队列" in prop["note"], prop["note"]
    assert Path(add[0]).name == "front_main"
    assert rec["review_queue"] and Path(rec["review_queue"]).exists(), \
        "要给出'先修哪几帧'的复核队列"
    rq = json.loads(Path(rec["review_queue"]).read_text(encoding="utf-8"))
    assert rq["per_view"] >= 1, "队列要按视角各取前 N 帧"
    assert {f["view"] for f in rq["frames"]} == {"front_main", "pillar_left"}
    st = [e["status"] for e in _events(loop, "c3")]
    assert "collecting" in st and "collect_ok" in st, st
    # 收尾必须把自己起的游戏关掉（实测缺陷：一局游戏挂了 1.5 小时，
    # 后面几轮的吞吐被拖慢一个量级、推理 p95 漂到 7.6 倍）
    assert "game_after_collect" in rec, rec.keys()
    assert rec["game_pids_before"] == [], "采集前没有游戏进程"


def test_run_with_collect_tech_puts_the_collected_group_in_the_first_round(
        loop, monkeypatch, tmp_path):
    """采集 → 训练：采集产出的数据因子排第一，且两臂命令都看得见它。"""
    calls = []

    def fake_rounds(args):
        calls.append(args)
        return 1

    monkeypatch.setattr(loop, "cmd_rounds", fake_rounds)
    order = []

    def fake_collect(args, cfg, *, log):
        order.append("collect")
        d = loop.exp_dir(args.run_id)
        d.mkdir(parents=True, exist_ok=True)
        pp = d / "proposals_collect_20260925_120000.json"
        pp.write_text(json.dumps(collection_proposal(
            role_dirs=["logs/m5_seg/collect_x/front_main"],
            candidate_id="collect-20260925_120000")), encoding="utf-8")
        return {"ok": True, "proposal_path": str(pp), "frames_total": 4,
                "map_name": "italy", "out_dir": "logs/m5_seg/collect_x"}, 0

    monkeypatch.setattr(loop, "collect_once", fake_collect)
    cfg = _config(tmp_path, proposals="")
    rc = loop.main(["run", "--run-id", "c4", "--no-dry-run", "--config",
                    str(cfg), "--user-inactive"])
    assert order == ["collect"] and calls, "先采集再训练"
    assert rc == 1
    a = calls[0]
    assert a.proposals and Path(a.proposals).exists()
    merged = json.loads(Path(a.proposals).read_text(encoding="utf-8"))
    assert merged["proposals"][0]["candidate_id"] == "collect-20260925_120000"
    assert a.runs == ["logs/m5_seg/diverse_town_20260924/front_main"], \
        "配置里的 runs 不被改写；新数据只通过 add_runs 因子进来"


def test_a_rejected_collection_stops_the_round_before_training(loop,
                                                              monkeypatch,
                                                              tmp_path):
    calls = []
    monkeypatch.setattr(loop, "cmd_rounds",
                        lambda args: calls.append(args) or 0)
    monkeypatch.setattr(loop, "collect_once",
                        lambda args, cfg, *, log: ({"ok": False}, 7))
    rc = loop.main(["run", "--run-id", "c5", "--no-dry-run", "--config",
                    str(_config(tmp_path)), "--user-inactive"])
    assert rc == 7 and not calls, "采集没过闸门就不许训练"


def test_collect_runs_under_the_resolved_interpreter(loop, monkeypatch,
                                                    tmp_path):
    """采集子进程必须用"装了 beamngpy 的那个"解释器，并把选择记进产物。"""
    monkeypatch.setattr(loop, "game_running_probe", lambda: False)
    monkeypatch.setattr(loop, "collector_python",
                        lambda root, configured="": ("X:/venv/python.exe",
                                                     "project-venv"))
    monkeypatch.setattr(loop, "python_can_import", lambda py, *a, **k: True)
    seen = {}

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        return R()

    monkeypatch.setattr(loop, "subprocess", _FakeSP(fake_run))
    rc = loop.main(["collect", "--run-id", "c6", "--config",
                    str(_config(tmp_path)), "--collect-roles", "front_main",
                    "--collect-frames", "2"])
    assert rc == 7, "假子进程没产出采集目录 -> 审计拒收（但命令已经拼好）"
    assert seen["cmd"][0] == "X:/venv/python.exe", seen["cmd"][:1]
    rec = json.loads(next(loop.exp_dir("c6").glob("collect_*.json"))
                     .read_text(encoding="utf-8"))
    assert rec["collector_python"] == "X:/venv/python.exe"
    assert rec["collector_python_source"] == "project-venv"


def test_collect_refuses_an_interpreter_without_beamngpy(loop, monkeypatch,
                                                        tmp_path):
    monkeypatch.setattr(loop, "game_running_probe", lambda: False)
    monkeypatch.setattr(loop, "collector_python",
                        lambda root, configured="": ("X:/bad/python.exe", "config"))
    monkeypatch.setattr(loop, "python_can_import", lambda py, *a, **k: False)
    called = []
    monkeypatch.setattr(loop, "subprocess", _FakeSP(
        lambda *a, **k: called.append(a) or None))
    rc = loop.main(["collect", "--run-id", "c7", "--config",
                    str(_config(tmp_path))])
    assert rc == 6 and not called, "解释器不合格时不该起游戏"
    evs = _events(loop, "c7")
    assert any(e["status"] == "collect_blocked" and "beamngpy" in e["note"]
               for e in evs), evs


def test_cli_collect_flags_override_the_config(loop, tmp_path):
    cfg = loop.LoopConfig.load(_config(tmp_path, collect_frames=30,
                                       collect_roles=["front_main"],
                                       collect_step_m=2.0))
    args = type("A", (), {"collect_frames": 7, "collect_roles": None,
                          "collect_step_m": None})()
    spec = loop.spec_from(cfg, args)
    assert spec.frames == 7, "命令行给了就用命令行"
    assert spec.roles == ("front_main",), "没给就回落配置"
    assert spec.step_m == 2.0
    assert isinstance(spec.roles, tuple)
