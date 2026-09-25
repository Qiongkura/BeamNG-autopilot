"""双击入口（``scripts/m5_training_view.py``）的离线测试。

它自己不画图、不起训练，价值全在"挑对了哪一轮 + 不伪造数据 + 不写 run
目录"三件事上，所以测试也钉这三件：

1. **挑哪一轮**：显式 ``--run-id`` 优先；否则按标记文件 mtime 取最新；没有
   候选时退出码 2 并打印原因与候选，绝不退回一个空页；
2. **不混别的实验的数字**：默认只带本 run 的 ``events.jsonl``（per-run 的
   ``probes/`` 自动带上），T13 历史与评估矩阵必须显式要才带；
3. **只读**：跑完 dashboard 与 monitor 后 run 目录逐文件不变（不新增
   ``metrics.jsonl``、不碰 ``events.jsonl``）。

全部走进程内 ``main([...])``：不起子进程、不连游戏、不占端口（``serve``
与浏览器都换成本地桩），秒级完成。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m5_training_view as tv
from beamng_autopilot.experiments.events import Event, EventLog, metric

BASE = dict(candidate_id="armA_seed42", dataset_id="ds_x", config_hash="cfg1",
            seed=42)


# ---------------------------------------------------------------------------
# 造样本
# ---------------------------------------------------------------------------
def _logs_root(tmp_path: Path, monkeypatch) -> Path:
    """把 config.LOGS_DIR 指到 tmp（模块在调用时才读它）。"""
    root = tmp_path / "logs"
    monkeypatch.setattr(tv.config, "LOGS_DIR", root)
    return root


def _write_events(run_dir: Path, *, run_id: str) -> Path:
    """一条合法的相位链（queued → auditing → training），epoch 级两个点。"""
    log = EventLog(run_dir)
    log.append(Event(run_id=run_id, phase="queued", status="ok",
                     ts="2026-09-24T10:00:00Z", **BASE))
    log.append(Event(run_id=run_id, phase="auditing", status="ok",
                     ts="2026-09-24T10:00:30Z", **BASE))
    log.append(Event(run_id=run_id, phase="training", status="ok", epoch=0,
                     ts="2026-09-24T10:01:00Z",
                     metrics={"train_loss": metric(2.0)}, **BASE))
    log.append(Event(run_id=run_id, phase="training", status="ok", epoch=1,
                     ts="2026-09-24T10:02:00Z",
                     metrics={"train_loss": metric(1.4)}, **BASE))
    return log.path


def _write_metrics(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "metrics.jsonl"
    p.write_text(json.dumps({"seq": 0, "kind": "train", "t": 1.0,
                             "run_id": run_dir.name, "step": 1,
                             "loss": 2.0}) + "\n", encoding="utf-8")
    return p


def _touch_older(path: Path, *, seconds: float) -> None:
    import os
    import time as _t
    stamp = _t.time() - seconds
    os.utime(path, (stamp, stamp))


def _tree(run_dir: Path) -> dict:
    """run 目录的逐文件快照（相对路径 → 大小 + 内容）。"""
    return {str(p.relative_to(run_dir)): (p.stat().st_size,
                                          p.read_bytes())
            for p in sorted(run_dir.rglob("*")) if p.is_file()}


class _StubServer:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def no_browser(monkeypatch):
    """记录浏览器被打开的地址，不真的开浏览器。"""
    opened: list[str] = []
    monkeypatch.setattr(tv.webbrowser, "open",
                        lambda url, *a, **k: (opened.append(url), True)[1])
    return opened


# ---------------------------------------------------------------------------
# 挑哪一轮
# ---------------------------------------------------------------------------
def test_newest_run_is_picked_by_marker_mtime(tmp_path, monkeypatch):
    logs = _logs_root(tmp_path, monkeypatch)
    old = logs / "experiments" / "run_old"
    new = logs / "experiments" / "run_new"
    _write_events(old, run_id="run_old")
    _write_events(new, run_id="run_new")
    _touch_older(old / "events.jsonl", seconds=600)

    found = tv.candidates(logs / "experiments", tv.MARKER_DASHBOARD)
    assert [c[1] for c in found] == ["run_new", "run_old"]

    run, reason = tv.resolve_run(logs / "experiments", tv.MARKER_DASHBOARD,
                                 None)
    assert reason == ""
    assert run is not None and run[0] == "run_new"


def test_explicit_run_id_wins_over_mtime(tmp_path, monkeypatch):
    logs = _logs_root(tmp_path, monkeypatch)
    old = logs / "experiments" / "run_old"
    new = logs / "experiments" / "run_new"
    _write_events(old, run_id="run_old")
    _write_events(new, run_id="run_new")
    _touch_older(old / "events.jsonl", seconds=600)

    run, _ = tv.resolve_run(logs / "experiments", tv.MARKER_DASHBOARD,
                            "run_old")
    assert run is not None and run[0] == "run_old"


def test_explicit_run_without_the_marker_says_which_file_is_missing(
        tmp_path, monkeypatch):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_metrics(logs / "experiments" / "only_metrics")

    run, reason = tv.resolve_run(logs / "experiments", tv.MARKER_DASHBOARD,
                                 "only_metrics")
    assert run is None
    assert "events.jsonl" in reason

    run, reason = tv.resolve_run(logs / "experiments", tv.MARKER_DASHBOARD,
                                 "nope")
    assert run is None
    assert "不存在" in reason


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
def test_dashboard_renders_the_selected_run_and_opens_it(
        tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    run_dir = logs / "experiments" / "run_a"
    _write_events(run_dir, run_id="run_a")

    assert tv.main(["dashboard"]) == 0
    out = logs / "experiments" / tv.DEFAULT_DASHBOARD_OUT
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # 看板默认视图 = 监控页同款卡片网格（两列卡片 + 状态栏 + 统计行）；
    # 文档视图（含"运行总览/学习曲线"大图）现在要显式 --view full。
    assert 'class="card' in text and "<main>" in text and "run_a" in text
    assert no_browser == [out.resolve().as_uri()]


def test_dashboard_default_does_not_blend_other_experiments(
        tmp_path, monkeypatch, no_browser):
    """默认 argv 里只有本 run 的事件流；T13 历史要显式要。"""
    logs = _logs_root(tmp_path, monkeypatch)
    run_dir = logs / "experiments" / "run_a"
    _write_events(run_dir, run_id="run_a")

    seen: list[list[str]] = []
    monkeypatch.setattr(tv.m5_seg_dashboard, "main",
                        lambda argv: (seen.append(list(argv)), 0)[1])

    assert tv.main(["dashboard", "--no-open"]) == 0
    assert seen[0] == ["render", "--out",
                       str(logs / "experiments" / tv.DEFAULT_DASHBOARD_OUT),
                       "--events", str(run_dir / "events.jsonl")]

    assert tv.main(["dashboard", "--no-open", "--t13-import"]) == 0
    assert seen[1][-1] == "--t13-import"


def test_dashboard_auto_attaches_only_this_runs_probes(
        tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    run_dir = logs / "experiments" / "run_a"
    _write_events(run_dir, run_id="run_a")
    (run_dir / "probes").mkdir()

    seen: list[list[str]] = []
    monkeypatch.setattr(tv.m5_seg_dashboard, "main",
                        lambda argv: (seen.append(list(argv)), 0)[1])
    assert tv.main(["dashboard", "--no-open"]) == 0
    assert "--probes" in seen[0]
    assert seen[0][seen[0].index("--probes") + 1] == str(run_dir / "probes")

    # 没有 probes/ 的 run 不该凭空带一个路径
    other = logs / "experiments" / "run_b"
    _write_events(other, run_id="run_b")
    assert tv.main(["dashboard", "--no-open", "--run-id", "run_b"]) == 0
    assert "--probes" not in seen[1]


def test_dashboard_without_any_run_exits_2_and_lists_nothing(
        tmp_path, monkeypatch, capsys, no_browser):
    _logs_root(tmp_path, monkeypatch)
    assert tv.main(["dashboard"]) == tv.EXIT_NO_DATA
    out = capsys.readouterr().out
    assert "没有可用数据" in out
    assert "events.jsonl" in out
    assert "（无）" in out
    assert no_browser == []


def test_dashboard_render_failure_is_reported_not_hidden(
        tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_events(logs / "experiments" / "run_a", run_id="run_a")
    monkeypatch.setattr(tv.m5_seg_dashboard, "main", lambda argv: 1)
    assert tv.main(["dashboard"]) == 1
    assert no_browser == []


def test_no_open_never_launches_a_browser(tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_events(logs / "experiments" / "run_a", run_id="run_a")
    assert tv.main(["dashboard", "--no-open"]) == 0
    assert no_browser == []


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------
def test_monitor_without_metrics_explains_metrics_run(
        tmp_path, monkeypatch, capsys, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_events(logs / "experiments" / "run_a", run_id="run_a")

    assert tv.main(["monitor"]) == tv.EXIT_NO_DATA
    out = capsys.readouterr().out
    assert "metrics.jsonl" in out
    assert "--metrics-run" in out
    assert no_browser == []


def test_monitor_serves_the_newest_run_with_metrics(
        tmp_path, monkeypatch, no_browser, capsys):
    logs = _logs_root(tmp_path, monkeypatch)
    exp = logs / "experiments"
    _write_events(exp / "run_a", run_id="run_a")
    _write_metrics(exp / "run_a")
    _write_metrics(exp / "run_b")
    _touch_older(exp / "run_a" / "metrics.jsonl", seconds=600)

    monkeypatch.setattr(tv, "_serving", lambda url: None)
    srv = _StubServer()
    seen: list[dict] = []

    def fake_serve(run_dir, **kw):
        seen.append({"run_dir": Path(run_dir), **kw})
        return srv, None, f"http://{kw['host']}:{kw['port']}/"

    monkeypatch.setattr(tv.monitor_server, "serve", fake_serve)
    monkeypatch.setattr(tv.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt))

    assert tv.main(["monitor", "--port", "8799"]) == 0
    assert seen[0]["run_dir"] == exp / "run_b"
    assert seen[0]["run_id"] == "run_b"
    assert seen[0]["port"] == 8799
    assert no_browser == ["http://127.0.0.1:8799/"]
    assert srv.shutdown_called
    assert "Ctrl+C 退出" in capsys.readouterr().out


def test_monitor_reuses_a_service_already_serving_the_same_run(
        tmp_path, monkeypatch, no_browser, capsys):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_metrics(logs / "experiments" / "run_b")
    monkeypatch.setattr(tv, "_serving", lambda url: "run_b")
    monkeypatch.setattr(tv.monitor_server, "serve",
                        lambda *a, **k: pytest.fail("不该重复起服务"))

    assert tv.main(["monitor", "--port", "8799"]) == 0
    assert no_browser == ["http://127.0.0.1:8799/"]
    assert "已经在看 run_b" in capsys.readouterr().out


def test_monitor_never_reuses_a_service_showing_another_run(
        tmp_path, monkeypatch, no_browser, capsys):
    """别的 run 占着端口：不顺手打开，换系统分配端口并说明原因。"""
    logs = _logs_root(tmp_path, monkeypatch)
    _write_metrics(logs / "experiments" / "run_b")
    monkeypatch.setattr(tv, "_serving", lambda url: "run_old")
    srv = _StubServer()
    ports: list[int] = []

    def fake_serve(run_dir, **kw):
        ports.append(kw["port"])
        return srv, None, f"http://{kw['host']}:{kw['port']}/"

    monkeypatch.setattr(tv.monitor_server, "serve", fake_serve)
    monkeypatch.setattr(tv.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt))

    assert tv.main(["monitor", "--port", "8799"]) == 0
    assert ports == [0]          # 明知被别的 run 占着，就不去试那个端口
    out = capsys.readouterr().out
    assert "另一个 run（run_old）" in out
    assert "改用" in out
    assert no_browser == ["http://127.0.0.1:0/"]


def test_monitor_busy_port_falls_back_to_a_free_one(
        tmp_path, monkeypatch, capsys, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_metrics(logs / "experiments" / "run_b")
    monkeypatch.setattr(tv, "_serving", lambda url: None)
    srv = _StubServer()
    ports: list[int] = []

    def flaky_serve(run_dir, **kw):
        ports.append(kw["port"])
        if kw["port"] == 8799:
            raise OSError("[WinError 10048] 通常每个套接字地址只允许使用一次")
        return srv, None, f"http://{kw['host']}:{kw['port']}/"

    monkeypatch.setattr(tv.monitor_server, "serve", flaky_serve)
    monkeypatch.setattr(tv.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt))

    assert tv.main(["monitor", "--port", "8799"]) == 0
    assert ports == [8799, 0]
    assert "起不来" in capsys.readouterr().out


def test_monitor_exits_3_when_even_a_free_port_fails(
        tmp_path, monkeypatch, capsys, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    _write_metrics(logs / "experiments" / "run_b")
    monkeypatch.setattr(tv, "_serving", lambda url: None)

    def boom(*a, **k):
        raise OSError("no socket for you")

    monkeypatch.setattr(tv.monitor_server, "serve", boom)
    assert tv.main(["monitor", "--port", "8799"]) == tv.EXIT_PORT_BUSY
    out = capsys.readouterr().out
    assert "系统分配端口也失败" in out
    assert no_browser == []


def test_serving_only_trusts_our_own_state_payload(monkeypatch):
    """别人占着这个端口时不能当成"已有监控服务"。"""
    import io

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def reply(body: bytes):
        return lambda url, timeout=0: _Resp(body)

    monkeypatch.setattr(tv.urllib.request, "urlopen",
                        reply(b'{"ok": true}'))          # 别人的 /health
    assert tv._serving("http://127.0.0.1:8799/") is None

    monkeypatch.setattr(tv.urllib.request, "urlopen",
                        reply(b'{"run_id": "run_b", "status": "running"}'))
    assert tv._serving("http://127.0.0.1:8799/") == "run_b"

    def refused(url, timeout=0):
        raise OSError("connection refused")

    monkeypatch.setattr(tv.urllib.request, "urlopen", refused)
    assert tv._serving("http://127.0.0.1:8799/") is None


# ---------------------------------------------------------------------------
# history：全仓台账（不挑某一轮）
# ---------------------------------------------------------------------------
def test_history_renders_the_whole_logs_tree_and_opens_it(
        tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    run_a = logs / "experiments" / "run_a"
    _write_events(run_a, run_id="run_a")
    (run_a / "train_hist.json").write_text(
        json.dumps({"epoch": [0], "train_loss": [2.1], "val_acc": [0.9],
                    "val_miou": [0.7], "val_line_iou": [0.2]}), encoding="utf-8")
    run_b = logs / "m5_seg" / "seg_model_v13b"
    run_b.mkdir(parents=True)
    (run_b / "train_hist.json").write_text(
        json.dumps({"epoch": [0], "train_loss": [2.0], "val_acc": [0.9],
                    "val_miou": [0.7], "val_line_iou": [0.3]}), encoding="utf-8")

    assert tv.main(["history"]) == 0
    out = logs / "experiments" / tv.DEFAULT_HISTORY_OUT
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "训练台账" in text and "本页不做排行" in text
    # 两种目录布局的训练 run 都在同一页里
    assert "run_a" in text and "seg_model_v13b" in text
    assert no_browser == [out.resolve().as_uri()]


def test_history_needs_no_run_and_honours_no_open(tmp_path, monkeypatch,
                                                 no_browser):
    """空 logs 也要出页（写 0 个 run），不是报错退出。"""
    logs = _logs_root(tmp_path, monkeypatch)
    logs.mkdir(parents=True)
    assert tv.main(["history", "--no-open"]) == 0
    text = (logs / "experiments" / tv.DEFAULT_HISTORY_OUT).read_text(
        encoding="utf-8")
    assert "训练台账" in text
    # 0 个 run 是"没有"，页面上就写 0（这是计数，不是缺列的指标）
    assert 'train_hist.json）</th><td class="num">0</td>' in text
    assert no_browser == []


# ---------------------------------------------------------------------------
# 只读：跑完两个模式，run 目录逐文件不变
# ---------------------------------------------------------------------------
def test_both_modes_leave_the_run_directory_untouched(
        tmp_path, monkeypatch, no_browser):
    logs = _logs_root(tmp_path, monkeypatch)
    run_dir = logs / "experiments" / "run_a"
    _write_events(run_dir, run_id="run_a")
    _write_metrics(run_dir)
    (run_dir / "probes").mkdir()
    (run_dir / "probes" / "manifest.json").write_text("{}", encoding="utf-8")

    before = _tree(run_dir)

    assert tv.main(["dashboard", "--no-open"]) == 0

    monkeypatch.setattr(tv, "_serving", lambda url: None)
    srv = _StubServer()
    monkeypatch.setattr(tv.monitor_server, "serve",
                        lambda run_dir, **kw: (srv, None, "http://x/"))
    monkeypatch.setattr(tv.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt))
    assert tv.main(["monitor", "--no-open"]) == 0

    assert _tree(run_dir) == before
    assert not (run_dir / "dashboard_latest.html").exists()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def test_list_shows_candidates_for_both_markers(tmp_path, monkeypatch,
                                               capsys):
    logs = _logs_root(tmp_path, monkeypatch)
    exp = logs / "experiments"
    _write_events(exp / "run_a", run_id="run_a")
    _write_metrics(exp / "run_b")

    assert tv.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "run_a" in out and "run_b" in out
    assert "events.jsonl" in out and "metrics.jsonl" in out
