"""T14 训练台账（``scripts/m5_training_history.py``）的离线测试。

台账的价值全在"它写出来的每个数字都能追到产物、缺的东西不被写成 0、不替
产物下结论"这三点上，所以测试钉这些：

1. **四种目录布局都要被扫到**（``m5_seg/<run>``、``m5_seg/<exp>/<run>``、
   ``experiments/<run>``、``experiments/<run>/<arm>/<seed>``），家族/子路径
   分组正确，排序确定（mtime 倒序）；
2. **缺列不写 0**：缺键、末轮 ``null``、``NaN``/``inf`` 都渲染成"缺列"，
   而**真记录到的 0 必须写 0**；缺列原因只读产物自报的字段
   （``line_ignored_frames`` / 有无该列），不推测；
3. **不替产物下结论**：完成状态只看 ``best.pt`` / ``checkpoint_last.pt``；
   最终集只有账本里**带结果**的条目才算确认，访问控制记录不许读成"通过"。

再加只读断言（跑完 render 后 tmp 树逐文件不变）与自包含断言（HTML 不引用
任何外部资源）。全部走进程内 ``main([...])``，秒级完成。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m5_training_history as hist


# ---------------------------------------------------------------------------
# 造样本
# ---------------------------------------------------------------------------
def _write_hist(dirpath: Path, blob: dict) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / "train_hist.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


def _full(*, n=3, line=0.30) -> dict:
    return {"epoch": list(range(n)),
            "train_loss": [2.0 - 0.2 * i for i in range(n)],
            "val_acc": [0.95] * n,
            "val_miou": [0.70 + 0.01 * i for i in range(n)],
            "val_line_iou": [line - 0.05 * i for i in range(n)]}


def _touch_older(path: Path, *, seconds: float) -> None:
    import os
    import time as _t
    stamp = _t.time() - seconds
    os.utime(path, (stamp, stamp))


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _decision(run_dir: Path, name: str = "decision_cand-A-r0.json",
              *, decision: str = "rejected", research_only: bool = False,
              candidate: str = "cand-A") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / name
    p.write_text(json.dumps({
        "candidate_id": candidate,
        "research_only": research_only,
        "pairings": {"line_iou": {"mean_delta": 0.079, "verdict": "candidate_better",
                                  "n": 5, "seeds_needed_for_effect": 2}},
        "hard_gate_violations": ["candidate_identity_rate: 0.14 < 0.6",
                                 "offroad_false_ratio: 0.11 > 0.1"],
        "decision": {"decision": decision,
                     "reasons": ["candidate_identity_rate 0.1417 < 0.6"]},
    }), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 扫描四种布局
# ---------------------------------------------------------------------------
def test_scans_all_four_layouts_and_groups_families(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "m5_seg" / "seg_model_v13b", _full())                  # d1
    _write_hist(logs / "m5_seg" / "t13" / "armA_seed42", _full())             # d2
    _write_hist(logs / "experiments" / "run_a", _full())                      # d1
    _write_hist(logs / "experiments" / "run_a" / "baseline" / "seed42",
                _full())                                                      # d3

    rows = hist.scan_runs(logs)
    got = {(r.root, r.family, r.sub) for r in rows}
    assert got == {
        ("m5_seg", "seg_model_v13b", ""),
        ("m5_seg", "t13", "armA_seed42"),
        ("experiments", "run_a", ""),
        ("experiments", "run_a", "baseline/seed42"),
    }
    # 下钻 run-id：只有 experiments 顶层的 run 目录才有
    by_rel = {r.rel: r for r in rows}
    key = next(k for k in by_rel if k.startswith("experiments") and k.endswith(
        "baseline\\seed42\\train_hist.json") or "baseline" in k)
    assert by_rel[key].drill_run_id == "run_a"
    assert by_rel[str(Path("m5_seg") / "t13" / "armA_seed42" / "train_hist.json")
                  ].drill_run_id == ""


def test_runs_are_sorted_newest_first(tmp_path):
    logs = tmp_path / "logs"
    old = _write_hist(logs / "experiments" / "run_old", _full())
    new = _write_hist(logs / "experiments" / "run_new", _full())
    _touch_older(old, seconds=900)
    rows = hist.scan_runs(logs)
    assert [r.family for r in rows] == ["run_new", "run_old"]


def test_missing_roots_are_not_an_error(tmp_path):
    assert hist.scan_runs(tmp_path / "logs") == []
    assert hist.scan_decisions(tmp_path / "logs") == []


# ---------------------------------------------------------------------------
# 缺列：不写 0，但真 0 要写 0
# ---------------------------------------------------------------------------
def test_absent_column_is_missing_not_zero(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "m5_seg" / "old_model",
                {"epoch": [0, 1], "train_loss": [2.0, 1.5],
                 "val_acc": [0.9, 0.9], "val_miou": [0.7, 0.71]})
    (row,) = hist.scan_runs(logs)
    assert row.last["val_line_iou"] is None
    assert row.best_dev is None
    assert any("无该列" in m for m in row.missing_keys)
    assert "早于标线通道" in row.missing_reason
    assert row.missing_kind == "产物没有 val_line_iou 列"


def test_null_and_non_finite_last_epoch_are_missing(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_null",
                {"epoch": [0, 1], "train_loss": [2.0, None],
                 "val_miou": [0.7, float("nan")], "val_line_iou": [0.3, None]})
    (row,) = hist.scan_runs(logs)
    assert row.last["train_loss"] is None
    assert row.last["val_miou"] is None
    assert row.last["val_line_iou"] is None
    # 最佳值是序列里有限的那个，不受末轮 null 影响
    assert row.best_dev == pytest.approx(0.3)


def test_a_recorded_zero_stays_zero(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_zero",
                {"epoch": [0, 1], "train_loss": [2.0, 1.9], "val_acc": [0.9, 0.9],
                 "val_miou": [0.7, 0.7], "val_line_iou": [0.0, 0.0]})
    ctx = hist.build_context(logs)
    (row,) = ctx["runs"]
    assert row.last["val_line_iou"] == 0.0
    assert row.missing_keys == ()
    text = hist.render_html(ctx)
    assert "0.0000" in text          # 真 0 就写 0
    assert ctx["scan"]["n_missing"] == 0


def test_line_ignored_frames_explains_the_missing_column(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_masked",
                {"epoch": [0, 1], "train_loss": [2.0, 1.8], "val_acc": [0.9, 0.9],
                 "val_miou": [0.7, 0.7], "val_line_iou": [0.2, None],
                 "line_ignored_frames": [600, 1200]})
    (row,) = hist.scan_runs(logs)
    assert row.line_ignored == 1200
    assert row.missing_reason == "line 通道被屏蔽（line_ignored_frames 末值 1200）"
    assert row.missing_kind == "line 通道被屏蔽"

    ctx = hist.build_context(logs)
    text = hist.render_html(ctx)
    assert "line 通道被屏蔽 ×1" in text       # 汇总行
    assert "line_ignored_frames 末值 1200" in text   # 单元格提示


def test_best_dev_takes_the_best_epoch_and_its_index(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_best",
                {"epoch": [0, 1, 2], "train_loss": [2.0, 1.8, 1.6],
                 "val_acc": [0.9] * 3, "val_miou": [0.7] * 3,
                 "val_line_iou": [0.20, 0.55, 0.31]})
    (row,) = hist.scan_runs(logs)
    assert row.best_dev == pytest.approx(0.55)
    assert row.best_dev_epoch == 1


# ---------------------------------------------------------------------------
# 状态只看产物
# ---------------------------------------------------------------------------
def test_status_comes_from_the_weights_on_disk(tmp_path):
    logs = tmp_path / "logs"
    a = logs / "experiments" / "run_done"
    b = logs / "experiments" / "run_half"
    c = logs / "experiments" / "run_bare"
    for d in (a, b, c):
        _write_hist(d, _full())
    (a / "best.pt").write_bytes(b"x")
    (a / "checkpoint_last.pt").write_bytes(b"x")
    (b / "checkpoint_last.pt").write_bytes(b"x")
    c.joinpath("train_hist.json").unlink()          # 空曲线也不许崩
    _write_hist(c, {"epoch": [], "train_loss": []})

    status = {r.family: r.status for r in hist.scan_runs(logs)}
    assert status["run_done"] == "完成（有 best.pt）"
    assert status["run_half"] == "中断（只到 last）"
    assert status["run_bare"] == "空曲线"


# ---------------------------------------------------------------------------
# 决策台账
# ---------------------------------------------------------------------------
def test_decisions_are_scanned_with_reasons_and_pairings(tmp_path):
    logs = tmp_path / "logs"
    _decision(logs / "experiments" / "t14_rounds_real")
    _decision(logs / "experiments" / "t14_rounds_real" / "nested",
              name="decision_cand-B-r1.json", decision="needs_evidence",
              research_only=True, candidate="cand-B")

    dec = hist.scan_decisions(logs)
    assert {d.candidate for d in dec} == {"cand-A", "cand-B"}
    by_cand = {d.candidate: d for d in dec}
    assert by_cand["cand-A"].decision == "rejected"
    assert by_cand["cand-A"].run == "t14_rounds_real"
    assert by_cand["cand-A"].gates == ("candidate_identity_rate: 0.14 < 0.6",
                                       "offroad_false_ratio: 0.11 > 0.1")
    assert by_cand["cand-A"].pairings == (("line_iou", 0.079,
                                           "candidate_better", 5),)
    assert by_cand["cand-B"].research_only is True

    ctx = hist.build_context(logs)
    assert ctx["scan"]["decisions"] == {"rejected": 1, "needs_evidence": 1}
    text = hist.render_html(ctx)
    assert "硬门判定台账" in text and "candidate_better" in text


def test_decision_without_pairings_says_missing(tmp_path):
    logs = tmp_path / "logs"
    d = logs / "experiments" / "run_x"
    d.mkdir(parents=True)
    (d / "decision_c-r0.json").write_text(
        json.dumps({"candidate_id": "c", "decision": {"decision": "rejected",
                                                      "reasons": []}}),
        encoding="utf-8")
    ctx = hist.build_context(logs)
    assert hist.MISSING in hist.render_html(ctx)


# ---------------------------------------------------------------------------
# 最终集：只有带结果的条目才算确认
# ---------------------------------------------------------------------------
def _ledger(logs: Path, entries: list) -> Path:
    p = logs / "experiments" / "run_seal" / "seal" / hist.FINAL_LEDGER
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return p


def _entry(**kw) -> dict:
    base = {"t": "2026-09-25T18:56:55", "protocol_hash": "e8f5", "candidate_id": "c",
            "caller": "m5_final_confirm", "purpose": "final_confirm",
            "result": "", "allowed": True, "reasons": [], "seal_digest": "67ed"}
    base.update(kw)
    return base


def test_final_set_records_without_a_result_are_not_a_confirmation(tmp_path):
    logs = tmp_path / "logs"
    _ledger(logs, [_entry(),
                   _entry(allowed=False,
                          reasons=["this final set is already consumed"])])
    ctx = hist.build_context(logs)
    assert ctx["final_set"]["n_entries"] == 2
    assert ctx["final_set"]["n_with_result"] == 0
    text = hist.render_html(ctx)
    assert "没有一条带确认结果" in text
    assert "不要把" in text and "allowed" in text


def test_final_set_with_a_result_is_reported_as_such(tmp_path):
    logs = tmp_path / "logs"
    _ledger(logs, [_entry(result="line_iou 0.62 (frozen set, one shot)")])
    ctx = hist.build_context(logs)
    assert ctx["final_set"]["n_with_result"] == 1
    assert "带结果的确认" in hist.render_html(ctx)


def test_no_ledger_says_untested(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    ctx = hist.build_context(logs)
    assert ctx["final_set"]["ledgers"] == []
    assert "未测" in hist.render_html(ctx)


# ---------------------------------------------------------------------------
# 生产权重
# ---------------------------------------------------------------------------
def test_production_model_reports_hash_and_size(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setattr(hist.config, "LOGS_DIR", logs)
    p = logs / "m5_seg" / "seg_model" / "best.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"weights")

    prod = hist.production_model()
    assert prod["path"].endswith("best.pt")
    assert prod["sha16"] == __import__("hashlib").sha256(b"weights").hexdigest()[:16]
    assert prod["size_mb"] == 0.0
    assert prod["experts"] == []


def test_production_model_absent_is_untested_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(hist.config, "LOGS_DIR", tmp_path / "logs")
    prod = hist.production_model()
    assert prod["path"] == ""
    assert "None" in prod["error"]
    ctx = {"scan": {"logs_dir": "", "n_runs": 0, "n_families": 0, "n_decisions": 0,
                    "n_missing": 0, "epochs": 0, "n_by_root": {}, "decisions": {},
                    "missing_by_reason": {}},
           "production": prod,
           "final_set": {"ledgers": [], "n_entries": 0, "n_with_result": 0,
                         "error": ""},
           "families": [], "runs": [], "decisions": []}
    text = hist.render_html(ctx)
    assert "未测" in text and "0.0 MB" not in text


# ---------------------------------------------------------------------------
# 渲染纪律与只读
# ---------------------------------------------------------------------------
def test_html_is_self_contained_and_says_it_is_not_a_ranking(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_a", _full())
    text = hist.render_html(hist.build_context(logs))
    assert "http://" not in text and "https://" not in text
    assert "<script src" not in text
    assert "本页不做排行" in text
    assert hist.LEVEL_DEV in text


def test_render_writes_only_the_requested_files(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    run = logs / "experiments" / "run_a"
    _write_hist(run, _full())
    (run / "best.pt").write_bytes(b"x")
    _write_hist(logs / "m5_seg" / "t13" / "armA_seed42", _full())
    monkeypatch.setattr(hist.config, "LOGS_DIR", logs)
    before = _tree(logs)

    out = tmp_path / "out" / "history.html"
    js = tmp_path / "out" / "history.json"
    assert hist.main(["render", "--out", str(out), "--json", str(js)]) == 0
    assert out.is_file() and js.is_file()
    assert "run 2 个" in capsys.readouterr().out

    # run 目录/产物一个字节都没动
    assert _tree(logs) == before
    # JSON 导出的是同一份 context
    blob = json.loads(js.read_text(encoding="utf-8"))
    assert len(blob["runs"]) == 2
    by_family = {r["family"]: r for r in blob["runs"]}
    assert by_family["run_a"]["status"] == "完成（有 best.pt）"
    assert by_family["t13"]["status"] == "无权重产物"
    assert set(blob) >= {"generated_at", "scan", "production", "final_set",
                         "families", "runs", "decisions"}


def test_list_prints_counts_and_decisions(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_a", _full())
    _decision(logs / "experiments" / "run_a")
    monkeypatch.setattr(hist.config, "LOGS_DIR", logs)

    assert hist.main(["list", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "1 个训练 run" in out
    assert hist.LEVEL_DEV in out
    assert "'rejected': 1" in out


def test_html_filter_hook_has_a_key_per_row(tmp_path):
    logs = tmp_path / "logs"
    _write_hist(logs / "experiments" / "run_alpha", _full())
    _write_hist(logs / "m5_seg" / "seg_model_v2", _full())
    text = hist.render_html(hist.build_context(logs))
    assert text.count('data-k="') == 2
    assert 'run_alpha' in text and "seg_model_v2" in text
