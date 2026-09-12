"""Presence scoring/agreement regressions: pure math, no model, no game."""

from __future__ import annotations

from beamng_autopilot.labeling.presence import (
    line_fraction, load_label_records, presence_agreement)


def test_line_fraction():
    assert line_fraction([[0, 0], [0, 0]]) == 0.0
    assert line_fraction([[1, 0], [0, 1]]) == 0.5
    assert line_fraction([]) == 0.0


def test_agreement_perfect_separation():
    labels = {"a.png": 1, "b.png": 1, "c.png": 0, "d.png": 0}
    scores = {"a.png": 0.9, "b.png": 0.5, "c.png": 0.1, "d.png": 0.0}
    rep = presence_agreement(labels, scores)
    assert rep["auc"] == 1.0
    assert rep["accuracy"] == 1.0
    assert rep["recall"] == 1.0 and rep["precision"] == 1.0
    assert rep["disagreements"] == []
    assert (rep["tp"], rep["fp"], rep["tn"], rep["fn"]) == (2, 0, 2, 0)


def test_agreement_inverted_and_ties():
    # 分数完全反向 -> AUC 0；并列分数取平均秩
    labels = {"a.png": 1, "b.png": 0, "c.png": 1, "d.png": 0}
    scores = {"a.png": 0.1, "b.png": 0.9, "c.png": 0.1, "d.png": 0.9}
    rep = presence_agreement(labels, scores)
    assert rep["auc"] == 0.0

    labels = {"a.png": 1, "b.png": 0}
    rep = presence_agreement(labels, {"a.png": 0.5, "b.png": 0.5})
    assert rep["auc"] == 0.5          # 全并列 => 等同随机
    assert rep["n"] == 2 and rep["pos"] == 1


def test_agreement_disagreements_sorted_and_skips_missing():
    labels = {"a.png": 1, "b.png": 1, "c.png": 0, "ghost.png": 0}
    scores = {"a.png": 0.0, "b.png": 0.8, "c.png": 0.9}   # c 高分误报
    rep = presence_agreement(labels, scores)
    assert rep["n"] == 3              # ghost 不在分数里，被跳过
    assert rep["fp"] == 1 and rep["fn"] == 0
    assert rep["disagreements"][0]["key"] == "c.png"

    # 两个误报按 |score - label| 降序：更离谱的排前面
    labels = {"x1.png": 1, "x2.png": 1, "y1.png": 0, "y2.png": 0}
    scores = {"x1.png": 0.0, "x2.png": 0.05, "y1.png": 1.0, "y2.png": 0.5}
    rep = presence_agreement(labels, scores)
    keys = [d["key"] for d in rep["disagreements"]]
    assert keys == ["y1.png", "y2.png"]


def test_load_label_records_skips_bad_lines(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(
        '{"path": "C:/a.png", "has_line": 1}\n'
        'not json\n'
        '{"path": "C:/b.png"}\n'
        '{"path": "C:/c.png", "has_line": 0}\n', encoding="utf-8")
    recs = load_label_records(p)
    assert [(r["path"], r["has_line"]) for r in recs] == [
        ("C:/a.png", 1), ("C:/c.png", 0)]
