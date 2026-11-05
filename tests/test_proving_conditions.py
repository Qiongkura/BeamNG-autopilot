# -*- coding: utf-8 -*-
"""试车场静止条件实验 runner：格名/门限/环境读回（fail-closed）。"""
import importlib.util
from pathlib import Path


def _mod():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mpc_test", str(root / "scripts" / "m5_proving_conditions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(ticks=12, true_n=12, cert=1.0):
    return {"summary": {
        "ticks": ticks, "true_pairs": true_n, "phantom_pairs": 0,
        "ambiguous_pairs": 0, "modes": {"pair": ticks},
        "cert_line_recall": {"both_hit_rate": cert},
    }}


def test_cell_names_and_grid_size():
    m = _mod()
    cells = list(m._cells())
    assert len(cells) == 12
    assert m._cell_name("night", "rain") == "gridmap_night_rain"


def test_score_pass(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("[fsd-drive] environment -> tod=12:00 weather=- "
                   "applied_tod=12:00:00\n", encoding="utf-8")
    row = m._score_cell(_report(), log)
    assert row["status"] == "PASS", row
    assert row["tod_after"] == "12:00:00"


def test_score_fail_on_true_rate(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("environment -> tod=12:00 weather=rainy applied_tod=12:00:00\n",
                   encoding="utf-8")
    row = m._score_cell(_report(ticks=12, true_n=5, cert=1.0), log)
    assert row["status"] == "FAIL", row


def test_score_unknown_without_readback(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("[fsd-drive] no environment line\n", encoding="utf-8")
    row = m._score_cell(_report(), log)
    assert row["status"] == "UNKNOWN", row
    assert "environment_readback_missing" in row["reasons"]


def test_score_unknown_on_too_few_ticks(tmp_path):
    m = _mod()
    log = tmp_path / "x.run.log"
    log.write_text("environment -> tod=12:00 applied_tod=12:00:00\n",
                   encoding="utf-8")
    row = m._score_cell(_report(ticks=3, true_n=3), log)
    assert row["status"] == "UNKNOWN", row
