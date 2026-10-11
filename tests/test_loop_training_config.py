"""Background training recipe survives loading, hashing and planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beamng_autopilot.experiments.controller import LoopConfig, plan_once


def test_training_recipe_round_trip_and_hash(tmp_path):
    cfg = LoopConfig(total_steps=120, init="weights/shared.pt")
    loaded = LoopConfig.load(cfg.save(tmp_path / "loop.json"))
    assert loaded.total_steps == 120 and loaded.init == "weights/shared.pt"
    assert loaded.hash() == cfg.hash()
    assert loaded.hash() != LoopConfig(total_steps=121, init=cfg.init).hash()
    assert loaded.hash() != LoopConfig(total_steps=120, init="weights/other.pt").hash()


@pytest.mark.parametrize("steps", [-1, True, 1.5, "120"])
def test_invalid_step_budget_is_rejected(steps):
    with pytest.raises(ValueError, match="total_steps"):
        LoopConfig(total_steps=steps)


@pytest.mark.parametrize("init", ["", " ", 42])
def test_invalid_initialization_is_rejected(init):
    with pytest.raises(ValueError, match="init"):
        LoopConfig(init=init)


def test_misspelled_config_cannot_silently_disable_training_budget(tmp_path):
    path = tmp_path / "loop.json"
    path.write_text(json.dumps({"total_step": 120}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown LoopConfig fields: total_step"):
        LoopConfig.load(path)


def test_dry_run_command_contains_configured_budget_and_init(tmp_path):
    cfg = LoopConfig(total_steps=120, init="weights/shared.pt")
    plan = plan_once(cfg, run_id="fixture", candidate_id="candidate",
                     dataset_id="data", python="python",
                     script_dir=Path("scripts"), seed=42)
    cmd = next(a.cmd for a in plan.actions if a.name == "train_candidate")
    assert cmd[cmd.index("--total-steps") + 1] == "120"
    assert cmd[cmd.index("--init") + 1] == "weights/shared.pt"
