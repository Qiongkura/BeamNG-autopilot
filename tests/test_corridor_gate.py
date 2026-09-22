"""The corridor adapter consumes real Scene measurements, not stub fields."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot import safety_monitor as sm
from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene


def _scene():
    grid = OccupancyGrid(60, 60, 0.5)
    grid.drivable[:] = 1.0
    grid.observed[:] = 1
    path = np.column_stack([np.linspace(0., 14., 29), np.zeros(29)])
    scene = Scene(pos=np.array([0., 0.]), heading=0., grid=grid,
                  lane_ref=path, strict_perception=True,
                  meta={"bev_age_s": 0.1})
    return scene, path


def _feasible(mon, scene, **kw):
    args = {"ego_speed_mps": 4., "closest_obs_m": 999., "bev_age_s": 0.1}
    args.update(kw)
    return mon._corridor_feasibility(scene, **args)


def test_corridor_gate_stays_default_off():
    assert sm.CORRIDOR_FEASIBILITY_GATE is False
    verdict = sm.SafetyVerdict()
    assert verdict.corridor_state == "unknown"
    assert verdict.corridor_evidence == {}


def test_disabled_gate_labels_legacy_boolean_evidence(monkeypatch):
    monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", False)
    scene, path = _scene()
    v = sm.SafetyMonitor().evaluate(scene, path, ego_speed_mps=4.)
    assert v.corridor_open
    assert v.corridor_state == "feasible"
    assert "structured feasibility disabled" in v.corridor_reason
    assert v.corridor_evidence == {"method": "legacy_bool", "enabled": False}


def test_open_observed_pavement_is_feasible():
    scene, _ = _scene()
    res = _feasible(sm.SafetyMonitor(), scene)
    assert res.feasible
    assert res.evidence["age_s"] == 0.1
    assert res.evidence["ego_speed_mps"] == 4.


def test_no_grid_is_unknown_and_closed():
    scene, _ = _scene()
    scene.grid = None
    res = _feasible(sm.SafetyMonitor(), scene)
    assert res.state == "unknown"
    assert not res.feasible


@pytest.mark.parametrize("field", ["ego_speed_mps", "closest_obs_m", "bev_age_s"])
@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), -1.])
def test_invalid_measurement_cannot_open_the_corridor(field, value):
    scene, _ = _scene()
    res = _feasible(sm.SafetyMonitor(), scene, **{field: value})
    assert res.state == "unknown"
    assert not res.feasible


def test_missing_explicit_inputs_are_unknown_not_stationary_and_fresh():
    scene, _ = _scene()
    res = sm.SafetyMonitor()._corridor_feasibility(scene)
    assert res.state == "unknown"
    assert not res.feasible


def test_stale_evidence_stays_closed():
    scene, _ = _scene()
    res = _feasible(sm.SafetyMonitor(), scene, bev_age_s=9.)
    assert res.state == "unknown"
    assert not res.feasible


def test_exception_stays_unknown(monkeypatch):
    primitive = importlib.import_module(
        "beamng_autopilot.planning.corridor_feasibility")

    def fail(*args, **kw):
        raise RuntimeError("grid reader failed")

    monkeypatch.setattr(primitive, "corridor_feasibility", fail)
    scene, _ = _scene()
    res = _feasible(sm.SafetyMonitor(), scene)
    assert res.state == "unknown"
    assert "failed" in res.reason


@pytest.mark.parametrize("closest", [8., 999.])
def test_adapter_passes_explicit_distance_not_scene_attributes(monkeypatch, closest):
    primitive = importlib.import_module(
        "beamng_autopilot.planning.corridor_feasibility")
    seen = {}

    def capture(scene, **kw):
        seen.update(kw)
        return primitive.CorridorFeasibility(state="feasible")

    monkeypatch.setattr(primitive, "corridor_feasibility", capture)
    scene, _ = _scene()
    _feasible(sm.SafetyMonitor(), scene, closest_obs_m=closest)
    assert seen["ego_speed_mps"] == 4.
    assert seen["required_distance_m"] == (None if closest == 999. else closest)
    assert seen["evidence"]["age_s"] == 0.1


def test_evaluate_wires_real_speed_clearance_and_bev_age(monkeypatch):
    primitive = importlib.import_module(
        "beamng_autopilot.planning.corridor_feasibility")
    seen = {}

    def capture(scene, **kw):
        seen.update(kw)
        return primitive.CorridorFeasibility(
            state="feasible", evidence=kw["evidence"])

    monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", True)
    monkeypatch.setattr(primitive, "corridor_feasibility", capture)
    scene, path = _scene()
    scene.grid.mark_obstacle_region(9., 0., 0.25, 0.25)
    assert not hasattr(scene, "speed_mps")
    assert not hasattr(scene, "closest_obs_m")
    assert not hasattr(scene, "bev_age_s")
    v = sm.SafetyMonitor(max_speed=6.).evaluate(scene, path, ego_speed_mps=5.)
    assert seen["ego_speed_mps"] == 5.
    assert seen["required_distance_m"] == pytest.approx(v.closest_obs_m)
    assert v.closest_obs_m < 999.
    assert seen["evidence"]["age_s"] == 0.1
    assert v.corridor_state == "feasible"
    assert v.corridor_evidence["evidence"]["ego_speed_mps"] == 5.


def test_evaluate_prefers_snapshot_bev_age(monkeypatch):
    monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", True)
    scene, path = _scene()
    scene.perception_snapshot = SimpleNamespace(
        head_age_s={"semantic": 0.1}, tracks=[],
        freshness=lambda: {"head_max_s": 0.1, "range_s": 0.1,
                           "bev_s": 9., "lane_s": 0.1, "max_s": 9.})
    v = sm.SafetyMonitor(max_speed=6.).evaluate(scene, path, ego_speed_mps=5.)
    assert v.corridor_state == "unknown"
    assert not v.corridor_open
    assert v.corridor_evidence["evidence"]["age_s"] == 9.


def test_evaluate_missing_speed_stays_unknown(monkeypatch):
    monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", True)
    scene, path = _scene()
    v = sm.SafetyMonitor(max_speed=6.).evaluate(scene, path)
    assert v.corridor_state == "unknown"
    assert not v.corridor_open
    assert "ego_speed_mps" in v.corridor_reason


def test_evaluate_unknown_age_cannot_relax_occupied_path(monkeypatch):
    monkeypatch.setattr(sm, "CORRIDOR_FEASIBILITY_GATE", True)
    scene, path = _scene()
    scene.meta.clear()
    scene.grid.mark_obstacle_region(12., 0., 4., 0.2)
    v = sm.SafetyMonitor(max_speed=6.).evaluate(scene, path, ego_speed_mps=5.)
    assert v.corridor_state == "unknown"
    assert not v.corridor_open
    assert v.path_occupied_frac >= sm.OCC_FRACTION_STOP
    assert v.level == "minimal_risk"
    assert v.target_speed == 0.
