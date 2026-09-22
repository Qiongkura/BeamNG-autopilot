"""The planner must pay for candidates whose BODY sweeps into boxes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from beamng_autopilot.occupancy import OccupancyGrid
from beamng_autopilot.planning import Scene
from beamng_autopilot.planning.constraints import (
    Constraints,
    _path_body_collision,
)


def _scene_with_wall(gap_clear: bool) -> Scene:
    """A straight road; an obstacle wall on the right from x=6 onward.

    ``gap_clear=False`` puts the wall face 0.4 m from the straight path
    (the body sweeps into it); ``True`` leaves 1.6 m of clearance.
    """
    grid = OccupancyGrid(n_rows=60, n_cols=60, res=0.5)
    # Straight path along world y=0 (ego at origin, +x forward).  Cells:
    # col c covers ego-ey in [15 - (c+1)*0.5, 15 - c*0.5]; world y maps to
    # ego ey = -y, so the body corner at world y=-0.9 lands in col 31.
    # tight: wall cells cover ey <= -1.0 (face 0.1 m from the corner);
    # clear: wall cells cover ey <= -2.0.
    c0 = 31 if not gap_clear else 34
    for r in range(60):
        ex = grid.extent - (r + 0.5) * grid.res      # forward metres
        if 5.0 <= ex <= 15.0:
            for c in (c0, c0 + 1):
                grid.obstacle[r, c] = 1
    return Scene(pos=np.array([0.0, 0.0, 0.0]), heading=0.0,
                 grid=grid, route=None, lane_ref=None,
                 strict_perception=False)


def _straight_path():
    return np.column_stack([np.linspace(0.0, 20.0, 41),
                            np.zeros(41)])


def test_body_collision_detects_the_sweep():
    scene = _scene_with_wall(gap_clear=False)
    bad, tot, near = _path_body_collision(scene, _straight_path())
    assert tot > 0 and bad > 0          # the body really enters the wall
    scene2 = _scene_with_wall(gap_clear=True)
    bad2, tot2, near2 = _path_body_collision(scene2, _straight_path())
    assert bad2 == 0 and near2 == 0     # clear wall never triggers


def test_cost_prefers_the_candidate_with_body_clearance():
    tight = Constraints(w_lane_align=0.0, w_curvature=0.0,
                        w_progress=0.0, w_collision=5.0)
    scene_tight = _scene_with_wall(gap_clear=False)
    scene_clear = _scene_with_wall(gap_clear=True)
    cand = SimpleNamespace(path=_straight_path(),
                           meta={"kind": "arc"})
    cost_tight, feas_tight = tight.score(scene_tight, cand)
    cost_clear, feas_clear = tight.score(scene_clear, cand)
    assert feas_tight and feas_clear    # soft cost, not a refusal
    assert cost_clear < cost_tight      # the clean one wins


def test_strict_with_an_empty_drivable_layer_is_infeasible():
    """Live 2026-09-19 14:31: the road mask vanished on a dirt shoulder
    (40 frames off the road on lane=sensor) because the off-drivable gate
    silently skips when the drivable layer carries no evidence.  Strict
    mode must fail closed: no road evidence -> no candidate."""
    scene = _scene_with_wall(gap_clear=True)
    scene.grid.drivable[:] = 0.0          # the mask vanished
    scene.strict_perception = True
    cand = SimpleNamespace(path=_straight_path(), meta={"kind": "arc"})
    cost, feas = Constraints().score(scene, cand)
    assert not feas and cost >= 1e8
    # the same scene WITH road evidence stays feasible
    scene2 = _scene_with_wall(gap_clear=True)
    scene2.grid.drivable[20:40, 26:34] = 1.0
    scene2.strict_perception = True
    scene2.lane_ref = _straight_path()    # a perception lane exists
    cost2, feas2 = Constraints(w_lane_align=0.0, w_curvature=0.0,
                               w_progress=0.0).score(scene2, cand)
    assert feas2


def test_body_collision_empty_sampling_window_preserves_three_values():
    scene = _scene_with_wall(gap_clear=True)
    path = np.array([[0.0, 0.0], [1.5, 0.0], [2.2, 0.0]])
    assert _path_body_collision(scene, path) == (0, 0, 0)
    assert _path_body_collision(scene, path[:1]) == (0, 0, 0)
    scene.grid = None
    assert _path_body_collision(scene, path) == (0, 0, 0)


def test_constraints_score_short_forward_candidate_without_unpack_error():
    scene = _scene_with_wall(gap_clear=True)
    path = np.array([[0.0, 0.0], [1.5, 0.0], [2.2, 0.0]])
    candidate = SimpleNamespace(path=path, meta={"kind": "arc"})
    cost, feasible = Constraints().score(scene, candidate)
    assert feasible and np.isfinite(cost)
