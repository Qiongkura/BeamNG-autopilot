"""Offline tests for the BEV occupancy grid (vector-space representation)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.occupancy import (
    OccupancyGrid,
    fuse_obstacles_to_grid,
    project_road_mask_to_grid,
)
from beamng_autopilot.perception import Obstacle
from beamng_autopilot.vision.projection import CameraModel


def _grid(res=0.5, n=20):
    return OccupancyGrid(n, n, res)


def test_grid_origin_centred() -> None:
    g = _grid(res=0.5, n=20)
    assert abs(g.extent - 5.0) < 1e-9
    assert g.center == (10, 10)
    assert g.occupancy.shape == (20, 20)


def test_ego_to_cell_front_left_maps_to_row0_col0() -> None:
    g = _grid(res=0.5, n=20)
    assert g.max_x == 5.0 and g.max_y == 5.0
    r, c = g.ego_to_cell(4.75, 4.75)
    assert (r, c) == (0, 0)
    # ego centre maps to the grid centre
    r, c = g.ego_to_cell(0.0, 0.0)
    assert (r, c) == (10, 10)
    # far rear-right corner
    r, c = g.ego_to_cell(-4.75, -4.75)
    assert (r, c) == (19, 19)
    # out of bounds
    assert g.ego_to_cell(50.0, 0.0) is None
    assert g.ego_to_cell(0.0, -50.0) is None


def test_world_to_cell_respects_origin_and_heading() -> None:
    g = OccupancyGrid(20, 20, 0.5, origin=(100.0, 200.0), heading=0.0)
    # a point 1 m ahead (+x) of the origin at the right lateral (-y)
    r, c = g.world_to_cell(101.0, 199.0)
    assert (r, c) is not None
    ex = (101.0 - 100.0)
    ey = (199.0 - 200.0)
    rr, cc = g.ego_to_cell(ex, ey)
    assert (r, c) == (rr, cc)


def test_obstacle_point_accumulates_and_decays() -> None:
    g = _grid(res=0.5, n=40)  # 10 m extent so 5 m samples fit
    g.origin = (0.0, 0.0)
    g.add_obstacle_point(5.0, 0.0, z=1.0)
    cell = g.world_to_cell(5.0, 0.0)
    assert cell is not None
    assert g.occupancy[cell] > 0.0
    assert g.sources[cell] == 1
    assert np.isfinite(g.height[cell])
    # free space decays the obstacle evidence
    g.add_drivable_point(5.0, 0.0)
    assert g.occupancy[cell] < 0.4
    # permanent barrier never cleared by free space
    g.mark_obstacle_region(0.0, 0.0, 1.5, 1.0)
    cell2 = g.world_to_cell(0.0, 0.0)
    assert g.obstacle[cell2] == 1
    g.add_drivable_point(0.0, 0.0)
    assert g.obstacle[cell2] == 1


def test_obstacle_region_floods() -> None:
    g = _grid(res=0.5, n=40)
    g.origin = (0.0, 0.0)
    g.mark_obstacle_region(4.0, 0.0, 1.0, 1.0)
    # all samples of the box centre line become obstacles
    for x in (4.0,):
        cell = g.world_to_cell(x, 0.0)
        assert cell is not None
        assert g.obstacle[cell] == 1
    # a cell well away stays drivable
    far = g.world_to_cell(-4.0, 0.0)
    assert g.obstacle[far] == 0


def test_query_path_cost_penalises_occupied() -> None:
    g = _grid(res=0.5, n=40)
    g.origin = (0.0, 0.0)
    g.mark_obstacle_region(4.0, 0.0, 1.0, 1.0)
    clean = [(x, 0.0) for x in np.linspace(-5, -1, 10)]
    blocked = [(x, 0.0) for x in np.linspace(2, 8, 10)]
    assert g.query_path_cost(clean) < 0.1
    # the path through the box is penalised (occupancy carries weight even
    # after averaging in the far, free end)
    assert g.query_path_cost(blocked) >= 0.3
    # a path sampled exactly inside the barrier is dominated by it
    in_box = [(4.0, 0.0), (4.0, 0.0), (4.0, 0.0)]
    assert g.query_path_cost(in_box) >= 0.5
    # out-of-grid samples count as occupied (unknown != drivable)
    assert g.query_path_cost([(100.0, 100.0)]) == 1.0


def test_project_road_mask_marks_drivable() -> None:
    from beamng_autopilot.vision.projection import CameraModel

    g = _grid(res=0.5, n=40)
    g.origin = (0.0, 0.0)
    w, h = 160, 120
    # A real front camera pitches down slightly (like the calibrated one),
    # so its lower-half rays actually meet the ground plane.
    cam = CameraModel(np.array([0.0, 1.0, 1.4]),
                      np.array([0.0, 0.9999, -0.02]),
                      np.array([0.0, 0.02, 0.9999]), 65.0, w, h)
    # a full-road mask: everything in the lower half projects to the road
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2:] = True
    project_road_mask_to_grid(g, mask, cam, np.array([0.0, 0.0, 0.0]),
                              0.0, step=6)
    # the front camera stamps drivable space forward of the car
    n_drv = int(g.drivable.sum())
    assert n_drv > 20, f"only {n_drv} drivable cells"
    # some forward cell (a few metres ahead) is marked drivable
    rr, cc = np.nonzero(g.drivable)
    xs = [(g.max_x - (r + 0.5) * g.res) for r in rr]
    assert any(2.0 <= x <= 8.0 for x in xs), f"forward drift missing: {xs}"
    # nothing behind the car should be stamped by the front camera
    assert g.drivable[g.ego_to_cell(-2.0, 0.0)] == 0.0


def test_fuse_obstacles_to_grid_boxes() -> None:
    g = _grid(res=0.5, n=40)
    g.origin = (0.0, 0.0)
    obs = Obstacle(x=3.0, y=0.0, half_w=1.0, half_h=1.0, category="lidar")
    fuse_obstacles_to_grid(g, [obs])
    cell = g.world_to_cell(3.0, 0.0)
    assert g.obstacle[cell] == 1
    assert g.query_path_cost([(3.0, 0.0)]) >= 0.5


def test_ray_hits_fuse_as_soft_evidence() -> None:
    g = _grid(res=0.5, n=40)
    g.origin = (0.0, 0.0)
    fuse_obstacles_to_grid(g, [], ray_hits=[(6.0, -1.0), (6.0, 1.0)])
    cell = g.world_to_cell(6.0, 0.0)  # between the two hits
    assert g.sources[cell] == 0  # soft hits are sparse; fine