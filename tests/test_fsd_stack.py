"""Offline tests for the integrated FSDStack (stubbed ring/range)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_stack import FSDStack, FSDTick, semantic_to_meta


class _StubRange:
    def scan(self, pos):
        from beamng_autopilot.runtime import RangeSample
        from beamng_autopilot.perception import Obstacle
        return RangeSample(
            obstacles=[Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                                category="lidar")],
            ray_hits=[(6.0, -1.0), (6.0, 1.0)])


class _StubRing:
    role = "front_main"

    def grab_ring(self):
        import numpy as np
        from beamng_autopilot.vision.projection import CameraModel
        model = CameraModel(np.array([0.0, 1.0, 1.4]),
                            np.array([0.0, 0.9999, -0.02]),
                            np.array([0.0, 0.02, 0.9999]), 65.0, 160, 120)
        frame = np.random.default_rng(0).integers(
            40, 220, (120, 160, 3), dtype=np.uint8)
        return {"front_main": (frame, model)}

    def close(self):
        return None


class _StubConn:
    def get_state(self):
        class S:
            pos = np.array([0.0, 0.0, 0.0])
            heading = 0.0
            speed = 5.0
        return S()


class _FakeSemantic:
    name = "semantic"

    def run(self, ctx):
        from beamng_autopilot.vision.hydra import TaskOutput
        h, w = ctx.frame_rgb.shape[:2]
        road = np.zeros((h, w), dtype=bool)
        road[h // 2:] = True  # "road" in the lower half
        return TaskOutput(masks={"road": road}, meta={"markings": [1, 2]})


def _stack():
    st = FSDStack.__new__(FSDStack)
    st.conn = _StubConn()
    st.ring = _StubRing()
    st.mode = "tech-stub"
    st.range_prov = _StubRange()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    st.hydra.add(_FakeSemantic())
    from beamng_autopilot.planning import Constraints
    st.constraints = Constraints(w_collision=5.0, w_curvature=0.5,
                                 w_lane_align=1.0)
    st.grid_n, st.grid_res = 60, 0.5
    return st


def test_fsd_tick_pipeline_runs() -> None:
    st = _stack()
    out = st.tick()
    assert isinstance(out, FSDTick)
    assert out.bev is not None and out.bev.shape == (60, 60)
    assert out.drivable is not None
    assert out.n_candidates >= 9
    assert "lane_markings" in out.meta


def test_fsd_tick_obstacle_present_in_bev() -> None:
    st = _stack()
    out = st.tick()
    # the stub LiDAR obstacle at (6, 0) should show as occupancy
    grid_occ = out.bev
    # world (6,0) -> ego (6,0) -> cell
    r = int((st.grid_n * st.grid_res * 0.5 - 6.0) / st.grid_res)
    assert 0 <= r < st.grid_n
    assert grid_occ[r, st.grid_n // 2] > 0.0


def test_fsd_tick_best_path_possible() -> None:
    st = _stack()
    out = st.tick()
    # with an obstacle at (6,0) the arc fan usually still finds a clear
    # candidate; at minimum the meta reports what happened
    assert "planner" in out.meta
    assert out.n_candidates >= 9
    # the planner meta explains the selection ("best-of-N" or
    # "no feasible candidate")
    assert out.meta["planner"]["why"] in ("best-of-N",
                                          "no feasible candidate")


def test_semantic_to_meta_flattens() -> None:
    out = semantic_to_meta({
        "semantic": type("O", (), {"meta": {"markings": [1]}})(),
        "traffic": type("O", (), {"meta": {"signal_state": "green",
                                            "signal_conf": 0.9}})(),
        "topology": type("O", (), {"meta": {"change_left": True,
                                             "change_right": False}})(),
    })
    assert out["lane_markings"] == 1
    assert out["signal_state"] == "green"
    assert out["change_left"] is True and out["change_right"] is False


def test_bev_drivable_center_returns_ahead_centerline() -> None:
    st = _stack()
    from beamng_autopilot.occupancy import OccupancyGrid
    import numpy as np

    grid = OccupancyGrid(60, 60, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    # a drivable corridor centred slightly left of the ego (ego local y>0)
    drv = np.zeros((60, 60), dtype=np.uint8)
    for r in range(60):
        # front rows (small r) drivable at col 25-35; rear rows none
        if r < 40:
            drv[r, 25:36] = 1
    grid.drivable = drv.astype(np.float32)
    lane = st._bev_drivable_center(grid, np.array([0.0, 0.0]), 0.0)
    assert lane is not None and len(lane) >= 3
    # the centreline includes points ahead of the ego within the corridor
    ahead = lane[lane[:, 0] > 2.0]
    assert len(ahead) >= 2, f"no ahead points: {lane}"
    # the centreline's lateral spread is near the drivable corridor middle
    assert abs(float(np.median(lane[:, 1]))) < 3.0


def test_bev_drivable_center_empty_returns_none() -> None:
    st = _stack()
    from beamng_autopilot.occupancy import OccupancyGrid
    import numpy as np

    grid = OccupancyGrid(30, 30, 0.5)
    grid.origin = (0.0, 0.0)
    grid.heading = 0.0
    assert st._bev_drivable_center(grid, np.array([0.0, 0.0]), 0.0) is None

def test_fsd_tick_lane_ref_anchored_near_ego() -> None:
    """The drivable-centreline lane reference must be anchored at the ego
    (start within ~3 m) so the planner's shift candidates pass the
    forward-progress gate - a far-first reference started 8 m ahead and
    made every candidate infeasible (town runs 2026-08-21)."""
    st = _stack()
    out = st.tick()
    assert out.lane_ref is not None and len(out.lane_ref) >= 3
    d0 = float(np.linalg.norm(np.asarray(out.lane_ref[0], dtype=float)))
    assert d0 <= 3.0, f"lane reference not anchored: start {d0:.1f} m away"


class _CountingSemantic(_FakeSemantic):
    """Semantic stub that counts how many times it actually runs."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        return super().run(ctx)


def test_fsd_tick_semantic_throttle_reuses_last_output() -> None:
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    sem = _CountingSemantic()
    st.hydra.add(sem)
    st.semantic_every_n = 2
    st._semantic_skip = 0
    st._last_heads = None
    out1 = st.tick()
    assert sem.calls == 1
    out2 = st.tick()
    assert sem.calls == 1          # second tick reuses the last output
    assert out2.head_outputs["semantic"] is out1.head_outputs["semantic"]
    out3 = st.tick()
    assert sem.calls == 2          # third tick runs the head again
    # the reused output still carries the road mask for BEV fusion
    assert "road" in out2.head_outputs["semantic"].masks


def test_fsd_tick_semantic_throttle_off_by_default() -> None:
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    sem = _CountingSemantic()
    st.hydra.add(sem)
    st.semantic_every_n = 1
    st._semantic_skip = 0
    st._last_heads = None
    st.tick()
    st.tick()
    assert sem.calls == 2          # default runs every tick


def test_fsd_tick_candidates_survive_progress_gate() -> None:
    """An ego-anchored lane reference must not silently kill the lane-shift
    candidates via the forward-progress gate.  At least the 11 arc fan plus
    the shift family should reach the selector (a couple of straight arcs
    can legitimately collide with the stub obstacle at (6, 0) and drop).
    Regression: the far-first lane reference once rejected every shift and
    left only the arcs (town runs 2026-08-21)."""
    st = _stack()
    out = st.tick()
    meta = out.meta.get("planner", {})
    if meta.get("n_eval") is not None:
        assert meta["n_eval"] >= 12, meta
