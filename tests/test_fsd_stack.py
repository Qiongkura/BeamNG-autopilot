"""Offline tests for the integrated FSDStack (stubbed ring/range)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.fsd_stack import (
    FSDStack,
    FSDTick,
    RANGE_REUSE_MAX_PREDICT_M,
    compensate_range_motion,
    semantic_to_meta,
)


class _StubRange:
    def scan(self, pos):
        from beamng_autopilot.runtime import RangeSample
        from beamng_autopilot.perception import Obstacle
        return RangeSample(
            obstacles=[Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                                category="lidar")],
            ray_hits=[(6.0, -1.0), (6.0, 1.0)])


class _CountingRange:
    """Range stub that counts how many real scans happened."""

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, pos):
        self.calls += 1
        return _StubRange().scan(pos)


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
    st._head_skip = {}
    st._last_heads = {}
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
    st._head_skip = {}
    st._last_heads = {}
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


class _CountingObject(_FakeSemantic):
    """Object stub that counts runs and returns one world obstacle."""

    name = "object"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        from beamng_autopilot.vision.hydra import TaskOutput
        from beamng_autopilot.perception import Obstacle
        return TaskOutput(obstacles=[Obstacle(
            x=8.0, y=0.0, half_w=0.8, half_h=0.8,
            category="vehicle")])


def _range_with_moving_car():
    from beamng_autopilot.perception import Obstacle
    from beamng_autopilot.runtime import RangeSample
    return RangeSample(
        obstacles=[
            Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=2.0,
                     category="vehicle", label="car",
                     velocity=np.array([5.0, 0.0]),
                     heading=0.0,
                     axis=np.array([1.0, 0.0]),
                     half_len=2.0, half_thick=1.0,
                     vehicle_id="v0"),
            Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                     category="lidar"),
        ],
        ray_hits=[(6.0, -1.0), (6.0, 1.0)])


class _MovingCarRange:
    """Range stub returning one moving car; counts real scans."""

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, pos):
        self.calls += 1
        return _range_with_moving_car()


def test_fsd_tick_object_throttle_reuses_last_output() -> None:
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    obj = _CountingObject()
    st.hydra.add(obj)
    st.object_every_n = 2
    st._head_skip = {}
    st._last_heads = {}
    out1 = st.tick()
    assert obj.calls == 1
    out2 = st.tick()
    assert obj.calls == 1          # second tick reuses the last output
    assert out2.head_outputs["object"] is out1.head_outputs["object"]
    out3 = st.tick()
    assert obj.calls == 2          # third tick runs the head again


def test_fsd_tick_object_obstacles_fused_into_bev() -> None:
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    obj = _CountingObject()
    st.hydra.add(obj)
    st.object_every_n = 1
    st._head_skip = {}
    st._last_heads = {}
    out = st.tick()
    assert out.meta.get("n_object_obstacles") == 1
    # world (8,0) -> ego (8,0) -> cell index in the 60x60@0.5 grid
    r = int((st.grid_n * st.grid_res * 0.5 - 8.0) / st.grid_res)
    assert 0 <= r < st.grid_n
    assert out.bev[r, st.grid_n // 2] > 0.0


def test_fsd_tick_range_throttle_reuses_last_scan() -> None:
    """range_every_n>1 polls LiDAR every n-th tick and reuses the last
    world-frame scan in between (static walls stay valid; the temporal
    occupancy filter bridges the gap)."""
    st = _stack()
    st.range_prov = _CountingRange()
    st.range_every_n = 3
    st._range_skip = 0
    st._last_range = None
    out1 = st.tick()
    assert st.range_prov.calls == 1
    out2 = st.tick()
    assert st.range_prov.calls == 1
    assert out2.ray_hits == out1.ray_hits
    assert out2.meta.get("n_obstacles", 0) >= 1
    out3 = st.tick()
    assert st.range_prov.calls == 1
    out4 = st.tick()
    assert st.range_prov.calls == 2          # fresh scan on the 4th tick


def test_fsd_tick_budget_defers_heavy_head_then_catches_up() -> None:
    """time_budget_s caps a tick: a due heavy head is deferred to a later
    affordable tick instead of freezing the control loop, and stays due
    until it actually runs."""
    import beamng_autopilot.fsd_stack as fs

    class _FakeClock:
        """Deterministic clock: advances 1 s between calls, so the tiny
        budget is always exceeded regardless of Windows timer jitter."""

        def __init__(self) -> None:
            self.t = 1000.0

        def __call__(self) -> float:
            self.t += 1.0
            return self.t

    _real_time = fs.time.time
    fs.time.time = _FakeClock()
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    sem = _CountingSemantic()
    st.hydra.add(sem)
    st.semantic_every_n = 2
    st._head_skip = {}
    st._last_heads = {}
    try:
        out0 = st.tick(time_budget_s=1e-12)
        assert sem.calls == 0                # deferred, not run
        assert out0.meta.get("tick_budget_skips") == ["semantic"]
        out1 = st.tick()                     # no budget: catch-up runs it
        assert sem.calls == 1
        assert out1.head_outputs["semantic"] is not None
        assert "semantic" not in st._head_retry  # retry set is cleared
    finally:
        fs.time.time = _real_time


def test_fsd_tick_equal_heavy_cadences_stagger_ticks() -> None:
    """semantic_every_n == object_every_n offsets the object head by half
    a cycle so one tick never runs both heavy heads (the stutter source)."""
    st = _stack()
    from beamng_autopilot.vision.hydra import HydraNet
    st.hydra = HydraNet()
    sem = _CountingSemantic()
    obj = _CountingObject()
    st.hydra.add(sem)
    st.hydra.add(obj)
    st.semantic_every_n = 2
    st.object_every_n = 2
    st._head_phase = {"object": 1}
    st._head_skip = {}
    st._last_heads = {}
    for _ in range(4):
        st.tick()
    # cadence 2 -> each head runs on 2 of the 4 ticks, never together
    assert sem.calls == 2
    assert obj.calls == 2


def test_compensate_range_motion_predicts_dynamic_box_ahead() -> None:
    """A moving vehicle box is shifted by its world velocity over the
    reuse gap and inflated with a small safety margin."""
    out = compensate_range_motion(_range_with_moving_car(), 1.0)
    car = [o for o in out.obstacles
           if getattr(o, "vehicle_id", None) == "v0"][0]
    assert car.x == pytest.approx(15.0, abs=1e-6)   # 10 + 5 m/s * 1 s
    assert car.y == pytest.approx(0.0, abs=1e-6)
    assert car.half_w > 1.0                          # inflated for safety
    assert car.velocity is not None                  # keep track identity


def test_compensate_range_motion_leaves_static_boxes_untouched() -> None:
    """Walls / lidar clusters have no velocity and never move."""
    out = compensate_range_motion(_range_with_moving_car(), 1.0)
    lidar = [o for o in out.obstacles if o.category == "lidar"][0]
    assert lidar.x == 6.0 and lidar.y == 0.0
    assert out.ray_hits == [(6.0, -1.0), (6.0, 1.0)]
    assert compensate_range_motion(None, 5.0) is None


def test_compensate_range_motion_caps_stale_velocity() -> None:
    """Stale/erroneous velocities cannot teleport the box absurdly far."""
    from beamng_autopilot.perception import Obstacle
    from beamng_autopilot.runtime import RangeSample
    sample = RangeSample(
        obstacles=[
            Obstacle(x=10.0, y=0.0, half_w=1.0, half_h=2.0,
                     category="vehicle",
                     velocity=np.array([40.0, 0.0]),
                     half_len=2.0, half_thick=1.0),
        ],
        ray_hits=[])
    out = compensate_range_motion(sample, 10.0)      # dt capped at 2 s
    car = out.obstacles[0]
    assert (car.x - 10.0) <= RANGE_REUSE_MAX_PREDICT_M + 1e-9


def test_fsd_tick_range_reuse_compensates_vehicle_motion() -> None:
    """A reused LiDAR scan predicts moving boxes forward instead of
    replaying the stale scan pose."""
    st = _stack()
    st.range_prov = _MovingCarRange()
    st.range_every_n = 3
    st._range_skip = 0
    st._last_range = None
    import beamng_autopilot.fsd_stack as fs
    orig = fs.compensate_range_motion
    seen: list[float] = []
    try:
        fs.compensate_range_motion = (
            lambda s, dt: (seen.append(float(dt)),
                           orig(s, dt))[1])
        out1 = st.tick()
        assert st.range_prov.calls == 1
        assert not seen                       # fresh scan: no compensation
        st._last_range_t -= 1.0               # simulate a 1 s reuse gap
        out2 = st.tick()
        assert st.range_prov.calls == 1       # reuse, no new scan
        assert len(seen) == 1 and seen[0] >= 0.99
        assert out2.ray_hits == out1.ray_hits
        assert out2.meta.get("n_obstacles", 0) >= 2
    finally:
        fs.compensate_range_motion = orig
