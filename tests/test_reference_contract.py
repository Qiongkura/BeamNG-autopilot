"""T02 contract counterexamples: one accepted reference, one authority.

Written from the round-5 plan (FINAL_INTEGRATED_PLAN_20260922 §2.2/§3.3),
which listed four defects as *component* counterexamples.  Each one here
is pinned at the level where the decision is actually taken, so a fix
cannot pass by fixing only the unit in isolation:

1. a CONSTRUCTED centre (painted centre line: ``paired=True`` yet only the
   left edge was measured) must never earn full steering authority;
2. a HELD / coasted reference is not fresh observation and must not
   promote itself to full authority by surviving;
3. a candidate whose centre was revoked by the on-pavement gate must not
   be re-published as ``sensor`` with an empty centre (plan §3.3-1/2);
4. the planner Scene must consume the SAME final geometry the control
   loop consumes - including after the slew limiter (plan §3.3-3/4);
5. an accepted perception reference reaches the Scene without depending on
   ``frame_used`` (plan §3.3-5: no ``frame_used`` deciding silently).

These are deterministic component/stack tests.  They do NOT show the car
behaving well: closed-loop behaviour is a separate evidence level
(plan §8.1).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamng_autopilot.fsd_realism import SRC_BEV_ROUTE, SRC_SENSOR
from beamng_autopilot.lane import LaneFrame
from beamng_autopilot.lane.reference import LaneReference
from beamng_autopilot.lane.stability import (
    AUTHORITY_FULL,
    AUTHORITY_LIMITED,
    ReferenceStabilityTracker,
)


# --------------------------------------------------------------------------
# 1 + 2: authority grading must use real two-sided freshness
# --------------------------------------------------------------------------
def _straight_ref(lat_m: float = -1.8, n: int = 12) -> np.ndarray:
    """A reference polyline parallel to the car at ``lat_m`` (+ = left)."""
    s = np.linspace(2.0, 26.0, n)
    return np.column_stack([s, np.full(n, float(lat_m))])


class TestAuthoritySemantics:
    def test_a_constructed_centre_line_never_earns_full_authority(self):
        """Painted centre line: ``paired=True`` but the right edge is a width
        prior, not a measurement (lane/pairing.py::_centre_line_own_lane)."""
        frame = LaneFrame(center=_straight_ref(), left=_straight_ref(-3.6),
                          right=None, width=3.6, paired=True,
                          left_kind="solid", right_kind=None)
        assert frame.two_sided_measured is False
        tr = ReferenceStabilityTracker()
        for _ in range(10):
            st = tr.update(ref=_straight_ref(), pos=np.zeros(3), heading=0.0,
                           paired=frame.two_sided_measured, fresh=True)
        assert st.authority == AUTHORITY_LIMITED

    def test_a_two_sided_measurement_does_earn_full_authority(self):
        """The positive case, so the fix cannot silently disable the gate."""
        frame = LaneFrame(center=_straight_ref(), left=_straight_ref(-3.6),
                          right=_straight_ref(0.0), width=3.6, paired=True,
                          left_kind="solid", right_kind="solid")
        assert frame.two_sided_measured is True
        tr = ReferenceStabilityTracker()
        for _ in range(4):
            st = tr.update(ref=_straight_ref(), pos=np.zeros(3), heading=0.0,
                           paired=frame.two_sided_measured, fresh=True)
        assert st.authority == AUTHORITY_FULL

    def test_a_held_reference_never_promotes_itself(self):
        """A hold is not this tick's observation: the streak must reset."""
        tr = ReferenceStabilityTracker()
        st = tr.update(ref=_straight_ref(), pos=np.zeros(3), heading=0.0,
                       paired=True, fresh=True)
        st = tr.update(ref=_straight_ref(), pos=np.zeros(3), heading=0.0,
                       paired=True, fresh=True)
        assert st.authority == AUTHORITY_FULL
        for _ in range(20):
            st = tr.update(ref=_straight_ref(), pos=np.zeros(3), heading=0.0,
                           paired=True, fresh=False)
            assert st.authority == AUTHORITY_LIMITED


class TestFrameProvenance:
    def test_a_centre_line_frame_declares_its_inferred_side(self):
        """Pinned on the REAL construction path, not a hand-built frame.

        ``paired=True`` here is about the own-lane geometry being usable;
        the right edge comes from the width prior, so the frame must
        declare ``inferred`` and must not read as two-sided.
        """
        from beamng_autopilot.lane.pairing import _centre_line_own_lane
        proj = np.column_stack([np.linspace(2.0, 12.0, 8),
                                np.full(8, -0.4)])
        cand = SimpleNamespace(color="yellow", kind="solid", conf=0.8,
                               span=9.0, med_lat=-0.4, proj=proj)
        frame = _centre_line_own_lane(cand, np.zeros(2),
                                      np.array([1.0, 0.0]), 3.6)
        assert frame is not None and frame.paired is True
        assert frame.right is None
        assert frame.inferred is True
        assert frame.two_sided_measured is False

    def test_a_mirrored_single_edge_is_not_two_sided(self):
        frame = LaneFrame(center=_straight_ref(), left=_straight_ref(-3.6),
                          right=_straight_ref(0.0), width=3.6, paired=False,
                          inferred=True)
        assert frame.two_sided_measured is False

    def test_a_real_pair_is_two_sided(self):
        frame = LaneFrame(center=_straight_ref(), left=_straight_ref(-3.6),
                          right=_straight_ref(0.0), width=3.6, paired=True)
        assert frame.two_sided_measured is True
        assert frame.inferred is False

    def test_a_hand_built_frame_with_a_missing_side_is_not_two_sided(self):
        """Belt and braces: a future caller that forgets ``inferred`` still
        cannot claim a two-sided measurement without two sides."""
        frame = LaneFrame(center=_straight_ref(), left=_straight_ref(-3.6),
                          right=None, width=3.6, paired=True)
        assert frame.two_sided_measured is False


# --------------------------------------------------------------------------
# 3: a revoked candidate must not be re-published as ``sensor``
# --------------------------------------------------------------------------
class TestPublicationInvariant:
    def test_a_sensor_reference_without_geometry_must_not_be_published(self):
        """Invariant (plan §3.3-1): ``sensor`` implies real centre geometry.

        The backstop is checked on the DATACLASS so every branch that sets
        ``src`` - present and future - is covered, not just the two
        branches that were found by reading.
        """
        ref = LaneReference(center=None, src=SRC_SENSOR, boundaries=True)
        assert ref.publishable_sensor is False

    def test_the_backstop_accepts_a_real_centre(self):
        ref = LaneReference(center=_straight_ref(), src=SRC_SENSOR)
        assert ref.publishable_sensor is True

    def test_the_backstop_ignores_non_sensor_sources(self):
        ref = LaneReference(center=None, src=SRC_BEV_ROUTE)
        assert ref.publishable_sensor is True   # not this rule's business


# --------------------------------------------------------------------------
# 4 + 5: Scene and control consume one geometry
# --------------------------------------------------------------------------
class TestSceneReference:
    def test_an_accepted_perception_reference_reaches_the_scene(self):
        """Without ``frame_used`` deciding it (divider / paved fallbacks)."""
        ref = LaneReference(center=_straight_ref(), src=SRC_SENSOR,
                            frame_used=False)
        assert ref.scene_ref is not None

    def test_the_bev_whole_road_centre_still_never_reaches_the_scene(self):
        ref = LaneReference(center=_straight_ref(), src=SRC_BEV_ROUTE,
                            frame_used=False)
        assert ref.scene_ref is None

    def test_geometry_id_is_a_function_of_the_geometry(self):
        a = LaneReference(center=_straight_ref(), src=SRC_SENSOR)
        b = LaneReference(center=_straight_ref(), src=SRC_SENSOR)
        c = LaneReference(center=_straight_ref(-2.4), src=SRC_SENSOR)
        assert a.geom_id == b.geom_id
        assert a.geom_id != c.geom_id

    def test_the_slew_writes_back_into_the_published_reference(self):
        """The slew must move the ONE accepted object, not a local copy.

        ``fsd_stack`` used to slew a local ``lane_ref`` while the Scene read
        ``lane_ref_out.scene_ref`` (the un-slewed centre): planner and
        controller then disagreed about where the lane was.
        """
        from beamng_autopilot.fsd_stack import FSDStack
        from beamng_autopilot.planning.lateral_ref import limit_reference_slew
        ref = LaneReference(center=_straight_ref(-3.0), src=SRC_SENSOR,
                            frame_used=True)
        before = ref.geom_id
        prev = _straight_ref(-0.5)
        slewed, _hold = limit_reference_slew(prev, 0.0, ref.center,
                                             now=1.0, pos=np.zeros(3),
                                             heading=0.0)
        assert slewed is not None
        assert not np.allclose(slewed, ref.center), (
            "the test needs a slew that actually moved the geometry")
        assert FSDStack._publish_reference_geometry(ref, slewed) is True
        np.testing.assert_allclose(ref.center, slewed)
        assert ref.geom_id != before


# --------------------------------------------------------------------------
# Stack level: the same two rules, on a real FSDStack.tick
# --------------------------------------------------------------------------
class _StubRange:
    def scan(self, pos):
        from beamng_autopilot.perception import Obstacle
        from beamng_autopilot.runtime import RangeSample
        return RangeSample(
            obstacles=[Obstacle(x=6.0, y=0.0, half_w=1.0, half_h=1.0,
                                category="lidar")],
            ray_hits=[(6.0, -1.0), (6.0, 1.0)])


class _StubRing:
    role = "front_main"

    def grab_ring(self):
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
        road[h // 2:] = True
        return TaskOutput(masks={"road": road}, meta={"markings": []})


def _stack():
    """The same stubbed stack tests/test_fsd_stack.py uses."""
    from beamng_autopilot.fsd_stack import FSDStack
    from beamng_autopilot.planning import Constraints
    from beamng_autopilot.vision.hydra import HydraNet

    st = FSDStack.__new__(FSDStack)
    st.conn = _StubConn()
    st.ring = _StubRing()
    st.mode = "tech-stub"
    st.range_prov = _StubRange()
    st.hydra = HydraNet()
    st.hydra.add(_FakeSemantic())
    st.constraints = Constraints(w_collision=5.0, w_curvature=0.5,
                                 w_lane_align=1.0)
    st.grid_n, st.grid_res = 60, 0.5
    # Perception-led strict mode: the plan's default acceptance baseline
    # (plan §4: ``--lane-mode sensor --strict``).
    st.lane_mode = "sensor"
    st.strict_sensor = True
    return st


def _pair_frame(obs_seq: int, inferred: bool = False, lat_m: float = -1.8):
    from beamng_autopilot.lane import LaneFrame
    left = _straight_ref(lat_m - 1.8)
    right = None if inferred else _straight_ref(lat_m + 1.8)
    return LaneFrame(center=_straight_ref(lat_m), left=left, right=right,
                     width=3.6, confidence=0.9, span_m=20.0,
                     paired=True, left_kind="solid", right_kind="solid",
                     inferred=inferred, obs_seq=int(obs_seq))


class TestStackReferenceContract:
    """The counterexamples of plan §2.2 as stack-level regressions."""

    def test_a_constructed_reference_does_not_get_full_authority(self):
        st = _stack()
        st._sensor_lane = lambda out, pos, heading: _pair_frame(
            obs_seq=int(getattr(st, "_tick_num", 0)), inferred=True)
        out = st.tick()
        assert out.meta.get("ref_authority") == AUTHORITY_LIMITED
        assert out.meta.get("lane_ref_two_sided") == 0

    def test_a_two_sided_fresh_reference_gets_full_authority(self):
        st = _stack()
        st._sensor_lane = lambda out, pos, heading: _pair_frame(
            obs_seq=int(getattr(st, "_tick_num", 0)))
        out = st.tick()
        assert out.meta.get("lane_ref_two_sided") == 1
        assert out.meta.get("lane_ref_fresh_obs") == 1
        # ...and the first observation cannot be "stable", so authority
        # only reaches full on the second consecutive agreeing tick.
        out2 = st.tick()
        assert out2.meta.get("ref_authority") == AUTHORITY_FULL

    def test_a_replayed_observation_never_promotes_the_reference(self):
        """Same geometry, same frame object state, but the observation
        behind it belongs to an earlier tick: a hold."""
        st = _stack()
        st._sensor_lane = lambda out, pos, heading: _pair_frame(
            obs_seq=int(getattr(st, "_tick_num", 0)))
        st.tick()
        st.tick()
        held = _pair_frame(obs_seq=1)          # stale observation number
        st._sensor_lane = lambda out, pos, heading: held
        out = st.tick()
        assert out.meta.get("lane_ref_fresh_obs") == 0
        assert out.meta.get("ref_authority") == AUTHORITY_LIMITED

    def test_the_planner_and_the_controller_share_one_geometry(self):
        """Slew limiter included: the write-back keeps both on one object.

        Tick 2 asks for a reference 4.2 m away from tick 1's; the limiter
        (``LANE_REF_SLEW_MAX_M = 0.8``) holds the old geometry.  Before the
        write-back the controller got the held geometry while the planner
        Scene got the requested one.
        """
        st = _stack()
        st.strict_sensor = True            # enables the slew limiter
        st._sensor_lane = lambda out, pos, heading: _pair_frame(
            obs_seq=int(getattr(st, "_tick_num", 0)), lat_m=-1.8)
        st.tick()
        st._sensor_lane = lambda out, pos, heading: _pair_frame(
            obs_seq=int(getattr(st, "_tick_num", 0)), lat_m=-6.0)
        out = st.tick()
        assert "scene_ref_geom_mismatch" not in out.meta
        assert out.scene is not None and out.scene.lane_ref is not None
        assert out.meta["scene_ref_geom_id"] == out.meta["lane_ref_geom_id"]
        np.testing.assert_allclose(np.asarray(out.lane_ref, dtype=float),
                                   np.asarray(out.scene.lane_ref, dtype=float))
        # and the limiter really did act (the requested -6.0 was refused)
        median_lat = float(np.median(np.asarray(out.lane_ref)[:, 1]))
        assert abs(median_lat - (-6.0)) > 1.0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
