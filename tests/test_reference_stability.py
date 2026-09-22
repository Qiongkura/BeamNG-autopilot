"""P1-2: a reference earns steering authority, it is not granted by default.

The round-5 review found the same stretch of road alternating between a
paired own-lane reference, an oncoming-lane read, the whole-road fused
centre and a single-edge mirror - and the crash run holding +0.55 steering
while the reference flipped.  These tests pin the rules the handoff asks
for: side + centre agreement for 2-3 ticks before full authority, small
corrections only for unpaired reads, and no promotion of a stale
reference just because it survived.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from beamng_autopilot.lane import (
    AUTHORITY_FULL,
    AUTHORITY_LIMITED,
    ReferenceStabilityTracker,
    near_reference_lat,
    reference_side,
)
from beamng_autopilot import fsd_drive as fd


def _ref(lat: float, n: int = 21, step: float = 1.0):
    """Straight reference polyline running parallel to the ego at ``lat``."""
    lon = np.arange(n, dtype=float) * step
    return np.column_stack([lon, np.full(n, float(lat))])


def test_reference_side_uses_a_dead_band():
    assert reference_side(0.0) == "center"
    assert reference_side(0.25) == "center"
    assert reference_side(1.2) == "left"
    assert reference_side(-1.2) == "right"
    assert reference_side(None) == "unknown"


def test_near_reference_lat_is_stable_under_forward_motion():
    pos0 = np.zeros(3)
    lat0 = near_reference_lat(_ref(-1.5), pos0, 0.0)
    pos1 = np.array([4.0, 0.0, 0.0])          # 4 m further along
    lat1 = near_reference_lat(_ref(-1.5), pos1, 0.0)
    assert lat0 == pytest.approx(-1.5, abs=0.2)
    assert lat1 == pytest.approx(lat0, abs=1e-6)


def test_full_authority_needs_paired_and_two_agreeing_ticks():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    first = t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True,
                     fresh=True)
    assert first.authority == AUTHORITY_LIMITED
    assert first.stable_ticks == 1
    assert "not yet stable" in first.reason
    second = t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True,
                      fresh=True)
    assert second.authority == AUTHORITY_FULL
    assert second.stable_ticks == 2


def test_an_unpaired_reference_never_earns_full_authority():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    for _ in range(6):
        st = t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=False,
                      fresh=True)
    assert st.authority == AUTHORITY_LIMITED
    assert "unpaired" in st.reason
    assert st.stable_ticks >= 2       # the streak is tracked, authority is not


def test_a_centre_jump_resets_the_streak_and_counts_a_flip():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True, fresh=True)
    st = t.update(ref=_ref(1.5), pos=pos, heading=hd, paired=True, fresh=True)
    assert st.stable_ticks == 1
    assert st.flip is True
    assert st.flips_total == 1
    assert st.side == "left"          # the reference is now on the left
    assert st.authority == AUTHORITY_LIMITED


def test_a_small_drift_inside_the_band_keeps_stability():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True, fresh=True)
    st = t.update(ref=_ref(-1.8), pos=pos, heading=hd, paired=True, fresh=True)
    assert st.stable_ticks == 2
    assert st.flip is False
    assert st.authority == AUTHORITY_FULL


def test_a_side_change_without_a_band_break_still_counts_as_a_flip():
    """Crossing the car's centre line is the change that matters."""
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    t.update(ref=_ref(0.4), pos=pos, heading=hd, paired=True, fresh=True)
    st = t.update(ref=_ref(-0.4), pos=pos, heading=hd, paired=True, fresh=True)
    assert st.side == "right" and st.flip is True
    assert st.authority == AUTHORITY_LIMITED


def test_a_stale_reference_cannot_be_promoted_by_surviving():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    for _ in range(3):
        t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True, fresh=True)
    assert t.ticks >= 2
    st = t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True,
                  fresh=False)
    assert st.authority == AUTHORITY_LIMITED
    assert "not fresh" in st.reason
    assert st.stable_ticks == 0
    # ...and the streak starts over afterwards
    st2 = t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True, fresh=True)
    assert st2.stable_ticks == 1


def test_no_reference_resets_and_reports_limited():
    t = ReferenceStabilityTracker()
    pos, hd = np.zeros(3), 0.0
    t.update(ref=_ref(-1.5), pos=pos, heading=hd, paired=True, fresh=True)
    st = t.update(ref=None, pos=pos, heading=hd, paired=True, fresh=True)
    assert st.authority == AUTHORITY_LIMITED
    assert st.side == "unknown"
    assert st.stable_ticks == 0


def test_the_stabiliser_reads_no_map_geometry():
    """AGENTS.md: nothing that can steer may use the nav route or an offset."""
    src = inspect.getsource(inspect.getmodule(ReferenceStabilityTracker))
    for banned in ("nav_route", "route_ref", "map_lane", "RIGHT_OFFSET",
                   "SNAP_LANE"):
        assert banned not in src, banned


class TestSteeringAuthorityApplication:
    """Arm C/D behaviour of the drive loop: limited authority = small nudge."""

    def test_the_switch_is_off_by_default(self):
        assert fd.REF_STABILITY_ENABLED is False

    def test_the_clamp_is_smaller_than_the_full_lock(self):
        assert fd.REF_STABILITY_LIMITED_STEER < 0.55
        assert fd.REF_STABILITY_LIMITED_STEER > 0.0
