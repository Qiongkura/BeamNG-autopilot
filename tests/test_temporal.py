"""Offline tests for temporal fusion (EMA occupancy + object tracking)."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.temporal import (
    TemporalOccupancyFilter,
    TrackedObject,
    WorldObjectTracker,
)


def test_filter_smooths_single_glitch() -> None:
    f = TemporalOccupancyFilter(n=30, res=0.5, conf_bias=0.5)
    # long history of wall present
    wall = np.zeros((30, 30), dtype=np.float32)
    wall[:, 15] = 1.0
    for _ in range(8):
        f.update(wall, 0.1)
    assert f.occupied_mask(0.6)[:, 15].mean() > 0.8
    # a single frame where the wall vanishes -> filter should keep most
    # of the wall (a one-frame LiDAR miss is not "wall gone")
    f.update(np.zeros((30, 30), dtype=np.float32), 0.1)
    assert f.occupied_mask(0.6)[:, 15].mean() > 0.5
    # but sustained disappearance eventually fades it
    for _ in range(16):
        f.update(np.zeros((30, 30), dtype=np.float32), 0.1)
    assert f.occupied_mask(0.6)[:, 15].mean() < 0.3


def test_filter_remembers_through_many_frames() -> None:
    f = TemporalOccupancyFilter(n=20, res=0.5)
    blob = np.zeros((20, 20), dtype=np.float32)
    blob[5, 5] = 1.0
    f.update(blob, 0.1)
    assert f.raster()[5, 5] > 0.5


def test_tracker_associates_and_smoothes_velocity() -> None:
    tr = WorldObjectTracker(match_m=2.0, dt_default=0.1)
    # an object moving +1 m/frame in x
    tr.update([(0.0, 0.0, "car")], 0.1)
    tr.update([(1.0, 0.0, "car")], 0.1)
    active = tr.update([(2.0, 0.0, "car")], 0.1)
    assert len(active) == 1
    t = active[0]
    assert t.matches == 3
    assert t.vx > 5.0  # ~10 m/s


def test_tracker_prunes_lost() -> None:
    tr = WorldObjectTracker(max_lost=2, dt_default=0.1)
    tr.update([(0.0, 0.0, "cone")], 0.1)
    tr.update([], 0.1)
    tr.update([], 0.1)
    active = tr.update([], 0.1)
    # after 3 missed frames (lost > max_lost=2) the track is gone
    assert len(active) == 0


def test_tracker_new_detection_new_track() -> None:
    tr = WorldObjectTracker(match_m=1.0)
    active = tr.update([(10.0, 10.0, "ped")], 0.1)
    assert len(active) == 1
    assert active[0].category == "ped"


def test_tracker_multiple_objects_distinct_tracks() -> None:
    tr = WorldObjectTracker(match_m=2.0)
    tr.update([(0.0, 0.0, "a"), (10.0, 0.0, "b")], 0.1)
    n2 = tr.update([(0.5, 0.0, "a"), (10.5, 0.0, "b")], 0.1)
    assert len(n2) == 2
    ids = {t.track_id for t in n2}
    assert len(ids) == 2