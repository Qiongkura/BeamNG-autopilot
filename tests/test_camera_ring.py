"""Offline tests for the FSD-style camera ring definition (game-free).

Checks that the eight mounts cover the whole surround (pans around the
car), that every CameraModel produces finite intrinsics, and that the
front cameras all look forward while the rears look backward, plus the
local-frame axis convention used by the Tech provider.
"""

from __future__ import annotations

import pytest

import numpy as np

from beamng_autopilot.vision.ring import (
    CAMERA_RING,
    FRONT_FISHEYE,
    FRONT_MAIN,
    FRONT_NARROW,
    PILLAR_LEFT,
    PILLAR_RIGHT,
    REAR,
    REAR_LEFT,
    REAR_RIGHT,
    RING_ROLES,
    camera_ring_models,
    pan_deg_of,
)

W, H = 1076, 806


def test_ring_has_eight_roles() -> None:
    assert len(CAMERA_RING) == 8
    assert RING_ROLES == (
        FRONT_MAIN, FRONT_NARROW, FRONT_FISHEYE,
        PILLAR_LEFT, PILLAR_RIGHT, REAR_LEFT, REAR_RIGHT, REAR,
    )


def test_ring_models_registry_complete() -> None:
    models = camera_ring_models(W, H)
    assert set(models.keys()) == set(RING_ROLES)
    for role, cam in models.items():
        assert cam.width == W and cam.height == H
        assert np.isfinite(cam.fx) and cam.fx > 0.0
        assert np.isfinite(cam.cy) and cam.cy == H / 2.0


def test_pan_computed_matches_definition() -> None:
    """The computed pan (from the local forward vector) must equal the
    declared pan: the cameras actually point the way they should."""
    for mount in CAMERA_RING:
        cam = mount.camera_model(W, H)
        computed = pan_deg_of(cam)
        assert computed is not None
        assert abs(computed - mount.pan_deg) < 1e-6, (
            f"{mount.role}: defined {mount.pan_deg} computed {computed}")


def test_ring_covers_surround() -> None:
    """Pans span the full circle: front 0, left neg / right pos, rear 180."""
    pans = {mount.role: mount.pan_deg for mount in CAMERA_RING}
    assert pans[FRONT_MAIN] == 0.0
    assert pans[FRONT_NARROW] == 0.0
    assert pans[FRONT_FISHEYE] == 0.0
    # left cameras point left (negative pan, since pan + = right)
    assert pans[PILLAR_LEFT] < -90.0
    assert pans[REAR_LEFT] < -120.0
    # right cameras point right (positive pan)
    assert pans[PILLAR_RIGHT] > 90.0
    assert pans[REAR_RIGHT] > 120.0
    assert pans[REAR] == 180.0
    # physical coverage: at least one camera looking in each quadrant
    left_ish = [p for p in pans.values() if p < -60.0]
    right_ish = [p for p in pans.values() if p > 60.0]
    front_ish = [p for p in pans.values() if abs(p) <= 60.0]
    assert left_ish and right_ish and front_ish


def test_front_fisheye_wider_than_main() -> None:
    fovs = {mount.role: mount.fov_deg for mount in CAMERA_RING}
    assert fovs[FRONT_FISHEYE] > fovs[FRONT_MAIN]
    assert fovs[FRONT_MAIN] > fovs[FRONT_NARROW]


def test_offsets_spread_around_vehicle() -> None:
    """The ring should be physically spread round the car (front +y,
    rear -y, sides on +/-x in the vehicle local frame)."""
    by_role = {mount.role: mount for mount in CAMERA_RING}
    assert by_role[FRONT_MAIN].offset[1] > 0.0   # front of centre
    assert by_role[REAR].offset[1] < 0.0         # back of centre
    assert by_role[PILLAR_LEFT].offset[0] < 0.0  # left side
    assert by_role[PILLAR_RIGHT].offset[0] > 0.0  # right side
    assert by_role[REAR_LEFT].offset[0] < 0.0
    assert by_role[REAR_RIGHT].offset[0] > 0.0


def test_fwd_local_points_where_pan_says() -> None:
    """The local forward vector generated from pan must be a unit vector:
    the Tech provider negates Y for BeamNG, so only these computed values
    are what the sensor really points along."""
    for mount in CAMERA_RING:
        f = mount.fwd_local
        assert np.isclose(np.linalg.norm(f), 1.0, atol=1e-9)
        if mount.role == FRONT_MAIN:
            assert f[1] > 0.99  # mostly +Y (forward)
        if mount.role == REAR:
            assert f[1] < -0.99  # mostly -Y (backward)
        if mount.role == PILLAR_LEFT:
            assert f[0] < 0.0  # lateral component points left (-x)
        if mount.role == PILLAR_RIGHT:
            assert f[0] > 0.0  # lateral component points right (+x)
        if mount.role == REAR_LEFT:
            assert f[0] < 0.0 and f[1] < 0.0  # left + back
        if mount.role == REAR_RIGHT:
            assert f[0] > 0.0 and f[1] < 0.0  # right + back