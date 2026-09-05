"""Geometric LiDAR obstacle classification: tree / guardrail / wall."""

from __future__ import annotations

import numpy as np
import pytest

from beamng_autopilot.perception import (
    classify_lidar_obstacle, lidar_obstacles, obstacle_class_counts,
)


# --- pure classifier ------------------------------------------------------
def test_slender_and_tall_is_tree() -> None:
    assert classify_lidar_obstacle(0.4, 0.4, z_top=5.0, ground_z=0.0) == "tree"


def test_long_and_low_is_guardrail() -> None:
    assert classify_lidar_obstacle(6.0, 0.15, z_top=0.9, ground_z=0.0) \
        == "guardrail"


def test_long_and_tall_is_wall() -> None:
    assert classify_lidar_obstacle(5.0, 0.4, z_top=3.2, ground_z=0.0) == "wall"


def test_small_low_blob_is_unclassified() -> None:
    # a boulder / bush: neither tall enough for a tree nor long for a rail
    assert classify_lidar_obstacle(0.8, 0.8, z_top=1.0, ground_z=0.0) is None


def test_slender_short_pole_is_unclassified() -> None:
    # a 1.5 m bollard: slender but too short for the tree canopy class
    assert classify_lidar_obstacle(0.3, 0.3, z_top=1.5, ground_z=0.0) is None


def test_classifier_handles_slope_ground() -> None:
    # a tree on a hill: top is measured against the LOCAL ground reference
    assert classify_lidar_obstacle(0.4, 0.4, z_top=187.0, ground_z=182.0) \
        == "tree"


# --- end-to-end on a synthetic cloud --------------------------------------
def _cloud():
    """A tree, a guardrail and a wall around the ego at (0, 0, 0)."""
    rng = np.random.default_rng(5)
    pts = []
    # tree trunk+canopy at (12, 3): slender footprint, hits up to 6 m
    for _ in range(160):
        pts.append((12.0 + rng.uniform(-0.3, 0.3),
                    3.0 + rng.uniform(-0.3, 0.3),
                    rng.uniform(0.3, 6.0)))
    # guardrail at (10, -8): a 12 m long, low strip
    for x in np.linspace(4.0, 16.0, 120):
        pts.append((float(x) + rng.uniform(-0.1, 0.1),
                    -8.0 + rng.uniform(-0.15, 0.15),
                    rng.uniform(0.4, 0.9)))
    # wall at (-15, 0): a long, tall face
    for y in np.linspace(-8.0, 8.0, 140):
        pts.append((-15.0 + rng.uniform(-0.2, 0.2),
                    float(y),
                    rng.uniform(0.5, 3.5)))
    # ground scatter
    for _ in range(300):
        pts.append((rng.uniform(-20, 20), rng.uniform(-12, 12),
                    rng.uniform(-0.2, 0.1)))
    return np.asarray(pts, dtype=float)


def test_lidar_obstacles_carries_geometric_labels() -> None:
    obs = lidar_obstacles(_cloud(), np.array([0.0, 0.0, 0.0]))
    labels = sorted(o.label for o in obs if o.label in
                    ("tree", "guardrail", "wall"))
    assert "tree" in labels
    assert "guardrail" in labels
    assert "wall" in labels


def test_lidar_obstacles_classified_categories_unchanged() -> None:
    # downstream code filters on category == "lidar"; the class must go
    # into the label, never the category
    for ob in lidar_obstacles(_cloud(), np.array([0.0, 0.0, 0.0])):
        assert ob.category == "lidar"


# --- counts helper ---------------------------------------------------------
def test_obstacle_class_counts_and_nearest() -> None:
    from beamng_autopilot.perception import Obstacle
    obs = [
        Obstacle(x=10.0, y=0.0, half_w=0.4, half_h=0.4, label="tree"),
        Obstacle(x=5.0, y=2.0, half_w=0.3, half_h=0.3, label="tree"),
        Obstacle(x=3.0, y=-4.0, half_w=2.0, half_h=0.2, label="guardrail"),
    ]
    counts, nearest = obstacle_class_counts(obs, np.array([0.0, 0.0]))
    assert counts == {"tree": 2, "guardrail": 1, "wall": 0}
    assert nearest["tree"] == pytest.approx(5.39, abs=0.01)
    assert nearest["guardrail"] == pytest.approx(5.0, abs=0.01)
    assert nearest["wall"] is None
