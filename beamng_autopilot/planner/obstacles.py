"""Obstacle geometry, classification, and path-collision helpers."""

from __future__ import annotations

import math

import numpy as np

from .geometry import _seg_hits_box, _box_seg_dist, _seg_seg_dist
from .constants import SPECK_RAYCAST_MAX_M, LIDAR_PATH_CLEAR_M


def _inflated_boxes(obstacles, margin: float):
    boxes = []
    for ob in obstacles:
        hw = ob.half_w + margin
        hh = ob.half_h + margin
        if hw < 0.15 or hh < 0.15:
            continue
        boxes.append((ob.x, ob.y, hw, hh))
    return boxes


def _obstacle_oriented(ob) -> bool:
    """True when the obstacle carries a real oriented footprint."""
    return (ob.axis is not None and ob.half_len > 0.0
            and bool(getattr(ob, "half_len", 0.0)))


def _seg_hits_obstacle(ax, ay, bx, by, ob, half_w: float) -> bool:
    """True when a segment intersects an obstacle's actual footprint.

    Raycast walls keep their oriented footprint (axis + extents) so a
    diagonal roadside wall does not turn into a world-aligned square that
    falsely blocks the lane.  Other obstacle sources stay axis-aligned.
    """
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux

        def tr(px, py):
            dx, dy = px - ob.x, py - ob.y
            return dx * ux + dy * uy, dx * vx + dy * vy

        ax1, ay1 = tr(ax, ay)
        bx1, by1 = tr(bx, by)
        return _seg_hits_box(
            ax1, ay1, bx1, by1, 0.0, 0.0,
            ob.half_len + half_w, max(0.0, ob.half_thick) + half_w)
    return _seg_hits_box(ax, ay, bx, by, ob.x, ob.y,
                         ob.half_w + half_w, ob.half_h + half_w)


def _obstacle_corners(ob):
    """World-space corners of an obstacle's actual footprint."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        hu = ob.half_len
        hv = max(0.0, ob.half_thick)
        return (
            (ob.x + ux * hu + vx * hv, ob.y + uy * hu + vy * hv),
            (ob.x + ux * hu - vx * hv, ob.y + uy * hu - vy * hv),
            (ob.x - ux * hu + vx * hv, ob.y - uy * hu + vy * hv),
            (ob.x - ux * hu - vx * hv, ob.y - uy * hu - vy * hv),
        )
    return (
        (ob.x - ob.half_w, ob.y - ob.half_h),
        (ob.x + ob.half_w, ob.y - ob.half_h),
        (ob.x + ob.half_w, ob.y + ob.half_h),
        (ob.x - ob.half_w, ob.y + ob.half_h),
    )


def _obstacle_half_extents(ob, fwd, lat):
    """Half extents of an obstacle projected onto (fwd, lat) axes."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        thick = max(0.0, ob.half_thick)
        half_lon = (ob.half_len * abs(fwd[0] * ux + fwd[1] * uy)
                    + thick * abs(fwd[0] * vx + fwd[1] * vy))
        half_lat = (ob.half_len * abs(lat[0] * ux + lat[1] * uy)
                    + thick * abs(lat[0] * vx + lat[1] * vy))
        return half_lon, half_lat
    return (ob.half_w * abs(fwd[0]) + ob.half_h * abs(fwd[1]),
            ob.half_w * abs(lat[0]) + ob.half_h * abs(lat[1]))


def _obstacle_footprint_area(ob) -> float:
    """Footprint area (m^2) of an obstacle's own, uninflated box."""
    if _obstacle_oriented(ob):
        return 4.0 * max(0.0, ob.half_len) * max(0.0, ob.half_thick)
    return 4.0 * max(0.0, ob.half_w) * max(0.0, ob.half_h)


def _obstacle_seg_dist(ob, ax, ay, bx, by) -> float:
    """Closest distance from a segment to an obstacle's actual footprint."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux

        def tr(px, py):
            dx, dy = px - ob.x, py - ob.y
            return dx * ux + dy * uy, dx * vx + dy * vy

        ax1, ay1 = tr(ax, ay)
        bx1, by1 = tr(bx, by)
        return _box_seg_dist(0.0, 0.0, ob.half_len,
                             max(0.0, ob.half_thick),
                             ax1, ay1, bx1, by1)
    return _box_seg_dist(ob.x, ob.y, ob.half_w, ob.half_h,
                         ax, ay, bx, by)


def is_sparse_raycast_speck(ob) -> bool:
    """True when a raycast cluster is too sparse to act as a path blocker.

    A real wall or a dense trunk cluster comes back as an elongated box
    (labelled "wall") or a compact box several metres across.  A single
    hit point becomes a 0.9 x 0.9 m artefact box, and an unlabelled fat
    raycast blob is usually a few points from two surfaces fused into one
    box.  These are kept for a gentle speed limit but must not pin the
    path to blocked, otherwise the car parks in an open lane.
    """
    if getattr(ob, "category", "") != "raycast":
        return False
    if getattr(ob, "label", "") == "wall":
        return False
    if _obstacle_oriented(ob):
        length = 2.0 * float(getattr(ob, "half_len", 0.0))
        thick = 2.0 * float(getattr(ob, "half_thick", 0.0))
        if length > 4.5 and thick > 2.5:
            return True
        return length < SPECK_RAYCAST_MAX_M \
            and thick < SPECK_RAYCAST_MAX_M
    # Single-hit raycasts have no oriented spread; the 0.9 m box is the
    # min_size floor, not a measured footprint.
    return (2.0 * ob.half_w <= 2.1
            and 2.0 * ob.half_h <= 2.1)


def is_small_lidar_clutter(ob) -> bool:
    """True when a LiDAR cluster is small enough to be roadside clutter.

    Dense town scenes return dozens of small lidar boxes (poles, trunks,
    mailboxes, wall corners) that inflate the A* grid until no detour
    exists.  Like ``is_sparse_raycast_speck`` these are kept for a gentle
    speed limit but must not pin the path to blocked.  A real vehicle or
    pedestrian is still covered by the Lua vehicle/scenario scans and the
    vision channel, so dropping the small lidar boxes does not remove a
    safety layer - it removes grid noise.
    """
    if getattr(ob, "category", "") != "lidar":
        return False
    if getattr(ob, "label", "") == "wall":
        return False
    if _obstacle_oriented(ob):
        return (2.0 * float(getattr(ob, "half_len", 0.0)) <= 2.1
                and 2.0 * float(getattr(ob, "half_thick", 0.0)) <= 2.1)
    return (2.0 * ob.half_w <= 2.1
            and 2.0 * ob.half_h <= 2.1)


def _path_clear_m(ob, half_w: float) -> float:
    """Path-clearance gap for one obstacle.

    Lidar clusters are voxel-quantised roadside noise: they get the tight
    A* gap (car half width + 0.3 m) instead of the full safety margin, so
    the generated detour is not rejected by its own collision check.  The
    speed planner still applies the full clearance, so safety is kept.
    """
    if getattr(ob, "category", "") == "lidar":
        return LIDAR_PATH_CLEAR_M
    return half_w


def _obstacle_aabb(ob, half_w: float):
    """Axis-aligned bounding box of an obstacle inflated by ``half_w``."""
    if _obstacle_oriented(ob):
        ux, uy = float(ob.axis[0]), float(ob.axis[1])
        vx, vy = -uy, ux
        hu = ob.half_len + half_w
        hv = max(0.0, ob.half_thick) + half_w
        pts = (
            (ob.x + ux * hu + vx * hv, ob.y + uy * hu + vy * hv),
            (ob.x + ux * hu - vx * hv, ob.y + uy * hu - vy * hv),
            (ob.x - ux * hu + vx * hv, ob.y - uy * hu + vy * hv),
            (ob.x - ux * hu - vx * hv, ob.y - uy * hu - vy * hv),
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return (ob.x - ob.half_w - half_w, ob.y - ob.half_h - half_w,
            ob.x + ob.half_w + half_w, ob.y + ob.half_h + half_w)


def _path_hit_index(pts, i0: int, i1: int, obstacles, half_w: float) -> int:
    """Index of the first path vertex whose next segment is blocked.

    ``pts`` is the full route; only the ``[i0, i1]`` window is inspected.
    Returns -1 when no segment in the window is blocked.
    """
    n = len(pts)
    boxes = [_obstacle_aabb(ob, _path_clear_m(ob, half_w))
             for ob in obstacles]
    for i in range(i0, min(i1, n - 1)):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_min_x = min(ax, bx)
        seg_max_x = max(ax, bx)
        seg_min_y = min(ay, by)
        seg_max_y = max(ay, by)
        for k, ob in enumerate(obstacles):
            x0, y0, x1, y1 = boxes[k]
            if (seg_max_x < x0 or seg_min_x > x1
                    or seg_max_y < y0 or seg_min_y > y1):
                continue
            if _seg_hits_obstacle(ax, ay, bx, by, ob,
                                  _path_clear_m(ob, half_w)):
                return i
    return -1


def _find_blocker(pts, i0: int, i1: int, obstacles, half_w: float):
    """First obstacle whose footprint intrudes into the planning window."""
    n = len(pts)
    boxes = [_obstacle_aabb(ob, _path_clear_m(ob, half_w))
             for ob in obstacles]
    for i in range(i0, min(i1, n - 1)):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_min_x = min(ax, bx)
        seg_max_x = max(ax, bx)
        seg_min_y = min(ay, by)
        seg_max_y = max(ay, by)
        for k, ob in enumerate(obstacles):
            x0, y0, x1, y1 = boxes[k]
            if (seg_max_x < x0 or seg_min_x > x1
                    or seg_max_y < y0 or seg_min_y > y1):
                continue
            if _seg_hits_obstacle(ax, ay, bx, by, ob,
                                  _path_clear_m(ob, half_w)):
                return ob
    return None


def _vehicle_speed_along(ob, seg_pts, seg_k: int) -> float:
    """Signed speed of a dynamic vehicle along the local route segment.

    Used by speed planning so a moving lead vehicle does not force the ego
    to brake as hard as for a static wall.
    """
    if ob is None or ob.velocity is None or seg_pts is None:
        return 0.0
    if seg_k < 0 or seg_k >= len(seg_pts) - 1:
        return 0.0
    ax, ay = seg_pts[seg_k]
    bx, by = seg_pts[seg_k + 1]
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0
    return max(0.0, float((ob.velocity[0] * dx + ob.velocity[1] * dy) / n))
