"""Multi-frame confirmation for camera obstacle detections.

YOLO on a moving game frame can report a phantom "car" that is really a
roadside object or pavement texture; back-projecting a single frame turns
it into a blocker two metres in front of the ego.  A real static obstacle
keeps a stable world position while the ego moves, while these phantoms
ride along with the camera.  A track is therefore confirmed only after it
has been seen on two scans and has survived one ego-motion check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from beamng_autopilot.perception import Obstacle

_VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}


@dataclass
class VisionTrack:
    """A candidate vision detection matched across camera frames."""

    x: float
    y: float
    label: str = ""
    half_w: float = 0.0
    half_h: float = 0.0
    hits: int = 1
    last_seen: float = 0.0
    last_ego: tuple[float, float] | None = None
    motion_seen: bool = False


def _labels_match(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return a == b or (a in _VEHICLE_LABELS and b in _VEHICLE_LABELS)


def update_vision_tracks(
    tracks: list[VisionTrack],
    detections: list[Obstacle],
    ego_pos,
    now: float,
    match_m: float = 1.8,
    confirm_hits: int = 2,
    ttl_s: float = 8.0,
    ego_gate_m: float = 0.8,
    ride_along_ratio: float = 0.6,
) -> tuple[list[VisionTrack], list[Obstacle]]:
    """Match new detections to existing tracks and return confirmed ones.

    A track becomes a confirmed obstacle only after it has been seen
    ``confirm_hits`` times and has been observed while the ego moved by
    more than ``ego_gate_m`` without the detection riding along with it.
    Detections that move with the ego are treated as camera phantoms and
    restarted instead of accumulating confirmation.
    """
    ex, ey = float(ego_pos[0]), float(ego_pos[1])
    live = [tr for tr in tracks if now - tr.last_seen <= ttl_s]
    used: set[int] = set()
    out: list[VisionTrack] = []

    for ob in detections:
        best_i = -1
        best_d = float(match_m)
        for i, tr in enumerate(live):
            if i in used or not _labels_match(tr.label, ob.label or ob.category):
                continue
            d = math.hypot(ob.x - tr.x, ob.y - tr.y)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            out.append(VisionTrack(
                x=ob.x, y=ob.y, label=ob.label or ob.category,
                half_w=ob.half_w, half_h=ob.half_h,
                last_seen=now, last_ego=(ex, ey)))
            continue

        tr = live[best_i]
        used.add(best_i)
        ego_d = math.hypot(
            ex - tr.last_ego[0], ey - tr.last_ego[1]) if tr.last_ego else 0.0
        obs_d = math.hypot(ob.x - tr.x, ob.y - tr.y)
        rides_along = (ego_d > ego_gate_m
                       and obs_d > ride_along_ratio * ego_d
                       and obs_d > 0.5)
        if rides_along:
            tr.x, tr.y = ob.x, ob.y
            tr.label = ob.label or ob.category
            tr.half_w, tr.half_h = ob.half_w, ob.half_h
            tr.hits = 1
            tr.motion_seen = False
            tr.last_seen = now
            tr.last_ego = (ex, ey)
            out.append(tr)
            continue

        tr.x, tr.y = ob.x, ob.y
        tr.label = ob.label or ob.category
        tr.half_w, tr.half_h = ob.half_w, ob.half_h
        tr.hits += 1
        if ego_d > ego_gate_m:
            tr.motion_seen = True
        tr.last_seen = now
        tr.last_ego = (ex, ey)
        out.append(tr)

    for i, tr in enumerate(live):
        if i not in used:
            out.append(tr)

    confirmed = [
        Obstacle(x=tr.x, y=tr.y, half_w=tr.half_w, half_h=tr.half_h,
                 category="vision", label=tr.label)
        for tr in out
        if tr.hits >= confirm_hits and tr.motion_seen
    ]
    return out, confirmed
