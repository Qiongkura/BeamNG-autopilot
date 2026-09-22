"""T05 geometry audit: independent oracle, error budget, resolution limits.

The plan's T05 asks for three things this script produces:

1. an **oracle independent of the code under test** - the ground point of a
   pixel is derived in closed form HERE (from the camera height, the focal
   length and the row index), not by calling the project's projection;
   two implementations sharing one formula would not be evidence;
2. a **metric error distribution** for the cases the plan lists: level and
   tilted vehicle attitude, two camera heights (main view vs fisheye), a
   sloped plane, and the lens edge;
3. the **resolution-distance budget** (survey L39: ``Np = C·d/w``): how far
   a 0.1 m marking can be resolved at each camera's resolution, and where
   that puts the near-field blind zone and the far-field reference.

Nothing here drives, trains or writes outside ``--json``.  It is a
measurement of the geometry baseline, not of driving behaviour.

Usage::

    .venv\\Scripts\\python.exe scripts/m5_geometry_audit.py --json logs/geom.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402

# Camera models come from the REAL ring definition (mount offsets, tilts,
# FOVs) at the live capture resolution - a synthetic mount would only audit
# my own arithmetic.  ``ring.camera_ring_models`` is the same builder the
# runtime uses.
from beamng_autopilot.vision.ring import (  # noqa: E402
    FRONT_FISHEYE,
    FRONT_MAIN,
    camera_ring_models,
)

RING = camera_ring_models(320, 240)
MAIN = RING[FRONT_MAIN]
FISHEYE = RING[FRONT_FISHEYE]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def oracle_ground_point(u: float, v: float, cam: CameraModel,
                        offset=None, fwd_local=None, up_local=None,
                        extra_pitch_rad: float = 0.0,
                        ground_gap_m: float | None = None
                        ) -> tuple[float, float] | None:
    """Independent closed-form ground hit of pixel (u, v), any mount.

    Built from first principles and expressed in the CAR frame
    (x forward, y left, z up) - never through the project's projection code,
    so a shared formula error cannot hide.  The mount is given as the ring
    stores it (offset and axes in the vehicle's (right, forward, up) local
    frame); this function does its own orthonormalisation instead of
    trusting the stored axes, and its own plane intersection.

    The ground is the horizontal plane through the contact patch, which in
    the origin frame sits at ``z = -EGO_GROUND_GAP_M`` (the origin is ABOVE
    the road - treating it as z = 0 was one of the harness bugs this script
    found; it was worth 0.3-0.7 m at 3-6 m range).

    Returns ``(forward_m, lateral_m)`` FROM THE VEHICLE ORIGIN, or ``None``
    when the ray does not descend to that plane.
    """
    fx = float(cam.fx)
    fy = float(cam.fy)
    if fx <= 0.0 or fy <= 0.0:
        return None
    gap = G.EGO_GROUND_GAP_M if ground_gap_m is None else float(ground_gap_m)
    off = np.asarray(cam.offset if offset is None else offset, dtype=float)
    f_loc = np.asarray(cam.fwd_local if fwd_local is None else fwd_local,
                       dtype=float)
    u_loc = np.asarray(cam.up_local if up_local is None else up_local,
                       dtype=float)
    # ring local (x right, y forward, z up) -> car frame (x fwd, y left, z up)
    def _to_car(vec):
        return np.array([vec[1], -vec[0], vec[2]], dtype=float)

    F = _to_car(f_loc)
    F = F / np.linalg.norm(F)
    U = _to_car(u_loc)
    if abs(float(extra_pitch_rad)) > 1e-12:
        # an extra tilt about the camera's right axis (positive = down)
        R = np.cross(F, U)
        R = R / np.linalg.norm(R)
        cp, sp = math.cos(float(extra_pitch_rad)), math.sin(
            float(extra_pitch_rad))
        F2 = F * cp + U * (-sp)
        U2 = U * cp + F * sp
        F, U = F2 / np.linalg.norm(F2), U2 / np.linalg.norm(U2)
    R = np.cross(F, U)
    R = R / np.linalg.norm(R)
    U = np.cross(R, F)
    P = _to_car(off)
    # image +u is the camera's RIGHT (R already points that way in the car
    # frame, being cross(forward, up)); flipping this sign inflated every
    # off-centre pixel's error to metres
    dirv = (F
            + R * ((float(u) - float(cam.cx)) / fx)
            + U * ((float(cam.cy) - float(v)) / fy))
    if float(dirv[2]) >= -1e-12:
        return None
    t = ((-gap) - float(P[2])) / float(dirv[2])
    if t <= 0.0:
        return None
    q = P + t * dirv
    return (float(q[0]), float(q[1]))


def _project_lateral_forward(cam: CameraModel, pos, u, v, ground_z,
                             pitch_rad: float = 0.0):
    """The PROJECT's answer for the same pixel, via its own code path.

    ``pitch_rad`` tilts the camera exactly like the oracle does, so the two
    differ only in HOW they compute the intersection.
    """
    from beamng_autopilot.vision.lanes import _back_project_many
    pts, ok = _back_project_many(np.array([float(u)]), np.array([float(v)]),
                                 cam, pos, 0.0, float(ground_z),
                                 pitch_rad=float(pitch_rad))
    if not bool(ok[0]):
        return None
    x, y = float(pts[0][0]), float(pts[0][1])
    # Heading 0 in the calls above means the camera's forward axis is world
    # +x and its left is world +y: the car-frame FORWARD offset is dx and
    # the LATERAL offset is dy.  (Getting this pair the wrong way round was
    # the first version's bug - it compared lateral to forward and reported
    # metre-scale "errors".)
    return (x - float(pos[0]), y - float(pos[1]))


def audit_oracle(cam: CameraModel, name: str,
                 pitches_deg=(0.0, 2.0, -3.0),
                 rows=(40, 100, 160, 200, 232),
                 cols=(10, 80, 160, 240, 310),
                 usable_range_m: float = 60.0) -> dict:
    """Oracle vs project, split into a USABLE band and a horizon band.

    A pixel row a fraction of a degree above the ground horizon maps to a
    point hundreds of metres away, so the ground model is ill-conditioned
    there: a 0.001 deg attitude difference moves the answer by metres.  The
    audit therefore reports the rows whose ground distance is inside
    ``usable_range_m`` separately from the rows that graze the horizon -
    folding the two together would produce a "median error" that describes
    neither regime.

    The camera height above the ROAD is a mount property:
    ``mount_up + EGO_GROUND_GAP_M``.  (Passing the mount height as the
    camera height was the first harness bug; the oracle is only evidence
    once its inputs are the same geometry.)
    """
    off = np.asarray(cam.offset, dtype=float)
    fwd_local = np.asarray(cam.fwd_local, dtype=float)
    up_local = np.asarray(cam.up_local, dtype=float)
    # camera height above the road = mount offset's up component (the ring
    # carries the offset in the mount frame) + the origin gap
    cam_z = float(np.array([off[1], -off[0], off[2]])[2])
    mount_pitch = math.degrees(math.atan2(-float(fwd_local[2]),
                                         float(fwd_local[1])))
    h_cam = cam_z + G.EGO_GROUND_GAP_M
    rows_out = []
    for pitch in pitches_deg:
        usable, horizon, misses = [], [], 0
        for v in rows:
            for u in cols:
                pos = np.array([0.0, 0.0, G.EGO_GROUND_GAP_M])
                got = _project_lateral_forward(
                    cam, pos, u, v, G.ego_ground_z(pos),
                    pitch_rad=math.radians(pitch))
                if got is None:
                    misses += 1
                    continue
                ref = oracle_ground_point(
                    u, v, cam, off, fwd_local, up_local,
                    extra_pitch_rad=math.radians(pitch))
                if ref is None:
                    misses += 1
                    continue
                # both sides are (forward, lateral) from the origin
                err = math.hypot(got[0] - ref[0], got[1] - ref[1])
                (usable if got[0] <= usable_range_m else horizon).append(err)
        def _stat(vals):
            a = np.asarray(vals, dtype=float)
            if not len(a):
                return {"n": 0, "p50_m": None, "p95_m": None, "max_m": None}
            return {"n": int(len(a)),
                    "p50_m": round(float(np.median(a)), 4),
                    "p95_m": round(float(np.percentile(a, 95)), 4),
                    "max_m": round(float(a.max()), 4)}
        rows_out.append({
            "camera": name, "camera_height_m": round(h_cam, 3),
            "mount_pitch_deg": round(float(mount_pitch), 3),
            "extra_pitch_deg": float(pitch),
            "usable_range_m": float(usable_range_m),
            "usable": _stat(usable), "horizon_band": _stat(horizon),
            "misses": int(misses),
        })
    return {"rows": rows_out}


def _oracle_on_slope(u: float, v: float, cam: CameraModel, offset,
                     fwd_local, up_local, slope_rad: float,
                     extra_pitch_rad: float = 0.0,
                     ground_gap_m: float | None = None) -> float | None:
    """Forward distance from the origin to a plane TILTED by ``slope_rad``.

    Same first-principles basis as ``oracle_ground_point`` but with the
    plane ``z = -gap + d*tan(slope)`` (d = forward distance), so the result
    isolates the FLAT-GROUND assumption's error from any difference between
    the two implementations.
    """
    fx = float(cam.fx)
    fy = float(cam.fy)
    if fx <= 0.0 or fy <= 0.0:
        return None
    gap = G.EGO_GROUND_GAP_M if ground_gap_m is None else float(ground_gap_m)
    off = np.asarray(offset, dtype=float)
    f_loc = np.asarray(fwd_local, dtype=float)
    u_loc = np.asarray(up_local, dtype=float)

    def _to_car(vec):
        return np.array([vec[1], -vec[0], vec[2]], dtype=float)

    F = _to_car(f_loc)
    F = F / np.linalg.norm(F)
    U = _to_car(u_loc)
    if abs(float(extra_pitch_rad)) > 1e-12:
        cp, sp = math.cos(float(extra_pitch_rad)), math.sin(
            float(extra_pitch_rad))
        F2, U2 = F * cp + U * (-sp), U * cp + F * sp
        F, U = F2 / np.linalg.norm(F2), U2 / np.linalg.norm(U2)
    R = np.cross(F, U)
    R = R / np.linalg.norm(R)
    U = np.cross(R, F)
    P = _to_car(off)
    dirv = (F + R * ((float(u) - float(cam.cx)) / fx)
            + U * ((float(cam.cy) - float(v)) / fy))
    dz = float(dirv[2])
    if dz >= -1e-12:
        return None
    dx = float(dirv[0])
    tn = math.tan(float(slope_rad))
    denom = dz - tn * dx
    if abs(denom) < 1e-12:
        return None
    s_ = (-gap - float(P[2]) + tn * float(P[0])) / denom
    if s_ <= 0.0:
        return None
    return float(P[0] + s_ * dx)


def audit_slope(cam: CameraModel, name: str,
                slopes_deg=(0.0, 1.0, 2.0, 5.0, 8.0),
                extra_pitch_deg: float = 0.0,
                rows=(170, 190, 210, 225, 232)) -> dict:
    """What the FLAT-GROUND assumption costs on a sloped road (modelling error).

    Both sides of this comparison are the independent oracle, so the number
    is the assumption's error alone, not an implementation difference.
    """
    off = np.asarray(cam.offset, dtype=float)
    fwd_local = np.asarray(cam.fwd_local, dtype=float)
    up_local = np.asarray(cam.up_local, dtype=float)
    gap = G.EGO_GROUND_GAP_M
    out = []
    for slope in slopes_deg:
        errs = []
        for v in rows:
            flat = oracle_ground_point(
                cam.cx, v, cam, off, fwd_local, up_local,
                extra_pitch_rad=math.radians(extra_pitch_deg))
            sloped = _oracle_on_slope(
                cam.cx, v, cam, off, fwd_local, up_local,
                math.radians(float(slope)),
                extra_pitch_rad=math.radians(extra_pitch_deg))
            if flat is None or sloped is None:
                continue
            errs.append(abs(flat[0] - sloped))
        if errs:
            a = np.asarray(errs)
            out.append({"camera": name, "slope_deg": float(slope),
                        "extra_pitch_deg": float(extra_pitch_deg),
                        "n": int(len(a)),
                        "p50_m": round(float(np.median(a)), 4),
                        "max_m": round(float(a.max()), 4)})
    return {"rows": out}


def budget(cam: CameraModel, name: str, mark_w_m: float = 0.1,
           min_px=(1.0, 2.0, 4.0)) -> dict:
    """Resolution-distance budget + blind zone for one camera."""
    p = np.array([0.0, 0.0, 1.42 + G.EGO_GROUND_GAP_M])
    near = G.nearest_ground_distance_m(cam, p)
    return {
        "camera": name,
        "fx_px": round(float(cam.fx), 2),
        "fov_deg": float(cam.fov_deg) if hasattr(cam, "fov_deg") else None,
        "width_px": int(cam.width),
        "mark_width_m": float(mark_w_m),
        "max_distance_m": {f"min_{m:g}px": (
            None if math.isnan(G.resolution_distance_m(cam.fx, mark_w_m, m))
            else round(G.resolution_distance_m(cam.fx, mark_w_m, m), 2))
            for m in min_px},
        "nearest_ground_m": (None if near is None else round(near, 2)),
        "ground_model": G.GROUND_MODEL_FLAT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T05 geometry audit")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    report: dict = {
        "ground_model": G.GROUND_MODEL_FLAT,
        "ego_ground_gap_m": G.EGO_GROUND_GAP_M,
        "ground_plane_unified": bool(G.GROUND_PLANE_UNIFIED),
        "pose_rotation_enabled": bool(G.POSE_ROTATION_ENABLED),
        "pose_label": G.pose_label(None),
        "budgets": [budget(MAIN, FRONT_MAIN), budget(FISHEYE, FRONT_FISHEYE),
                    budget(MAIN.camera_model_640()
                           if hasattr(MAIN, "camera_model_640")
                           else CameraModel(offset=np.asarray(MAIN.offset),
                                            fwd_local=np.asarray(MAIN.fwd_local),
                                            up_local=np.asarray(MAIN.up_local),
                                            fov_deg=float(MAIN.fov_deg),
                                            width=640, height=480),
                           "front_main@640x480")],
        "oracle": {},
        "slope": {},
    }
    for role, cam in RING.items():
        report["oracle"][role] = audit_oracle(cam, role)
        report["slope"][role] = audit_slope(cam, role)

    print(f"[geom] ground model={report['ground_model']} "
          f"gap={report['ego_ground_gap_m']} m "
          f"unified={report['ground_plane_unified']} "
          f"rotation={report['pose_rotation_enabled']}")
    for b in report["budgets"]:
        print(f"[geom] {b['camera']:16s} fx={b['fx_px']:6.1f}px "
              f"nearest_ground={b['nearest_ground_m']} m  "
              f"0.1 m marking resolvable to " + ", ".join(
                  f"{k}={v} m" for k, v in b["max_distance_m"].items()))
    for name, rep in report["oracle"].items():
        print(f"[geom] oracle vs project ({name}, rows whose ground hit is "
              f"<= {rep['rows'][0]['usable_range_m']} m):")
        for row in rep["rows"]:
            u, hb = row["usable"], row["horizon_band"]
            print(f"        h={row['camera_height_m']} m "
                  f"mount_pitch={row['mount_pitch_deg']:+.2f} "
                  f"extra_pitch={row['extra_pitch_deg']:+.1f} deg | "
                  f"usable n={u['n']:3d} p50={u['p50_m']} p95={u['p95_m']} "
                  f"max={u['max_m']} m | horizon-band n={hb['n']} "
                  f"max={hb['max_m']} m | misses={row['misses']}")
    for name, rep in report["slope"].items():
        print(f"[geom] flat-plane assumption error ({name}) for a sloped road:")
        for row in rep["rows"]:
            print(f"        slope={row['slope_deg']:.1f} deg  n={row['n']:3d} "
                  f"p50={row['p50_m']:.3f} max={row['max_m']:.3f} m")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
        print(f"[geom] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
