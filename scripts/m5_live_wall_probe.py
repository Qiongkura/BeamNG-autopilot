"""Live static-wall probe: verify raycast, wall labelling and planning.

Attaches to the RUNNING session, spawns a real physics-capable TSStatic
wall ahead of the ego along its heading, then runs the exact perception +
planning stack used by ``m5_autopilot.py`` and reports whether the
raycast fan sees a ``wall`` obstacle and whether the local planner marks
the corridor blocked with a path truncated before the wall.  An annotated
camera + birdview frame is saved under ``logs/m5_wall/`` and the wall is
deleted afterwards, so the session is left exactly as it was.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import (
    errors_summary,
    scan_obstacles_all,
    scan_obstacles_raycast,
)
from beamng_autopilot.planner import LocalPlanner
from beamng_autopilot.vision.projection import default_camera
from beamng_autopilot.vision.detection import live_camera_model
from beamng_autopilot.vision.projection import CameraModel
from beamng_autopilot.visionview import (
    render_birdview,
    render_camera_overlay,
)

WALL_DIST = 28.0
WALL_LEN_M = 12.0
WALL_HEIGHT_M = 0.94
WALL_THICK_M = 1.8
RAY_RADIUS = 55.0
RAY_COUNT = 90
PIECES = 4
PIECE_GAP = 0.15
SHAPE = "/art/shapes/objects/jerseybarrier_3m.dae"


def _lua_spawn_wall(conn, pos, heading: float) -> list[dict]:
    x, y, z = (float(v) for v in pos)
    # A single scaled TSStatic only exposes a small collision patch around
    # its origin, so the probe wall is built from several unscaled jersey
    # barriers.  The low ray fan (0.45 m) sees this wall; the pieces are
    # deleted again before the session is closed.
    chunk = (
        "local q = quatFromEuler(0, 0, %(heading)s + math.pi / 2.0)\n"
        "local px, py = -math.sin(%(heading)s), math.cos(%(heading)s)\n"
        "local cx, cy = %(cx)s, %(cy)s\n"
        "local z = %(z)s\n"
        "local n = %(n)s\n"
        "local gap = %(gap)s\n"
        "local total = %(total)s\n"
        "local piece = (total - (n - 1) * gap) / n\n"
        "local start = -total / 2.0\n"
        "local ids = {}\n"
        "for i = 0, n - 1 do\n"
        "  local along = start + i * (piece + gap)\n"
        "  local wx = cx + px * along\n"
        "  local wy = cy + py * along\n"
        "  local obj = createObject('TSStatic')\n"
        "  obj:setField('shapeName', 0, '%(shape)s')\n"
        "  obj.scale = vec3(1, 1, 1)\n"
        "  obj.useInstanceRenderData = true\n"
        "  obj:setField('instanceColor', 0, '1 1 1 1')\n"
        "  obj:setField('collisionType', 0, 'Collision Mesh')\n"
        "  obj:setField('decalType', 0, 'Collision Mesh')\n"
        "  obj.canSave = false\n"
        "  obj:registerObject('wall_probe_piece_' .. i)\n"
        "  local grp = scenetree.MissionGroup\n"
        "  if grp then grp:addObject(obj) end\n"
        "  obj:setPosRot(wx, wy, z, q.x, q.y, q.z, q.w)\n"
        "  pcall(function() obj:enableCollision() end)\n"
        "  ids[#ids + 1] = {i = i, id = tostring(obj:getId()), "
        "x = wx, y = wy}\n"
        "end\n"
        "return jsonEncode(ids)"
        % {
            "heading": heading,
            "cx": x + math.cos(heading) * WALL_DIST,
            "cy": y + math.sin(heading) * WALL_DIST,
            "z": z - 0.35,
            "n": PIECES,
            "gap": PIECE_GAP,
            "total": WALL_LEN_M,
            "shape": SHAPE,
        }
    )
    try:
        resp = conn.bng.queue_lua_command(chunk, response=True)
        return json.loads(str(resp)) or []
    except (ValueError, TypeError) as exc:
        print(f"[probe] wall spawn response not JSON: {resp!r} ({exc})")
        return []


def _lua_delete_wall(conn) -> int:
    try:
        resp = conn.bng.queue_lua_command(
            "local n = 0\n"
            "for i = 0, %d - 1 do\n"
            "  local o = scenetree.findObject('wall_probe_piece_' .. i)\n"
            "  if o and o.delete then o:delete() n = n + 1 end\n"
            "end\n"
            "return jsonEncode({deleted = n})" % PIECES, response=True)
        return int(json.loads(str(resp)).get("deleted", 0))
    except Exception as exc:
        print(f"[probe] wall delete failed: {exc}")
        return 0


def _save_overlay(conn, st, route, drive, obstacles):
    out_dir = config.LOGS_DIR / "m5_wall"
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_w, cam_h = 1076, 806
    try:
        try:
            img = conn.grab_screen()
            if img is None or getattr(img, "size", 0) == 0:
                raise RuntimeError("empty screenshot")
        except Exception as exc:
            print(f"[probe] screenshot fallback: {exc}")
            img = np.zeros((cam_h, cam_w, 3), np.uint8)
        h, w = img.shape[:2]
        if (w, h) == (cam_w, cam_h):
            cam_model = default_camera(cam_w, cam_h)
        else:
            vmodel = live_camera_model(conn.bng, w, h, st.pos, st.heading)
            cam_model = CameraModel(
                offset=vmodel.offset,
                fwd_local=vmodel.fwd_local,
                up_local=vmodel.up_local,
                fov_deg=vmodel.fov_deg,
                width=cam_w,
                height=cam_h,
            )
        img = cv2.resize(img, (cam_w, cam_h))
        if drive is not None and len(drive) >= 2:
            img = render_camera_overlay(
                img, drive, st.pos, st.heading, cam_model,
                obstacles=obstacles)
        bv = np.full((cam_h, cam_h, 3), (22, 24, 30), np.uint8)
        render_birdview(
            bv, route_xy=drive, obstacles=obstacles,
            goal_xy=route[-1][:2] if len(route) else None,
            pos=st.pos, heading=st.heading)
        frame = np.hstack([img, bv])
        p = out_dir / (
            f"wall_probe_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(str(p), frame)
        print(f"[probe] overlay frame -> {p}")
    except Exception as exc:
        print(f"[probe] overlay frame error: {exc}")
        traceback.print_exc()


def main() -> None:
    conn = BeamNGConnector(config.DEFAULT_MAP, config.DEFAULT_VEHICLE,
                           port=config.PORT)
    wall_ids: list[dict] = []
    try:
        conn.open(launch=False)
        conn.attach_vehicle(vid=None, already_open=True)
        st = conn.get_state()
        pos = st.pos
        heading = float(st.heading)
        hx, hy = math.cos(heading), math.sin(heading)
        print(f"[probe] ego pos=({pos[0]:.1f}, {pos[1]:.1f}) "
              f"heading={heading:.2f} speed={st.speed:.2f}")

        obs0 = scan_obstacles_all(conn.bng, conn.vehicle.vid, pos,
                                  radius=RAY_RADIUS)
        print(f"[probe] baseline before wall: {len(obs0)} obstacles "
              f"(errors={errors_summary()!r})")

        wx = pos[0] + hx * WALL_DIST
        wy = pos[1] + hy * WALL_DIST
        wz = float(pos[2]) - 0.35
        wall_ids = _lua_spawn_wall(
            conn, (float(pos[0]), float(pos[1]), float(pos[2])), heading)
        if not wall_ids:
            print("[probe] FAILED to spawn wall - aborting")
            return
        print(f"[probe] wall spawned: {len(wall_ids)} pieces at "
              f"({wx:.1f}, {wy:.1f}) z={wz:.1f}")
        for piece in wall_ids:
            print(f"[probe]   piece {piece['i']}: id={piece['id']} "
                  f"at ({piece['x']:.1f}, {piece['y']:.1f})")
        time.sleep(1.0)
        conn.step(20)

        st = conn.get_state()
        rays = scan_obstacles_raycast(conn.bng, st.pos,
                                      radius=RAY_RADIUS, rays=RAY_COUNT)
        obstacles = scan_obstacles_all(
            conn.bng, conn.vehicle.vid, st.pos, radius=RAY_RADIUS)
        print(f"[probe] raycast={len(rays)} merged={len(obstacles)} "
              f"(errors={errors_summary()!r})")
        wall_obs = None
        wall_near = None
        for o in obstacles:
            d = float(np.hypot(o.x - st.pos[0], o.y - st.pos[1]))
            lon = (o.x - st.pos[0]) * hx + (o.y - st.pos[1]) * hy
            lat = (o.x - st.pos[0]) * (-hy) + (o.y - st.pos[1]) * hx
            print(f"[probe]   obs d={d:6.1f}m lon={lon:+6.1f} "
                  f"lat={lat:+6.1f} box=({o.half_w:.1f}x{o.half_h:.1f}) "
                  f"cat={o.category} label={o.label!r}")
            near = math.hypot(o.x - wx, o.y - wy)
            if near < 8.0 and wall_near is None:
                wall_near = o
            if o.label == "wall" and near < 15.0 and wall_obs is None:
                wall_obs = o

        route = np.array([
            [st.pos[0] + hx * d, st.pos[1] + hy * d]
            for d in np.arange(0.0, 70.0, 1.5)
        ])
        planner = LocalPlanner()
        drive, blocked = planner.plan(route, obstacles, st.pos, heading, 0)
        drive = np.asarray(drive, dtype=float)
        path_end = 999.0
        if len(drive) >= 2:
            path_end = float(np.linalg.norm(
                drive[-1, :2] - st.pos[:2]))
        speed, obs_dist = planner.speed(drive, obstacles, st.pos, heading,
                                        0, 9.0)
        print(f"[probe] planner mode={planner.last_mode} "
              f"blocked={blocked} speed={speed:.1f} m/s "
              f"nearest={obs_dist:.1f} m, path pts={len(drive)} "
              f"path_end={path_end:.1f} m wall_at={WALL_DIST:.1f} m")
        if planner.last_blocker is not None:
            print(f"[probe] blocker={planner.last_blocker}")

        _save_overlay(conn, st, route, drive, obstacles)

        wall_seen = wall_obs is not None or (
            wall_near is not None and wall_near.label == "wall")
        ok = wall_seen and planner.last_mode == "blocked" and blocked
        ok = ok and path_end < WALL_DIST
        if wall_near is not None:
            print(f"[probe] wall box near centre: "
                  f"({wall_near.half_w:.1f}x{wall_near.half_h:.1f}) "
                  f"label={wall_near.label!r} labelled={wall_obs is not None}")
        if wall_obs is not None:
            near = float(np.hypot(wall_obs.x - wx, wall_obs.y - wy))
            print(f"[probe] labelled wall box: "
                  f"({wall_obs.half_w:.1f}x{wall_obs.half_h:.1f}) "
                  f"label={wall_obs.label!r} centre_delta={near:.1f} m")
        result = ("PASS - wall seen and corridor blocked" if ok
                  else "FAIL - wall perception/planning did not react")
        print(f"[probe] RESULT: {result}")
    finally:
        if wall_ids:
            removed = _lua_delete_wall(conn)
            time.sleep(0.5)
            try:
                conn.step(10)
            except Exception:
                pass
            print(f"[probe] wall removed: {removed} pieces")
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
