"""Tech-only diagnostic: LiDAR obstacle fusion against the real instance.

Polls the 360 LiDAR on the running BeamNG.tech, runs the production
pipeline (lidar_obstacles + merge with the Lua sources) and prints cloud /
timing / box stats so the fusion can be tuned on real data.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.perception import (
    LidarClusterTracker,
    merge_obstacles,
    scan_obstacles_all,
    lidar_obstacles,
)
from beamng_autopilot_tech.providers import (
    CAMERA_DIR,
    CAMERA_UP,
    LIDAR_DENSITY,
    LIDAR_MAX_DIST,
    LIDAR_POS,
    LIDAR_VERTICAL_RES,
)


def main() -> None:
    conn = BeamNGConnector(port=64257)
    conn.open(launch=False)
    conn.attach_vehicle(already_open=True)
    from beamngpy.sensors import Lidar

    lidar = Lidar(
        "lidar_probe", conn.bng, conn.vehicle,
        requested_update_time=0.1,
        pos=LIDAR_POS, dir=CAMERA_DIR, up=CAMERA_UP,
        vertical_resolution=LIDAR_VERTICAL_RES,
        max_distance=LIDAR_MAX_DIST, density=LIDAR_DENSITY,
        is_360_mode=True, is_using_shared_memory=True,
        is_visualised=False,
    )
    tracker = LidarClusterTracker()
    ego = conn.vehicle.get_bbox()
    fl = np.asarray(ego["front_bottom_left"], dtype=float)[:2]
    fr = np.asarray(ego["front_bottom_right"], dtype=float)[:2]
    rl = np.asarray(ego["rear_bottom_left"], dtype=float)[:2]
    rr = np.asarray(ego["rear_bottom_right"], dtype=float)[:2]
    half_len = max(0.5, float(np.linalg.norm((fl + fr) / 2.0 - (rl + rr) / 2.0))) + 0.3
    half_w = max(0.5, float(np.linalg.norm((fr + rr) / 2.0 - (fl + rl) / 2.0))) + 0.3
    for i in range(15):
        conn.step(10)
        with conn.io_lock:
            data = lidar.poll()
        cloud = np.asarray(data.get("pointCloud"), dtype=float)
        st = conn.get_state()
        if cloud.ndim != 2 or len(cloud) == 0:
            print(f"[{i}] empty cloud")
            continue
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        heading = float(np.arctan2(float(st.dir[1]), float(st.dir[0])))
        t0 = time.perf_counter()
        boxes = lidar_obstacles(cloud, st.pos, radius=45.0,
                                self_rect=(half_len, half_w, heading))
        t1 = time.perf_counter()
        tracker.update(boxes, time.time())
        lua_obs, hits = scan_obstacles_all(conn.bng, conn.vehicle.vid,
                                           st.pos, radius=55.0,
                                           return_hits=True)
        merged = merge_obstacles(lua_obs + boxes)
        t2 = time.perf_counter()
        n_wall = sum(1 for b in boxes if b.label == "wall")
        n_moving = sum(1 for b in boxes
                       if b.velocity is not None
                       and np.linalg.norm(b.velocity) > 0.5)
        big = [b for b in boxes if max(b.half_w, b.half_h) > 6.0]
        print(f"[{i}] cloud={len(cloud)} lidar_boxes={len(boxes)} "
              f"(walls={n_wall} moving={n_moving}) "
              f"cluster={1000*(t1-t0):.0f}ms total={1000*(t2-t0):.0f}ms")
        print(f"    lua={len(lua_obs)} hits={len(hits)} merged={len(merged)}")
        for b in boxes[:10]:
            v = ("v=%.1f" % np.linalg.norm(b.velocity)) if b.velocity is not None else ""
            print(f"    lidar {b.label or 'obj':6s} x={b.x:.1f} y={b.y:.1f} "
                  f"hw={b.half_w:.1f} hh={b.half_h:.1f} len={b.half_len:.1f} {v}")
        if big:
            print("    !! BIG BOXES:")
            for b in big[:5]:
                print(f"    big x={b.x:.1f} y={b.y:.1f} hw={b.half_w:.1f} "
                      f"hh={b.half_h:.1f} label={b.label or '-'}")
    lidar.remove()
    conn.close()


if __name__ == "__main__":
    main()