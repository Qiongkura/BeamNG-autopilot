"""BeamNG.tech sensor providers built on beamngpy Camera/LiDAR sensors."""

from __future__ import annotations

import os
import time

import numpy as np

from beamng_autopilot.perception import (
    LidarClusterTracker,
    Obstacle,
    _cluster_points,
    _split_raycast_sectors,
    downsample_cloud,
    last_error,
    lidar_obstacles,
    merge_obstacles,
    scan_obstacles_all,
)
from beamng_autopilot.runtime import CameraProvider, RangeProvider, RangeSample
from beamng_autopilot.vision.projection import CameraModel

# Sensor names must be unique per game instance: two Python processes
# attaching to the same BeamNG.tech (e.g. the lane viewer and the autopilot,
# or a stale viewer left running) would collide on a shared name and one of
# them gets None/empty polls.  A per-process suffix makes cross-process
# collisions impossible; within one process the name stays stable so
# reconnects do not leak sensors.
_PID = os.getpid()


CAMERA_POS = (0.0, -1.216, 1.386)
CAMERA_DIR = (0.0, -0.99994, -0.0112)
CAMERA_UP = (0.0, -0.0112, 0.99994)
CAMERA_FOV_DEG = 65.0

LIDAR_POS = (0.0, 0.0, 1.7)
LIDAR_VERTICAL_RES = 16
LIDAR_MAX_DIST = 120.0
LIDAR_DENSITY = 12
LIDAR_POLL_RETRIES = 3
LIDAR_MAX_POINTS = 6000
LIDAR_SELF_MARGIN = 0.3
LIDAR_OBSTACLE_RADIUS = 45.0

BLACK_FRAME_MAX_MEAN = 1.0
BLACK_FRAME_RETRIES = 2
BLACK_FRAME_RETRY_DELAY_S = 0.1


def _frame_is_black(frame, max_mean: float = BLACK_FRAME_MAX_MEAN) -> bool:
    """True when a camera frame is an all-black/stale shared-memory read."""
    if frame is None or frame.size == 0:
        return True
    return float(np.mean(np.asarray(frame, dtype=np.float32))) < max_mean


class TechCameraProvider(CameraProvider):
    """Front camera attached to the ego vehicle on BeamNG.tech."""

    def __init__(self, conn, width: int = 1076, height: int = 806,
                 annotations: bool = False) -> None:
        from beamngpy.sensors import Camera

        self.conn = conn
        self.name = f"autopilot_front_{_PID}"
        self.annotations = bool(annotations)
        with conn.io_lock:
            self.camera = Camera(
                self.name,
                conn.bng,
                conn.vehicle,
                requested_update_time=0.05,
                pos=CAMERA_POS,
                dir=CAMERA_DIR,
                up=CAMERA_UP,
                resolution=(int(width), int(height)),
                field_of_view_y=CAMERA_FOV_DEG,
                near_far_planes=(0.05, 150.0),
                is_using_shared_memory=True,
                is_render_colours=True,
                is_render_annotations=self.annotations,
                is_render_instance=False,
                is_render_depth=False,
                is_visualised=False,
            )
        self.width = int(width)
        self.height = int(height)

    def _poll(self) -> dict:
        last_error: RuntimeError | None = None
        for attempt in range(BLACK_FRAME_RETRIES):
            with self.conn.io_lock:
                data = self.camera.poll()
            image = data.get("colour")
            if image is None:
                last_error = RuntimeError(
                    "tech camera returned no colour frame")
            else:
                frame = np.ascontiguousarray(
                    np.asarray(image), dtype=np.uint8).copy()
                if not _frame_is_black(frame):
                    return data
                last_error = RuntimeError(
                    "tech camera returned a black frame")
            if attempt + 1 < BLACK_FRAME_RETRIES:
                time.sleep(BLACK_FRAME_RETRY_DELAY_S)
        raise last_error

    def grab(self) -> np.ndarray:
        data = self._poll()
        return np.ascontiguousarray(
            np.asarray(data.get("colour")), dtype=np.uint8).copy()

    def grab_annotated(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (RGB frame, annotation frame); requires annotations=True."""
        data = self._poll()
        rgb = np.ascontiguousarray(
            np.asarray(data.get("colour")), dtype=np.uint8).copy()
        ann = data.get("annotation")
        if ann is None:
            raise RuntimeError("tech camera returned no annotation frame "
                               "(annotations mode not enabled)")
        return rgb, np.ascontiguousarray(
            np.asarray(ann), dtype=np.uint8).copy()

    def camera_model(self, pos, heading, width, height,
                     fallback: CameraModel | None = None,
                     rotation=None) -> CameraModel:
        # Real-vehicle logic (Tesla / Xpeng style): sensor extrinsics are
        # fixed by calibration and only the vehicle's full 6-DOF pose (the
        # rotation quaternion) is fed in at runtime.  The pose is applied
        # inside CameraModel.camera_pose(), so slopes (pitch/roll) are
        # handled correctly and no per-frame GE round-trips are needed.
        # (The earlier get_position/get_direction approach cost two
        # round-trips per frame and its 0.15 s cache could not survive the
        # 6 Hz viewer loop, so it was replaced by this zero-query design.)
        #
        # CAMERA_* are in BeamNG vehicle-local axes (forward = -Y); the
        # core CameraModel uses (right, fwd, up) local axes, so the y
        # components are negated.
        return CameraModel(
            offset=np.array([CAMERA_POS[0], -CAMERA_POS[1], CAMERA_POS[2]]),
            fwd_local=np.array([CAMERA_DIR[0], -CAMERA_DIR[1],
                                CAMERA_DIR[2]]),
            up_local=np.array([CAMERA_UP[0], -CAMERA_UP[1], CAMERA_UP[2]]),
            fov_deg=CAMERA_FOV_DEG,
            width=int(width),
            height=int(height),
        )

    def close(self) -> None:
        with self.conn.io_lock:
            self.camera.remove()


class TechRangeProvider(RangeProvider):
    """LiDAR plus scenario/vehicle obstacle sources on BeamNG.tech."""

    def __init__(self, conn) -> None:
        from beamngpy.sensors import Lidar

        self.conn = conn
        self.name = f"autopilot_lidar_{_PID}"
        with conn.io_lock:
            self._ego_half_len, self._ego_half_w = self._ego_extents()
            self.lidar = Lidar(
                self.name,
                conn.bng,
                conn.vehicle,
                requested_update_time=0.1,
                pos=LIDAR_POS,
                dir=CAMERA_DIR,
                up=CAMERA_UP,
                vertical_resolution=LIDAR_VERTICAL_RES,
                max_distance=LIDAR_MAX_DIST,
                density=LIDAR_DENSITY,
                is_360_mode=True,
                is_using_shared_memory=True,
                is_visualised=False,
            )
        self._lidar_tracker = LidarClusterTracker()

    def _ego_extents(self) -> tuple[float, float]:
        """Approximate ego footprint from the current BeamNG bbox."""
        try:
            bbox = self.conn.vehicle.get_bbox()
            fl = np.asarray(bbox["front_bottom_left"], dtype=float)[:2]
            fr = np.asarray(bbox["front_bottom_right"], dtype=float)[:2]
            rl = np.asarray(bbox["rear_bottom_left"], dtype=float)[:2]
            rr = np.asarray(bbox["rear_bottom_right"], dtype=float)[:2]
            length = float(np.linalg.norm((fl + fr) / 2.0 - (rl + rr) / 2.0))
            width = float(np.linalg.norm((fr + rr) / 2.0 - (fl + rl) / 2.0))
            return max(0.5, length / 2.0), max(0.5, width / 2.0)
        except Exception:
            return 2.4, 1.0

    def _ego_heading(self) -> float | None:
        """Heading of the last polled vehicle state, if available."""
        try:
            st = self.conn.vehicle.state
            d = st.get("dir") if st else None
            if d is None:
                return None
            return float(np.arctan2(float(d[1]), float(d[0])))
        except Exception:
            return None

    def scan(self, pos, ego_vid=None, radius: float = 55.0) -> RangeSample:
        with self.conn.io_lock:
            obstacles: list[Obstacle] = []
            pts: list[tuple[float, float]] = []
            ox, oy = float(pos[0]), float(pos[1])
            cloud = np.empty((0, 3), dtype=float)
            heading: float | None = None
            try:
                for _ in range(LIDAR_POLL_RETRIES):
                    data = self.lidar.poll()
                    cloud = np.asarray(data.get("pointCloud"), dtype=float)
                    if cloud.ndim == 2 and len(cloud) > 0:
                        break
                    time.sleep(0.15)
                if cloud.ndim != 2 or cloud.shape[1] < 3:
                    cloud = np.empty((0, 3), dtype=float)
                cloud = cloud[np.isfinite(cloud).all(axis=1)]
                oz = float(pos[2])
                heading = self._ego_heading()
                # Lane-corridor hits (unchanged semantics): world z window
                # around the ego, self footprint removed, voxel-capped.
                dist = np.hypot(cloud[:, 0] - ox, cloud[:, 1] - oy)
                keep = ((dist >= 2.5) & (dist <= radius)
                        & (np.abs(cloud[:, 2] - oz) <= 4.0))
                if heading is not None:
                    uf = np.array([np.cos(heading), np.sin(heading)])
                    ur = np.array([-uf[1], uf[0]])
                    local = cloud[:, :2] - np.asarray(pos[:2], dtype=float)
                    on_car = ((np.abs(local @ uf)
                               <= self._ego_half_len + LIDAR_SELF_MARGIN)
                              & (np.abs(local @ ur)
                                 <= self._ego_half_w + LIDAR_SELF_MARGIN))
                    keep = keep & ~on_car
                kept = downsample_cloud(cloud[keep])
                pts = [(float(x), float(y)) for x, y, _ in kept]
                last_error["raycast"] = None
            except Exception as exc:
                last_error["raycast"] = str(exc)
                print(f"[tech] lidar scan failed: {exc}")
            try:
                # Scenario / vehicle registry / raycast fan (shared with
                # Steam).  The dense 360 LiDAR is now a first-class
                # obstacle channel on top of this (lidar_obstacles below)
                # instead of a fallback, so vehicles / pedestrians /
                # unexpected objects stay covered even without the camera.
                obstacles, ray_hits = scan_obstacles_all(
                    self.conn.bng, ego_vid, pos, radius=radius,
                    return_hits=True)
                if ray_hits:
                    pts = ray_hits
                last_error["raycast"] = None
            except Exception as exc:
                last_error["raycast"] = str(exc)
                print(f"[tech] raycast scan failed: {exc}")
            # Tech-native LiDAR obstacles: cluster the dense 360 cloud
            # (with local ground removal) and fuse with the Lua sources.
            # Lua boxes come first so a vehicle's registry velocity
            # survives the merge; lidar-only clusters carry a tracker
            # velocity estimate instead.
            if len(cloud) >= 4:
                try:
                    boxes = lidar_obstacles(
                        cloud, pos,
                        radius=min(radius, LIDAR_OBSTACLE_RADIUS),
                        self_rect=(self._ego_half_len + LIDAR_SELF_MARGIN,
                                   self._ego_half_w + LIDAR_SELF_MARGIN,
                                   float(heading if heading is not None
                                         else 0.0)))
                    self._lidar_tracker.update(boxes, time.time())
                    obstacles = merge_obstacles(obstacles + boxes)
                except Exception as exc:
                    last_error["raycast"] = str(exc)
                    print(f"[tech] lidar obstacle fusion failed: {exc}")
            elif not obstacles and pts:
                # Last-resort fallback when both the Lua scan and the fused
                # lidar channel are unavailable: cluster the corridor hits.
                for sector in _split_raycast_sectors(pts, (ox, oy)):
                    obstacles.extend(_cluster_points(sector, split_walls=True))
            return RangeSample(obstacles=obstacles, ray_hits=pts)

    def close(self) -> None:
        with self.conn.io_lock:
            self.lidar.remove()
