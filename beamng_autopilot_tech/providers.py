"""BeamNG.tech sensor providers built on beamngpy Camera/LiDAR sensors."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from beamng_autopilot.perception import (
    LIDAR_GROUND_CLEARANCE_M,
    LIDAR_MAX_HEIGHT_M,
    LidarClusterTracker,
    Obstacle,
    _cluster_points,
    _local_ground_z,
    _split_raycast_sectors,
    downsample_cloud,
    last_error,
    lidar_obstacles,
    merge_obstacles,
    scan_obstacles_all,
)
from beamng_autopilot.runtime import CameraProvider, RangeProvider, RangeSample
from beamng_autopilot.vision.ring import (
    CAMERA_RING,
    FRONT_MAIN,
    CameraMount,
    camera_ring_models,
)
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
LIDAR_POLL_RETRIES = 3  # (kept for reference; scan() now polls once per
                        # worker cycle and retries on the next cycle)
LIDAR_MAX_POINTS = 6000
LIDAR_SELF_MARGIN = 0.3
LIDAR_OBSTACLE_RADIUS = 45.0

BLACK_FRAME_MAX_MEAN = 1.0
BLACK_FRAME_RETRIES = 2
BLACK_FRAME_RETRY_DELAY_S = 0.1


# ── 图形质量前置检查 ──────────────────────────────────────────────
# BeamNG.tech 的 Camera/LiDAR 传感器依赖 GPU prepass buffer：当画质
# 处于 'Lowest' 时引擎不生成该 buffer，传感器一旦创建，GPU Request
# Manager 会以每秒数千条的频率刷 "Failed to get prepass buffer"，渲染
# 线程被拖死，游戏窗口直接卡成"未响应"（实测 4 份 beamng.log 全部复现）。
# 因此在创建任何 Tech 传感器之前先读 settings.json，命中 Lowest 就打印
# 明确的中文提示并中止启动，而不是让游戏默默卡死。
_GRAPHICS_QUALITY_KEYS = (
    "GraphicLightingQuality",
    "GraphicShaderQuality",
    "GraphicMeshQuality",
    "GraphicTextureQuality",
    "GraphicShadowsQuality",
)


def check_graphics_quality(user_dir) -> None:
    """Raise RuntimeError when the Tech user's graphics preset is 'Lowest'.

    读取 ``<user_dir>/current/settings/settings.json``，只要任一关键画质项
    为 ``Lowest`` 就中止（该画质下 Tech 传感器会刷爆 GPU prepass buffer，
    导致游戏窗口未响应）。设置文件缺失/不可读时只打印警告并放行。
    """
    path = Path(user_dir) / "current" / "settings" / "settings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        print("[tech] WARNING: 无法读取图形设置 "
              f"{path}（继续创建传感器，若卡死请检查画质是否为 Lowest）")
        return
    lowest = [k for k in _GRAPHICS_QUALITY_KEYS
              if str(data.get(k, "")).strip().lower() == "lowest"]
    if not lowest:
        return
    print("[tech] ✗ 检测到图形质量为 'Lowest'，无法创建 Camera/LiDAR 传感器：")
    for k in lowest:
        print(f"[tech]   - {k} = Lowest")
    print("[tech] 'Lowest' 画质下引擎不生成 GPU prepass buffer，Tech 传感器")
    print("[tech] 一创建就会把游戏卡成窗口未响应（实测稳定复现）。")
    print("[tech] 请在游戏 选项 → 图形 里把 光照质量(Lighting Quality) 至少")
    print("[tech] 调到 Low（建议 Normal 或更高），然后重启游戏再试。")
    raise RuntimeError(
        "graphics quality is 'Lowest' which breaks Tech camera/LiDAR "
        "sensors; raise Lighting Quality to Low+ in the game settings "
        "and restart BeamNG.tech first")


def _frame_is_black(frame, max_mean: float = BLACK_FRAME_MAX_MEAN) -> bool:
    """True when a camera frame is an all-black/stale shared-memory read."""
    if frame is None or frame.size == 0:
        return True
    return float(np.mean(np.asarray(frame, dtype=np.float32))) < max_mean


class TechCameraProvider(CameraProvider):
    """Front camera attached to the ego vehicle on BeamNG.tech."""

    def __init__(self, conn, width: int = 1076, height: int = 806,
                 annotations: bool = False) -> None:
        check_graphics_quality(conn.user_dir)
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
        check_graphics_quality(conn.user_dir)
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
        # Only the sensor reads (LiDAR poll, Lua fan) need the connector
        # lock.  The CPU-side clustering / fusion below can take 200+ ms
        # on a dense 360 cloud and must NOT hold io_lock: the control loop
        # steps/commands through that same lock, so holding it across the
        # clustering would cap the control cadence at ~1 Hz and the car
        # would weave on bends (run 42-45).  Read under the lock, compute
        # outside it.
        obstacles: list[Obstacle] = []
        pts: list[tuple[float, float]] = []
        ox, oy = float(pos[0]), float(pos[1])
        cloud = np.empty((0, 3), dtype=float)
        heading: float | None = None
        with self.conn.io_lock:
            try:
                data = self.lidar.poll()
                cloud = np.asarray(data.get("pointCloud"), dtype=float)
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
                # Near-field 360-LiDAR points (1-2.5 m) are kept ONLY for
                # the raw-sensor emergency-stop / approach-speed layer
                # (forward_clearance_m).  The cluster/corridor pool still
                # starts at 2.5 m so the own car's tail / wheel guards do
                # not become road obstacles, but a wall 1-2 m off the
                # bonnet must remain visible or the car floors the throttle
                # into it while standing still (throttle=94.7% at v=0).
                near_keep = ((dist >= 1.0) & (dist < 2.5)
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
                    near_keep = near_keep & ~on_car
                kept = downsample_cloud(cloud[keep])
                pts = [(float(x), float(y)) for x, y, _ in kept]
                # Drop the local ground plane from the near field the same
                # way lidar_obstacles does: on flat ground the 1-2.5 m band
                # is full of ground returns that project onto the emergency
                # corridor and pin the car forever (EMERGENCY STOP raw
                # clear=0.7m everywhere, v never leaves 0).  A wall / tree /
                # vehicle right off the bonnet still has points above the
                # ground clearance and stays visible to the safety layer.
                near_cloud = downsample_cloud(cloud[near_keep])
                if len(near_cloud) >= 4:
                    gnd = _local_ground_z(near_cloud, ox, oy)
                    above = ((near_cloud[:, 2] - gnd
                              >= LIDAR_GROUND_CLEARANCE_M)
                             & (near_cloud[:, 2] - gnd
                                <= LIDAR_MAX_HEIGHT_M))
                    if np.any(above):
                        near_cloud = near_cloud[above]
                    else:
                        near_cloud = near_cloud[:0]
                near_pts = [(float(x), float(y))
                            for x, y, _ in near_cloud]
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
                # Merge the 360-LiDAR near field on top of the Lua fan so
                # a wall right in front of the bonnet is never invisible,
                # whichever sensor channel happened to fire this frame.
                pts = pts + near_pts
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


def _mount_to_tp(mount: CameraMount) -> tuple[tuple, tuple, tuple]:
    """Map a ring mount into BeamNG vehicle-local Sensor inputs.

    BeamNG vehicle-local axes use forward = -Y; the ring defines forward
    on the +Y axis (matching ``CameraModel``), so the Y components of the
    position and of the direction/up vectors are negated here - the same
    convention the existing front ``CAMERA_POS/DIR/UP`` use.
    """
    pos = (float(mount.offset[0]), -float(mount.offset[1]),
           float(mount.offset[2]))
    d = mount.fwd_local
    direction = (float(d[0]), -float(d[1]), float(d[2]))
    u = mount.up_local
    up = (float(u[0]), -float(u[1]), float(u[2]))
    return pos, direction, up


class TechCameraRingProvider(CameraProvider):
    """The full eight-camera FSD-style ring on BeamNG.tech.

    One beamngpy ``Camera`` per ring mount, all polled from the same
    snapshot so the multi-view frames are close in time (the shared-memory
    sensors are queried back to back under the connector lock).  ``grab()``
    returns the front-main frame for the legacy front-only consumers;
    ``grab_ring()`` returns ``{role: (rgb, CameraModel)}`` for the
    perception pipeline.
    """

    def __init__(self, conn, width: int = 1076, height: int = 806,
                 roles: tuple[str, ...] | None = None) -> None:
        check_graphics_quality(conn.user_dir)
        from beamngpy.sensors import Camera

        self.conn = conn
        self.width = int(width)
        self.height = int(height)
        if roles is None:
            self._roles = tuple(m.role for m in CAMERA_RING)
        else:
            wanted = set(roles)
            self._roles = tuple(m.role for m in CAMERA_RING
                                if m.role in wanted)
        if FRONT_MAIN not in self._roles:
            self._roles = (FRONT_MAIN,) + self._roles
        self._mounts: dict[str, CameraMount] = {
            m.role: m for m in CAMERA_RING if m.role in self._roles}
        # Ring warm-up is one-shot: the GPU prepass buffer is re-authored
        # right after sensor creation, so the first polls can be black.
        # Re-doing the throwaway pass on EVERY grab_ring would double the
        # poll cost (16 camera polls per FSD tick); individual black
        # frames are still retried inside _poll afterwards.
        self._warmed = False
        with conn.io_lock:
            self.cameras: dict[str, object] = {}
            for mount in CAMERA_RING:
                if mount.role not in self._roles:
                    continue
                name = f"autopilot_ring_{mount.role}_{_PID}"
                pos, direction, up = _mount_to_tp(mount)
                self.cameras[mount.role] = Camera(
                    name, conn.bng, conn.vehicle,
                    requested_update_time=0.05,
                    pos=pos, dir=direction, up=up,
                    resolution=(int(width), int(height)),
                    field_of_view_y=mount.fov_deg,
                    near_far_planes=(0.05, 150.0),
                    is_using_shared_memory=True,
                    is_render_colours=True,
                    is_visualised=False,
                )

    def _poll(self, cam: object) -> np.ndarray:
        # Ring cameras share the GPU prepass buffer with the other Tech
        # sensors; a single frame can come back black right after the
        # buffer is re-authored (fresh sensor creation, scenario load).
        # Retry a few times with a small delay (the next poll usually
        # returns a real frame) and only give up after several
        # consecutive failures.  The shared-memory camera keeps the
        # latest valid image, so a genuinely black mount would fail every
        # attempt and surface as an error instead of silently feeding a
        # black frame to the perception pipeline.
        last_error: RuntimeError | None = None
        for attempt in range(BLACK_FRAME_RETRIES + 4):
            with self.conn.io_lock:
                data = cam.poll()
            image = data.get("colour")
            if image is None:
                last_error = RuntimeError(f"{cam} returned no colour frame")
            else:
                frame = np.ascontiguousarray(
                    np.asarray(image), dtype=np.uint8).copy()
                if not _frame_is_black(frame):
                    return frame
                last_error = RuntimeError(f"{cam} returned a black frame")
            time.sleep(0.1)
        raise last_error

    def grab(self) -> np.ndarray:
        return self._poll(self.cameras[FRONT_MAIN])

    def grab_ring(self) -> dict[str, tuple[np.ndarray, CameraModel]]:
        """Poll the whole ring; ``{role: (rgb, CameraModel)}``."""
        models = camera_ring_models(self.width, self.height)
        # Warm-up only ONCE per provider (first grab after creation): the
        # GPU prepass buffer is re-authored on a fresh ring and the first
        # polls frequently come back black.  A throwaway pass on every
        # tick would double the ring poll cost; per-frame black frames are
        # still retried inside _poll, so a later black streak recovers.
        if not self._warmed:
            for cam in self.cameras.values():
                try:
                    self._poll(cam)
                except RuntimeError:
                    pass
            self._warmed = True
        out: dict[str, tuple[np.ndarray, CameraModel]] = {}
        for role, cam in self.cameras.items():
            out[role] = (self._poll(cam), models[role])
        return out

    def camera_model(self, pos, heading, width, height,
                     fallback: CameraModel | None = None,
                     rotation=None) -> CameraModel:
        # Front-main model for the legacy single-camera consumers.
        return camera_ring_models(int(width), int(height))[FRONT_MAIN]

    def close(self) -> None:
        with self.conn.io_lock:
            for cam in self.cameras.values():
                cam.remove()
        self.cameras = {}
