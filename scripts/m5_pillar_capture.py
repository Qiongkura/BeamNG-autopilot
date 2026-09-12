"""P1b 侧向相机探针：**行驶中**采集 pillar_left/right 并量化近场盲区增益。

背景（docs §19 / §27）：前视相机反投影最近只到 3.59 m，车旁 (<3.59 m) 的
车道线结构性不可见；相机环里的 B 柱侧向相机（pan ±118°, FOV 90°）从未被
消费。§27 还记了一个测量完整性陷阱：**teleport 静态采集会静默缺路面漆画
decal**（RGB 本体缺线），所以本探针只在**行驶中**采集——AI span 沿当前
道路开车（``m5_lane_truth_probe --drive`` 的已验证模式），连续采三视角。

* **capture**（需 BeamNG.tech 在线）：AI 沿路行驶，每帧采
  front_main / pillar_left / pillar_right 的 RGB+标注（536×403）+ 位姿，
  存 ``logs/m5_pillar/pillar_<ts>.npz``；
* **--analyze**（离线）：量化 (a) <3.59 m 近场漆画覆盖增益，(b) 本车道
  右边界"最近可见纵向距离" front-only vs front+pillar（镜像窗 3.0 m 要求
  是否变得可满足），(c) 语义模型在侧视角上的迁移。

Usage::

    .venv\Scripts\python.exe scripts\m5_pillar_capture.py --frames 90
    .venv\Scripts\python.exe scripts\m5_pillar_capture.py --analyze ^
        logs\m5_pillar\pillar_<ts>.npz
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.vision.detection import back_project
from beamng_autopilot.vision.ring import (CAMERA_RING, FRONT_MAIN,
                                          PILLAR_LEFT, PILLAR_RIGHT)

RES_W, RES_H = 536, 403
# The front camera's measured near bound: image bottom row maps here.
FRONT_BLIND_M = 3.59
# Near-field bin the mirror window needs points in.
MIRROR_NEAR_M = 3.0
LINE_CLS = 2


def _mounts():
    by_role = {m.role: m for m in CAMERA_RING}
    return [("front", by_role[FRONT_MAIN]),
            ("pl", by_role[PILLAR_LEFT]),
            ("pr", by_role[PILLAR_RIGHT])]


def _mounts():
    by_role = {m.role: m for m in CAMERA_RING}
    return [("front", by_role[FRONT_MAIN]),
            ("pl", by_role[PILLAR_LEFT]),
            ("pr", by_role[PILLAR_RIGHT])]


def _stations_from_episode(ep: Path, n: int) -> list[tuple[float, float, float]]:
    d = np.load(ep, allow_pickle=True)
    x, y, h = d["x"], d["y"], d["heading"]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [(float(x[i]), float(y[i]), float(h[i])) for i in idx]


def capture(out: Path, frames: int, speed: float) -> int:
    from beamngpy.sensors import Camera
    from beamng_autopilot_tech.annotations import annotation_palette, to_label
    from beamng_autopilot_tech.providers import _mount_to_tp

    # Vehicle lock: two driver sessions share one game/vehicle, and both
    # may execute the same next action at once - a second capture silently
    # invalidates both runs (double-instance incidents 2026-09-12).
    lock = config.LOGS_DIR / "m5_pillar" / ".vehicle_lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8", errors="replace").strip()
        print(f"[pillar] another capture holds {lock} (pid {holder}); "
              f"exiting - clean up the stale lock only after checking "
              f"the process table")
        return 1

    conn = BeamNGConnector("italy", "etk800",
                           port=config.runtime_port("tech"),
                           home=config.runtime_home("tech"))
    conn.open(launch=False)
    try:
        conn.attach_vehicle(already_open=True)
    except Exception:
        conn.load_scenario()
        conn.step(60)
        conn.attach_vehicle(already_open=True)

    with conn.io_lock:
        annotation_session_palette = annotation_palette(
            conn.bng.get_annotations())

    cams = {}
    with conn.io_lock:
        for role, mount in _mounts():
            pos, direction, up = _mount_to_tp(mount)
            cams[role] = Camera(
                f"pillar_probe_{role}", conn.bng, conn.vehicle,
                requested_update_time=0.05, pos=pos, dir=direction, up=up,
                resolution=(RES_W, RES_H),
                field_of_view_y=mount.fov_deg,
                near_far_planes=(0.05, 150.0),
                is_using_shared_memory=True, is_render_colours=True,
                is_render_annotations=True, is_render_instance=False,
                is_render_depth=False, is_visualised=False)

    # AI span drive: the verified way to be on painted road (§27 - a
    # teleported static world silently lacks the lane-paint decals).
    with conn.io_lock:
        try:
            conn.vehicle.ai.set_mode("span")
            conn.vehicle.ai.set_speed(speed, mode="limit")
            print(f"[pillar] AI driving at {speed} m/s (span)")
        except Exception as exc:
            print(f"[pillar] AI drive failed ({exc}); aborting - a static "
                  f"capture would miss the lane-paint decals (docs §27)")
            conn.close()
            os.remove(lock)
            return 1

    def _snap():
        snap = {}
        with conn.io_lock:
            for role, cam in cams.items():
                for _ in range(4):          # black-frame retries
                    data = cam.poll()
                    rgb = data.get("colour") if data else None
                    ann = data.get("annotation") if data else None
                    if (rgb is not None and ann is not None
                            and np.asarray(rgb).mean() > 8.0):
                        snap[role] = (np.ascontiguousarray(
                            np.asarray(rgb, np.uint8)),
                            to_label(np.ascontiguousarray(
                                np.asarray(ann, np.uint8)),
                                road_colors=annotation_session_palette["road"],
                                line_colors=annotation_session_palette["line"]))
                        break
                    time.sleep(0.06)
        return snap

    snap = _snap()                            # warm-up / black-frame pass
    frames_store = {k: [] for k in ("front", "pl", "pr")}
    poses: list[tuple[float, float, float]] = []
    n_ok = 0
    for i in range(frames):
        t0 = time.time()
        with conn.io_lock:
            conn.step(10)
            st = conn.get_state()
        snap = _snap()
        if any(v is None for v in snap.values()):
            print(f"  frame {i}: missing view "
                  f"{[k for k, v in snap.items() if v is None]}, skipped")
            continue
        for role in snap:
            frames_store[role].append(snap[role])
        poses.append((float(st.pos[0]), float(st.pos[1]),
                      float(st.heading)))
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  frame {i}: ok ({n_ok}/{frames}) "
                  f"v={float(st.speed):.1f} m/s ({time.time() - t0:.2f}s)")

    with conn.io_lock:
        try:
            conn.vehicle.ai.set_mode("disabled")
        except Exception:
            pass
        for cam in cams.values():
            try:
                cam.remove()
            except Exception:
                pass
    conn.close()
    try:
        os.remove(lock)
    except OSError:
        pass

    if not n_ok:
        print("no usable frames")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{f"{k}_rgb": np.stack([f[0] for f in frames_store[k]])
           for k in frames_store},
        **{f"{k}_label": np.stack([f[1] for f in frames_store[k]])
           for k in frames_store},
        x=np.array([p[0] for p in poses]),
        y=np.array([p[1] for p in poses]),
        heading=np.array([p[2] for p in poses]),
        meta=np.array(json_meta({"schema": "pillar_probe_drive",
                                 "frames": n_ok,
                                 "res": [RES_W, RES_H]})),
    )
    print(f"saved {n_ok} frames -> {out}")
    return 0


def json_meta(d: dict) -> bytes:
    import json
    return json.dumps(d).encode("utf-8")


def _world_line_pts(label, mount, pos, hd, max_m: float):
    """Back-project GT line pixels of one view to ego-frame (lon, lat)."""
    cam = mount.camera_model(RES_W, RES_H)
    rr, cc = np.where(label == LINE_CLS)
    if len(rr) == 0:
        return np.empty((0, 2))
    if len(rr) > 2500:
        sel = np.random.default_rng(3).choice(len(rr), 2500, replace=False)
        rr, cc = rr[sel], cc[sel]
    fwd = np.array([np.cos(hd), np.sin(hd)])
    left = np.array([-fwd[1], fwd[0]])
    pts = []
    for r_, c_ in zip(rr, cc):
        wp = back_project(float(c_), float(r_), cam, pos, hd, 0.0)
        if wp is None:
            continue
        rel = np.asarray(wp) - pos[:2]
        lon = float(rel @ fwd)
        lat = float(rel @ left)
        if 0.2 <= lon <= max_m:
            pts.append((lon, lat))
    return np.asarray(pts)


def analyze(path: Path) -> int:
    d = np.load(path, allow_pickle=True)
    by_role = {m.role: m for m in CAMERA_RING}
    mounts = {"front": by_role[FRONT_MAIN],
              "pl": by_role[PILLAR_LEFT], "pr": by_role[PILLAR_RIGHT]}
    n = len(d["x"])
    gain = {"right": 0, "left": 0}
    vis = {k: 0 for k in ("front", "pl", "pr")}
    near_start = {"front_only": [], "front_plus": []}
    seg_hit = {"pl": [0, 0], "pr": [0, 0]}
    for i in range(n):
        pos = np.array([float(d["x"][i]), float(d["y"][i]), 0.0])
        hd = float(d["heading"][i])
        pts = {k: _world_line_pts(d[f"{k}_label"][i], mounts[k], pos, hd,
                                  40.0) for k in mounts}
        # (a) near-field paint (<3.59 m) per view, per ego-lane side
        def _near(view, side):
            p = pts[view]
            if not len(p):
                return False
            lat = p[:, 1]
            lon = p[:, 0]
            m = (lon < FRONT_BLIND_M) & (
                (lat < -1.0) if side == "right" else (lat > 1.0))
            return bool(m.sum() >= 5)

        for side, view in (("right", "pr"), ("left", "pl")):
            f, s = _near("front", side), _near(view, side)
            if s:
                vis[view] += 1
            if s and not f:
                gain[side] += 1
        # (b) nearest visible longitudinal of the ego RIGHT boundary
        def _nearest(view):
            p = pts[view]
            m = (p[:, 1] < -1.0) if len(p) else np.empty(0, bool)
            return float(p[m, 0].min()) if len(p) and m.any() else None

        nf = _nearest("front")
        pp = _nearest("pr")
        if nf is not None:
            near_start["front_only"].append(nf)
        if nf is not None or pp is not None:
            near_start["front_plus"].append(
                min(v for v in (nf, pp) if v is not None))
        # (c) seg-model transfer on pillar views (GT line px vs mask px)
        for view in ("pl", "pr"):
            gt_px = int((d[f"{view}_label"][i] == LINE_CLS).sum())
            if gt_px >= 200:
                seg_hit[view][0] += 1
                try:
                    from beamng_autopilot.vision.heads.semantic import \
                        SemanticHead
                    from beamng_autopilot.vision.hydra import FrameContext
                    from beamng_autopilot.vision.segmentation import \
                        Segmenter
                    if not hasattr(analyze, "_sem"):
                        analyze._sem = SemanticHead(segmenter=Segmenter())
                    out = analyze._sem.run(FrameContext(
                        frame_rgb=d[f"{view}_rgb"][i],
                        cam=mounts[view].camera_model(RES_W, RES_H),
                        pos=pos, heading=hd, ground_z=0.0,
                        role=view))
                    mask = out.masks.get("line")
                    if mask is not None and np.asarray(mask).sum() >= 200:
                        seg_hit[view][1] += 1
                except Exception as exc:
                    print(f"  seg transfer failed on {view}: {exc}")
                    break

    print(f"\n=== pillar probe analysis ({n} stations, {path.name}) ===")
    print(f"  near-field (<{FRONT_BLIND_M} m) ego-lane paint visible:")
    print(f"    front sees it : right n/a by design / left {vis['pl']}/{n}")
    print(f"    pillar sees it: right {vis['pr']}/{n}  left {vis['pl']}/{n}")
    print(f"  BLIND-ZONE GAIN (pillar sees, front cannot): "
          f"right {gain['right']}/{n}  left {gain['left']}/{n}")
    for k, v in near_start.items():
        if v:
            a = np.asarray(v)
            print(f"  right-boundary nearest visible lon [{k}]: "
                  f"p10/p50={np.percentile(a, 10):.2f}/{np.median(a):.2f} m "
                  f"(n={len(a)})")
    print(f"  mirror window (<= {MIRROR_NEAR_M} m satisfiable): "
          f"front_only "
          f"{sum(1 for v in near_start['front_only'] if v <= MIRROR_NEAR_M)}"
          f"/{len(near_start['front_only'])}  vs front+pillar "
          f"{sum(1 for v in near_start['front_plus'] if v <= MIRROR_NEAR_M)}"
          f"/{len(near_start['front_plus'])}")
    for view in ("pl", "pr"):
        tot, hit = seg_hit[view]
        if tot:
            print(f"  seg model on {view}: line mask present "
                  f"{hit}/{tot} views with GT paint")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="pillar camera probe (drive)")
    ap.add_argument("--frames", type=int, default=90,
                    help="frames captured while the AI drives")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="AI speed limit in m/s")
    ap.add_argument("--analyze", default=None,
                    help="analyze a captured npz instead of capturing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.analyze:
        return analyze(Path(args.analyze))

    out = (Path(args.out) if args.out else
           config.LOGS_DIR / "m5_pillar"
           / f"pillar_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    return capture(out, args.frames, args.speed)


if __name__ == "__main__":
    sys.exit(main())
