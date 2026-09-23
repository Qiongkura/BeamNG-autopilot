"""Multi-view segmentation labels: one drive round, all eight cameras.

Why this exists: the deployed UNet is trained on the front camera only, so
"add a camera" looked like "label a new dataset by hand".  It does not have
to.  The simulator renders a per-pixel annotation for ANY camera, and the
road class it provides is dense and automatic - measured 2026-09-21, road
IoU of the deployed model against it is 0.89-0.96 on hand-checked frames.
So a ring-wide label pass is a collection problem, not a labeling problem.

What the annotation does NOT provide is the painted LINE class: on this map
the game colours the lane paint as ASPHALT (measured: 3 px of SOLID_LINE in
a frame with thousands of white-paint pixels).  Line supervision therefore
still comes from the manual annotator (``m5_annotate_manual.py``) - and
only once, because a line looks the same from every camera.

Layout written (one run directory per mount, which is what
``m5_train_seg.py --split per-run`` expects, so each view becomes its own
held-out group):

    logs/m5_seg/ring_<stamp>/<role>/frame_00000.npz   colour + label
    logs/m5_seg/ring_<stamp>/meta.json                 palette + provenance

Labels follow the training contract: 0 = background, 1 = road, 2 = line
(empty here - see above).  ``prepare_annotation_sample`` drops pixels whose
annotation colour the live palette does not map; those become background,
which is the same convention the single-camera collector uses.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_collect_seg_ring.py --runtime tech \\
        --attach --frames 40 --roles front_main front_fisheye pillar_left pillar_right
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beamng_autopilot import config
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot.runtime import build_camera_ring_provider
from beamng_autopilot.vision.ring import camera_ring_models

# Collection resolution: the segmentation training contract is 536x403.
W, H = 536, 403


def session_map_name(bng, fallback: str) -> tuple[str, str]:
    """``(map_name, source)`` read from the RUNNING session, not the argument.

    Measured defect: this collector wrote the literal ``"italy"`` while the
    session had ``east_coast_usa`` / ``gridmap_v2`` loaded, so two ring
    collections carry the wrong map in their meta and the training entry's
    map identity is wrong for them.  ``--map`` is only what the connector
    was TOLD to load, and on ``--attach`` that says nothing about the
    session.  The source is returned too, so a reader can see whether the
    identity is real or a fallback.
    """
    try:
        sc = getattr(bng, "scenario", None)
        cur = sc.get_current() if sc is not None else None
        level = str(getattr(cur, "level", "") or "")
        if level:
            return level, "session.get_current().level"
    except Exception:                        # noqa: BLE001
        pass
    return str(fallback), "argument-fallback"


def follow_road_step(rn, pos, heading: float, advance_m: float = 2.0,
                     snap_m: float = 6.0):
    """One ROADNET-guided placement step: ``(target_xy, heading, note)``.

    Picks the neighbour of the nearest road node whose direction best
    matches the current heading, and returns a point ``advance_m`` along
    that edge.  ``(None, heading, reason)`` when the road ends or the
    graph is not ready, so the caller stops instead of guessing.

    When the car is OFF the network (measured on east_coast_usa: the spawn
    sat at ``road_px=22``, i.e. not on a road, and the very first step
    found "no forward neighbour"), the step SNAPS to the nearest node and
    aligns the heading with an incident edge - otherwise a walk can never
    start from a spawn point that is not on the network.
    """
    import numpy as np

    if rn is None or not getattr(rn, "ready", False) or rn.nodes is None:
        return None, heading, "roadnet not ready"
    p = np.asarray(pos, dtype=float)[:2]
    i = int(rn._nearest(p))
    here = np.asarray(rn.nodes[i], dtype=float)[:2]
    best = None
    for j, d in rn.adj.get(i, []):
        tp = np.asarray(rn.nodes[j], dtype=float)[:2]
        v = tp - here
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        u = v / n
        align = float(u[0] * np.cos(heading) + u[1] * np.sin(heading))
        if best is None or align > best[0]:
            best = (align, tp)
    off_network = float(np.linalg.norm(here - p)) > float(snap_m)
    if best is None:
        if off_network:
            return None, heading, "off the roadnet and the nearest node has no edge"
        return None, heading, "no neighbour on the roadnet"
    _, tp = best
    if off_network:
        u = tp - here
        n = float(np.linalg.norm(u))
        if n < 1e-6:
            return None, heading, "degenerate edge at the snap node"
        u = u / n
        return here, float(np.arctan2(u[1], u[0])), "snapped onto the roadnet"
    align, tp = best
    if align <= 0.3:
        return None, heading, "no forward neighbour on the roadnet"
    u = (tp - p)
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None, heading, "degenerate edge"
    u = u / n
    tgt = p + advance_m * u
    return tgt, float(np.arctan2(u[1], u[0])), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default="italy",
                    help="map for the connector; the ATTACH path uses whatever "
                         "the running session already has loaded, so this only "
                         "matters when the collector must load the scenario")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--roles", nargs="*", default=None,
                    help="ring roles to collect (default: the whole ring)")
    ap.add_argument("--out", default=None,
                    help="output root (default logs/m5_seg/ring_<stamp>)")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--save-annotation", action="store_true",
                    help="also store the engine palette frame (annotation_raw) "
                         "for the requested roles: material-level diagnosis "
                         "(building/sky/grass vs road) needs the palette, not just "
                         "the 3-class label")
    ap.add_argument("--follow-road", action="store_true",
                    help="place each step along the ROADNET instead of "
                         "dead-reckoning on the heading: keeps a curving "
                         "street's captures on the pavement, and avoids the "
                         "map-wide teleports that killed the session on "
                         "east_coast_usa")
    ap.add_argument("--step-m", type=float, default=0.0,
                    help="place the car this far along its heading before each "
                         "grab: a STATIONARY car yields 20 identical frames "
                         "(measured: first and last pose identical), which is a "
                         "duplicate sample, not a sequence")
    ap.add_argument("--step", type=int, default=10,
                    help="simulation steps between grabs (moves the car "
                         "slowly if it is already rolling)")
    args = ap.parse_args()

    from beamng_autopilot_tech.annotations import annotation_palette
    from beamng_autopilot.labeling.tech_annotation import (
        prepare_annotation_sample)

    conn = BeamNGConnector(
        str(args.map), "etk800",
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out) if args.out else (config.LOGS_DIR / "m5_seg"
                                            / f"ring_{stamp}")
    conn.open(launch=not args.attach)
    try:
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        if args.teleport:
            x, y, yaw = args.teleport
            conn.safe_teleport(float(x), float(y), heading_deg=float(yaw))
        get_ann = getattr(conn.bng, "get_annotations", None)
        tech_ann = get_ann() if callable(get_ann) else None
        palette = annotation_palette(tech_ann)
        map_name, map_name_source = session_map_name(conn.bng, args.map)
        print(f"[ring-collect] session map: {map_name} "
              f"(source: {map_name_source})", flush=True)
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, W, H, roles=(tuple(args.roles)
                                             if args.roles else None),
            annotations=True)
        if ring is None:
            print("[ring-collect] the ring needs BeamNG.tech (no ring on "
                  "this runtime)", flush=True)
            return 2
        print(f"[ring-collect] runtime={mode} roles="
              f"{','.join(sorted(ring.cameras))}", flush=True)

        rn = None
        if args.follow_road:
            from beamng_autopilot.roadnet import RoadNetwork
            rn = RoadNetwork()
            _t0 = time.time()
            while not rn.ready and time.time() - _t0 < 120:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                time.sleep(1)
            print(f"[ring-collect] follow-road: roadnet ready="
                  f"{bool(rn.ready)}", flush=True)

        counts: dict[str, int] = {r: 0 for r in ring.cameras}
        skipped: dict[str, int] = {r: 0 for r in ring.cameras}
        audits: dict[str, dict] = {}
        # One grab = one EXPOSURE seen by every view.  The counter is what
        # lets the split keep all views of one instant on the same side
        # (T10: 跨视角同一曝光必须一起归组); without it the training entry
        # cannot even check that leak.
        frame_records: list[dict] = []
        for i in range(max(1, int(args.frames))):
            if i > 0 and (float(args.step_m) > 0.0 or args.follow_road):
                st_prev = conn.get_state()
                hdg = float(st_prev.heading)
                p_prev = np.asarray(st_prev.pos, dtype=float)
                adv = float(args.step_m) if float(args.step_m) > 0.0 else 2.0
                if args.follow_road:
                    # ROADNET-GUIDED placement: step along the road through
                    # the nearest node instead of dead-reckoning along the
                    # heading.  Measured need: on curving streets (and on
                    # other maps) a straight walk leaves the pavement within
                    # a few metres, and map-wide teleport sampling killed
                    # the session on east_coast_usa - staying ON the road
                    # network between successive placements is what avoids
                    # both.
                    tgt, hdg_new, why = follow_road_step(
                        rn, p_prev, hdg, adv)
                    if tgt is None:
                        print(f"[ring-collect] step {i}: follow-road "
                              f"stopped ({why})")
                        break
                    nx, ny = float(tgt[0]), float(tgt[1])
                    hdg = hdg_new
                else:
                    nx = float(p_prev[0]) + adv * float(np.cos(hdg))
                    ny = float(p_prev[1]) + adv * float(np.sin(hdg))
                if not conn.safe_teleport(nx, ny, heading_deg=float(np.degrees(hdg))):
                    print(f"[ring-collect] step {i}: placement failed; stopping")
                    break
                conn.step(max(1, int(args.step)))
            labels = ring.grab_ring_labels()
            if not labels:
                print("[ring-collect] no annotated frames returned "
                      "(annotations were not enabled on the cameras)",
                      flush=True)
                return 3
            t_wall = time.time()
            st = conn.get_state()
            rec_pos = [round(float(v), 3) for v in st.pos]
            rec_hdg = round(float(getattr(st, "heading", 0.0) or 0.0), 5)
            for role, (colour, ann) in labels.items():
                sample, audit = prepare_annotation_sample(
                    colour, ann, width=W, height=H, palette=palette)
                if int(audit.get("road_pixels") or 0) <= 0:
                    # A frame with no road at all is the failure signature
                    # of an annotation that did not render (cold buffer) or
                    # of a palette that mapped nothing.  Writing it would
                    # train "no road here" for a frame nobody verified.
                    skipped[role] = skipped.get(role, 0) + 1
                    continue
                d = root / role
                d.mkdir(parents=True, exist_ok=True)
                _save_ann = bool(args.save_annotation) and role in set(
                    args.roles or [role])

                idx = counts.get(role, 0)
                _payload = {
                    "colour": np.asarray(sample["colour"], dtype=np.uint8),
                    "label": np.asarray(sample["label"], dtype=np.uint8)}
                if _save_ann:
                    _payload["annotation_raw"] = np.asarray(
                        ann, dtype=np.uint8)
                    _payload["palette_version"] = np.array(
                        [str(palette.get("version") or "")])
                np.savez_compressed(d / f"frame_{idx:05d}.npz", **_payload)
                counts[role] = idx + 1
                audits[role] = dict(audit)
                frame_records.append({
                    "i": idx, "view": role, "exposure": i,
                    "t_wall": round(float(t_wall), 3),
                    "pos": rec_pos,
                    "heading": rec_hdg,
                    "line_pixels": int(audit.get("line_pixels") or 0),
                    "pixels": int(W) * int(H),
                    "path": f"{role}/frame_{idx:05d}.npz",
                })
            if args.step:
                conn.step(int(args.step))
        meta = {
            "stamp": stamp, "roles": {r: counts[r] for r in sorted(counts)},
            "width": W, "height": H,
            "classes": ["background", "road", "line"],
            "map_name": map_name,
            "map_name_source": map_name_source,
            "source_id": f"ring_{stamp}",
            "label_source": "beamng_annotation (road dense; line class is "
                            "NOT provided by the game - see module doc)",
            "audits_last_frame": audits,
            "palette_source": palette.get("source"),
            # The camera model per view, so a later identity check can
            # project pixels to the ground plane.  Without it a collection
            # can only be compared in the image plane: measured on the
            # first frozen holdouts, per-candidate identity needs the
            # extrinsics and the ring collector did not save them.
            "cameras": {r: {"offset": [float(v) for v in m.offset],
                            "fwd": [float(v) for v in m.fwd_local],
                            "up": [float(v) for v in m.up_local],
                            "fov_deg": float(m.fov_deg),
                            "width": int(m.width),
                            "height": int(m.height)}
                        for r, m in sorted(camera_ring_models(W, H).items())
                        if r in counts},
            "frames": frame_records,
            "note": "one run dir per mount so --split per-run holds each "
                    "view out separately; every view of one grab shares "
                    "`exposure`, so the same instant can never be split",
        }
        (root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        for role in sorted(counts):
            print(f"[ring-collect] {role:14s} frames={counts[role]:4d} "
                  f"skipped={skipped.get(role, 0):3d} "
                  f"road_px={audits.get(role, {}).get('road_pixels')}",
                  flush=True)
        print(f"[ring-collect] wrote {root}", flush=True)
    finally:
        try:
            ring.close()
        except Exception:
            pass
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
