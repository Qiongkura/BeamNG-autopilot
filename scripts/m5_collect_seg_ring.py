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

# Collection resolution: the segmentation training contract is 536x403.
W, H = 536, 403


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--roles", nargs="*", default=None,
                    help="ring roles to collect (default: the whole ring)")
    ap.add_argument("--out", default=None,
                    help="output root (default logs/m5_seg/ring_<stamp>)")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--step", type=int, default=10,
                    help="simulation steps between grabs (moves the car "
                         "slowly if it is already rolling)")
    args = ap.parse_args()

    from beamng_autopilot_tech.annotations import annotation_palette
    from beamng_autopilot.labeling.tech_annotation import (
        prepare_annotation_sample)

    conn = BeamNGConnector(
        "italy", "etk800",
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

        counts: dict[str, int] = {r: 0 for r in ring.cameras}
        skipped: dict[str, int] = {r: 0 for r in ring.cameras}
        audits: dict[str, dict] = {}
        # One grab = one EXPOSURE seen by every view.  The counter is what
        # lets the split keep all views of one instant on the same side
        # (T10: 跨视角同一曝光必须一起归组); without it the training entry
        # cannot even check that leak.
        frame_records: list[dict] = []
        for i in range(max(1, int(args.frames))):
            labels = ring.grab_ring_labels()
            if not labels:
                print("[ring-collect] no annotated frames returned "
                      "(annotations were not enabled on the cameras)",
                      flush=True)
                return 3
            t_wall = time.time()
            st = conn.get_state()
            rec_pos = [round(float(v), 3) for v in st.pos]
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

                idx = counts.get(role, 0)
                np.savez_compressed(
                    d / f"frame_{idx:05d}.npz",
                    colour=np.asarray(sample["colour"], dtype=np.uint8),
                    label=np.asarray(sample["label"], dtype=np.uint8))
                counts[role] = idx + 1
                audits[role] = dict(audit)
                frame_records.append({
                    "i": idx, "view": role, "exposure": i,
                    "t_wall": round(float(t_wall), 3),
                    "pos": rec_pos,
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
            "map_name": "italy",
            "source_id": f"ring_{stamp}",
            "label_source": "beamng_annotation (road dense; line class is "
                            "NOT provided by the game - see module doc)",
            "audits_last_frame": audits,
            "palette_source": palette.get("source"),
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
