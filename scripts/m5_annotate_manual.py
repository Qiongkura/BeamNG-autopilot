"""Manual frame annotator: paint lane/road labels for training.

The user paints over captured frames; every saved frame is a
``colour``/``label`` npz in the EXACT format ``m5_train_seg.py``
consumes (0 = background, 1 = road, 2 = line), so manual annotations
fold directly into the segmentation training set.

Sources:
    --episode latest   frames from a shadow episode (live camera views)
    --frames-dir DIR   existing frame npz/png directory
    --grab             capture N fresh frames from the live game first

Controls (OpenCV window, 2x zoom):
    left-drag  paint with the active class
    1 / 2 / 3  active class = line / road / background(erase)
    [ / ]      brush size down / up
    z          toggle zoom 2x / 1x
    s          save this frame (npz + preview png) and advance
    q          quit

    .venv\Scripts\python.exe scripts\m5_annotate_manual.py --episode latest
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from beamng_autopilot import config

CLS_LINE, CLS_ROAD, CLS_BG = 2, 1, 0
CLASS_COLOUR = {CLS_LINE: (0, 255, 0), CLS_ROAD: (255, 120, 0),
                CLS_BG: (60, 60, 60)}
CLASS_NAME = {CLS_LINE: "line", CLS_ROAD: "road", CLS_BG: "background"}


def _frames_from_episode(path: str):
    d = np.load(path, allow_pickle=True)
    return [(np.asarray(rgb, dtype=np.uint8), i)
            for i, rgb in enumerate(d["rgb"])]


def _frames_from_live(conn, n: int):
    from beamng_autopilot.runtime import build_camera_ring_provider
    ring, _ = build_camera_ring_provider(conn, "tech", 400, 300,
                                         roles=("front_main",))
    out = []
    for i in range(n):
        snap = ring.grab_ring()
        role = "front_main" if "front_main" in snap else next(iter(snap))
        out.append((snap[role][0], i))
        conn.step(10)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="manual lane annotation")
    ap.add_argument("--episode", type=str, default=None)
    ap.add_argument("--frames-dir", type=str, default=None)
    ap.add_argument("--grab", type=int, default=0,
                    help="capture N fresh frames from the live game first")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    frames = []
    if args.grab:
        from beamng_autopilot.connector import BeamNGConnector
        conn = BeamNGConnector("italy", "etk800",
                               port=config.runtime_port("tech"),
                               home=config.runtime_home(args.runtime
                                                        if hasattr(args, "runtime")
                                                        else "tech"))
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        frames = _frames_from_live(conn, int(args.grab))
    elif args.frames_dir:
        for f in sorted(glob.glob(os.path.join(args.frames_dir, "*.npz"))):
            d = np.load(f)
            frames.append((np.asarray(d["colour"], dtype=np.uint8), 0))
    elif args.episode:
        ep = (sorted(glob.glob(str(config.LOGS_DIR / "m5_e2e"
                               / "shadow_fsd_*.npz")),
                    key=os.path.getmtime)[-1]
              if args.episode == "latest" else args.episode)
        frames = _frames_from_episode(ep)
    if not frames:
        print("no frames to annotate (use --episode / --frames-dir / --grab)")
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out) if args.out
               else config.LOGS_DIR / "m5_seg" / f"manual_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[annotate] {len(frames)} frames | output -> {out_dir}")
    print("[keys] 1=line 2=road 3=erase | [ ]=brush | z=zoom | s=save+next "
          "| q=quit")

    brush = 6
    cls = CLS_LINE
    zoom = 2
    fi = 0
    rgb, fidx = frames[0]
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    painting = False
    last_pt = None

    def _render():
        ov = rgb.copy()
        for c, col in ((CLS_ROAD, (255, 120, 0)), (CLS_LINE, (0, 255, 0))):
            m = label == c
            ov[m] = (ov[m] * 0.55 + np.array(col) * 0.45).astype(np.uint8)
        big = cv2.resize(ov, (ov.shape[1] * zoom, ov.shape[0] * zoom),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"[{fi+1}/{len(frames)}] class={CLASS_NAME[cls]} "
                         f"brush={brush}  s=save+next q=quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3)
        cv2.putText(big, f"[{fi+1}/{len(frames)}] class={CLASS_NAME[cls]} "
                         f"brush={brush}  s=save+next q=quit",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow("annotate", big)

    def _paint(x, y):
        r, c = int(y / zoom), int(x / zoom)
        h, w = label.shape
        cv2.circle(label, (min(max(c, 0), w - 1), min(max(r, 0), h - 1)),
                   brush, cls, -1)

    def _mouse(event, x, y, flags, param):
        nonlocal painting, last_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            painting = True
            last_pt = (x, y)
            _paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and painting:
            if last_pt is not None:
                cv2.line(label,
                         (int(last_pt[0] / zoom), int(last_pt[1] / zoom)),
                         (int(x / zoom), int(y / zoom)), cls,
                         thickness=brush * 2)
            last_pt = (x, y)
            _paint(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            last_pt = None

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", _mouse)
    _render()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            cls = CLS_LINE
        elif key == ord("2"):
            cls = CLS_ROAD
        elif key == ord("3"):
            cls = CLS_BG
        elif key == ord("["):
            brush = max(1, brush - 2)
        elif key == ord("]"):
            brush = min(40, brush + 2)
        elif key == ord("z"):
            zoom = 1 if zoom == 2 else 2
        elif key == ord("s"):
            fp = out_dir / f"frame_{fidx:05d}.npz"
            np.savez_compressed(str(fp), colour=rgb, label=label)
            prev = out_dir / f"preview_{fidx:05d}.png"
            ov = rgb.copy()
            ov[label == CLS_ROAD] = (ov[label == CLS_ROAD] * 0.6
                                     + np.array([255, 120, 0]) * 0.4
                                     ).astype(np.uint8)
            ov[label == CLS_LINE] = (0, 255, 0)
            cv2.imwrite(str(prev), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            print(f"[saved] {fp.name}")
            fi += 1
            if fi >= len(frames):
                print("[annotate] all frames done")
                break
            rgb, fidx = frames[fi]
            label = np.zeros(rgb.shape[:2], dtype=np.uint8)
        _render()
    cv2.destroyAllWindows()
    print(f"[annotate] annotations in {out_dir} - training-ready by "
          "m5_train_seg.py --runs <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
