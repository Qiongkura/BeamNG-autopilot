"""Manual frame annotator: paint lane-line labels for training.

Paint the LINE markings on captured frames; every save writes a
``colour``/``label`` npz in the exact ``m5_train_seg.py`` contract
(0 = background, 1 = road, 2 = line), so hand annotations fold directly
into the segmentation training set.  Road class stays available for a
later pass; the typical line-only session just uses class 1.

Tools:
    pen (left-drag)         paint with the active class
    bucket (right-click/f)  flood-fill the connected same-label region
                            with the active class - draw a closed
                            outline, then click inside it
Controls:
    1 / 2 / 3   class = line / road / background(erase)
    b           toggle pen / bucket
    f           switch directly to bucket
    p           switch directly to pen
    u           undo (last stroke / fill / clear)
    c           clear the whole label
    trackbar    brush size 1-40
    z           zoom 2x / 1x
    s           save + next frame
    q           quit

    .venv\\Scripts\\python.exe scripts\\m5_annotate_manual.py \\
        --frames-dir logs/m5_seg/manual_capture
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
CLASS_NAME = {CLS_LINE: "line", CLS_ROAD: "road", CLS_BG: "erase"}
WIN = "annotate"


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
        rt = getattr(args, "runtime", "tech")
        conn = BeamNGConnector("italy", "etk800",
                               port=config.runtime_port(rt),
                               home=config.runtime_home(rt))
        conn.open(launch=False)
        conn.attach_vehicle(already_open=True)
        frames = _frames_from_live(conn, int(args.grab))
    elif args.frames_dir:
        for f in sorted(glob.glob(os.path.join(args.frames_dir, "*.npz"))):
            d = np.load(f)
            try:
                idx = int(os.path.basename(f).split("_")[-1].split(".")[0])
            except ValueError:
                idx = len(frames)
            frames.append((np.asarray(d["colour"], dtype=np.uint8), idx))
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

    cls = CLS_LINE
    tool = "pen"                     # pen | bucket
    zoom = 2
    fi = 0
    save_i = [0]
    undo_stack: list = []

    rgb, fidx = frames[0]
    label = np.zeros(rgb.shape[:2], dtype=np.uint8)
    painting = False
    last_pt = None

    def _push_undo():
        undo_stack.append(label.copy())
        if len(undo_stack) > 25:
            undo_stack.pop(0)

    def _render():
        ov = rgb.copy()
        m_road = label == CLS_ROAD
        m_line = label == CLS_LINE
        ov[m_road] = (ov[m_road] * 0.6
                      + np.array([255, 120, 0]) * 0.4).astype(np.uint8)
        ov[m_line] = (0, 255, 0)
        big = cv2.resize(ov, (ov.shape[1] * zoom, ov.shape[0] * zoom),
                         interpolation=cv2.INTER_NEAREST)
        tool_txt = f"tool={tool} class={CLASS_NAME[cls]} " \
                   f"undo={len(undo_stack)}"
        for txt, row in (
                (f"[{fi + 1}/{len(frames)}] src#{fidx} {tool_txt}", 20),
                ("1/2/3 class  b=tool  u=undo  c=clear  s=save+next  "
                 "q=quit", 40)):
            cv2.putText(big, txt, (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(big, txt, (8, row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, big)

    def _paint(x, y):
        r, c = int(y / zoom), int(x / zoom)
        h, w = label.shape
        cv2.circle(label, (min(max(c, 0), w - 1), min(max(r, 0), h - 1)),
                   brush, cls, -1)

    def _bucket(x, y):
        r, c = int(y / zoom), int(x / zoom)
        h, w = label.shape
        if not (0 <= r < h and 0 <= c < w):
            return
        old = int(label[r, c])
        if old == cls:
            return
        m = (label == old).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(m, ff, (c, r), 0, loDiff=0, upDiff=0, flags=4)
        region = (m == 0) & (label == old)
        label[region] = cls

    def _mouse(event, x, y, flags, param):
        nonlocal painting, last_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            _push_undo()
            if tool == "bucket":
                _bucket(x, y)
            else:
                painting = True
                last_pt = (x, y)
                _paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and painting:
            if last_pt is not None:
                cv2.line(label,
                         (int(last_pt[0] / zoom), int(last_pt[1] / zoom)),
                         (int(x / zoom), int(y / zoom)), cls,
                         thickness=max(1, brush * 2))
            last_pt = (x, y)
            _paint(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            last_pt = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            _push_undo()
            _bucket(x, y)

    def _brush_cb(v):
        pass

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, _mouse)
    cv2.createTrackbar("brush", WIN, 6, 40, _brush_cb)
    _render()
    while True:
        brush = max(1, cv2.getTrackbarPos("brush", WIN))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            cls = CLS_LINE
        elif key == ord("2"):
            cls = CLS_ROAD
        elif key == ord("3"):
            cls = CLS_BG
        elif key == ord("b"):
            tool = "bucket" if tool == "pen" else "pen"
        elif key == ord("f"):
            tool = "bucket"
        elif key == ord("p"):
            tool = "pen"
        elif key == ord("u") and undo_stack:
            label[:] = undo_stack.pop()
        elif key == ord("c"):
            _push_undo()
            label[:] = 0
        elif key == ord("z"):
            zoom = 1 if zoom == 2 else 2
        elif key == ord("s"):
            save_i[0] += 1
            fp = out_dir / f"frame_{save_i[0]:05d}.npz"
            np.savez_compressed(str(fp), colour=rgb, label=label)
            prev = out_dir / f"preview_{save_i[0]:05d}.png"
            ov = rgb.copy()
            ov[label == CLS_ROAD] = (ov[label == CLS_ROAD] * 0.6
                                     + np.array([255, 120, 0]) * 0.4
                                     ).astype(np.uint8)
            ov[label == CLS_LINE] = (0, 255, 0)
            cv2.imwrite(str(prev), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            print(f"[saved] {fp.name} (src#{fidx})")
            fi += 1
            if fi >= len(frames):
                print("[annotate] all frames done")
                break
            rgb, fidx = frames[fi]
            label = np.zeros(rgb.shape[:2], dtype=np.uint8)
            undo_stack.clear()
        _render()
    cv2.destroyAllWindows()
    print(f"[annotate] annotations in {out_dir} - training-ready by "
          "m5_train_seg.py --runs <this dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
