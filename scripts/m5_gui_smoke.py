"""GUI smoke test for m5_launcher (no game required).

Runs headful-once on Windows; verifies:
  - last-session telemetry panel: auto-popup, toggle, summary label, image draw
  - env auto-fill of map/vehicle from live telemetry
  - BirdView production render path (roads / rte / obstacle boxes)
  - ENDED frame mapping
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from beamng_autopilot import config
from beamng_autopilot.vision.lanes import LaneMarking
from beamng_autopilot.vision.projection import default_camera
from beamng_autopilot.visionview import render_camera_overlay

# Redirect to a private temp dir so a live m5_autopilot instance cannot
# clobber the fixture files this test writes/reads.
config.LOGS_DIR = Path(tempfile.mkdtemp(prefix="m5_gui_smoke_"))

import m5_launcher as launcher_mod
from m5_launcher import BirdView, LauncherApp, MODE_CN, MODE_COLOR


def main() -> int:
    tele = config.LOGS_DIR / "telemetry"
    tele.mkdir(parents=True, exist_ok=True)
    live_path = tele / "live.json"
    last_path = tele / "last_session.json"
    png_path = tele / "smoke_chart.png"

    orig = {
        p: (p.read_bytes() if p.exists() else None)
        for p in (live_path, last_path, png_path)
    }

    img = Image.new("RGB", (320, 180), "#0b0f14")
    for i in range(0, 320, 8):
        img.paste((63, 185, 80), (i, 160 - i // 4, i + 3, 170))
    img.save(png_path)

    last = {"ts": 1754990000, "png": str(png_path), "duration": 83.4,
            "max_speed": 21.3, "avg_speed": 9.8,
            "throttle_ratio": 0.42, "brake_ratio": 0.13}
    last_path.write_text(json.dumps(last, ensure_ascii=False), encoding="utf-8")

    def write_live(mode: str, creep: int = 0) -> None:
        live = {"t": 12.5, "speed": 8.3, "throttle": 0.55, "brake": 0.0,
                "steer": 0.02, "g_lat": 0.1, "g_lon": -0.05, "heading": 1.2,
                "pos": [0.0, 0.0, 0.0],
                "extra": {"auto": True, "mode": mode, "ended": 1,
                          "creep": creep,
                          "cruise": 16.7,
                          "target": 8.5, "obs": 1, "obs_d": 18.0,
                          "vis": 1, "sen": "OK", "route": 1, "goal_d": 120.0,
                          "blk": "", "lanes": 2,
                          "env": {"map": "west_coast_usa",
                                  "vehicle": "etk800"},
                          "roads": [[[5.0, -3.0], [25.0, -3.0],
                                     [45.0, -3.0]]],
                          "markings": [
                              {"color": "white", "kind": "solid",
                               "poly": [[8.0, -2.0], [20.0, -2.0],
                                        [32.0, -2.0]]},
                              {"color": "yellow", "kind": "dashed",
                               "poly": [[8.0, 2.0], [20.0, 2.0],
                                        [32.0, 2.0]]},
                          ],
                          "rte": [[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]],
                          "boxes": [[15.0, 0.0, 2.3, 1.1, "car"]]}}
        live_path.write_text(json.dumps(live), encoding="utf-8")

    write_live("follow")

    checks: list[tuple[str, bool]] = []

    def ok(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(("PASS" if cond else "FAIL"), name)

    root = tk.Tk()
    root.geometry("1220x820")
    app = LauncherApp(root)

    # 1. initial state before any poll
    ok("chart hidden before poll", app._chart_visible is False)
    ok("map default before poll",
       app.map_var.get() == config.DEFAULT_MAP)
    ok("runtime selector default",
       app.runtime_var.get() in ("auto", "steam", "tech"))
    ok("nav world default off", app.nav_world_var.get() is False)
    btn_idle = app.btn_chart["text"]

    # 1b. speed limit apply sends a live set_speed command to the helper
    app.speed_var.set("45")
    real_alive = app._m5_alive
    app._m5_alive = lambda: True
    app.apply_speed()
    app._m5_alive = real_alive
    ctl = json.loads((config.LOGS_DIR / "autopilot_ctl.json").read_text(
        encoding="utf-8"))
    ok("speed apply sends set_speed",
       ctl.get("cmd") == "set_speed"
       and abs(float(ctl.get("value")) - 12.5) < 1e-9
       and app.speed_var.get() == "45")

    # 2. real poll path auto-pops the chart (new last_session.json)
    app.poll()
    ok("chart auto-visible via poll", app._chart_visible is True)
    ok("chart button text changed",
       app.btn_chart["text"] != btn_idle and app.btn_chart["text"] != "")
    s = app.lbl_session["text"]
    ok("session summary label",
       "83s" in s and "77" in s and "42%" in s and "13%" in s)

    # 3. chart canvas actually draws the image
    root.update()
    app._redraw_chart()
    ok("chart canvas has image item",
       len(app.cv_chart.find_all()) > 0)
    ok("cruise limit label", "60" in app.lbl_cruise["text"])

    # 4. toggle hide / show
    app.toggle_chart()
    root.update()
    ok("toggle hides chart",
       app._chart_visible is False and not app.cv_chart.winfo_ismapped())
    app.toggle_chart()
    root.update()
    ok("toggle shows chart",
       app._chart_visible is True and app.cv_chart.winfo_ismapped())

    # 5. same mtime does not re-trigger / no crash
    app._check_last_session()
    ok("repeat check no crash", app._chart_visible is True)

    # 5b. front-camera overlay draws lane markings without a route
    cam = default_camera(320, 240)
    pos = np.array([0.0, 0.0, 0.0])
    markings = [
        LaneMarking(world=np.array(
            [[5.0, -1.8], [10.0, -1.8], [15.0, -1.8], [20.0, -1.8]]),
            pixels=np.zeros((4, 2)), color="white", kind="solid"),
        LaneMarking(world=np.array(
            [[5.0, 1.8], [10.0, 1.8], [15.0, 1.8], [20.0, 1.8]]),
            pixels=np.zeros((4, 2)), color="yellow", kind="dashed"),
    ]
    base = np.full((240, 320, 3), 60, np.uint8)
    out = render_camera_overlay(base.copy(), None, pos, 0.0, cam,
                                lane_markings=markings)
    white_px = int(np.sum(np.all(out == (255, 255, 255), axis=2)))
    yellow_px = int(np.sum(np.all(out == (0, 255, 255), axis=2)))
    ok("overlay markings drawn without route",
       white_px > 50 and yellow_px > 50)

    # 6. env auto-fill (reset to default first)
    app.map_var.set(config.DEFAULT_MAP)
    app.veh_var.set(config.DEFAULT_VEHICLE)
    app._update_eid()
    ok("env map auto-filled", app.map_var.get() == "west_coast_usa")
    ok("env veh auto-filled", app.veh_var.get() == "etk800")
    ok("stat map label", app.stat_labels["map"]["text"] == "west_coast_usa")
    ok("stat veh label", app.stat_labels["veh"]["text"] == "etk800")
    ok("stat lanes label", app.stat_labels["lanes"]["text"] == "2")

    # 7. BirdView production render (roads/markings/rte/boxes)
    root.update()
    w, h = app.bird.winfo_width(), app.bird.winfo_height()
    items = len(app.bird.find_all())
    print(f"    bird size {w}x{h}, items {items}")
    ok("birdview rendered",
       w > 60 and h > 60 and items > 8)

    # 8. ENDED frame mapping
    write_live("ENDED")
    app._update_eid()
    ok("ENDED mode label", app.lbl_mode["text"] == MODE_CN["ENDED"])
    ok("ENDED cruise label", "60" in app.lbl_cruise["text"])
    ok("ENDED mapped to cyan", MODE_COLOR["ENDED"] != "")
    ok("ENDED reason badge not empty", bool(MODE_CN["ENDED"]))

    # 8b. creep indicator on the mode label
    write_live("follow", creep=1)
    app._update_eid()
    ok("creep mode suffix", "爬行" in app.lbl_mode["text"])

    # 8c. manual-run audit log: a launcher-spawned helper must leave a
    #     per-run autopilot.log + run.json behind.
    class _FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("fake line one\nfake line two\n")
            self.pid = 424242

        def poll(self) -> int:
            return 0

    old_popen = launcher_mod.subprocess.Popen
    try:
        launcher_mod.subprocess.Popen = _FakePopen
        app.start_m5()
    finally:
        launcher_mod.subprocess.Popen = old_popen
    run_dirs = sorted((config.LOGS_DIR / "manual_runs").glob("run_*"))
    ok("manual run dir created", len(run_dirs) == 1)
    run_dir = run_dirs[0]
    log_path = run_dir / "autopilot.log"
    meta_path = run_dir / "run.json"
    ok("autopilot.log created", log_path.exists())
    ok("run.json created", meta_path.exists())
    deadline = time.time() + 2.0
    while app.m5_log is not None and time.time() < deadline:
        time.sleep(0.01)
    log_txt = (log_path.read_text(encoding="utf-8")
               if log_path.exists() else "")
    ok("log captured helper output",
       "fake line one" in log_txt and "fake line two" in log_txt)
    ok("log records helper exit", "helper exited rc=0" in log_txt)
    meta = (json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists() else {})
    ok("run.json records exited",
       meta.get("status") == "exited" and meta.get("exit_code") == 0)
    meta_args = meta.get("args") or []
    expected_runtime = app.runtime_var.get()
    ok("run.json passes raw --runtime for auto-detect",
       "--runtime" in meta_args and expected_runtime in meta_args)
    ok("nav world hidden arg passed",
       "--nav-world" in meta_args and "0" in meta_args)

    # 9. BirdView standalone render with roads only (fresh canvas in its
    #    own Toplevel; packing into the app-filled root never maps it)
    sub = tk.Toplevel(root)
    bv = BirdView(sub, width=300, height=300)
    bv.pack()
    root.update()
    bv.render({"pos": [0, 0, 0], "heading": 0.0,
               "extra": {"roads": [[[10.0, -3.0], [30.0, -3.0],
                                    [50.0, -3.0]]],
                         "markings": [
                             {"color": "white", "kind": "solid",
                              "poly": [[12.0, -1.0], [30.0, -1.0],
                                       [48.0, -1.0]]},
                             {"color": "yellow", "kind": "dashed",
                              "poly": [[12.0, 1.0], [30.0, 1.0],
                                       [48.0, 1.0]]},
                         ],
                         "rte": [[0.0, 0.0], [20.0, 0.0]],
                         "boxes": [[12.0, 0.0, 2.3, 1.1, "car"],
                                   [18.0, 2.0, 2.6, 1.4, "truck"],
                                   [22.0, -2.0, 1.0, 1.0, "person"],
                                   [30.0, 5.0, 3.0, 2.0, "wall"]]}})
    bv_items = len(bv.find_all())
    item_types = [bv.type(i) for i in bv.find_all()]
    line_types = [t for t in item_types if t == "line"]
    dash_lines = [i for i in bv.find_all()
                  if bv.type(i) == "line" and bv.itemcget(i, "dash")]
    ok(f"standalone birdview ({bv_items} items)", bv_items > 8)
    ok("marking lines drawn", len(line_types) >= 4)
    ok("dashed marking drawn", len(dash_lines) >= 1)

    sub.destroy()
    root.destroy()

    for p, data in orig.items():
        if data is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(data)

    failed = [n for n, c in checks if not c]
    print(f"\n=== {len(checks) - len(failed)}/{len(checks)} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
