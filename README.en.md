# BeamNG Autopilot — Visual Autopilot Research

<div align="center">

[中文](README.md) | **English**

</div>

An autonomous driving research project based on BeamNG.drive + BeamNGpy. The roadmap: first layered (perception/decision/control), then upgrade to end-to-end imitation learning and image RL.

## Milestones

- M1 Closed-loop tracking: Pure Pursuit + PID, vehicle follows closed loop automatically ✅
- M2 Visual perception: Camera calibration + tire mark strip detection + path projection ✅
- M3 End-to-end imitation learning: DAVE-2 style CNN (single frame image -> steering) 🔨 In progress
- M4 Decision layer: DQN discrete actions (cruise/decelerate/lane change/overtake), Stable-Baselines3
- M5 Integration demo: Perception -> Decision -> Control, new energy vehicle interface (front view + BEV + dashboard)
- M6 Image end-to-end RL (SAC/PPO from pixels)

## Environment

- BeamNG.drive 0.39+ (Steam version; BeamNG.tech optional enhancement). Installation directory auto-detected
  (Steam registry / `libraryfolders.vdf` / common paths), fallback to environment variable
  `BEAMNG_HOME` if not found; user directory defaults to current user's
  `%LOCALAPPDATA%\BeamNG.drive\<version>`, can be overridden with environment variable `BEAMNG_USER`.
- Python 3.10 + venv (`--system-site-packages`, reuse system torch cu128 / OpenCV)
- GPU recommended: 6GB VRAM or higher (for YOLO detection + HUD)

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Before first use, it's recommended to run environment self-check, outputting dependencies / game paths / resources / runtime status checklist:

```powershell
.venv\Scripts\python.exe scripts\m5_env_check.py
```

Frame capture performance probe (optional, requires game window on primary display; dxcam needs manual `pip install dxcam`):

```powershell
.venv\Scripts\python.exe scripts\bench_grab_screen.py
```

## Learned Pavement/Marking Segmentation (M5-B)

Traditional CV color threshold fails on Tech rendered frames (measured marking recall 1.5%, boundary error 0.8-1.0m). Alternative: train lightweight UNet using BeamNG.tech annotation pixel ground truth (background/asphalt/line 3 classes), inference only takes RGB frames, works on both Steam / Tech.

```powershell
# 1. Collect training data (Tech instance running, AI drives along road automatically)
.venv\Scripts\python.exe scripts\m5_collect_seg.py --frames 500

# 2. Train (multiple runs can be combined; augmentation + class weighting built-in)
.venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_* `
    --epochs 60 --lr 5e-4 --out logs\m5_seg\seg_model

# 3. Offline evaluation (no game needed, direct model verification)
.venv\Scripts\python.exe scripts\m5_eval_seg.py --runs logs\m5_seg\run_* `
    --model logs\m5_seg\seg_model\best.pt --save

# 4. Real-world ground truth comparison (CV vs learned, requires Tech instance)
.venv\Scripts\python.exe scripts\m5_lane_truth_probe.py --frames 60 --drive
.venv\Scripts\python.exe scripts\m5_lane_truth_probe.py --frames 60 --drive `
    --model logs\m5_seg\seg_model\best.pt
```

When the model is ready, `m5_autopilot.py` and `m5_lane_state_view.py` will automatically load `logs/m5_seg/seg_model/best.pt` (or specify with `--seg-model <path>`), with automatic fallback to classic CV if loading fails.

## Dual Runtime (Steam / Tech) Parallel Development

Core codebase maintains Steam version compatibility and open-source readiness; BeamNG.tech exclusive features are placed in separate package `beamng_autopilot_tech/`, only lazily imported when Tech is detected, regular users using open-source repository won't be affected.

- `BEAMNG_RUNTIME=auto|steam|tech`, default `auto`: auto-selects after connection based on BeamNGpy's `tech_enabled()`.
- `BEAMNG_TECH_HOME` points to Tech installation directory; Steam still uses `BEAMNG_HOME`.
- `BEAMNG_TECH_USER` points to Tech user directory, defaults to `...\BeamNG.drive\0.38`, separate from Steam's `0.39` to avoid cross-contamination of saves/settings.
- `BEAMNG_PORT` optional, default `64256`; when running Steam and Tech instances simultaneously, use different ports.
- Startup/main flow entry points (`launch_game.py` / `m5_launcher.py` / `m5_autopilot.py` / `m5_e2e_test.py` / `m5_drive_test.py`), M1-M3 collection/calibration scripts (`m1_smoke_test.py` / `m1_record_track.py` / `m1_follow_track.py` / `m1_calibrate_heading.py` / `m2_capture.py` / `m2_calibrate_camera.py` / `m3_collect_bc.py` / `m3_drive_bc.py`) and M5 probe/validation scripts (`m5_nav_route_test.py` / `m5_obstacle_test.py` / `m5_twitch_verify.py` / `m5_vision_probe.py` / `m5_loop_probe.py`) all support `--runtime auto|steam|tech`; launcher UI left settings area also has the same "Runtime" dropdown. Before startup `auto` prioritizes Tech (when Tech installation directory exists locally), otherwise falls back to Steam.
- Runtime state/control/perception probes (`m5_probe_state.py` / `m5_probe_elec.py` / `m5_ctrl_probe.py` / `m5_handover_test.py` / `m5_emergency_stop.py` / `m5_steer_sign_probe.py` / `m5_perception_probe.py` / `m5_route_scan_probe.py`) also support `--runtime auto|steam|tech`.
- Steam path: existing window screenshot + Lua ray/scene/vehicle perception, no Tech license required.
- Tech path: `beamng_autopilot_tech.providers` lazily creates beamngpy `Camera` / `Lidar` sensors.
- `diag_*`, `m5_wall_*`, `m5_live_wall_probe*.py`, `m5_live_route_probe.py` etc. pure diagnostic probes maintain Steam semantics, no dual runtime dispatch; they are only for Steam version troubleshooting.
- Open-source release can omit `beamng_autopilot_tech/` entirely; core modules don't import it at top level.

```powershell
.venv\Scripts\python.exe scripts\launch_game.py --runtime steam
.venv\Scripts\python.exe scripts\launch_game.py --runtime tech
.venv\Scripts\python.exe scripts\m5_autopilot.py --runtime auto --attach
```

## M2 Conclusions (Visual Perception)

- Camera solution: Early Steam version without tech activated could only use `set_relative_camera` + PrintWindow screenshot (1064x772). Now BeamNG.tech license is active (tech channel `127.0.0.1:64256` TechCom / TechVE both working), can gradually switch to beamngpy real sensors (Camera / LiDAR / IMU).
- With map priors, tire mark strip detection fidelity is high: at 4m r=+0.93 (mean error 0.21m), 6m r=+0.90 (0.25m), 12m r=+0.91 (0.75m).
- Lateral offset difference drift(12m−4m) correlates with steering input r=+0.79, indicating "far point left/right offset" is indeed an effective steering feature.
- But pure vision without priors (center seeding + temporal tracking) at 2.4fps low frame rate is unreliable: lat12 with steering correlation r≈−0.33, locks onto shadows/other dark features in curves.
- Conclusion: handcrafted features have low ceiling and rely on precise calibration → transition to M3 neural network end-to-end imitation learning.

## M3 Imitation Learning (BC)

- Collection: `scripts/m3_collect_bc.py` high-frequency screenshot + expert (Pure Pursuit) steering/throttle/brake labels; frames captured before commands to ensure image-label alignment.
- Training: `scripts/m3_train_bc.py` DAVE-2 CNN (200x66 RGB -> steer), temporal 80/20 split, monitors val MAE / R², automatically saves best model and curves.
- Inference: `scripts/m3_drive_bc.py` loads model for real-world closed-loop driving, PID maintains target speed.
- Proof of concept: used M2's 229 frames (2.4fps) to run through full pipeline, val MAE=0.048, but model degenerated to constant prediction (R²≈0) — frame rate too low, steering labels almost all zero, must use M3 high-frequency collection (target ≥10fps, multiple curves, multiple laps) to have training value.

## Running

```powershell
# Smoke test: connect to game and drive straight for 3 seconds
.venv\Scripts\python.exe scripts\m1_smoke_test.py

# Record trajectory: AI drives along closed loop, saves reference trajectory
.venv\Scripts\python.exe scripts\m1_record_track.py

# Track following: Pure Pursuit drives along trajectory automatically for 2 laps
.venv\Scripts\python.exe scripts\m1_follow_track.py --track data\track_smallgrid.npz

# M2 collection (low frame rate, for perception calibration)
.venv\Scripts\python.exe scripts\m2_capture.py --track data\track_smallgrid.npz --laps 1

# M3 high-frequency collection (BC training data, 320x180 frames)
.venv\Scripts\python.exe scripts\m3_collect_bc.py --track data\track_smallgrid.npz --speed 8.0 --laps 3

# M3 training
.venv\Scripts\python.exe scripts\m3_train_bc.py --runs logs\m3_bc\20260811_* --epochs 60 --out logs\m3_bc\bc_steer

# M3 real-world inference (BC driving)
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --speed 8.0 --duration 180
```

Running will launch BeamNG.drive window, allowing real-time observation of vehicle. Telemetry and datasets saved in `logs/`.

## Real-time Telemetry HUD

`m3_drive_bc.py` / `m1_follow_track.py` by default pop up visualization telemetry window (throttle, brake, steering, speed, G-force meter, heading, lap count), with front camera preview; press `q` / `ESC` to end driving early. Disable with:

```powershell
# Turn off entire HUD window
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --no-hud
# Only remove camera preview, keep dashboard
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --no-camera
```

Can also launch dashboard in separate terminal (without camera), reading real-time telemetry from running process:

```powershell
.venv\Scripts\python.exe scripts\m4_dashboard.py
```

Decision layer real-time visualization (reads autopilot actual published telemetry, displays perception fusion / planner mode / speed decision chain / traffic rules / control output + bird's eye view + speed time series):

```powershell
.venv\Scripts\python.exe scripts\m5_decision_view.py
```

## M5 In-game Autopilot Assistant (Manually Activatable)

Activate/deactivate autopilot via hotkeys in game, vehicle strictly follows game built-in navigation route (blue arrow route generated by clicking destination on large map), with Tesla-style visual overlay and post-run telemetry charts.

When not started with `--attach`, script automatically loads `italy` (simulated Italy) map's `spawn_crossroads` intersection spawn point, places vehicle on nearby road network node aligned with road direction (avoids spawning under map for maps like `hirochi_raceway` where origin isn't on road surface). Route directly grabs game built-in navigation (no longer uses road network A* pathfinding), script only responsible for grabbing then following.

### Startup (Recommended: start game first, enter map, position vehicle)

```powershell
.venv\Scripts\python.exe scripts\m5_autopilot.py --attach
```

Without `--attach`, script will start game itself and load `italy` map intersection spawn point (`spawn_crossroads`):

```powershell
.venv\Scripts\python.exe scripts\m5_autopilot.py
```

### Hotkeys (Global, press directly in game)

| Key | Function |
| --- | --- |
| `F8` | Visual overlay on/off (3D world route + front view projection + bird's eye view) |
| `F9` | Autopilot on/off |
| `F10` | Grab in-game navigation route (first press `M` to open large map, click destination to generate blue route) |
| `F11` | Clear route |
| `F12` | Exit |

### Gameplay

1. Press `M` in game to open large map, click destination, after closing map blue navigation route appears;
2. Press `F10` to grab this navigation route (prompts "navigation route grabbed: N pts");
3. Press `F9` to activate autopilot: Pure Pursuit + curve speed limiting + PID following, strictly follows game-generated navigation route, automatically brakes at destination to end;
4. During/after autopilot can use `F8` to toggle 3D world overlay and front view projection + bird's eye view in HUD;
5. Each time autopilot ends (arrived / manually stopped / timeout) automatically pops up "throttle / brake / speed" three-segment continuous bar chart, while saving PNG to `logs/telemetry/m5_telemetry_*.png`;
6. After autopilot closes, vehicle control is returned to you (manual driving).

Common parameters: `--speed 10` cruise speed (m/s), `--max-run 600` maximum seconds per run, `--no-hud` turn off HUD window, `--no-show` only save charts without popping up. Console's `Speed Limit (km/h)` will be passed to assistant as `--speed`, and can be updated during runtime via `set_speed`.

## Console Interface (One-click Startup + EID Environment Information Display)

`scripts/m5_launcher.py` is a software-like launch console: no need to memorize commands, no need to open multiple windows, buttons can complete "start game → start assistant → one-click autopilot", right side displays environment information (EID) in real-time.

Double-click `启动自动驾驶.vbs` (Start Autopilot.vbs) in project root directory to directly open console interface (auto-locates venv, hides black window, no need to open terminal); can also manually run:

```powershell
.venv\Scripts\python.exe scripts\m5_launcher.py
```

### Interface Layout

- Left side buttons: `启动游戏` (Start Game) (auto-launches BeamNG in TCP mode), `启动助手` (Start Assistant) (runs m5_autopilot.py), `一键自动驾驶(F9)` (One-click Autopilot (F9)), `抓取路线(F10)` (Grab Route (F10)), `清除路线` (Clear Route), `停止自动驾驶` (Stop Autopilot);
- Left side settings area: `限速 (km/h)` (Speed Limit (km/h)) input + `应用` (Apply) button (takes effect immediately when assistant running, saves for next startup if not), map, vehicle model, `--attach` and marker point toggle;
- Right side EID panel: current speed (large digit) + target speed + speed limit display, mode badge (manual / cruise / obstacle avoidance), throttle / brake / steering bars, G-force graph, obstacle count, nearest obstacle distance, visual detection targets, sensor status, route point count, distance to destination, elapsed time, heading angle;
- Center `BirdView`: top-down map, real-time drawing of route points, obstacle bounding boxes and vehicle heading;
- Bottom log area: real-time scrolling display of assistant process output.

### Button Flow

1. Click `启动游戏` (Start Game), wait for game to enter map and position vehicle;
2. Press `M` in game to open large map and select destination (blue navigation route appears);
3. Return to console and click `抓取路线(F10)` (Grab Route (F10)) (corresponds to in-game hotkey F10), then click `一键自动驾驶(F9)` (One-click Autopilot (F9));
4. To stop, click `停止自动驾驶` (Stop Autopilot) (corresponds to pressing F9 again), exiting won't close game.

The `限速 (km/h)` (Speed Limit (km/h)) in settings area will be passed as `--speed` when starting assistant; clicking `应用` (Apply) while assistant running will immediately send `set_speed`, if not running will save and take effect on next startup.

### Communication Notes

Console communicates with m5 assistant via `logs/autopilot_ctl.json` command bridge (with monotonically increasing sequence watermark for replay prevention), assistant writes EID data in real-time to `logs/telemetry/live.json`, console reads per frame for refresh. Besides toggle commands like F9/F10/F11, command bridge also supports `set_speed` with numeric value for runtime cruise speed updates. Only needed for this interface, command-line gameplay unaffected.

## M5 Visual Obstacle Avoidance (YOLO Front View + Back-projection)

M5 autopilot enables visual obstacle perception by default: background thread warms up YOLOv8n, main loop grabs game window frame approximately every 0.33s (`--vision-rate 3`), YOLO detects pedestrians/cars/motorcycles/buses/trucks, then through current game camera's real pose (Lua query `getCameraPosition / Forward / Up / FovDeg`) back-projects detection box bottom center to ground plane (z=0), obtaining obstacle world coordinates, then merges with scene/ray obstacles before passing to planner for avoidance.

- Dependencies: `weights/yolov8n.pt` (~6.5MB, auto-downloads to project directory if missing) + `.yolo/` (ultralytics config directory, avoids writing to AppData causing permission crashes).
- Automatic degradation when game window unavailable/blank frame: silently skips visual scanning and counts, doesn't affect other perception and control.
- HUD overlays cyan detection boxes (category + confidence), status line shows `vis=N` (current visual obstacle count).
- Same vehicle detected by both visual and scene/ray merges into single obstacle by distance (`merge_obstacles`), avoiding duplicate avoidance.

### Visual Probe (Recommended to run first to verify detection effect)

Run after entering game map, no need to start autopilot:

```powershell
# Single frame: print detected obstacles (world coordinates / distance / size)
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --once

# Continuous detection and save annotated frames to logs\m5_vision\
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --save

# Pop-up real-time preview (press q to exit)
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --show
```

### Visual-related Parameters

- `--no-vision-obstacles`: Disable visual obstacle avoidance (only use scene + ray obstacles).
- `--vision-conf 0.35`: YOLO confidence threshold (default 0.35, increase if many false positives).
- `--vision-rate 3`: Visual scanning frequency Hz (default 3; can increase to 5-10 on RTX 5070).
- `--max-dist 55` (probe): Ignore detections beyond this distance.

Note: Visual detection relies on current game camera real pose (Lua query), works consistently with player free camera; `m5_autopilot.py --attach` has visual enabled by default.

## Development Log

Recorded in chronological order; entries before 2026-08-14 reconstructed from file timestamps, README and docs, after that based on git commits. New changes appended to end of this section.

### 2026-08-11

- Project start and M1 closed-loop tracking: set up venv and `requirements.txt`, implemented trajectory recording/replay (`track.py`), PID (`control/pid.py`), Pure Pursuit (`control/pure_pursuit.py`), `m1_smoke_test.py` / `m1_record_track.py` / `m1_follow_track.py` run through closed loop, and saved sample trajectory `data/track_smallgrid.npz`.
- M2 visual perception start: `vision/band.py` tire mark strip detection, `m2_capture.py` / `m2_calibrate_camera.py` complete camera calibration and low frame rate collection; conclusion: 4m strip r=+0.93, 12m r=+0.91, but pure vision without priors unreliable at 2.4fps.
- M3 imitation learning start: `bc.py` + `m3_train_bc.py` implement DAVE-2 CNN training pipeline, used M2's 229 frames to run through PoC (val MAE=0.048, R²≈0), confirmed low frame rate, near-zero steering labels data has no training value.
- Telemetry visualization: `hud.py`, `telemetry_chart.py`, `m4_dashboard.py` dashboard prototype, M1/M3 driving can pop HUD real-time view throttle/brake/steering/speed.

### 2026-08-12

- Hotkey framework: `beamng_autopilot/hotkeys.py`, laying foundation for M5 in-game F8/F9/F10/F11/F12 control.
- Control chain diagnostics: `diag_parkingbrake.py` / `diag_disconnect.py` / `diag_r_latch_drive.py` / `diag_gear_map.py` / `diag_gearbox_info.py` / `diag_gearbox_list.py` / `diag_arcade_standstill.py` / `diag_arcade_neutral.py`, troubleshooting handbrake, disconnection, reverse gear self-locking, gear mapping and arcade control issues.
- Perception probes: `m5_rayframe_probe.py` / `m5_rayground_probe.py` / `m5_castray_struct.py` / `m5_castray_compare.py` / `m5_live_blocker_probe.py` / `m5_watchdog_probe.py`, added `watchdog.py` watchdog; recorded twitch/park scenarios, later incorporated into `m5_offline_validate.py` regression.
- Visual detection: `vision/detection.py` added YOLO detection and ground back-projection, downloaded `weights/yolov8n.pt`; `control/speed.py` speed control; `m5_vision` detection results implemented.

### 2026-08-13

- Wall/route/obstacle avoidance troubleshooting: `m5_wall_shape_probe.py` / `m5_wall_fan_probe.py` / `m5_wall_multi_probe.py` / `m5_live_wall_probe.py` / `m5_live_wall_probe2.py` / `m5_wall_route_probe.py` / `m5_live_route_probe.py` / `m5_live_planner_diag.py`, covering wall shape, fan/multi-wall, live route and planner diagnostics.
- Gear control: `control/gearbox.py` + `m5_gearbox_diag.py`.
- Perception and telemetry: `perception.py` scene/ray/visual fusion, `vision/tracking.py` target tracking, `visionview.py` front view overlay, `control/handover.py` human-machine handover, `telemetry.py` real-time telemetry, `m5_watchdog_beat_test.py` heartbeat test; lane debug run55-57 for lane state troubleshooting.

### 2026-08-14

- Dual runtime: `beamng_autopilot_tech/providers.py` lazily creates Tech `Camera` / `Lidar`, `launch_game.py` / `runtime.py` / `download_beamng_tech.py` / `bridge.py` and `BEAMNG_RUNTIME` series environment variables; same day completed Steam/Tech multiple rounds of e2e verification.
- M5 integration: `m5_e2e_test.py` / `m5_drive_test.py` e2e/real-world testing, `m5_launcher.py` + `m5_gui_smoke.py` console interface, `traffic.py` traffic, `connector.py` extension, and `启动自动驾驶.vbs` (Start Autopilot.vbs) / `启动车道状态窗口.vbs` (Start Lane State Window.vbs).
- Lane state and local planning: `lane.py` lane geometry/state, `planner.py` local planning, `roadnet.py` road network, with supporting `m5_lane_state_probe.py` / `m5_lane_center_capture.py` / `m5_lane_state_annotate.py` / `m5_lane_state_view.py`, lane state data saved to `logs/m5_lane_state`.
- Offline regression and planner baseline: `m5_offline_validate.py` large-scale offline regression; `docs/planner_baseline_20260814.md` records Steam run 98 (median_lat=1.76, centered_ratio=0.901), and clarifies improvement direction "navigation route only determines path, not lateral reference within lane".
- M2 closing analysis: `m2_steering_signal.py` / `m2_steering_vision.py` / `m2_validate_projection.py` / `m2_visualize.py`, solidifying steering signal/visual correlation conclusions into re-runnable scripts.
- Engineering implementation: 14:37 initialized git repository with `AGENTS.md` / `.gitignore` / README first version, 23:42 first commit `9702e71` (full repository snapshot, 107 files, 28981 lines).

### 2026-08-15

- `00:06 68cbfc0`: environment self-check `m5_env_check.py` and frame capture performance probe `bench_grab_screen.py`, README usage updated.
- `00:41 cf973d3`: `CameraModel.camera_pose` supports vehicle 6DOF pose (pitch/roll participate in back-projection, BeamNG quaternion convention verified with real state); Tech camera changed to calibrated extrinsics + pose driven, removed per-frame GE query; `m5_lane_state_view.py` frame capture failure degraded.
- `00:58 2481774`: `m5_lane_truth_probe.py` compares classic CV with BeamNG.tech pixel ground truth; italy AI 80 frames measured pavement IoU 0.734, marking recall 0.015, boundary error 0.8-1.0m, quantitatively confirming traditional CV unusable on rendered frames.
- `01:01 onward (currently uncommitted)`: learned segmentation route. Added `vision/segmentation.py` (lightweight UNet, ~1.3M parameters, background/asphalt/line 3 classes), `m5_collect_seg.py` (Tech colour + annotation half-resolution collection), `m5_train_seg.py` (temporal 80/20, median frequency weighting, mIoU monitoring); `lanes.py` extracted shared `_mask_to_markings` pipeline, `lane_overlay.py`'s `estimate_pavement_edges` supports learned off-road mask; `m5_lane_truth_probe.py --model` and `m5_autopilot.py --seg-model` integrated, falls back to classic CV when no model available.