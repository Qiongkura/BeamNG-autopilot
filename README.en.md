# BeamNG-autopilot

<div align="center">

[中文](README.md) | **English**

[![Star](https://img.shields.io/github/stars/Qiongkura/BeamNG-autopilot.svg)](https://github.com/Qiongkura/BeamNG-autopilot/stargazers)
[![Issues](https://img.shields.io/github/issues/Qiongkura/BeamNG-autopilot.svg)](https://github.com/Qiongkura/BeamNG-autopilot/issues)

</div>

An autonomous driving research project based on BeamNG.drive + BeamNGpy,采用分层架构（感知/决策/控制），逐步演进到端到端模仿学习与图像强化学习。

- **Layered Architecture**：Clear perception, decision, and control modules for research and extension；
- **Dual Runtime Support**：Compatible with Steam and BeamNG.tech, with automatic detection and adaptation；
- **Visual Perception and Obstacle Avoidance**：Integrated YOLO real-time object detection and BEV ground back-projection for visual obstacle avoidance。

## Features

| Feature | Description |
| --- | --- |
| M1 Closed-loop Tracking | Pure Pursuit + PID control for automatic vehicle following of pre-recorded trajectories |
| M2 Visual Perception | Camera calibration, tire mark strip detection, path projection for steering features |
| M3 End-to-end Imitation Learning | DAVE-2 style CNN model predicting steering angles directly from single frames |
| M4 Decision Layer | DQN discrete actions (cruise/decelerate/lane change/overtake) using Stable-Baselines3 |
| M5 In-game Autopilot Assistant | Hotkey activation for automatic driving along game navigation routes with real-time HUD |
| M5 Visual Obstacle Avoidance | YOLOv8n front-view detection + ground back-projection, fused with scene/ray obstacles for avoidance |
| Learned Pavement Segmentation | Lightweight UNet model distinguishing background/pavement/markings, replacing traditional CV thresholds |
| Real-time Telemetry & Visualization | Telemetry HUD, dashboard, decision visualization, bird's eye view, etc. |

## Architecture Design

The project adopts a layered modular design:

- **Perception Layer** (`beamng_autopilot/perception.py`, `vision/`)：Fuses scene, ray, and visual obstacle data for environmental perception。
- **Decision Layer** (`beamng_autopilot/planner.py`)：Local path planning and obstacle avoidance decisions。
- **Control Layer** (`beamng_autopilot/control/`)：Pure Pursuit path tracking, PID speed control, gearbox management, human-machine handover。
- **Runtime** (`beamng_autopilot/runtime.py`)：Steam/Tech dual runtime adaptation with lazy import of Tech-exclusive features。
- **Visualization** (`beamng_autopilot/hud.py`, `telemetry.py`, `visionview.py`)：Real-time telemetry and perception overlay。

## 📦 Environment Dependencies

```bash
BeamNG.drive 0.39+ (Steam version; BeamNG.tech optional enhancement)
Python 3.10 + venv (--system-site-packages)
GPU with 6GB VRAM or higher (required for YOLO detection + HUD visualization)
```

## Install & Usage

1. **Install BeamNG.drive**：Install version 0.39+ via Steam。
2. **Create Virtual Environment**：
   ```powershell
   python -m venv --system-site-packages .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. **Environment Self-check** (recommended before first use)：
   ```powershell
   .venv\Scripts\python.exe scripts\m5_env_check.py
   ```
4. **Start Game** (or via console)：
   ```powershell
   .venv\Scripts\python.exe scripts\launch_game.py --runtime steam
   ```
5. **Run Autopilot Assistant**：
   ```powershell
   # Attach to running game
   .venv\Scripts\python.exe scripts\m5_autopilot.py --attach
   # Or auto-start game and load map
   .venv\Scripts\python.exe scripts\m5_autopilot.py
   ```
6. **Use GUI Console**：Double-click `启动自动驾驶.vbs` (Start Autopilot.vbs) in project root or manually run：
   ```powershell
   .venv\Scripts\python.exe scripts\m5_launcher.py
   ```

## Usage Example

```powershell
# Smoke test: connect to game and drive straight for 3 seconds
.venv\Scripts\python.exe scripts\m1_smoke_test.py

# Record trajectory: AI drives along closed loop, saves reference trajectory
.venv\Scripts\python.exe scripts\m1_record_track.py

# Track following: Pure Pursuit drives along trajectory automatically for 2 laps
.venv\Scripts\python.exe scripts\m1_follow_track.py --track data\track_smallgrid.npz

# M3 high-frequency collection (BC training data)
.venv\Scripts\python.exe scripts\m3_collect_bc.py --track data\track_smallgrid.npz --speed 8.0 --laps 3

# M3 training
.venv\Scripts\python.exe scripts\m3_train_bc.py --runs logs\m3_bc\20260811_* --epochs 60 --out logs\m3_bc\bc_steer

# M3 real-world inference (BC driving)
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --speed 8.0 --duration 180

# M5 visual probe (verify detection effect)
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --show
```

## Configuration

| Key | Description | Default |
| --- | --- | --- |
| `BEAMNG_HOME` | BeamNG.drive installation directory (Steam version) | Auto-detected |
| `BEAMNG_USER` | BeamNG.drive user data directory | `%LOCALAPPDATA%\BeamNG.drive\<version>` |
| `BEAMNG_RUNTIME` | Runtime selection: `auto`/`steam`/`tech` | `auto` |
| `BEAMNG_TECH_HOME` | BeamNG.tech installation directory | None (manual setup required) |
| `BEAMNG_TECH_USER` | BeamNG.tech user data directory | `...\BeamNG.drive\0.38` |
| `BEAMNG_PORT` | Connection port | `64256` |
| `--speed` | Cruise speed (m/s) | `10` |
| `--vision-conf` | YOLO confidence threshold | `0.35` |
| `--vision-rate` | Visual scanning frequency (Hz) | `3` |
| `--seg-model` | Semantic segmentation model path | Auto-loads best model |

## Testing

The project provides various testing and validation scripts:

- **Environment Self-check**：`.venv\Scripts\python.exe scripts\m5_env_check.py`
- **Offline Regression Validation**：`.venv\Scripts\python.exe scripts\m5_offline_validate.py`
- **End-to-end Testing** (requires running game)：`.venv\Scripts\python.exe scripts\m5_e2e_test.py --attach`
- **Real-world Driving Testing** (requires running game)：`.venv\Scripts\python.exe scripts\m5_drive_test.py --speed 6 --run 10`
- **GUI Smoke Test**：`.venv\Scripts\python.exe scripts\m5_gui_smoke.py`
- **Various Diagnostic Probes**：`diag_*.py`, `m5_*_probe.py` scripts for specific function verification

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/xxx`)
3. Commit your changes (`git commit -m 'feat: add xxx'`)
4. Push to the branch (`git push origin feature/xxx`)
5. Open a Pull Request

## License

This project is licensed under the [MIT](LICENSE) license.

## Contact

- GitHub: https://github.com/Qiongkura
- WeChat: Qiongkura

## Known Limitations

- Low frame rate (<10fps) limits end-to-end imitation learning model training effectiveness, requiring high-frequency data collection；
- Traditional CV color thresholds are almost ineffective on BeamNG.tech rendered frames, requiring learned segmentation；
- Pure vision path tracking without priors is unreliable in complex curves, often locking onto shadows or dark features；
- End-to-end imitation learning (M3) is still in progress, current PoC model degenerates to constant prediction；
- Decision layer (M4) DQN training not yet completed, offline validation based on rule-based baseline；
- Visual detection relies on real-time game window visuals, degrades to no-visual mode when window is minimized or obscured。

## Related Projects

- [BeamNG.drive](https://beamng.com/)：Physics simulation driving game platform
- [BeamNGpy](https://github.com/BeamNG/BeamNGpy)：Python API for BeamNG.drive
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)：Real-time object detection framework
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)：Reinforcement learning algorithm library
- [DAVE-2](https://arxiv.org/abs/1604.07316)：End-to-end autonomous driving neural network architecture