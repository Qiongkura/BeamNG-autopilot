# BeamNG-autopilot（BeamNG自动驾驶）

<div align="center">

**中文** | [English](README.en.md)

[![Star](https://img.shields.io/github/stars/Qiongkura/BeamNG-autopilot.svg)](https://github.com/Qiongkura/BeamNG-autopilot/stargazers)
[![Issues](https://img.shields.io/github/issues/Qiongkura/BeamNG-autopilot.svg)](https://github.com/Qiongkura/BeamNG-autopilot/issues)

</div>

基于 BeamNG.drive + BeamNGpy 的自动驾驶研究项目，采用分层架构（感知/决策/控制），逐步演进到端到端模仿学习与图像强化学习。

- **分层架构**：清晰的感知、决策、控制模块，便于研究与扩展；
- **双运行时支持**：兼容 Steam 与 BeamNG.tech，自动检测并适配不同版本；
- **视觉感知与避障**：集成 YOLO 实时目标检测与 BEV 地面反投影，实现视觉避障。

## 功能

| 功能 | 说明 |
| --- | --- |
| M1 循迹闭环 | Pure Pursuit + PID 控制，车辆沿预录轨迹自动行驶 |
| M2 视觉感知 | 相机标定、轮胎印条带检测、路径投影，提供转向特征 |
| M3 端到端模仿学习 | DAVE-2 风格 CNN 模型，单帧图像直接预测转向角 |
| M4 决策层 | DQN 离散动作（巡航/减速/变道/超车），基于 Stable-Baselines3 |
| M5 游戏内自动驾驶助手 | 通过热键激活，沿游戏内置导航路线自动行驶，带实时 HUD |
| M5 视觉避障 | YOLOv8n 前视检测 + 地面反投影，与场景/射线障碍融合绕行 |
| 学习式路面分割 | 轻量 UNet 模型，区分背景/路面/标线，替代传统 CV 阈值方法 |
| 实时遥测与可视化 | 遥测 HUD、仪表盘、决策层可视化、鸟瞰图等 |

## 架构设计

项目采用分层模块化设计：

- **感知层**（`beamng_autopilot/perception.py`、`vision/`）：融合场景、射线、视觉障碍数据，提供环境感知。
- **决策层**（`beamng_autopilot/planner.py`）：局部路径规划与避障决策。
- **控制层**（`beamng_autopilot/control/`）：Pure Pursuit 路径跟踪、PID 速度控制、档位管理、人机交接。
- **运行时**（`beamng_autopilot/runtime.py`）：Steam/Tech 双运行时适配，惰性导入 Tech 专属功能。
- **可视化**（`beamng_autopilot/hud.py`、`telemetry.py`、`visionview.py`）：实时遥测与感知叠加。

## 📦 环境依赖

```bash
BeamNG.drive 0.39+（Steam 版；BeamNG.tech 可选增强）
Python 3.10 + venv（--system-site-packages）
GPU 显存 6GB 以上（YOLO 检测 + HUD 可视化需要）
```

## 安装与使用

1. **安装 BeamNG.drive**：通过 Steam 安装 0.39+ 版本。
2. **创建虚拟环境**：
   ```powershell
   python -m venv --system-site-packages .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. **环境自检**（首次使用前建议运行）：
   ```powershell
   .venv\Scripts\python.exe scripts\m5_env_check.py
   ```
4. **启动游戏**（或通过控制台启动）：
   ```powershell
   .venv\Scripts\python.exe scripts\launch_game.py --runtime steam
   ```
5. **运行自动驾驶助手**：
   ```powershell
   # 附着到已运行的游戏
   .venv\Scripts\python.exe scripts\m5_autopilot.py --attach
   # 或自动启动游戏并加载地图
   .venv\Scripts\python.exe scripts\m5_autopilot.py
   ```
6. **使用图形控制台**：双击项目根目录的 `启动自动驾驶.vbs` 或手动运行：
   ```powershell
   .venv\Scripts\python.exe scripts\m5_launcher.py
   ```

## 📝 使用示例

```powershell
# 冒烟测试：连接游戏直行 3 秒
.venv\Scripts\python.exe scripts\m1_smoke_test.py

# 录制轨迹：AI 沿闭环跑，保存参考轨迹
.venv\Scripts\python.exe scripts\m1_record_track.py

# 循迹：Pure Pursuit 沿轨迹自动跑 2 圈
.venv\Scripts\python.exe scripts\m1_follow_track.py --track data\track_smallgrid.npz

# M3 高频采集（BC 训练数据）
.venv\Scripts\python.exe scripts\m3_collect_bc.py --track data\track_smallgrid.npz --speed 8.0 --laps 3

# M3 训练
.venv\Scripts\python.exe scripts\m3_train_bc.py --runs logs\m3_bc\20260811_* --epochs 60 --out logs\m3_bc\bc_steer

# M3 实车推理（BC 驾驶）
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --speed 8.0 --duration 180

# M5 视觉探针（验证检测效果）
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --show
```

## ⚙️ 配置说明

| 配置项 | 说明 | 默认 |
| --- | --- | --- |
| `BEAMNG_HOME` | BeamNG.drive 安装目录（Steam 版） | 自动检测 |
| `BEAMNG_USER` | BeamNG.drive 用户数据目录 | `%LOCALAPPDATA%\BeamNG.drive\<版本>` |
| `BEAMNG_RUNTIME` | 运行时选择：`auto`/`steam`/`tech` | `auto` |
| `BEAMNG_TECH_HOME` | BeamNG.tech 安装目录 | 无（需手动设置） |
| `BEAMNG_TECH_USER` | BeamNG.tech 用户数据目录 | `...\BeamNG.drive\0.38` |
| `BEAMNG_PORT` | 连接端口 | `64256` |
| `--speed` | 巡航速度（m/s） | `10` |
| `--vision-conf` | YOLO 置信度阈值 | `0.35` |
| `--vision-rate` | 视觉扫描频率（Hz） | `3` |
| `--seg-model` | 语义分割模型路径 | 自动加载最佳模型 |

## 🧪 测试

项目提供多种测试与验证脚本：

- **环境自检**：`.venv\Scripts\python.exe scripts\m5_env_check.py`
- **离线回归验证**：`.venv\Scripts\python.exe scripts\m5_offline_validate.py`
- **端到端测试**（需游戏运行）：`.venv\Scripts\python.exe scripts\m5_e2e_test.py --attach`
- **实车驾驶测试**（需游戏运行）：`.venv\Scripts\python.exe scripts\m5_drive_test.py --speed 6 --run 10`
- **GUI 冒烟测试**：`.venv\Scripts\python.exe scripts\m5_gui_smoke.py`
- **各类诊断探针**：`diag_*.py`、`m5_*_probe.py` 等脚本用于特定功能验证

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/xxx`)
3. 提交你的修改 (`git commit -m 'feat: 新增xxx功能'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 打开 Pull Request

## 📄 许可证

本项目未指定许可证。请在使用前联系项目维护者确认许可条款。

## 📮 联系方式

- GitHub：https://github.com/Qiongkura
- 微信：Qiongkura

## 已知限制

- 低帧率（<10fps）下端到端模仿学习模型训练效果有限，需要高频采集数据；
- 传统 CV 颜色阈值在 BeamNG.tech 真渲染帧上几乎失效，必须使用学习式分割；
- 纯视觉无先验的路径跟踪在复杂弯道中不可靠，易锁定阴影或深色特征；
- 端到端模仿学习（M3）仍在进行中，当前 PoC 模型退化为常数预测；
- 决策层（M4）DQN 训练尚未完成，离线验证基于规则基线；
- 视觉检测依赖游戏窗口实时画面，窗口最小化或遮挡时会降级为无视觉模式。

## 与相关项目的关系

- [BeamNG.drive](https://beamng.com/)：物理仿真驾驶游戏平台
- [BeamNGpy](https://github.com/BeamNG/BeamNGpy)：BeamNG.drive 的 Python API
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)：实时目标检测框架
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)：强化学习算法库
- [DAVE-2](https://arxiv.org/abs/1604.07316)：端到端自动驾驶神经网络架构