# BeamNG-autopilot（BeamNG自动驾驶）

<div align="center">

**中文** | [English](README.en.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
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
| **M6 FSD 结构栈** | 参考特斯拉 FSD（AI Day 2021/2022）：环视相机环 → 多任务 HydraNet 感知头 → BEV/占用向量空间 → 分层规划 → 安全监控与仲裁 → 影子录制 → 端到端骨架（详见下文「FSD 结构栈」） |

## FSD 结构栈（参考特斯拉全自动驾驶 FSD 架构）

数据采集不急于规模化，重点是把**结构和代码对齐特斯拉 FSD** 的分层数据流。
方向严格参照 FSD（AI Day 2021/2022）公开架构：**环视多相机 → 共享视觉骨干 →
BEV/向量空间 → 占用 → 规划 → 安全 → 影子数据闭环**。现有规则驾驶（M5，
94.6% 山地路线完成度）始终作为执行层保留，FSD 栈以并行包落地、逐步切换。

```
相机环(8) ─► HydraNet 多任务头 ─► BEV 占用 / 向量空间 ─► 分层规划
   ▲              │                      │                    │
   │         语义/目标/交通灯/            │             轨迹+速度剖面
   │         车道拓扑(共享帧)             │                    │
   └──── 时序融合 + 目标追踪 ◄────────────┘             安全监控+仲裁
                                                              │
                        影子录制 ◄── 真实驾驶闭环 ◄────────────┘
```

### 1) 环视多相机（`beamng_autopilot/vision/ring.py`）

- 8 个相机挂载，对齐 FSD 相机布局：前主 / 前窄(长焦) / 前鱼眼(侧盲区) /
  B 柱左·右 / 后视镜左·右 / 后尾。
- 每个挂载 = 固定外参标定 + 内参（`CameraModel`），运行时只喂整车 6-DOF 位姿。
- Tech 侧 `beamng_autopilot_tech.providers.TechCameraRingProvider` 在
  BeamNG.tech 上创建 8 路 beamngpy `Camera` 传感器（含 GPU prepass 预热与
  黑帧重试）；Steam 回退为单前视。

### 2) HydraNet 多任务感知头（`beamng_autopilot/vision/hydra.py`、`heads/`）

单个共享帧分发给多个任务头，每个头输出统一 `TaskOutput`，失败头被隔离：
- 语义头：学习式 UNet → 路面/标线掩码 + 车道路面落地（`SemanticHead`）
- 目标头：YOLO → 世界坐标系障碍 + 像素框（`ObjectHead`）
- 交通灯头：HSV 红/黄/绿检测，紧凑度 + 单色优势门控，免疫琥珀色大范围偏色
  （`TrafficSignalHead`）
- 车道拓扑头：左/右邻车道存在性与可穿越性（实线/护墙=不可并线）（`LaneTopologyHead`）

### 3) BEV 占用 / 向量空间（`occupancy.py`、`bev_fusion.py`、`temporal.py`）

- `OccupancyGrid`：车身系网格，每个 cell 含占用/可行驶/障碍/高度/来源计数；
  相机掩码经地面反投影（`project_road_mask_to_grid`）写入可行驶，LiDAR/射线
  灌入障碍；`query_path_cost` 供轨迹打分。
- `BEVFeatureMap`：FSD 向量空间——各路相机反投影证据经 log-odds 融合成
  obstacle/drivable/lane/sign 语义通道，预留 attention 权重接口（未来学习式
  交叉注意力）。
- 时序融合：`TemporalOccupancyFilter`（EMA，正向证据增高、空帧只衰减）让
  单帧 LiDAR 抖动不产生幻影墙、也不抹掉真实墙；`WorldObjectTracker` 跨帧
  匹配目标并估计速度。

### 4) 分层规划（`beamng_autopilot/planning/`）

- `scene.py`：规划读取的统一场景快照（占用 + 车道/路线参考 + 规则）。
- `trajectory.py`：候选轨迹扇（单车模型弧线 + 车道偏移），物理曲率上界，
  不走回头路。
- `constraints.py`：碰撞 / 曲率 / 车道对齐成本；`corridor_free_band` 空间
  连通门控——仅当前方横向带真被连续堵死才判无路（散射路边柱不再误杀候选）。
- `speed_profile.py`：沿轨迹每点最大速度（弯道/障碍刹车带/巡航上限），
  轨迹与速度一起规划——补上纵向规划。
- `selector.py`：best-of-N 择优并附带速度剖面。
- `intent.py`：从导航路线推断路口意图（直行/左转/右转/U 型）与建议车速。
- `arbiter.py`：FSD 轨迹 vs 规则兜底仲裁——FSD 无路时降级到规则路径慢开，
  不再瞬态死停。

### 5) 路网导航路线（`roadnet.py`、`fsd_stack.py`）

- `RoadNetwork`：查询 DecalRoad 中心线构建路网图；**交叉路口缝合**
  （近邻节点建边）把碎片化路网连通，A* 才能跨路口导航（此前 start/goal
  往往落在不同连通分量，A* 直接返回 None）。
- `m5_fsd_drive.py` 优先用 road-graph A* 生成沿路线（并 snap 到最近路网
  节点），不再用单点 `core_groundMarkers.setPath` 的**直线插值**——那种
  直线会横穿路面/车辆/建筑导致逆行、压线与撞墙。
- `choose_plan_route`：当 map route 前方被占用而感知车道（BEV 可行驶
  中线）前方通畅时，计划沿感知车道走——避免「路线直穿墙还把车往墙上推」。

### 5) 安全监控与仲裁（`safety_monitor.py`、`control/reverse_guard.py`）

- `SafetyMonitor`：影子健康 + 最小风险策略——对比轨迹与占用、检查车道保持、
  监测传感器/规划时效，输出 Safe / Degraded / MinimalRisk 与目标车速；
  尊重走廊连通门控。
- `ReverseGuard`：FSD 驾驶永不倒车——沿车头前进速度分量为负即刹车，
  滞后释放（撞墙弹回后防止「无脑倒车」）。

### 6) 真实驾驶闭环（`fsd_stack.py`、`scripts/m5_fsd_drive.py`）

- `FSDStack`：一条可调用完整管线——相机环 → HydraNet → 时序占用 → 分层
  规划（带速度剖面）→ 控制提示；teleport 后 `reset_temporal()` 防止旧占用
  泄漏成幻影墙；BEV 可行驶空间中线作为车道参考（相机车道不可用回退导航路线）。
- `m5_fsd_drive.py`：实车 FSD 模式驾驶——FSDStack 规划 → 安全监控仲裁 →
  PurePursuit 转向 + SpeedController 纵向，仲裁含规则兜底与反向防护；
  兜底路径强制使用世界坐标（避免车身系参考把车甩进墙）。

### 7) 影子录制 + 端到端骨架（`recording.py`、`neural/`）

- `ShadowRecorder` / `EpisodeDataset`：每次驾驶录制（真值控制、影子预测、
  BEV 栅格、轨迹）对齐成 .npz 数据，产出 `(bev_raster, action)` 训练对。
- `E2ENet`：FSD v12 形状的端到端骨架——BEV → 轨迹 + 动作，前向/反向/
  合成训练闭环均已打通，**暂不训练真实数据**（数据采集后续再做）。

### 运行 FSD 栈

```powershell
# 1) 环视相机会采（需 Tech，游戏运行中）
.venv\Scripts\python.exe scripts\m5_ring_probe.py --runtime tech --attach

# 2) 单次完整感知+规划 tick（HydraNet → BEV → planner）
.venv\Scripts\python.exe scripts\m5_fsd_stack_probe.py --runtime tech --attach

# 3) 分层规划候选评分可视化
.venv\Scripts\python.exe scripts\m5_planning_probe.py --runtime tech --attach

# 4) 真实驾驶（FSD 模式，带安全监控 + 仲裁 + 反向防护）
.venv\Scripts\python.exe scripts\m5_fsd_drive.py --runtime tech --attach `
    --seconds 30 --speed 6 --teleport 729.6 763.9 45

# 5) 影子录制一集（规则开 + FSD 栈影子预测）
.venv\Scripts\python.exe scripts\m5_shadow_drive.py --runtime tech `
    --attach --seconds 15 --drive
```

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

抓帧性能对比探针（可选，需游戏窗口位于主屏；dxcam 需自行 `pip install dxcam`）：

```powershell
.venv\Scripts\python.exe scripts\bench_grab_screen.py
```

## 学习式路面/标线分割（M5-B）

传统 CV 颜色阈值在 Tech 真渲染帧上失效（实测标线 recall 1.5%、边界误差
0.8-1.0m）。替代方案：用 BeamNG.tech 的 annotation 像素真值训练轻量 UNet
（背景 / 路面 / 标线 3 类），推理只吃 RGB 帧，Steam / Tech 通用。

```powershell
# 1. 采集训练数据（Tech 实例运行中，AI 沿路行驶自动采集）
.venv\Scripts\python.exe scripts\m5_collect_seg.py --frames 500

# 2. 训练（可多个 run 一起训；增强 + 类别加权已内置）
.venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_* `
    --epochs 60 --lr 5e-4 --out logs\m5_seg\seg_model

# 3. 离线评估（无需游戏，直接验证模型）
.venv\Scripts\python.exe scripts\m5_eval_seg.py --runs logs\m5_seg\run_* `
    --model logs\m5_seg\seg_model\best.pt --save

# 4. 实车真值对比（CV vs 学习式，需 Tech 实例）
.venv\Scripts\python.exe scripts\m5_lane_truth_probe.py --frames 60 --drive
.venv\Scripts\python.exe scripts\m5_lane_truth_probe.py --frames 60 --drive `
    --model logs\m5_seg\seg_model\best.pt
```

模型就绪后 `m5_autopilot.py` 和 `m5_lane_state_view.py` 会自动加载
`logs/m5_seg/seg_model/best.pt`（或用 `--seg-model <路径>` 指定），
加载失败自动回退经典 CV。

## 双运行时（Steam / Tech）可并行开发

**开发前提（2026-08 起）：优先基于 BeamNG.tech 研发**。感知以真传感器
（Camera / LiDAR）与 annotation 像素真值为准，学习数据从 Tech 采集；
Steam 兼容路径（窗口截屏、Lua 射线、经典 CV 回退、YOLO 2D 反投影）只
保底不坏、不再投入，统一留给后期下放适配。下面的双运行时机制保持不变，
仅作为运行选择使用。

核心代码库保持 Steam 版兼容、可开源；BeamNG.tech 专属能力放在独立包
`beamng_autopilot_tech/`，只在检测到 Tech 时惰性导入，普通玩家直接使用
开源仓库不会受到影响。

- `BEAMNG_RUNTIME=auto|steam|tech`，默认 `auto`：连接后根据 BeamNGpy
  的 `tech_enabled()` 自动选择。
- `BEAMNG_TECH_HOME` 指向 Tech 安装目录；Steam 仍使用 `BEAMNG_HOME`。
- `BEAMNG_TECH_USER` 指向 Tech 用户目录，默认 `...\BeamNG.drive\0.38`，
  与 Steam 的 `0.39` 分开，避免两个版本互相污染存档/设置。
- `BEAMNG_PORT` 可选，默认 `64256`；Tech 固定用 `64257`
  （`BEAMNG_TECH_PORT` 可覆盖）。**端口与运行时绑定**：Steam 始终
  `64256`、Tech 始终 `64257`，两套实例可并行，启动器/助手/探针按所选
  运行时自动对应端口（`config.runtime_port(mode)`），无需手动改端口。
- 启动/主流程入口（`launch_game.py` / `m5_launcher.py` /
  `m5_autopilot.py` / `m5_e2e_test.py` / `m5_drive_test.py`）、M1-M3
  采集标定脚本（`m1_smoke_test.py` / `m1_record_track.py` /
  `m1_follow_track.py` / `m1_calibrate_heading.py` / `m2_capture.py` /
  `m2_calibrate_camera.py` / `m3_collect_bc.py` / `m3_drive_bc.py`）以及
  M5 探针/验证脚本（`m5_nav_route_test.py` / `m5_obstacle_test.py` /
  `m5_twitch_verify.py` / `m5_vision_probe.py` / `m5_loop_probe.py`）均
  支持 `--runtime auto|steam|tech`；启动器界面左侧设置区也有同一个
  “运行时”下拉框。启动前 `auto` 优先选 Tech（本机存在 Tech 安装目录
  时），否则退回 Steam。
- 运行状态/控制/感知探针（`m5_probe_state.py` / `m5_probe_elec.py` /
  `m5_ctrl_probe.py` / `m5_handover_test.py` / `m5_emergency_stop.py` /
  `m5_steer_sign_probe.py` / `m5_perception_probe.py` /
  `m5_route_scan_probe.py`）同样支持 `--runtime auto|steam|tech`。
- Steam 路径：现有窗口截屏 + Lua 射线/场景/车辆感知，不需要 Tech 授权。
- Tech 路径：`beamng_autopilot_tech.providers` 惰性创建 beamngpy
  `Camera` / `Lidar` 传感器。
- `diag_*`、`m5_wall_*`、`m5_live_wall_probe*.py`、
  `m5_live_route_probe.py` 等纯诊断探针保持 Steam 语义，不走双运行时
  分派；它们只用于 Steam 版排障。
- 开源发布时可整体不包含 `beamng_autopilot_tech/`；核心模块不在顶层
  导入它。

```powershell
.venv\Scripts\python.exe scripts\launch_game.py --runtime steam
.venv\Scripts\python.exe scripts\launch_game.py --runtime tech
.venv\Scripts\python.exe scripts\m5_autopilot.py --runtime auto --attach
```

## M2 结论（视觉感知）

- 相机方案：早期 Steam 版未激活 tech 时只能用 `set_relative_camera` + PrintWindow 截屏（1064x772）。现在 BeamNG.tech 授权已生效（tech 通道 `127.0.0.1:64256` 的 TechCom / TechVE 均正常），后续可逐步换成 beamngpy 真传感器（Camera / LiDAR / IMU）。
- 地图先验下，轮胎印条带检测保真度很高：4m 处 r=+0.93（平均误差 0.21m），6m r=+0.90（0.25m），12m r=+0.91（0.75m）。
- 横向偏移差 drift(12m−4m) 与转向输入相关 r=+0.79，说明"远处点偏左/偏右"确实是有效转向特征。
- 但纯视觉无先验（中心播种 + 时间追踪）在 2.4fps 低帧率下不可靠：lat12 与转向相关 r≈−0.33，弯道中会锁到阴影/其他深色特征。
- 结论：手工特征上限低，且依赖精确标定 → 转向 M3 神经网络端到端模仿学习。

## M3 模仿学习（BC）

- 采集：`scripts/m3_collect_bc.py` 高频截屏 + 专家（Pure Pursuit）转向/油门/刹车标签；帧先于指令抓取，保证图像与标签对齐。
- 训练：`scripts/m3_train_bc.py` DAVE-2 CNN（200x66 RGB -> steer），时间序 80/20 划分，监控 val MAE / R²，自动保存最优模型与曲线。
- 推理：`scripts/m3_drive_bc.py` 加载模型实车闭环驾驶，PID 维持目标速度。
- PoC：用 M2 的 229 帧（2.4fps）跑通全流程，val MAE=0.048，但模型退化为常数预测（R²≈0）——帧率太低、转向标签几乎全是 0，必须用 M3 高频采集（目标 ≥10fps、多弯道、多圈）才有训练价值。

## 运行

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

本项目采用 [MIT](LICENSE) 许可证。

## 📮 联系方式

- GitHub：https://github.com/Qiongkura
- 微信：Qiongkura

## 已知问题与限制

### FSD 结构栈现状

| 问题 | 影响 | 优先级 | 状态 |
| --- | --- | --- | --- |
| **FSD 栈默认不驱动、仅供验证** | `m5_fsd_drive` 是独立验证入口，`AutopilotSession` 仍走 M5 规则驾驶 | 中 | FSD 栈与旧路径并行，尚未切换为主驾驶 |
| **多任务头是「结构对齐」非「共训练」** | 各 head 仍调用原有独立算法，未真正共享一个训练过的骨干 | 中 | 目标仅对齐 HydraNet 数据流形状 |
| **FSD 模式在多变场景偏向保守** | 十字路口密集 LiDAR 时可能降速/回退规则路径 | 中 | 已有走廊连通门控 + 仲裁兜底，仍待长测 |
| **端到端骨架未训练** | `E2ENet` 仅保证前向/反向/合成 loss 可降 | 低 | 数据采集后续再做 |

### 架构层面

| 问题 | 影响 | 优先级 | 状态 |
| --- | --- | --- | --- |
| **`planner.py` 单文件 2790 行** | 可维护性极差 | 高 | ✅ 已拆分为 `planner/` 包（6 个子模块） |
| **`lane.py` 1500 行** | 维护困难 | 高 | ✅ 已拆分为 `lane/` 包（6 个子模块） |
| **`m5_autopilot.py` 主循环 2516 行** | 脚本不可测试、不可复用 | 高 | ✅ 已拆分为 `autopilot.py`（AutopilotSession 类）+ 薄脚本 |
| **53 处裸 `except Exception`** | 真实错误被掩盖 | 中 | ✅ 已清理（6 个模块，140 行改动） |

#### 重构方案

**`planner.py` → `planner/` 包** ✅ 已完成：

```
beamng_autopilot/planner/
├── __init__.py      # re-export LocalPlanner，保持外部 import 不变
├── constants.py     # 53 个魔数常量 + _MapLaneBoundary 类
├── geometry.py      # 18 个纯数学函数（曲率、投影、距离）
├── obstacles.py     # 14 个障碍物几何/分类/碰撞检测函数
├── solid.py         # 5 个实线检测/噪声过滤/禁止穿越函数
└── core.py          # LocalPlanner 类 + creep_speed
```

**`lane.py` → `lane/` 包** ✅ 已完成：

```
beamng_autopilot/lane/
├── __init__.py      # re-export pair_lane_markings, LaneTracker 等
├── constants.py     # 55 个车道追踪常量
├── pairing.py       # 19 个视觉标线配对函数
├── lidar.py         # LiDAR 射线走廊估算
├── fusion.py        # 视觉 + LiDAR 传感器融合
└── tracking.py      # LaneTracker 类 + 帧间跟踪/稳定性检查
```

**`m5_autopilot.py` → 薄脚本 + 库** ✅ 已完成：

```
beamng_autopilot/
├── autopilot.py     # AutopilotSession 类：主循环、状态机、遥测（从脚本提取）
├── hud.py           # 已有，保持不变
└── hotkeys.py       # 已有，保持不变

scripts/
└── m5_autopilot.py  # 薄入口：解析参数、创建 AutopilotSession、run()
```

脚本只保留 ~100 行的参数解析和组装逻辑，核心循环移入 `beamng_autopilot/autopilot.py` 作为可测试的类。

**执行节奏**：建议按 planner → lane → autopilot 的顺序分三个 PR 逐步拆分，每个 PR 独立可回滚。

### 感知与视觉

| 问题 | 影响 | 优先级 |
| --- | --- | --- |
| **传统 CV 阈值在 Tech 真渲染帧上几乎失效**：标线 recall 仅 1.5%，边界误差 0.8-1.0m | 没有学习式分割模型时，Steam 回退路径的车道检测基本不可用 | 已缓解（有 UNet 回退） |
| **学习式分割模型泛化能力未验证**：当前 UNet 仅在 italy/smallgrid 两个地图上训练 | 换地图可能需要重新采集训练数据 | 中 |
| **视觉检测依赖游戏窗口实时画面**：窗口最小化或被遮挡时降级为无视觉模式 | 多任务切换时自动驾驶功能失效 | 中 |
| **LiDAR 聚类有 117-139ms 延迟**：360 点云的体素降采样 + 聚类管线耗时较长 | 高速场景下障碍物更新可能滞后 | 低 |

### 决策与规划

| 问题 | 影响 | 优先级 |
| --- | --- | --- |
| **超车决策基于规则状态机**：`OvertakeStateMachine` 的阈值（1.5s 持续、0.4s 确认）是硬编码的 | 不同道路场景下超车行为可能过于保守或过于激进 | 中 |
| **A* 路径规划网格分辨率固定 0.5m**：`GRID_RES = 0.5`，网格范围 55m × 40m | 密集场景下计算量大；稀疏场景下精度浪费 | 低 |
| **实线检测的噪声过滤参数多且脆弱**：`SOLID_*` 系列有 12 个常量，用于过滤误检 | 换地图或光照变化后可能需要重新调参 | 中 |
| **PurePursuit `find_target` 未使用 `adaptive_lookahead`**：`adaptive_lookahead(speed)` 方法存在但 `find_target` 仍用固定 `self.lookahead` | 高速时预瞄距离不足，S 弯/缓弯中转向指令可能剧烈抖动 | 中 |

### 模仿学习（M3）

| 问题 | 影响 | 优先级 |
| --- | --- | --- |
| **低帧率（<10fps）下训练效果有限**：早期 PoC 仅 2.4fps、229 帧，模型退化为常数预测 | 需要高频采集（≥10fps）+ 多弯道多圈数据才有训练价值 | 已缓解（Tech 采集可达 7fps） |
| **单地图训练的模型泛化差**：`bc_tech_smallgrid.pt` 仅在 smallgrid 上验证 | 换地图或换车型后模型可能失效 | 中 |
| **缺乏闭环评估指标**：目前只有 val MAE / R²，没有实际驾驶中的横向偏差、碰撞率等指标 | 无法量化模型在真实驾驶中的表现 | 中 |

### 决策层（M4）

| 问题 | 影响 | 优先级 |
| --- | --- | --- |
| **DQN 训练尚未完成**：离线验证基于规则基线，没有真正的 RL 策略 | M4 功能名存实亡 | 低（当前优先 M5） |

### 运行时与工程

| 问题 | 影响 | 优先级 |
| --- | --- | --- |
| **双运行时的 Tech 路径未经 CI 测试**：Tech 专属功能（annotation、LiDAR）只能在有 Tech 授权的机器上手动验证 | 回归测试覆盖不完整 | 中 |
| **缺少类型注解**：虽然使用了 `from __future__ import annotations`，但多数函数缺少参数和返回值类型标注 | IDE 补全和静态检查效果差 | 低 |
| **日志体系不统一**：部分模块用 `print`，部分用 `logging`，遥测走 JSON 文件 | 排障时信息分散，难以聚合分析 | 低 |

## 与相关项目的关系

- [BeamNG.drive](https://beamng.com/)：物理仿真驾驶游戏平台
- [BeamNGpy](https://github.com/BeamNG/BeamNGpy)：BeamNG.drive 的 Python API
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)：实时目标检测框架
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)：强化学习算法库
- [DAVE-2](https://arxiv.org/abs/1604.07316)：端到端自动驾驶神经网络架构
