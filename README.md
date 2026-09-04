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

> **FSD 底层拟真要求（硬性规定）**：车道保持/道路边界必须 100% 来自感知
> （语义分割 + LiDAR → BEV），地图/导航只负责"去哪"，感知不可用时降级停车
> 而不是骑地图中线。完整规则与代码落点见 **`docs/fsd_realism.md`**，
> 用 `--lane-mode sensor --strict` 进入 FSD 拟真驾驶模式。
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
- `local_route.py`：导航路线的车头局部窗口——按弧长取 ego 前方 40m 路线、
  禁止倒着走线；Catmull-Rom 重采样把路网折线圆角化（发卡弯不再被裁成
  直段）；`map_lane_edges` 按圆角道路中心线给出车道参考，修掉「压中线
  误判」与圆角出口尖点。
- `speed_profile.py`：沿轨迹每点最大速度——曲率公式改为真曲率
  `2*|cross|/(n1*n2*(n1+n2))`（与采样步长无关）+ 12m 前瞻刹车带 +
  弯道速度上限（`COMFORT_LAT` 收紧到 2.0 m/s²）；轨迹与速度一起规划。
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

### 5) 安全监控与仲裁（`safety_monitor.py`、`control/reverse_guard.py`、`control/reverse_maneuver.py`）

- `SafetyMonitor`：影子健康 + 最小风险策略——对比轨迹与占用、检查车道保持、
  监测传感器/规划时效，输出 Safe / Degraded / MinimalRisk 与目标车速；
  尊重走廊连通门控。
- `ReverseGuard`：FSD 驾驶永不倒车——沿车头前进速度分量为负即刹车，
  滞后释放（撞墙弹回后防止「无脑倒车」）。
- `ReverseManeuver`：与「永不倒车」互补的**受控有界倒车**——检测到前方
  无路、后方有空间时才挂 R 挡倒出死胡同，按距离/时间/后向净空限制单次
  倒车（默认最多 1.5m / 2.5s / -0.4m/s），停下后重新规划前进；溜坡失控
  （低于 -3.0m/s）立即终止整个 maneuver；纯状态机可离线单测。

### 6) 真实驾驶闭环（`fsd_stack.py`、`scripts/m5_fsd_drive.py`）

- `FSDStack`：一条可调用完整管线——相机环 → HydraNet → 时序占用 → 分层
  规划（带速度剖面）→ 控制提示；teleport 后 `reset_temporal()` 防止旧占用
  泄漏成幻影墙；BEV 可行驶空间中线作为车道参考（相机车道不可用回退导航路线）。
- `m5_fsd_drive.py`：实车 FSD 模式驾驶——FSDStack 规划 → 安全监控仲裁 →
  PurePursuit 转向 + SpeedController 纵向，仲裁含规则兜底与反向防护；
  兜底路径强制使用世界坐标（避免车身系参考把车甩进墙）。
  **确定性控制时序**：每 tick 先 `pause()` 仿真，感知+规划完成后按
  `step(45)`（0.75s 仿真突发）推进——FSD tick 要 1.4~7s 墙钟，若不停仿真
  车会用上一帧控制盲跑数秒（首帧约 9m，直接错过发卡弯）；首帧转向 dt
  从 0 改为 1.5s，首帧即可打方向。
  **弯道能力**：低速转向上限提到 0.55、曲率前馈 `near_m=1.5 / max_ff=0.40`
  （小转向时附加、满打方向时禁用）、弯前 12m 内有 <15m 弯时计划速度降为
  `sqrt(1.3*R)`、`v > plan_speed + 0.8` 时硬刹、横摆阻尼只在小转向生效；
  爬坡辅助窗口 3.5s，避免坡底停顿被误判为 stuck/倒车。

### 7) 影子录制 + 端到端骨架（`recording.py`、`neural/`）

- `ShadowRecorder` / `EpisodeDataset`：每次驾驶录制（真值控制、影子预测、
  BEV 栅格、轨迹）对齐成 .npz 数据，产出 `(bev_raster, action)` 训练对。
- `E2ENet`：FSD v12 形状的端到端骨架——BEV → 轨迹 + 动作，前向/反向/
  合成训练闭环均已打通；真实影子数据已训练 temporal 多模态 CNN（val 0.0599，
  `logs/m5_e2e/best_temporal.pt`），批量回放报告见 `logs/m5_e2e/report.json`。

### 8) 实车验证记录（FSD 模式，压缩）

| 版本 | 场景 | 关键结果 |
| --- | --- | --- |
| fix56（08-27） | 山地发卡弯 75s | 全程 safe，0 倒车/卡死，首弯干净通过（修复控制时序 + 首帧转向） |
| opt14（08-28） | 山地 290m 30s | safe，速度 3.8~6.9 m/s，未压线/未上草（修复车道门控、转向振荡、护栏、安全重生） |
| opt23/opt24（08-28） | 完整路线 + 三修复 | 0 卡死 / 0 急刹，完整到终点（plan 限幅 + corridor-open + 弯前 governor 40° 门控 + 踏板限幅） |
| opt26（08-28） | 城镇 118.6m + 山地 284m | 均 0 压线 / 0 上草 / 0 倒车（路口转向门控；城镇目标需为路上的点） |
| opt32/35/42（08-28） | 终点停车回正 | 停车车道中央（中线右 2.28m）、车头偏差 0.6°（trim_backtrack + 低速回正 + 停后归零） |
| lane-mode 对比（08-28） | map / auto / sensor | sensor 感知车道质量不足（压线 18 次）；护栏兜底有效；感知质量是 FSD 车道参考的真正瓶颈 |

### 实车验证已知注意

- LiDAR 隔帧复用已带 egomotion 运动补偿：复用框按自身位移外推，动态障碍（来车）不再滞后整帧。
- 城镇路线目标必须是 road-graph 上的点，否则 A* 尾段是跨草地的直线（上草是路线问题而非控制问题）。
- 逐轮详细日志保留在 `logs/`（`fsd_eval_*.json` / `fsd_town_*.json`），此处不再展开。

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

# 推荐（v10 起）：只保留标线密集帧 + 每个 run 各取尾部做验证。
# 全局时间尾部分割会让"拼接后最后几个稀疏 run"主导验证集、line IoU 失真；
# 标线稀疏帧（无标线路段）混进训练只会稀释 line 监督。
.venv\Scripts\python.exe scripts\m5_train_seg.py `
    --runs logs\m5_seg\run_dense_* --min-line-frac 0.003 `
    --split per-run --epochs 60 --out logs\m5_seg\seg_model

# 中断后续训（每轮自动落盘 checkpoint_last.pt，不会白训）
.venv\Scripts\python.exe scripts\m5_train_seg.py --runs logs\m5_seg\run_* `
    --epochs 60 --lr 5e-4 --out logs\m5_seg\seg_model `
    --resume logs\m5_seg\seg_model\checkpoint_last.pt

# 3. 离线评估（无需游戏，直接验证模型）
.venv\Scripts\python.exe scripts\m5_eval_seg.py --runs logs\m5_seg\run_* `
    --model logs\m5_seg\seg_model\best.pt --save
#    除总体指标外还会按 run 分组报告 line IoU / 标线像素占比，
#    一眼看出哪类路段弱、哪个 run 是劣质稀疏数据；训练占用 GPU 时
#    追加 --device cpu 避免互抢。

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

- 相机方案：早期 Steam 版未激活 tech 时只能用 `set_relative_camera` + PrintWindow 截屏（1064x772）。现在 BeamNG.tech 授权已生效，FSD 栈已用 beamngpy 真传感器（Camera / LiDAR / IMU）采集与感知（`beamng_autopilot_tech/providers.py`）。
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
| **第一发卡弯入弯偏晚（此前最大表现缺口）** | 先直行冲到弯心外侧，再靠倒车+重规划回路线 | 高 | ✅ 已解决（确定性控制时序 + 首帧转向 + 弯前 governor，fix56 实车干净通过） |
| **重生/传送可能埋到地下** | 固定高度 teleport 在较高地形把车埋进地下 | 高 | ✅ 已解决（`safe_teleport` 先垂直射线量真实地面再放置并复查，`load_scenario` 自动抬升，任意地图生效） |
| **无导航路线直行上草地** | 游戏重启后无路线，直线参考把车带到草地蠕动 | 高 | ✅ 已解决（无路线限速 1 m/s；地图真实左右边缘护栏：越界即降速，不允许在路上之外行驶） |
| **传感器车道误检带偏（假左弯）** | 车道线被路边杂物带偏 55°，生成假大弯撞墙 | 高 | ✅ 已解决（sensor lane 与导航方向差 >35° 即拒绝，回退地图车道） |
| **高速转向振荡（左右晃）** | 7~8 m/s 下控制滞后，车头 ±20° 摆动撞墙 | 高 | ✅ 已解决（车头偏离导航 >12° 自动降速收敛） |
| **坡道/撞墙后倒车滚回数米** | 反向溜坡、重复倒车 | 高 | ✅ 已解决（`ReverseManeuver` 有界倒车 + R 挡油门门控，fix56 全程 0 次倒车） |
| **FSD 栈默认不驱动、仅供验证** | `m5_fsd_drive` 是独立验证入口，`AutopilotSession` 仍走 M5 规则驾驶 | 中 | FSD 栈与旧路径并行，尚未切换为主驾驶 |
| **多任务头是「结构对齐」非「共训练」** | 各 head 仍调用原有独立算法，未真正共享一个训练过的骨干 | 中 | 目标仅对齐 HydraNet 数据流形状 |
| **路口几何误读蠕行** | 路口圆角局部半径 R~3m 触发弯前 governor，plan 锁 1.97 m/s 爬行 5s+ | 高 | ✅ 已解决（governor 改按 12m 前瞻真实累计转向 ≥40° 门控，`_route_turn_deg` 按段航向差求和，opt24 实车确认：无持续蠕行） |
| **起步/重启油门冲击** | 重规划后 0→0.8 油门一步给满，速度每帧跳 >1.5 m/s | 高 | ✅ 已解决（`rate_limit_pedal` 对最终踏板限幅 0.8-1.2/s，安全分支绕过，opt24 实车确认：跳变 13→4 帧） |
| **路口/终点 plan 速度跳变（卡死-重启）** | LiDAR 杂波使 plan 在 1.0↔6.0 逐帧跳，急刹后全油门重启，路口卡死 | 高 | ✅ 已解决（plan 限幅 + 弧长投影 + corridor-open 障碍下限，opt23 实车 0 卡死） |
| **速度节流轻微振荡** | 加速→轻刹→再加速，不影响安全但观感差 | 中 | ✅ 已缓解（目标速度 ramp + 滞回 + 下坡 governor 提前柔介入，opt14 实车 0 刹停、速度 3.8~6.9 m/s） |
| **FSD tick 慢** | 感知规划 tick 曾 1.4~7s，实车一顿一顿 | 中 | ✅ 已缓解（实时模式 ~2Hz 连续控制、无暂停卡顿；warm-up + 只抓前视 + LiDAR 每 3 tick 轮询 + 扫描范围 120→80m / 聚类半径 40→30m + 向量化 + 密度 12→8 / 点数 6000→4500，opt24 实车确认：total p90 364ms / >300ms 42 帧） |
| **长测覆盖** | 山地 284m 完整路线 + 城镇 118.6m 连续道路 | 中 | ✅ 均 0 压线 / 0 上草 / 0 倒车（opt26）；多场景长测继续 |
| **端到端已训练（影子数据）** | temporal 多模态 CNN（RGB+分割+BEV+速度 → 轨迹+动作） | 中 | ✅ val 0.0599（`best_temporal.pt`），批量回放报告 + 最差帧截图可查；实车闭环后续 |


### 离线评测与数据工具（2026-08-30）

- `m5_e2e_probe.py --data <dir> --weights <ckpt>`：批量回放评测，输出动作误差 /
  接管率 / 轨迹误差到 `logs/m5_e2e/report.json`，并把最差 Top-N 帧截图存到
  `logs/m5_e2e/worst/`（直接定位模型失效场景）。
- `m5_train_e2e.py --drop-takeover-ge 0.5 --dedup`：训练前自动剔除回放中高接管率
  的坏集；`--dedup` 跳过连续近似重复帧，缩短训练并减少过拟合。
- `m5_collect_seg.py --town`：限定城镇标线密集区采集分割真值
  （`--area-center / --area-radius` 可自定义区域）。

- 真值抽检（2026-09-01）：山区/沿海路段约一半帧没有任何标线像素（line 类
  稀疏），line IoU 0.33 的瓶颈主要是**数据构成**而非模型；城镇采集
  （`--town`，900 帧 / 60.9 万标线像素 / 677 每帧）已补上标线密集段。

- 车道侧与录制（2026-09-02/03）：`m5_fsd_drive` 起步/重启落**右车道**（不再从
  路线中线起步）；2026-09-03 起起步定位改为**纯感知**：语义头 warm 后投影漆画
  线，把车放到"标线右侧 1.5m=本车道中心"，彻底删除路线中线+偏移常量
  （`SNAP_LANE_OFFSET_M`），感知不可见才留在路面安全点。录制（`m5_shadow_drive`）
  用同样规则 + map_lane 右车道参考 + lane_center 候选，标签不再骑中线；
  `fsd_drive` 在线**漆画线横向指标**（`line_lat`，每帧投影语义标线报车在标线
  哪侧，结束汇总）。`m5_autopilot --fsd` 可把 FSD 分层规划作为转向前端（规则
  路径兜底，默认关闭）。2026-09-03 终点刹停同样改为**纯感知**：最后 20m 的
  横向参考来自漆画线投影（本车道中心），感知不可见时**不加任何地图/导航线横向
  拉拽**，只保持当前航向直行刹停；标线在终点段闪烁/消失时会短时沿用**最近一次
  感知到的本车道中心**（≤2s 且车仍在该线附近）继续横向收敛，telemetry 新增
  `end_ref`（live-perception / last-good-hold / straight-hold）可区分是哪一种。
  `m5_fsd_drive` 驾驶路径已无「导航线+偏移」残留（唯一保留的 `right_offset`
  在旧 M5 规则驾驶 `LocalPlanner` 里，作为兼容兜底，FSD 栈不使用）。
- 稳态标线修正（2026-09-03）：新增 `PaintedLineLateralCorrector`
  （`vision/lanes.py`）——行驶中每个 tick 复用语义标线反投影，把近端 12m
  转向路径按"感知本车道中心"限速拉正（死区 5cm / 限幅 1.0m / 速率 1.2m/s，
  感知掉线先 hold 2s 再衰减），遥测新增 `plc_shift/plc_desired`；全链路
  仍是纯感知横向，无导航线/地图线偏移。
- PLC 激活门控（2026-09-04）：`PaintedLineLateralCorrector` 增加策略门控
  （`vision/lanes.py::painted_line_correction_active`）——只在感知标线可信、
  当前不是规则来源兜底帧、且未进入终点停靠段的帧才介入，门控状态写入
  telemetry 并接进 `m5_fsd_drive` 的 engage 判定，避免在感知漂移/兜底帧
  里把车往错误方向拉。
- 分割训练流水线（2026-09-04）：断点续训落地（每轮自动落盘
  `checkpoint_last.pt`，`--resume` 可从中断轮次续训，不再白跑 2h+）、
  实时日志改进、`torch.amp.GradScaler` API 清理弃用告警。
- 分割模型 v8/v9 对照与 held-out 评估（2026-09-04）：v9 用 10 个 run /
  6532 帧（train 5226 / val 1306）训 40 epochs，val mIoU **0.6277**、val acc
  **0.976**；但用**完全不参与 v9 训练**的城镇/桥上两个 run 做 held-out 评估
  （925 帧）line IoU 仅 **0.110**，且 v9 自身 val line IoU（0.103）< v8
  （0.205），说明 v9 标线整体退步而非数据构成问题。**同时发现此前 v8 的
  line IoU 0.2218 是训练集泄漏**（v8 训练 run 包含这两个评估 run，数字不可信）；
  推测 v9 混入大量标线稀疏 run（line 像素占比 ~0.0006-0.0025）拉低了学习。
  部署目录 `logs/m5_seg/seg_model/best.pt` **保持 v8 不覆盖**，v9 仅作记录；
  下一步用标注密集 run 重训 v10 + 评估按标注密度分组，held-out line IoU
  达标（≥0.35）才部署。
- `m5_pipeline.py --cycles N --dedup --drop-takeover-ge 0.5`：录 → 训 → 回放评测
  一键循环，报告自动落盘。

- 实验记录（2026-08-31）：`--line-morph`（分割标线形态扰动）、`--dedup` 与
  `--drop-takeover-ge`（E2E）在离线指标上目前均为**负优化**（v7 标线
  line IoU 0.33→0.03；E2E v2/v3 接管率 ≥ 默认全量 v1），故默认关闭/不用，
  保留为数据量增大后的消融选项。


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
| **PurePursuit 预瞄自适应** | `find_target`/`steering` 现接受 `speed=`，内部走 `adaptive_lookahead`（向后兼容，不传仍用固定值） | 高速 S 弯/缓弯转向平稳 | ✅ 已接线（API + 测试）；rule/FSD 驾驶循环此前已手动设置 lookahead |

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
