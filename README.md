# BeamNG Autopilot — 视觉智驾研究

基于 BeamNG.drive + BeamNGpy 的自动驾驶研究项目。路线：先分层（感知/决策/控制），再升级端到端模仿学习与图像 RL。

## 里程碑

- M1 循迹闭环：Pure Pursuit + PID，车沿闭环自动跑圈 ✅
- M2 视觉感知：相机标定 + 轮胎印条带检测 + 路径投影 ✅
- M3 端到端模仿学习：DAVE-2 风格 CNN（单帧图像 -> 转向）🔨 进行中
- M4 决策层：DQN 离散动作（巡航/减速/变道/超车），Stable-Baselines3
- M5 整合 demo：感知 -> 决策 -> 控制，新能源车机界面（前视 + BEV + 仪表）
- M6 图像端到端 RL（SAC/PPO from pixels）

## 环境

- BeamNG.drive 0.39+（Steam 版；BeamNG.tech 可选增强）。安装目录自动探测
  （Steam 注册表 / `libraryfolders.vdf` / 常见路径），找不到时用环境变量
  `BEAMNG_HOME` 指定；用户目录默认使用当前用户的
  `%LOCALAPPDATA%\BeamNG.drive\<版本>`，可用环境变量 `BEAMNG_USER` 覆盖。
- Python 3.10 + venv（`--system-site-packages`，复用系统 torch cu128 / OpenCV）
- 显卡建议 6GB 显存以上（YOLO 检测 + HUD 需要）

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次使用前建议跑一遍环境自检，输出依赖 / 游戏路径 / 资源 / 运行时状态清单：

```powershell
.venv\Scripts\python.exe scripts\m5_env_check.py
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

# M2 采集（低帧率，用于感知标定）
.venv\Scripts\python.exe scripts\m2_capture.py --track data\track_smallgrid.npz --laps 1

# M3 高频采集（BC 训练数据，320x180 帧）
.venv\Scripts\python.exe scripts\m3_collect_bc.py --track data\track_smallgrid.npz --speed 8.0 --laps 3

# M3 训练
.venv\Scripts\python.exe scripts\m3_train_bc.py --runs logs\m3_bc\20260811_* --epochs 60 --out logs\m3_bc\bc_steer

# M3 实车推理（BC 驾驶）
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --speed 8.0 --duration 180
```

运行时会启动 BeamNG.drive 窗口，可实时观察车辆。遥测与数据集保存在 `logs/`。

## 实时遥测 HUD

`m3_drive_bc.py` / `m1_follow_track.py` 运行时默认弹出可视化遥测窗口（油门、刹车、转向、速度、G 力表、航向、圈数），带前置摄像头预览；按 `q` / `ESC` 可提前结束驾驶。禁用方式：

```powershell
# 关掉整个 HUD 窗口
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --no-hud
# 只去掉摄像头预览，保留仪表
.venv\Scripts\python.exe scripts\m3_drive_bc.py --model logs\m3_bc\bc_steer.pt --track data\track_smallgrid.npz --no-camera
```

也可以在第二个终端单独开仪表盘（无摄像头），读取运行中的实时遥测：

```powershell
.venv\Scripts\python.exe scripts\m4_dashboard.py
```

决策层实时可视化（读 autopilot 实际发布的遥测，显示感知融合 / planner
模式 / 速度决策链 / 交通规则 / 控制输出 + 鸟瞰 + 速度时序图）：

```powershell
.venv\Scripts\python.exe scripts\m5_decision_view.py
```

## M5 游戏内自动驾驶助手（可手动激活）

在游戏里通过热键手动激活/关闭自动驾驶，车辆严格沿游戏内置导航路线（大地图里点选目的地生成的蓝色箭头路线）自己跑，并带特斯拉式视觉叠加和结束后的遥测图表。

非 `--attach` 启动时，脚本会自动加载 `italy`（模拟意大利）地图的 `spawn_crossroads` 十字路口出生点，并把车放到附近路网节点上按道路方向摆正（避免像 `hirochi_raceway` 这种原点不在路面上的地图把车生成到地图下面）。路线直接抓取游戏内置导航（不再自己用路网 A* 寻路），脚本只负责抓取后跟车。

### 启动（推荐：先开游戏，进地图，摆好车）

```powershell
.venv\Scripts\python.exe scripts\m5_autopilot.py --attach
```

不带 `--attach` 时脚本会自己启动游戏并加载 `italy` 地图的十字路口出生点（`spawn_crossroads`）：

```powershell
.venv\Scripts\python.exe scripts\m5_autopilot.py
```

### 热键（全局，游戏内直接按）

| 按键 | 功能 |
| --- | --- |
| `F8` | 视觉叠加 开/关（3D 世界路线 + 前视投影 + 鸟瞰图） |
| `F9` | 自动驾驶 开/关 |
| `F10` | 抓取游戏内导航路线（先按 `M` 开大地图，点选目的地生成蓝色路线） |
| `F11` | 清除路线 |
| `F12` | 退出 |

### 玩法

1. 在游戏里按 `M` 打开大地图，点一个目的地，关闭地图后出现蓝色导航路线；
2. 按 `F10` 抓取这条导航路线（提示"navigation route grabbed: N pts"）；
3. 按 `F9` 激活自动驾驶：Pure Pursuit + 弯道限速 + PID 跟车，严格沿游戏生成的导航路线行驶，到达目的地自动刹车结束；
4. 自动驾驶期间/之后可用 `F8` 开关 3D 世界叠加与 HUD 里的前视投影 + 鸟瞰图；
5. 每次自动驾驶结束（到达 / 手动关闭 / 超时）自动弹出"油门 / 刹车 / 速度"三段连续条形统计图，同时保存 PNG 到 `logs/telemetry/m5_telemetry_*.png`；
6. 自动驾驶关闭后车辆控制交还给你（手动驾驶）。

常用参数：`--speed 10` 巡航速度（m/s）、`--max-run 600` 单次最长秒数、`--no-hud` 关掉 HUD 窗口、`--no-show` 只存图表不弹窗。控制台的 `限速 (km/h)` 会以 `--speed` 传给助手，并可在运行中通过 `set_speed` 更新。

## 控制台界面（一键启动 + EID 环境信息显示）

`scripts/m5_launcher.py` 是一个像软件一样的启动控制台：不用记命令、不用开多个窗口，按钮就能完成"启动游戏 → 启动助手 → 一键自动驾驶"，右侧实时显示环境信息（EID）。

双击项目根目录的 `启动自动驾驶.vbs` 即可直接打开控制台界面（自动定位 venv、隐藏黑窗口，不需要开终端）；也可以手动运行：

```powershell
.venv\Scripts\python.exe scripts\m5_launcher.py
```

### 界面布局

- 左侧按钮：`启动游戏`（自动以 TCP 模式拉起 BeamNG）、`启动助手`（运行 m5_autopilot.py）、`一键自动驾驶(F9)`、`抓取路线(F10)`、`清除路线`、`停止自动驾驶`；
- 左侧设置区：`限速 (km/h)` 输入 + `应用` 按钮（助手运行中立即生效，未运行则保存到下次启动）、地图、车型、`--attach` 与标记点开关；
- 右侧 EID 面板：当前速度（大数字）+ 目标速度 + 限速显示、模式徽章（手动 / 巡航 / 避障中）、油门 / 刹车 / 转向条、G 力图、障碍物数量、最近障碍距离、视觉检测目标、传感器状态、路线点数量、距目标距离、已运行时间、航向角；
- 中部 `BirdView`：俯视地图，实时画路线点、障碍物包围框和车头朝向；
- 底部日志区：实时滚动显示助手进程输出。

### 按钮流程

1. 点 `启动游戏`，等游戏进地图、把车摆好；
2. 在游戏里按 `M` 打开大地图选目的地（出现蓝色导航路线）；
3. 回控制台点 `抓取路线(F10)`（对应游戏内热键 F10），再点 `一键自动驾驶(F9)`；
4. 想停就点 `停止自动驾驶`（对应 F9 再次按下），退出时不会关游戏。

设置区里的 `限速 (km/h)` 会在启动助手时作为 `--speed` 传入；助手运行中点 `应用` 会立即下发 `set_speed`，未运行时则先保存、下次启动生效。

### 通信说明

控制台通过 `logs/autopilot_ctl.json` 与 m5 助手做命令桥（带单调序号水印防重放），助手把 EID 数据实时写进 `logs/telemetry/live.json`，控制台按帧读取刷新。除 F9/F10/F11 这类开关命令外，命令桥还支持带数值的 `set_speed`，用于运行时更新巡航限速。仅此界面需要，命令行玩法不受影响。

## M5 视觉避障（YOLO 前视 + 反投影）

M5 自动驾驶默认开启视觉障碍物感知：后台线程预热 YOLOv8n，主循环里约每 0.33s（`--vision-rate 3`）抓一次游戏窗口帧，YOLO 检测人 / 车 / 摩托车 / 公交车 / 卡车，再通过当前游戏镜头的真实位姿（Lua 查询 `getCameraPosition / Forward / Up / FovDeg`）把检测框底边中心反投影到地平面（z=0），得到障碍物世界坐标，随后与场景 / 射线障碍合并后交给 planner 绕行。

- 依赖：`weights/yolov8n.pt`（约 6.5MB，缺省时自动下载到项目目录）+ `.yolo/`（ultralytics 配置目录，避免写入 AppData 导致无权限崩溃）。
- 拿不到游戏窗口 / 画面空白时自动降级：静默跳过视觉扫描并计数，不影响其他感知与控制。
- HUD 叠加青色检测框（类别 + 置信度），状态行显示 `vis=N`（当前视觉障碍数量）。
- 同一辆车被视觉与场景 / 射线同时看到时按距离合并成一个障碍（`merge_obstacles`），避免重复绕行。

### 视觉探针（建议先跑这个验证检测效果）

进游戏地图后运行，不需要开自动驾驶：

```powershell
# 单帧：打印检测到的障碍物（世界坐标 / 距离 / 尺寸）
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --once

# 持续检测并保存标注帧到 logs\m5_vision\
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --save

# 弹窗实时预览（按 q 退出）
.venv\Scripts\python.exe scripts\m5_vision_probe.py --attach --show
```

### 视觉相关参数

- `--no-vision-obstacles`：关闭视觉避障（仅用场景 + 射线障碍）。
- `--vision-conf 0.35`：YOLO 置信度阈值（默认 0.35，误报多就调高）。
- `--vision-rate 3`：视觉扫描频率 Hz（默认 3；RTX 5070 上可调到 5–10）。
- `--max-dist 55`（探针）：忽略超过该距离的检测。

注意：视觉检测依赖当前游戏镜头真实位姿（Lua 查询），玩家自由视角下同样自洽；`m5_autopilot.py --attach` 默认已开启视觉。

## 开发日志

按时间正序记录；2026-08-14 之前的条目由文件时间戳、README 与 docs 重建，之后以 git 提交为准。新增改动追加到本节末尾。

### 2026-08-11

- 项目起步与 M1 循迹闭环：搭建 venv 与 `requirements.txt`，实现轨迹录制/回放（`track.py`）、PID（`control/pid.py`）、Pure Pursuit（`control/pure_pursuit.py`），`m1_smoke_test.py` / `m1_record_track.py` / `m1_follow_track.py` 跑通闭环，并保存样例轨迹 `data/track_smallgrid.npz`。
- M2 视觉感知起步：`vision/band.py` 轮胎印条带检测，`m2_capture.py` / `m2_calibrate_camera.py` 完成相机标定与低帧率采集；结论为 4m 条带 r=+0.93、12m r=+0.91，但纯视觉无先验在 2.4fps 下不可靠。
- M3 模仿学习起步：`bc.py` + `m3_train_bc.py` 实现 DAVE-2 CNN 训练流程，用 M2 的 229 帧跑通 PoC（val MAE=0.048，R²≈0），确认低帧率、近零转向标签的数据没有训练价值。
- 遥测可视化：`hud.py`、`telemetry_chart.py`、`m4_dashboard.py` 仪表盘雏形，M1/M3 驾驶可弹 HUD 实时查看油门/刹车/转向/速度。

### 2026-08-12

- 热键框架：`beamng_autopilot/hotkeys.py`，为 M5 游戏内 F8/F9/F10/F11/F12 控制打基础。
- 控制链路诊断：`diag_parkingbrake.py` / `diag_disconnect.py` / `diag_r_latch_drive.py` / `diag_gear_map.py` / `diag_gearbox_info.py` / `diag_gearbox_list.py` / `diag_arcade_standstill.py` / `diag_arcade_neutral.py`，排查手刹、断开、倒挡自锁、挡位映射与 arcade 控制问题。
- 感知探针：`m5_rayframe_probe.py` / `m5_rayground_probe.py` / `m5_castray_struct.py` / `m5_castray_compare.py` / `m5_live_blocker_probe.py` / `m5_watchdog_probe.py`，并新增 `watchdog.py` 看门狗；记录 twitch/park 场景，后续纳入 `m5_offline_validate.py` 回归。
- 视觉检测：`vision/detection.py` 加入 YOLO 检测与地面反投影，下载 `weights/yolov8n.pt`；`control/speed.py` 速度控制；`m5_vision` 检测结果落地。

### 2026-08-13

- 墙体/路线/避障排障：`m5_wall_shape_probe.py` / `m5_wall_fan_probe.py` / `m5_wall_multi_probe.py` / `m5_live_wall_probe.py` / `m5_live_wall_probe2.py` / `m5_wall_route_probe.py` / `m5_live_route_probe.py` / `m5_live_planner_diag.py`，覆盖墙面形状、扇形/多墙、live 路线与 planner 诊断。
- 挡位控制：`control/gearbox.py` + `m5_gearbox_diag.py`。
- 感知与遥测：`perception.py` 场景/射线/视觉融合、`vision/tracking.py` 目标跟踪、`visionview.py` 前视叠加、`control/handover.py` 人机交接、`telemetry.py` 实时遥测，`m5_watchdog_beat_test.py` 心跳测试；lane debug run55-57 用于车道状态排障。

### 2026-08-14

- 双运行时：`beamng_autopilot_tech/providers.py` 惰性创建 Tech `Camera` / `Lidar`，`launch_game.py` / `runtime.py` / `download_beamng_tech.py` / `bridge.py` 与 `BEAMNG_RUNTIME` 系列环境变量；同日完成 Steam/Tech 多轮 e2e 验证。
- M5 整合：`m5_e2e_test.py` / `m5_drive_test.py` 端到端/实车测试、`m5_launcher.py` + `m5_gui_smoke.py` 控制台界面、`traffic.py` 交通、`connector.py` 扩展，以及 `启动自动驾驶.vbs` / `启动车道状态窗口.vbs`。
- 车道状态与局部规划：`lane.py` 车道几何/状态、`planner.py` 局部规划、`roadnet.py` 路网，配套 `m5_lane_state_probe.py` / `m5_lane_center_capture.py` / `m5_lane_state_annotate.py` / `m5_lane_state_view.py`，车道状态数据落到 `logs/m5_lane_state`。
- 离线回归与 planner 基线：`m5_offline_validate.py` 大型离线回归；`docs/planner_baseline_20260814.md` 记录 Steam run 98（median_lat=1.76、centered_ratio=0.901），并明确“导航线只决定路线、不决定车道内横向基准”的改进方向。
- M2 收尾分析：`m2_steering_signal.py` / `m2_steering_vision.py` / `m2_validate_projection.py` / `m2_visualize.py`，把转向信号/视觉相关性结论固化为可重跑脚本。
- 工程落地：14:37 初始化 git 仓库与 `AGENTS.md` / `.gitignore` / README 首版，23:42 首次提交 `9702e71`（整仓快照，107 个文件、28981 行）。

### 2026-08-15

- `00:06 68cbfc0`：环境自检 `m5_env_check.py` 与抓帧性能探针 `bench_grab_screen.py`，README 补充用法。
- `00:41 cf973d3`：`CameraModel.camera_pose` 支持车辆 6DOF 姿态（pitch/roll 参与反投影，BeamNG 四元数约定用真实 state 验证）；Tech 相机改为标定外参 + 姿态驱动，移除逐帧 GE 查询；`m5_lane_state_view.py` 抓帧失败降级。
- `00:58 2481774`：`m5_lane_truth_probe.py` 对比经典 CV 与 BeamNG.tech 像素真值；italy AI 80 帧实测路面 IoU 0.734、标线 recall 0.015、边界误差 0.8-1.0m，量化确认传统 CV 在真渲染帧上不可用。
- `01:01 起（当前未提交）`：学习式分割路线。新增 `vision/segmentation.py`（轻量 UNet，约 1.3M 参数，background/asphalt/line 三类）、`m5_collect_seg.py`（Tech colour + annotation 半分辨率采集）、`m5_train_seg.py`（时间序 80/20、中位频率加权、mIoU 监控）；`lanes.py` 抽取共享 `_mask_to_markings` 管线，`lane_overlay.py` 的 `estimate_pavement_edges` 支持学习式 off-road mask；`m5_lane_truth_probe.py --model` 与 `m5_autopilot.py --seg-model` 接入，无模型时回退经典 CV。
- `3925723`：动态交通 ACC 跟车/超车。`traffic.py` 新增 `find_lead_vehicle` / `follow_speed` / `should_overtake` / `vehicle_along_speed`（沿路线投影找前车、时间间隙跟车、慢前车触发超车请求）；场景车辆扫描带速度/航向/ID（Lua `getVelocity`），`merge_obstacles` 合并时保留动态状态；`planner.speed` 运动学限速改为 `sqrt(v_lead^2 + 2ad)`，移动前车不再当静态墙刹停、对向来车仍按静态处理；`m5_autopilot.py` 在 blocked / 可绕行两分支接入 ACC 跟车（moving lead 跟车、静态障碍仍停车），遥测新增 `lead_d` / `lead_v` / `follow` / `overtake` 字段。
