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

## 双运行时（Steam / Tech）可并行开发

核心代码库保持 Steam 版兼容、可开源；BeamNG.tech 专属能力放在独立包
`beamng_autopilot_tech/`，只在检测到 Tech 时惰性导入，普通玩家直接使用
开源仓库不会受到影响。

- `BEAMNG_RUNTIME=auto|steam|tech`，默认 `auto`：连接后根据 BeamNGpy
  的 `tech_enabled()` 自动选择。
- `BEAMNG_TECH_HOME` 指向 Tech 安装目录；Steam 仍使用 `BEAMNG_HOME`。
- `BEAMNG_TECH_USER` 指向 Tech 用户目录，默认 `...\BeamNG.drive\0.38`，
  与 Steam 的 `0.39` 分开，避免两个版本互相污染存档/设置。
- `BEAMNG_PORT` 可选，默认 `64256`；需要同时跑 Steam 和 Tech 两套实例
  时改成不同端口即可。
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
