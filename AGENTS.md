# AGENTS.md

## 项目定位

基于 BeamNG.drive + BeamNGpy 的自动驾驶研究项目。当前路线是先做分层（感知 / 决策 / 控制），再升级端到端模仿学习与图像 RL。

`README.md` 是当前权威说明，改动前先读它，并顺着相关里程碑（M1-M6）理解上下文。

## 开发前提（Tech-first）

1. 优先基于 BeamNG.tech 开发：感知以真传感器（Camera / LiDAR）与 annotation 像素真值为准，学习数据从 Tech 采集。
2. Steam 兼容路径（窗口截屏、Lua 射线、经典 CV 回退、YOLO 2D 反投影）只保底不坏：不为其新增功能、不做优化，统一留给后期下放适配。
3. 涉及真实游戏的改动默认用 Tech 验证（`--runtime tech`）；纯逻辑改动跑离线回归（pytest + `m5_offline_validate.py`）。

## 目录职责

- `beamng_autopilot/`：可复用库。核心模块包括 `config`、`connector`、`perception`、`planner`、`lane`、`control/`、`vision/`。
- `scripts/`：按里程碑拆分的入口脚本、探针与测试。脚本保持薄，业务逻辑尽量放库。
- `tests/`：pytest 纯逻辑回归（不依赖游戏），覆盖 traffic / planner / perception / lane 等库模块。
- `data/`：轨迹等数据；只提交 `data/track_smallgrid.npz` 这个样例。
- `logs/`：运行产物、遥测、训练输出，全部 gitignore，不提交。
- `weights/`、`.yolo/`：模型与 YOLO 配置，属于运行时产物，不提交。

## 铁律（不可违反）

**禁止「导航线/地图线 + 固定偏移」作为横向定位逻辑**。横向参考（车应该处在
车道内的哪个横向位置、压没压线、终点停在哪）只能来自感知 —— 语义分割 / 漆画线
投影 / 可行驶边界等传感器输出（如 `painted_line_lane_center`：标线右侧
`lane_half_m` = 本车道中心）。以下写法一律禁止：

- 用导航路线中心线、地图车道中心线 + 固定偏移（`RIGHT_OFFSET_M`、
  `SNAP_LANE_OFFSET_M`、`route + 2.0m` 等同类常量/表达式）做横向落点、
  转向参考或压线判定；
- 感知不可用时回退到地图/导航线做横向兜底。

感知不可用时的合法降级只有：刹停、保持当前航向直行、落到路面安全点。

旧 M5 规则驾驶（`LocalPlanner` / `m5_autopilot` 的 `right_offset`）是唯一例外，
仅作兼容兜底保留，FSD 栈及其入口（`m5_fsd_drive`、`m5_shadow_drive`）不得使用或
重新引入同类逻辑；新代码不得新增任何"导航线/地图线 + 偏移"常量。横向逻辑是否合规
以此为准：能否用感知指标（如 `line_lat`）回答"车相对感知标线在哪一侧"，而不是
相对导航中线/偏移。

## 关键约定

1. 复用优先。已有 `PurePursuit`、`SpeedController`、`LocalPlanner`、`LaneTracker`、`VisionDetector`、`BeamNGConnector`、`ControlBridge` 等组件，先搜索再实现，不要重复造轮子。
2. 新增能力放进 `beamng_autopilot/`，入口脚本放进 `scripts/`。脚本只负责参数解析、组装与运行流程。
3. 常量就近放在模块顶部，或放到 `beamng_autopilot/config.py`。路径统一用 `PROJECT_ROOT`、`DATA_DIR`、`LOGS_DIR`，不硬编码机器绝对路径。
4. 与 BeamNG 的通信集中在 `connector`、`watchdog`、`control` 相关模块，不在业务循环里散落 Lua 字符串。
5. GUI 与 autopilot 的进程间命令走 `ControlBridge`（`logs/autopilot_ctl.json`），实时遥测走 `logs/telemetry/live.json`；不要另起一套文件协议。
6. 不修改运行时生成物：`logs/`、`weights/`、`.yolo/`、`probe_out.txt`、`probe_route.json` 等。
7. Python 3.10，模块开头使用 `from __future__ import annotations`。保持现有命名、注释与换行风格，不顺手格式化无关文件。
8. 工作区已有未提交改动时，把它们当作用户工作：先读再改，不还原、不覆盖无关修改。

## 改动流程

1. 先读：README、受影响模块、调用链和相似脚本。
2. 再方案：说明要改哪些文件、复用哪些组件、数据怎么流动、影响哪些现有功能、如何验证。
3. 后实现：保持最小 diff，不夹带无关重构。
4. 验证：纯逻辑改动至少跑 `pytest tests/` 与 `scripts/m5_offline_validate.py`；涉及真实游戏的改动默认用 Tech（`--runtime tech`）验证对应探针或测试，Steam 路径只需确认不回归。
5. 收尾：报告实际改动与验证结果，不假装没跑过的测试通过。

## 常用验证

- 纯逻辑回归（不需要游戏）：`.venv\Scripts\python.exe -m pytest tests/ -q`
- 深度离线回归（不需要游戏）：`.venv\Scripts\python.exe scripts\m5_offline_validate.py`
- 端到端（需要 Tech 游戏）：`.venv\Scripts\python.exe scripts\m5_e2e_test.py --attach --runtime tech`
- 真实驾驶（需要 Tech 游戏）：`.venv\Scripts\python.exe scripts\m5_drive_test.py --runtime tech --speed 6 --run 10`
- GUI 冒烟：`.venv\Scripts\python.exe scripts\m5_gui_smoke.py`
