# AGENTS.md

## 项目定位

基于 BeamNG.drive + BeamNGpy 的自动驾驶研究项目。当前路线是先做分层（感知 / 决策 / 控制），再升级端到端模仿学习与图像 RL。

`README.md` 是当前权威说明，改动前先读它，并顺着相关里程碑（M1-M6）理解上下文。

## 目录职责

- `beamng_autopilot/`：可复用库。核心模块包括 `config`、`connector`、`perception`、`planner`、`lane`、`control/`、`vision/`。
- `scripts/`：按里程碑拆分的入口脚本、探针与测试。脚本保持薄，业务逻辑尽量放库。
- `data/`：轨迹等数据；只提交 `data/track_smallgrid.npz` 这个样例。
- `logs/`：运行产物、遥测、训练输出，全部 gitignore，不提交。
- `weights/`、`.yolo/`：模型与 YOLO 配置，属于运行时产物，不提交。

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
4. 验证：纯逻辑改动至少跑 `scripts/m5_offline_validate.py`；涉及真实游戏的改动再跑对应探针或测试。
5. 收尾：报告实际改动与验证结果，不假装没跑过的测试通过。

## 常用验证

- 离线（不需要游戏）：`.venv\Scripts\python.exe scripts\m5_offline_validate.py`
- 端到端（需要游戏）：`.venv\Scripts\python.exe scripts\m5_e2e_test.py --attach`
- 真实驾驶（需要游戏）：`.venv\Scripts\python.exe scripts\m5_drive_test.py --speed 6 --run 10`
- GUI 冒烟：`.venv\Scripts\python.exe scripts\m5_gui_smoke.py`
