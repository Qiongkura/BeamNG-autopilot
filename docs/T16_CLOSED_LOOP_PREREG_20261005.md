# T16 闭环对照预注册：v8 候选范围对**实车驾驶**的影响（2026-10-05）

- 目的：补上 A 前置的**步骤 ③ 实机闭环**。离线部分已完成
  （`docs/T16_PLANNER_IMPACT_V8_OFFLINE_20261005.md`：配对净 −3~−4 帧、车道中心
  中位误差 0.254→0.408 m、八个配对旋钮 0/12 恢复 → 损失内生于候选范围）。
  本预注册先定判据、再跑车。
- 工具：`scripts/m5_fsd_benchmark.py`（**场景注册表 + 记分卡**，仓库里就是为
  "跨提交可比"设计的）；`--score` 可离线重打分。

## 1. 单因子与口径

- **唯一变量**：候选范围协议 —— v7（默认）vs v8（`BEAMNG_PROTOCOL=v8`）。
  其它一律不动：场景注册表里的 `teleport/goal/seconds/speed/seg_model/lane_mode/strict`
  全部用注册表默认值（不改模型、不改相机、不改安全阈值、不改规划参数）。
- 场景：注册表里的 **`town` 与 `mountain`**（前者 `strict+sensor+require_goal`，
  后者是 README 的实车记录起点）。
- 重复：每配置每场景 **2 次**（驾驶非确定，单次不可信；取两次并报两次）。

## 2. 指标（全部来自记分卡，不改口径）

| 指标 | 为什么看它 |
|---|---|
| **`lane_paired_rate`**（主） | 感知配对可用率——v8 的候选范围直接作用在这里 |
| `lane_sensor_rate`、`lane_src_hist` | 车道参考来源构成（配对/单侧/传感器） |
| `off_road_frac` / `off_road_episodes` / `off_road_max_m` | 出铺装（安全） |
| `cross_centre_frames` / `cross_right_frames` | 压线（安全） |
| `stall_frac` / `stall_events` / `stuck_frames` | 停顿（可用性） |
| `checks.status`（pass/fail/unknown） | 记分卡的总体判定 |

## 3. 判定规则（先定后跑）

- **v8 判定为"驾驶侧不可接受"**：任一场景在 v8 下出现
  `off_road_episodes` 或 `cross_centre_frames` 或 `cross_right_frames` **由 0 变正**
  （即 v7 全 0 而 v8 >0），或 `stall_frac` 恶化 >50%（相对值）；
- **v8 判定为"驾驶侧可接受（配对降但无安全后果）"**：`lane_paired_rate` 明显下降
  但上述安全项不变差、`checks.status` 不劣化；
- **不确定**：两次重复结果矛盾，或出现 `unknown`（记分卡自己会说 unknown）。
- 无论结论如何**都不改默认**（本预注册只产证据；采纳 A 仍需计划层裁决）。

## 4. 复现命令

```pwsh
# v7（默认）
.venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --scenarios town,mountain
# v8（候选范围）
$env:BEAMNG_PROTOCOL="v8"; .venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --scenarios town,mountain
```
（记分卡输出路径以脚本实际输出为准；两次重复各存一份。）

## 5. 执行记录与一处例外（**必须随证据一起读**）

- 首次启动被 `check_exclusivity` 拒绝：`refusing to drive; pass --allow-contaminated
  to override`，被点名的"另一个控制器"是 **pid 13944 = 本会话的启动器 shell**
  （cmdline `bash.exe -c . .../snapshot-bash-....sh`，不含任何 `CONTROLLER_MARKERS`）
  ——即代码注释里已记录的**启动器误报类**（"the shim is this process's parent and
  carries the same command line... every run refuses to start"）。
- 因此两次运行都带 **`--allow-contaminated`**：这是守卫**显式提供**的开关，且
  **manifest 会记录该冲突**（可审计）。这不是"绕过检查"，而是把已知误报记在案；
  若将来有人读这批数据，必须知道这一条。
- 除该例外，命令与 §4 完全一致（场景注册表默认、单因子、两次重复）。
