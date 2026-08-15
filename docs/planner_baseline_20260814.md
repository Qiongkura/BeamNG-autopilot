# Planner 基线记录（2026-08-14）

## 当前横向寻路逻辑

`LocalPlanner.plan()` 的横向基准按传感器车道状态分三种：

1. 双边车道确认（`sensor_lane.paired == True`）
   - 直接使用检测到的车道中线作为路径。
   - 导航线只负责取窗口方向，不叠加横向偏移。
   - 对应 `beamng_autopilot/planner.py` 的 `lane_primary` 分支。

2. 只有单边边界（`lidar|right`、`lidar|left`、单边 `vision`）
   - 导航线仍是主路径。
   - 若 map 有合法车道，先执行 `_right_offset_path(..., offset=map_lane.preferred_offset_m)`。
   - 该偏移在 LHD 双车道 `lanes="-+"` 上通常是 `1.75m`（有节点半径时为 `2.0m`）。
   - 之后再叠加单边边界保护，最大约 `0.35-0.40m`。

3. 完全没有传感器车道
   - 使用导航线 + map 合法车道偏移。
   - map 中线边界还会把路径推到合法侧。

因此当前绝大多数单边帧的横向位置是：

```text
导航线 + map 合法车道偏移（1.75m） + 单边边界微调
```

这不是真车意义上的“车道中线跟随”，导航线在这里同时承担了宏观路线和横向基准两个职责。

## run 98 Steam 实车基线

命令：

```powershell
.venv\Scripts\python.exe scripts\m5_drive_test.py --runtime steam --speed 6 --run 98
```

关键指标：

| 指标 | 值 |
| --- | --- |
| reason | goal reached |
| frames | 873 |
| driven_m | 271.7 |
| median_lat | 1.76 |
| max_abs_lat | 2.21 |
| median_lane_lat | -0.35 |
| centered_ratio | 0.901 |
| lane_src | lidar|right=513, lidar|left=96, vision=116, vision|lidar=50 |
| plan_offset 中位数 | 1.75 |

解释：

- `median_lat` 是车相对游戏内导航折线的垂直距离，不是相对车道中心。
- `median_lane_lat` 是车相对视觉/LiDAR 检测车道中心的位置，`centered_ratio=0.901` 说明视觉上车在车道内 90.1% 时间居中。
- 单边帧占多数，`plan_offset=1.75` 表明 map 合法车道偏移在主动把路径推离导航线。
- 逐段统计显示偏导航线不是终点才发生：`175-200m` 段 median_lat=1.96，`250-275m` 段 median_lat=1.73。

## 待完善方向

按真车自动驾驶分层：

- 导航线只决定走哪条路，不决定车道内横向位置。
- 车道中线由视觉/LiDAR 感知给出。
- 单边感知只做边界保护，不主动推导航线。
- map 只阻止逆行、跨实线，不作为横向目标。

对应的最小改动点：

- `LocalPlanner.plan()` 单边/无感知分支不再应用 `map_lane.preferred_offset_m`。
- `last_lane_offset` 不再把 map 偏移记成 `plan_offset`。
- 离线测试同步改成“导航线不因 map 被右移”。
- 用 `m5_offline_validate.py` 和 Steam `m5_drive_test.py` 验证。

## 现状更新（2026-08-16）

上述横向基准改造已完成：

- `plan()` 中 `map_offset` 恒为 `None`，`preferred_offset_m` 不再参与任何路径
  生成（`traffic.legal_lane_view` 仍计算它，但只是 map 自身的车道信息，无
  运行时消费者）。
- 无感知/单边分支的横向基准改为固定 `RIGHT_OFFSET_M=1.5`（`_safe_right_offset`
  会按障碍/边界收缩），不再随 map 车道宽度/节点半径变化；run 98 的
  `plan_offset=1.75` 已成历史。
- `last_lane_offset` 记录的是安全右偏量，不再是 map 偏移。
- 离线 `sensor-plan` 系列用例断言的就是“导航线 + 1.5m 固定右偏”行为。
- `_safe_lateral_offset`（map 偏移时代的“目标侧被堵则收缩”助手）已无运行时
  调用方，仅被 `m5_offline_validate.py` 作为回归保留。
- 新增超车意图管理（`traffic.OvertakeStateMachine`）：慢前车持续 1.5s 才进入
  requested、再确认 0.4s 才 active；对向来车 / 左侧实线取消请求；active 期间
  解除 ACC 跟车限速让 planner 的 detour/bypass 完成超车，前车提速或消失即回
  落 none 恢复跟车。
