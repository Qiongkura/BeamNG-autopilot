# 修复后实机验证结果（2026-10-11）

预注册：[T16_SCENE_SEND_GATE_LIVE_PREREG_20261011.md](T16_SCENE_SEND_GATE_LIVE_PREREG_20261011.md)
（commit 先于运行）；冻结版本 HEAD `97a45db`，全量门 `gate_20261011_171740.log`
（pytest 3404 项 + 深度离线 ALL PASS）。
结果目录：`logs/experiments/e5_postfix_sendgate_20261011/pose_00`。

## 1. 运行身份

| 项 | 值 |
|---|---|
| status / qualified | **invalid**（`telemetry exposure too short`）——按预注册原样保留 |
| 放置 | `placement_rc=0`，实际放置误差 0.0215 m / 0.0755°（契约 0.10 m / 1° 内） |
| 帧 / 跨度 | 44 帧，23.664 s（资格阈值之下 ~1.1 s，与 A/B `04_O` 同型的曝光不足） |
| 驾驶读数 | 净行进 2.449 m、最大速度 1.260 m/s、被接受路径 36 帧、sensor 车道 36 帧 |
| 安全字段 | damage 增量 0、倒车 0 帧；主因 no drivable path 19、obstacle very close 9 |
| owned cleanup | `closed=true`、`unverified=[]`、`still_running=[]`；收尾后 BeamNG 进程 0（两次 kill 尝试失败条目原样保留） |
| E2E 单帧最大 | 831.1 ms（本轮无 A/B `04_O` 型 5.8 s 冷启动巨滞） |

## 2. 预注册判据逐条

1. **场景链（判据 1）——通过**：`--probe-geometry` 在 44/44 帧捕获
   `control-geometry-v1` 完整生产链（scene_ref / planner_path / candidates /
   selected_path / safety_assessment / verdict / command）；每帧
   `geometry_probe.command.cmd_seq` 与该帧 `cmd_seq` 一致（44/44），场景→候选→
   安全评估→最终命令可逐帧回放连接。诊断在场景选择后只读：
   `plan_dev_m` 23 帧、`ref_gap_m` 36 帧（其余为引用缺失时的合法 None）。
2. **发送时效门（判据 2）——通过**：44 帧中 1 帧记录
   `stale_at_send_s=1.846`（>1.5）；该帧 `final_stop=true`（因 no drivable path
   已在硬停）、油门 0.0 / 刹车 1.0，故 `send_stale_stop_reason` 按设计缺席
   （门未改变结果时不写原因）。**全轮不存在 `stale_at_send_s > 1.5` 且油门 > 0
   的帧**；发送年龄有界记录，未伪装采集年龄。
3. **曝光资格**：不合格，原因原样保留；不把 invalid 读成驾驶失败或成功。

## 3. 边界与未完成

* 曝光差 ~1.1 s 的机制（冷启动/节拍）与复核方案阶段一第 3 项（**预热与完整控制计时**）
  对应，该项**尚未实施**——本轮不归因、不调阈值。
* 本轮配置（probe 开启）与源身份均不同于 E5 A/B 批次，按预注册不与 A/B 做等曝光
  并列或重复性比较。
* 验证的是两项修复在实机的可观测行为；不声称可靠驾驶、安全或学习收益。

## 4. 第二轮：完整批次代码的合格运行（f64d6bd）

第一批代码补齐阶段一第 3 项（曝光前预热 + 感知返回→发送后链计时）并重新冻结
（HEAD `f64d6bd`，pytest 3406 项 + 深度离线 ALL PASS，`gate_20261011_173542.log`）后，
按同一预注册参数重跑一轮：`logs/experiments/e5_postfix_sendgate2_20261011/pose_00`。

| 项 | 值 |
|---|---|
| status / qualified | **completed / qualified=true**（`source_unchanged=true`） |
| 放置 | `placement_rc=0`，误差 0.0215 m / 0.0750° |
| 帧 / 跨度 | 47 帧，24.709 s（曝光资格通过） |
| 发送时效门 | 0 帧 `stale_at_send_s>1.5`（本轮后链健康：`send_chain_ms` p50 63 / max 94 ms）；门在新鲜帧静默 ✓ |
| 几何链 | 47/47 帧 `geometry_probe.command.cmd_seq` 与 `cmd_seq` 一致 ✓ |
| 诊断 | `plan_dev_m` 25 帧（选择后只读） |
| 驾驶读数 | 净行进 0.463 m、最大速度 0.577 m/s、被接受路径 32 帧、sensor 车道 32 帧；主因 no drivable path 22、planned body cross 19 |
| 安全字段 | damage 增量 0、倒车 0；cleanup closed、收尾后 BeamNG 进程 0 |

**两轮合并读法**：第一轮（97a45db）证明发送门在真实停滞帧正确记录过期年龄且无正油门；
第二轮（f64d6bd）证明完整批次在合格曝光下门静默、后链时长可测、几何链逐帧可回放连接。
仍不声称：可靠驾驶（净行进仍处于已知可用率阻塞）、安全或学习收益。曝光不足的冷启动
机制（第一轮）与预热的关系需在阶段二的确定性归因中继续检验。
