# 驾驶正确性修复：场景选择分离与最终发送时效门（2026-10-11）

依据：[项目复核与下一阶段方案](PROJECT_NEXT_DEVELOPMENT_REVIEW_20261011.md) §4-A/§4-B、§5 阶段一第 1–2 项。
本轮只完成该批次的代码与回归；**尚未运行 Tech 实测**（按方案阶段三：代码、定向与全量门、
源码冻结全部完成之后再实测）。

## 1. P0-A：诊断不再改变安全场景的选择

**缺陷**（离线复现于复核报告）：`fsd_drive.py` 驾驶循环中，F-C 只读诊断的
`try/except/else` 插在了 `if scene is not None:` 与其回退 `else:` 之间，把回退分支
劫持为「诊断成功分支」——诊断成功时反而用 `out.bev` 另建 Scene 并重新评估，
替换了规划器的有效场景；诊断抛异常时回退反而被跳过。

**修复**：

* 回退构造分支归还给 `if scene is None:`（只在原 Scene 缺失时进入）；
* 诊断移到场景选择**完成之后**，只读 `out.lane_ref` / `scene.lane_ref` / `best`；
  诊断异常只影响诊断状态（`plan_dev_m`/`ref_gap_m` 缺席），不触碰场景与裁定。

## 2. P0-B：最终发送边界的时效重判

**缺陷**（离线复现）：`_sensor_snapshot_age` 与安全年龄汇总读取**发布时刻冻结**的
head/BEV/车道年龄，不随规划、安全评估、神经头、仲裁的下游耗时推进。实测证据：
E5 `04_O` 单帧 E2E 最大 5781.9 ms，按 `cmd_t − source_poll_started_t` 复算的
range 代理年龄在发送时最大 6.159 s，而遥测发布值仅 0.273 s。

**修复**（不改既有阈值标定，不新增控制协议）：

* 新增 `_snapshot_age_at_send(out, recv_mono)`：发布年龄 + 感知返回后的单调时长
  （纯时长差，与时钟基准无关，不重复计计算耗时）。无真实采集时间戳时它是**下界**，
  不伪装成采集年龄；证据无效（inf）时原样返回，裁定权留给 SafetyMonitor 的既有
  stale 路径。
* 感知返回处记录单调时刻 `_out_mono`；最终 `_commands.send` 之前重判：
  发送年龄 > `STALE_PIPELINE_S`（1.5 s，健康观测之上、看门狗之下）时按**既有硬停
  契约**收尾（`_final_stop_controls(stop=True)` → 0 油门/满刹/回中），并把
  `stale_at_send_s`（下界年龄）与 `send_stale_stop_reason="stale_snapshot"`
  （仅当该门实际改变了本次指令时写入）记入遥测。
* 子步路径已有 `_substep.decide(plan_age_s=...)` 时效回交，保持不变。
* 既有字段原定义不变（`tick_wall_ms` 等照旧）；新增字段只增不改。

## 3. 回归测试（tests/test_fsd_drive_pipeline.py，22 → 29 项）

| 测试 | 钉住的契约 |
|---|---|
| `test_diagnostic_success_cannot_replace_a_valid_planner_scene` | 诊断成功时监控器收到的必须是 Stack 创建的同一 Scene 对象（身份断言） |
| `test_diagnostic_failure_keeps_the_planner_scene` | 诊断异常：场景不变、驾驶照常、诊断字段缺席 |
| `test_missing_scene_still_enters_the_explicit_fallback` | 只在缺 Scene 时进入兼容构造分支 |
| `test_real_monitor_verdict_is_a_pure_function_of_scene_content` | 真实 SafetyMonitor：同一场景重复评估一致，内容变化才改变裁定 |
| `test_stale_perception_at_send_cannot_send_throttle` | 评估后停滞 2 s：发送年龄下界 ≥2 s，硬停收尾，全部命令 0 油门，原因可见 |
| `test_fresh_tick_records_no_send_hold` | 无停滞时发送门静默，不改变既有指令路径 |
| `test_snapshot_age_at_send_is_a_monotonic_lower_bound` | 年龄推进的单调性与下界语义；时钟回拨不产生负年龄；无效证据不重判 |

## 4. 验证与边界

* 定向回归 29/29 通过；全量 pytest 另行记录（见本轮提交说明）。
* **不声称**：该修复对 E5 停车/压线现象的贡献尚未实测；发送年龄仍是下界
  （真实 sensor capture timestamp 仍 UNKNOWN）；阈值未重标定——可用率归因
  （阶段二）先于任何阈值调整。
