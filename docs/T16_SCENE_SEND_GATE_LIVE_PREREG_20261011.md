# 修复后实机验证预注册（2026-10-11）

状态：**预注册，先于本轮任何运行提交**。本轮是[驾驶正确性修复](T16_SCENE_AND_SEND_GATE_FIX_20261011.md)
（场景选择分离 + 最终发送时效门，commit `a409c22`）在冻结版本上的实机验证；
冻结依据：pytest 3404 项 + 深度离线 ALL PASS（`logs/experiments/gate_20261011_171740.log`，
HEAD `97a45db`）。

## 1. 验证问题（按复核方案阶段三第 6 条顺序）

1. **场景一致性**：修复后诊断在场景选择之后只读——实机帧中
   `plan_dev_m`/`ref_gap_m` 应在诊断成功时存在；`--probe-geometry` 捕获的生产
   场景/候选/最终命令链可回放连接。
2. **最终发送时效门**：每帧应有有限 `stale_at_send_s`（发布年龄 + 单调时长的下界）；
   `stale_at_send_s > 1.5` 的帧必须无正油门；`send_stale_stop_reason` 只在该门实际
   改变指令的帧出现。
3. **曝光资格**：`fixed_actual_pose`（0.10 m / 1°）放置与曝光资格照旧执行。

## 2. 冻结参数

- 地图 italy；anchor **B 臂** `(779.4392068, 734.5823346, −13.107691)`（沿用 E5 A/B
  预注册的 baseline 实测锚点）；goal `(868.3, 744.9)`；
- lane-mode sensor、strict、`seg_model_hand/best_task.pt`、25 s、6 m/s、
  `--max-wall-s 150`、`fixed_actual_pose`、容差 0.10 m / 1°；
- **`--probe-geometry` 开启**（诊断捕获，不影响控制——与 A/B 批次的
  `probe_geometry=false` 配置不同，因此本轮**不与 A/B 运行做等曝光并列**）；
- 结果目录：`logs/experiments/e5_postfix_sendgate_20261011/`；
- 入口：`pwsh scripts/m5_pose_sweep.ps1 --lateral 0 --yaw 0`（单 pose）。

## 3. 预注册判据

- 放置/曝光资格不合格、启动失败、UNKNOWN 一律原样保留并单独报告；
- 判据 1（场景链）：合格帧的几何捕获与 `plan_dev_m`/`ref_gap_m` 可回放连接；
- 判据 2（发送门）：全部帧 `stale_at_send_s` 有限；不存在
  `stale_at_send_s > 1.5` 且油门 > 0 的帧；
- **不声称**：与 A/B 的因果或重复性比较（配置与源身份均不同）；任何安全或
  学习收益结论。
