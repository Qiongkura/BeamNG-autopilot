# BeamNG non-shared depth 编码校准结果（2026-10-12）

输入：

* `logs/m5_seg/collect_e5line_known4_20261011/frames/known_line_a0s0_depthcal.npz`
  + `scene_known_line_a0s0.json`（truth points 60，实际投影可用 53）；
* `collect_e5line_known3_20261011`：**没有 depthcal 文件**，只有普通共享内存帧，不能冒充
  浮点深度标定输入。

工具：`scripts/m5_depth_calibration_audit.py`；报告：
`logs/experiments/t16_linepool_20261011/depth_calibration_audit_known4.json`、
`depth_calibration_audit_known3.json`。

## 1. 拟合规则

* 每个 truth point 由采集帧 camera JSON 投影到 depthcal 图像；取投影点 3×3 邻域
  raw `depth_ndc` 中位数；与相机光轴方向的真实世界深度配对；
* 单调 PAVA（原始值→真实米制深度）拟合；每第五个点留作 holdout；
* 只有 holdout 数量 ≥2 且 p95 绝对残差 ≤1m 的 5m bin 才叫 `supported`。这只是
  depth 编码的有效范围，不改变任何投影/准入冻结门。

## 2. known4 结果

| 深度 bin | holdout n | RMSE (m) | p95 abs residual (m) | 状态 |
|---|---:|---:|---:|---|
| 0–5m | 1 | 1.010 | 1.010 | 样本不足/不支持 |
| **5–10m** | 2 | 0.175 | **0.221** | **supported** |
| **10–15m** | 2 | 0.476 | **0.636** | **supported** |
| 15–20m | 2 | 4.912 | 6.602 | 拒绝 |
| 20–25m | 2 | 2.148 | 2.267 | 拒绝 |
| 25–30m | 2 | 2.877 | 2.996 | 拒绝 |

全体 11 个 holdout 点：RMSE 2.628m、p95 4.996m。raw buffer 总体范围
0.0162–0.3259（背景 raw 最大约 1.0003）；这解释了为什么直接套 OpenGL
near/far 反解会得到约 3.1e6m 的背景伪值。

## 3. 决定

* **实际有效范围只能暂定为 5–15m 的两个支持 bin**，且每 bin 只有 2 个 holdout
  点，属于探索性校准，不可作为安全/遮挡充分性证明；15m 外深度判据明确拒绝。
* `collect_e5line_known3` 缺少 depthcal，保持 UNKNOWN；不能回填或推断。
* 本校准**不能证明链-贴花已经分离，也不能证明生成器已修复**；它只把原先
  “深度编码未知”收敛为“5–15m 在本次 known4 的有限样本上可拟合，15m 外不可用”。
* frozen projection gate 仍独立有效；known_line 窄化站点的 `after_px_median=1.271 px`,
  仍超过 1.0 px，因此新线池仍未准入。

下一步若要用深度量化链-贴花分离，需要在多个站点/多个距离重复采样，扩大 holdout
样本，并把深度有效范围作为判据前置；不能拿本次 11 点校准替代真实验证。
