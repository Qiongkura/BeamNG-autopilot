# 多站点/多距离链-贴花分离验证结果（2026-10-12）

预注册：`docs/T16_DEPTH_SEPARATION_MULTISITE_PREREG_20261012.md`（commit `5ce3688`）。
运行目录：`logs/m5_seg/collect_e5line_depthcheck_20261012/`；端口 `50000`。

## 1. 冻结投影门（先于深度判据）

| 站点 | 场景 | 覆盖 | 校准后均值 | 校准后中位 | 门 |
|---|---|---:|---:|---:|---|
| known_line_a0s0 | 平/直 | 0.9582 |  — | — | **通过** |
| material_mix_a0s1 | 材质混合 | 1.0000 | — | — | **通过** |
| slope_curve_a0s2 | 弯/坡 | 1.0000 | — | — | **通过** |

采集器批次报告：三站均 `eligible`，校准后 mean=0.883 px、median=0.826 px、
`<=2px=3/3`。导出器冻结准入：**3/3 站、9 帧全部 `engine_verified`，0 隔离**。
每站 3 帧、2m 步进；没有用旧格式 depthcal 或降门补齐。

## 2. 深度编码有效范围

`m5_depth_calibration_audit.py` 报告：
`logs/experiments/t16_linepool_20261011/depth_calibration_audit_depthcheck.json`。

| 站点 | truth 样本 | holdout | 支持 bin（p95≤1m） | 结论 |
|---|---:|---:|---|---|
| known_line_a0s0 | 52 | 11 | 5–10m (0.583m), 10–15m (0.605m) | 支持 |
| material_mix_a0s1 | 52 | 11 | 5–10m (0.511m) | 只支持 5–10m |
| slope_curve_a0s2 | 27 | 6 | 无（每 bin holdout=1，不满足 n≥2） | UNKNOWN，不外推 |

这一步没有把 depth 当成普遍可靠的 OpenGL NDC；超出 supported bin 的深度分离
读数明确拒绝。坡/弯站点的冻结投影门已经通过，但 depth 证据因 holdout 不足保持 UNKNOWN。

## 3. 链-贴花分离与生成器决策

* 本轮**没有发现需要立即修改生成器链采样的投影门失败**：三种几何条件（直、材质混合、
  弯/坡）均通过冻结投影门；因此不能把此前已隔离站点的失败直接归因于生成器，而应
  保留旧批次的链-贴花分离诊断为历史边界。
* 深度判据只在已支持的 5–15m（具体按站点 bin）可用；本轮坡/弯站点深度仍 UNKNOWN，
  不能用它声称“深度证实分离为零”。
* 新池可作为**第二线帧池族**入库：三站 9 帧、全为线帧、engine_verified。

## 4. 跨池族隔离

旧人工修订池 `annotate_pkg_e2_it3/front_fisheye`（15 帧）+ 新 engine_verified
池（9 帧）重跑 `m5_seg_dataset_audit.py`：

* train 24 帧，road/paint/trainable 均 24；
* `group_overlap=0`、`content_overlap=0`、`exposure_overlap=0`（checked=true）；
* 新旧池族可作为后续跨池族配比实验的两个独立组。

这一步解锁的是**配比实验输入资格**，不是模型晋级或驾驶安全结论。
