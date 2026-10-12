# 多站点/多距离链-贴花分离验证预注册（2026-10-12）

状态：**预注册，先于本轮 Tech 运行提交**。依据 depth 校准报告
`docs/T16_DEPTH_CALIBRATION_RESULT_20261012.md`：仅 5–15m bin 暂时支持，15m 外拒绝。

## 冻结配置

* Tech RPC：`BEAMNG_TECH_PORT=50000`（64257 段在本机 WinError 10013）；
* 入口：修复后的 `scripts/m5_controlled_scenes.py`；
* Italy，1 anchor，3 sites，显式场景 `known_line material_mix slope_curve`；
  `--frames 3 --step-m 2.0 --anchor-offset 36 --line-convention tiers`；
* 每站保存普通帧 + `_depthcal.npz`（非共享内存 float depth/raw fallback）；
* 输出：`logs/m5_seg/collect_e5line_depthcheck_20261012/`。

## 预注册判据

1. **depthcal 存在**：每个含线站点必须有 `_depthcal.npz`，包含 `depth_ndc` 与
   `depth_m`；若缺失，标记 UNKNOWN，不用旧格式文件补齐；
2. **depth 编码有效范围**：调用 `m5_depth_calibration_audit.py`；按已冻结规则，
   5–15m holdout p95 ≤1m 才算 supported，15m 外拒绝；
3. **冻结投影门先于入库**：每站 `after_px_median ≤1.0px`、覆盖 ≥0.8、外观一致率 ≥0.9；
   不降门；任一失败站隔离；
4. **链-贴花分离读数**：只在 supported depth bins 中报告 depth residual；
   不在 15m 外外推；没有足够 depth 点则 UNKNOWN；
5. **不声称**：生成器已修复、跨池族配比已解锁或安全收益，除非站点通过全部门。

所有 UNKNOWN、隔离、启动失败与清理证据原样保留。
