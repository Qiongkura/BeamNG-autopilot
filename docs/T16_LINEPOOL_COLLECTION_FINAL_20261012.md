# 修复后线帧池窄化验证结果（2026-10-12）

## 1. 运行身份

* Tech 通过显式 RPC 端口 `50000` 启动；64257–64260 在本机返回 `WinError 10013`
  （Windows 端口权限/保留范围），不是许可证错误。受控入口用 `BEAMNG_TECH_PORT=50000`。
* 代码冻结：`2744b3b`（相机局部光轴修复、浮点深度原始 buffer 回退、depthcal 导出器
  隔离；pytest 3418 + 深度离线 ALL PASS）。
* 入口：修复后的 `scripts/m5_controlled_scenes.py`，Italy，known_line，1 站、3 帧、
  step 2m、offset 24。

## 2. 结果

* 采集成功：覆盖 1.000、线点 60、annotation 线像素 1849、外观证据两实例均 1.0；
  采集侧校准 mean 1.541px / median 1.541px。
* 冻结准入仍拒绝：`after_px_median=1.271px > 1.0px`（逐帧 after mean 1.393px）。
  这是**近门失败**，不是覆盖或外观失败；导出器不把它降格为合格。
* `depthcal` 文件确实生成（1 个）：`frames/known_line_a0s0_depthcal.npz`，包含
  `depth_ndc`、`depth_m`、near/far 与相机 JSON。非共享内存相机的异步 poll 曾返回空 depth，
  现已加入 raw float buffer 回退，并将 keys/types/shapes 写入状态。
* 但本次 depth 数值还**不能作为链-贴花分离的定量证据**：NDC 原始分布含接近 1 的背景
  饱和，按通用 near/far 反解得到约 3.1e6m 的背景值；在 53 个投影真值点上，直接对
  原始 float 做真值距离 affine 拟合 RMSE 约 4.15m、p95 约 8.55m。该深度路径的实际
  编码/处理仍需独立标定，不能把它读成真实深度。

## 3. 结论边界

1. 相机姿态修复确实把一部分问题压低：前一批 material_mix 跨帧残差约 10.2px，
   修复后约 2.5px；但 known_line 窄化站点仍在 1.27px 近门失败。
2. 本轮**没有获得合格的新线真值池**；跨池族配比实验仍未解锁。
3. `_depthcal.npz` 已验证存在与落盘链路；链-贴花深度分离尚未被可靠量化，不能声称
   生成器几何已修复。
4. 下一代码批次应先为 BeamNG 的 non-shared depth buffer 建立真实编码校准（使用已知
   几何点/多距离标定，输出可复算 residual 与有效范围），再决定是否需要改生成器链采样。
   冻结门不放宽。
