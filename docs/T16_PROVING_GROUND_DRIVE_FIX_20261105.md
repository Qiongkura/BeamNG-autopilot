# T16 试车场驾驶打通：三处根因修复后行驶 29.4m（2026-11-05）

## 结果

GridMap 纯地面试车场（decal 双线、right_lane、无铺装），同一场景连续
修复后：

| 阶段 | 位移 | vmax | stop 帧 | 关键现象 |
| --- | --- | --- | --- | --- |
| 修复前（`gridmap_plain_drive`） | 0.02m | 0.01 | 34/38 | placed=True 但永远不走 |
| + stuck/climb 修复（`prev_brk`、`+inf` 净空） | 2.69m | 2.92 | 22/38 | 起步 2.7m 后右线消失、永久停车 |
| + 候选合并半径 0.5m | 8.14m | 2.11 | 15/37 | 右线时有时无 |
| + 填充率门修正（minAreaRect） | 21.23m | 6.35 | 9/42 | 右线持续存在，但周期 `reject=side` |
| **+ 场景道路中线 route_ref** | **29.44m** | **7.01** | **3/42** | **0 次 reject，42 帧仅 2 帧 lane 不可用** |

## 三处根因与修复

### 1. 起步自锁（已修，`44d0451`）

- `_counts_as_stuck` 收到的是 controller **原始** `brk`（目标速度 > 车速时
  恒 0），"刹车释放重叠"过滤等于没过滤；应传上一帧**实际下发**的
  `prev_brk`。
- 无障碍道路 `path_grid_clearance_m = +inf`，旧代码 `np.isfinite(inf)`
  为 False，`climb` 永不触发；改为 `_climb_clearance_ok`（接受 +inf）。
- 效果：位移 0.02→2.69m，vmax 0.01→2.92m/s。

### 2. 右线整条被 `too_sparse` 误丢（已修，本次 `vision` 提交）

- 逐帧遥测 + 离线复现：运动 2m 后右线的学习掩码组件仍在（area 4332、
  bbox 259×178），但轴对齐 bbox 填充率 **9.4% < 10%**，被
  `_mask_to_markings` 判成稀疏噪声整条丢弃。
- 斜向长线在自身 bbox 里天然稀疏；改用**最小外接旋转矩形
  （minAreaRect）**复核：线状组件在旋转矩形内致密（保留），空心方框/
  散点仍被拒。
- 效果：右线在运动中持续存在（离线帧 12/13 两条线均保留），
  位移 8.1→21.2m。

### 3. `reject=side`：路线参考退化成"过自车直线"（已修，本次 `fsd_drive` 提交）

- 逐帧遥测：`mode=pair` 但 `lane_reject=side`。strict 侧向门要求本车道
  中心在参考线右侧 ≥0.2m；无地图路线时参考线退化成 **过自车的直线**，
  车道中心与参考线重合 → 必然拒绝。
- 修复：`--scenario-json` 模式下用场景 JSON 的 `mid/dir` 生成**道路中线**
  局部参考（只作 `route_ref`，不设 `nav_route`，避免 snap 把车拉回道路
  中心）。修复后 `reject` 归零。

## 负结果

- `BEAMNG_THIN_MERGE=1`（thin 共线碎段合并）**无效**：右线不是多个
  thin 碎段，而是整条被 `too_sparse` 丢弃，合并拿不到输入；该轮 stop
  反而升到 26。保留为默认关的研究开关。

## 判定

- 试车场 decal 车道**已可连续行驶约 29.4m**（decal 全长 30m 量级），
  0 次侧向拒绝；距离/速度/拒绝三项均有逐帧遥测证据。
- 仍非"完成品"：速度呈 stop-go（3 次安全停车，v 在 0–7 m/s 间波动），
  条件矩阵（TOD×天气）尚未跑；`--scenario-json` 路线参考是试车场
  专用逻辑，生产默认未变。
- 新默认行为（填充率门）已影响所有地图的候选提取：需用 T16 回归 +
  全量 pytest + 固定 4 折离线评测确认无回归（本提交后执行）。

## 产物

- `logs/m5_seg/gridmap_stuckfix_drive.log`
- `logs/m5_seg/gridmap_merge05_probe.{log,json}`
- `logs/m5_seg/gridmap_fillfix_probe.{log,json}`
- `logs/m5_seg/gridmap_routeref_probe.{log,json}`
- `logs/m5_e2e/shadow_fsd_1793888174_20261105_221651.npz`（42 帧）
- 单测：`tests/test_lane_fill_gate.py`、`tests/test_fsd_drive_helpers.py`
