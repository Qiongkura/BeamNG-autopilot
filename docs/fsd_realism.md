# FSD 底层拟真要求(FSD Realism Invariants)

本文件是项目的**底层硬性要求**:任何与"FSD 一样的能力"冲突的实现都必须按此整改。
目标不是模拟器里能跑就行,而是逻辑结构上等于真实 FSD(传感器 → 向量空间 → 规划 → 安全 → 影子闭环)。

## 1. 车道保持与道路边界:100% 来自感知

- **禁止**:用地图/导航路线做"车道中心/车道边界"(DecalRoad、A* 折线、HD-map 车道几何)。
- **必须**:车道中心 = 语义分割线/路缘 + LiDAR 自由走廊配对出的**本车道**中心;
  道路边界 = 感知掩码投影到 BEV 的 drivable 边界。
- 代码: `beamng_autopilot/lane/perception_guard.py`(横向纠偏,无任何 map 导入)、
  `lane/pairing.py + lidar.py + fusion.py`(感知车道)。
- 状态: ✅ shadow_drive 已用感知纠偏;❌ fsd_stack 默认模式仍把地图车道当硬护栏(见 §4)。

## 2. 地图/导航只负责"去哪",不负责"路在哪"

- 导航路线(route)是**目的地意图**,允许用于:选路、路口方向门控、终点刹停。
- 不允许用于:横向纠偏、车道中心、道路边界。
- 状态: ✅ 已拆分;route 只进 Scene 作为意图。

## 3. 规划 = 向量空间上的规划,不是地图折线

- 规划器输入是 BEV/向量空间(occupancy + drivable + 感知车道),候选轨迹在感知空间采样与评分。
- 地图折线只作为导航意图参与仲裁,不作为唯一参考。
- 状态: ✅ planning/ 分层规划;传感器模式感知车道优先。

## 4. 感知不可用时的降级(严格模式)

- **FSD 行为**:看不见车道时**不**靠高精地图车道继续开,而是降级(减速/停车/谨慎巡航)。
- 本项目默认 `lane_mode="map"` 是**旧规则兼容模式**(标注 non-FSD fallback,不得作为 FSD 验收);
  `lane_mode="sensor" + strict_sensor=True` 才是 FSD 拟真模式:
  无配对感知车道 → `lane_ref=None`,src=`perception-unavailable`,禁止地图车道进入规划。
- 状态: ✅ `FSDStack(strict_sensor=True)` 已实现并单测。

## 5. 传感器真值优先,模拟器特权禁止进入推理

- 推理路径只允许:真实 Camera/LiDAR 传感器、真实物理状态。
- 禁止:Lua 地面真值、annotation 像素、地图查询直接进网络/规划输入(只能用于离线标注/评测)。
- 状态: ✅ 感知管线只吃传感器;annotation 仅用于分割训练。

## 6. 影子闭环

- 数据 = 真实规则驾驶执行 + 感知栈影子预测对齐,坏样本(卡墙/压草)自动剔除。
- 状态: ✅ recording.py 多模态 + 质量门控 + 卡死集剔除。

## 验收

- `pytest tests/test_fsd_realism.py` 必须全过;
- `scripts/m5_fsd_drive.py --lane-mode sensor --strict` 实车必须:无压线、无逆行、无倒车、
  感知不可用时停车/降级而不是骑地图中线。
