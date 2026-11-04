# T16 试车场真值链复测：配对模式修好，几何宽度仍未对齐；驾驶缺可行驶参考（2026-11-04）

## 问题

2026-11-04 首测：GridMap + decal 车道，严格+sensor 放置 rc=3。此前的
解释是"右线碎段/取景裁切"。本轮做分层诊断，回答三件事：

1. 到底是感知没看见线，还是候选/配对层把线丢了？
2. 换铺装能不能修？
3. 放置成功后车能不能在试车场开（条件矩阵的前提）？

## 1. 根因：不是模型，是经典白漆臂把干净的两条线并碎

对已保存的 live 帧做离线分层（`gridmap_f1_r0.frames/frame_00005.npz`）：

| 层 | 结果 |
| --- | --- |
| 纯学习掩码（`predict`） | **2 条 solid，34/33 点，世界跨度 ~30m** |
| 纯掩码过 `_mask_to_markings` | 2 条，`gate_line_candidates` 全保留 |
| `line | cv_white`（生产路径） | **0 marking**：浅色地面被经典亮度臂当白漆，与两条线并成无效大块 |
| 经典黄色臂（drive 的兜底） | 2 条 1m 碎段（`cv_yellow_px` 仅 161） |

live 里最终进配对的正是那两条 1m 黄色碎段：重叠
`s_hi-s_lo = 1.0m < LANE_PAIR_OVERLAP_M=1.5m`，配对循环**静默跳过**
（所以 `pair_rejects={}`），镜像又因两条黄线的中心线互斥只剩
`mirror_left`。放置要求双侧配对，故 rc=3。

## 2. 修复 A：`BEAMNG_CV_WHITE=0`（研究开关，默认 1 不变）

关闭经典白漆臂后，候选只来自学习掩码（+黄色臂）。离线同一帧：

- 2 条 solid，世界跨度 **39.0/32.4m**，`pair` 模式，宽度按车体系 ~3.95m。

live 探针（12 ticks，`gridmap_cvoff_r0`）：

| 指标 | 值 |
| --- | --- |
| 配对模式 | **pair 12/12**（无镜像） |
| cert 线召回 | left 0.833 / right 1.000 / both 0.833 |
| 配对判定 | true 0 / phantom 1 / **ambiguous 11** |
| `result.width_m` | **4.83–4.88m**（GT 3.96m） |

**结论：配对模式与候选跨度修好了，但几何宽度仍偏宽 ~0.9m**，探针因此
判 ambiguous 而不是 true。这不是"模式"问题，是 `_pair_world_geometry`
（世界系按站匹配）与车体系 `med_lat` 读数不一致：同一对线车体系宽
3.95m、世界系 4.88m。未解决前，这条真值不能作为条件矩阵的参考。

补充定位（同一帧、离线、`BEAMNG_CV_WHITE=0`）：

- GT 右线车体系 `gt_right_lat = -1.7996m`（自车确实在车道中心）；
- 检测：右线 `-2.36m`（向外偏 0.56m）、左线 `+1.59m`（向内偏 0.21m），
  整对向右偏 0.385m、宽 3.95m；
- 左线是 178 点的 `thin` 折线，其世界 x 在近场从 500.2 漂到 499.2
  （GT 为 499.3 的直线）；右线 34 点 `solid` 世界 x≈503.4 基本笔直。
  即**左线近场回投被拉向自车**，导致世界宽度沿线变化
  `pair_width_spread=2.91m`（近 3.2m、远 4.3m）。
- 两条世界线本应平行（同一平面、同一方向）；一条直一条斜 → 近场
  回投/候选折线有问题，不是 `_pair_world_geometry` 的数学问题（对
  合成平行线它返回 3.6m，对近端内弯 0.9m 的线仍返回 3.59m）。

## 3. 修复 B（已否决）：铺 20m 深色沥青

- `AsphaltRoad_variation_01` 在 gridmap_v2 **不存在**，引擎回退成橙色
  棋盘（首轮 `gridmap_pave_r0`，cand=0）；
- 换 gridmap 自带 `m_road_asphalt_rough_01` 后场景正常，live 探针
  12/12 true（两轮：pair 7/mirror 5、pair 6/mirror 6），cert_both 0.583；
- drive 严格+sensor **placed=True（本环境首次）**；
- 但铺装纹理仍触发经典白漆臂（`cv_white_px≈7k`），线被切成 3–9m
  碎段；线抬高 2cm 不改变（`gridmap_pave3_r0` 与 `pave2` 同）；
- 因此铺装不是干净解，`proving_lane` 默认已改回**不铺装**
  （`pave_width_m` 机制保留为可选）。

## 4. 放置成功≠能开：试车场缺可行驶参考

`placed=True` 的那轮（`gridmap_pave_drive.log`）车 20s 内几乎不动：
37 帧 21 次停车，`throttle=0.27` 与 `brake=1.0` 交替，
`src=none reason=no drivable path` 周期出现；shadow 轨迹
`trajectory_ok=False` 的帧 13/37。

直接控制对照（`_gridmap_engine_probe.py`，诊断脚本不入库）：
同一场景 ego 在 `throttle=0.6, gear=2` 下 0→10 m/s 约 2s，
位置从 y=500 走到 y=510。**车、发动机、挡位、地面都正常**。

结论：停车来自**规划/安全层在空 GridMap 上没有可行驶证据/路线**，
不是试车场或车辆故障。条件矩阵若在空地上跑，绝大多数格会是停车，
指标退化成"放置成功率"，不能回答"什么条件下还能开"。

## 5. 判定（fail-closed）

- 试车场 decal 真值链**已能出双侧 pair，但宽度未对齐 GT**：
  条件矩阵**暂不启动**（不拿未验证的参考去跑批）。
- 默认行为未变：`BEAMNG_CV_WHITE` 默认 1、`proving_lane` 默认不铺装、
  配对门限未动；所有诊断开关都需显式开启。
- 驾驶侧需要**可行驶参考**：把 decal 车道铺到 GridMap 自带道路上，
  或给试车场一条导航路线；否则只能做"静止放置+配对"条件实验。

## 6. 下一步（按优先级）

1. **修世界系宽度**：在 live 旋转帧上对比 `_pair_world_geometry` 与
   车体系 `med_lat`；重点查左右线按站匹配时的近场点（相机姿态/地面
   平面误差会被放大成 ~0.9m 宽度偏差）。修好前试车场只作诊断场。
2. **可行驶参考**：在 GridMap 自带道路（有路面 class、无标线）上铺
   同一 decal 车道，复测 placed + 能开；或实现试车场直线导航路线。
3. **静止条件实验**（不依赖驾驶）：TOD×天气下只测 placement 与
   pair/cert 率，12 格 × 短探针；需先预注册。
4. 文档与提交：本轮代码提交见 `75502b0`/`8e5b48b`/`2f928b8`/
   `f639477`/`a6ce37f`。

## 产物

- `logs/m5_seg/gridmap_cvoff_r0.{json,jsonl,probe.log,frames/}`
- `logs/m5_seg/gridmap_pave2_r0.*`、`gridmap_pave3_r0.*`
- `logs/m5_seg/gridmap_pave_drive.log`、`gridmap_pave_drive.jsonl`
- `logs/m5_seg/_gridmap_engine_probe.log`（直接控制对照）
- 掩码叠加：`_gridmap_mask_frame5.png`；铺装帧：`_gridmap_pave3_frame5.png`
