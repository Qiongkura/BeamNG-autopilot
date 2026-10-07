# 结果：第二张地图（E4）尝试——被引擎的逐地图标注类阻断

日期：2026-10-07　依据：`docs/paper/REVIEW_20261007.md` 的 E4（「扩展自动训练证据：新道路自动真值确认」）
性质：**负结果**。没有生成可用于第二张地图的线真值；本文的结论是「为什么不能」，且每个数字都是实测。

## 一、动机

论文的局限之一是「所有感知结果来自同一张地图」。审查建议在第二张地图上复测定义性发现
（不需要新增人工标注）。本轮把这条从「待办」推进到「已界定的阻断点」。

## 二、步骤与实测

### 1. 离线扫描：哪些地图有标线贴花（不需要游戏）

扫描 `content/levels/*.zip` 里各地图的 `items.level.json`（材质引用计数）：

| 地图 | 标线类材质 | 备注 |
|---|---|---|
| italy | `italy_road_markings_line_thin`（+ yellow/blue 变体） | 历史批次与封存集所用地图 |
| west_coast_usa | `line_white`(3112)、`line_yellow`(1470)、`line_dashed_short`(215)、`line_dashed_short_yellow`(75) | 标线材质丰富 |
| jungle_rock_island | `line_yellow_damage_02` | 仅一条破损黄线 |
| 其余 17 张（gridmap_v2/smallgrid/utah/east_coast_usa/johnson_valley/…） | 无 | — |

### 2. 生成器加「地图档案」（可复用资产）

`scripts/m5_controlled_scenes.py` 新增 `MAP_PROFILES`：把材质从写死的 italy 名字改成按地图取，
italy 默认值逐位不变（既有批次可复现）；未建档案的地图必须显式给 `--mat-*`，否则报错而不是静默用错材质。
west_coast_usa 档案：铺装 `road_asphalt_2lane`、白 `line_white`、黄 `line_yellow`、蓝 `line_dashed_short`、碎石 `m_dirt_road_gravels`。

### 3. 在 west_coast_usa 上生成受控场景（真跑）

命令：`--map west_coast_usa --scenes known_line known_no_line --anchors 3 --sites 2 --frames 6`

* 锚点自动挑选成功（合格 119 个），投影标定自动完成；
* 生成侧真值正常（每个 known_line 站 60 个真值点）；
* **但逐站验收全部隔离**：`annotation 线类像素 = 0`（6/6 站），
  即引擎的 annotation 通道**不把这些 DecalRoad 判成线类**，标签通道因此为空。

### 4. 原生采集对照（同一张地图，不用生成器）

`m5_collect_seg_ring.py --map west_coast_usa --frames 30 --roles front_main --follow-road --save-annotation`：

| 地图 | 帧数 | 线类像素合计 | 每帧均值 |
|---|---|---|---|
| west_coast_usa（本轮） | 30 | **803** | ≈27 |
| utah（2026-09-27 采集） | 77（正向 1 帧） | **0** | 0 |
| italy（既有采集） | 多批 | 稠密（线类覆盖是既有管线的前提） | — |

## 三、结论

1. **阻断点是引擎的逐地图标注类，不是我们的管线。** 线真值（管线使用的标签通道）在本安装的
   20 张地图里只有 italy 稠密；west_coast_usa 每帧约 27 个线类像素（比 italy 低两个数量级，
   远低于认证门要求的逐实例 0.8 覆盖），utah 为 0。
2. **因此第二张地图需要另一条真值通道**（在该地图上做人工标注，或改引擎的标注类映射），
   而不是「重跑一次生成器」。这一点现在是**实测结论**，写进了论文的局限与未完成工作。
3. 本轮留下的可复用资产：地图档案机制、一次被正确隔离的批次
   （`logs/experiments/t16_scenes_wcu_20261007/`，6 站全隔离 + 原因）、
   一次原生采集（`logs/m5_seg/collect_e4_wcu_20261007/`，30 帧）。
4. 收尾：本轮泄漏的 9 个 BeamNG 进程已按项目纪律全部关闭（`game_pids()` 归零）。

## 四、对论文的改动

局限「单一仿真器、单一地图」与未完成工作第 8 条改写为实测口径：三张地图的线类像素计数、
「第二张地图需要另一条真值通道」的结论，以及生成器已具备地图档案这一可复用条件。
