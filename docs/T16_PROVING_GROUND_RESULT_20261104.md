# T16 GridMap 无标线试车场首测：镜像配对 20/20 true（2026-11-04）

## 动机

五锚点三图全部被地图标线污染，live 贴花参考不可建立。Phase 2 方案：
在 **GridMap**（天然无标线平地）用 decal 程序化铺一条 7m 宽车道 + 两条
0.4m 线（right_lane 约定，余量 0.6），让合成贴花成为画面里唯一的漆。

## 执行

- 地面探测：GridMap (0,0)/(500,500) 等点 ground_z≈100.0；原点有巨墙遮挡，
  改用 **(500,500)** 空旷区。
- 场景：`logs/m5_seg/gridmap_scene.json`（mid (500,500,100)、dir (0,1)、
  半宽 3.5）；材质覆盖 `line_yellow_double`（白/黄）+ `AsphaltRoad_variation_01`
  （pave）+ `m_dirt_road_gravels`（gravel）。
- 探针：`m5_pairing_probe --map gridmap_v2 --scene gridmap_scene.json
  --save-frames`；两轮：默认 span 门 vs S1（`--arm span`，thin span→1.0）。

## 结果

| 配置 | ticks | 配对 | true | phantom | 模式 | collect 丢弃 |
| --- | --- | --- | --- | --- | --- | --- |
| 默认 span 门 | 20 | 0 | 0 | 0 | none | `thin:span` 40（2/轮） |
| **S1（span→1.0）** | 20 | **20** | **20** | **0** | `mirror_left` 20 | `thin:span` 20（1/轮） |

- 贴花**确实渲染**：画面中可见两条橙/黄色线（GridMap 白色地面上），
  说明材质覆盖方案有效、无地图标线污染。
- 候选提取把线碎成 **0.92–1.09m** 段（`collect_stats`），默认 1.5m span
  门全拒；S1 放开到 1.0m 后**左侧线**进入候选，配对以 `mirror_left` 形成
  车道，中心/宽度与 GT 匹配（true 20/20、phantom 0）。
- **右侧线仍未成为候选**（right recall 0.0）：右线位于画面右下角边缘，
  近端被画面裁切、可检段更短，span 仍 <1.0m。这是候选层/取景问题，
  不是配对门问题。

## 判读

1. **无标线试车场可行**：GridMap + 材质覆盖 + decal 车道可以建立
   "贴花唯一漆源"的 live 参考；S1 下配对中心与 GT 一致（20/20），
   这是本环境首次 live 参考可归因。
2. **限制**：目前是单侧（镜像）配对；右侧线候选召回 0（画面裁切/碎段）。
   要实现双侧 `pair` 与 placement 的严格确认，需要：
   - 车辆/相机相对车道居中调整（右线不入画面边缘），或
   - 候选碎片合并（S1 预注册里提到但未实现的共线合并）。
3. 该试车场下一步：把场景构建接进 `m5_fsd_drive`（`--scenario-json`），
   即可在无标线场地跑条件矩阵（TOD×天气），不再受地图标线污染。

## 产物

`logs/m5_seg/gridmap_probe2_r0.*`（默认）、`gridmap_probe2_span_r0.*`（S1）、
`logs/m5_seg/gridmap_scene.json`；探针新增 `--mat-*` 覆盖与
`m5_marking_replay --paint-mode line`。
