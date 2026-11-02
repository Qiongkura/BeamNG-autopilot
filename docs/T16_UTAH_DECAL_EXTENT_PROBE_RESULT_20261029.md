# T16 Utah 贴花渲染截断探针结果：单 Scenario 多路对象截断（2026-10-29）

## 执行

按 `T16_UTAH_DECAL_EXTENT_PROBE_PREREG_20261029.md` 两轮，`scripts/m5_decal_annotation_probe.py`
（utah 化 144c38c、第二轮 70f0c32），`--map utah --site-index 4 --material
line_yellow_double`，产物 `logs/m5_seg/diag_decal_probe2_utah_20261029_234338`（第一轮
utah/utah_prio）与 `..._234831`（第二轮 utah_18/utah_full），probe_report.json 落盘。

## 第一轮（预注册变体）判读

- **utah_z 基线未复现截断**：lift000（road-edge 线位 ±4.07=站点 lat_left/right 夹紧位）
  在 22-34 bin solid 7/11——单站场景全跨渲染。**判据 A 的"截断复现"不成立**。
- **lift 序列无效且有害**：lift000→035 随提升 annotation 与 paint 同步变差
  （all-bin solid 22→16、paint 26→20），复现 east 的"抬升伤渲染"教训。
- **优先级反向**：补丁还原后 p030/p060 的 annotation 大幅塌陷
  （all-bin solid 7-8/27-28 对 lift000 的 16-22）——显式高优先级**更差**，
  优先级战争假说排除（beamngpy idx 0..N 的默认行为反而是观测里较好的）。

## 第二轮（批次忠实变体，±1.8 批次线位）判读

| 场景 | 结构 | 左线 solid/30 | 右线 solid/30 | 22-34 bin |
| --- | --- | --- | --- | --- |
| utah_18 | 单站 2 路（idx 0/1） | 30/30 | 29/30 | 左 11/11、右 11/11 |
| utah_full | 全 5 站×2 线（批次结构，目标 idx 8/9） | 30/30 | **12/30** | 左 11/11、右 **0/11**（14-22 也仅 2/8） |

**判据 A 判决：批次截断在"批次路对象结构"下复现、在单站结构下消失**——
机制 = **单 Scenario 多路对象**（后创建的路渲染截断、beauty 同步塌陷），
不是站点几何、线位、坡度、节点 z、优先级、材质或多站资源位置的效应。
与批次站型证据一致（a0/a1 全跨、a2 部分、a3 左 26/右 0、a4 左 15/右 14，
随插入序劣化）；与 east ECA3 的"a2 right coverage 0.4216"同签名。

## 修复（scripts 提交 53850c1）

- `per_site_scenario_mode(map_name)`：非 GROUND_ALIGN_MAPS 地图（italy/utah…）
  采集改为**每站一个 Scenario**（2 条贴花路，探针实测该结构全跨渲染），
  逐站 build→load→start→capture；GROUND_ALIGN_MAPS（west/east）的加载后
  ground 重检依赖全量 Scenario 已加载，保持原流程、另行迁移。
- 回归测试 `test_per_site_scenario_mode_excludes_ground_align_maps`；
  全模块 pytest 41 通过。冻结导出门一字未改。

## 重采预注册

见 `T16_LINE_MEASURED_UTAH2_PREREG_20261029.md`：唯一变更因子 = 采集结构
（每站 Scenario），站点/线位/材质/线宽/帧数/步进/冻结门与
T16_LINE_MEASURED_UTAH_PREREG 逐项一致；探针产物不进训练池。

## 纪束

- 探针两轮均为诊断运行：产物仅落 diag_* 目录，不进训练池、不进导出池、
  不改门、不晋级模型、不消费最终集；probe_report.json 原样保留。
