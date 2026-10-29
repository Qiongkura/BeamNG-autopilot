# T16 Utah 贴花渲染截断引擎诊断预注册（2026-10-29）

## 目的与前置

承接 `T16_UTAH_PROJECTION_ATTRIB_20261018.md`（a7a3a9e）：a4s0 双带贴花渲染范围世界固定止于 stn≈15（链 4→33）、a3s0 右贴花对象级全缺、a3s0 左带 stn≈22 后截断。**纯离线证据已排除**：

- 位姿链漂移（2.00m/≤21mm/≤1°）、相机标定（left 实例全程 ~1px 自证）、链-贴花横移（有像素处 ≤0.2m）；
- **埋地/z-clamp**：f2 全分辨率横截面显示 a4s0 远站表面低于链 z 4–8cm（贴花浮置、应当渲染）；全部 5 站 `z_profile` 覆盖 stn 0..34.4（scene_report 实测），链 z 全程跟随坡度（a4s0 5.8cm/m 与引擎表面一致）；
- 距离/mip 截断（截断站位三帧世界固定、不随步进平移）、a0s0 型阴影写入失败（beauty 同缺、形态相反）。

**交叉站型证据**（渲染跨 = 三帧并集、1m 箱密度）：a0s0（坡度最陡 6.94%）双带全跨（L 5..34/R 5..29）；a1s0 全跨（7..36/3..34）；a2s0 L 3..19（密 0.94）/R 4..28；a3s0 L 5..26/R **0**；a4s0 L 4..15/R 4..14。**站点特异性、非站序单调、与坡度无关**。

**两个待判机制**（Utah.zip `roads/items.level.json` 3006 条 DecalRoad 实测）：

1. **节点 z 竞争**（east ECA2 已证同族机制）：覆盖我链的地图叠置对象每站存在（a4s0：AsphaltRoad_variation_01/p9、tread_marks_worn/p11、DirtRoad_variation_01/p1 等；a3s0：dirt_road_flows/p7、m_utah_dirt_variation_04/p7-8），其节点 z 高于链 z +0.02..+0.20m/站——但 a0s0 同样有覆盖叠置高于链 z（21/31 站）却全跨渲染，纯节点 z 比较不判别；
2. **优先级战争**：beamngpy `_get_roads_list` 把场景路 render_priority 强制为插入序 idx 0..9（显式值被丢弃），地图贴花 renderPriority 2–17——a0s0 线 idx 0/1、a4s0 线 idx 8/9，与地图对象的胜负关系未知。

## 变量矩阵（单帧/变体；`scripts/m5_decal_annotation_probe.py --scenes utah`）

场景 **utah_z**（不补丁 → render_priority=idx，即批次行为）：

| 变体 | 线位 | z 提升 | 检验什么 |
| --- | --- | --- | --- |
| ut_r_lift000 / ut_l_lift000 | 站点 lat_right / lat_left | 0 | **V0 单站复现基线**（单站场景是否复现截断/全缺——排除多对象资源效应） |
| ut_r_lift010 / 020 / 035 | lat_right | +0.10/+0.20/+0.35 | **V2 z 竞争序列**（随提升量恢复远端？） |
| ut_l_lift020 | lat_left | +0.20 | 左线对照 |

场景 **utah_prio**（`_restore_render_priority` 补丁后，显式优先级生效）：

| 变体 | 线位 | priority | 检验什么 |
| --- | --- | --- | --- |
| ut_r_p030 / ut_r_p060 | lat_right | 30 / 60 | **V1 优先级战争**（高于地图最大 17） |
| ut_l_p060 | lat_left | 60 | 左线对照 |

- 材质统一 `line_yellow_double`@0.40m（utah 实测唯一 annotation 可写，MAP_PROFILES）。
- 站点：`--site-index 4`（a4s0）与 `--site-index 3`（a3s0）各一轮：

```
.venv/Scripts/python scripts/m5_decal_annotation_probe.py --map utah \
  --site-report logs/m5_seg/collect_e5line_utah1_20261018/scene_report.json \
  --site-index 4 --material line_yellow_double --scenes utah
# 同命令 --site-index 3
```

## 判读（预注册判据）

读回 = 每变体 `S_BINS=(4-8, 8-14, 14-22, 22-34)` 的 solid2px/solid6px/label2px/paint2px（`_analyze` 同款）：

- **判据 A（单站复现）**：utah_z 的 lift000 在 22-34 bin label2px=0 且 14-22 显著低于 4-14 → 批次截断在单站复现（非多对象资源效应）；若 lift000 全跨渲染 → 修复方向=减少单 Scenario 路对象数/拆分场景。
- **判据 B（z 竞争）**：lift 序列随提升量**单调恢复** 14-22/22-34 bin → 机制=节点 z 竞争；修复=线节点 z 取实际覆盖面上沿+余量（生成器内做、真值同源同步）。**约束**：paint2px 相对 lift000 塌陷 >50% 的 lift 档不得作为修复依据（east lift_015 教训：抬升可能伤 beauty）。
- **判据 C（优先级）**：补丁后 p030/p060 恢复远端 bin 而 lift000 不恢复 → 修复=优先级还原入库（beamngpy 补丁随生成器走）。
- **判据 D（对象级）**：A 复现且 B/C 均不恢复 → 引擎贴花网格生成/距离语义，另立变量组（材质对照/宽度/节点数/反序）。
- **a3s0 右全缺**：lift000 右线 0 像素复现 → lift/prio 任一恢复即归因相应机制；均不恢复 → 对象创建/路幅问题（该侧地面为土肩，beauty 无漆）。
- 每轮同时记录 annotation 与 beauty 双 pass（label2px 与 paint2px 分开判，不混用）。

## 纪束

- 诊断输出仅落 `logs/m5_seg/diag_decal_probe2_utah_*/`；**不进训练池、不进导出池、不改任何冻结门、不晋级模型、不消费最终集**。
- 本诊断不重采资格批次；修复判决后另走修复提交（代码模块）+ 按 `T16_LINE_MEASURED_UTAH_PREREG` 同协议重采（冻结门一字不改）。
- 探针变体路不写真值导出（capture 仅诊断读回）；probe_report.json 落盘为诊断证据，原样保留。
