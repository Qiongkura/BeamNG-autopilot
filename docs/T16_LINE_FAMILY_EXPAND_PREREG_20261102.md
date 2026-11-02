# T16 数据族扩量预注册：utah 新锚点 + east（2026-11-02）

## 目的

跨族泛化坍缩（`T16_LINE_SEG_RETRAIN_RESULT_20261102.md`：内部 val 0.77 /
同族 0.30 / 跨族留出 0.0）判决数据广度是绑定约束。用户批准恢复采集。
本预注册覆盖两批扩量：utah 新锚点族 + east 新锚点族，目标把认证族数
从 5 扩到 15+，随后多族重训 + 跨族留出评测。

## 变更因子

- 采集结构：**每站 Scenario 全图统一**（`per_site_scenario_mode` 恒 True；
  east 从全量 Scenario 迁移，a2 right 0.4216 截断签名同族修复；
  GROUND_ALIGN_MAPS 的加载后 ground 重检下沉到每站循环，>2 m 偏离时
  重对齐并重建该站 Scenario）。
- 其余与各图既有预注册逐项一致：utah（`T16_LINE_MEASURED_UTAH_PREREG_20261018`
  线材质 line_yellow_double@0.40m）、east（`T16_LINE_MEASURED_ECA3_PREREG_20261018`
  同材质同宽）；`--anchors 5 --sites 1 --scenes known_line --frames 3 --step-m 2`。
- **冻结准入一字不改**：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance。

## 批次与锚点

| 批 | 地图 | anchor-offset | 目的 |
| --- | --- | --- | --- |
| A | east_coast_usa | 5 | east 新锚点族（绕开 ECA3 已用 offset 0） |
| B | utah | 15 | utah 新锚点族（绕开批次 1-3 的 offset 0/5/10） |

（批 B 顺延在批 A 之后单引擎串行跑；每批完成即导出判读 + 哈希归档。）

## 判据

- 目标：两批合计新增 ≥2 站/≥6 帧 engine_verified（每批各自 ≥1 站）；
  族数从 5 → 15+（每站 3 帧为一族）。
- 隔离站原因逐站保留，不降门；如 east 的 ground 重检触发，重对齐记录入
  `post_load_ground_recheck`。
- 每批产物立即归档到 `logs/evidence_archives/`（哈希清单）。

## 纪束

不降门、不消费最终集、无人工标注、不晋级模型；探针/诊断产物不进池。
