# utah 资格采集重采预注册（2026-10-29，每站 Scenario 修复后）

* **本轮唯一变更因子**：采集结构 = **每站一个 Scenario**（`per_site_scenario_mode`，
  scripts 53850c1）——utah 探针两轮判决单 Scenario 10 条贴花路会让后创建的路
  渲染截断（`T16_UTAH_DECAL_EXTENT_PROBE_RESULT_20261029.md`）。其余与
  T16_LINE_MEASURED_UTAH_PREREG（2026-10-18）逐项一致：`--map utah --anchors 5
  --sites 1 --scenes known_line --frames 3 --step-m 2 --anchor-offset 0`、
  线材质 line_yellow_double@0.40m、线位 symmetric（默认）、Tech RPC、
  同一采集/导出/审计代码路径（除每站 Scenario 结构本身）；
* **冻结准入一字不改**：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance；两处已知非门证据照旧记录（严格外观掩码对
  utah 偏暗渲染欠覆盖；导出 loose paint 判据的红土假阳性/阴影不不变性已在
  a0s0/a3s0 定因记录在案）；
* 预期改善（判据，不作门改动）：批次截断站（a2/a3/a4）的贴花渲染范围应恢复
  全跨（探针单站结构 22-34 bin solid 11/11 实测）；a3 right 应从 0 恢复；
  若仍隔离 → 隔离原因逐实例保留，不降门、不预防性放宽；
* 目标 ≥2站、≥6帧 engine_verified；锚点耗尽/道路链失败/隔离/UNKNOWN/启动与
  cleanup 失败全部保留；探针与诊断产物不进训练池；
* 首次启动挂起处置沿用原批次协议：杀进程后同命令重试一次，attempt 日志保留。
