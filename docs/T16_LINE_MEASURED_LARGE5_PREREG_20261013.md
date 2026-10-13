# 第五 LINE=measured 独立池族预注册（2026-10-13）

* Tech 50000；Italy；5 anchors × 1 known_line site × 3 frames；`step_m=2`；
  `anchor-offset=140`；
* 冻结准入：投影 median≤1px、coverage≥0.8、appearance≥0.9、engine_verified
  truth_provenance；失败/隔离/UNKNOWN原样记录；
* 目标至少 3 站准入、9 帧；
* 通过后保持既定初始化、5 seeds `{47..51}`、60 steps、同8开发包、硬门做配对训练；
* 不消费最终集、不改门、不新增人工标注。

## 修订 A（2026-10-13）

offset=140 超出确定性 anchor 数量（合格 140、跳过后 0），失败原样保留；改为
`anchor-offset=130`（剩余 10 个 anchor），其它参数不变。

## 修订 B（2026-10-13）

offset=130 的 5 个候选道路链均为 10–19m，无法布站；失败原样保留。改试
`anchor-offset=110`，其它参数完全不变；若仍无合格站点，本批次按地图道路网络容量
边界收口，不再改变 offset 盲扫。
