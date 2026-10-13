# 不同地图 LINE=measured 受控采集预注册（2026-10-13）

状态：预注册先于采集。Italy 的道路链容量已达到边界，本轮切换到已建档的
`west_coast_usa` MAP_PROFILE（材质来自离线扫描，非 Italy 材质）。

* Tech RPC 50000；`--map west_coast_usa`；`--anchors 3 --sites 1`
  `--scenes known_line --frames 3 --step-m 2 --anchor-offset 0`
  `--line-convention tiers`；
* 输出：`logs/m5_seg/collect_e5line_wcu2_20261013/`；导出
  `logs/experiments/t16_linepool_wcu2_20261013/auto_truth/`；
* 目标 ≥2 站、≥6 帧 LINE=measured；
* 冻结门不变：projection median≤1px、coverage≥0.8、appearance≥0.9、
  `truth_provenance.report.verified=true`；数据审计 group/content/exposure overlap=0；
* 任何材质缺失、道路链失败、projection/coverage/appearance隔离、UNKNOWN、启动失败和
  cleanup 异常原样保留，不降低门、不消费最终集、不新增人工标注。
