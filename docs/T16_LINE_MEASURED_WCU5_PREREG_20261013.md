# west_coast 第四批独立 LINE=measured 采集预注册（2026-10-13）

* 地图 west_coast_usa，Tech RPC 50000；
* `--anchors 5 --sites 1 --scenes known_line --frames 3 --step-m 2 --anchor-offset 9`；
* 冻结准入：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance；
* 目标≥2站、≥6帧；道路链失败、隔离、UNKNOWN、启动/cleanup失败全部保留；
* 不降门、不消费最终集、不新增人工标注。
