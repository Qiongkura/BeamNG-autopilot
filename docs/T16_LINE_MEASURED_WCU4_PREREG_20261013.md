# west_coast 第三批独立 LINE=measured 采集预注册（2026-10-13）

* 地图 west_coast_usa，Tech RPC 50000；
* 新 offset=6，`--anchors 5 --sites 1 --scenes known_line --frames 3 --step-m 2`；
* 冻结门：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance；
* 目标至少 2 站/6 帧；道路链不足、材质/annotation隔离、UNKNOWN、启动/cleanup失败
  全部原样保留；不降门、不消费最终集、不新增人工标注。
