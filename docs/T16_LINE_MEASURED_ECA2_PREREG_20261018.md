# east_coast 第二批 LINE=measured 采集预注册（2026-10-18，生成器修复后重采）

* 前置：ECA1 0/5 站准入，根因为 road-network z 低于地形 ~28 m 使生成路链
  埋地（见 `T16_LINE_MEASURED_ECA1_RESULT_20261018.md`）；修复已入库
  （`GROUND_ALIGN_MAPS` 泛化 + post-load 复测重建 + 回归断言）；
* east_coast_usa、Tech RPC 50000、`--anchors 5 --sites 1 --scenes known_line
  --frames 3 --step-m 2 --anchor-offset 0`（与 ECA1 同锚点族，只改生成器修复）；
* 冻结准入一字不改：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance；
* 目标≥2站、≥6帧；锚点耗尽、道路链失败、隔离、UNKNOWN、启动/cleanup失败
  全部保留；不降门、不消费最终集、不新增人工标注；
* 本轮证据应显示：站点 z 已对齐实测地形（`ground_alignment` measured）、
  post-load 复测键为 `post_load_ground_recheck`。
