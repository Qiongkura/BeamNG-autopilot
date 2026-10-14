# west_coast 第二批 LINE=measured 扩展预注册（2026-10-13）

* Tech RPC 50000；`west_coast_usa`；`--anchors 5 --sites 1 --scenes known_line`
  `--frames 3 --step-m 2 --anchor-offset 3 --line-convention tiers`；
* 新 offset=3，避开首批 offset=0 的 3 站；目标至少 3 个准入站、9 帧；
* 冻结门：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance；数据 audit group/content/exposure overlap=0；
* 所有道路链失败、材质/annotation 隔离、UNKNOWN、启动失败和 cleanup 状态原样保留；
* 不降低硬门、不消费最终集、不新增人工标注；若目标未达成，按实际结果收口，不补采改参数。
