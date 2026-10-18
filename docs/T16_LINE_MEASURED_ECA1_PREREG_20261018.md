# east_coast 第一批独立 LINE=measured 采集预注册（2026-10-18）

* 第三张地图 `east_coast_usa`，Tech RPC 50000；
* 材质档案（提交前已入库 + 回归断言）：white=line_white、yellow=line_yellow、
  pave=AsphaltRoad_variation_two_way、gravel=DirtRoad_variation_01、blue=无
  （2026-10-18 全嵌套 `items.level.json` 复扫：line_white 72 站、line_yellow 15 站
  DecalRoad 实例，四材质均有 map 级定义）；
* `--anchors 5 --sites 1 --scenes known_line --frames 3 --step-m 2 --anchor-offset 0`
  （新地图无已消费锚点，offset=0）；
* 冻结准入：projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified truth_provenance（与 italy/west_coast 同门，一字不改）；
* 目标≥2站、≥6帧；锚点耗尽、道路链失败、隔离、UNKNOWN、启动/cleanup失败全部保留；
* 不降门、不消费最终集、不新增人工标注；
* 已知未知项：该地图 engine annotation 对生成 DecalRoad 线的承载能力、
  ground 对齐偏移、材质实际渲染外观均为首次实测——一切以冻结 gate 读数为准，
  失败即记录，不以 west_coast 经验外推。
