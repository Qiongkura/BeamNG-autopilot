# 第五 LINE=measured 独立池族预注册（2026-10-13）

* Tech 50000；Italy；5 anchors × 1 known_line site × 3 frames；`step_m=2`；
  `anchor-offset=140`；
* 冻结准入：投影 median≤1px、coverage≥0.8、appearance≥0.9、engine_verified
  truth_provenance；失败/隔离/UNKNOWN原样记录；
* 目标至少 3 站准入、9 帧；
* 通过后保持既定初始化、5 seeds `{47..51}`、60 steps、同8开发包、硬门做配对训练；
* 不消费最终集、不改门、不新增人工标注。
