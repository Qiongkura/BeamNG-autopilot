# west_coast 修复后准入池配对训练预注册（2026-10-13）

* H human：`t16_crosspool_old_subset_20261012/front_main`，9帧；
* Wfix west_coast：`t16_linepool_wcu_reload_20261013/auto_truth/m5auto_a0_known_line_a0s0/front_main`
  + `m5auto_a1_known_line_a1s0/front_main`，6帧，冻结门通过的 engine_verified LINE/ROAD measured；
* 初始化、评测集、硬门、batch/lr/split 与既有 west_coast 配对一致；
* seeds `{47,48,49,50,51}`、total_steps=60；
* 逐种子记录 IoU/recall/precision、Wfix−H 差值、collapse 与 UNKNOWN；
* 不改硬门、不消费最终集、不晋级模型。
