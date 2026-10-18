# west_coast 第五批 LINE=measured 配对训练预注册（2026-10-13）

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* W6 west_coast：`t16_linepool_wcu6_20261013/auto_truth/m5auto_a2_known_line_a2s0/front_main`，
  3 帧（engine_verified；采集目标 ≥2 站/≥6 帧**未达成**——冻结 gate 仅准入 1 站，
  a0/a1/a3/a4 隔离原因全部保留于 `export_report.json`，不降门补采替代）；
  跨池审计：全部既往导出池 + H 共 81 帧 81 唯一摘要、0 重叠；
* 初始化、5 seeds `{47..51}`、60 steps、per-run split、batch/lr、8开发包和硬门全冻结
  （与 WCU/WCU3/Wfix/WCU4/WCU5 配对同一协议）；
* 记录逐种子 IoU/recall/precision、W6−H、collapse、UNKNOWN；
* 不改硬门、不消费最终集、不晋级模型。
