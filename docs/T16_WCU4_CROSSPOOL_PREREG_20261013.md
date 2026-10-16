# west_coast 第三批 LINE=measured 配对训练预注册（2026-10-13）

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* W4 west_coast：`t16_linepool_wcu4_20261013/auto_truth/m5auto_a0_known_line_a0s0/front_main`
  + `m5auto_a2_known_line_a2s0/front_main` + `m5auto_a3_known_line_a3s0/front_main`，9 帧；
* 初始化、5 seeds `{47..51}`、60 steps、per-run split、batch/lr、8开发包和硬门全冻结；
* 记录逐种子 IoU/recall/precision、W4−H、collapse、UNKNOWN；
* 不改硬门、不消费最终集、不晋级模型。
