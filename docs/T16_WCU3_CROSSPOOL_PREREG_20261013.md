# west_coast 第二批 LINE=measured 配对训练预注册（2026-10-13）

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* W3 west_coast：`t16_linepool_wcu3b_20261013/auto_truth/m5auto_a0_known_line_a0s0/front_main`
  + `m5auto_a1_known_line_a1s0/front_main`，6 帧，LINE/ROAD measured；
* 初始化固定：`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；60 steps、per-run split、batch=4、lr=.001、cuda；
* 同一 8 个 reviewed_full 开发包、同一评测口径和硬门；
* 逐种子记录 W3−H line IoU/recall/precision、collapse、UNKNOWN；precision 无分母保持 UNKNOWN；
* 不改硬门、不消费最终集、不晋级模型。

输出：`logs/experiments/t16_wcu3_crosspool_20261013/`。
