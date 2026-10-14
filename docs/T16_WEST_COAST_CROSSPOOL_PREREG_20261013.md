# west_coast LINE=measured 配对训练预注册（2026-10-13）

状态：预注册先于训练。west_coast 修复后首个准入池为 3 帧，隔离站点不纳入。

* H human：`logs/experiments/t16_crosspool_old_subset_20261012/front_main`，9 帧；
* W west_coast engine：`logs/experiments/t16_linepool_wcu_final_20261013/auto_truth/m5auto_a0_known_line_a0s0/front_main`，3 帧；
* 初始化：`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；60 steps、per-run split、batch=4、lr=.001、cuda；
* 同一 8 个 reviewed_full 开发包、同一 line IoU/recall/precision 口径和硬门；
* 逐种子记录 W−H 指标差值、collapse、UNKNOWN；precision 无分母保持 UNKNOWN；
* 不消费最终集、不改硬门、不晋级模型；小池结果只作独立地图资格/探索性读数。
