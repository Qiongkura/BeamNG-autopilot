# 第四 LINE=measured 池配对训练预注册（2026-10-13）

状态：预注册先于训练。输入为第四池 offset100 采集后通过冻结准入的两个 known_line 站、
共 6 帧；隔离站点不进入训练。

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* E4 engine：`t16_linepool_large4b_20261013/auto_truth/m5auto_a0_known_line_a0s0/front_main`
  + `m5auto_a3_known_line_a3s0/front_main`，6 帧；
* 初始化固定：`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；
* 60 steps、per-run split、val-frac=.2、batch=4、lr=.001、cuda；
* 评测固定同一 8 个 reviewed_full 开发包，line IoU/recall/precision 同口径；
* 记录逐种子指标、E4−H 差值、collapse 与 UNKNOWN；不改硬门、不晋级模型、不消费最终集。

输出：`logs/experiments/t16_crosspool_e4_20261013/`。
