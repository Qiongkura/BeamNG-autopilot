# 第三 LINE=measured 池配对训练预注册（2026-10-13）

状态：预注册先于训练。因子为第三个 LINE=measured engine 池族 vs 固定 human 池；
其余完全冻结。

* H：`t16_crosspool_old_subset_20261012/front_main`，9 帧 human_revision；
* E3：`t16_linepool_large3b_20261013/auto_truth/m5auto_a{1,2,3}_known_line_*/front_main`，9 帧，
  engine_verified LINE/ROAD measured；
* 输入审计：18 帧，4 个 group，group/content/exposure overlap=0；
* 初始化：`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；
* `total_steps=60`、per-run split、val-frac=.2、batch=4、lr=.001、cuda；
* 评测：同一 8 个 reviewed_full 开发包、line IoU/recall/precision 同口径；
* 不改硬门、不消费最终集、不晋级模型；
* 记录逐种子指标、E3−H 差值、collapse 与 UNKNOWN；precision 缺有效分母时保持 UNKNOWN。

输出：`logs/experiments/t16_crosspool_e3_20261013/`。
