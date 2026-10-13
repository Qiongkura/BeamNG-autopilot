# 更大 LINE=measured 合并池配对训练预注册（2026-10-13）

状态：预注册先于训练。本轮仅扩大 E engine pool 的独立场景数量，保持协议不变。

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* E-large engine：合并两个已通过冻结准入的 engine 池族，共 18 帧：
  * `t16_linepool_20261011/depthcheck_auto_truth_v2/m5auto_a0_{known_line,material_mix,slope_curve}/front_main`；
  * `t16_linepool_large3b_20261013/auto_truth/m5auto_a{1,2,3}_known_line_a*s0/front_main`；
* 输入审计：27 帧、group/content/exposure overlap=0、rejected=0；
* 初始化固定；seeds `{47,48,49,50,51}`；`total_steps=60`、per-run split、batch=4、
  lr=.001、同一 8 个开发包和硬门；
* 逐种子记录 H/E-large line IoU、recall、precision、E−H 差、collapse、UNKNOWN；
* 不改门、不消费最终集、不晋级模型；precision 无分母保持 UNKNOWN。

输出：`logs/experiments/t16_crosspool_large_20261013/`。
