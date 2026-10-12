# 跨池族配对第二独立重复预注册（2026-10-13）

状态：**预注册，先于本轮训练提交**。第一轮结果见
`docs/T16_CROSSPOOL_RESULT_20261013.md`：5/5 配对正向，E−H line IoU +0.1003±0.069。
本轮只做独立 seed 重复，不改变因子、数据、评测集、初始化、预算或硬门。

## 冻结协议

* 臂 H：`logs/experiments/t16_crosspool_old_subset_20261012/front_main`，9 帧；
* 臂 E：`logs/experiments/t16_linepool_20261011/depthcheck_auto_truth/m5auto_a0_{known_line,material_mix,slope_curve}/front_main`，9 帧；
* `m5_train_seg.py`：`--split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001
  --total-steps 60 --device cuda`；
* 初始化固定：`logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* 新配对种子：**{47,48,49,50,51}**；
* `m5_seg_eval_matrix.py`：同一 8 个 reviewed_full 开发包、同一 line IoU/recall/precision
  口径；不消费封存最终集；不改阈值、不晋级模型。

## 预注册读数

逐种子记录 H/E 三个指标、E−H 差值、UNKNOWN/评测失败；主读数 line IoU。
报告 mean±std、正向种子数与 collapse 诊断（复用 `PairedResult`，不改变原 verdict）。
5 个新种子为探索性独立重复；若方向不复现，不能把第一轮结果写成稳定池族效应。

输出：`logs/experiments/t16_crosspool_repeat_20261013/{H,E}/seed{47..51}`、
`dev_eval_{H,E}_s*.json`、`crosspool_repeat_summary.json`。
