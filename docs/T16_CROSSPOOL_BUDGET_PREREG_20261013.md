# 跨池步预算扫描预注册（2026-10-13）

状态：**预注册，先于本轮训练提交**。依据第二次跨池重复不复现（E−H IoU
−0.0574±0.2356，2/5 正向，两个 engine collapse seed）。本轮只改变总优化步数，
判断不稳定是否由 60-step 预算不足/过度造成。

## 冻结因子

* H：`logs/experiments/t16_crosspool_old_subset_20261012/front_main`，9 帧；
* E：`logs/experiments/t16_linepool_20261011/depthcheck_auto_truth/m5auto_a0_{known_line,material_mix,slope_curve}/front_main`，9 帧；
* 初始化：`logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* 配对种子：`{47,48,49,50,51}`；
* 开发评测：与两轮跨池完全相同的 8 个 `reviewed_full` 包、line IoU/recall/precision 口径；
* 唯一因子：`total_steps ∈ {30, 60, 120}`。其余 split、val-frac、batch、lr、
  训练入口和硬门不变。

## 预注册读数

每个预算 × 臂 × 种子逐项记录 line IoU、recall、precision；每个预算逐种子记录
E−H 差值、正向种子数、mean±std、paired verdict、collapse indices 与 UNKNOWN/失败。
不把排除 collapse 后的诊断均值替换原始 paired verdict。

## 输出与边界

输出根目录 `logs/experiments/t16_crosspool_budget_20261013/steps{30,60,120}/`
及 `budget_summary.json`；30 训练 + 30 评测串行运行，GPU 分钟照账。
探索性结果，不晋级模型、不改硬门、不消费最终集、不新增人工标注；若某预算失败，
该格保持 UNKNOWN，不补 0。
