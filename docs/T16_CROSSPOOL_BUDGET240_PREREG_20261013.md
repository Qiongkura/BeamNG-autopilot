# 跨池 240-step 独立预算重复预注册（2026-10-13）

状态：**预注册，先于本轮训练提交**。基于 30/60/120-step 扫描中 Engine 臂
未显示稳定优势，本轮只增加一个未测试预算 `total_steps=240`，不改变池、初始化、
种子或评测集。

## 冻结项

* H：`logs/experiments/t16_crosspool_old_subset_20261012/front_main`，9 帧；
* E：`logs/experiments/t16_linepool_20261011/depthcheck_auto_truth/m5auto_a0_{known_line,material_mix,slope_curve}/front_main`，9 帧；
* 初始化：`logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；
* 评测：相同 8 个 reviewed_full 开发包、line IoU/recall/precision 口径；
* 唯一变化：`total_steps=240`；
* 不改硬门、不消费最终集、不晋级模型、不新增人工标注。

## 预注册输出

逐臂逐种子保存训练曲线/checkpoint、评测 JSON；汇总 H/E 指标与 E−H 差值，
记录正向种子数、paired verdict、collapse indices、UNKNOWN 和失败。collapse
诊断只作为附加报告，不替换原始配对 verdict。

输出：`logs/experiments/t16_crosspool_budget240_20261013/`，
`budget240_summary.json`；结果限定为单池族、单初始化、240-step 探索性重复。
