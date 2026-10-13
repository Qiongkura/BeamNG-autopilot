# 第三个独立池跨池配对预注册（2026-10-13）

状态：**预注册，先于本轮训练提交**。前两轮跨池结果受小池/预算交互影响：
第一轮 E−H +0.1003，第二轮 −0.0574，240-step −0.4106。本轮只替换 H 池族，
其余初始化、评测、种子和硬门全部冻结。

## 因子与输入

* H2（新 human pool）：`annotate_pkg_e1_jv_20260927/front_main`，固定排序抽帧索引
  `[0,1,2,4,5,6,8,9,10]`，复制到 `logs/experiments/t16_crosspool_h2_20261013/front_main`；
  9 帧、Johnson Valley、`human_revision`；
* E（既有 engine pool）：多站点验证后准入的 9 帧三站 engine pool，路径不变；
* 输入审计必须确认 group/content/exposure overlap=0；任何冲突/UNKNOWN 原样保留，
  不补 0。

## 共同配置

`m5_train_seg.py --split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001
--total-steps 60 --device cuda`；初始化固定为
`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
配对种子 `{47,48,49,50,51}`；评测固定为同一 8 个 reviewed_full 开发包；硬门不变。

## 预注册读数

逐种子记录 H2/E 的 line IoU、recall、precision，E−H2 差值、正向数、collapse
indices 与 UNKNOWN；复用 `PairedResult`，不以排除 collapse 后均值替换原始 verdict。
不把 H2 的 human provenance 或 E 的 engine provenance 解释成普遍质量排序，结论限定
为这两个小池在冻结预算下的探索性差异。

输出：`logs/experiments/t16_crosspool_h2_20261013/`，
`dev_eval_{H2,E}_s*.json`、`h2_summary.json`。
