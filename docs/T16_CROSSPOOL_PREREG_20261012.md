# 第二线帧池族跨池配对实验预注册（2026-10-12）

状态：**预注册，先于本轮训练提交**。依据多站点验证
`T16_DEPTH_SEPARATION_MULTISITE_RESULT_20261012.md`：新 engine-verified 池 9 帧、
旧 human-revision 池确定性抽样 9 帧，二者 train/content/exposure overlap=0。

## 因子与臂

单因子 = **线帧池族身份**；帧数、初始化、步预算、种子、评测集全部冻结：

| 臂 | 输入 | 帧数 | truth 来源 |
|---|---|---:|---|
| H（human） | `t16_crosspool_old_subset_20261012/front_main`（从 e2_it3 15 帧按固定索引抽 9） | 9 | human_revision |
| E（engine） | `t16_linepool_20261011/depthcheck_auto_truth/m5auto_a0_{known_line,material_mix,slope_curve}/front_main` | 9 | engine_verified |

旧池子集索引冻结为原池按文件排序的 `[0,1,2,6,7,8,12,13,14]`，只改变池族，不改变
每臂帧数；子集复制只生成在 logs 下，保留原 source_id/identity。

## 共同配置与评测

* `m5_train_seg.py --split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001
  --total-steps 60 --device cuda`；
* 初始化：`logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* 配对种子 `{42,43,44,45,46}`；
* `m5_seg_eval_matrix.py` 同一 8 个 reviewed_full 开发包、同一 line IoU/recall/precision 口径；
* 主差值：E−H，逐种子、mean±std、正向数；崩坏种子诊断字段保留但不改变原 verdict。

## 边界

* 探索性 n=5；池族身份同时包含 truth provenance / 场景构成语义，不能解释为单纯
  “人工 vs 引擎质量”或泛化结论；
* 不消费最终集、不改门、不晋级模型、不新增人工标注；训练/评测串行；
* 任一输入审计失败、UNKNOWN 或训练失败原样保留，实验不补 0。

输出：`logs/experiments/t16_crosspool_20261012/{H,E}/seed{42..46}`，
`dev_eval_{H,E}_s*.json` 与 `crosspool_summary.json`。
