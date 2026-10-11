# 负例剂量正式配对实验预注册（2026-10-11）

状态：**预注册，先于本轮任何训练提交**。基于试点
([T16_DOSE_PILOT_RESULT_20261011.md](T16_DOSE_PILOT_RESULT_20261011.md)：两个配对种子上
B−A = +0.041/+0.039，方向支持稀释假说) 冻结正式设计。复核方案阶段二第 6 项：
相同初始化、种子与优化步预算，仅改变一种数据配比。

## 1. 研究问题

**开发集 line IoU 是否随训练池的无线帧占比（负例剂量）单调变化？**
试点只测了 0.20 与 1.00 两档；本轮扫描 4 档剂量画趋势。

## 2. 因子与臂（冻结）

单因子 = 训练池组成（line_frame_share，池内含线像素帧占比）。池子全部来自
R3 已审计的同一批目录的**子集组合**，不新增采集：

| 臂 | 组成 | 帧数 | 线帧占比 |
|---|---|---:|---:|
| d20（锚点=R3 配比） | it3/fisheye + jv/{fisheye,main,pillar_left,pillar_right} | 75 | 15/75 = 0.20 |
| d33 | it3/fisheye + jv/{fisheye,main} | 45 | 15/45 = 0.33 |
| d50 | it3/fisheye + jv/fisheye | 30 | 15/30 = 0.50 |
| d100 | it3/fisheye | 15 | 15/15 = 1.00 |

**如实声明的混杂**：剂量臂的唯一帧数随占比同步变化（15–75 帧）——「组成」与
「独立帧曝光」在本因子定义下不可分（R3/E2 的剂量口径同样如此）；结论按
「负例剂量（占比）」陈述，不拆分两者。

## 3. 共同配置（与试点/R3 baseline 逐项一致）

`m5_train_seg.py --split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001
--device cuda --save-every-epoch --total-steps 60`；
`--init logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
**种子 {42,43,44,45,46}（5 个，跨臂配对）**；
评测 `m5_seg_eval_matrix.py`（无 `--dev-runs`，与试点同口径）对 R3 的 8 个
`reviewed_full` 开发包逐检查点评测，主读数 = 开发集 line IoU（tp/fp/fn 口径）。

## 4. 分析规则（预注册）

* 主表：4 臂 × 5 种子的 dev line IoU，逐种子给出；每臂 mean±std；
* 配对差：d33/d50/d100 相对锚点 d20 的逐种子差值（5 对/臂）与差值均值±std；
* 趋势：剂量单调性按方向一致次数陈述（5 对中正/负的个数），**不做显著性
  显著声称**（n=5，探索性）；
* 次级：recall/precision 形态、per-group 明细原样保留；
* 训练/评测失败与 UNKNOWN 原样保留并单独列出。

## 5. 判据与边界

* **不改任何门、不晋级模型**（交付模型仍为 `seg_model_hand`，sha 19f54e…）；
* 不消费封存最终集；不新增人工标注；训练/评测与其它重活串行；
* 训练/开发隔离沿用 R3 已审计的组隔离证据（同一批目录、同一 dev 集）；
* 结论限定：单地图池族、单一初始化、60 步预算下的配比方向；
  GPU 分钟照记（`gpu_minutes` 台账）。

## 6. 运行身份

输出 `logs/experiments/t16_dose_sweep_20261011/d{20,33,50,100}/seed{42..46}`；
metrics-run `t16_dose_sweep_20261011-d{share}-s{seed}`；评测 JSON
`dev_eval_d{share}_s{seed}.json` 与汇总 `dose_sweep_summary.json`。
