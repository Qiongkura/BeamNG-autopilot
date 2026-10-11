# 配对配比试点预注册（阶段二第 6 项第一试，2026-10-11）

状态：**预注册，先于本轮任何训练提交**。依据 [复核方案](PROJECT_NEXT_DEVELOPMENT_REVIEW_20261011.md)
§4-D（学习收益未成立）与 §5 阶段二第 5/6 项。R3 实测：75 训练帧中 60 帧来自
verified 无线目录（`line_pixels=0`），候选 IoU 0.480→0.370/0.367 被拒。

## 1. 研究问题

**R3 候选 IoU 下降是否主要由训练池的无线帧占比（15/75 = 0.20 线帧）导致？**

单因子 = 训练池组成；其余全部冻结为与 R3 baseline 逐项一致。

## 2. 臂定义（冻结）

| 臂 | 训练池 | 帧数 | 线帧占比 |
|---|---|---:|---:|
| A（R3 配比复现） | `annotate_pkg_e2_it3_20260927/front_fisheye` + `annotate_pkg_e1_jv_20260927/{front_fisheye,front_main,pillar_left,pillar_right}` | 75 | 0.20 |
| B（仅线帧池） | `annotate_pkg_e2_it3_20260927/front_fisheye` | 15 | 1.00 |

## 3. 共同配置（与 R3 baseline 命令逐项一致）

`--split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001 --device cuda
--save-every-epoch --total-steps 60`；种子 **{42, 43}（配对）**；
`--init logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
训练入口 `scripts/m5_train_seg.py`（R3 round0 同款 argv）。

评测：R3 的 8 个 `reviewed_full` 开发包，`scripts/m5_seg_eval_matrix.py`
（循环同口径）；主读数 = **开发集 line IoU**（总体 + per-group），按种子配对
报告 B−A 差值；R3 baseline（0.48025）作为 Arm A 的复现参照（同配置应近似复现）。

## 4. 预注册判据与解释边界

* 报告逐种子 dev line IoU 与配对差值；**任何方向都不改门、不晋级模型**；
* 2 种子/臂为探索性试点：只读方向与幅度量级，不做显著性声称；
* 已知并如实记录：Arm B 仅 15 帧（val 3 帧）有过拟合风险；per-run 切分下两臂
  的 val 组成不同（属配比因子的一部分）；
* 评测失败/UNKNOWN 原样保留；不消费封存最终集；不新增人工标注。

## 5. 运行身份

输出目录 `logs/experiments/t16_dose_pilot_20261011/{ArmA,ArmB}/seed{42,43}`；
每作业记录 `train_hist.json` 的 source_sha256/commit；评测 JSON 同目录保存。
