# 更大真实凭证池通道掩码训练回归预注册（2026-10-13）

状态：**预注册，先于本轮训练提交**。这是通道资格掩码真实小训练之后的更大池回归，
不改变初始化、评测集、硬门或优化预算。

## 输入池

`logs/experiments/t16_autotruth_big_20260928/` 中 3 个 engine-verified 场景族：

* `m5auto_a0_material_mix_a0s4/front_main`
* `m5auto_a1_slope_curve_a1s3/front_main`
* `m5auto_a27_known_line_a27s0/front_main`

数据审计：24 帧、24 trainable、0 rejected、0 group/content/exposure overlap（池内审计）。
每个目录有真实 `annotation.json` + `truth_provenance`；LINE/ROAD/SHOULDER/ROLE
资格状态由训练器从凭证读取，UNKNOWN/N-A/unverified 不进入对应监督。

## 冻结训练/评测

* 初始化：`t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`；
* seeds `{47,48,49,50,51}`；
* `m5_train_seg.py --split per-run --val-frac 0.2 --epochs 3 --batch 4 --lr 0.001
  --total-steps 60 --device cuda`；
* 评测：同一 8 个 reviewed_full 开发包、同一 line IoU/recall/precision 口径；
* 不消费最终集、不改硬门、不晋级模型；
* 逐种子保存 `train_hist.channel_qualification`、checkpoint `train_args`、指标、
  collapse 与 UNKNOWN；
* 目标输出：`logs/experiments/t16_largepool_20261013/`。

## 预注册判据

* 5/5 训练/评测成功才报告总体；失败格 UNKNOWN，不补 0；
* 主读数逐种子 line IoU、mean±std、collapse indices；
* 该回归验证的是“通道资格掩码在更大真实凭证池上能运行且分母不被污染”，不是
  provenance 优越性或驾驶收益实验。
