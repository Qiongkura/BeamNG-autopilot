# 更大真实凭证池通道掩码训练回归结果（2026-10-13）

预注册：`docs/T16_LARGEPOOL_MASK_REGRESSION_PREREG_20261013.md`（commit `a5d85da`）。
训练/评测目录：`logs/experiments/t16_largepool_20261013/`；5 个种子，
5 训练 + 5 评测，0 失败。

## 1. 真实通道资格统计

每个 seed 的 `train_hist.json` / checkpoint 都记录相同统计（训练共 240 帧，
3 个 engine-verified 场景族、24 原始帧在 per-run 切分后）：

```json
{
  "frames": 240,
  "masked_frames": 240,
  "ROAD": {"measured": 240},
  "LINE": {"unverified_labels": 240},
  "SHOULDER": {"unknown": 240},
  "ROLE": {"unverified_labels": 240}
}
```

这是真实 `annotation.json`/`truth_provenance` 状态的实际消费：LINE 没有被偷偷当作
背景负例；SHOULDER/ROLE 没有被升级成可监督目标；ROAD measured 正常参与损失。

## 2. 开发集读数

5 个种子的 line IoU 和 recall 均为 0.0；precision 五个种子全部 UNKNOWN（没有有效
正例分母）。因此没有计算伪造的均值/差值，也没有把全零臂判作 collapse——collapse
规则在臂中位数为零时不触发。

这不是模型收益实验，而是一个重要的真实凭证回归：**当真实凭证把 LINE 标成
unverified 时，训练与评测会停止消费 LINE 监督，结果明确显示为未测/无有效检出，
而不是制造一个“低质量 line 模型”数字。**

## 3. 边界

* UNKNOWN/precision 缺失原样保留；不补 0、不晋级模型、不改硬门；
* 该大池的现有凭证 LINE 状态是 unverified，因此不能用于证明 engine pool 的学习收益；
* 若要获得可评测 LINE 回归，必须先取得 LINE=measured 的独立凭证池，或完成人工/独立
  验证链，不能通过训练参数绕过资格掩码。
