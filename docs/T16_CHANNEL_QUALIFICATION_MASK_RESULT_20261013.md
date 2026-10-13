# 通道资格掩码训练支持（2026-10-13）

提交：`b4b0a41`（训练接线）与前一提交的 loss 支持；开发分支已推送。
验证：全量 `pytest tests/` 通过，深度离线 `m5_offline_validate.py` ALL PASS。

## 1. 数据契约

训练帧现在可携带 `qualification.channels`：

* `LINE`
* `ROAD`
* `SHOULDER`
* `ROLE`

优先读取目录凭证 `annotation.json` / `meta.json` 中的 `channels` 或
`truth_provenance.labels.channel_valid_area`；旧目录没有凭证时保持历史训练路径，
不猜测通道状态。凭证存在但通道缺失时按 UNKNOWN 处理。

允许状态只有按证据解释：`measured` 才能进入对应监督；`unknown`、`not_applicable`
和 `unverified_labels` 均不升级为 PASS。

## 2. 损失与评测行为

当前 3 类网络中可直接映射的通道是：ROAD→class 1、LINE→class 2；SHOULDER/ROLE
状态随帧保留并进入报告，但当前模型没有对应输出类，因此不会被假装成可训练目标。

对于每个 batch：

1. 非 `measured` 的 ROAD/LINE 目标像素改为 `255` ignore；
2. `class_mask[B,C]` 从 softmax 分母、交叉熵和 IoU 分母共同移除对应类；
3. 混合 batch 不再报错：LINE 的 Tversky/clDice 只对 LINE `measured` 的样本计算；
   UNKNOWN/N-A/unverified 样本不贡献区域损失；
4. `train_hist.json` 和 checkpoint `train_args` 记录通道状态计数与被屏蔽帧数。

这修复了旧行为中“某通道未测却仍被当成背景负例”的风险。

## 3. 回归覆盖

新增/更新回归包括：

* 混合 measured/unverified LINE batch 的区域损失可计算且有限；
* UNKNOWN/N-A/unverified 的 LINE/ROAD 目标从损失和评测分母移除；
* SHOULDER/ROLE 状态被保留为通道审计信息；
* 无凭证旧目录不凭空生成资格状态；
* 全量训练、监控和非有限损失回归保持通过。

本批次没有重新训练候选模型，也没有改变硬门或晋级模型。
