# 真实通道凭证小规模训练回归（2026-10-13）

目标：验证真实 `annotation.json` / `truth_provenance` 的 LINE/ROAD/SHOULDER/ROLE 状态
实际进入训练损失和评测分母，而不是只通过纯函数测试。

## 运行

* 输入：`logs/experiments/t16_linepool_20261011/depthcheck_auto_truth/m5auto_a0_known_line_a0s0/front_main`
* 真实凭证：`annotation.json`，`label_source=engine_verified`，
  `truth_provenance.report.verified=true`；通道状态为：
  `LINE=unverified_labels`、`ROAD=measured`、`SHOULDER=unknown`、`ROLE=unverified_labels`；
* 训练：CPU、`total_steps=2`、batch 2、全新输出
  `logs/experiments/t16_channel_mask_smoke_20261013/`；未改模型门、未消费最终集。

## 实测结果

`train_hist.json` 和 `checkpoint_last.pt` 都记录：

```json
{
  "frames": 6,
  "masked_frames": 6,
  "statuses": {
    "ROAD": {"measured": 6},
    "LINE": {"unverified_labels": 6},
    "SHOULDER": {"unknown": 6},
    "ROLE": {"unverified_labels": 6}
  }
}
```

* 两个优化步均有有限 loss（0.75996、0.51978）和正梯度范数（3.240、2.610）；
* `val_line_iou` 为 0.0，且不是伪造的低分：LINE 全部为 unverified，LINE 目标和
  IoU 分母被屏蔽；
* ROAD 是 measured，因此仍参与训练/评测；SHOULDER/ROLE 状态被保留为审计信息，
  当前 3 类网络没有对应输出类，不会被伪造为训练目标；
* 训练进程正常结束，checkpoint/hist 的资格统计一致。

## 验证范围

这是真实凭证小训练回归，不是学习收益实验；它证明的是通道资格状态从凭证到
loss/metric 的接线有效。代码提交 `b4b0a41` 已通过全量 pytest 和深度离线门。
