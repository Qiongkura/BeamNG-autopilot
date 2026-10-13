# LINE=measured 独立真值池与通道掩码训练回归结果（2026-10-13）

## 1. 准入池

输入：`logs/m5_seg/collect_e5line_depthcheck_20261012/`，三个受控站点
（known_line、material_mix、slope_curve），9 帧。

冻结准入结果：

* 投影、覆盖、外观门：3/3 站通过；
* 独立导出：3/3 站、9 帧 `engine_verified`；
* 重导出修复：此前 exporter 在 `verify_batch` 通过后仍把 scene 的 label_source
  写成 `engine_annotation`，造成训练侧 LINE 被错误降级为 unverified；现已改为
  `engine_verified`，并把 `rep.channels` 写入 `truth_provenance.labels.channel_valid_area`；
* 凭证状态（每个站点相同）：`LINE=measured`、`ROAD=measured`、
  `SHOULDER=unknown`、`ROLE=unknown`。

## 2. 真实小训练

输入目录：重导出的 `depthcheck_auto_truth_v2` 三站 front_main；
训练：CPU、seed 23、batch 2、`total_steps=2`、全新输出
`logs/experiments/t16_channel_mask_measured_smoke_20261013/`。

实测 `train_hist.json` 与 `checkpoint_last.pt`：

```json
{
  "frames": 7,
  "masked_frames": 0,
  "ROAD": {"measured": 7},
  "LINE": {"measured": 7},
  "SHOULDER": {"unknown": 7},
  "ROLE": {"unknown": 7}
}
```

* 2 个优化步正常完成；loss=3.0791 → 2.8002；梯度范数 8.503 → 3.230；
* `val_line_iou=0.0` 是小预算/CPU smoke 的真实读数，不是分母屏蔽；LINE=measured
  已进入损失与评测分母；
* SHOULDER/ROLE 仍保留 UNKNOWN，当前 3 类模型没有对应输出头，不伪造监督；
* UNKNOWN/N-A/unverified 掩码路径已由前一轮真实凭证回归验证，本轮验证了
  `LINE=measured` 路径也能正确进入训练。

## 3. 边界

这是资格接线回归，不是学习收益或模型晋级实验；硬门、初始化、最终集使用规则均未改变。
