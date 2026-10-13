# 第三 LINE=measured 大池族采集结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_LARGE3_PREREG_20261013.md`，修订 A/B 记录因道路链
长度约束导致的布站失败。最终采用 5 anchors × 1 known_line site。

## 结果

* 采集 5 个 anchor / 5 个站点 / 15 帧目标；2 个站点被冻结导出门隔离，3 个站点准入；
* 准入池：`known_line_a1s0`、`known_line_a2s0`、`known_line_a3s0`，共 **9 帧**；
* 每个准入池均 `engine_verified`，通道状态：LINE/ROAD=`measured`，
  SHOULDER/ROLE=`unknown`；
* 9 帧均 trainable，数据审计 `rejected=0`、group/content/exposure overlap=0；
* 隔离站点原因原样保留：一个 projection median 1.45px；另一个 coverage 0.59/0.79、
  residual 3.74px、appearance 0.726；没有降低门或纳入训练。

## 结论边界

这是第三个 LINE=measured 独立池族，规模 9 帧，较前一轮 3 帧增加；已满足“更多受控
站点 + 冻结准入 + truth_provenance”要求。它不是学习收益结论；后续可用于配对训练，
但仍需注意 9 帧小池的种子方差。
