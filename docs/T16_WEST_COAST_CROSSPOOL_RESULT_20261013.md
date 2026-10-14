# west_coast LINE=measured 配对训练结果（2026-10-13）

预注册：`docs/T16_WEST_COAST_CROSSPOOL_PREREG_20261013.md`（commit `aa16cdf`）。
H=9 human 帧、W=3 west_coast engine LINE=measured 帧；同初始化、5 seeds、60 steps、
8 开发包、硬门；10 作业0失败、IoU/recall/precision UNKNOWN=0。

## W−H 结果

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | −0.1709±0.1497 | 0/5 | inconclusive | 0 |
| recall | −0.1753±0.1430 | 0/5 | inconclusive | 0 |
| precision | −0.0590±0.2680 | 1/5 | inconclusive | W seed49 |

逐种子 IoU 差值：−0.0473、−0.3410、−0.3266、−0.0497、−0.0901。
precision 排除 collapse 后诊断均值 +0.0606，但不替换原始 verdict。

## 结论边界

west_coast 的 LINE=measured 准入链已验证，但 3 帧小池配对训练没有显示收益：
IoU/recall 五个种子全部低于 Human 对照。不能把地图、来源或材质写成普遍排序，
模型不晋级，硬门不变；结果作为独立地图资格与小池高方差证据保留。
