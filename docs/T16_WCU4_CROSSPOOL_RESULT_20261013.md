# west_coast 第三批 LINE=measured 配对训练结果（2026-10-13）

预注册：`docs/T16_WCU4_CROSSPOOL_PREREG_20261013.md`（commit `f45eeb1`）。
W4=9帧 west_coast LINE=measured、H=9帧 human；5 seeds、60 steps、同初始化/8开发包/硬门；
10作业0失败、UNKNOWN=0。

## W4−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | −0.1072±0.0960 | 1/5 | inconclusive | 0 |
| recall | −0.0844±0.1158 | 1/5 | inconclusive | 0 |
| precision | −0.1600±0.0850 | 0/5 | **champion_better** | 0 |

逐种子 line-IoU 差值：−0.0715、−0.1413、−0.2340、+0.0279、−0.1171。
UNKNOWN=0；collapse=0；原始 paired verdict 未被任何排除诊断替换。

## 结论边界

第三批 west_coast 9帧准入池没有在本冻结协议下显示学习收益；precision 还显示
Human 对照更高。准入、真值来源和训练收益保持分离：不晋级模型、不改硬门，结果只
支持“west_coast 小池高方差/内容依赖”，不支持地图或 engine 来源普遍排序。
