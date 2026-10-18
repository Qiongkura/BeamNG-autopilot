# west_coast 第四批 LINE=measured 配对训练结果（2026-10-13）

预注册：`docs/T16_WCU5_CROSSPOOL_PREREG_20261013.md`（commit `2ee89cf`）。
W5=9帧 west_coast（offset9、a0/a2/a3、engine_verified、与 WCU4 零内容重叠）、
H=9帧 human；5 seeds、60 steps、同初始化/8开发包/硬门；
10作业0失败、UNKNOWN=0、collapse=0。

## W5−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | −0.1345±0.0568 | 0/5 | **champion_better** | 0 |
| recall | −0.0817±0.1022 | 1/5 | inconclusive | 0 |
| precision | −0.2094±0.1562 | 0/5 | **champion_better** | 0 |

逐种子 line-IoU 差值：−0.1578、−0.1446、−0.1925、−0.0402、−0.1374。
UNKNOWN=0；collapse=0；原始 paired verdict 未被任何排除诊断替换。

## 结论边界

第四批 west_coast 9帧准入池在本冻结协议下同样没有显示学习收益：IoU 与 precision
均为 Human 对照更高（0/5 正向），recall 不确定。与 WCU3/WCU4 方向一致——小池
高方差、预算/内容交互主导，不支持地图或 engine 来源的普遍排序。不晋级模型、
不改硬门；准入、真值来源和训练收益保持分离。
