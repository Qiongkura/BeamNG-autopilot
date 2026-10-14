# west_coast 第二批 LINE=measured 配对训练结果（2026-10-13）

预注册：`docs/T16_WCU3_CROSSPOOL_PREREG_20261013.md`（commit `7532a6e`）。
H=9 human，W3=6 west_coast LINE=measured；5 seeds、60 steps、同初始化/8开发包/硬门；
10作业0失败、IoU/recall/precision UNKNOWN=0。

## W3−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | −0.2374±0.1786 | 0/5 | champion_better | W3 seed49/51 |
| recall | −0.2499±0.1808 | 0/5 | champion_better | W3 seed49/51 |
| precision | −0.1263±0.3158 | 3/5 | inconclusive | W3 seed51 |

排除 collapse 后仅作诊断均值：IoU −0.1137、recall −0.1263、precision +0.0146；
不替换原始预注册裁决。

## 边界

west_coast 第二批新增池通过准入，但小池配对训练在 IoU/recall 上 0/5 正向，且两个
seed collapse；资格通过不等于学习收益。模型未晋级、硬门未改、UNKNOWN=0 原样记录。
