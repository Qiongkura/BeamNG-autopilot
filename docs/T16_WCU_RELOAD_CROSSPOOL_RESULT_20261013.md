# west_coast 修复后准入池配对训练结果（2026-10-13）

预注册：`docs/T16_WCU_RELOAD_CROSSPOOL_PREREG_20261013.md`（commit `98630f7`）。
Wfix=6帧 west_coast、H=9帧 human；5 seeds、60 steps、同初始化/8开发包/硬门；
10作业0失败、UNKNOWN=0。

## Wfix−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | **−0.2211±0.0614** | 0/5 | **champion_better** | 0 |
| recall | **−0.2035±0.0459** | 0/5 | **champion_better** | 0 |
| precision | **−0.2382±0.1866** | 0/5 | **champion_better** | 0 |

逐种子 IoU 差值：−0.2139、−0.2639、−0.2745、−0.1200、−0.2334。

## 结论边界

west_coast 修复后准入链已通过，但 6 帧小池的配对训练在 IoU/recall/precision 五个
种子全部低于 Human 对照；这不是安全或来源质量结论，而是小池内容/初始化/训练配方
下的探索性结果。模型未晋级、硬门未改；准入通过与学习收益明确分开。
