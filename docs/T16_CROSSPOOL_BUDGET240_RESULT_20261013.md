# 跨池 240-step 独立预算重复结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_BUDGET240_PREREG_20261013.md`（commit `6276f8c`）。
产物：`logs/experiments/t16_crosspool_budget240_20261013/`；10 个训练 + 10 个评测作业，
line IoU/recall 无失败、无 UNKNOWN；precision 全部缺少有效正例分母，按 UNKNOWN 保留。

## 1. 配对结果（E−H）

| 指标 | 逐种子差值（47–51） | mean±std | 正向 | verdict | UNKNOWN |
|---|---|---:|---:|---|---:|
| line IoU | −0.4514, −0.3877, −0.4379, −0.4288, −0.3473 | **−0.4106±0.0426** | **0/5** | **champion_better** | 0 |
| recall | −0.4651, −0.3981, −0.4493, −0.4398, −0.3549 | **−0.4214±0.0447** | **0/5** | **champion_better** | 0 |
| precision | — | — | — | `needs_evidence` | **5/5** |

240 steps 下 Engine 池在 line IoU 和 recall 上五个种子全部低于 Human 池；precision
没有有效分母，未补零或猜测。

## 2. 解释边界

* 这是在相同池、初始化、种子和评测集下仅改变预算的独立预算重复；它进一步确认
  Engine 池优势不是稳定主效应；
* 240-step 的巨大反向差异与早先 30/60/120 扫描一致地表明，小池、训练预算和
  初始化交互主导结果；
* 不修改硬门、不晋级模型，不把 precision UNKNOWN 当失败或通过；
* 结论限定为当前两个 9 帧池族和单一初始化，不能外推到来源质量或一般训练规律。
