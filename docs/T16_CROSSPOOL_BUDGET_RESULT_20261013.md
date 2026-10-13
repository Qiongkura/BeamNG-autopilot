# 跨池族步预算扫描结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_BUDGET_PREREG_20261013.md`（commit `cf74188`）。
产物：`logs/experiments/t16_crosspool_budget_20261013/`；30 训练 + 30 评测作业，
0 失败、0 UNKNOWN。

## 1. 配对 line IoU：E（engine）−H（human）

| total_steps | 逐种子差值（47,48,49,50,51） | mean±std | 正向 | verdict |
|---:|---|---:|---:|---|
| 30 | −0.179, +0.133, −0.131, −0.091, −0.210 | **−0.0957±0.1354** | 1/5 | inconclusive |
| 60 | +0.159, −0.382, −0.035, +0.142, −0.196 | **−0.0622±0.2297** | 2/5 | inconclusive |
| 120 | −0.159, −0.353, −0.144, −0.355, −0.057 | **−0.2139±0.1340** | **0/5** | **champion_better** |

Collapse diagnostic：steps30 标出 H seed47；steps60 标出 E seed48/51；steps120 标出
E seed50。排除 collapse 后的诊断均值不替换预注册原始 verdict。

## 2. 次级读数

* recall：steps30 −0.1000±0.1321（1/5）、steps60 −0.0711±0.2379（2/5）、
  steps120 **−0.2337±0.1455（0/5，champion_better）**；
* precision：steps30 −0.0360±0.0877（1/5）、steps60 −0.0007±0.0667（2/5）、
  steps120 +0.0429±0.0543（4/5，inconclusive）。

## 3. 结论

1. **步数解释了第二轮不稳定的一部分**：60 steps 的 E−H 高度异质（collapse 2 对）；
   120 steps 下 engine 池在 line IoU/recall 上 0/5 正向，human 池稳定更好；
   30 steps 也偏向 human 但噪声仍大。
2. 不能把 engine 池的首轮 +0.1003 写成池族固有效应：在相同池族、同种子、只改变
   total_steps 的扫描中，engine 臂没有一个预算得到可靠的正向 IoU 结论；120 steps
   反而给出 `champion_better`。
3. 该结果支持“**小池 + 初始化 + 步预算交互主导**”而不是简单的 truth provenance
   主效应。所有结果仍为探索性，模型不晋级，不改硬门。
