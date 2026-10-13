# 第三个独立池跨池配对结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_H2_PREREG_20261013.md`（commit `ac38003`）。
执行：H2 human pool 9 帧 vs E engine pool 9 帧；5 seeds、60 steps、10 训练 + 10 评测，
0 训练失败、0 IoU/recall UNKNOWN；precision 5/5 UNKNOWN（无有效正例分母）。

## 结果（E−H2）

| 指标 | 逐种子差值 | mean±std | 正向 | verdict |
|---|---|---:|---:|---|
| line IoU | 0, 0, +0.0332, 0, +0.0262 | +0.0119±0.0164 | 2/5 | inconclusive |
| recall | 0, 0, +0.0352, 0, +0.0262 | +0.0123±0.0171 | 2/5 | inconclusive |
| precision | — | — | — | needs_evidence |

H2 human 臂的 5 个开发集 line IoU 全部为 0；engine 臂只有 seed49/51 出现非零（0.0332/
0.0262），所以这个独立池试验没有提供可确认的池族收益，只说明该 H2 小池/训练配方
本身无法在当前预算下学到可评测的线输出。collapse 诊断未标出异常对，因为 H2 的中位数
本身为 0；不能把全零臂伪装成 collapse 后收益。

## 边界

* 这是第三个独立池输入，不把 H2 的全零结果解释为 human provenance 的质量结论；
  采样、池大小、场景内容和单初始化均有限；
* E/H2 仍不改变硬门、不晋级模型；precision UNKNOWN 原样保留；
* 综合前三次跨池/预算结果：没有稳定的 engine 池优势，池内容和预算交互明显强于
  provenance 单因子；后续应扩大池/增加独立场景，而不是继续用小池宣称来源排序。
