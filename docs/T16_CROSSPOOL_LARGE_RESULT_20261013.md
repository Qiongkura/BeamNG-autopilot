# 更大 LINE=measured 合并池配对结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_LARGE_PREREG_20261013.md`（commit `3d64dd0`）。
E-large 是两个已准入 engine 池族合并，共18帧；H固定9帧；5 seeds、60 steps、
同初始化/8开发包/硬门。10作业0失败/UNKNOWN。

## E-large − H

| 指标 | 逐种子差值 | mean±std | 正向 | verdict | collapse |
|---|---|---:|---:|---|---:|
| line IoU | +0.0823, −0.1295, −0.0130, +0.1863, −0.0038 | +0.0245±0.1177 | 2/5 | inconclusive | 0 |
| recall | +0.0969, −0.1507, −0.0198, +0.2171, −0.0040 | +0.0279±0.1376 | 2/5 | inconclusive | 0 |
| precision | −0.0182, +0.0459, +0.0237, −0.0325, −0.0002 | +0.0037±0.0316 | 2/5 | inconclusive | 0 |

UNKNOWN：所有三项 0/5；collapse：0/5。

## 结论

扩大 E 池到 18 帧后，E-large−H 的 IoU/recall 仍不能稳定区分（2/5 正向、原始
paired verdict inconclusive）；均值接近零而方差较大。相较第三池小规模结果的
+0.114，增加池族后优势消失，进一步支持“池内容/种子/预算交互”而非来源主效应。
不晋级模型、不改变硬门，不把该结果解释为 engine 或 human 来源排序。
