# 第四 LINE=measured 池配对训练结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_E4_PREREG_20261013.md`（commit `788dcfb`）。
H=9 human frames、E4=6 engine LINE=measured frames；同初始化、5 seeds、60 steps、
同8开发包；10作业0失败、UNKNOWN=0。

## E4−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---|---|
| line IoU | −0.2222±0.2925 | 1/5 | inconclusive | E4 seed49/51 |
| recall | −0.1959±0.3420 | 2/5 | inconclusive | E4 seed49/51 |
| precision | −0.4295±0.3911 | 0/5 | inconclusive | E4 seed49/51 |

逐种子 line-IoU 差值：−0.3069、−0.0426、−0.4863、+0.1930、−0.4683。
排除 collapse 后的诊断均值仅作附加读数：IoU −0.0522、recall +0.0031、precision −0.1491；
不替换原始 paired verdict。

## 结论边界

第四池的 LINE=measured 资格链通过，但 6 帧小池训练在两个种子上 collapse，整体没有
可确认收益；这进一步证明“资格通过”不等于“学习收益”，小池/seed 方差仍主导。
不晋级模型、不改硬门、不把 collapse 排除后结果当主结果。
