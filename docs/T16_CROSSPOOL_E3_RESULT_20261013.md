# 第三 LINE=measured 池配对训练结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_E3_PREREG_20261013.md`（commit `4c49880`）。
产物：`logs/experiments/t16_crosspool_e3_20261013/`；10 作业全部完成、0 失败、
IoU/recall/precision UNKNOWN=0。

## 逐种子结果（E3 engine measured − H human）

| seed | H IoU | E3 IoU | Δ IoU | Δ recall | Δ precision |
|---:|---:|---:|---:|---:|---:|
| 47 | 0.3523 | 0.5529 | +0.2006 | +0.2504 | −0.0457 |
| 48 | 0.5168 | 0.5918 | +0.0751 | +0.0751 | +0.0199 |
| 49 | 0.4956 | 0.5816 | +0.0860 | +0.1412 | −0.0998 |
| 50 | 0.3386 | 0.5003 | +0.1617 | +0.1821 | −0.0174 |
| 51 | 0.4264 | 0.4732 | +0.0468 | +0.0330 | +0.0684 |
| **mean±std** | — | — | **+0.1140±0.0644** | **+0.1364±0.0859** | **−0.0149±0.0639** |

* IoU：5/5 正向，paired verdict=`candidate_better`，collapse=0；
* recall：5/5 正向，paired verdict=`candidate_better`，collapse=0；
* precision：2/5 正向（−,+,−,−,+），paired verdict=inconclusive，
  seeds_needed_for_effect=74；UNKNOWN=0。

## 结论边界

第三 LINE=measured 池在本次 9 帧池、单初始化、60 steps、5 seeds 下比固定 H human
子集显示稳定的 IoU/recall 方向，但 precision 不改善且方差/池内容仍混杂；这不是
provenance 的普遍排序，也不是驾驶收益或模型晋级证据。与 H2 全零池、第二重复不复现
和预算扫描反向结果合并阅读：**池内容/资格/训练配方交互仍主导**，本结果只证明
第三池可以进入后续更大、更多独立池的配对研究。
