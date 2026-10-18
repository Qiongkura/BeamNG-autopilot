# west_coast 第五批 LINE=measured 配对训练结果（2026-10-13）

预注册：`docs/T16_WCU6_CROSSPOOL_PREREG_20261013.md`。
W6=3帧 west_coast（offset12、a2、engine_verified、采集目标 ≥2站/≥6帧未达成——
冻结 gate 仅准入 1 站，4 站隔离证据保留；与全部既往导出池+H 零内容重叠）、
H=9帧 human；5 seeds、60 steps、同初始化/8开发包/硬门；
10作业0失败、UNKNOWN=0、collapse=0。
H 臂复现性（事后更正）：仅 seed47 与 WCU5 配对逐位一致，seed48–51 有
运行间差异（IoU 最大 0.0356@seed50、recall 最大 0.0375、precision 最大 0.0094，
GPU 非确定性）——同一次配对运行内的双臂对比不受影响（同条件同轮训练），
但**跨实验比较带有该噪声下限**；初稿「完全一致」的说法以 seed47 单点外推，
已在本更正中撤回。

## W6−H

| 指标 | mean±std | 正向 | verdict | collapse |
|---|---:|---:|---:|---:|
| line IoU | +0.1161±0.1195 | 4/5 | inconclusive | 0 |
| recall | +0.1229±0.1372 | 4/5 | inconclusive | 0 |
| precision | +0.0195±0.0467 | 3/5 | inconclusive | 0 |

逐种子 line-IoU 差值：+0.2638、−0.0177、+0.0205、+0.2066、+0.1072。
UNKNOWN=0；collapse=0；原始 paired verdict 未被任何排除诊断替换。

## 结论边界

第六批是六个 west_coast 池中**首个均值方向为正**的池（IoU/recall 4/5 正向），
但 5 种子下三项均 `inconclusive`（IoU 95% CI 半宽 0.148 > 均值 0.116），
不构成 `candidate_better`。这与前五批（24/25 配对为负、3/5 `champion_better`）
合看，进一步说明：**小池内容、种子与预算交互主导**本研究，6–9 帧量级的单池
方向可以整体翻转而无稳定来源排序。前置条件注意：本批是唯一 3 帧池（WCU 首批
亦为 3 帧，方向为负），且 a2 站与既往 a2 同索引不同 offset。
不晋级模型、不改硬门；准入、真值来源和训练收益保持分离。
