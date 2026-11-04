# T16 SSL 双种子集成部署化评估结果：0.5392 精确复现（2026-11-04）

## 执行

预注册 `T16_ENSEMBLE_DEPLOY_PREREG_20261104.md`。实现部署组件
`EnsembleSegmenter`（双模型概率平均 → 与单模型相同的后处理），评测入口
`--ensemble-deploy`；固定 4 折（HOLD 集与 SSL 复核一致）、模型对
`v28_ssl<fold>`（种子 42）+ `v29_ssl2<fold>`（种子 43）。

## 复现结果（R-Repro）

| 折 | 帧数 | 部署路径 line IoU(近场) | 既有 `--tta ens` 记录 | 差 |
| --- | --- | --- | --- | --- |
| A | 50 | 0.5048 | 0.5048 | 0.0000 |
| B | 18 | 0.6897 | 0.6897 | 0.0000 |
| C | 150 | 0.4232 | 0.4232 | 0.0000 |
| W | 93 | 0.4312 | 0.4312 | 0.0000 |
| **宏平均 A/B/C** | — | **0.539246** | 0.5392 | **+0.00005** |

**R-Repro 通过**（阈值 ≤0.005；逐折差 0）。部署组件与评测路径数学一致，
0.5392 可在部署链路复现。

## 时延（L-Deploy，536×403，20 次 p50）

| 配置 | p50 | mean |
| --- | --- | --- |
| 单模型 | 18.76 ms | 18.65 ms |
| 双模型集成 | 29.00 ms | 29.01 ms |
| 比值 | **1.55×** | — |

（不是 2×：后处理只跑一次，省掉了一份形态学/连通域成本。）

## 安全 gate（S-*）

- **S-FailClosed 通过**：成员抛异常 → `EnsembleSegmenter.predict` 向上抛出，
  不返回掩码；由 stack 转 perception-unavailable → 刹停。单测覆盖。
- **S-Post 通过**：集成结果走 `masks_from_model_probs`（与单模型同一后处理
  链），单测钉住调用路径与概率平均值（0.5/0.5 → 0.5）。
- **S-Default 通过**：评测/驾驶默认不启用集成（`--ensemble-deploy` 显式
  开启；drive 侧未接线）。单测/源检查钉住。

## 部署条件与限制

1. 组件已就绪、复现成立；但**尚未接进 `m5_fsd_drive`**——按预注册纪束，
   live 接线需要单独预注册（放置/驾驶链路、时延预算、双模型显存占用、
   fail-closed 行为审计）。
2. 显存：两份 SSL checkpoint 同时驻留（每份约数十 MB 权重，GPU 显存
   增量 <200MB，实测设备 12GB 足够）。
3. 模型仍不晋级：本结果是"部署候选"评估，不是默认启用。

## 产物

`logs/m5_seg/seg_model_v28_ssl{ABCW}/eval_fold*_deploy.json`（4 份）、
`logs/m5_seg/ensemble_deploy_latency.json`；
组件见 `beamng_autopilot/vision/segmentation.py::EnsembleSegmenter`。
