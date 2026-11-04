# T16 离线 SSL 双种子集成部署化评估 · 预注册（2026-11-04）

## 1. 目标

SSH 双种子集成的固定 4 折结果（0.5148→0.5392，+4.7%）目前只存在于评测脚本的
`--tta ens` 路径。本任务把集成做成**可部署组件**（`EnsembleSegmenter`，与
`Segmenter` 同一接口子集），在该部署路径上复跑固定 4 折，确认 0.5392 可复现；
同时给出时延与安全 gate 的结论。**不做默认启用、不晋级模型**。

## 2. 部署组件定义

- `EnsembleSegmenter(model_paths=[m1, m2])`：
  - `predict(frame)`：两模型各自 `probability_maps_small` → 算术平均 →
    `masks_from_model_probs`（与单模型完全相同的后处理）；
  - `detect_lines`：用集成掩码委托给第一个成员的提取器（几何/标定不变）；
  - `reset`、`model_path`（可追溯的复合名）、`last_timing_ms`、`route_is_dirt`
    兼容语义头；
  - **fail-closed**：任一成员异常向上抛出（由 stack 转成 perception-
    unavailable → 刹停），禁止"一个成功就用一个"的静默降级。
- 默认不在 drive 入口启用；研究/评测显式传参。

## 3. 复现评测（固定 4 折，配对）

- 折：A/B/C/W，HOLD 集与 SSL 复核完全一致（训练∩留出=0 已核验）。
- 模型对：`v28_ssl<fold>`（种子 42）+ `v29_ssl2<fold>`（种子 43）。
- 指标：`line IoU(近场)`，宏平均=A/B/C。
- **R-Repro**：部署路径宏平均与 0.5392 的差 **≤0.005**；逐折与 ens 记录差 ≤0.01。
- 若不一致 → 记录部署路径与评测路径的差异（不允许改判据凑数）。

## 4. 时延与安全 gate

- **L-Deploy**：536×403 单帧 `predict` 的 p50；集成应为单模型约 2×，
  报告实测值（不设硬阈值，供部署预算参考）。
- **S-FailClosed**：注入一个抛异常成员 → `predict` 必须抛，不得返回掩码；
  单测覆盖。
- **S-Post**：集成走与单模型相同的 `masks_from_model_probs`（后处理链一致），
  单测钉住调用路径。
- **S-Default**：drive/eval 默认路径不启用集成（None），单测钉住。

## 5. 纪束与交付

- 不训练、不晋级、不改默认门、不新增标注、不消费最终集；
- 交付：本预注册、`EnsembleSegmenter` + 单测、部署路径 4 折结果、时延数据、
  `T16_ENSEMBLE_DEPLOY_RESULT_*.md`、证据 zip。
- 即使复现成功，集成是否进 drive 仍需**单独的部署接线预注册**（时延预算与
  fail-closed 审计在此先行给出）。
