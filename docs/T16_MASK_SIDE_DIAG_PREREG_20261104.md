# T16 语义掩码侧右线诊断 · 预注册（2026-11-04）

## 1. 背景

候选层三臂（Y1 黄线阈值 / S1 thin 跨度 / I1 CLAHE）全部失败：右线候选召回
在所有臂与轮次中 ≈0（最大 +0.08），east_hard 的 true_rate 反被拉低；west 控制
零变化。预注册已判定 **H0：损失在语义掩码/检测层**。本诊断回答"到底在哪一层"。

## 2. 问题与假设

- **Q1**：模型**原始 argmax 线掩码**里有没有右侧黄线像素？
- **Q2**：若有，是否在**后处理**（形态学/路面约束/外观门）中被删除？
- **Q3**：若无，是否碎片化（连通段过短）导致候选层无法使用？

假设（互斥，可判定）：
- **HA 模型未检出**：raw 线掩码在黄线 GT 上的召回 <0.1 → 模型/数据域问题，
  本环境内无候选层或后处理杠杆，记录为外部阻塞。
- **HB 后处理删除**：raw 召回 ≥0.3 且 post 召回 / raw 召回 <0.5 → 后处理
  （含外观门）是删除点，属本环境内可修。
- **HC 碎片化**：raw 与 post 召回都 ≥0.3，但预测黄线连通段平均跨度 <1.5m
  → 候选 span 门是瓶颈（注意 S1 已失败，若 HC 成立需解释 S1 为何无效）。

## 3. 数据与装置（离线，不连引擎、不训练）

- 认证池帧（engine_verified）：east off45 `a0s0`（3 帧）、`a1s0`（3 帧）、
  `a2s0`（3 帧）；west off40 `a3s0`（3 帧）作对照。帧为 144×192
  colour+label npz。
- 模型：`seg_model_v28_sslA`（与全部候选层实验同一 pin）。
- GT：`label==2` 的线像素，按漆色拆成 **yellow GT**（HSV：hue 12–45、
  sat ≥30、val ≥100）与 **white GT**（其余线像素）。这是"角色"的代理：
  报告每帧 yellow/white GT 像素数，若某帧 yellow GT≈0 则该帧不参与黄线
  召回统计（避免 0/0 虚高）。
- 预测：
  - raw：`Segmenter.predict_proba()` 返回的 raw_line（argmax、未后处理）；
  - post：`Segmenter.predict()` 的 line（完整后处理管线的最终掩码）。
- 指标（每帧）：
  - `gt_yellow_px`、`gt_white_px`；
  - raw/post 在 yellow GT、white GT 上的召回；
  - `drop_frac = 1 − post_recall/raw_recall`（黄线）；
  - 预测黄线连通段：数量、平均/最大跨度（像素→按帧高折算米，粗估）；
  - 预测类别像素计数（mask_counts 同源：line/bg/road）。
- 聚合：按场景报均值；east 三站合并 vs west 对照。

## 4. 判定（跑前冻结）

- **HA**：east 的 raw yellow 召回均值 <0.1 → "模型未检出"，记录外部阻塞
  （需新数据/域授权），本 recipe 内的标线提拉线就此收口。
- **HB**：raw yellow ≥0.3 且 drop_frac >0.5 → 后处理/外观门可修，进入
  "后处理研究臂"（需新预注册）。
- **HC**：raw/post 均 ≥0.3 且跨度 <1.5m → span 门；但 S1 已失败，需先解释
  矛盾，否则按 HA/HB 处理。
- 若 east 与 west 的结论不一致，分别记录，不取平均掩盖。

## 5. 纪束与交付

- 离线只读；不训练、不改门、不新增标注、不消费最终集；
- 交付：本预注册、诊断脚本 + 单测、`T16_MASK_SIDE_DIAG_RESULT_*.md`、
  JSON 结果与证据 zip。
