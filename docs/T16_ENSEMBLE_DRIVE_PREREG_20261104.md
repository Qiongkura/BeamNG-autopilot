# T16 集成接 drive 部署接线 · 预注册（2026-11-04）

## 1. 目标

把 `EnsembleSegmenter`（已验证：固定 4 折宏平均 0.539246、时延 1.55×、
fail-closed 单测通过）接进 `m5_fsd_drive` 的放置/驾驶链路，作为**显式可选**
的感知后端（默认单模型不变）。验证部署条件后才允许进入默认启用讨论。

## 2. 接线

- 入口：`--seg-model2 <checkpoint>`（与 `--seg-model` 配对）。
- 行为：两者都给时，`_setup_runtime` 构建
  `EnsembleSegmenter([seg_model, seg_model2], post_norm_width=...)`；
  只给 `--seg-model` 时行为与现在完全一致（默认路径不变）。
- 行分割模型（`--line-seg-model`）保持单模型（集成只替换语义分割头）。
- 记录：运行记录/日志打印 `model_path`（含两个成员名），可追溯。

## 3. 验证判据（跑批前冻结）

- **D1 时延预算**：离线微基准已测 p50 18.76→29.00ms（1.55×）；live 放置
  期间不出现超时/看门狗触发（日志无 watchdog/timeout）。
- **D2 显存增量**：双模型驻留相对单模型的 `torch.cuda.memory_allocated`
  增量 **≤200 MB**（12GB 卡）。
- **D3 fail-closed**：任一成员缺失/损坏 → 感知被禁用（现有
  "segmentation model disabled" 路径）→ 不放置、刹停；不得静默退化为
  单模型。单测 + 一次 live 负面注入。
- **D4 放置可用性不下降**：同一站点/条件，单模型 vs 集成各 2 轮；
  集成的 paired 帧率与放置成功率不低于单模型 **−5pp**（探索性基线；
  live 贴花 GT 不可用，故只比可用性，不涉 IoU）。
- 任一 D 不达 → 不启用、记录原因。

## 4. 纪束

- 默认路径不变（无 `--seg-model2` 时行为逐位一致）；不训练、不晋级、
  不改默认门、不新增标注；全部运行/结果归档。
- live 右线 GT 仍不可用（环境阻塞），本任务只验证部署条件，不重开
  感知指标结论。
