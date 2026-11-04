# T16 集成接 drive 部署接线结果：D1–D4 全部通过（2026-11-04）

## 接线

`m5_fsd_drive --seg-model2` → `_setup_runtime` 构建
`EnsembleSegmenter([seg_model, seg_model2], post_norm_width=...)`；
仅给 `--seg-model` 时行为与现行完全一致。探针 `--model2` 透传并记录。

## D1 时延预算

- 离线微基准（536×403）：单模型 p50 18.76ms → 集成 29.00ms（**1.55×**）。
- live 4 轮（single×2 / ensemble×2）：日志无 watchdog 触发、无 timeout、
  无 Traceback（仅正常的 "input watchdog armed"）。

## D2 显存增量

| 配置 | `torch.cuda.memory_allocated` |
| --- | --- |
| 单模型 | 1.7 MB |
| 第二模型 | +1.7 MB |
| 集成总计 | 3.4 MB |

远低于 200MB 预算（12GB 卡）。注：CUDA context 开销不计入该指标，但
单/双模型共享同一 context，增量可忽略。

## D3 fail-closed

- **单测**：成员抛异常 → `EnsembleSegmenter.predict` 向上抛出；构建失败 →
  drive 走 "segmentation model disabled"。
- **live 负面注入**（第二模型路径不存在）：
  `[fsd-drive] segmentation model disabled: [Errno 2] ...` →
  `src=perception-unavailable` → `holding brake`，无崩溃、无静默退单模型。

## D4 放置可用性（west a3s0，2 重复）

| 臂 | 轮 | paired 率 | placed | placement_rc |
| --- | --- | --- | --- | --- |
| 单模型 | r0 | 1.00 | True | 0 |
| 单模型 | r1 | 1.00 | False | 3 |
| 集成 | r0 | 1.00 | False | 3 |
| 集成 | r1 | 1.00 | True | 0 |

- paired 帧率两者均 **1.00**；放置成功 **1/2 vs 1/2**（同分）。
- **D4 通过**（无下降）；注意该站点放置本身双稳态（各 1 成 1 败），
  2 重复不足以给出更强的可用性结论，只作不下降判定。

## 判定

全部 D1–D4 通过：集成接线可用、时延与显存代价可接受、fail-closed 行为
正确、放置可用性不下降。集成仍**不作为默认**（显式 `--seg-model2`）；
live 贴花 GT 仍不可用（环境阻塞），本任务不重开感知 IoU 结论。

## 产物

`logs/m5_seg/ensemble_placement_table.json`、`ens_place_*`（探针/日志/
session）、`ens_place_negative.*`（负面注入）、`ensemble_vram.json`。
