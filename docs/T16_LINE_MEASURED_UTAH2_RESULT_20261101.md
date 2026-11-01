# T16 utah 重采结果：每站 Scenario 修复后 2 站/6 帧首次达标（2026-11-01）

## 执行

- 预注册：`T16_LINE_MEASURED_UTAH2_PREREG_20261029`（唯一变更因子 = 每站 Scenario
  采集结构，scripts 53850c1；冻结门一字未改）。
- attempt1（2026-10-29 夜）首站 Scenario 加载即模拟器断连（BNGDisconnectedError，
  无帧）；2026-11-01 同命令重试一次成功（协议内，首启不稳定族第二次复现）。
- 产物：`logs/experiments/t16_scenes_20261101_224341`（采集）与
  `logs/experiments/t16_utah2_export_20261101`（导出）；
  证据档 `logs/evidence_archives/t16_utah2_20261101_evidence.zip`
  （35 项、sha256 清单、complete）。

## 采集批次验收（collector 门）

**5/5 全合格、隔离 0**（10-18 批次 0/5 全隔离）：

| 站 | 覆盖 | annotation 线像素 |
| --- | --- | --- |
| a0s0 | 1.0 | 3547 |
| a1s0 | 0.871 | 777 |
| a2s0 | 0.9133 | 2931 |
| a3s0 | 0.9944 | 3623 |
| a4s0 | 1.0 | 5921 |

校准后 mean 1.391 / median 1.001 / ≤2px 4/5。截断机制（10-18 的 a2/a3/a4
渲染范围止于 stn≈15、a3 右全缺）在生产批次中**消失**。

## 冻结导出门裁决（未改一字）

**准入 2 站 / 6 帧 engine_verified（utah 首次达到 ≥2站/≥6帧 目标）**：

| 站 | 实例覆盖 L/R | after px (med) | paint | 裁决 |
| --- | --- | --- | --- | --- |
| a3s0 | 1.0 / 0.9889 | 0.398 (0.359) | 1.0 | **准入** |
| a4s0 | 1.0 / 1.0 | 0.936 (0.696) | 1.0 | **准入** |

修复前后对照：a3s0 从（右覆盖 0.0 + 14.35px）→ 0.398px/paint 1.0；
a4s0 从（右 0.5472 + 逐帧 1.6→6.6px 漂移）→ 0.936px。

**隔离 3 站（原因逐站保留）**：
- a0s0：median 1.05px（超 1.0 一线）+ paint 左实例 0/86 —— `T16_UTAH_A0S0_OCCLUSION_ATTRIB_20261018.md`
  已定因的站点阴影（太阳角稳定属性，非回归；paint 判据照度敏感语义再次实证）；
- a1s0：投影 2.86px > 2.0（paint 1.0/1.0）—— 纯几何前沿；
- a2s0：median 1.76px > 1.0（paint 0.994）—— 纯几何前沿。

## 外部证据删除事件（2026-11-01 发现）

10-29 深夜至 11-01 之间 `logs/m5_seg/` 整目录与 `logs/experiments/` 历史产物
（evidence pack、training_results.html 等）被外部删除：无保留清单
（m5_logs_retention 只删 .pt/.pth/.ckpt 且先落 manifest，最后一次运行 10-20），
磁盘 68G 空闲非空间触发，采集器无删除逻辑。**损失**：v13b 部署模型、全部历史
池帧（E3/H/WCU 等 105 帧语料）、10-29 探针 probe_report.json；**幸存**：全部
docs/代码/测试（git）、bc_best.pt、presence_model、v13b holdout 评测记录。
后果与对策：历史池不可再跑（H 人工池不可再生成——不新增人工标注约束下为
永久损失，已记录）；本文档与后续批次产物即时归档
（`evidence_archives/`，哈希清单），后续导出池默认纳入保留清单（见保留加固提交）。

## 标线识别基线（幸存记录）

`logs/goal_20260921/holdout_lines_eval_seg_model_v13b.json`：39 帧 holdout，
**line_iou_near = 0.116**、miou 0.670、pixel_acc 0.961、线像素占比 0.35%。
这就是"标线识别"当前的绝对基线；模型本体已随删除损失。标线识别提升计划的
第一步 = 用本批开始重建的 engine_verified 池重新训练并按同一 holdout 口径评测。

## 纪束

冻结门未改；模型未晋级；最终集未消费；无人工标注；新池为 engine_verified。
