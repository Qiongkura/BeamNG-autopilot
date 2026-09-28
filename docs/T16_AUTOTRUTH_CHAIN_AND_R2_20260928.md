# T16：自动真值接入链 + 首轮自动真值增量实验（2026-09-28）

本文两件事：**(1)** 把受控场景的合格批次落成**带凭证的自动线真值**并接进训练链
（方案 §4.3 的"接进采集链"）；**(2)** 用固定 init/预算 + 冻结 seeds 跑首轮
"自动真值增量"实验，主看身份率 0.60 门（Order 5 的 R2 前置）。

## 1. 自动真值接入链（新增/修复）

| 环节 | 实现 | 关键实测 |
|---|---|---|
| 导出入口 | `scripts/m5_auto_truth_export.py`（新，测试 5 例） | 两道门：**annotation 线类覆盖 ≥0.6**（标签就是 annotation，外观不能当标签来源）+ 校准后残差 ≤2 px；不通过只记 isolate 清单 |
| 逐点认证 | `certify_points` | 只保留"校准后 2 px 内有 annotation 线像素"的真值点，丢弃比例写进凭证（§4.3"不确定的像素 ignore，有效面积与拒绝率分开统计"） |
| 出口复核 | 套场景的**静态投影标定** + 离线重建调色板 + 三路齐全（rgb/annotation/depth） | 缺调色板 → `PALETTE_CHANGED`；缺 depth → `RESOLUTION_MISMATCH`；不套标定 → 残差假失败（都实测踩到） |
| 凭证 | `write_truth_credentials`（engine_verified + `truth_provenance` v1） | 训练包与采集包同构：`meta.json`（`source_id=m5auto_a<锚点>` = 场景族）+ `annotation.json` + `front_main/frame_*.npz` |
| **接线缺口修复** | `manifest.py` 调 `audit_label` 时**没传凭证** | `paint_valid_frames` **0 → 2**（声明 engine_verified 的目录此前被判无凭证降级 = 等于没接上）；配 4 条回归测试（带 v1 凭证算真值；无凭证/未验证/契约不对一律 0） |
| 遮挡通道资格 | `verify_batch(occlusion_mode="report")` | uint8 量化 + 标定残差 ~0.6 m 的深度在 17 m 处分不出 1.5 m 的差，逐点判码会把生成批次全判 `OCCLUSION_INSERT`（10/11 站）→ 该通道对本批次记**不可判**（上报不判码，写进凭证），遮挡能力本身在合成反例套件里已验证 |

**规模批次**（30 路段 × 5 站 × 8 帧沿线序列）：140 站 / **1120 帧**采集，
批次验收 合格 50 / 隔离 62 / N-A 28（校准后 median **1.376 px**、33/50 ≤2 px）。
导出：11 站 / **88 帧 / 7 个场景族**，全部 `engine_verified`（`verified=True`）。

离线接线验证：导出站点 → `read_dir_credentials`（engine_verified + truth_verified）
→ `resolve_paint_rank` = verified → `audit_label` usable/valid=True（281 线像素）
→ 清单 `paint_valid_frames` 2/2。

## 2. 首轮自动真值增量实验

| 项 | 值 |
|---|---|
| 两臂共享 | `--init t14_e1_promotable.../baseline/seed42/checkpoint_last.pt`、`--total-steps 120`、seeds **42/43/44**、同一 8 目录开发集 |
| 基线 | 5 个已验证人工目录（75 帧，`human_revision`） |
| 候选 | 基线 + **11 个自动真值包（88 帧，`engine_verified`）** |
| 来源资格 | 13 个来源**全部 verified**、`research_only=False`（可晋级） |
| 数据确实进入损失 | sampler `unique_seen` **60 → 131**（全池配额轮换，两臂同预算 120/120 步） |

结果（dev 集）：

| 指标 | 基线 | 候选 | 判定 |
|---|---|---|---|
| line IoU（逐 seed） | 0.468 / 0.361 / 0.391 | 0.356 / 0.542 / 0.355 | 有正有负 |
| 候选身份率 | — | **0.3980**（micro，`M/R = 156/392`） | < 0.60 门 |
| 可测候选覆盖率 | — | 0.9776 | ✓ |
| 左右角色一致率 | 0.7247（上一轮 init 对照） | **0.6090** | < 0.70 门 |
| seed 42 精度 | — | 0.3799 | < 0.40 门 |

判定：**rejected**（identity 0.398 < 0.60、role 0.609 < 0.70、seed42 precision 0.380 < 0.40），
`round_outcome=meaningful_no_gain`；两臂未到平台期 → 判定标为**暂行**。


## 2b. 归因与第二轮/第三轮单因子实验（同 init/预算/seeds 42-44）

**错误分桶**（`m5_candidate_failure_attribution.py`，上一轮候选 seed 42）：
136 帧（76 帧有线真值）；掩码 recall=0 仅 **1** 帧、有真值无候选 **1** 帧
——掩码/候选提取不是瓶颈。候选 1305：有参考 525（覆盖 0.40）→ 匹配 205
（身份 0.39）→ 角色一致 124/205（0.605）。**780 个候选所在侧的真值参考像素 <1**
（=假线），306 个未匹配候选离最近引擎线 1–8 m，假线平均路外占比 **0.67**。
→ 失败主因是**多画线**，不是漏线，也不是参考/匹配机制。

于是按"无线场景 = 合格负例"（§4.2）把生成器的 `known_no_line` 站点也导出成
**带凭证的负例包**（证据：生成器声明 `line_generated=False` + 帧内 annotation
线类像素为 0 + 外观也无线状像素），同一 init/预算/seeds 跑剂量-反应：

| 臂（单因子：数据组成） | line IoU（逐 seed 均值） | 身份率 | 假线帧率（候选 vs 基线） | 假线像素占比（候选 vs 基线） | 最大连通域（候选 vs 基线） |
|---|---|---|---|---|---|
| +88 帧有线自动真值（池 131） | 0.418 | **0.398** | 0.974 vs 1.000 | 0.0101 vs 0.0068 | 53,010 vs 58,923 |
| +128 帧无线负例（占池 75%） | 0.245 | 0.344 | 1.000 vs 0.968 | 0.0121 vs 0.0069 | 73,639 vs 62,806 |
| +16 帧无线负例（占池 20%） | 0.365 | 0.378 | 1.000 vs 1.000 | 0.0109 vs 0.0082 | 70,525 vs 60,391 |

（三轮都 `rejected`、都标"暂行"（未到平台期）；三轮的来源资格全 verified、
`research_only=False`。）

**剂量-反应结论**：

1. **负例不是越多越好**：75% 负例把召回打崩（line IoU 0.245、身份 0.344）；
   20% 负例基本中性（0.365 / 0.378）。要压假线，靠"堆负例"无效。
2. **有线自动真值是目前唯一动过身份率的因子**（0.370→0.398，+0.028），
   并且**降低了最大连通域**（53k vs 59k）——但角色一致率掉到 0.609、精度 0.380。
   即：它让模型"更敢画且更成段"，但画的位置/角色与开发集约定不符。
3. 身份率 0.60 门仍未过，角色 0.70 门也未过。下一轮应按开发集实测的车道约定
   生成（车道宽度/角色定义贴近目标域），并单独诊断角色率；不再靠加负例。

## 3. 结论

1. **自动真值链打通且可晋级**：生成 → 两道门 → 逐点认证 → 凭证 v1 →
   清单/审计按凭证认账（`paint_valid_frames>0`）→ 进训练（`unique_seen` 翻倍）。
   这是方案 §4.3 要求的能力，本轮第一次全链路实跑。
2. **这批自动真值不是净收益**：身份率从 0.370（init 对照）升到 0.398（+0.028），
   但**角色一致率掉到 0.609**、seed 42 精度 0.380——生成场景的几何（任意锚点上
   的 ±1.8 m 车道）与开发集（italy 城镇/环路）的**车道约定不一致**，角色标签
   因此不匹配。按 §4.4"先加路段再加近邻帧"的方向是对的，但**标签约定要贴近
   目标域**：下一步应按开发集实测的车道宽度/角色定义来生成，而不是固定 1.8 m。
3. 身份率仍是最硬的门（0.398 < 0.60），角色率现在是第二硬的门。
4. 实验按纪律只改了一个因子（数据组成），init/预算/seed 全部固定；判定是暂行，
   不作晋级依据。

## 4. 下一步

1. 生成几何对齐目标域：从开发集（已有人工标注）实测车道宽度与线-角色关系，
   生成时用实测值（而不是固定 ±1.8 m）；再跑同一实验（同 init/预算/seeds）；
2. 角色一致率单独诊断（它已连续两轮是失败项）：看是"角色标签定义"还是
   "候选匹配口径"的问题（`m5_candidate_failure_attribution.py` 可分层）；
3. 扩量方向不变：更多路段/材质/曲率/光照，按锚点分场景族，不回流评价帧；
4. R2 仍不成立（0.398 < 0.60），不启动最终确认。

## 5. 复现

```pwsh
# 采集（30 路段 × 5 站 × 8 帧沿线序列）
.venv\Scripts\python.exe scripts\m5_controlled_scenes.py `
  --out logs\experiments\t16_scenes_big_20260928 --frames 8 --step-m 2.0 --anchors 30 --sites 5

# 导出（两道门 + 逐点认证 + 凭证）
.venv\Scripts\python.exe scripts\m5_auto_truth_export.py `
  --batch logs\experiments\t16_scenes_big_20260928 `
  --out logs\experiments\t16_autotruth_big_20260928

# 增量实验（固定 init/预算，冻结 seeds）
.venv\Scripts\python.exe scripts\m5_seg_autoloop.py rounds `
  --run-id t16_r2_autotruth_20260928 --rounds 1 `
  --runs <5 个人工目录> --eval-runs <8 个开发目录> `
  --proposals logs\experiments\t16_autotruth_big_20260928\proposal.json `
  --seeds 42 43 44 --total-steps 120 --init <父 checkpoint> `
  --paint-source <5 个人工目录>=human_revision
```

产物（logs 不提交）：`t16_scenes_big_20260928/`（140 站/1120 帧 + 报告）、
`t16_autotruth_big_20260928/`（88 帧凭证数据 + export_report.json）、
`t16_r2_autotruth_20260928/`（判定 + champion + 逐 seed 训练产物）。
