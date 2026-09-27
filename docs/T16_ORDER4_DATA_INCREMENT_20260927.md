# T16 Order 4（2）：固定 init 下的数据增量实验（2026-09-27）

方案 §3.4 的顺序要求："先独立检验初始化（见
`docs/T16_ORDER4_INIT_AB_20260927.md`）；**选定新基线后，再做自动数据增量实验，
不把初始化与数据组成同时变化解释成单一因子收益**。"本文是选定 init 基线后的
第一次数据增量实验，跑在 Order 1 接好的新协议上（步预算 + 全池采样 + init 谱系）。

## 1. 设置

| 项 | 值 |
|---|---|
| 入口 | `scripts/m5_seg_autoloop.py rounds`（单因素：数据组成） |
| 两臂共享 | `--init t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`、`--total-steps 120`、batch 4、lr 1e-3、seeds **42/43/44**、`--paint-source ...=human_revision`（8 个目录逐条声明，凭证校验通过） |
| 基线数据 | 5 个已验证正例目录（75 帧）：it3 front_fisheye + jv 四视角 |
| 候选增量 | **+3 个未进基线的已验证视角**（it3 的 front_main / pillar_left / pillar_right，+45 帧）；未使用评价帧或相邻路段（方案 §P2 纪律） |
| 评估 | 8 个 reviewed 开发目录；判定器与阈值不变（COVERAGE_GATE_FROZEN 等照旧） |

## 2. 协议生效的证据（方案 §3 验收项）

- **等步数**：两臂逐 seed `steps_done=120/120`、`stopped_by=step_budget`。
- **增加的数据确实进入损失**：`train_meta` 里 sampler 记账
  **`unique_seen` 基线 60 → 候选 96**（全池配额轮换；`sampling=quota_full_pool`）。
- **起始身份可追溯**：判定里 `init_from` 指向父 checkpoint；
  `train_meta.*.init.init_source=random`、`provenance_complete=false`
  （父是旧产物，按 §3.5 只可用于探索）。
- **适用性修复在真实判定里生效**：`scene_not_applicable` 两条
  （`italy/ring_20260926_123638` / `_123950`），依据
  `rank_verified_zero_line_pixels`（弱档，理由里写明"没有负例计数，证据较弱"），
  不再进缺测通道。
- 判定含 `counts`（P=60 C=306 R=297 M=112 L=112 A=88）、`counts_ratios`、
  `hard_gate_violations`、`missing_metrics`（空）、`step_budget`、`sampling`。

## 3. 结果

| 指标 | 基线（init，120 步） | 候选（init + 45 帧，120 步） |
|---|---|---|
| line IoU（逐 seed） | 0.4701 / 0.3863 / 0.3762（均值 0.411） | 0.4486 / 0.4138 / 0.3083（均值 0.390） |
| 候选身份率（micro） | — | 0.3771（`M/R = 112/297`），覆盖率 0.9706 |
| 硬门违反 | — | 12 条（含逐 seed/分场景的 identity 0.38–0.40 < 0.60） |
| 判定 | — | **rejected**；`round_outcome=meaningful_no_gain` |
| 平台期 | 两臂都在动 → 判定按方案标为**暂行**，不作结论引用 | |

**横向一致**：本轮的基线臂（5 目录 + init + 120 步 + seeds 42/43/44）line IoU
均值 0.411，与 Order 4（1）里 `m5_init_ab.py` 的 init 臂 0.415 一致（差 0.004）
——两个独立入口在同一配置上给同一答案，新协议自洽。

## 4. 结论

1. **同一场景包内加视角（+45 帧、+60% 唯一样本）没有带来 line IoU 收益**
   （0.411 → 0.390，逐 seed 有正有负），身份率不动（0.377）。
   结合 §4.4 的方向：下一步该加的是**新路段/新材料**（受控场景扩量已经给出
   可复现入口与逐站资格），而不是同一场景的更多近邻视角。
2. 判定是**暂行**的（两臂未到平台期、父谱系不完整）；本实验只用于选方向，
   不作晋级依据。
3. 开发集已被反复使用：按方案纪律，它只能算**开发证据**，不能升级为独立最终集；
   R2/最终确认需要独立参考与完整谱系基座（Order 5 的前置条件，见 §5）。

## 5. Order 5 状态（R2 / 签收）

- **R2 仍不成立**：身份率 0.38（micro）< 0.60 门；本轮与上一轮（init 对照
  0.370）都卡在同一项。
- 未启动最终确认程序（§10.3：确认记录必须绑定候选权重；当前没有过门候选）。
- 已具备的签收前置：判定链可重放、计数契约 v5、N/A 适用性、负例 v2 指标、
  受控场景的可复现真值入口与逐站资格门（合格站 median 1.93 px）。

## 6. 复现

```pwsh
$srcs = @(
 'logs\experiments\annotate_pkg_e2_it3_20260927\front_fisheye',
 'logs\experiments\annotate_pkg_e1_jv_20260927\front_fisheye',
 'logs\experiments\annotate_pkg_e1_jv_20260927\front_main',
 'logs\experiments\annotate_pkg_e1_jv_20260927\pillar_left',
 'logs\experiments\annotate_pkg_e1_jv_20260927\pillar_right')
$args = @('rounds','--run-id','t16_data_inc_it3views_20260927','--rounds','1',
 '--runs') + $srcs + @('--eval-runs') + (Get-ChildItem logs\experiments\review_pack_20260926\reviewed_full -Directory | ForEach-Object { "$($_.FullName)\front_main" }) + @(
 '--proposals','logs\experiments\t16_data_inc_it3views_20260927_proposal.json',
 '--seeds','42','43','44','--total-steps','120',
 '--init','logs\experiments\t14_e1_promotable_20260927\baseline\seed42\checkpoint_last.pt')
foreach ($d in ($srcs + @('logs\experiments\annotate_pkg_e2_it3_20260927\front_main',
 'logs\experiments\annotate_pkg_e2_it3_20260927\pillar_left',
 'logs\experiments\annotate_pkg_e2_it3_20260927\pillar_right'))) { $args += @('--paint-source',"$d=human_revision") }
.venv\Scripts\python.exe scripts\m5_seg_autoloop.py @args
```

产物（logs 不提交）：`logs/experiments/t16_data_inc_it3views_20260927/`
（`decision_add-it3-views-r0.json`、`champion.json`、`train_meta`、
`round0/seed*/`、`baseline/seed*/`）。
