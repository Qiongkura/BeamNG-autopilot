# S6 定向学习实验设计（冻结稿，2026-09-26）

本文件是 **S6 的运行前冻结**：两臂定义、数据准入、选择规则、最小有意义收益与
非劣余量、决策规则都在跑之前写死。方案 v2 §S6 要求"运行前冻结最小有意义收益与
非劣余量"；跑完后不得改这些数字，只允许如实报告"无改善/证据不足"。

前置状态：D1 已签收（本轮补 T10/T11 并重测），S4/S5 已落地并通过测试与页面实检；
`docs/T14_AUDITED_TRUTH_REMEASURE_20260926.md` 给出审计后读数（身份率
0.3115/0.3280，远低于 0.60）。**当前无可晋级候选**，本轮实验只做定向探索。

## 1. 数据准入与隔离（先冻结，再训练）

| 角色 | 目录 | 档位 | 说明 |
|---|---|---|---|
| 训练（线通道，研究臂） | `logs/m5_seg/line_truth_agent_full_20260925/town/front_main` | agent（弱） | 两臂共有；判定记 `research_only`，不允许晋级 |
| 开发/评价（固定） | `logs/m5_seg/line_truth_agent_full_20260925/wide|plain/front_main` + 审计后人工包 | agent / human | 评价集本实验内**不变**，不得入训 |
| 困难负例候选（E1） | 待定，见 §4 | — | **不得**取自评价帧或其空间邻近段 |

纪律：`--equal-steps` 保持两臂**实测总步数相同**（不是"最接近的整数轮"）；两臂共享
起始权重、划分、优化器、有效 batch、训练步数与评价路径；一次只改一个因子；
选择 checkpoint 的规则见 §3（跑前写死，不允许"结果出来后挑更好的那一种"）。

## 2. E2：路口/弯道漏检——用**方向**而不是**幅度**做单因子实验

**原假设**：line 类漏检（FN）与假线（FP）同时存在（人工包上 recall 0.30/0.54、
身份 0.31/0.33；agent 真值集上 precision 0.26–0.30），而**唯一还没被试过的旋钮
是 FP/FN 的比值**。

**先查历史（本轮实测，防止重复实验）**：

| 历史轮 | 配置 | 因子 | 结果 |
|---|---|---|---|
| `t14_soak4h_i18/i19/i37/i38/i56` | **road-only**（`allow_road_only=true`） | `line_tversky_weight` 0.5/1.0 | 旗标照记但 line 类被整通道屏蔽 → **因子不可能生效**；5 轮全 `needs_evidence`。本轮 S5 因子门已把这类提议判 inactive 并拒训（T14 用例） |
| `t14_auto_20260925_agentline` | 真实线监督 | `line_tversky_weight: 2.0` | 候选臂 recall 0.6896 / precision 0.302 / 身份 0.1577 → rejected |
| `t14_auto_20260925_line` | 真实线监督 | `line_tversky_weight: 2.0` | 候选臂 recall 0.728 / precision 0.2617 / 身份 0.1485 → rejected |

结论：**幅度方向（1.0→2.0）已经试过两轮**，不是未测领域；再跑同一因子是重复实验。
T15 实测（`docs/T15_SEG_LOSS_DIRECTION_20260926.md`）：实现是
`TP/(TP + alpha*FP + beta*FN)`，**只有 `beta/alpha` 比值决定 FP/FN 方向**，
提高整体权重只把方向"再推一档"，而 `alpha` **目前不在训练器白名单里**
（`m5_train_seg.py` 的 `--line-tversky-alpha` 不存在 → 提议不了）。

**因此 E2 的定义（本文件冻结）**：

1. **前置代码工作**（S6 前置，不是实验本身）：把 `line_tversky_alpha` 接进
   训练器参数与 `proposer.ALLOWED_KEYS["loss_weights"]`，并加"参数真的进了训练
   命令"的接线测试（与 `--line-tversky-weight` 同一路径）。这是**唯一**允许的
   代码改动；改动本身不得与实验轮混在一起跑（先冻结配方，再跑实验）。
2. **两臂**（唯一差别 = `beta/alpha`）：baseline 保持默认
   `alpha=0.3, beta=0.7`（比值 2.33，偏向 recall）；candidate 用
   `alpha=0.5, beta=0.5`（比值 1.0，对称）。两臂 `--line-tversky-weight 1.0`
   不变、`--epochs 24 --batch 4 --lr 0.001 --equal-steps`、seeds 42/43/44。
3. **主指标**：`line_precision` 与 `candidate_identity_rate`（micro，固定开发集），
   因为当前过不去的是这两项；`line_recall` 作为**非劣**约束（不得下降超过
   -0.03，方向实验可能牺牲一点 recall）。
4. **方向预期（T15 数值）**：比值 2.33 → 1.0 会让 |g_FN|/|g_FP| 下降（更偏 precision）。
   若实测方向相反，先按 T15 的方法复算再解释，不口头解释。

## 2.1 线监督来源（研究臂）

两臂都**不加** `--ignore-line-class`；线监督来自 agent 档位标签
（`--paint-source front_main=agent_revision`）→ 判定记 `research_only`，
**不允许晋级**。要用可晋级的线标签做同样实验，需要评价集之外的新人工线标注
（与 E1 同一约束，见 §4）。

## 3. 选择规则与判定规则（跑前冻结）

* **checkpoint 选择**：每臂同时报告 `best` 与 `checkpoint_last`，主结论以
  `checkpoint_last` 为准（两臂步数相同，避免"谁训得久"混入）；`best` 只作参考，
  不用于挑结论。
* **配对**：同 seed 的两臂差值；3 个 seed 给均值与区间（t 区间，n=3 时明确标
  样本极少）。
* **最小有意义收益**：`line_recall` 配对均值 **≥ +0.03（绝对）**。
  依据：历史单段数据组成变化的观测效应 <1 点且跨 0（`t14_e0_20260925`），
  小于 3 点的变化不作为改善证据。
* **非劣余量**：`line_recall` 不得下降超过 **-0.01**；合格负例的假线帧率相对
  变差不得超过 **+20%**（防止"多画线"换 recall）。
* **结论口径**：区间跨 0 → "证据不足"（合法结果）；达标 → "支持原假设（研究臂，
  不可晋级）"；未达标且超非劣余量 → "推翻"。
* **无收益停止**：只统计**有意义且成功完成**的实验（`round_outcome ==
  meaningful_no_gain`）；无效因子/缺标注/资格失败另有原因，不算"没有提升空间"。

## 4. E1：困难负例组成——当前**受阻**，需要先补输入

**原假设**：用已确认的土路/易混淆纹理负例替换固定比例训练样本，能降低假线
（土路 28/28 误报的历史观测）。

**阻塞原因（逐条可查）**：

1. "**已确认**负例"按 T10 必须是 `verified` 档位（人工复核）的全零标签；训练侧
   现有目录的档位是 agent/engine → 不构成合格负例（T10 用例钉住）。
2. 现有 verified 负例只在人工复核包里，而那是**开发/评价帧**（E1 明文禁止
   "取自当前固定开发/校准/最终评价帧或其空间邻近段"）。
3. 因此 E1 需要的输入是：**新的人工确认负例**（非评价集、空间上远离评价集），
   或一份书面决定把 E1 降级为研究臂（用 agent 标签负例，判定 `research_only`）。

**解除条件（按性价比排序）**：① 用户对若干非评价集的土路/碎石采集做一次人工
"确认无线"标注（最小量：≥30 帧/场景，用于替换比例 20%）；② 或明确授权研究臂
版本，并按 §3 的规则同样冻结。**在此之前不跑 E1**，也不拿 E1 的"无收益"当结论。

## 5. 命令模板（跑前核对 `--help`，参数以实际实现为准）

```pwsh
[Console]::OutputEncoding = [Text.Encoding]::UTF8; .venv/Scripts/python.exe scripts/m5_seg_autoloop.py run --run-id <新run-id> --config <冻结配置.json> --dry-run
```

```pwsh
[Console]::OutputEncoding = [Text.Encoding]::UTF8; .venv/Scripts/python.exe scripts/m5_seg_autoloop.py run --run-id <已核对run-id> --config <冻结配置.json> --no-dry-run
```

E2 的提议（单因子，经 `factor_activity` 判定线监督下 active）：

```json
{"proposals": [{"candidate_id": "e2-line-tversky-2", "family": "loss_weights",
                "factor": {"line_tversky_weight": 2.0}}]}
```

## 6. 运行后必须报告

原假设 → 是否真正激活（optimizer step、权重差异、实际损失项、生效步数）→
可复现输入（含模型/数据哈希与设备）→ 实测结果（逐 seed、逐场景、C/R/M/L/A、
负例诊断、时延）→ 支持/推翻/证据不足 → 未知与下一步。**研究臂结果不得写成
"候选可晋级"**；`best` 与 `last` 分开报。
