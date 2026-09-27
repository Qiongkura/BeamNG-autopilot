# T16 Order 0 冻结：新数据/预算/init/凭证契约（2026-09-27）

方案：`docs/T16_AUTONOMOUS_LEARNING_ROADMAP_20260927.md`（= 桌面副本，逐字一致）。
本文是该方案 §8 次序 0 的交付：复现"12 帧、72 步、无 init"、划清新旧口径、
冻结 A/B/C 并行编码要用的接口。**本文冻结的是接口与语义，不是实现**。

## 1. 基线与 dirty 清单（冻结，勿覆盖）

- 基线 commit：`ee3132e`（分支 `fix/round3-hardening-20260921`）。
- 用户未提交（**唯一写入者是用户，本轮任何 agent 不得修改**）：
  `README.md`、`beamng_autopilot/experiments/monitor_server.py`、
  `beamng_autopilot/experiments/monitor_ui.py`、`scripts/m5_train_monitor.py`、
  `docs/TRAINING_MONITOR_20260924.md`、`docs/T14_DEVELOPMENT_EXECUTION_PLAN_V2_20260926.md`、
  `docs/T14_NEXT_STEP_20260927.md`。
  → 看板/监控新面板（方案 §7）本轮只出**补丁建议文档**，不改这些文件。
- 未跟踪：`docs/T16_AUTONOMOUS_LEARNING_ROADMAP_20260927.md`（方案本体，随本轮提交）、
  `Item`（0 字节误重定向产物，保留不动）。

## 2. "12 帧 / 72 步 / 无 init" 复现链（Order 0 完成条件）

| 环节 | 证据 | 值 |
|---|---|---|
| 判定文件 | `logs/experiments/t14_e1_promotable_20260927/decision_e1-neg20-promotable-r0.json` | `n_train_frames_by_arm` 两臂各 seed = 12；`steps_by_arm` 各 seed = 72；`all_at_plateau=True` |
| 判定文件 | `logs/experiments/t14_line_promotable2_20260927/decision_line-add-verified-r0.json` | 同上 12/72；`all_at_plateau=False`（该轮非收敛结论） |
| 基线记录 | 两个实验的 `champion.json` | `base_n_train=12`、`max_train_frames=12`、`equal_steps=True`、`epochs=24`；`arm_runs=['logs\experiments\annotate_pkg_e2_it3_20260927\front_fisheye']`（1 个目录） |
| 步数公式 | `loop_config.json`：`epochs=24, batch=4`；`steps_per_epoch=ceil(12/4)=3` | 24×3 = 72 步/seed |
| 截帧机制 | `scripts/m5_train_seg.py:333 cap_train_frames()`，在数据加载后调用一次；autoloop 只在候选臂加 `--max-train-frames` | 候选 48 帧被按 run 配额固定抽到 12；**新增视角未进损失** |
| 无 init | `scripts/m5_seg_autoloop.py:train_cmd()` 不传 `--init`/`--resume`；`loop_config.json` 无相关键 | 两轮都是小样本**从头**训练 |
| init 能力 | `scripts/m5_train_seg.py:615 --init`（`897` 行只加载 `state_dict`，优化器从头）、`613 --resume`（同任务续训） | 训练器已支持，autoloop 未接 |

结论：流程可运行、来源可晋级，但"样本利用（48→12）+ 训练预算（72 步）+ 初始化（随机）"
三者同时受限；**两轮 rejected 只支持"该配方被拒"，不支持对新增数据/学习能力的一般结论**。

## 3. 冻结契约 v1

### 3.1 预算契约（budget v1）

- 主停止条件 = **实际 optimizer step 数**：`--total-steps N`（每臂每 seed）。
  达到 N 即停；`epochs` 在给定 `--total-steps` 时由
  `steps_per_epoch=ceil(n_train/batch)` 反推，不再单独作为主条件。
- `train_hist.json` 增：`steps_done`、`total_steps`、`stopped_by`
  （`"step_budget"` / `"epochs"`）。
- checkpoint 的 `train_args` 增：`total_steps`、`steps_done`、`sampler`、
  `sampler_state_digest`、`init_from`、`init_sha16`、`init_source`、
  `provenance_complete`。
- `--resume` 恢复 model/opt/sched/`steps_done`/sampler 状态，续训**逐位可复现**。
- 等步数对照由**预算**保证（两臂同 N），不再靠截数据；`--max-train-frames`
  只保留为历史协议，判定里必须标 `sampling="legacy_cap"`。
- 未给 `--total-steps` → 完全保持旧 epoch 行为（现有 3004 测试不得受影响）。

### 3.2 采样契约（sampling v1）

- 池 = 划分后的全部合格训练帧；采样器对**全池**做按场景（run 目录）/视角配额轮换。
- 逐帧身份 = 内容 hash（`frame_hash`，sha256 前 16 位，来源见实现）；
  计数：`unique_available`（池大小）、`unique_seen`（**真实进入损失**的不同
  frame_hash 数）、`exposures_total`、`exposures_by_run`、正/负例曝光数。
- `unique_seen` **必须来自训练采样**，不得由池清单或目录文件数派生
  （方案 §7：禁止把候选池 60 个帧文件显示成"60 帧已学习"）。
- 确定性：给定 seed + 池 + `--total-steps`，采样序列可复现；checkpoint 里
  `sampler_state` 支持精确恢复。
- 报告：`train_hist.json` 增 `sampler_report`（上述计数字典）。

### 3.3 初始化契约（init v1）

- `--init CKPT` = **权重初始化**（优化器/调度器从头）；`--resume CKPT` = 恢复
  同一训练任务；两者同时给出 → 直接报错退出。
- `--init` 记录：`init_from`（路径）、`init_sha16`、`init_source`
  （`champion`/`production`/`random`，由父 checkpoint 自身的 `train_args` 链判定）、
  `init_arch`（`width`）、`parent_history`（父的 run_id/steps_done/dataset_id 摘要）、
  `provenance_complete`（父链能追到随机初始化根且各环节有记录 = True）。
- 架构不一致（`width` 不同）→ 报错拒绝，不做静默部分加载。
- 未给 `--init` → 显式记 `init_source="random"`。
- 谱系纪律：`provenance_complete=False` 的父模型（含当前生产模型）只可用于
  **探索微调**；进入最终确认前必须有完整谱系的新基座或证明未见过的场景。
- 效果结论用冻结配对 seeds 42/43/44；1 seed 只作冒烟（梯度/输入/预算）。

### 3.4 自动真值凭证契约（credentials v1，P1）

sidecar `annotation.json` 增 `truth_contract: "v1"` + `truth_provenance`：

```
truth_provenance = {
  "generator": {"name", "version", "sha"},
  "asset":     {"map", "segment", "sha"},
  "run":       {"id", "scene_seed", "game_version", "renderer"},
  "camera":    {"name", "calibration_sha", "frame_ids"},
  "labels":    {"source_image_sha", "label_sha", "channel_valid_area",
                "unknown_reason"},
  "report":    {"test_report_sha", "verifier_version", "verified": true},
}
```

- `label_source="engine_verified"` **只有在** `truth_provenance.report.verified
  is True` 且 `truth_contract=="v1"` 时才算 verified 档；否则降到 `absent`
  （不得仅改 sidecar 字符串把旧不完整标签提升）。人工路径
  （`human_revision`）语义不变。
- 反例库（必须被拒绝，带拒绝码）：相机翻转、尺寸变化、时间错帧、遮挡物插入、
  标签篡改；另有黑图/分辨率错位/palette 变化/异步旧帧检测。
- 逐通道资格（LINE / ROAD / 铺装-土肩 / 左右角色）各自独立；某通道合格不替
  其他通道背书。不确定边界像素 ignore，有效面积与拒绝率分开统计。
- 遮挡后的不可见漆线**不计漏检**（该区域 ignore，不是 background）。

### 3.5 适用性契约（N/A vs UNKNOWN，P0 修复）

- **已证明无线**（`label_rank=="verified"` 且负例判据：无 255、无 class 2、
  非空、`eligible_frames>0`）→ 线通道指标（`line_recall`/`line_precision`/
  `line_iou`）记 `not_applicable`，**不进缺测通道**（现状 bug：判定文件里
  `scene_applicability=not_applicable` 与 `missing_metrics` 里的
  `scene ...: line_recall: UNKNOWN` 自相矛盾，来源 = `gates.scene_report`
  像素度量路没有像 `scene_count_violations` 那样过滤无线场景）。
- 标签不明（无档位/非 verified/未测）→ 仍 `UNKNOWN` → `needs_evidence`。
- 该修复**不取消**有线场景召回门（0.70 不动），也**不得**令"永远不画线"过门：
  无线场景转由负例指标约束（假线帧率、像素占比、最大连通域、进入控制区域的
  假候选数），全零输出在有线场景仍触发漏线违反。
- 判定里必须能看到：哪些场景按 N/A 排除、依据（负例 eligible 计数）、以及
  `unknown` 与 `not_applicable` 的区分。

## 4. 新旧口径分界

| 维度 | 旧（历史协议，只读不改） | 新（本契约） |
|---|---|---|
| 样本利用 | `--max-train-frames` 固定截帧 | 全池配额轮换 + `unique_seen` 计数 |
| 停止条件 | `epochs` | `--total-steps`（optimizer step） |
| 初始化 | 随机（无记录） | `--init` + 谱系记录；`random` 也显式记 |
| 来源档位 | sidecar 字符串 | 字符串 + `truth_provenance` 证据 |
| 无线场景 | 混在缺测里 UNKNOWN | N/A（有证据）或 UNKNOWN（无证据） |
| 判定字段 | 无 `sampling`/`init_*` | 判定必须带 `sampling`、`init_provenance`、`step_budget` |

旧判定不得回填新字段（`legacy_replay_note` 语义不变）；新实验的判定必须带
上表"新"列字段，缺任一 → 不算完成。

## 5. 并行阶段文件归属（互斥，不得越界）

| 归属 | 文件 |
|---|---|
| A（自动真值） | 新增 `beamng_autopilot/experiments/auto_truth.py`、`scripts/m5_auto_truth_probe.py`、`tests/test_auto_truth_*.py`；`beamng_autopilot/experiments/credentials.py`（只加 `truth_provenance` 暴露与兼容读取） |
| B（训练/采样） | `scripts/m5_train_seg.py`、新增 `beamng_autopilot/vision/sampling.py`、`tests/test_train_step_budget.py`、`tests/test_sampling_rotation.py`、`tests/test_train_init_provenance.py` |
| C（来源/适用性） | `beamng_autopilot/experiments/gates.py`、`candidate_metrics.py`、`negative_scenes.py`、`labels.py`、`tests/test_applicability_linefree.py`、`tests/test_negative_extra_metrics.py`、`tests/test_labels_engine_verified_proof.py` |
| 主 agent | `scripts/m5_seg_autoloop.py`、`scripts/m5_marking_identity_probe.py`、docs/报告、看板补丁建议 |

规则：只改自己名下的文件；**不提交**（主 agent 按模块逐个提交）；不跑训练、
不启动 Tech/游戏、不占 GPU；测试只跑自己的新文件 + 受影响的小集合；报告里给出
文件、测试结果、未决问题与复现命令。实现前先核对已有函数，能复用不新增同义模块。

## 6. 本阶段明确不做

- 不改监控/看板文件（用户未提交）。
- 不放宽任何任务门；不给 `road_off=0`/缺列/UNKNOWN 当通过。
- 不把 `pseudo`/弱标签升格；不用单次 `damage=0` 或目录数量证明能力。
- 不跑全量回归与 Tech（属 Order 2 串行步骤，由主 agent 排队执行）。
