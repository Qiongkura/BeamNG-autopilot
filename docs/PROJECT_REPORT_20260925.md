# 项目开发报告（2026-09-25）

范围：本报告覆盖 2026-09-24 夜到 2026-09-25 的开发轮（T14 后台自学习线的加固与推进）。
逐轮原始证据在 `logs/experiments/t14_auto_20260925*/`，本轮技术细节见
`logs/experiments/t14_auto_20260925/report.md`（§0–§20）。
历史报告：`docs/PROJECT_REPORT_20260924.md`（含 §10 增补与缺陷编号 1–35）。

---

## 1. 一句话结论

**后台自学习闭环已经能无人值守跑通**（自己开游戏采集 → 身份审计 → 数据/标签因子
对照 → 判定落盘 → 自停），并且这一轮把两件长期卡住的事做掉了：
**① 标线通道从"完全不产出"变成"可训练、可测量"**（line IoU 0 → 0.50–0.58，
硬门第一次拿到实数）；**② 推理延迟从"取决于这一帧预测得多碎"变成稳定的 ~14 ms**
（p95 121.7 → 14.4 ms）。模型侧三条旋钮（数据量 / 训练预算 / 容量）都已测到回报边界，
本轮还把 agent 真值**扩到整个训练集（640 帧）**并据此训练了一轮：因子（Tversky 2.0）效果更强（**+0.079，只需 2 个 seed**），但**换真值训练本身没有带来可测量的改进**（同 seed 差 <0.01，远小于 seed 间波动）；同时量出**同一模型在不同场景上的标线 IoU 差 3 倍**（0.15–0.31 vs 0.50–0.58）——指标必须带场景。

---

## 2. 能力矩阵（现状）

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 无人值守采集 | **可跑**：入口自己开 BeamNG.tech、采多视角、身份读自运行中的会话、采完关掉自己起的游戏 | `collect_20260925_121718.json`（4×30 帧、rc=0）、`close_started_game`（缺陷 29） |
| 数据准入 | 逐通道真值 + 身份 + 整组隔离 + 内容重复；缺身份一律拒收 | `rounds_dataset.json`、`m5_seg_dataset_audit.py` |
| 标线真值 | **三级口径**：人（`human_revision`，可晋级）/ agent 逐帧核对（`agent_revision`，可训练可测）/ 引擎弱标签（`engine_annotation_partial`） | `experiments/labels.py` 的 `PAINT_SOURCE_RANK`、`ClassQuality.usable` |
| 训练 | 单视角/多视角（8 视角数据齐）、road-only 或 line 监督、等步数对照、平台期判据（两臂都判） | `m5_train_seg.py`、`rounds` 的 `plateau_baseline_by_seed` |
| 评估 | 像素/候选/几何(UNKNOWN)/性能四层；平凡基线；成对 seed；缺测记 UNKNOWN 不记 0 | `m5_seg_eval_matrix.py`、`decision_*.json` |
| 判定与晋级 | 硬门取自实测；研究臂（弱/agent 真值）**不允许晋级**；评估口径 = `checkpoint_last.pt`（已复核） | `_block_research_promotion`、`eval_criterion.json` |
| 资源与安全 | 单实例锁、资源门、机器级 GPU 账本（上限可配，`<=0` = 不设上限） | `controller.py`、`gpu_ledger_machine.json` |
| 看板 | 一页全量：99 个 run、采集总表、判定/超参/GPU 表；42 个表头全部带含义；超参表读最新 checkpoint | <http://127.0.0.1:8761/>、`logs/dashboards/index.html` |
| 计时协议 | 静默前置检查 + 重复测量取较小值；`timing_suspect` 只查同 seed，跨 seed 要看分解 | `m5_seg_timing_retest.py` |

---

## 3. 本轮新增能力（7 项）

1. **标线通道打开**：`engine_annotation_partial`（弱监督）与 `agent_revision`
   （逐帧核对式）两种来源；训练器按 `usable` 而非 `valid` 决定是否屏蔽 line 通道。
2. **agent 逐帧核对式标注工具**（`scripts/m5_line_truth_agent.py`）：规则提议 +
   逐帧目视复核（32 帧 4 视角全看过），扩到整个训练/开发集（640 帧）；
   未知区写 `255(ignore)`（指标与损失都尊重它）。
3. **容量旋钮** `SegUNet(width=...)`（默认 1.0 与原模型逐位一致）+ 评估链按 checkpoint
   自证结构建模型（`arch_args.width`）；`best.pt` 与 `checkpoint_last.pt` 同 payload。
4. **后处理向量化**：`constrain_line_to_road` / `filter_line_shape` 从"每块扫全帧"
   改成直方图 + LUT（判据逐位不变）。
5. **评估口径可复核工具**（`m5_seg_eval_criterion.py`）：last vs best 两种口径都算出来
   再定；两个 run 上结论一致 → 维持 `last`。
6. **研究臂纪律**：`--paint-source RUN=SOURCE` + `--research-arm`，弱真值下
   `shadow_candidate` 自动降级为 `needs_evidence`。
7. **机器级 GPU 账本**（每日上限按机器合计，`<=0` 不设上限）。

---

## 4. 实测结论汇总（都可复现）

### 4.1 因子对照（两臂、等步数、成对 seed、两臂平台期）

| 因子 | 结果 | 判定 | 需要 seed |
| --- | --- | --- | --- |
| 加一段新采集数据（30–120 帧） | +0.005 | inconclusive | 14 |
| 训练预算 120→240 步 | **+0.0092** | **candidate_better** | 4 |
| 训练预算 240→480 步 | +0.004 | inconclusive | 18 |
| 容量 width 1.0→2.0（参数 ×4） | +0.004 / −0.006（两口径符号相反） | inconclusive | 61 / 10 |
| **line 监督开**（road-only → line） | **line IoU 0 → 0.13–0.32**（弱真值）/ **0.50–0.58**（agent 真值，collection #3 的 16 帧） | 能力达成 | — |
| 训练标签：引擎弱标签 → agent 真值 | 同 seed 差 <0.01（0.1601→0.1506 / 0.2062→0.2006） | **无可测量改进** | — |
| **line Tversky 1.0→2.0（agent 真值下）** | **+0.0790**（5/5 为正） | **candidate_better** | **2** |
| line Tversky 权重 1.0→2.0 | **+0.052** | **candidate_better** | 4–5 |
| 后处理离路阈值 0.5→0.9 | +0.057 / −0.025 / +0.004（跨 seed 跨零） | **不改默认值** | — |

### 4.2 跨路泛化（用今天采的三段新路，均未参与训练）

* **道路**通道泛化良好：未见过的路段上 road IoU **0.919–0.939**（width 2 / width 1）。
* **标线**通道此前为 0（road-only 配方完全不产出标线）。

### 4.3 延迟（静默协议重测）

| 量 | 改前 | 改后 |
| --- | --- | --- |
| 端到端 p95（碎掩码那类 checkpoint） | 121.69 ms | **14.42 ms** |
| 跨 seed 离散 | 8.7× | **1.03×** |
| 纯前向（各 seed） | 4.2–4.3 ms（本来稳定） | 4.2–4.3 ms |

### 4.4 硬门实数（第一次有数字，来源 = agent 真值）

`line_recall 0.690–0.728`（门槛 0.70，取决于场景，**擦线**）、
`line_precision 0.262–0.649`（门槛 0.40，取决于真值完整度与场景）、
`offroad_false_ratio 0.203–0.223`（门槛 0.10 **不过**）、
`candidate_identity_rate 0.149–0.158`（门槛 0.60 **不过**，与历史 6–19% 一致）、
`inference_ms_p95 15.6 ms`（门槛 45 ✓）。

**场景相关性（本轮新量出，重要）**：同一模型、同一套真值口径，
在 collection #3 的 16 帧上 line IoU **0.50–0.58**，在 wide+plain 的 55 帧上只有
**0.15–0.31**——差 3 倍；过预测程度也随场景变化（wide/plain 上预测 26 万 px vs 真值约
8 万 px，collection #3 上 precision 0.64 并不过预测）。**任何标线指标都必须带场景**。

---

## 5. 真值体系与本轮最重要的一课

**指标可以因为真值不完整而系统性变差。** 实测：同一批模型在引擎弱真值（漏标约四成
漆线）下 line IoU 只有 0.13–0.32；换上逐帧核对的完整真值后是 **0.50–0.58**。
之前"precision 差 = 模型过预测"的诊断，有相当一部分是**真值偏差**造成的假象。

三级口径（`PAINT_SOURCE_RANK`）：

| 来源 | valid（可当门槛真值） | usable（可训练/可测） | 用途 |
| --- | --- | --- | --- |
| `human_revision` | ✅ | ✅ | 晋级判定 |
| `agent_revision` | ❌（需人确认） | ✅ | 训练 + 测量 + 报数 |
| `engine_annotation_partial` | ❌ | ✅ | 训练（弱监督） |
| `engine_annotation`（默认） | ❌ | ❌ | 与历史行为一致（屏蔽 line 通道） |

**agent 标注的诚实边界**：32 帧 4 视角逐帧目视复核（`front_main`/`front_fisheye`/
`pillar_left` 24/24 与可见漆线一致），复核过程抓到并修掉两处问题：① `pillar_right`
一处假阳（树丛边亮块）→ 该视角不补 RGB 线；② 饱和度阈值 45→20（真漆线 1.2–12 vs
天际线雾 30–32；亮度与形状都分不开）——这是"逐帧看"抓出来的，纯统计扫不出来。

---

## 6. 缺陷清单（本轮新增，编号接 20260924 报告的 23）

| # | 缺陷 | 修法 |
| --- | --- | --- |
| 24 | 采集子进程用系统 Python（无 `beamngpy`）→ 白起一局 | `collector_python()` + 启动前 `python_can_import()` |
| 25 | 游戏进程名只按 Steam 版写（Tech 是 `BeamNG.tech.x64.exe`） | `GAME_IMAGES` 补齐 + 测试 |
| 26 | 采集输出被父进程用管道收走，游戏继承写端 → 父进程卡 30 min | 输出重定向到日志文件 |
| 27 | 看板把"没有帧"的采集漆线写成 0（0 帧是未测） | 三态显示 + 测试 |
| 28 | 因子改了轮数而判定文件仍写基线轮数 | `factor_epochs()` |
| 29 | **采集后不关自己启动的游戏**（4.4 GB 挂 1.5 h，拖慢后续几轮） | `close_started_game()`（只杀本次新起的差集） |
| 30 | 容量臂 checkpoint 装不进评估链（评估链不读 width；`best.pt` 缺 metadata） | 评估链读 `arch_args.width`；`best.pt` 同 payload |
| 31 | `all_at_plateau` 赋值在判定字典之后 → 写盘崩（白花 15 min GPU） | 赋值提前 + "整轮走到写判定"回归测试 |
| 32 | 平台期守卫只判候选臂 | 两臂都判（`plateau_baseline_by_seed`） |
| 33 | 每日 GPU 上限按 run 记账，换 run-id 归零 | 机器级账本 + 门禁取机器合计 |
| 34 | 后处理延迟取决于预测内容（每块扫全帧） | 直方图 + LUT（判据逐位不变） |
| 35 | 不完整真值系统性压低标线指标 | 三级真值口径 + agent 核对式标注 |
| 36 | `--paint-source` 的键与 manifest 查找口径不一致（静默回落默认来源，审计报告失真） | 同时登记规范化路径键 + 测试 |

历史缺陷（1–23）见 `docs/PROJECT_REPORT_20260924.md` §3.2 与 §10.4。

---

## 7. 卡点（需要你）

1. **晋级仍被真值卡住**：agent 真值 `valid=False`（机器画的不能自己给自己发通行证）。
   两条路：① 你花 ~10 分钟用标注器把这批标注复核一遍（它会把这批当初值，
   只需改错的地方：命令见 `logs/m5_seg/line_truth_agent_20260925/README.md`），
   之后按 `human_revision` 记 → 硬门可用于晋级；② 你直接认可 `agent_revision`
   作为真值（改一行 rank），我照此执行——但记录会写清"真值由 agent 提供并经用户认可"。
2. **空闲时间窗 / 用户活动暂停**仍未指定（当前 0–24 全天、不做活动检测；
   今日机器上还有 `cs2` 在跑，训练/计时都会受影响）。GPU 上限已按你的指示取消
   （`<=0` 不设上限，账本照记）。
3. **驾驶闭环（阶段 F）**仍然未测：完整 tick deadline、压线/出铺装计数、
   非终点停车时长都需要真车/闭环运行，而它们又依赖第 1 条。

---

## 8. 下一步（按优先级）

1. **标线真正的提升**：目前主要矛盾是 precision（假线），且训练集只有 25 帧
   front_main。建议：① 用 agent 真值把**多视角**数据（town 8 视角 200 帧）纳入训练
   （本轮已标注完，只差一轮对照）；② 或按你的确认升级到 human_revision。
2. **预算/容量/数据三条线已到边界**：不要在这三条上继续烧 GPU（有实测数字支撑）；
   要动就动**输入/任务**（多视角融合、时序）。
3. **跨路稳定性口径**：建议把"最差路段 IoU"作为固定报告项（现在只有单点数字）。
4. **看板**：把"每轮采集 → 候选 → 判定"串成时间序（现在三张表各自独立）。

---

## 9. 复现命令

```pwsh
# 质量门（离线，不需要游戏）
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\dev_validate.ps1

# 无人值守一轮（采集 + 训练 + 判定）
python -u scripts\m5_seg_autoloop.py run --run-id <id> --no-dry-run --config <entry_config.json>

# agent 逐帧核对式标线标注
python scripts\m5_line_truth_agent.py --config <agent_config.json> --out <out_dir>

# 静默协议重测推理延迟
python scripts\m5_seg_timing_retest.py --model s42=<ckpt> --dev-runs <dirs> --repeats 3 --out <json>

# 评估口径复核（last vs best）
python scripts\m5_seg_eval_criterion.py --run-dir <run> --arm baseline --arm round0 --dev-runs <dirs>

# 人工修订任务包（画完回传即可训练/判定）
python scripts\m5_annotate_package.py --review-queue <queue.json> --out <pkg> --per-view 8
```

---

## 10. 证据索引

| 主题 | 产物 |
| --- | --- |
| 本轮逐轮细节（§0–§20） | `logs/experiments/t14_auto_20260925/report.md` |
| 四/五/六轮判定与容量复现 | `logs/experiments/t14_auto_20260925{,_r2,_r3,_steps,_steps96,_width2,_width2b}/decision_*.json` |
| 标线通道（弱真值 / agent 真值） | `logs/experiments/t14_auto_20260925_line/`、`..._agentline/` |
| agent 逐帧核对式标注 | `logs/m5_seg/line_truth_agent_20260925/`（含 README）、`logs/m5_seg/line_truth_agent_full_20260925/`（640 帧） |
| 采集记录与复核队列 | `logs/experiments/t14_auto_20260925/collect_*.json`、`review_queue_collect_*.json` |
| 计时与口径复核 | `timing_retest_round3.json`、`timing_retest_after_vec.json`、`*_width2*/eval_criterion.json` |
| 看板 | `logs/dashboards/index.html`（服务地址 <http://127.0.0.1:8761/>） |
| 历史项目报告 | `docs/PROJECT_REPORT_20260924.md`（缺陷 1–23、§10 增补） |
| 授权与运行参数记录 | `docs/T14_PROGRESS_20260924.md` §4–§6 |
