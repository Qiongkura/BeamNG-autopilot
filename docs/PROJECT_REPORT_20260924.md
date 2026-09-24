# 项目完整报告（beamng-autopilot，2026-09-24）

本报告汇总**当前工作树的真实状态**：做了什么、数字是多少、哪些还没测、下一步卡在哪。
每条结论都能顺着指针回到原始产物；**未测项一律写"未测"，不写成 0，也不写成"通过"**。

- 工作树：分支 `fix/round3-hardening-20260921`，HEAD `8f4c5e2`，538 个提交、518 个跟踪文件、约 **12.4 万行 Python**、**156 个测试文件**
- 质量门（最近一次全量）：`pytest tests/` **2469 passed**；`scripts/m5_offline_validate.py` **ALL PASS**；`scripts/check_commit_scope.py` 干净；已跟踪文件无未提交改动
- 计划依据：`docs/FINAL_INTEGRATED_PLAN_20260922.md`（T01–T12、§6 阶段门、§7 验收口径、§9 禁止事项）、
  `docs/T14_BACKGROUND_MODEL_ITERATION_PLAN_20260924.md`（阶段 0/A–F）、`docs/LITERATURE_DEVELOPMENT_20260922.md`

---

## 1. 一句话结论

**工程侧**已经到"可以无人值守跑离线实验迭代"的水平：数据准入、可复现训练与恢复、四层评估、
事前冻结的晋级门槛、自动淘汰与自行停止、实时可视化与固定探针，全部落地并在真实数据上跑过。
**模型侧**仍是**研究级、不能上路**：本图冻结集上比生产模型好约 2 倍，但 seed 不稳（同配方跨路
IoU 0.0–0.68，会出现整条路失效）、候选身份率 6–19% 远低于 60% 门槛、所有驾驶闭环项**未测**，
生产权重**一个都没替换**。

---

## 2. 模型水平（感知候选模型）

### 2.1 标线像素层（冻结集 50 帧 = 两条从未参与任何选择的路段，`frozen_holdout_t13` digest `5828001add8ae9a5`）

| 模型 | precision | recall | IoU | 路外假线像素 | 推理 p50 |
| --- | --- | --- | --- | --- | --- |
| 生产 `logs/m5_seg/seg_model/best.pt`（sha256 `b16a734d67f3c0de…`） | 0.2577 | 0.5856 | **0.2180** | 82 802 | 12.3 ms |
| 微调初值 `seg_model_v13b/best.pt` | 0.1990 | 0.9832 | 0.1984 | 88 961 | 25.1 ms |
| armA seed42（173 帧，3 轮） | **0.5535** | 0.8181 | **0.4929** | 6 252 | 11.3 ms |
| armA seed43 / seed44 | 0.4164 / 0.4965 | 0.7686 / 0.8133 | 0.3700 / 0.4457 | 38 626 / 22 094 | 13.8 / 11.8 ms |
| armB seed42 / 43 / 44（227 帧） | 0.5055 / 0.5277 / 0.5402 | 0.8625 / 0.7334 / 0.8386 | 0.4678 / 0.4428 / 0.4893 | 24 981 / 13 029 / 17 160 | 11.1 / 11.8 / 11.8 ms |

证据：`logs/m5_seg/seg_t13_data_20260924/06_eval_matrix.json`、报告 `docs/T13_LOCAL_TRAINING_20260924.md`。
**数据层修复是有效且幅度大的**（IoU 1.7–2.3×、precision 1.6–2.1×），但**本轮的"只加场景数据"单因子没有净收益**
（armA 均值 0.4362±0.0620 vs armB 0.4666±0.0233，差值小于臂内 seed 波动）——按 T13 方案的自动停止条件，该实验已停止。

### 2.2 候选层（逐候选引擎漆线确认 + 零假设对照，两段冻结路各 25 帧）

| 模型 | 匹配率 | 零假设 | ×机会 | 角色一致率 | 落漆线候选 | 路外候选占比 |
| --- | --- | --- | --- | --- | --- | --- |
| 生产 | 0.103 / 0.209 | 0.084 / 0.008 | 1.2 / 26.8 | 0.55 / 0.59 | 1 / 6 | 74% / 94% |
| v13b | 0.215 / 0.162 | 0.089 / 0.078 | 2.4 / 2.1 | 0.45 / 0.55 | 1 / 4 | 70% / 46% |
| armA42 | 0.235 / 0.142 | 0.046 / 0.025 | **5.2 / 5.7** | **0.94 / 0.82** | **28 / 7** | 71% / 77% |
| armB44 | 0.197 / 0.127 | 0.066 / 0.017 | 3.0 / 7.5 | 0.79 / 0.73 | 19 / 5 | 76% / 80% |

证据：`logs/m5_seg/seg_t13_data_20260924/ident_t13_test*.json`。

### 2.3 三个真瓶颈（都有数字）

1. **候选身份率 6–19%，门槛 60%**：T14 三轮自动循环**全被这条硬门槛拦下**；T08 的结论一致。
   这不是"线画得好不好"，而是"感知候选 ↔ 真实标线"的对应关系。
2. **seed 不稳 / 跨路失效**：同配方跨 seed 在开发路上 IoU 0.0–0.68，出现**整条路 IoU=0**；
   `--epochs 3→6` 无改善。T13 冻结集上 0.37–0.49 的稳定区间**不能外推**到别的路。
3. **跨地图结构性缺口**：跨图召回 0.322→0.454（加数据）但精度掉到 0.197；本图 0.94–0.99。
   已登记为**不阻塞接线**的研究项。

### 2.3b 身份率的归因（最新一轮实测，决定下一步做什么）

用已有探针产物（8 份 `ident_*.json`、1055 个候选、870 个未匹配）拆解，**两条设疑都已判定**：

| 未匹配候选的归因 | 数量 | 占比 |
| --- | --- | --- |
| 离引擎漆线 **≥2 m**（帧内有漆线）→ 候选在别的笔画上 | **766** | **88.0%** |
| 其中"镜像位置"（带符号距 ≥1 m、绝对距 <1 m） | 139 | 占未匹配 16.0% |
| 贴近（<1 m）却没配上 → 匹配规则边界 | 21 | 2.4% |
| 帧内完全没有引擎线 → 参考不可用 | 19 | 2.2% |

1. **镜像 100% 是假笔画，不是投影符号问题**：判据是候选**自己的像素**是否落在漆线上
   （`on_line_frac ≥ 0.5` 且被投影到镜像位置才算几何缺陷）——实测 **0 个**属此列，
   **139 个全是不在漆线上的假笔画**。**投影几何没有被抓到缺陷**（不必去动 T05 的几何链）。
2. **主因是候选错位/假线**：88% 的未匹配候选离任何真漆线 ≥2 m；按并集口径分层，
   远距候选 634 个中**铺装内错位 117、铺装外假线 517**（82% 是铺装外、且以 `thin` 细笔画为主）。
3. **匹配规则与参考质量都不是主因**（2.4% / 2.2%）；按帧内漆线像素分层的匹配率
   0.080 → 0.149 → 0.194 说明参考越密越好，但最密也只有 19%。

**口径提醒（可复用的教训）**："铺装外"必须按**路面 ∪ 漆线**判（两者是互斥类）；
若只看"路面类"，落在漆线上的候选会被必然算成"铺装外"。仓库工具的 `candidates_off_road`
本来就是并集口径（在 T13 探针集上按路 70–94%、合并 **68.2%**）；本轮我临时脚本的第一版用了
单类口径（77.9%/82.8%），属**我自己的分析错误**，已在本报告更正。

### 2.4 明确"不能声称"的

- **不能上路**：闭环未测（见 §5），几何层无独立真值 = UNKNOWN。
- **GPU 上不能声称"逐位可复现"**：CUDA 没有 `nll_loss2d` 的确定性实现，只能给实测容差（见 §3.3）。
- **不能把 IoU 当安全成绩**：IoU 是训练诊断与辅助门槛；主指标是身份/漏线/精度/路外假线/延迟（§2 门槛表）。

---

## 3. 工程水平（自主迭代基础设施，T14 阶段 0/A/B/C/D）

### 3.1 阶段表（对照 T14 实施顺序表）

| 阶段 | 状态 | 关键证据 |
| --- | --- | --- |
| 0 冻结协议 | **完成** | `docs/t14_thresholds_v2.json`（config_hash `f350b13313baa9db`，v1 `29a6f387201db6a4`）：改一个阈值就被拒；**加载时校验 config_hash**，手改当场拒绝 |
| A 数据与标签 | **完成** | 逐通道覆盖（road/paint/pavement 分开）；**标线真值不可用 → 拒绝训练并进 `needs_review`**（实测 173 帧里 168 290 个"标线像素"被判不可信）；不可变 `dataset_id`；内容/曝光/组泄漏检查 |
| B 训练与恢复 | **完成** | **中断续训 = 未中断**（CPU+`--deterministic` 逐位一致 n_diff=0，固化为 `tests/test_seg_resume.py`）；**GPU 续训容差实测**（控制组 3.26e-3、续训 2.87e-3、比值 **0.88** → 落在噪声带内，判据 1e-2）；**NaN / 标签污染立即失败**（故障注入测过）；小样本过拟合 2.82→1.07 证明梯度链 |
| C 学习看板 + 探针 | **完成** | 实时 12 图监控页（`docs/TRAINING_MONITOR_20260924.md`，含三张实机截图）；7 视图看板；**固定探针图三档场景 + 同帧四阶段后处理对照 + `full_predict` 交叉检查** |
| D 自动搜索 | **完成（离线）** | 真实数据上的**净负收益自动淘汰**；**连续 2–3 轮无人值守跑完并自行停止**；**决策重放逐字一致**；`run --once --dry-run` 只打印不执行。**3h 轮补齐接线**：数据因子真改输入、基线臂独立、按 seed 配对、未生效即拒绝训练（提交 `b7e687c`） |
| E 新数据闭环 | **未测** | 需要用户授权（§6） |
| F 驾驶晋级 | **未测** | 同上；离线判定最高只给 `shadow_candidate`，且明确不覆盖生产模型 |

### 3.2 本轮修掉的真实缺陷（每条都有反例/触发证据）

| # | 缺陷 | 触发证据 | 修法 |
| --- | --- | --- | --- |
| 1 | **新写出的 checkpoint 在部署链加载失败** | `torch.load(weights_only=True)`（torch 2.6+ 默认）拒绝 `TorchVersion` 对象与 numpy 数组 | 数据侧改存纯字符串/整数列表；加回归测试 |
| 2 | 非有限损失会一路写进权重，且"跑完 N 轮"看着像成功 | 无守卫 | 逐 step 检查非有限即失败 + 写 `failed` 记录（故障注入测过） |
| 3 | 标签污染被静默当背景 | 无校验 | 加载时校验标签 ∈{0,1,2,255}，否则报类别与文件 |
| 4 | 重复内容/同名路径的泄漏审计误报 | 6 采集清单产生 39 个假重复组 | 路径按 (run,path) 判；`checked` 语义改为"证据是否可执行" |
| 5 | 采集器硬编码 `map_name="italy"` | EC/gridmap 两组采集身份错误 | 从 `scenario.get_current().level` 读会话地图 + 记录来源 |
| 6 | 训练入口 `per_run` 用 basename 为键 | 6 个 `front_main` 塌成 1 组 → `--split by-map-scene` 只训练 25/173 帧却照常打印"共 173 帧" | 键改为路径唯一；父目录 meta 按视角过滤 |
| 7 | 一次非有限值毁掉整包 JSON | `Infinity` 非法 JSON → 页面 0 条数据 | 写入前消毒 + `allow_nan=False`；读取端救回旧文件 |
| 8 | 直方图把假轴对象传给标注函数 → 中断整页渲染 | `axis.sy is not a function`，硬件图停在"未采集" | 传真轴 + 每图单独 try/catch |
| 9 | 判定把次要指标 0 差异当"不合格" | 阻止了合法晋级 | 主指标要"≥1 项可信改善且不得可信变差"；辅助指标只需非劣 |
| 10 | 提议器在最高错误率桶无对应因子时掉到最次要族 | `candidate_off_road` 94% 却提议 `epochs` | 该桶触发场景配比；factor 只留真正改动的键 |
| 11 | 计划打印的评估命令跑不通 | `--checkpoint` vs CLI 的 `--pairings/--hard-gate` | 修正命令 + 测试断言"计划里的 flag 必须存在于子命令 help" |
| 12 | **冠军没有跨轮持久化** | `round1/2_decision.json` 的 `champ` 为空 → "净收益"判据从未被比较 | 冠军写 `champion.json` 逐轮读取；seed 数不一致取较小者并警告 |
| 13 | 轮与候选身份混用 / 候选 id 复用 | `rejected -> evaluating` 被状态机拒绝（两次） | 每候选独立事件流；候选 id 带轮号 |
| 14 | 提议因子（数据集组成键）被当命令行开关 | argparse 报 `unrecognized arguments` | 白名单映射训练器参数，跳过项记账 |
| 15 | 探针目录残留上一轮旧图 | 目录里的图与清单不一致 | 写入前清理本目录探针图 |
| 16 | 我自己的实验设计错：改 `--epochs` 当"中断" | 续训那轮 `lr=0`（换的是 LR 计划） | 加 `--stop-after`；把这条坑写进测试文档串 |
| 17 | **手工标注导出丢身份** | 48 帧人工修订全部被判 `no map identity` → 标签无法分组/定位/划分 | 导出写 `map_name/source_id/pos/heading` 进 npz + `<out>/meta.json` 边车；缺什么记 `identity_missing`，**不默认地图名**（提交 `204a94e`） |
| 18 | **旧 meta 的地图身份无来源** | 28 个采集目录只有 8 个有 `map_name_source`；20 个（2508 帧）值无来源，其中 5 个（`gm_*`/`ecrev_*`/`holdout_eastcoast`）目录名与 `italy` 矛盾 | 审计对"有值无来源"点名（提示不改判）；旧值在独立证据前不得用于决策（提交 `4532cb7`） |
| 19 | **`rounds` 三处接线缺陷** | ① 数据因子只打印 `skipped_factors` 就照训（候选臂=基线臂）；② 第 0 轮拿候选自己的 IoU 当 champion；③ 按位置配对 seed、数量不一致就截断；④ 判定文件命名让 `replay` 重放不到 | `add_runs/drop_runs` 真改 `--runs`，未生效即拒绝训练；第 0 轮训练**基线臂**并落 `champion.json`；按 seed 键控配对，不一致拒绝；判定改名 `decision_{cand}.json` 并带 replay 四字段（提交 `b7e687c`） |
| 22 | **"忽略某类"被实现成"权重置零"** | 权重置零只取消正样本拉力，未标注像素仍在 softmax 分母里当负样本 → 教"未标注的可见漆线=背景"（方案 §1 禁止） | `masked_cross_entropy`：被屏蔽类移出分母，梯度恒为 0；训练器 `--ignore-line-class`（判据来自标签审计）；评估补 `road_iou`；`rounds` road-only 配方与主指标（提交 `6bc1f37`/`081f047`/`2ce6483`/`41ea53d`） |
| 23 | **硬门输入用命令行阈值冒充测量** | `hard_gate.inference_ms_p95 = 45.0`（是阈值不是测量） | 四项硬门输入改取本轮实测，缺测记 UNKNOWN；`--hard-*` 标注废弃 |
| 21 | **提议器发训练器没实现的因子** | `scene_mix` 发 `group_weights`、困难采样发 `hard_neg_manifest`；`loss_weights` 的 `continue` 连带跳过 `epochs` 族 | `APPLICABLE_KEYS` 只放可执行键；场景配比族改发 `add_runs`（`--available-runs` 的真实目录）；无数据时记 blocked 并停下；每族独立判断（提交 `60512f1`、`d5d5722`） |
| 20 | **`--allow-road-only` 只是"审计层许可"，训练侧不成立** | `mask_line_for_loss` 只改**已标注**的 line 像素；训练器是带类别权重的 CE，无类屏蔽 → 置零 line 权重 = 把未标注漆线当负例（方案 §1 禁止） | 审计门内建进 `rounds`（真实数据实测：不带开关判 `needs_review`、带开关判 `road_only_not_implemented`，均退出 3 且不训练）；整通道忽略列为下一步能力项 |

### 3.3 可复现性的适用域（重要）

| 场景 | 结论 | 证据 |
| --- | --- | --- |
| CPU + `--deterministic`，同配置两次 | **逐位一致**（n_diff 0/106） | `tests/test_seg_resume.py` |
| CPU + `--deterministic`，中断续训 vs 未中断 | **逐位一致**（n_diff 0/106） | 同上 |
| GPU（AMP）+ 同配置两次 | 不一致，max_abs 1.4e-3…3.9e-2（噪声基线） | `logs/experiments/t14_gpu_tol/tolerance.json` |
| GPU（AMP）+ 中断续训 | 差异 **≤** 噪声基线（比值 0.88）→ 按容差 1e-2 判为等价 | 同上 + `docs/t14_thresholds_v2.json` |

CUDA 上无法逐位：`nll_loss2d`（本项目交叉熵）**没有确定性实现**，`use_deterministic_algorithms(True)` 直接报错。

---

## 4. 还没做/做不到的（三态口径，不混写）

| 项 | 三态 | 原因/缺口 |
| --- | --- | --- |
| T07 边界精度（定位误差/漏边界/假边界率） | **UNKNOWN** | 无独立几何真值（场景矩阵 8 行已齐，缺精度类证据） |
| T08 地图先验"减少错误关联" | **FAIL（实测 0 处净减少）** | 路口 8/8 硬拒绝全在路外候选（与道路存在性判据重复）；折线切线在路口**有害**（3/3 漆线确认被误拒） |
| T09 横向风险实车量测 | **UNKNOWN** | 36/36 tick 的 gap/first_crossing 为 UNKNOWN（该路段无双边边界） |
| 压线/出铺装/碰撞/非终点停车/deadline | **未测** | 无 Tech 闭环（八项 ③④⑤⑦） |
| 跨图召回 | **部分** | 0.322→0.454（加数据）但精度 →0.197；结构性 |
| CUDA OOM 状态 | **未注入测试** | 只有预算耗尽与数据缺失路径有回归 |
| T10 标注 GUI 与"分钟/有效标签" | **未做** | 需人工点击 |
| 手工标注集的**可信身份** | **缺口（已定量）** | 28 个带 `map_name` 的目录只有 8 个有来源；20 个（2508 帧）无来源、5 个目录名与 `italy` 矛盾；48 帧人工修订因无身份被拒且**不可恢复**（见 3h 轮 `report.md` 结论 1–4） |
| 3h 轮的训练臂（步骤 3/5） | **未开始（停止门）** | 数据准入门不通过：无"带身份 + 可信漆线真值"的数据；按纪律不开训练 |
| 训练器"整条标线通道忽略" | **已实现** | `masked_cross_entropy`（移出 softmax 分母、梯度恒为 0）+ `--ignore-line-class`（审计驱动）+ 评估补 `road_iou`；真实两臂 3 seed 已跑（机制测试，因果结论 needs_evidence：非等步数） |
| 引擎 line 类作为漆线真值 | **不可用（已量化）** | 三条开发路引擎线像素左/右 = 0.021/0.234/0.193；`diverse_wide` 19/30 帧左侧为 0；放大图里**沥青上的白色左边缘线没有引擎标注**。→ 身份率/假线比在"无参考侧"一律 UNKNOWN（268 个未匹配候选中 102 个属此类），列 `review_queue_candidates.json` |

---

## 5. 铁律与禁止事项自查（计划 §9 / AGENTS.md）

- **横向定位只用感知**：全链条未出现"导航线/地图线 + 固定偏移"做横向控制或压线判断；
  T08 的地图先验全程**影子**，且 `AssociationResult` 结构上不存在横向几何（三条测试钉住）。
- **不把 UNKNOWN 当 0 / 不当 PASS**：监控页与看板对未测显式写"未测/不提供"；三角态台账
  `docs/ACCEPTANCE_T07_T08_T09_20260924.md` 逐项给 PASS/FAIL/UNKNOWN。
- **不修改运行时生成物**：`logs/`、`weights/`、`.yolo/` 等只增不改（本轮所有产物写在新目录）。
- **不还原用户改动**：工作树既有未提交改动一律先读再改；未 `reset --hard`、未 `clean`。
- **提交纪律**：改动按模块（`beamng_autopilot/<子包>` / `scripts` / `tests` / `docs` / 顶层文件）分开提交；
  每个提交自身通过 `pytest`；推送前跑 `check_commit_scope.py`。
- **不把 4×4 多视角样例说成 200 帧能力**、**不用单次 `damage=0` 证明安全**：本报告的所有能力声明都跟随其样本量。

---

## 6. 卡在用户侧的唯一事项（T14 §136）

阶段 E（新数据闭环）与 F（驾驶晋级）**未测**，因为没有这两项授权（已问过两次，未答复，按默认继续）：

1. **是否允许无人值守时自动启动 BeamNG.tech 采集**：当前 **`collect=off` 长期运行**，只用现有数据；
2. **空闲时间窗 / 每日 GPU 时长上限 / 用户使用电脑时是否立即暂停**：建议 **00:00–08:00 / 120 min 日 / 立即暂停**
   （已写进 `docs/t14_loop_config.example.json` 的 `_pending_user_params`）。

**授权后的执行顺序（runbook 已在 `docs/T14_PROGRESS_20260924.md`）**：
填配置 → `run --once --dry-run` 复核命令 → 起采集前确认无驾驶会话 → 准入门（标线真值不可用则转复核队列）
→ 新标注内容哈希进下一版数据 → 只有离线判定为 `shadow_candidate` 的候选才进 `--seg-model` 短程 Tech 对照
（错误接受不增加、压线/出铺装不增加、非终点停车净改善、deadline 不恶化）→ 八项证据齐全才提请人工确认部署，
生产权重不自动替换并保留回滚。

---

## 7. 下一步优先级（建议）

1. **候选身份率**（唯一能显著改变结论的单项）：现在 6–19% vs 门槛 60%。**归因已做**（§2.3b）：
   主因是"候选不在真线上"（88%），其中 82% 是**铺装外假细笔画**、18% 是铺装内错位，
   镜像不是几何问题、匹配规则也不是瓶颈。→ 下一步**不是**候选门：把"连 learned 候选也要求在模型自己路面区域内"做成开关实测后**被否**——
   收紧恰好删掉 21 个真标线候选（落漆线 21→0、匹配率 0.217→0.017），而路外假线只减 75→50。
   原因是**空间不一致**：引擎标签空间的"铺装外"不等于模型路面掩码之外（漆线就在掩码更窄处）。
   而且最直觉的"路面 ∪ 线掩码"写法是**同义反复**（`learned_frac` 本就是"在 line 掩码里的占比"），
   被单测当场抓住。→ 真正要动的是**独立的铺装面来源**或**线通道的数据/标签**。
2. **跨路稳定性**：同配方跨路 IoU 0.0–0.68 → 需要更多路段的训练数据与"最差路段"报告口径。
3. **授权 E/F**，让八项 ③④⑤⑦ 从"未测"变成实数；否则驾驶层永远只有离线结论。
4. 补 **CUDA OOM 注入测试**与 `--deterministic` 对训练速度的影响（成本低、有明确做法）。

---

## 8. 复现入口（命令清单）

```pwsh
# 质量门（离线，不需要游戏）
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\dev_validate.ps1

# 数据集准入审计（逐通道覆盖 + 泄漏 + 内容重复）
.venv\Scripts\python.exe scripts\m5_seg_dataset_audit.py --runs <帧目录> --digest-scan

# 多 checkpoint 像素/性能矩阵（同一批冻结输入）
.venv\Scripts\python.exe scripts\m5_seg_eval_matrix.py --model 名字=权重 --runs <目录> --json <报告>

# 固定探针图（三档场景 + 四阶段后处理对照）
.venv\Scripts\python.exe scripts\m5_seg_probe_images.py --model <ckpt> --runs <目录...> --run-id <run>

# 训练 + 实时监控（12 图）
.venv\Scripts\python.exe scripts\m5_train_seg.py --runs <目录> --epochs 3 --metrics-run <run> --out <目录>
.venv\Scripts\python.exe scripts\m5_train_monitor.py serve --run-id <run> --port 8760

# 离线自动循环（审计→提议→评估→判定→重放→多轮）
.venv\Scripts\python.exe scripts\m5_seg_autoloop.py audit|propose|evaluate|replay|rounds|run --once --dry-run

# 两臂命令逐项 diff（只列命令，不训练）：基线臂 vs 候选臂
.venv\Scripts\python.exe scripts\m5_seg_autoloop.py rounds --run-id <run> `
    --runs <训练目录...> --baseline-runs <基线目录...> --eval-runs <开发目录...> `
    --proposals <提议.json> --plan-only

# 手工标注（导出携带来源身份：npz 里 map_name/source_id/pos/heading + meta.json 边车；
# 记录里没有的身份会写进 identity_missing，审计据此拒绝，绝不默认地图名）
.venv\Scripts\python.exe scripts\m5_annotate_manual.py --frames-dir <帧目录>

# 续训等价性：GPU 容差测量 / CPU 逐位验证
.venv\Scripts\python.exe scripts\m5_seg_resume_tolerance.py --runs <目录> --seeds 42 43 44
.venv\Scripts\python.exe scripts\m5_train_seg.py --device cpu --deterministic --stop-after 2 --resume ...
```

---

## 9. 证据索引（本会话主要产物）

| 主题 | 文档 | 原始产物 |
| --- | --- | --- |
| T13 本地训练（基线冻结、数据准入、两臂 3 seed、四层评估、停止判定） | `docs/T13_LOCAL_TRAINING_20260924.md` | `logs/m5_seg/seg_t13_data_20260924/*`、`logs/goal_20260921/frozen_holdout_t13.json` |
| T14 阶段进度与四条结论 | `docs/T14_PROGRESS_20260924.md` | `logs/experiments/t14_*`（tolerance.json、rounds*_decision.json、probes/*） |
| 训练过程可视化（实时 12 图监控页） | `docs/TRAINING_MONITOR_20260924.md` | `logs/experiments/monitor_shot_*.png`、`logs/experiments/t14_monitor_live/metrics.jsonl` |
| T07/T08/T09 三态验收台账 | `docs/ACCEPTANCE_T07_T08_T09_20260924.md` | `docs/MAP_ASSOCIATION.md` §7/§8、`docs/BOUNDARY_EVIDENCE.md` §5i |
| 冻结阈值与运行配置 | `docs/t14_thresholds_v2.json`、`docs/t14_loop_config.example.json` | — |
| 本轮 16 个缺陷的证据 | 各提交信息 + §3.2 表 | `git log --oneline -20` |
| 3 小时计划轮（停止门 + 身份修复 + rounds 接线） | `docs/T14_PROGRESS_20260924.md` 结论 9/10 | `logs/experiments/t14_3h_20260924_1906/`（`report.md` 八项、`02_dataset_audit.json`、`03_needs_review.json` 48 帧、`04/05` 身份证据、`06/07/08` 回归门日志） |
