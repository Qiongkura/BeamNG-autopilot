# 分阶段发展方案（2026-09-13 起）

> 基于当日遍历：M1–M5 规则栈成熟，M6 FSD 结构已齐；卡点是 **italy 成对车道率**
> 与 **east_coast 实验可信度**（机位/provenance），不是缺算法框架。
> 完成定义、铁律与里程碑编号沿用 `ROADMAP_20260912.md` / `AGENTS.md` / `README.md`。

---

## 0. 项目定位与前景

### 定位
在 **BeamNG.tech** 上做可复现的自动驾驶研究栈：**感知优先的分层 FSD**（对齐特斯拉
AI Day 架构），保留规则/BC/DQN/残差 RL 作仲裁与增量，**不以地图横向兜底换安全**。

### 前景（现实、可达成）

| 层级 | 前景 | 依赖 |
| --- | --- | --- |
| **研究/教学** | 多图感知→BEV→规划→控制闭环、任务口径评测、影子数据闭环，可写课程实验/论文素材 | S0–S3 |
| **工程展示** | 城镇/山地/US 短途“无接管”演示；benchmark scorecard | S2–S3 |
| **学习升级** | 稳定 paired 后的横向残差 RL、E2E 提升；再谈时序/Transformer | S4 |
| **非目标（近中期）** | 真车、大规模 RL in-the-loop、V2X、LLM 决策 | 列为远期，不进主验收 |

### 产品形态
- 主交付：库 `beamng_autopilot/` + 薄入口 `scripts/` + 可复现 pin 权重策略
- 次交付：标注/训练/评测流水线（P5）、benchmark 场景注册表

---

## 1. 现状一页纸（遍历结论）

```
感知   vision/ (UNet+YOLO+黄线/植被先验) + lane/ (配对/融合/参考) + occupancy/BEV
决策   planning/ (轨迹扇/约束/速度剖面/仲裁) + rl/ (DQN 速度 + 横向残差离线)
控制   control/ (PurePursuit / Speed / 反向防护) + safety_monitor
闭环   fsd_stack + fsd_drive + shadow recording + eval/benchmark
数据   labeling/ (宫格/手涂/Tech annotation) + pseudo_us + 土路
测试   tests/ 约 80+ 文件，离线可跑
脚本   scripts/ 大量 probe/train/collect（含临时 _tmp_ 需后续清理）
```

| 能力 | 状态 |
| --- | --- |
| italy 规则驾驶 / FSD 安全项 | 达标（cross/off≈0） |
| italy stall（连续性） | 未达标（paired~19%） |
| east_coast 黄线联合模型离线 | AUC~0.945 |
| east_coast 实车开走 | 未验证（灌木机位无效） |
| 横向残差 RL | 离线 40k 过门；sim 未接 |
| 权重复现 P6 | 仍是最大工程债 |

---

## 2. 阶段总览

```text
S0 实验可信度     ████░░░░░░  本周
S1 italy 连续性   ██████░░░░  2–4 周   ← 主验收
S2 FSD 默认主驾   ████████░░  S1 后 1–2 周
S3 多图泛化       ██████░░░░  与 S1 可部分并行
S4 学习栈硬化     ████░░░░░░  S1 达标后
S5 研究/发布      ██░░░░░░░░  随里程碑
```

---

## S0 — 实验可信度（本周，先做）

**目标**：任何“能力结论”都建立在合法机位与真地图标签上。

| 子项 | 改法 | 验收 |
| --- | --- | --- |
| S0.1 模块提交 | 已完成 2026-09-13 四笔；保持按模块 | `git status` 干净策略 |
| S0.2 provenance | `fsd_drive` shadow `map` 用真实关卡/`args.map`，废除写死 italy；`data_contract` 同步 | 影子 meta 与游戏地图一致；单测 |
| S0.3 出生点门 | photo tour / teleport：可行驶像素、植被占比、亮度门；落 `stops.json` | 灌木点被拒；路面点被收 |
| S0.4 有效判决 | `(246.44, 877.87, -107.62)` 或人筛点，**120s** 严格感知短跑 | 日志可判 paired 连续性 |
| S0.5 数据卫生 | 灌木影子标负例，不进正样本集 | 训练集 manifest 不引用坏影子 |

**不要做**：在树丛点上调 hold/规划/模型。

---

## S1 — italy 成对车道率（主验收，2–4 周）

**目标**：钉住 10 集 `paired 显著 >33%` 且 `in-lane ≥86%`；实车 `stall p50 ≤83`，
且 `crossC/crossR/off = 0`（≥5 臂 × 2 会话）。

### 1.1 优化方向（按证据排序）

1. **数据（唯一已证明）**  
   - 继续 P1a：line-only 手涂 + 宫格；每批 20–40 帧  
   - 只收“当前模型 paired=0 但肉眼有双边线”的帧（主动学习）  
   - **禁止**再走“全量配方±少量真值、val_mIoU 选点”

2. **配对几何（工程，非调参碰运气）**  
   - `pair_lane_markings`：双侧间距 ∈ [2.8, 4.0]m、方向一致、span 门  
   - 闪断：EMA 平滑线位置；连续 3 帧中心一致再置 paired  
   - `HOLD_NONE_FRAMES` 仅作 A/B，默认保持 4

3. **任务口径训练**  
   - `m5_train_seg`：`--task-eval-every 1 --task-min-in-lane` + 固定 episode 名  
   - 选 `best_task.pt`，不选 mIoU best

4. **控制侧**  
   - 已有 PLC/单 owner；只在 paired 时做残差，禁止地图横向

### 1.2 阶段验收命令

```pwsh
.venv\Scripts\python.exe scripts\m5_seg_task_eval.py --model <ckpt> --episode-names <pinned10>
.venv\Scripts\python.exe scripts\m5_live_ab.py   # 实车 A/B，≥5 臂
```

---

## S2 — FSD 成为默认主驾（S1 达标后，1–2 周）

**目标**：`AutopilotSession` / GUI 默认走 FSD，规则栈降级为兜底。

| 项 | 内容 |
| --- | --- |
| 开关 | `--driver fsd|rule` + 场景默认表（mountain/town/free） |
| 回退 | FSD MinimalRisk → rule 慢开，遥测标明 driver_src |
| 冒烟 | `m5_gui_smoke` + 离线 fsd_closed_loop |
| 验收 | 默认路径下 P1 指标不回退；山地无倒车/无上草 |

---

## S3 — 多图泛化（可与 S1 后半并行）

**目标**：east_coast / west_coast 等短途可信；italy 不回退。

| 项 | 改法 |
| --- | --- |
| 数据 | photo tour 人筛 → 宫格 → 伪标签（黄线+veg）→ line-only 手涂关键段 |
| 模型 | joint 优先；难图再训 map specialist（恢复 by_map 仅当任务指标证明） |
| 配对 | US 双黄/白边几何与滞回专项；固定 hold 默认 |
| 规划 | 路网 A* 连通、走廊/障碍裕量分场景常数（禁新“地图横向偏移”） |
| 验收 | 人筛机位 ≥3 臂：放置成功 + 短途无倒车/无上草；italy benchmark 不退 |

**hirochi 路缘 / 赛道**：低优先级，作“域外难例”专项。

---

## S4 — 学习栈硬化（S1 达标后）

| 顺序 | 项 | 门禁 |
| --- | --- | --- |
| 1 | 横向残差 `mode=sim` 接 FSDStack | 仅 paired 帧生效；安全层可否决 |
| 2 | M4 DQN 观测对齐实车分布 | contract fail-closed |
| 3 | M3 BC 数据扩域（多图 shadow） | `validate_learned_path` |
| 4 | E2E 时序模型 | 任务接管率 |
| 5 | 远期：Decision Transformer / REM-SAC | 有稳定数据闭环再评估 |

铁律：学习路径一律过 `planning.validate_learned_path` + safety 仲裁。

---

## S5 — 工程债与可发布性（贯穿）

1. **P6 pin 权重**：`weights/pinned/MANIFEST.json`（hash + 场景 + git + 配方）  
2. **scripts 治理**：`_tmp_*`、重复 probe 归档到 `scripts/archive/` 或删除  
3. **一键报告**：一条命令出 paired / in-lane / 实车 A/B 三段结论  
4. **文档**：HANDOFF 每次实车轮必写；ROADMAP 阶段完成定义勾选  
5. **实车协议**：串行、短实验默认 120s、`python -u`、按 PID 停

---

## 3. 各阶段目标一览

| 阶段 | 一句话目标 | 主指标 | 典型产出 |
| --- | --- | --- | --- |
| **S0** | 实验可信 | 合法机位率、provenance 正确 | spawn 门、真 map 影子 |
| **S1** | italy 连续开 | paired>33%、in-lane≥86%、stall≤83 | 新 town pin、A/B scorecard |
| **S2** | FSD 默认 | 默认 driver=FSD 且指标不回退 | 开关 + 冒烟 |
| **S3** | 多图短途 | US 人筛机位 ≥3 臂过门 | 场景常数、US 数据 |
| **S4** | 学习增量 | 残差/决策在环且安全可否决 | sim RL、契约扩展 |
| **S5** | 可复现 | pin 可被他人加载复现 | MANIFEST、流水线 |

---

## 4. 推荐修改热点（按包）

| 包 | 优化点 |
| --- | --- |
| `lane/` | 配对间距/闪断；US 黄线几何；hold 仅实验 |
| `vision/` | 黄线/植被进伪标签管线；存在性+任务双指标 |
| `planning/` | 路口走廊、障碍高度滤、分场景裕量 |
| `fsd_drive` / `data_contract` | map provenance、出生点门 |
| `rl/` | sim 接入门禁、残差仅 paired 生效 |
| `scripts/` | photo tour 合法性、live_ab、任务选点一键化 |
| `labeling/` | score 策略主动学习、批次命名规范 |

---

## 5. 风险与“先别做”

| 风险 | 缓解 |
| --- | --- |
| 并行会话踩工作树 | 按模块提交；长实验前 HANDOFF 锁车 |
| 用错机位下结论 | S0 门 + 人眼抽检帧 |
| IoU/AUC 选模型 | 只认 paired+in-lane+实车安全 |
| 大 hold 掩盖真问题 | 默认 4，A/B 有完整收尾才改默认 |
| 学习栈越权横向 | 铁律 + validate + 仲裁 |

---

## 6. 建议 30 天节奏

| 周 | 焦点 |
| --- | --- |
| W1 | S0 全套 + 120s 有效判决；italy 开始新标注批 |
| W2 | S1 训练 1–2 轮 + 钉住集；配对几何小修 |
| W3 | S1 实车 A/B；若过门→S2 默认切换设计 |
| W4 | S2 落地或 S1 再一轮；S3 east_coast 人筛机位第二判决 |

---

## 7. 完成定义（项目级，沿用 ROADMAP）

1. 城镇/山地默认路径：安全项全 0 且 stall≤83（跨会话）  
2. 横向全来自感知  
3. pin 权重可复现  
4. pytest + offline_validate 全绿  

**扩展定义（可选，用户确认后）**：east_coast 人筛机位短途达标纳入“多图完成”。

---

*本文档为规划口径；阶段关闭以 ROADMAP/HANDOFF 实测数字为准，不在此虚构通过项。*
