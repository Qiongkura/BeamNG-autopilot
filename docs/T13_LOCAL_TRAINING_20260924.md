# T13 本地自主训练交付报告（Tech 标线分割，2026-09-24）

按用户下达的《本地自主训练执行方案》逐条执行。**结论一句话**：本轮的单因子（**只增加有场景差异的 Tech 数据**）
在冻结测试集上**没有净收益**（臂内 seed 波动大于臂间差值），按方案 §4 的自动停止条件**停止该实验、保留日志、
不提接线**；同时训练/评估链上**修掉 5 个会静默污染证据的工具缺陷**，并把"新采集+冻结的最终测试集"建立起来。

---

## 0. 结论索引（每项按 原假设 → 是否激活 → 输入 → 实测 → 支持/推翻 → 未知与下一步）

### 结论 1：单因子"只增加场景差异数据"是否改善标线通道？→ **推翻（无净收益）**

- **原假设**：在固定配方下，向 173 帧基线**只增加**路口 + 弯道两组新采集（54 帧 → 227 帧），能改善冻结测试集上的标线像素/候选层指标。
- **是否激活**：是（臂 A = 173 帧，臂 B = +`t13_junction2/front_main`、`t13_corner/front_main`；其余配方逐字相同）。
- **输入**：`frozen_holdout_t13.json`（digest `5828001add8ae9a5`，50 帧，两条**从未参与任何选择**的路段）；引擎标注为参照；3 个 seed（42/43/44）。
- **实测**（冻结集，`best.pt`，全局累加）：

  | 臂 | line precision | line recall | line IoU | 路外假线像素 |
  | --- | --- | --- | --- | --- |
  | A(173) | 0.4888 ± 0.0689 | 0.8000 ± 0.0273 | **0.4362 ± 0.0620** | 22 324 ± 16 188 |
  | B(227) | 0.5245 ± 0.0176 | 0.8115 ± 0.0687 | **0.4666 ± 0.0233** | 18 390 ± 6 070 |

  逐 seed IoU：A = 0.4929 / 0.3700 / 0.4457，B = 0.4678 / 0.4428 / 0.4893。
  **差值 +0.030 远小于臂内 seed 标准差（A 0.062、B 0.023），两臂区间大幅重叠**（A 有全场最高 0.4929，B 有 0.4893）。
- **支持/推翻**：**推翻**。单因子在该测试集上**无可辨增益**；且 seed 波动（IoU 0.37→0.49）**大于**任何臂间差异——
  这也意味着过去单 seed 的实验根本无法分辨这类差值。
- **未知与下一步**：n=3、单一测试集（两条路、单次采集、晴天）；若要判定"场景多样数据无效"这一更宽的命题，
  需要更多 seed 或更多测试路段；**按方案 §4 自动停止，不进入弱线采样/损失权重/曲线头等后续轮次**。

### 结论 2：两个微调臂都远好于生产模型与 v13b（但**不是**本轮的新因子）

- **原假设**：数据层修复（T11 结论）在同源标注上应显著提高标线 precision/IoU。
- **输入/实测**（同一冻结 50 帧）：production `seg_model/best.pt`（sha256 `b16a734d…`）P **0.2577** / R 0.5856 / IoU **0.2180**；
  初值 `v13b/best.pt` P 0.1990 / R 0.9832 / IoU 0.1984；两个微调臂 IoU **0.4362 / 0.4666**、precision 0.49 / 0.52。
- **支持**：支持（IoU 2.0–2.1×，precision 1.9–2.0×），与 T11 的既有结论一致。
- **限制**：这正是 T11 已登记并**未接线**的配方（本轮的臂 A 就是它），**不构成新的接线理由**；
  且它在本轮的候选层仍留下 70–78% 的路外候选（见结论 4）。

### 结论 3：`--follow-road` 在路口节点不可用（实测，登记）

- `t13_junction` 首采只得到 **1 帧**（`follow-road stopped (no forward neighbour on the roadnet)`），
  改用普通 `--step-m 2.0` 步进后得到 29–30 帧/视角。→ 路口节点（拼接后度 ≥3）上"最佳前向邻居"判据会失败，
  采到 **1 帧的"集合"** 是真实风险（与 T10 的 `--min-frames` 守卫同类）；已登记，未改路网步进逻辑。

### 结论 4：候选层随微调改善身份一致率，但**路外候选比例没有改善**

- **输入**：冻结集两段路各 25 帧，逐候选引擎漆线确认 + **零假设对照**（引擎线横移 3.0 m）。
- **实测**：

  | 模型 | 路段 | 候选 | 匹配率 | 零假设 | ×机会 | 角色一致率 | 落漆线 | 仅路面 | 路外 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | production | testA2 | 107 | 0.1028 | 0.0841 | 1.2 | 0.545 | 1 | 5 | 101 |
  | production | testB | 129 | 0.2093 | 0.0078 | 26.8 | 0.593 | 6 | 27 | 96 |
  | v13b | testA2 | 135 | 0.2148 | 0.0889 | 2.4 | 0.448 | 1 | 40 | 94 |
  | v13b | testB | 192 | 0.1615 | 0.0781 | 2.1 | 0.548 | 4 | 100 | 88 |
  | **armA42** | testA2 | 132 | 0.2348 | 0.0455 | **5.2** | **0.936** | **28** | 10 | 94 |
  | **armA42** | testB | 120 | 0.1417 | 0.0250 | 5.7 | **0.824** | 7 | 21 | 92 |
  | armB44 | testA2 | 122 | 0.1967 | 0.0656 | 3.0 | 0.792 | 19 | 10 | 93 |
  | armB44 | testB | 118 | 0.1271 | 0.0169 | 7.5 | 0.733 | 5 | 19 | 94 |

- **支持/推翻**：**角色一致率**（0.54→0.82–0.94）与**落漆线候选数**（1→28）随微调明显改善；
  但**路外候选占比 70–80% 在所有模型上都没有改善**（production 74–94%）→ 方案 §4 的"候选指标改善但路外假接受增加也不晋级"
  没有触发（两者都不变），但**也没有可晋级的候选层净收益**。armA 的候选层优于 armB（落漆线 28 vs 19；角色一致 0.936 vs 0.792）。
- **未知**：候选层是 image-plane 与投影两种量（探针报 p50 per-frame precision 0.0075–0.11，而像素层全局 IoU 0.44–0.49），
  两者**定义不同**（探针用候选提取用的线掩码含 cv_white 融合；像素层用学习掩码），本报告不把它们混用；**待澄清**（未做）。

### 结论 5：阶段评估未出现 `production_mismatch`（本轮指标有效）

- `m5_seg_stage_eval.py` 三段（raw→morph→road 约束→shape 过滤）与 `full_predict` 完全一致（如 armA42 0.4804=0.4804），
  生产臂亦无 mismatch → 本轮冻结集指标**未被作废**。
- 顺带实测：**后处理对弱模型伤害更大**——production 从 raw 0.2638 掉到 final 0.1912（−28%），
  而 armA42 从 0.5056 到 0.4804（−5%），armB44 0.4808→0.4360（−9%）。

### 结论 6：性能层（离线）无恶化

- 冻结集逐帧推理（GPU，含后处理，`Segmenter.last_timing_ms["total"]`）：production p50 12.30 / p95 14.52 ms；
  armA p50 11.31–13.77 / p95 13.37–21.37；armB p50 11.13–11.96 / p95 13.47–15.54 ms。
- **未测**：完整 tick P95、新源频率、deadline 违反（需要实车闭环，见 §⑦）。

---

## 1. 执行前冻结基线（方案 §1）

| 项 | 值 |
| --- | --- |
| HEAD（冻结时刻） | `cae11db72c`（本轮结束另有 3 个提交：`99e260c` vision、`cb1a1bb` collector、训练入口 1 个） |
| 工作树 | 0 个已跟踪文件被修改；3 个**未跟踪**目录（`.git.bak-20260920/`、`.workbuddy-ai/`、`rescue-20260920/`）——**未清理、未覆盖** |
| 生产 checkpoint | `logs/m5_seg/seg_model/best.pt`，**sha256 `b16a734d67f3c0de…`**，mtime 2026-09-13T17:23:56，3 375 899 B |
| 微调初值 | `logs/m5_seg/seg_model_v13b/best.pt`（40 轮，val_mIoU 0.5599） |
| GPU | RTX 5070 12 227 MiB；冻结时 used 5 861 / free 6 083 MiB（**游戏在跑**）；训练前已关闭游戏（used 3 232 / free 8 712 MiB） |
| torch | 2.12.0.dev20260408+cu128，CUDA 12.8，capability (12, 0)，可用 |
| 脚本 sha256(前 8) | train `696f77b5`、eval `c5625fa7`、stage `3bf3aadb`、identity `b756ef6c`、freeze `de5936c8`、ring `9d1f7e29` |
| 数据清单 | `logs/m5_seg/seg_t13_data_20260924/00_baseline_freeze.json` + `01_data_inventory.json`（158 个含帧目录、12 874 帧） |
| 训练输出位置 | 全部在 `logs/m5_seg/seg_t13_data_20260924/` 下，**未写 `logs/m5_seg/seg_model/`** |

## 2. 数据准入（方案 §2）

- **训练集（臂 A，173 帧）**：文档化配方 `ROUND8_HANDOVER_20260924.md:20` 的 188 帧清单
  **减去** 3 个 `_ft_probe5_*` 目录的 15 帧——它们是**逐字节复制**：`_ft_probe5_junction/*≡ident_probe_20260923/front_main/0000{0..4}`、
  `_ft_probe5_straight/*≡ident_probe_straight…`、`_ft_probe5_plain/*≡ident_probe_plain…`（内容 digest 证据见 `03_frame_digests.json`）。
  其中 5 帧**与清单内帧重复**（同一张图被计两次权重）。→ 两臂**同样**去掉这 15 帧，单因子保持"只加数据"。
- **来源元数据**：173 帧中 **158 帧**带完整身份（`map_name`/`source_id`/`t_wall`/`exposure`/`view`），
  其余 15 帧是上述无 meta 的复制帧（已删）。**map_name 缺陷（已修）**：采集器把地图硬编码为 `"italy"`，
  导致 east_coast_usa / gridmap_v2 两组采集都声称 italy（坐标 (240,855)/(0,0) 与 italy 城镇区不符）。
- **划分**：`--split by-map-scene`（每组取时间尾部 20%）。入口打印：**6 训练组 / 6 验证组、帧重叠 0、共享组 6**
  → **是 temporal-tail 开发验证，不是组隔离测试**（方案要求的注记）。
- **泄漏审计五段**（`split_audit`，173 refs）：帧重叠 0、组重叠 0、时间邻近 0 对（采集步距 2.57 s > 0.5 s，成立）、
  **复制样本 0 组**（证据：path 173 / exposure 173 / wall_clock 173 全部可得）、跨视角**不可判定且已如实标注**（单视角清单，原因文案已修）。
- **数据门判定**：**通过**（来源明确、训练/测试零重叠、泄漏检查可执行）。
- **最终测试集**：新采 `t13_testA2`（198807 东段，paint 经**仅计 class-2** 验证 LINE=958）+ `t13_testB`（198807 东北走廊），
  冻结为 `logs/goal_20260921/frozen_holdout_t13.json`，**digest `5828001add8ae9a5`**，50 帧；
  与训练集/开发集的内容 digest **重叠 0**（`05_overlap_check.json`）。
  **诚实登记**：第一版测试路段 `t13_testA` 作废——它是我用**自己的 scout bug**（把 road|line 两类相加）选出来的，
  实际无漆线；该 11 帧仍留在盘上作证据。

## 3. 训练顺序（方案 §3）

- **冒烟（独立目录）**：`--epochs 1`，45 帧 → `device=cuda`、损失 2.2475 有限、`best.pt`+`checkpoint_last.pt` 落盘；
  `--resume` 从 epoch 1 续训（跳过 0）、权重/优化器恢复（源码：resume 在 init 之后覆盖），
  **未出现 NaN/OOM**（`--vram-frac 0.9`）。
- **固定配方**（两臂逐字相同）：`--split by-map-scene --val-frac 0.2 --epochs 3 --batch 4 --lr 3e-4 --init seg_model_v13b/best.pt`；
  **只变** seed 与数据清单。臂 A 173 帧（140 训练/33 验证）、臂 B 227 帧。
- **内部开发验证**（同一入口打印，仅作参考、集合与臂绑死）：A mIoU 0.7409/0.7297/0.7286、line IoU 0.329/0.286/0.285；
  B mIoU 0.7504/0.7420/0.7378、line IoU 0.350/0.323/0.308。
- **`best.pt` vs `checkpoint_last.pt` 明确区分**：两文件都在冻结集上评估（`06_eval_matrix.json`）。
  臂 A 三个 seed 两者**完全相同**（最优即最后 epoch）；臂 B seed44 两者不同（best P0.5402/R0.8386/IoU0.4893；
  last P0.5193/R0.8461/IoU0.4745）。→ **仅凭文件名不能认定驾驶任务最优**，本轮两列并列报告。

## 4. 自动评估与淘汰（方案 §4）

- **四层口径**：像素层（本报告结论 1/2，冻结集全局累加 + 路外假线 = 预测标线像素落在 `label==0`）、
  候选层（结论 4，含零假设）、几何层（**UNKNOWN**：无独立几何标签，不报米制误差）、性能层（结论 6）。
- **生产臂 `production_mismatch`**：无（结论 5）→ 指标有效。
- **自动停止判定**：主要指标无净收益 → **停止**，保留全部日志（`armA/B_seed*.log`、`*.json`、`curve.png`）。

## 5. 接入门槛（方案 §5）：**不提出接线**

未进入 `--seg-model` 影子/短程 Tech 对照：触发条件（离线胜出）未满足（结论 1 的单因子无净收益），
且结论 2 的"好于生产"并非本轮新因子。**未接线、未改默认模型**，符合"本方案不授权自动替换驾驶默认模型"。

## 6. 本轮修掉的工具/数据缺陷（每条都有反例）

| # | 缺陷 | 反例/量级 | 提交 |
| --- | --- | --- | --- |
| 1 | `duplicate_groups` 用 `path` 单键 → 不同采集的**同名相对路径**被判重复 | 6 采集清单产生 **39 个假重复组**，而全部帧唯一 | `99e260c` |
| 2 | 重复检查的 `checked` 含义 = "发现了东西" | 清洁清单报 `checked=False` + "no duplicate evidence"，而 173/173 ref 都有 path/exposure/时钟 | `99e260c` |
| 3 | 跨视角检查把"单视角不可发生"写成"缺曝光计数" | 173/173 ref 有曝光，仍报缺计数 | `99e260c` |
| 4 | 采集器**硬编码** `map_name="italy"` | east_coast_usa / gridmap_v2 两组采集身份错误 | `cb1a1bb` |
| 5 | 训练入口 `per_run` 用**目录 basename** 为键 | 6 个 `front_main` 塌成 1 组 → `--split by-map-scene` 下**只训练 25/173 帧却仍打印"共 173 帧"** | 训练入口提交 |
| 6 | 训练入口读不到 ring 采集的 meta（在采集根目录） | 6 个采集的身份全部回退为目录名 | 训练入口提交 |
| 7 | （我的）scout 把 `road|line` 相加当漆线 | 选出无漆线的测试路段（已作废 `t13_testA`） | —（脚本未入库） |
| 8 | 采集期偶发丢相机 | `t13_offpavement2` 的 front_main/front_narrow **0 帧**而其余视角 15–19 帧 | —（登记） |

## 7. 逐项交付物（方案"交付报告"要求的 ①–⑧）

| 项 | 状态 |
| --- | --- |
| ① commit/config/run 与控制权 | HEAD `cae11db7`→3 提交；两臂 6 个 run 目录 + 日志；**未改任何默认模型/开关**（控制权未变） |
| ② 覆盖、缺列、UNKNOWN | 冻结 50 帧（2 路段 × 25）；训练 173 帧覆盖 6 组场景；**几何层 UNKNOWN**（无独立标签）；候选层两段路各自报；跨视角检查对单视角清单标注"不可发生" |
| ③ 强制刷新与新源消费 | **未测**（本轮无实车闭环） |
| ④ 几何正反例及最终命令 | 未涉及（离线）；命令全部记录在 `armA/B_seed*.log` 与 §3 |
| ⑤ 路面门真阳性/漏检/误停 | **未测**（无实车闭环）；仅离线"路外假线像素"= 6252–38626（臂内波动大于臂间差） |
| ⑥ 性能与 deadline | 推理 p50 11.1–13.8 ms / p95 13.4–21.4 ms（离线）；**完整 tick P95 与 deadline 未测** |
| ⑦ 碰撞/压线/出铺装/停车时长 | **未测**（本轮**未跑 Tech 闭环**，按方案要求写"未测"，不写"安全通过"） |
| ⑧ 通过、失败、未测 | 通过：冒烟/数据门/阶段一致性/离线性能不恶化；失败：单因子无净收益（结论 1）；未测：闭环全部、几何层、tick P95、跨视角（单视角清单） |

## 8. 复现命令（关键路径）

```text
# 采集（会话地图身份现在取自 scenario.get_current().level）
.venv\Scripts\python.exe scripts\m5_collect_seg_ring.py --attach --runtime tech \
  --teleport 1274.3 648.5 0.0 --frames 25 --step-m 2.0 --follow-road \
  --out logs\m5_seg\t13_testA2

# 冻结最终测试集（50 帧，digest 5828001add8ae9a5）
.venv\Scripts\python.exe scripts\m5_freeze_holdout.py \
  --runs logs\m5_seg\t13_testA2\front_main logs\m5_seg\t13_testB\front_main \
  --holdout-all --min-frames 5 \
  --out logs\goal_20260921\frozen_holdout_t13.json

# 两臂 × 3 seed：配方逐字固定（§3），臂 A 用 173 帧、臂 B 再加 t13_junction2/t13_corner；
# 每臂三个 seed（42/43/44），逐条命令即 §3 的配方加 --seed 与 --out。
# 当时的批量驱动脚本是工作区临时件（未入库），配方与清单已完整写在 §2/§3，可据此重放。

# 评估（已入库的工具；同一批冻结输入、多 checkpoint 一行一个）
.venv\Scripts\python.exe scripts\m5_seg_eval_matrix.py ^
  --model production=logs\m5_seg\seg_model\best.pt ^
  --model armA42=logs\m5_seg\seg_t13_data_20260924\armA_seed42\best.pt ^
  --model armB44=logs\m5_seg\seg_t13_data_20260924\armB_seed44\best.pt ^
  --runs logs\m5_seg\t13_testA2\front_main logs\m5_seg\t13_testB\front_main ^
  --dev-runs logs\m5_seg\holdout_wide2_20260924\front_main ^
  --json logs\m5_seg\seg_t13_data_20260924\06_eval_matrix.json
.venv\Scripts\python.exe scripts\m5_seg_stage_eval.py --runs <frozen dirs> --model <ckpt>
.venv\Scripts\python.exe scripts\m5_marking_identity_probe.py --run <dir> --meta <meta> \
  --model <ckpt> --null-shift-m 3.0

# 数据集准入审计（T14 阶段 A 的入口，本轮当时是临时脚本）
.venv\Scripts\python.exe scripts\m5_seg_dataset_audit.py --runs <帧目录...> --digest-scan ^
  --out logs\experiments\dataset_audit.json
```

**产物**：`logs/m5_seg/seg_t13_data_20260924/`（00 冻结、01 清单、02 准入、03 digest、04 冻结报告、05 重叠检查、
06 评估矩阵、stage_*.json、ident_*.json、arm*_seed*.log、六个 run 目录）；冻结集 `logs/goal_20260921/frozen_holdout_t13.json`。
