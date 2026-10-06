# T16 论文图表集（2026-10-06，34 张）

**生成入口（两个脚本，只读已记录的数据文件；不手写、不估计）**

```pwsh
.venv\Scripts\python.exe scripts\m5_paper_figures.py          # fig1–fig8
.venv\Scripts\python.exe scripts\m5_paper_figures_ext.py      # fig9–fig34
.venv\Scripts\python.exe scripts\m5_paper_figures_ext.py --only fig21
```

输出：`logs/paper_figures/fig*.png`（300 dpi；`logs/` 不提交，图可随时重建）。
`fig30–fig34` 是**示意图**（schematic），按代码/文档里的既有定义绘制，标题里已标注。

## 索引（图 → 数据来源）

| 图 | 内容 | 来源 |
|---|---|---|
| 1 | R2 门限矩阵（两套定义各差一门） | `r2_verdict_{v7,v8_ADOPTED}_20261005.json` |
| 2 | 边界图（膨胀的收益-代价） | `r2_verdict_v8_paint_{nodilate_mean,dil1b,dil2}_20261005.json` |
| 3 | 剂量-效应（身份率） | `dev_full_arms_20261005.json`、`dev_full_dose_arms_s4{3..7}.json` |
| 4 | 一次性最终确认 | `final_set_v5_20261005/seal/confirmation_base6x-seed42.json` |
| 5 | 死锁解剖 | `logs/fsd_benchmark/town_*.json`（12 次运行） |
| 6 | 单因子阶梯 | `logs/fsd_benchmark/{scorecard,manifest}_*.json` |
| 7 | 车道接受证据 | `logs/fsd_benchmark/town_1791272{266,479}.json` |
| 8 | 闭环权衡 | `scorecard_1791{201426,204924,206113,206348,202995,205151,206744,206973}.json` |
| 9 | 剂量 × 五指标 | 同图 3 |
| 10 | 身份率双口径（label vs surface） | `t16_dual_scope_10seed.json` |
| 11 | 召回双口径 + 非漆占比 | `r2_verdict_v8_ADOPTED_20261005.json` |
| 12 | 横向口径扫描 | `dev_scan_lateral_20261004.json`、`r3b_scan_lateral_20261004.json` |
| 13 | 并行合并扫描 | `dev_scan_parallel_20261004.json`、`r3b_scan_parallel_20261004.json` |
| 14 | 外观门消融（像素） | `dev_pixel_{base,appgate}_20261004.json`、`r3b_pixel_{lat3,appgate}_20261004.json` |
| 15 | 阈值敏感性（keep/elong） | `r3b_pixel_{keep0.6,0.7,0.8,elong0.4,0.6,0.8}_20261004.json` |
| 16 | SWA 单因子（否决） | `r2_verdict_swa_20261005.json` + 采纳判定 |
| 17 | β=0.8（Tversky）臂的负例代价 | `dev9_v8_dilate1_20261005.json`、`dev9_beta08_v8d1_20261005.json` |
| 18 | R3 池逐 seed 候选级门 | `r3_acceptance_20261004.json` |
| 19 | 实例级参考覆盖 vs 横向位置 | `r3_instance_ann_cov.json` |
| 20 | 负例诊断全家（各臂） | `logs/experiments/*_pixel_*.json` |
| 21 | 训练曲线 | `logs/experiments/**/train_hist.json` |
| 22 | 驾驶硬门清单热图 | `logs/fsd_benchmark/scorecard_*.json`（最近 32 次） |
| 23 | 仲裁原因直方图 | 12 次 town 遥测的 `reason` 字段 |
| 24 | 逐帧时间线（速度/车道偏差） | `town_1791221139.json`、`town_1791273507.json` |
| 25 | 各臂分布箱线图 | 各 A/B 的 scorecard + manifest |
| 26 | 车道偏差分布（闭环两臂） | 8 次闭环遥测 |
| 27 | 安全裕度分布 | 4 次验收遥测（`closest_obs_m`/`min_ttc`/`path_occ_frac`） |
| 28 | 帧计数 vs 按行进量归一 | 各 A/B 的 scorecard（bC 与 travelled） |
| 29 | 最终集组成 | `final_set_v5_20261005/seal/final_set_seal.json` |
| 30 | 计数契约 v5（**示意图**） | 定义取自 `AGENTS.md`/`candidate_metrics` |
| 31 | 候选口径流水线（**示意图**） | 代码 + 本会话实测（202/202 撤销由 obstacle 项触发） |
| 32 | 安全仲裁阶梯（**示意图**） | `safety_monitor.ARBITRATION_RULES` / `RULE_WORST_LEVEL` |
| 33 | 协议开关矩阵（**示意图**） | `experiments/protocol.py`（含实时哈希） |
| 34 | 一次性最终集流程（**示意图**） | `final_set.py` + v5 封存/消费账 |

---

## 图注（论文可直接用）

**Fig. 1.** Gate metrics vs frozen thresholds for the two protocol definitions, dev (a) and
R3 (b). v7 misses identity in both pools plus precision/role on R3; adopted v8 clears all
but **label-scope recall on dev** (hatched, 0.56 < 0.70).

**Fig. 2.** Mask dilation (0/1/2 px) buys recall and pays precision; R3 pays ~3× more
(0.65→0.27 vs 0.89→0.63). At 1 px R3's precision and role drop below their gates while dev
passes; at 0 px the reverse holds for one dev seed (s45 = 0.671). No single value clears both.

**Fig. 3.** Candidate identity rate vs negative-example dose: 0.706 (0×) → 0.712 (4×) →
**0.759 (6×)** → 0.700 (8.5×); dashed = 0.60 gate. Per-seed dots, mean bars.

**Fig. 4.** One-shot final confirmation on the sealed road-disjoint set: (a) per-group
label-scope recall/precision (three groups at 0.17–0.20 recall; a4 has no line truth);
(b) negative side (22/105 FP frames, 0.08% of pixels, max CC 1164 px). Overall recall
0.4603 / precision 0.7629.

**Fig. 5.** Deadlock anatomy over 512 "planned body crosses" frames: crossing at median
2.50 m (just inside the 4 m stop threshold); 96% stationary; current body inside in all.

**Fig. 6.** Driving single-factor ladder: travelled (a) and stall fraction (b) per arm with
worst-case body-centre-cross frames; every arm still fails `no_stall` (0 frames) and the
hard gates stay 0/4.

**Fig. 7.** (a) Baseline lane/path availability (415 → 116 → 79 frames); (b) the 202 revoked
lanes were revoked by the **obstacle** term (median 6 cells) with off-mask ≈ 0.

**Fig. 8.** Closed-loop trade-off: v8 sits at zero centre crossings in 4/4 runs (v7 8/8/8/13);
sensor-source rate lower on median (0.29 vs 0.78) with overlapping ranges.

**Fig. 9.** Dose response across five metrics (identity, role, coverage, off-road candidate
fraction, merge groups) — per seed and mean.

**Fig. 10.** Identity scope changes the reading, not the model: same checkpoints read in
label vs surface scope (base 0.44 → 0.71; base6x 0.44 → 0.73).

**Fig. 11.** Where the two recall scopes disagree (dev, 6 seeds): label vs paint recall with
the non-paint fraction of label line pixels overlaid.

**Fig. 12.** Lateral-scope scan: identity/role vs threshold on dev and R3 (grey = adopted 5.5 m).

**Fig. 13.** Parallel-merge scan: candidates merged away and kept, with the identity
consequence (twin axis).

**Fig. 14.** Appearance-gate ablation on the pixel metrics (line IoU / precision / recall),
dev and R3, seed 42.

**Fig. 15.** Threshold sensitivity of the mask post-processing (keep / elongation). The R3
pool carries no eligible negative frames, so the FP series is not measurable there.

**Fig. 16.** SWA single-factor result: no seed is rescued by last-5-epoch averaging; the arm
was rejected.

**Fig. 17.** Tversky β=0.8 arm vs base on the negative side (FP frames, FP pixels, max CC).

**Fig. 18.** R3 limited-class pool per seed: the three candidate-level gates with the gate
outcome marked (x = below gate).

**Fig. 19.** Instance-level reference coverage vs lateral offset across the R3 certified
scenes, coloured by role (near/far left, near right, straddled).

**Fig. 20.** Negative-side diagnostics across all recorded pixel arms (FP frame rate vs FP
pixel fraction; marker size = max connected component).

**Fig. 21.** Training curves from the recorded `train_hist.json` runs (val line IoU and
train loss per epoch).

**Fig. 22.** Driving hard-gate checklist over the last 32 town runs (green = pass, red = fail;
UNKNOWN counts as fail).

**Fig. 23.** Why the car does not move: safety-arbitration reasons over 2443 settled frames
(12 town runs).

**Fig. 24.** Per-run time series (speed top, path-to-lane deviation bottom) for a baseline
acceptance run and an F-E lane-gate run.

**Fig. 25.** Per-arm distributions (travelled, stall fraction) for the F-A/F-C/F-E A/Bs.

**Fig. 26.** Per-frame lane-placement deviation in the closed-loop round (v7 vs v8;
black bar = median).

**Fig. 27.** Safety-margin distributions (closest obstacle, min TTC, path occupancy) over
4 acceptance runs; axes clipped to the measurement range (see caption text).

**Fig. 28.** Why the safety rule needed re-specifying: raw body-centre-cross frame counts
scale with how far the car moved (log axis for the per-100 m normalisation).

**Fig. 29.** Sealed final-set composition (200 frames across 7 groups; digest and protocol
hash printed in the title).

**Fig. 30.** Counting contract v5 (schematic): every ratio has an explicit numerator and
denominator (coverage R/C, identity M/R, role A/L; precision/recall are pixel-level).

**Fig. 31.** Candidate-scope pipeline (schematic) with the measured revocation (34% of paired
frames; the obstacle term did all the revoking in 202/202 frames).

**Fig. 32.** Safety arbitration ladder (schematic, from `ARBITRATION_RULES`): soft rules
accumulate caps, `minimal_risk` rules stop the tick; colour = worst level.

**Fig. 33.** Protocol switch matrix (schematic) and the live hash: one constant switches all
three changes; the hash moves with the version so a seal cannot be mislabelled.

**Fig. 34.** One-shot final-confirmation flow (schematic) with the live seal (frames, digest,
protocol hash, confirmation count).

---

## 论文写作时要一并写明的口径与限制

1. **判定单位**：R2/R3 是 **6-seed 臂**（均值判门 + 逐 seed 上报）；最终确认是**单个
   checkpoint**（按索引规则选定的 seed 42，不按分数挑）。
2. **最终集只消费一次**：图 4/29/34 的数据来自那一次消费；身份率/左右角色/参考覆盖探针
   **未在该集上运行**（记录内明文 UNKNOWN）。
3. **两套"线"的定义**（图 1/2/11/12/14）：漆范围用于采纳判定，标签范围与 v7 可比；
   膨胀是两套集合的取舍（图 2 的脚注写明 0 px 行是 `mean`、1/2 px 行是 `strict`）。
4. **帧计数会被动作量污染**（图 6/28）：`body_cross_centre_frames` 随行进量缩放，
   F-C/F-E 的否决按预注册的"两臂最大值"规则作出，后续已另行预注册按行进量归一的读法。
5. **剂量各档 seed 数不等**（图 3/9）：base 1、4× 5、6× 1、8.5× 5（6× 即交付候选组成）。
6. **R3 池没有合格负例帧**（图 15）：该池上 FP 不可测，图中已标注，不当作 0。
7. **示意图**（图 30–34）按代码/文档既有定义绘制，不是测量结果。
8. **原始产物**：逐 epoch 快照与两代前的运行目录已在 2026-10-06 清理中删除（审计
   `logs/_cleanup_20261006.txt`）；判定 JSON、scorecard/manifest、封存最终集与 200 帧、
   交付候选、模型 pin 全部保留，**34 张图可由此完整重建**。
