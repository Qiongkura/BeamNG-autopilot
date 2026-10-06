# T16 论文图表集（2026-10-06）

**生成入口**：`scripts/m5_paper_figures.py`（只从**已记录的数据文件**取数；不手写、不估计）

```pwsh
.venv\Scripts\python.exe scripts\m5_paper_figures.py          # 全部 8 张
.venv\Scripts\python.exe scripts\m5_paper_figures.py --only fig5
```

输出：`logs/paper_figures/fig*.png`（300 dpi；`logs/` 不提交，图由脚本随时可重建）。
每张图的数字来源见下表；图注即论文可直接用的 caption。

| 图 | 文件 | 数据来源（逐文件） |
|---|---|---|
| 1 | `fig1_gate_matrix.png` | `logs/experiments/r2_verdict_v7_20261005.json`、`r2_verdict_v8_ADOPTED_20261005.json` |
| 2 | `fig2_boundary_map.png` | `r2_verdict_v8_paint_nodilate_mean_20261005.json`、`r2_verdict_v8_paint_dil1b_20261005.json`、`r2_verdict_v8_paint_dil2_20261005.json` |
| 3 | `fig3_dose_response.png` | `dev_full_arms_20261005.json`（base / base6x）、`dev_full_dose_arms_s4{3..7}.json`（neg4x / neg85x） |
| 4 | `fig4_final_confirm.png` | `logs/experiments/final_set_v5_20261005/seal/confirmation_base6x-seed42.json` |
| 5 | `fig5_deadlock_anatomy.png` | `logs/fsd_benchmark/town_*.json`（12 次运行逐帧遥测，512 个"规划车体越界"帧） |
| 6 | `fig6_factor_ladder.png` | `logs/fsd_benchmark/{scorecard,manifest}_*.json`（按 manifest 开关 + A/B 时间窗分臂） |
| 7 | `fig7_lane_gate_evidence.png` | `logs/fsd_benchmark/town_1791272{266,479}.json`（铺装门分级诊断轮） |
| 8 | `fig8_closed_loop_tradeoff.png` | `logs/fsd_benchmark/scorecard_1791{201426,204924,206113,206348,202995,205151,206744,206973}.json` |

---

## 图 1 — R2 验收：两套"线"的定义各差一门

> **Fig. 1.** Gate metrics (mean over 6 seeds) against the frozen thresholds (dashed)
> for the two protocol definitions on the dev pool (a) and the R3 limited-class pool (b).
> v7 (label scope) misses the identity gate in both pools and additionally precision and
> role on R3; the adopted v8 (paint scope) clears every gate except the **label-scope
> recall on dev** (hatched bar, 0.56 < 0.70) — the disagreement between the two
> definitions is one gate each, and it is a *definitional* difference (dark paint),
> not a model-quality difference.

**诚实性注记**：v8 的召回有**两个口径**（漆范围 = 采纳判定用；标签范围 = 与 v7 可比的口径），
两者都画出来了——只画漆范围会把"v8 在标签范围不过门"这件事藏起来。

## 图 2 — 边界图：膨胀买召回、付精度，R3 付约 3 倍

> **Fig. 2.** Effect of mask dilation (0/1/2 px) on recall (paint), precision and role
> agreement, on dev (a) and R3 (b). Dilation buys recall in both pools but costs
> precision, and R3 pays ~3× more (0.65 → 0.27 vs 0.89 → 0.63); at 1 px R3's precision
> and role fall below their gates while dev passes, at 0 px the reverse holds for one
> dev seed (s45 = 0.671). **No single dilation value clears both pools.**

**诚实性注记**：0 px 行的口径是 `mean`、1/2 px 行是 `strict`（记录如此）；本图画的是
**逐 seed 均值**，与口径无关，门限判定以记录文件里的 `verdict` 字段为准（见脚注文字）。

## 图 3 — 训练组成剂量-效应：身份率在 6× 达峰、8.5× 饱和转差

> **Fig. 3.** Candidate identity rate vs negative-example dose (×, relative to the base
> pool), per seed (dots) and mean (bar). Identity peaks at 6× (0.759) and falls back at
> 8.5× (0.700); the base and 4× arms sit at 0.706 / 0.712. The dashed line is the frozen
> 0.60 identity gate.

**诚实性注记**：**每档 seed 数不等**——base 1 个、4× 5 个、6× 1 个、8.5× 5 个
（6× 就是交付候选 `base6x` 的组成，只有 s42）；y 轴自 0.40 截断（图内已标注）。

## 图 4 — 一次性最终确认（已消费）：交付臂在道路级不相交的 200 帧上

> **Fig. 4.** (a) Per-group pixel metrics on the sealed, road-disjoint final set
> (label scope): three groups (a27/a1/a2) sit at recall ≈ 0.17–0.20 while precision
> stays 0.86–0.92; a4 carries no line ground truth ("n/a"). (b) Negative side over the
> same 200 frames: 22/105 eligible frames carry false line pixels, but they are 0.08% of
> eligible pixels and the largest connected component is 1164 px.
> Overall label-scope recall 0.4603 / precision 0.7629 — **below the R2/R3 pools'
> readings**, which is the headline result of the independent confirmation.

**诚实性注记**：身份率/左右角色/参考覆盖探针**未在该集上运行**（记录内明文 UNKNOWN），
且该集**已消费**不可再读；(b) 的三个量纲不同，柱高做了归一化、原始值标在柱上。

## 图 5 — 死锁解剖：停歪的车被拒绝每一条路径

> **Fig. 5.** Anatomy of the driving deadlock over 512 "planned vehicle body crosses lane
> boundary" frames (12 town runs): (a) the planned crossing sits at a median 2.50 m
> (509/512 within 2–4 m), just inside the 4 m hard-stop threshold; (b) 96% of those
> frames the ego is stationary, and in **all** of them the current body is inside the
> lane — a parked, off-centre car is refused every path and can never re-centre.

## 图 6 — 驾驶层单因子阶梯：都让车动了，但都没过门

> **Fig. 6.** Single-factor ladder (town, alternating A/B, 4 runs per arm unless noted):
> (a) median travelled distance per arm; (b) median stall fraction with the worst-case
> body-centre-cross frames per arm (control→factor). The lane-gate factor (F-E) roughly
> doubles the distance and the recentre factor (F-C) improves the worst case most, but
> **no arm clears `no_stall` (0 frames) and the hard gates stay 0/4 everywhere**.

**诚实性注记**：F-F 为**部分数据**（对照 3 次 / 因子 2 次，A/B 被中途暂停，未判定）；
F-C 的臂取自**修好之后**的那一轮 A/B（此前一轮代码静默失效，on 臂等于对照臂）。

## 图 7 — 车道接受证据：'铺装门'其实被 BEV 障碍层驱动

> **Fig. 7.** (a) Lane/path availability in the baseline configuration (2 diagnostic runs):
> of 415 settled frames, 116 (28%) have a perception lane, 79 (19%) a planner path, and
> only 28 (7%) a path whose deviation exceeds 0.30 m. (b) Why 202 lanes were revoked:
> the revoked frames' samples are obstacle cells (median 6, IQR 5–8) with off-mask cells
> ≈ 0 — i.e. the "on-pavement" gate is driven by the BEV head's obstacle predictions,
> not by "the centre is off the pavement".

## 图 8 — 闭环权衡：v8 把中心压线清零，代价是可用率

> **Fig. 8.** Closed-loop trade-off (town, 4 runs per arm): centre-crossing frames vs
> sensor lane-source rate. v8 sits at zero centre crossings in 4/4 runs while v7 shows
> 8/8/8/13; the sensor-source rate is lower for v8 on median (0.29 vs 0.78, dotted lines)
> but the ranges overlap — availability is given up, safety on this metric is gained.

---

## 论文写作时要一并写明的口径与限制

1. **判定单位**：R2/R3 是 **6-seed 臂**（均值判门 + 逐 seed 上报）；最终确认是**单个
   checkpoint**（按索引规则选定的 seed 42，不按分数挑）。
2. **最终集只消费一次**：图 4 的数据来自那一次消费，之后不可重读。
3. **规则会被动作量污染**：`body_cross_centre_frames` 是**帧计数**，而各臂行进量差可达
   2 倍（F-E：6.85 → 14.1 m）。图 6 因此把"最坏帧数"与"行进"并排给出；F-C/F-E 的
   否决就是按"两臂最大值直接比"的预注册规则作出的（后续已另行预注册按行进量归一的读法）。
4. **UNKNOWN 不释放门**：任何未测口径（身份率/角色/参考覆盖在最终集上）都不得写作"已确认"。
5. **原始产物**：`epoch_*.pt` 等逐 epoch 快照与两代前的运行目录已在 2026-10-06 清理中删除
   （审计清单 `logs/_cleanup_20261006.txt`）；所有**判定 JSON、scorecard/manifest、
   封存最终集与其 200 帧、交付候选 checkpoint、模型 pin** 均保留，图 1–8 可由此完整重建。
