# S7 前置：驾驶入口控制权审计 + R2 就绪表（2026-09-27）

方案 §S7 原文：「R2 通过后执行 Tech：先固定模型的影子记录，再在有导航目标的限定
场景低速闭环。**核对现有入口是否控车**，命令中记录实际 runtime/goal/sensor/strict/
模型/控制权。**禁止假定"shadow"名字就代表没有车辆控制**。」「R2 要逐项列出适用场景、
来源资格、硬门、独立性与未测项。」

本文就是对这两条的**执行证据**：入口控制权逐条核过（file:line），R2 就绪表逐项列出。
本轮**没有**启动任何驾驶；本文只做静态核对 + 引用既有实测。

## 1. "shadow" 不等于"不控车"——实测结论

| 入口 | 会不会动车 | 证据 | "shadow" 在它那里指什么 |
|---|---|---|---|
| `scripts/m5_shadow_drive.py` | **会**（`conn.control(...)` ×4：`:364/:419/:541/:582`） | 文件头 docstring："**Drives the car** with a simple rule autopilot (PurePursuit …) while the FSD-style stack runs in shadow" | FSD 栈处于影子**预测**（记录 executed control vs shadow trajectory），车由规则 autopilot 开 |
| `scripts/m5_fsd_drive.py` | **会**（走 `fsd_drive.py` 的控制链：`:1574` 及多处刹停分支） | docstring："instead of only recording shadow data, **it drives the car** with the layered planner's chosen trajectory" | **影子录像**：`--no-shadow` 关的是 `ShadowRecorder`（`fsd_drive.py:2322`），不是控制 |
| `scripts/m5_collect_seg.py` | **会**（由游戏 AI 开：`conn.vehicle.ai.set_mode("span")`，`:227`） | 文件头："游戏 AI 沿路行驶（span 模式），同时保存 RGB 帧与 3 类标签" | 不涉及 |
| `scripts/m5_fsd_replay.py` | 不会（离线重放 `.npz`，"No game needed"） | 文件头第 4 行 | 影子**episode 数据** |

另外：全仓库共 **33 个脚本**会发车辆控制（`grep -l "conn.control" scripts/*.py`），
包括 `m1_*`/`m3_*`/`m4_*`/`diag_*`/`m5_*_test.py` 等。**没有任何一个入口把 "shadow"
用作"不控车"的开关**；`beamng_autopilot/autopilot.py` 也没有 dry-run/不控车模式
（9 处 `conn.control(...)` 无条件走）。

**对 S7 的直接影响**：S7 的"影子记录"阶段必须显式指定**控制者**——要么接受
`m5_shadow_drive.py` 用规则 autopilot 开（那么它不是"纯观察"，碰撞/压线等驾驶
事件指标同样要测），要么让游戏 AI 开（`m5_collect_seg.py` 的 span 模式只采数据）。
命令里必须写清 runtime/goal/sensor/strict/模型/控制权（方案原文要求），不能靠名字。

## 2. R2 就绪表（逐项列出适用场景、来源资格、硬门、独立性、未测项）

| 项 | 现状 | 证据/来源 |
|---|---|---|
| **适用场景** | 有线场景 6 个（人工真值 136 帧，R=15…148）；确认真无线 2 个场景（58 帧）记 `not_applicable`；负例诊断按 T10 只算 verified | `T14_AUDITED_TRUTH_REMEASURE_20260926.md` |
| **来源资格** | 训练侧只有 agent/引擎弱标签（研究臂）；**评价集之外的 verified 线标签 = 0**（正例包已采回、待标注） | `T14_E1_READINESS_20260926.md` §5 |
| **硬门（总体）** | 覆盖 0.9632 ✓（门 0.80）、角色 0.7281 ✓（0.70）、**身份 0.3115 ✗（0.60）** | 同上（协议 v5 计数契约） |
| **硬门（逐场景/逐 seed）** | 逐场景 R≥30 下限在真实路径生效（4/6 场景样本不足）；逐 seed 硬门与分场景违反逐条落盘 | `test_scene_counts_wiring.py`、`decision_*.json` |
| **独立性** | 136 帧里 24 帧确定 in-sample（候选训练组重叠）；其余 112 帧无内容重叠，但生产模型 13 个训练目录**无地图身份** → 组键比对是键空间差异，**不能**读作"无泄漏" | `T14_D1_EVAL_CHAIN_REPORT_20260926.md` |
| **最终集** | **未封存、未消费**（R2 未开始） | `m5_final_set.py` 未运行 |
| **未测项** | 最终集独立确认、Tech 驾驶（碰撞/压线/倒车/出铺装/停车分类）、驾驶 tick deadline、受控计时复测 | 本轮无驾驶 |
| **任务范围（负责人决定 2026-09-27）** | **纯土山路（无车辙）不属于驾驶可用性目标**：真车也难判断，不要求"走得通"；在这些场景上**停车是可接受行为**（与 §2.3"边界不可信时停车"一致）。它们**只**作为线通道负例（E1）使用，**不得**用来声称或否决驾驶能力 |
| **结论** | **R2 未通过**：身份门 0.3115 < 0.60，且已实测"身份门上界 ≈0.55 < 0.60"（容差/过滤/形状都不足以过门） | `T14_CANDIDATE_FAILURE_ATTRIBUTION_20260926.md` §2/§5/§6/§7 |

## 3. S7 执行时**必须**记录的字段（照方案原文，先列清单，跑时照填）

命令与产物里必须能回答：**runtime**（tech/drive）、**goal**（`--goal X Y`，演示不得
无目标）、**sensor**（`--lane-mode sensor` 且 `--strict` 时地图线不得领先）、
**模型**（完整路径 + sha16）、**控制权**（谁在开：规则 autopilot / FSD 栈 / 游戏 AI；
本次运行是否 attach 到已有会话）、**速度上限**（限定场景低速）、以及**事件指标**
（碰撞、压线、倒车、出铺装、分类停车时长）与其**检测覆盖**（缺检测不填 0）。

## 4. 状态

* 本文只做**静态核对**与就绪表；**未启动驾驶、未消费最终集、未改控制链**。
* R2/R3 的前置未变：先标注正例包（评价集之外的 verified 线标签）→ 可晋级线通道
  训练 → 身份门有希望通过 → 才谈最终集确认与 Tech 驾驶。
* 本文的入口清单可直接用于 S7 的"控制权不唯一就停止"检查（方案 §8.2）：
  **任何一次驾驶前，先确认 33 个可能控车的入口里只有一个是本次运行的**。
