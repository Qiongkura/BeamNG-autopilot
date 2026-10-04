# Order 5 受限场景 Tech R3 设计（T16 §10.3 裁决 B 指定的下一路径）

- 日期：2026-10-04
- 目的：在**无/少"像线的非漆结构"**的路线类上做端到端验收——身份率 0.60 门的
  结构性上限（§18.5）在那里不成立，因此这是"不改口径、不新增标注"前提下
  唯一能把身份率门验出来的路。
- 依据：`docs/T16_DECISION_RECORD_IDENTITY_SCOPE_B_20261004.md`（裁决 B）。

## 1. 场景判据（**预注册**，与模型无关）

**近场背景类结构密度**（只读人工标签；实现 `scripts/m5_scene_structure_density.py`，
纯函数 `frame_structure_density` + 单测）：

```
band(row) = [u_min - M, u_max + M]      # 该行线类(label==2)像素左右极值 ± M
M         = 0.10 × 帧宽
structure = band 内 label == 0 的像素    # 标签说"背景"：墙/护栏/路缘/植被/阴影
density   = Σ structure_px / Σ band_px  # 近场（rows ≥ 0.55H）逐帧池化
```

**阈值（预注册，先于任何筛选/打分）**：**density ≤ 0.08**。
依据（已实测，非事后选取）：受控生成路（无结构）0.045–0.047；开发集 14 个有线
场景 0.024–0.321，其中最低的两档是宽路包（`pkg_wide` 0.0246、`pkg_pkg_wide`
0.0236），其余城镇/路口/弯道场景 0.09–0.32。0.08 落在"宽路包"与"其余场景"
之间，且**远低于**受控路的值——即它挑的是"近场车道带里几乎没有背景物"的路线。

**为什么必须预注册**：绝不允许"看模型在哪些场景分高再选场景"（那是挑结果）。
判据只看标签与真值线位置，不看任何预测；阈值在本文件里先定死。

## 2. 场景池与筛选流程

1. **选路线（地图侧，模型无关）**：用铺装半宽与结构密度两条筛
   - 宽铺装优先：实机扫描（2026-09-30）481 条路里半宽 ≥4.0 m 有 43 条、
     ≥4.6 m 有 15 条；`pkg_wide` 的密度最低说明"宽"与"少结构"相关。
   - 采集池：在宽路锚点上采 N 条路线 × 每线若干站点（`m5_controlled_scenes.py`
     的站点规划只用于**锚点定位**；R3 采的是**真实地图行驶帧**，用采集器
     `collect` 子命令沿这些路线开）。
2. **筛选**：对池里每个场景跑 `m5_scene_structure_density.py`，保留
   `density ≤ 0.08` 的场景 → **R3 场景集**。
3. **预注册的兜底**：若合格场景 < 3 个（池太小/阈值太严），报告池的分布并
   **另行预注册**一个次级集合（池内最低四分位），明确标注为"次级口径"，
   不得与主集合混报。

## 3. 验收口径（冻结门，v7 主口径；双口径只报不判）

| 门 | 阈值 | 说明 |
|---|---|---|
| 覆盖率 | ≥ 0.80 | `candidate_reference_coverage`（v7） |
| **身份率** | **≥ 0.60** | `candidate_identity_rate`（**v7**，本 R3 要验的就是它） |
| 精度 | ≥ 0.40 | `line_precision` |
| 召回 | ≥ 0.70 | `line_recall` |
| 角色一致率 | ≥ 0.70 | `left_right_agreement` |

- 逐 seed 判定（seeds 42–47 起，不足则扩到 51）+ 配对/均值双报；
- `identity_surface_scope`（A′ 次报）与 `offroad_false_ratio` 一并报告，
  但**判定按 v7**（协议仍 `t14-protocol-v7`）；
- 缺测一律 UNKNOWN（不当通过）。

## 4. 采集/探针清单（可照抄执行）

```pwsh
# 1) 池采集（真实地图行驶；宽路锚点由宽度扫描给出）
.venv\Scripts\python.exe scripts\m5_seg_autoloop.py collect --run-id <r3_pool_YYYYMMDD> `
    --collect-map italy --collect-frames 40 --collect-step-m 3.0 `
    --collect-timeout-s 900 --collect-attach   # 采集后接审计/探针（同一入口）

# 2) 密度筛选（离线，只读 npz）
.venv\Scripts\python.exe scripts\m5_scene_structure_density.py `
    --dir logs\experiments\<r3_pool_YYYYMMDD> --out logs\experiments\r3_density.json

# 3) 验收测量（冻结模型 = 新基座 6× 臂的 checkpoint；v7 主口径 + 次报）
.venv\Scripts\python.exe scripts\m5_arm_gate_measure.py `
    --arm base6x-s42=logs\experiments\t16_negdose6x_20261001\round0\seed42\checkpoint_last.pt `
    --dev-runs <合格场景目录...> --out logs\experiments\r3_acceptance.json
```

## 5. 报告要求

- R3 结论必须写明：场景集（含 density 与筛选命令）、模型/协议版本、五门逐 seed
  与均值、以及与全量 dev 的对照（"结构性上限在受限场景上是否消失"）；
- 若身份率在合格场景集上 ≥0.60：把"0.60 门在受限场景类上成立、在全量 dev 上受
  标签集约束"作为结论记入方案（与决策记录 B 并列）；
- 若仍 <0.60：如实记录（说明结构性上限不是唯一原因），并把失败归因按 §16.1
  的层归因重跑一次。

## 6. 已知约束与风险

- 当前**开发集里只有 2 个场景**（`pkg_wide` 6 帧、`pkg_pkg_wide` 1 帧）满足
  `density ≤ 0.08` → R3 必须**新采**（宽路路线池），不能用现有 dev 直接当 R3；
- 采宽路路线时注意：宽 ≠ 无结构（停车场/广场也宽）——所以密度筛选不可省；
- 采集器在真实行驶中会同时采到"线状结构"帧，筛选后**不得**把被筛掉的帧
  回流训练（它们是评价帧）。

## 7. 先导测量（现有合格场景，7 帧；**样本不足，不作为验收结论**）

对开发集里唯一两个 `density ≤ 0.08` 的场景（`pkg_wide` 6 帧 + `pkg_pkg_wide`
1 帧）用新基座（6× 臂，seeds 42–44）跑了一次先导：

| seed | C | R | M | 覆盖 | 身份 | 角色 | 路外候选 |
|---|---|---|---|---|---|---|---|
| 42 | 24 | 14 | 6 | 0.5833 | 0.4286 | 0.6667 | 8 |
| 43 | 30 | 20 | 5 | 0.6667 | 0.25 | 0.8 | 10 |
| 44 | 30 | 20 | 7 | 0.6667 | 0.35 | 0.8571 | 9 |

**两条必须先记下的教训**（都进 R3 的执行纪律）：

1. **样本量不足**：R = 14–20 < 冻结口径的 `per_scene_min_candidates=30` → 这组
   数字**不可作为 R3 验收**（缺测/样本不足既不是通过也不是失败）。→ R3 必须
   新采，且采集池要保证**筛选后**每个场景 R ≥ 30（按经验每场景 ≥8 帧、
   合格率按 ~50% 估，池要 ≥ 3 倍于目标场景数）。
2. **覆盖率也可能不过**：先导的覆盖 0.58–0.67 < 0.80 门。宽路上线更远、
   annotation 覆盖更稀——所以 R3 的结论必须**五门一起报**，不能只看身份率；
   若覆盖率在新采池上系统性不达标，则 R3 的结论是"受限场景类在覆盖率门上也
   不达标"，这同样是有价值的结论（说明宽路段的标注覆盖是下一个瓶颈）。

数据：`logs/experiments/t16_r3_pilot_wide.json`；判据实现与单测见
`scripts/m5_scene_structure_density.py` 与 `tests/test_candidate_metrics.py`。
