# T16 受限场景 Tech R3 验收报告（第一轮，2026-10-04）

按 `docs/T16_ORDER5_R3_LIMITED_SCENE_DESIGN_20261004.md` 预注册的口径执行：
宽路池采集 → 密度筛选（≤0.08，只读标签）→ 严格认证导出 → 冻结五门（v7 主口径，
seeds 42–47）。**本报告不改判据、不放宽任何门**；结论按预注册纪律给（样本不足
既不算通过也不算失败）。

## 0. 结论摘要

1. **判定：本轮 R3 验收不成立（样本量纪律未满足）**。3 个认证场景中只有 1 个
   （`known_line_a4s0`，R = 50–53）达到冻结口径的 `per_scene_min_candidates=30`；
   另两个场景 R = 14–17（a0s2）与 22–30（a1s0）低于门槛 → 池化数只作**指示值**。
   唯一达标的 a4s0 上身份率 0.500–0.520 < 0.60 → **没有任何场景给出"0.60 在受限
   类上成立"的证据**。
2. **方向性结果（指示值，不判定）**：池化五门均值 = 覆盖 0.8601 ✓、身份 0.5640 ✗
   （门 0.60）、精度 0.5282 ✓、召回 0.7473 ✓、角色 0.7029（均值勉强过，但 3/6 seed
   单独不过）。身份率相对全量 dev 基座 0.4410 上移 **+0.12**，但仍差 0.036。
3. **归因（本轮最重要的产出）**：身份率缺口**不是近线错配**，而是**远场路外假候选**——
   未匹配候选 39 个里 36 个路外（`off_road_frac ≥ 0.5`）、38 个 |lat| ≥ 3 m
   （中位 7.5 m，n_px 中位 20）。预注册的密度判据**天然看不见**这些结构（见 §3.3），
   所以"选干净场景"这条路清得掉近中带、清不掉远场尾带。
4. 与决策记录 B 的关系：尾带与 B 记录里的"路外候选占分母"是同一现象
   （A′ 表面口径下身份率 **0.9385**、`matched_lost` 仅 0–1，即 v7 缺口几乎
   全部来自标签说在路外的候选）；R3 用受限场景并没有让它消失，宽路场景下它
   反而**更高**（dev 0.99 个/帧 → R3 1.50 个/帧）。

## 1. 场景集（采集 → 筛选 → 认证的漏斗）

| 阶段 | 口径 | 结果 |
|---|---|---|
| 池采集 | `collect --run-id t16_r3_pool_20261004`（真实地图行驶，宽路锚点） | 30 场景 × 8 帧 = 240 帧 |
| 密度筛选 | 近场背景类结构密度 ≤0.08（预注册，只读标签） | 14/30 场景合格 |
| 认证导出 | 四道门（逐实例 annotation 覆盖 ≥0.8 / 逐帧残差中位 ≤1.0px / \|lat\| ≤2.0 m / 漆面一致 ≥0.9） | **3 场景 / 24 帧** |

认证场景明细（`logs/experiments/t16_autotruth_r3_20261004/export_report.json`）：

| 场景 | 密度 | 认证角色（逐实例覆盖） | 残差中位 | 漆面一致 |
|---|---|---|---|---|
| `known_line_a0s2` | 0.0211 | `near_left` (1.00)；`far_left` 3.447 m 被横向门排除 | 0.577 px | 1.00 |
| `known_line_a1s0` | 0.0133 | `near_right` (0.86)、`straddled` (1.00) | 0.900 px | 1.00 |
| `known_line_a4s0` | 0.0415 | `near_left` (0.98)、`near_right` (0.98) | 0.903 px | 1.00 |

被隔离的 27 个场景里，主因是**逐实例 annotation 覆盖不足**（22 个：引擎标签在
这些路段不携带线类像素），另有 2 个横向档位全超 2.0 m、3 个投影残差中位
1.06–1.52 px。→ 认证漏斗总通过率 3/30（密度合格段内 3/14）。

**排除实例的参考可信性核查**（回应"远场实例是不是标注坏了"）：a0s2 被排除的
`far_left`（3.447 m）逐帧 annotation 覆盖实测 **1.00**（`logs/experiments/r3_instance_ann_cov.json`）
——标注在远场没有坏，导出的 label 里两条线都在。远场不匹配**不是参考缺失/偏移**
造成的。

## 2. 冻结五门（v7 主口径；seeds 42–47；24 帧）

| seed | C | R | M | 覆盖 ≥0.80 | 身份 ≥0.60 | 精度 ≥0.40 | 召回 ≥0.70 | 角色 ≥0.70 |
|---|---|---|---|---|---|---|---|---|
| 42 | 101 | 86 | 47 | 0.8515 ✓ | 0.5465 ✗ | 0.5443 ✓ | 0.7266 ✓ | 0.7021 ✓ |
| 43 | 110 | 96 | 54 | 0.8727 ✓ | 0.5625 ✗ | 0.5005 ✓ | 0.7837 ✓ | 0.7037 ✓ |
| 44 | 113 | 95 | 55 | 0.8407 ✓ | 0.5789 ✗ | 0.4906 ✓ | 0.7821 ✓ | 0.6909 ✗ |
| 45 | 119 | 99 | 58 | 0.8319 ✓ | 0.5859 ✗ | 0.4870 ✓ | 0.7690 ✓ | 0.6724 ✗ |
| 46 | 96 | 86 | 45 | 0.8958 ✓ | 0.5233 ✗ | 0.5702 ✓ | 0.6680 ✗ | 0.6889 ✗ |
| 47 | 106 | 92 | 54 | 0.8679 ✓ | 0.5870 ✗ | 0.5765 ✓ | 0.7546 ✓ | 0.7593 ✓ |
| **均值** | 107.5 | 92.3 | 52.2 | **0.8601 ✓** | **0.5640 ✗** | **0.5282 ✓** | **0.7473 ✓** | **0.7029 ✓\*** |

\* 角色门均值过线但 3/6 seed 单独不过（0.6724–0.7593），按逐 seed 判定记为**混合**。

来源：`logs/experiments/r3_acceptance_20261004.json`（候选口径）、
`logs/experiments/r3_acceptance_pixel_20261004.json`（像素口径）。
次报项（不判定）：`identity_surface_scope`（A′）0.9388（seed 42，含召回代价
`matched_lost=1`）、路外候选占比 0.46–0.51。

**逐场景**（冻结口径要求逐场景 R ≥ 30，且池化会掩盖单场景样本不足）：

| 场景 | R（6 seed） | 覆盖 | 身份 | 角色 | R≥30 |
|---|---|---|---|---|---|
| `known_line_a0s2` | 14–17 | 0.52–0.61 ✗ | 0.786–0.867 | 0.64–0.93 | ✗ |
| `known_line_a1s0` | 22–30 | 0.81–0.96 | 0.409–0.583 | 0.67–0.86 | ✗（仅 1 seed=30） |
| `known_line_a4s0` | 50–53 | 1.00 ✓ | 0.500–0.520 | 0.59–0.67 | ✓ |

→ 唯一满足样本门槛的 a4s0 上身份率约 0.51，**低于 0.60**。

## 3. 归因

### 3.1 未匹配候选的横向/路外分布（seed 42）

| | 0–1.2 m | 1.2–2.0 m | 2.0–3.0 m | 3.0–5.0 m | ≥5.0 m | 合计 |
|---|---|---|---|---|---|---|
| R3 未匹配 | 1 | 0 | 0 | 2 | **36** | 39 |
| dev 未匹配（136 帧，含重复目录） | 5 | 5 | 19 | 39 | **135** | 203 |

- R3 未匹配候选：36/39 路外（`off_road_frac ≥ 0.5`）、|lat| 中位 **7.49 m**、
  `n_px` 中位 20（p25 10 / p75 24）、类型 thin 29 / solid 7 / dashed 3 —— 远处
  细小亮条（墙/护栏/杆/植被边缘），不是线。
- 按帧归一：近中带（2–5 m）未匹配 dev **0.43/帧** → R3 **0.08/帧**（约 5 倍
  下降，这正是密度筛选的效力）；≥5 m 尾带 dev 0.99/帧 → R3 **1.50/帧**
  （宽路把更多路外内容带进画面）。

### 3.2 按角色分解（by_role，seed 42 / 47）

| 模型自报角色 | R（42） | 身份（42） | 身份（47） |
|---|---|---|---|
| `near_left` | 16 | **1.000** | **1.000** |
| `near_right` | 17 | 0.765 | 0.941 |
| `straddled` | 4 | 0.750 | 0.750 |
| `far_left` | 21 | 0.476 | 0.522 |
| `far_right` | 28 | **0.179** | 0.219 |

→ 近线身份接近满分，缺口全部集中在 `far_*`；`far_right` 最差。

角色混淆（matched 对上的 model_role × engine_role，seed 42）：
`far_left→near_left 8`、`far_left→far_left 2`；`near_right→near_right 10`、
`near_right→far_right 3`、`straddled→straddled 3`。→ 角色一致率不达标的另一半
来源是**同一条线被重复检出**：外侧那个副本被排成 `far_*`，却匹配到同一条
engine 近线（0.8 m 容差内），记成角色不一致。这与"远场尾带假候选"是两个不同
机制，分别对应角色门与身份门的缺口。

### 3.3 角色门的重复检出机制（seed 42，R3 认证集）

匹配上但角色不一致的候选（model_role ≠ engine_role）14 个：与**同侧另一个
候选**的 |Δlat| 中位 **0.16 m**（10/14 在 0.5 m 内）→ 同一条线被检出两次，
外侧副本被排成 `far_*` 却匹配到同一条 engine 近线，记成角色不一致。
角色一致的匹配候选 33 个：同侧最近邻 |Δlat| 中位 1.71 m（真·两条线）。
→ 角色一致率缺口的一半是**并行重复检出**，属提取侧可解（v7 的合并只处理共线
碎片，抓不住并行重复）；另一半是远场路外假候选（未匹配，见 §3.1）。

### 3.4 预注册判据的盲区（实测，不改判据）

密度判据的 band 是**逐行、锚定在"含线像素的行"上的线像素列 ±10% 帧宽**
（`scripts/m5_scene_structure_density.py`）。两条实测：

- 把行范围从 0.55H 放到全帧（1.0）**密度完全不变**（0.0133/0.0211/0.0415 三个
  场景逐字节相同）——因为线像素只出现在近场行，远场行没有线像素就被跳过；
- 因此**路外结构（|lat| ≥ 5 m 的那 36 个）在判据里不可见**：它们既不在线的
  列带里、也不在线出现的行里。

→ 结论：判据度量的是"线带内有没有非漆结构"，不是"画面里有没有像线的结构"。
R3 的设计前提（低密度 ⇒ 少线状非漆结构）在**近中带**成立、在**远场**不成立。
下一轮若继续这条路线，判据必须扩展为**全帧、非线锚定、模型无关**的度量
（例如标签背景类在整帧的细长亮条密度），且按纪律**先预注册再使用**。

## 4. 与全量 dev 的对照（R3 设计 §5 要求）

| 口径（base6x，seeds 42–47） | 全量 dev（18 目录） | R3 受限集（24 帧） |
|---|---|---|---|
| v7 身份率（均值） | 0.4283 | 0.5640（指示值） |
| A′ 表面口径身份率（均值） | 0.7717 | **0.9385** |
| A′ 召回代价 `matched_lost`（逐 seed） | 40–53 | **0–1** |
| 近中带（2–5 m）未匹配/帧 | 0.43 | 0.08 |
| ≥5 m 未匹配/帧 | 0.99 | 1.50 |

dev 口径来源 `logs/experiments/t16_dual_scope_10seed.json`（base6x-s42…s47）。
R3 上 A′ 口径几乎**零代价**（0–1 个匹配丢失）就把身份率抬到 0.94 —— 这是
"v7 缺口全部来自标签说在路外的候选"的直接证据；dev 上同样的口径要丢
40–53 个匹配（代价真实）。

→ "结构性上限在受限场景上是否消失"：**近中带的那部分消失了**（这是真实、
可复现的机制：0.4410 → 0.5640），**远场尾带那部分没有**，且在宽路上更重。
v7 身份率在 R3 上仍未达 0.60，与决策记录 B 的"标签集约束"框架一致——但
约束的具体内容从"近带结构"细化为"**远场路外假候选**"。

## 5. 下一步（按预注册纪律，先定义后使用）

1. **扩池重测（R3 的验收前置）**：目标每个认证场景 R ≥ 30。按本轮漏斗
   （密度合格 14/30、认证 3/14），需追加采集（如 `--anchor-offset 40 --anchors 12`）
   并优先在**已认证的三个锚点邻域**增加帧数（8 → ≥16 帧/场景），使 R 翻倍。
2. **判据扩展提案（新判据，需预注册）**：全帧"像线结构"密度（标签背景类 +
   RGB 亮细条，模型无关），在现池上先报分布再定阈值；用于挑选**远场也干净**的
   路段。
3. **或按决策 B 框架收口**：承认 v7 身份率在受限类上仍受路外候选约束（A′ 0.94），
   把 R3 作为"近中带机制已证实、远场尾带待解"的阶段性记录，R2 以该口径推进。
4. 角色门缺口单独排期：`far_*` 并行重复检出（§3.3：10/14 在 0.16 m 内）是
   提取侧可解问题（与 v7 共线合并同族），它同时压低角色一致率与身份率。
5. **下一单因子提案已写**：`docs/T16_NEXT_FACTOR_FAR_OFFROAD_PROPOSAL_20261004.md`
   （远场路外假候选：经典侧横向门 / 远场结构负例对照 / 并行重复合并 = v8 候选，
   含各自的风险与先离线扫描的纪律）。

## 6. 复现命令

```pwsh
# 密度筛选（预注册判据）
.venv\Scripts\python.exe scripts\m5_scene_structure_density.py `
    --dir logs\experiments\t16_r3_pool_20261004 --out logs\experiments\t16_r3_pool_density_scenes.json

# 认证导出
.venv\Scripts\python.exe scripts\m5_auto_truth_export.py `
    --batch logs\experiments\t16_r3_pool_20261004 `
    --out logs\experiments\t16_autotruth_r3_20261004 --map italy

# 五门（候选口径 + 像素口径）
.venv\Scripts\python.exe scripts\m5_arm_gate_measure.py `
    --arm base6x-s42=logs\experiments\t16_negdose6x_20261001\round0\seed42\checkpoint_last.pt `
    --arm base6x-s43=logs\experiments\t16_negdose6x_20261001\round0\seed43\checkpoint_last.pt `
    --arm base6x-s44=logs\experiments\t16_negdose6x_20261001\round0\seed44\checkpoint_last.pt `
    --arm base6x-s45=logs\experiments\t16_negdose6x_20261001\round0\seed45\checkpoint_last.pt `
    --arm base6x-s46=logs\experiments\t16_negdose6x_20261001\round0\seed46\checkpoint_last.pt `
    --arm base6x-s47=logs\experiments\t16_negdose6x_20261001\round0\seed47\checkpoint_last.pt `
    --dev-runs logs\experiments\t16_autotruth_r3_20261004\m5auto_a0_known_line_a0s2\front_main `
               logs\experiments\t16_autotruth_r3_20261004\m5auto_a1_known_line_a1s0\front_main `
               logs\experiments\t16_autotruth_r3_20261004\m5auto_a4_known_line_a4s0\front_main `
    --out logs\experiments\r3_acceptance_20261004.json

.venv\Scripts\python.exe scripts\m5_seg_eval_matrix.py `
    --model base6x-s42=logs\experiments\t16_negdose6x_20261001\round0\seed42\checkpoint_last.pt `
    --model base6x-s47=logs\experiments\t16_negdose6x_20261001\round0\seed47\checkpoint_last.pt `
    --runs logs\experiments\t16_autotruth_r3_20261004\m5auto_a0_known_line_a0s2\front_main `
           logs\experiments\t16_autotruth_r3_20261004\m5auto_a1_known_line_a1s0\front_main `
           logs\experiments\t16_autotruth_r3_20261004\m5auto_a4_known_line_a4s0\front_main `
    --device cuda --json logs\experiments\r3_acceptance_pixel_20261004.json
```

## 7. 本轮改动（代码，随报告一起提交）

- `scripts/m5_auto_truth_export.py`：导出包写入 `cameras` 块（探针建相机模型用；
  缺它整 run 判 UNKNOWN——R3 首轮验收 C=0/R=0 的根因）；合成/采集两种相机位姿
  形态都支持。
- `scripts/m5_arm_gate_measure.py`：逐场景明细（`per_scene`，支撑 per-scene
  R ≥ 30 判定）+ 身份率的角色分解（`by_role`）与角色混淆矩阵
  （`role_confusion`、`reference_instances_matched`）。
- 测试：`tests/test_auto_truth_export.py`（cameras 块与位姿换算）、
  `tests/test_candidate_metrics.py`（逐场景计数不变量、角色分解）。
