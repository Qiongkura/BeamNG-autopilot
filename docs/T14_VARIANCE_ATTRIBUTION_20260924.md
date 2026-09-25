# 训练波动归因报告（下一阶段方案第 2 项）

日期：2026-09-24 · 范围：离线、只读已有 checkpoint（**不训练**任何模型）
工具：`scripts/m5_seg_variance_report.py`（纯函数 + 逐 checkpoint 评估）
数据：`logs/experiments/t14_3h_roads3f`、`logs/experiments/t14_3h_roads3d`
（各 2 臂 × 3 seed × 3 epoch = 36 个 checkpoint）；开发集 = `diverse_wide`(30) +
`diverse_plain`(25) 共 55 帧。

---

## 结论 1：波动不在 seed 之间，而在**同一个 seed 的不同 epoch 之间** → 推翻原假设

- **原假设**（我上一轮的说法）："seed 方差主导"——同配方 `road_iou` 跨 seed 0.43–0.69。
- **是否真正激活**：激活（36 个 checkpoint 逐个实测，含逐帧与逐路段）。
- **可复现输入**：`logs/experiments/t14_variance/variance_roads3f_3d.json`；
  命令
  ```pwsh
  .venv\Scripts\python.exe scripts\m5_seg_variance_report.py `
      --runs logs\experiments\t14_3h_roads3f logs\experiments\t14_3h_roads3d `
      --dev-runs logs\m5_seg\diverse_wide_20260924\front_main `
                 logs\m5_seg\diverse_plain_20260924\front_main `
      --out logs\experiments\t14_variance\variance_roads3f_3d.json
  ```
- **实测结果**（离差平方和口径，可相加＝总平方和）：

  | 来源 | 平方和 | 占比 |
  | --- | --- | --- |
  | 臂间（数据/因子本身） | 0.0024 | **0.2%** |
  | seed 间 | 0.0354 | **3.4%** |
  | **seed 内（不同 epoch）** | **1.0097** | **96.4%** |

  逐 checkpoint 典型值（平凡基线见结论 2）：`baseline/seed43`: ep0 **0.0** → ep1 0.4261
  → ep2 0.4241；`round0/seed43`: 0.0 → 0.4266 → **0.5396**；`baseline/seed42`:
  0.42389 → 0.42389 → **0.59314**（同一 seed 内极差 0.17）。
- **结论**：**推翻**"seed 方差主导"。正确表述：**同一次训练的不同 epoch 之间就在平凡解
  与"学到一点"之间摆动**，而评估固定取 `checkpoint_last.pt`，于是"取到哪一轮"被
  记成了"seed 差异"。
- **未知与下一步**：为什么会摆到 0.0（塌陷）与回摆？见结论 3；训练到平台期后是否
  仍摆动，需要按第 3 项的固定步数重跑一次。

## 结论 2：`road_iou` 存在**白送分数的平凡基线**（0.4025），此前所有读数需要重新解读

- **原假设**：`road_iou ≈ 0.42` 表示模型学到了一部分路面。
- **是否真正激活**：激活（把开发集**全部像素预测成路面**，直接算 IoU）。
- **可复现输入**：评估矩阵新字段 `road_iou_trivial_all_road` /
  `road_iou_trivial_all_background`（`scripts/m5_seg_eval_matrix.py`）。
- **实测结果**：开发集标签里路面占 **40.245%**；因此"全预测成路面"的 IoU = **0.40245**。
  而此前被当作"稳定基线"的 `0.4239` = 平凡基线 **+0.0215**；被当作"提升了"的
  `0.5396/0.593/0.694` = 平凡基线 **+0.137 / +0.191 / +0.291**；`0.0` = 比平凡基线
  **低 0.402**（模型塌成"不预测路面"）。
  实测那一个"塌陷"checkpoint 的输出：**95.1% 的像素被判为路面**（≈全路面）。
- **结论**：**支持**"指标被类别不平衡饱和"。**任何 `road_iou` 读数必须与平凡基线一起报**，
  否则 0.42 会被读成"学会了路面"。
- **未知与下一步**：line 通道有同样的风险（line 像素占比更极端）；线指标的平凡基线
  （全背景预测 → IoU 0，且 recall 无分母）已在 `totals_to_metrics` 里体现，
  但**可信标线真值到位前不退结论**（第 1 项）。

## 结论 2b（补跑）：把基线训到 **12 epoch**，它才真的学会，而且到平台期后测量稳定

- **原假设**：15–30 步的读数已经算"学到一些路面"。
- **是否激活**：激活。同配方、同数据（`diverse_town`，20 训练帧）、同 batch，
  只把 **epochs 3 → 12**（15 → 60 步），三个 seed 各跑一遍（共 3 次训练）。
- **可复现输入**：`logs/experiments/t14_baseline_plateau/`（含逐 epoch checkpoint）；
  分析 `logs/experiments/t14_variance/plateau_baseline.json`；命令：
  ```pwsh
  .venv\Scripts\python.exe scripts\m5_train_seg.py `
      --runs logs\m5_seg\diverse_town_20260924\front_main --split tail --val-frac 0.2 `
      --epochs 12 --batch 4 --lr 1e-3 --seed 42 --device cuda --save-every-epoch `
      --ignore-line-class --line-tversky-weight 0 --line-cldice-weight 0 `
      --out logs\experiments\t14_baseline_plateau\seed42
  ```
- **实测结果**（`road_iou`，平凡基线 0.40245）：

  | epoch | 3 | 4 | 6 | 8 | 10 | **12** |
  | --- | --- | --- | --- | --- | --- | --- |
  | seed44 | 0.424 | 0.577 | 0.667 | 0.746 | 0.849 | **0.876** |
  | seed43 | 0.430 | — | — | — | — | **0.9035** |
  | seed42 | 0.424 | — | — | — | — | **0.876** |

  三个 seed 在最后一轮收敛到 **0.876 / 0.9035 / 0.876（±0.03）**；
  分解仍是"seed 内 96%"，但那已经是**单调的学习进程**而不是噪声。
- **结论**：**支持**"训练太短"是主因。**3 epoch 的读数（0.42 ≈ 平凡基线 +0.02）
  说明模型当时什么都没学会**——此前所有两臂比较（+0.1476 / +0.0401 / +0.0389 /
  +0.0068）都发生在**瞬态区间**，作为"因子效果"的科学结论**作废**（机制结论仍有效）。
  平台期判据：最后 3 轮验证指标变化 ≤ 0.02。
- **未知与下一步**：12 epoch 是否是"够用"的步数，取决于数据量；换数据后要重测平台期。
  `rounds` 现在会把 `plateau_by_seed` / `all_at_plateau` 写进判定，未到平台期时打印
  警告并把该轮标记为**暂行判定**。

## 结论 3：`0.0` 不是"较差的模型"，而是**塌陷输出**；三次运行都没到平台期

- **原假设**：0.0 只出现在少数帧（少数场景主导）。
- **是否真正激活**：激活（逐帧列表：**55/55 帧全部 0.0，两条路都是 0.0**）。
- **可复现输入**：同上 JSON 的 `records[*].frames`（含最差帧与 `gt_px`）。
- **实测结果**：塌陷帧的 `gt_px ≈ 87k–91k`（并非空标签），`n_no_denominator = 0`;
  同一 seed 的下一轮又能回到 0.43–0.54。三次运行的优化步数只有 **15–30 步**
  （20–40 帧 × 3 epoch ÷ batch 4）。
- **结论**：**支持**"训练太短、仍在瞬态"。数据因子的两臂比较（`+0.039`/`+0.007` 等）
  是在**瞬态区间**里做的比较——机制有效（能跑、能判定、能重放），但**数字不作数**。
- **未知与下一步**（这也是第 3/4 项的前置条件）：
  1. 训练长度要**到平台期**（例如固定 ≥N 步并检查开发指标连续 k 轮不变），
     两臂同一步数；
  2. **评估 checkpoint 选取要冻结并写明**（当前固定 `checkpoint_last.pt`，已在
     判定文件里记录 `eval_checkpoint`/`epochs`/平凡基线），不允许"取最好的一轮"；
  3. 在这些条件满足前**不再增加 seed、不比较新因子**（与方案第 2 项一致）。

## 八项对应

- **① commit/config/run**：本报告不改训练；工具与评估字段见提交（`scripts` 模块）。
- **② 覆盖率与 UNKNOWN**：本报告 55 帧全部有路面真值（`n_no_denominator = 0`）；
  标线指标仍 **UNKNOWN**（没有可信漆线真值）。
- **③④⑤⑦**：未测（无新源、无几何/控制链、无路面门闭环、无驾驶）。
- **⑥ 性能**：本工具同时记录 p50/p95，但**不作为模型延迟**（并发负载会污染，
  前一轮已实测 158 ms vs 安静 16.7 ms；可疑时 `rounds` 记 None）。
- **⑧ 通过/失败/未测**：通过＝波动归因与平凡基线量化；失败＝原"seed 方差主导"结论被推翻；
  未测＝平台期训练后的稳定性、可信漆线指标。
