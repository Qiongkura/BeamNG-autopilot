# T15 分割损失方向实测报告（方案 v2 §S6/E2 第二条 + 验收 T15）

日期：2026-09-26 ｜ 基线 HEAD：`2ad703a`
实测代码：`beamng_autopilot/vision/seg_losses.py`（**只读，未改动**）
实测测试：`tests/test_seg_losses_direction.py`（15 个用例，CPU、无 GPU、无下载）
复现命令：`.venv/Scripts/python.exe -m pytest tests/test_seg_losses_direction.py -q`
（连同既有 `tests/test_seg_losses.py` 一起跑：25 passed）

**一句话结论**：本实现的 Tversky 分母是 `TP + alpha*FP + beta*FN`，方向只由 `beta/alpha`
比值决定（默认 0.7/0.3 = 2.333 → FN 更贵、偏召回），但完整训练损失的 FP/FN 方向被
median-frequency CE 类别权重主导（实测 |g_FN|/|g_FP| ≈ 68.7，CE 单独就有 66.2），
`w_tversky` 只加幅度且把它**继续推向召回**，因此"Tversky 权重 2 = 压 FP"是错的；
"`beta<1` 提升召回"也不成立（`beta<1` 不是方向条件，只有 `beta/alpha>1` 才是）。

---

## 1. 原假设（被检验的说法）

| 来源 | 说法 | 实测判定 |
|---|---|---|
| 方案 v2 §S6/E2（`docs/T14_DEVELOPMENT_EXECUTION_PLAN_V2_20260926.md:308`） | "现实现分母含 `alpha*FP + beta*FN`" | **成立**（§2、§3） |
| 同处禁止的口头判断 | "权重 2 就是压 FP" | **不成立**（§4.3：实测方向相反） |
| 同处禁止的口头判断 | "`beta<1` 就提升召回" | **不成立**（§4.2：只有 `beta/alpha` 定方向） |
| `seg_losses.py:9,63` docstring | "alpha<beta：FN 罚得比 FP 重，提细线召回"；"beta>alpha favours line recall" | **成立**（与实现一致） |
| `docs/W1_VERIFIED_TRUTH_136_20260926.md:64` | "Tversky 权重 2.0 是压 FP 的 …… 应测**反向**（beta<1 或提高 line 权重）" | **不成立**，见 §7 发现 1 |
| `seg_losses.py:99` docstring | 被屏蔽类"反向梯度恒为 0" | 对 CE 项成立；对**区域项**不成立（§7 发现 2，当前配方走不到该路径） |

## 2. 实际公式（逐行引用）

**区域项 Tversky**（`beamng_autopilot/vision/seg_losses.py:60-69`）：

```python
def tversky_line_loss(prob, target, valid, alpha: float = 0.3, beta: float = 0.7):
    """Tversky loss on the line channel; beta>alpha favours line recall."""
    p = prob * valid
    t = target * valid
    tp = (p * t).sum()
    fp = (p * (1.0 - t) * valid).sum()
    fn = ((1.0 - p) * t * valid).sum()
    return 1.0 - tp / (tp + alpha * fp + beta * fn + _EPS)
```

* `alpha` 乘 FP、`beta` 乘 FN（`_EPS = 1e-6`，`:29`）。
* 输入是 **softmax 概率**（`:193` `prob = F.softmax(logits.float(), dim=1)[:, LINE_CLASS]`），
  所以"FP/FN 数"是**软计数**（概率质量），不是硬像素计数；hard 0/1 概率时两者一致。
* `valid = (target != 255).float()`（`:195`）——ignore 像素被乘 0，不进 tp/fp/fn。
* 无 reduction：分子分母都是整帧求和，输出是**单个标量比值**（帧内归一化，与像素数无关）。

**完整损失**（`LineSegLoss`，`:128-203`）：

```
L = CE(weighted, mean over non-ignored pixels)          # :185-190
  + w_tversky * (1 - TP/(TP + alpha*FP + beta*FN + 1e-6))   # :197-199
  + w_cldice  * (1 - 2*tprec*trec/(tprec+trec+1e-6))         # :200-202（soft-clDice，空骨架返回 0）
```

* 默认构造：`w_tversky=1.0, w_cldice=1.0, tversky_alpha=0.3, tversky_beta=0.7,
  ignore_index=255`（`:135-140`）。
* CE 用 `F.cross_entropy(..., weight=weight, ignore_index=255)`，reduction 为默认
  **mean**：分母是**非 ignore 像素数**（`:189-190`）；`class_mask` 给定时走
  `masked_cross_entropy`（`:91-125`），被屏蔽类在 softmax 分母里置 `-inf`。
* 整帧/整批全 ignore 时返回 `logits.sum() * 0.0`（`:122-123`、`:165-166`）——
  零损失但保留计算图，避免 `F.cross_entropy` 的零有效像素 NaN。

## 3. 合成实测数字

固定 TP=20、256 像素的二值 prob/target，`alpha=0.3, beta=0.7`：

| 用例 | 解析 `1-20/(20+0.3·FP+0.7·FN)` | 实测 | 备注 |
|---|---|---|---|
| FP=10, FN=0 | 3/23 = 0.130435 | 0.130434871 | 10 个纯 FP |
| FP=0, FN=10 | 7/27 = 0.259259 | 0.259259284 | 10 个纯 FN |
| FP=5, FN=5 | 5/25 = 0.200000 | 0.200000048 | 对称 |
| **值之比 FN/FP** | β/α = 2.3333 | **1.987653** | 分母也含惩罚项，值之比 ≠ 权重比 |

边际敏感度（同一点 TP=20, FP=5, FN=5，D = 25）：

| 量 | 解析切线 | 有限差分（1 像素步长） |
|---|---|---|
| dL/dFP = α·TP/D² | 0.009600 | 0.009486 |
| dL/dFN = β·TP/D² | 0.022400 | 0.021790 |
| **比值** | **β/α = 2.3333** | 2.297（割线 < 切线，因损失对计数是凹的） |

* 凹性实测：FP 从 0→1 使损失 +0.010728，9→10 只有 +0.008642（饱和，边际非恒定）。
* 幅度 vs 方向实测：`(α,β)` 同乘系数**不是**不变式。状态 (TP=1,FP=0,FN=1)：
  (0.3,0.7)=0.411765、(0.6,1.4)=0.583333、(0.15,0.35)=0.259259 —— 幅度随系数走，比值不变。
* 闭式梯度校验（float64，区域项对 prob）：`-(t·D - TP·(t + α(1-t) - β·t))/D²`
  与 autograd 最大误差 **2.8e-17**；把 α/β 对调后的闭式与实测梯度差 **0.0686**
  → 实现确实是 α 乘 FP、β 乘 FN。

完整 `LineSegLoss` 在一个"FP 侧/FN 侧状态对称"的探针上（所有像素 p_line=0.5，
目标里 1 个真线像素被预测成背景；`ln2` 的 line logit）：

| 配置 | L | \|g_FN\|/\|g_FP\| |
|---|---|---|
| CE 单独，无类别权重（w_tv=0） | 1.381481 | **1.0000**（对称，控制组） |
| CE 单独，真实权重 [0.604, 1.0, 40.0] | 1.166892 | **66.23** |
| + Tversky(w_tv=1, 0.3/0.7) | 2.144471 | **68.72** |
| + Tversky(w_tv=2, 0.3/0.7)（历史 "Tversky 2.0" 因子） | 3.122049 | **71.06** |
| + Tversky(w_tv=1, 0.7/0.3)（α/β 对调） | 2.157031 | 67.32 |
| weight=None + Tversky(w_tv=1, 0.3/0.7) | 2.359059 | 4.12 |
| weight=None + Tversky(w_tv=1, 0.7/0.3) | 2.371619 | 2.39 |

（"真实权重"= `logs/_train_balanced.txt:8` 记录的 `[train] 类别权重: [0.6039878726005554, 1.0, 40.0]`；
其它 run 记录到的是 `[0.6049801707267761, 1.0, 40.0]`（`logs/_trainB.log:18`）与
`[0.6048, 1.0, 60.0]`（`logs/_train_v13_cldice.log:12`，那次 line_weight=3.0）——
**line 类权重在所有记录里都顶在 20×line_weight 的裁剪上限**。）

## 4. 方向结论

### 4.1 方向表（本实现）

| 操作 | 损失 | 说明 |
|---|---|---|
| FP↑（FN 固定） | 单调↑、凹（边际递减） | dL/dFP = α·TP/D² > 0 |
| FN↑（FP 固定） | 单调↑、凹 | dL/dFN = β·TP/D² > 0 |
| `beta/alpha` ↑ | 平衡点移向"少漏线"（偏召回） | 只有比值定方向；切线比恰为 β/α |
| `alpha`、`beta` 同乘 k | 值/梯度幅度变（k↑→损失↑，D 变大后饱和），**方向不变** | 非不变式，不能当"整体权重"用 |
| `w_tversky` ↑ | 只加**幅度**，方向仍是 β/α；在只召回不足的场景把总损失推向召回 | 与 α/β 比值不是同一个量 |
| CE 类别权重（`--line-weight`、median-freq） | **改变方向的主力** | 实测 CE 单独就 66.2×，区域项只在此之上加 ~2.5 |
| `w_cldice` | 治断线/连通性；本探针状态下其对 FP/FN 边际的贡献 < 1e-6（未单独量化，见 §8） | |

### 4.2 `beta < 1` 的真实含义（明确回答）

* **本实现里 β 是 FN 的权重**（`alpha * fp + beta * fn`，已由闭式梯度反证）。
* 因此"β<1"**什么方向都不代表**；有意义的只是 `β/α`：`β/α>1` 偏召回，`β/α<1` 偏精度。
* 实测反例：`(α=0.3, β=0.7)` 偏召回；`(α=0.7, β=0.3)` 里 β=0.3<1 **但方向相反**（偏精度，
  切线比 3/7=0.4286，同状态割线比实测 0.4353）。
* 默认配方 `β=0.7<1` 只是"β 比 α 大"的巧合写法；把 β 从 0.7 调到 0.5 仍是 β<1，方向却**更弱**
  （β/α 从 2.333 降到 1.667）。所以"推 β<1"作为"反向实验"是没有信息量的（默认早已 β<1）。
* 与"常见说法"的关系：在把 β 记在 FN 上的命名体系里，"β>α 偏召回"与本实现一致（不算反转）；
  但文献里 α/β 的命名**不统一**（有的论文把 α 记在 FN 上），任何"照论文调参"的结论都必须
  先核对该论文分母里谁乘 FP。本仓库的权威口径只有本实现。

### 4.3 "提高 Tversky 权重" vs "改 FP/FN 比值"

* `w_tversky` 1→2（历史因子 "Tversky 2.0"）：|g_FN|/|g_FP| 68.72→**71.06**，损失 2.144→3.122。
  即**更偏召回**，不是"压 FP"。把反向因子（β<α）与它混为一谈会推错实验方向。
* 想要"压 FP"只有两条路：`β/α < 1`（把 β 调到 α=0.3 以下），或降低 CE 的 line 类权重
  （`--line-weight`）。在当前配方里后者是量级更大的旋钮。

## 5. ignore 区验证

构造：第 0 行是真线；(4:6,4:6) 是 2×2 ignore 块（target=255），块内 line logit 取 ±6
（+6 = 若参与监督就是 4 个 FP；-6 = 中性内容）。

| 检查 | 实测 |
|---|---|
| 损失（块内 FP 内容 vs 中性） | **逐位相同**（`torch.equal` 为 True），`w_cldice`=0/1/3 都一样 |
| 梯度（同上两变体） | **逐位相同**；ignore 块内梯度元素**全 0** |
| 非空转反证：把同 4 像素从 255 改成 line 目标 | 损失变化 **0.059339**（若忽略无效则应为 0） |
| 整帧 255（默认 w_tversky=w_cldice=1） | 损失恰 **0.0**、梯度范数 **0.0**、非 NaN |
| 整帧 255 走 `masked_cross_entropy` | 损失恰 **0.0**、梯度范数 **0.0** |
| 区外监督仍在 | 非空转反证即证据（区外内容/标签变化会改损失）；探针用例里 g_FN、g_FP 均非 0 |

结论：ignore(255) 既不进 CE 的分子也不进其 mean 分母（分母是非 ignore 像素数），
区域项经 `valid` 屏蔽（`tp/fp/fn` 三处都乘了 valid 或已把 p、t 归零），梯度侧同样为 0。

## 6. 训练路径实际参数（当前配方）

| 项 | 值 | 位置 |
|---|---|---|
| 构造 | `LineSegLoss(weight=weights, ignore_index=255, w_tversky=args.line_tversky_weight, w_cldice=args.line_cldice_weight, tversky_alpha=..., tversky_beta=...)` | `scripts/m5_train_seg.py:907-912` |
| ignore_index | **255**（硬编码） | 同上 |
| `--line-tversky-weight` | 默认 **1.0** | `m5_train_seg.py:580` |
| `--line-cldice-weight` | 默认 **1.0** | `m5_train_seg.py:584` |
| `--line-tversky-alpha` | 默认 **0.3**（help 写 "Tversky FP weight"） | `m5_train_seg.py:587-588` |
| `--line-tversky-beta` | 默认 **0.7**（help 写 "Tversky FN weight (beta>alpha 提细线召回…)"） | `m5_train_seg.py:589-591` |
| CE 类别权重 | `median_freq_weights`：`clip(med/h, 0.1, 20)`，再 `w[2] *= line_weight`（`--line-weight` 默认 **2.0**） | `m5_train_seg.py:249-262`、`:560` |
| 记录到的真实权重 | `[0.6039878726005554, 1.0, 40.0]`（line 顶到 20×2 上限） | `logs/_train_balanced.txt:8` |
| 记录到的区域项开关 | `[train] line region loss: tversky=1.0 cldice=1.0` | `logs/_train_v13_cldice.log:13` |
| checkpoint 里的真实配方 | `line_weight=2.0, line_tversky_weight=1.0, line_cldice_weight=1.0, alpha=0.3, beta=0.7`（9 个 `logs/experiments/t14_*/seed*/best.pt` 的 `train_args` 实测：5 个 line 臂为 `(2.0, 1.0, 1.0, 0.3, 0.7, paint_source=agent_revision)`，4 个 road-only 臂为 `(2.0, 0.0, 0.0, 0.3, 0.7)`） | `ckpt["train_args"]`（`m5_train_seg.py:1262-1265`） |
| 历史实验因子 | 只用过 `--line-tversky-weight`：`2.0`（`logs/experiments/t14_auto_20260925_agentline/decision_line-tv2-r0.json`）与 `0.5`（`logs/experiments/t14_soak4h_i18/decision_soak4-ltw-0.5-r0.json`）；**没有任何 proposal/decision 用过 `line_tversky_beta`（α 更不在白名单里）** | 同左 |
| stdout 可观测性 | 只打印 `tversky=<w> cldice=<w>`，**不打印 α/β** | `m5_train_seg.py:913-915` |
| 落盘元数据 | `line_tversky_weight / line_cldice_weight / line_tversky_alpha / line_tversky_beta` 四项都写 | `m5_train_seg.py:1262-1265, 1404-1408` |
| 循环可提议的键 | `line_weight, line_tversky_weight, line_cldice_weight, line_tversky_beta`（**没有 alpha**） | `beamng_autopilot/experiments/proposer.py:31-34`；`scripts/m5_seg_autoloop.py:1312-1313` |
| 因子→旗标 | `--line-tversky-beta <v>`（`factor_to_flags`） | `m5_seg_autoloop.py:1330-1346` |
| road-only 配方 | `--ignore-line-class --line-tversky-weight 0 --line-cldice-weight 0`（两臂共享） | `m5_seg_autoloop.py:1416-1417` |

要点：当前配方是 `1.0·Tversky(α=0.3, β=0.7) + 1.0·clDice + 加权 CE（line 类权重 40）`。
实验循环里 **α 冻结在 0.3、只能动 β**；想真正翻转成"偏精度"必须 β<0.3，否则
`β/α` 始终 >1。另外：**已记录的所有损失实验都只动了整体权重这一个旋钮**
（`line-tv2` 用 `--line-tversky-weight 2.0`，`soak4-ltw-0.5` 用 `0.5`；
没有任何 proposal/decision 用过 `line_tversky_beta`，也没有 checkpoint 里出现过
非默认的 α/β），所以"FP/FN 相对方向"这一维在本仓库**从未被实验过**——
这正是 T15 要求先量方向、再设计 E2 实验的原因。

## 7. 与文档/计划声称的差异（发现，不在本任务修复范围）

1. **`docs/W1_VERIFIED_TRUTH_136_20260926.md:64` 方向判断错误**：
   "Tversky 权重 2.0 是压 FP 的 …… 应测反向（beta<1 或提高 line 权重）"。
   实测：(a) `w_tversky` 1→2 使 |g_FN|/|g_FP| 从 68.72 升到 71.06（更偏召回，不是压 FP）；
   (b) 默认 β=0.7 已经 <1，"推 β<1"不是反向；(c) 反向是 `β/α<1`（β<0.3）。
   建议后续在 W1 报告或后续计划里更正该句（本任务只记录，不改文档）。
2. **区域项与 masked-CE 的 softmax 口径不一致**：`seg_losses.py:193` 的区域项用
   `F.softmax(logits)`（**未加 class_mask**），而 CE（`:124`）用
   `masked_fill(blocked, -inf)` 后的 softmax。实测 `class_mask=[True,False,True]`
   （屏蔽 asphalt、允许 line）：把 asphalt logit 手动置 -inf 后损失从 2.258158 变到
   2.248027（差 0.010130），line logit 在 (0,0) 处仍拿到 0.011837 的梯度。
   这与 `:99` "被屏蔽的类……反向梯度恒为 0"的说法（若读成覆盖整个损失）不符。
   当前配方只屏蔽 line 类 → `region_w=0` 提前返回（`:191-192`），该路径**不可达**；
   风险留给"将来屏蔽非 line 类 + 开区域项"的组合。
3. **事件 `config_hash` 不含 α/β/clDice 权重**：`m5_train_seg.py:711`
   `config_hash=f"{lr}/{batch}/{line_tversky_weight}"` —— 只改 α/β 或 `w_cldice`
   的两个候选会写出**相同**的 config_hash（训练元数据 `:1262-1265` 仍记录了四项，
   所以数据没丢，但以 config_hash 做候选身份/去重时会看混）。未测试下游是否真的
   用它做身份判定。
4. 计划 §S6/E2 对分母的描述（`alpha*FP + beta*FN`）**正确**；`seg_losses.py:9,63`
   的两处 docstring 方向描述也正确。无需改损失实现。

## 8. 未知 / 未测

* **clDice 项的 FP/FN 方向未单独量化**：探针状态下 `w_cldice` 0↔1 的损失差 < 1e-6
  （单像素线目标下该项贡献极小），因此 §3 的比值数字实质是 CE+Tversky。clDice 对
  连通性的作用（既有 `tests/test_seg_losses.py` 已覆盖方向）与 FP/FN 的交互未测。
* **状态依赖性**：所有边际数字都在特定状态（TP=20 或 p_line=0.5 探针）测得；真实帧里
  line 占比 0.07–0.57%，`D` 与 FP 软质量完全不同，**不能**把 68.7 / 2.333 当作全数据集常量，
  只能当作"方向"证据。
* **未训练、未跑真实数据**：本任务不训练，因此"`w_tversky` 1→2 实测 +0.079 IoU"
  这类报告结论与本次方向分析是否自洽，未验证（方向分析只说它更偏召回）。
* **AMP/GPU 数值行为未测**（全 CPU float32/float64）；`_EPS=1e-6` 在 fp16 下的影响未测。
* **数据增强/采样对 ignore 的交互未测**（`_augment`、`balanced_indices` 等）。
* `class_mask=[B,C]` 逐样本混批路径（`:170-177` 的报错与拆批要求）本次只做了静态阅读，
  未新增用例（既有 `tests/test_seg_losses.py` 已覆盖）。
* 文献 α/β 命名冲突（§4.2 末）未做文献核对（只确认了本实现的口径）。

## 9. 验收 T15 映射

| T15 要求 | 对应实测用例 |
|---|---|
| 人为 FP/FN | `test_tversky_hard_count_values_and_value_ratio_is_not_beta_over_alpha`、`test_tversky_marginal_fp_fn_sensitivity_ratio_is_beta_over_alpha`、`test_tversky_saturates_so_the_next_fp_costs_less` |
| 损失权重方向符合实现 | `test_tversky_only_the_beta_over_alpha_ratio_sets_the_fp_fn_direction`、`test_scaling_alpha_and_beta_together_changes_magnitude_not_direction`、`test_region_gradient_matches_closed_form_and_rejects_swapped_convention`、`test_recipe_loss_direction_is_dominated_by_ce_class_weights`、`test_raising_the_tversky_weight_is_more_fn_averse_not_fp_averse` |
| ignore 不参与对应监督 | `test_ignore_region_content_changes_neither_loss_nor_gradient`（w_cldice 0/1/3）、`test_ignore_is_not_vacuous_same_pixels_as_line_target_change_the_loss`、`test_all_ignore_frame_gives_zero_loss_and_zero_grad`、`test_masked_non_line_class_leaves_the_region_term_on_the_raw_softmax` |
