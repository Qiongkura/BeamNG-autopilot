# T16 Order 4：初始化对照（冻结 seeds 42/43/44）2026-09-27

方案 §3.3/§3.4 要求的"**先独立检验相同数据/步数下随机初始化与已有模型初始化**"，
用预先冻结的配对 seeds（§3.6）跑完。入口：`scripts/m5_init_ab.py`
（可复用，纯逻辑测试 `tests/test_init_ab.py` 3 例）。

## 1. 设置（两臂只差 `--init`）

| 项 | 值 |
|---|---|
| 数据 | 5 个已验证正例目录（`annotate_pkg_e2_it3` + `annotate_pkg_e1_jv` 四视角）：训练 60 / 验证 15 帧 |
| 预算 | `--total-steps 120`，batch 4、lr 1e-3、cuda；两臂逐 seed 都 `steps_done=120/120`、`stopped_by=step_budget` |
| 采样 | quota 全池轮换（两臂 `unique_available=60`、`unique_seen=60` 一致） |
| 差异 | A 臂（random）无 `--init`；B 臂（init）`--init t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt` |
| 评估 | 8 个 reviewed 开发目录，口径 `_pixel_eval` + `identity_metrics` |
| seeds | **42 / 43 / 44（预先冻结的配对 seeds）** |

## 2. 结果（3 seed 均值 + 判定器同一条成对比较）

| 指标 | random | init | Δ | 成对判定 |
|---|---|---|---|---|
| line IoU | 0.2379 | **0.4152** | +0.177 | candidate_better |
| line precision | 0.2684 | **0.5099** | +0.242 | candidate_better |
| line recall | 0.6776 | 0.6959 | +0.018 | inconclusive |
| road IoU | 0.7275 | 0.8253 | +0.098 | inconclusive |
| 路外假线像素 | 218,417 | **30,625** | **−86%** | candidate_better |
| 候选身份率 | 0.3245 | **0.3704** | +0.046 | candidate_better |
| 可测候选覆盖率 | 0.970 | 0.978 | +0.008 | candidate_better |
| 左右角色一致率 | 0.6804 | 0.7247 | +0.044 | inconclusive |

逐 seed（line IoU）：random 0.285 / 0.241 / 0.188 vs init **0.413 / 0.454 / 0.379**
——**3/3 seed 全部高于对方全部 seed**；precision 与路外假线像素同样 3/3 分离。

## 3. 结论与边界

1. **初始化在同等数据/预算下影响很大**（3 seed 一致）：精度 +0.24、line IoU
   +0.18、路外假线 −86%；召回几乎不动（+0.018，判定 inconclusive）。
   → 此前两轮可晋级实验（都是随机初始化）测的是一个**受限配置**；后续数据增量
   实验必须以 **init 基座**为对照（方案 §3.4 的顺序要求）。
2. **身份率仍不过门**：0.37 vs 0.60 门 → **R2 仍不成立**（方案 §1.3-3：
   身份率不是唯一阻塞，但也没被这一因子解决）。Order 5 的签收条件未达成，
   不启动最终确认。
3. **谱系如实**：`--init` 的父 checkpoint 是旧产物（`train_args` 无 init 字段）
   → `init_source="random"`、`provenance_complete=false`、`parent_history` 为 null
   ——按方案 §3.5，这只可用于**探索**；进入最终确认前需要完整谱系的新基座或
   证明未见过的场景。
4. **固定预算，不是收敛结论**：两臂都停在 120 步预算（`stopped_by=step_budget`），
   未判平台期；这里的比较是"同预算下初始化差异"，不是"收敛后的差异"（§3.6）。

## 4. 复现

```pwsh
.venv\Scripts\python.exe scripts\m5_init_ab.py `
  --runs logs\experiments\annotate_pkg_e2_it3_20260927\front_fisheye `
         logs\experiments\annotate_pkg_e1_jv_20260927\front_fisheye `
         logs\experiments\annotate_pkg_e1_jv_20260927\front_main `
         logs\experiments\annotate_pkg_e1_jv_20260927\pillar_left `
         logs\experiments\annotate_pkg_e1_jv_20260927\pillar_right `
  --eval-runs logs\experiments\review_pack_20260926\reviewed_full\pkg_*\front_main `
  --seeds 42 43 44 --total-steps 120 `
  --init logs\experiments\t14_e1_promotable_20260927\baseline\seed42\checkpoint_last.pt `
  --out logs\experiments\t16_init_ab_20260927
```

产物（logs 不提交）：`init_ab.json`（逐 seed 指标 + 成对比较）、两臂
`seed{42,43,44}/`（checkpoint、train_hist、metrics）。
