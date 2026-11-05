# T16 试车场静止条件实验（预注册，2026-11-05）

## 1. 问题

在**无地图标线污染**的受控场地上，感知链路（分割→候选→配对）在
TOD×天气变化下还能不能稳定给出双侧配对？这是条件矩阵的感知前置问题；
驾驶格暂时无法成立（试车场无可行驶参考，见
`T16_PROVING_GROUND_NEAR_WIDTH_RESULT_20261105.md` §判定）。

## 2. 场地与配置（冻结）

- 地图/场地：`gridmap_v2`，`logs/m5_seg/gridmap_scene.json`，
  `scene_type=proving_lane`（无铺装，decal 双线，right_lane，余量 0.6m）。
- 模型 pin：`seg_model_v28_sslA`（与条件矩阵 A3 同 pin）。
- 试车场诊断配置（非生产默认，逐格记录）：
  `BEAMNG_CV_WHITE=0 BEAMNG_LINE_MERGE=0 BEAMNG_PAIR_WIDTH_NEAR=1`。
  依据：浅色地面会被经典白漆臂并碎（T16_PROVING_GROUND_GROUNDTRUTH_RESULT），
  地面接缝会被 1.0m 合并半径并入左线，远场回投会把宽度带偏
  （T16_PROVING_GROUND_NEAR_WIDTH_RESULT）。三者均为研究开关，生产默认不变。
- 采样：每格 12 个 tick（静止、无 goal），`--save-frames` 保留帧证据。

## 3. 条件格

4 时刻 × 3 天气 = 12 格；每格 1 轮（首轮不做重复；方差在结果里单列）：

| 维度 | 取值 |
| --- | --- |
| 时刻 | dawn 06:30 / noon 12:00 / dusk 18:30 / night 21:30 |
| 天气 | clear（不调预设）/ rain=`rainy` / fog=`foggy_morning` |

格名 = `gridmap_<tod>_<weather>`，例如 `gridmap_night_rain`。

环境应用必须读回：引擎 `get_tod()` 的 `tod_after` 与请求一致才有效；
缺失/空 → UNKNOWN（不静默当晴天）。

## 4. 指标（冻结口径）

- 主指标：`true_pairs / ticks`（探针按 GT 中心/宽度判 true）；
- 副指标：`cert_line_recall.both_hit_rate`（GT 左右线都命中漆色）、
  `phantom_pairs`、`ambiguous_pairs`、`modes`；
- 聚合：逐格报告，不跨格平均（条件不可互换）。

## 5. 判定门（预注册，不因结果调整）

- **PASS**：`ticks ≥ 8` 且环境读回有效 且 `true_rate ≥ 0.80`
  且 `cert_both ≥ 0.80`；
- **FAIL**：环境有效但任一指标低于门；
- **UNKNOWN**：`ticks < 8`、环境读回缺失、报告缺失/损坏。
- 任一格 UNKNOWN 不得在汇总里当作 PASS；夜间/雨雾格失败不得降门重跑
  （只允许同配置重复以量化方差）。

## 6. 运行

```
.venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --plan
.venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --run
.venv\\Scripts\\python.exe scripts/m5_proving_conditions.py --score
```

产物：`logs/m5_seg/proving_conditions/<cell>.{json,jsonl,run.log,frames/}`
与 `conditions_table.json`。

## 7. 纪束

诊断/探针输出不进训练池；不新增人工标注；不改生产默认；不晋级模型；
失败与 UNKNOWN 证据全部保留。
