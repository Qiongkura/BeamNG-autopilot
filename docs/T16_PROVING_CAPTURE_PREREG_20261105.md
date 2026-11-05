# 试车场低照度/雨天采集扩池实验（预注册，2026-11-05）

## 1. 问题与假设

12 格静止条件实验（`T16_PROVING_GROUND_CONDITIONS_RESULT_20261105.md`）
显示：**noon_clear / noon_fog PASS（12/12 true），其余 10 格 FAIL**
（低照度 9 格 `no_markings`，noon_rain `none`）。配对/宽度门均已修好，
剩余失败发生在候选层之前，判断为**输入域/光照覆盖不足**。

- H1：在低照度/雨天条件下采集 `engine_verified` 帧并混入训练池，能让
  10 个 FAIL 格中的多数转为 PASS；
- H2：该增益来自光照/天气覆盖，不是对评测场地的记忆——因此采集站点
  与评测站点**几何分离**，并新增一个未参与训练的 P 折验证泛化；
- H3（可证伪）：若混练后 FAIL 格无改善或原 4 折宏平均退步，则判负结果，
  不改门、不晋级。

## 2. 数据设计（冻结）

### 2.1 站点

评测站点固定为 `logs/m5_seg/gridmap_scene.json`（mid 500,500）。
采集站点**不使用该点**，三个采集站 + 一个 P 折留出站：

| 站点 | mid (x,y,z) | dir | edge_margin_m | 用途 |
| --- | --- | --- | --- | --- |
| S1 | (300, 700, 100) | (0,1,0) | 0.8 | 训练 |
| S2 | (700, 300, 100) | (1,0,0) | 1.0 | 训练 |
| S3 | (500, 900, 100) | (0,-1,0) | 1.2 | 训练 |
| P  | (600, 800, 100) | (0,1,0) | 0.6 | **留出（不训练）** |

所有站点 `scene_type=proving_lane`、`right_lane`、`half_width_m=3.5`、
`line_yellow_double`（白/黄同一材质，GridMap 实测唯一可写实线）、
无铺装（`m_road_asphalt_rough_01` 仅作为 CLI 材质覆盖占位，不建铺装）。

### 2.2 条件格

12 格与条件实验完全一致（4 时刻 × 3 天气）：

- 时刻：dawn 06:30 / noon 12:00 / dusk 18:30 / night 21:30；
- 天气：clear（默认）/ rain=`rainy` / fog=`foggy_morning`；
- 每格每站采集 **30 帧**（去重后；不足 20 帧的格记 INCOMPLETE，不补拍）。

总量目标：3 站 × 12 格 × 30 = **1080 帧**；P 折：4 格
（noon_clear、noon_rain、dusk_clear、night_clear）× 30 = 120 帧。

### 2.3 感知配置（冻结）

`BEAMNG_CV_WHITE=0 BEAMNG_LINE_MERGE=0 BEAMNG_PAIR_WIDTH_NEAR=1`，
模型 pin `logs/m5_seg/seg_model_v28_sslA/best.pt`（仅用于采集时的可用性
记录，不参与训练标签生成）。采集与评测使用同一套场景/材质。

## 3. 采集与真值（engine_verified）

### 3.1 采集

- 入口：`m5_pairing_probe.py --scene <site>.json --scene-type proving_lane
  --save-frames --tod <HH:MM> --weather-preset <name> --frames N`；
- 帧：probe 保存的 `frame/pos/heading/cam/t` npz（536×403）；
- 环境应用必须读回（`applied_tod` 与请求一致），否则该格记 UNKNOWN；
- 采集中途引擎/探针失败：保留失败日志，格记 INCOMPLETE；不静默补拍。

### 3.2 真值（来自生成侧几何，不新增人工标注）

- 线中心线：`right_lane_lats(3.5, edge_margin)` + `_line_nodes(span=34,
  step=1.0)`，与渲染使用同一组节点；
- 线宽：材质档案 `line_width`（GridMap=0.15m）；
- 标签合成：把每条线按宽度扩成世界系四边形条带，用帧内 `cam` 投影后
  栅格化为 `line`(2)；车道范围（两线外扩 1.0m）栅格化为 `road`(1)，
  其余 0；
- `truth_points`：节点链每 1.7m 取一点（与既有 `_line_truth` 一致），
  `class=2`，带 role。

### 3.3 导出冻结门（与既有认证池同口径，不因结果调整）

以下全部通过才入池（`verify_projection(line_evidence="appearance")` +
`line_evidence_coverage`）：

1. 覆盖率 `coverage ≥ 0.8`；
2. 投影残差中位 `≤ 1 px`、最大 `≤ 2 px`；
3. 漆色一致率 `≥ 0.9`（合成标签线上的像素必须是黄/橙漆色）；
4. `truth_points` 全部来自生成侧定义，来源标记 `engine_verified`；
5. 环境读回有效；
6. 帧内容 hash 去重（同站同格重复帧只留一帧）。

不满足的帧隔离到 `<cell>/quarantine/` 并保留原因，不进入训练池。

## 4. 训练协议（冻结）

- 基线：`seg_model_v28_sslA`（当前最高水位单模型）；
- 混练池：原 71 族 214 帧 + 新增 S1/S2/S3 的 1080 帧（同族不跨
  训练/评测：原 4 折的留出族保持不变，新帧全部属于**新的 proving 域**）；
- 采样：按域等量（原池:新池 = 1:1），epoch 数与学习率沿用
  `m5_train_seg.py` 既有 SSL 微调配方，不做新超参搜索；
- 种子：42/43 两次独立训练；只有两 seed 同向增益才采信；
- 不做：曝光/白平衡归一、TTA、集成加权——本实验只测**数据覆盖**。

## 5. 评测（冻结）

1. **12 格试车场条件矩阵**（评测站点，原封不动）：
   - 主判据：PASS 格数 ≥ **8/12**（基线 2/12）；
   - 其中低照度 9 格至少 **5 格 PASS**；
   - `noon_clear`/`noon_fog` 必须保持 PASS（12/12 true、cert=1.0）。
2. **原固定 4 折**（A/B/C/W）：宏平均 A/B/C ≥ **0.539246 − 0.005**，
   且任一折相对基线退步 ≤ **0.5pp**。
3. **P 折（未见站点）**：noon_clear + dusk_clear + night_clear 的
   line IoU 相对基线同折提升 ≥ **+0.05**（证明不是场地记忆）。
4. 部署门（仅当 1–3 全过才做）：D1 时延 ≤1.55×、D2 显存 ≤200MB、
   D3 fail-closed、D4 放置可用性不降。

## 6. 失败条件（预注册）

- 12 格 PASS < 8 或低照度 PASS < 5 → 判**负结果**，不降门、不重扫阈值；
- 原 4 折宏平均 < 0.534246 或单折退步 >0.5pp → 判**回归**，不晋级；
- P 折提升 < +0.05 → 判**未证明泛化**，即使 12 格变好也不晋级；
- 任一格环境读回失败 → 该格 UNKNOWN，不计入 PASS 数。

## 7. 复现与产物

```
# 1) 生成站点 JSON
python scripts/m5_proving_capture.py --build-sites

# 2) 采集（逐站逐格；环境读回缺失=UNKNOWN）
python scripts/m5_proving_capture.py --capture --site S1 --cell dawn_rain

# 3) 导出 + 门禁（步骤 2 的 pilot 先做 label 合成验证）
python scripts/m5_proving_export.py --cell <site>/<cell> --out <pool>

# 4) 训练 / 评测（沿用 m5_train_seg.py 与固定 4 折脚本）
```

产物：`logs/m5_seg/proving_sites/`（站点 JSON）、
`logs/m5_seg/proving_capture/<site>/<cell>/`（帧+日志）、
`logs/m5_seg/proving_pool/<site>/<cell>/`（入池 npz + manifest）、
`logs/m5_seg/proving_conditions/`（重跑后的 12 格表）。

## 8. 纪束

不新增人工标注；不消费最终集；不降门；失败/隔离帧全保留；
每步产物含 hash；模型未过 1–3 前不晋级、不改默认。

## 9. Pilot 标定（非约束，2026-11-05）

- 探针原来的保存帧只来自**放置阶段**（~15 帧）；已给探针加
  `--drive-seconds`（默认 1s，采集传 20s），并在 drive 循环按同一 npz
  格式保存 `frame_d*.npz`。S1/noon_rain 复测：**51 帧**（15 放置 +
  36 行驶）、`status=ok`、环境读回 12:00/rainy。
- 采集帧数按「放置+行驶」合计，导出时按里程/时间均匀抽 **30 帧/格**；
  不足 20 帧记 INCOMPLETE（本 pilot 已过线）。
- 非约束观察：同一 noon_rain 条件在 S1 给出 **16/16 true、cert=1.0**，
  而评测试车场 (500,500) 是 `none`（cands 空）。说明"雨天失败"可能
  混合了**站点几何/阴影**因素，不纯是天气——这正是本实验用 3 采集站 +
  P 留出站、并在评测试车场单独复测的原因；该观察不改变任何门。
