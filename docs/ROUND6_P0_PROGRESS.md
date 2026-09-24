# Round6 阶段 0 进展（回应 `HANDOFF_20260921_round5_REVIEW.md`）

日期：2026-09-21 ｜ 分支 `fix/round3-hardening-20260921`（**未提交**）

本文件只报**已完成并验证**的部分，并明确列出**未完成**的部分。所有数字都绑定
命令和产物，符合交接单第 5 节"统一记录要求"与第 10 节"完成定义"的前两条。

---

## 1. 已完成：P0-1 统一停车口径

- **代码**：`beamng_autopilot/eval.py::stop_digest()`（新增），阈值与规则以常量
  `STOP_SPEED_MPS=0.3`、`CREEP_SPEED_MPS=1.0`、`STOP_INTERVAL_RULE`、
  `STOP_HARD_RULE` 固化，并由返回值的 `spec` 字段自描述。
- **一条命令**：
  ```pwsh
  .venv\Scripts\python.exe scripts\m5_run_metrics.py --hist <run.json> [...] --table
  ```
  输出（节选，`--settle-s` 默认 8.0＝benchmark 口径）：

  | run | settled | stop_f | stop_s | longest_s | hard_f | 主因 | 横向覆盖 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `linesfix1` | 63 | 54 | 31.11 | 12.26 | 48 | `no_drivable_path` 25.9 s | `line_lat=75`、`lat_left=UNKNOWN` |
  | `models2_nodqn`（撞车） | 60 | 56 | 34.09 | **25.92** | 52 | `no_drivable_path` 27.8 s | `line_lat=69`、`lat_left=UNKNOWN` |
  | `divider3` | 89 | 89 | **52.14** | **52.14** | 89 | `no_drivable_path` 52.1 s | `line_lat=101` |
  | `models1` | 59 | 51 | 31.47 | 16.19 | 39 | `no_drivable_path` 13.0 s | `lat_left=2` 帧 |
  | `divider1` | 41 | 38 | 20.26 | 15.77 | 26 | `obstacle_very_close` 9.9 s | `lat_left=UNKNOWN` |

  **结论：8 个跑次里 6 个的停车主因是 `no_drivable_path`**（近场/远场可行驶证据缺失），
  不是障碍规则——这直接回答了交接单 §3 P1-6 的方向问题。
- **缺测规则**：列为空 → `None` + 计入 `unknown[]`，绝不写 0。实现中修掉了我自己
  第一版的同类错误：布尔列（`final_stop`/`emergency`）与文本列（`reason`）被当数值读时
  会静默变成"缺失→0"，现在按列类型分别解析（`_col(kind="flag"/"str"/"num")`）。
- **测试**：`tests/test_stop_digest.py`（11 例：阈值、蠕行分离、settle 规则、
  时间戳规则、硬停 vs 速度、终点区拆分、按原因秒数、四类缺列 → UNKNOWN）。

## 2. 已完成：P0-2 横向字段契约

- **文档**：`docs/LATERAL_FIELD_CONTRACT.md`（字段 = reference + frame + sign + unit +
  coverage，含 `0` 与 `None` 的区别，以及一帧实例读法）。
- **机器可读**：`beamng_autopilot/telemetry_contract.py::LATERAL_FIELD_SPECS` +
  `lateral_digest()`（逐字段覆盖率/缺列/中位/极值，缺列 → `status="UNKNOWN"`）。
- **修掉一处同类陷阱**：`safety_monitor._lane_deviation` 在没有参考/没有路径/无可投影样本时
  原本返回 `0.0`（＝"完全对齐"），现在返回 `None`；`SafetyVerdict.lane_dev_m` 默认值也改为
  `None`，消费端（DQN 观测、遥测、日志）全部做了 None 防护。**行为不变**（无参考的帧
  本来就已 fail-closed），1973 项测试全绿。
- **测试**：`tests/test_lateral_contract.py`（10 例，含"文档必须与代码逐字段一致"、
  "文档必须解释 `+0.91` 与 `0.368` 那对看似矛盾的数字"）。

## 3. 已完成：P0-3 分割评估分阶段 + P1-4 四臂离线对照

- **实现**：把后处理抽成与生产线**同一批函数**（`Segmenter._morph_close_line`、
  `_temporal_line_gate`、`constrain_line_to_road`、`filter_line_shape`），
  新脚本 `scripts/m5_seg_stage_eval.py` 调用同一批函数并**交叉校验**：
  stage5（完整 `Segmenter.predict`）必须等于 stage4，不等就打印告警。
- **产物**：`logs/goal_20260921/seg_stages_hand.json`（含命令、数据集、类别映射、
  ignore=255、逐阶段 IoU/像素/连通域/**路面外线像素**）。
- 数据：40 帧人工留出集（`manual_mountain_labeled` + `manual_review_batch_labeled`），
  checkpoint = `seg_model_hand/best_task.pt`，**平均帧 line IoU**：

| 阶段 | 无约束 | **像素式（旧线上）** | 连通域 0.5 | **两级（现产线）** |
| --- | --- | --- | --- | --- |
| 网络原始输出 | 0.3353 | 0.3353 | 0.3353 | 0.3353 |
| 形态学闭运算后 | 0.3303 | 0.3303 | 0.3303 | 0.3303 |
| 路面约束后 | 0.3303 | **0.0011** | 0.3283 | 0.3283 |
| 形状过滤后 | 0.3603 | **0.0000** | 0.3582 | **0.3582** |
| 完整 `predict()` | 0.3582 | 0.3582 | 0.3582 | 0.3582 |
| 路面外线像素（形状过滤后） | 34216 | 0 | 32385 | **32334** |

**三条结论（与 Round5 报告口径不同，此处为准）**：

1. **旧的像素式约束是灾难性的**：40 帧上把线几乎删光（约束后 0.0011、形状过滤后 0，
   每帧平均剩 1 个像素），这解释了"识别像狗屎"的直接观感；Round5 里"0.218→0.358"
   的数字正是这一段的对照。
2. **连通域/两级与"不加约束"在 IoU 上基本持平**（0.3582 vs 0.3603），
   它只删掉 5.5% 的路面外线像素（34216 → 32334）。也就是说：**这条约束的既定目标
   （删假线）几乎没达成，而它的风险（删真线）是真实的**。
3. **真正在清理假线的是形状过滤**：像素 6334 → 5269、连通域 2813 → 287，IoU 反而
   从 0.3303 升到 0.3603。→ P1-4 的建议应改为"保留连通域版本（不破坏输出），
   但**不要**指望它解决假线；假线需要形状/几何判别或更好的标签"。

## 4. 已完成：P1-5 撤回车道的停车代价

- **实现**：`scripts/m5_run_metrics.py::stop_by_lane_drivable()`，按"停车帧上记录的
  `lane_src_sel` + `lane_drivable.reason`"分类累计秒数。
- 实测（**关联统计，不等于因果**）：

| 跑次 | 分类 | 停车帧 | 秒 |
| --- | --- | --- | --- |
| `linesfix1` | `sensor \| centre off observed pavement` | 35 | **19.47** |
| | `perception-unavailable \| no_record` | 7 | 4.08 |
| | `sensor \| not enough observed samples` | 4 | 2.63 |
| `away1` | `sensor \| no_record` | 26 | 15.55 |
| | `perception-unavailable \| centre off observed pavement` | 24 | 13.29 |
| | `sensor \| not enough observed samples` | 10 | 5.35 |

→ 我的路面门槛在 `linesfix1` 上覆盖了 63% 的停车秒数。**门槛不能简单调低**
（交接单明确禁止），但没有前鱼眼提供的"真非铺装 vs 没看到"的区分，它就只能这样统计。

## 4b. 已完成：P1-2 车道参考跨帧稳定门

- **实现**：新模块 `beamng_autopilot/lane/stability.py`
  （`ReferenceStabilityTracker`，纯状态 + 纯函数）：
  - 侧别与**近场中心**（车体坐标系，`near_reference_lat`）连续一致 `need_ticks=2`
    才给 `full` 权限；中心带 `0.5 m`、侧别死区 `0.3 m`；
  - **未配对**（单侧镜像/分隔线回退/保持）**永远只给 `limited`**；
  - **不新鲜**的参考（hold/陈旧）重置连击——旧参考不会因为"活得久"变成合法参考；
  - 逐步累计 **侧别翻转**次数与**中心跳变**；
  - 全程不读导航线/地图（有测试断言源码里没有 `nav_route/route_ref/map_lane` 等）。
- **接线**：`fsd_stack.tick` 在 slew 限速之后运行稳定器，逐帧发布
  `ref_authority / ref_stability_reason / ref_side / ref_lat_m /
  ref_stable_ticks / ref_side_flips / ref_flip`。
- **行为开关**（交接单 arm C/D）：`BEAMNG_REF_STABILITY=1` 时，
  `limited` 权限把方向盘限制在 `REF_STABILITY_LIMITED_STEER=0.15`（原本可达 0.55），
  并记录 `steer_authority_clamped`。**默认关**，已登记进运行清单。
- **踩到并修掉一个真 bug**：第一版在 `fresh=` 里用了**尚未赋值**的局部
  `lane_src_sel`，被 `try/except` 吞掉 → 四个实车跑次的稳定器遥测**全部缺失**。
  已改为 `lane_ref_out.src`，并补**栈级**守卫测试
  （`tests/test_fsd_stack.py::test_the_tick_publishes_the_reference_stability_verdict`）
  —— 纯模块单测覆盖不到接线，这条测试专门守它。
- **测试**：`tests/test_reference_stability.py`（12 例：死区、前进不变性、
  paired+2 tick 才满权、未配对永不満权、中心跳变重置并计翻转、带内漂移保持稳定、
  跨中线计翻转、陈旧不得升级、无参考重置、源码不含地图量）+
  `tests/test_fsd_drive_pipeline.py` 3 例（limited 夹紧、full 不夹紧且需求确实超限、
  默认关只记录不夹紧）+ `tests/test_stop_digest.py` 2 例（稳定性指标聚合与缺列）。

### 首次 A/C 固定条件对照（n=2/臂，ABBA 交错，同起点/同 checkpoint/同开关）

`logs/goal_20260921/_ab/ab_p12.py`（arm A = 门关；arm C = 门开）：

| 指标 | arm A（关） | arm C（开） |
| --- | --- | --- |
| 停车秒数 | 37.04 / 32.22（均值 34.6） | 34.83 / 32.77（均值 33.8） |
| 里程 | 2.7 / 6.6 m | 3.1 / 5.5 m |
| 侧别翻转帧 | 7 / 5 | 4 / 6 |
| 中心跳变 p95 | 0.662 / 0.813 m | **0.487 / 0.208 m** |
| 中心跳变 max | 0.784 / 1.062 m | **0.733 / 0.736 m** |
| limited/full 帧 | 32/43、64/20 | 39/26、53/30 |
| 压线 C / 车身越界 C / 出铺装 / 碰撞 / damage | 0/0/0/0/0 | 0/0/0/0/0 |
| 停车主因 | body_crosses_boundary / no_drivable_path | no_drivable_path / no_drivable_path |

**读法（诚实）**：门**确实压住了参考中心跳变的尾部**（p95 与 max 都下降），
而 full 权限仍在 26–30 帧被授予（不是永久限权）；里程与停车秒数在噪声内
（n=2，按仓库自己的纪律**不足以判定差异**）。要满足交接单阶段 1，需要
**每臂 ≥5 次交错**，且 B/D 臂还依赖 P1-1 的鱼眼输入。

## 4c. 已完成：P1-1 前鱼眼接入 + 0–4 m 覆盖实验

- **输入接线**（只改相机输入，不动 planner/安全阈值/模型/DQN）：
  - `occupancy.nearfield_coverage()`：逐帧报告 0–4 m 带内的 `band_cells /
    observed_cells / observed_frac / drivable_cells / drivable_frac`
    （**`observed_frac` 才是头条**；`drivable_frac` 的分母可能只有 10 格，
    单独引用会把"看到 10 格"读成"全覆盖"）。
  - `occupancy.project_road_mask_to_grid(..., ground_z=)`：新增可选地面高度，
    默认行为不变；鱼眼用**路面**平面（`pos.z − EGO_ORIGIN_GROUND_GAP_M`），
    因为 0.17 m 的高度偏置在 2 m 处就是几十厘米。
  - `fsd_stack.NEARFIELD_CAM`（`BEAMNG_NEARFIELD_CAM`，默认 `off`）：
    `off` / `fisheye`（同一 Segmenter 跑鱼眼帧→投影）/ `fuse`（前视 ∪ 鱼眼）/
    **`band`（仅上界实验：注入固定宽度可行驶条，非感知、不得作为方案）**；
    `NEARFIELD_EVERY_N` 控节流。`fsd_drive` 仅在开关打开时才多请求一路相机。
  - 逐帧遥测：`nearfield_mode/ms/road_px/skipped/error/injected_band_m/cov`。
- **又踩到两个真 bug（都由新测试当场抓住）**：
  1. 近场带的**栅格行方向搞反**（行号越小才是车前方），覆盖指标与注入带都在量
     "车后方"；
  2. 分割模型挂在 **head 实例**（`stack.hydra._heads["semantic"]`）上，而
     `out.head_outputs["semantic"]` 是 **TaskOutput**；第一版据此取 segmenter，
     每帧 `nearfield_error`，整条鱼眼链路静默失效（实测 83/83 帧报错）。
- **测试**：`tests/test_nearfield_input.py`（8 例：UNKNOWN 语义、只数车前带、
  注入带必须自报注入、缺鱼眼降级为可记录的 skip、鱼眼用**自己的**相机模型与地面高度、
  默认只轮询一路相机、投影函数保留 `ground_z` 参数）。

### 四模式交错对照（每模式 n=2，ABFU / U F B A 交错；同地图/车/起点/goal/checkpoint/开关）

| 模式 | 0–4 m 观测格（/540） | `observed_frac` 中位 | `no_drivable_path` 秒 | 里程 | 额外耗时 p50 | 压线/出铺装/碰撞/damage |
| --- | --- | --- | --- | --- | --- | --- |
| `off` | **10** | 0.019 | 20.88 / 22.61 | 6.9 / 7.3 m | — | 0/0/0/0 |
| `fisheye` | **70** | 0.130 | 8.84 / 35.83 | 8.9 / 4.4 m | **16.9 / 15.6 ms** | 0/0/0/0 |
| `fuse` | 70 | 0.130 | 34.64 / 36.75 | 3.5 / 6.5 m | 15.9 / 18.0 ms | 0/0/0/0 |
| `band`（**注入上界**） | 74 | 0.137 | **2.75 / 4.93** | 3.8 / **14.9 m** | — | 0/0/0/0 |

**读法（诚实）**：

1. **假设成立**：0–4 m 带今天几乎是盲的（540 格里 10 格），接鱼眼后 7 倍到 70 格，
   代价 ~16 ms/tick；这与几何预测（前视最近可见地面 3.55 m、鱼眼 2.22 m）一致。
2. **"有近场证据"本身不解锁行驶**：fisheye/fuse 的 `no_drivable_path` 没有下降
   （fuse 两轮 ≈35 s，与 off 持平）。这正是交接单第 2 步要回答的问题，答案是
   **仍然停车**。
3. **卡点是形状不是数量**：注入带 74 格 ≈ 鱼眼 70 格，却把 `no_drivable_path`
   压到 2.8/4.9 s、一轮跑到 14.9 m。注入的是干净连通走廊，鱼眼给的是外域模型在
   120° 视场上的稀疏斑块。→ 下一步必须二选一或并行：**(a)** 鱼眼视角域内微调
   （交接单第 4 步），**(b)** 查规划器证据门为何在稀疏 70 格下仍判无路（P1-3 范围）。
4. `band` 是**上界实验**：它是注入的直条，不是感知；不得据此宣称"接鱼眼就好了"。

## 4d. 已完成：P1-3 证据门归因 + 车身 drivable 覆盖保护

两件事分开做：**(a)** 先回答"70 格稀疏可行驶证据下为何仍判 `no_drivable_path`"，
**(b)** 再补"车身穿在铺装上"的独立测量与保护（默认关闭，只做测量）。

### (a) 归因：不是可行驶证据空，是**车道门先开火**

- `planning/constraints.py`：新增 `reject_counts` 与 `_reject(reason)`，把原先
  "静默丢弃候选"的 10 个出口各自打上标签（`no_path_geometry` /
  `strict_no_perception_lane` / `strict_empty_drivable_layer` /
  `no_forward_progress` / `ref_start_yaw` / `lane_cross` / `body_cross` /
  `off_drivable_fraction` / `off_drivable_near` / `blind_path`）；
  `selector` 每 tick 重置并随 `meta["rejects"]` 上抛，`fsd_stack` 落到
  `out.meta["plan_rejects"]`，`fsd_drive` 写入 hist。
- 实测（`logs/goal_20260921/p13_fisheye.json`，84 帧，鱼眼近场开启）：
  `plan_blocked={'no_perception_lane': 72}`、`plan_rejects={'strict_no_drivable_evidence': 936}`
  ——**当时那版的标签打错了行**：当帧覆盖是 `observed_cells 70 / drivable_cells 42`
  （不是"没有可行驶证据"），真正开火的是**严格车道门**（`lane_src` 不是 sensor 时
  候选先被否，可行驶证据根本没被咨询）。标签已改正并加了
  `tests/test_selector_safety.py::TestRejectionAttribution`（4 例）钉住。
- **结论**：`no_drivable_path` 这个名字在那些帧上是**误名**——它是"没有可用的
  感知车道参考"，而 §4c 的近场证据再多也解不开它。这正是交接单要求"先写清
  哪道门开火"的原因。

### (b) 车身覆盖：四态测量 + 覆盖下限（含一条由实测抓出的假正例）

- `occupancy.body_drivable_coverage(grid, pos, heading, half_len, half_width)` →
  `on_road / off_road / unknown`（另含 `coverage / observed_cells /
  observed_frac / footprint_cells / drivable_cells`）。**`unknown` 不等于
  `on_road`**：观测不足时不许给判决。
- `safety_monitor`：`_body_coverage_gate()` 每 tick 测量（无论开关），
  `BEAMNG_BODY_COVERAGE_GATE`（默认关）才行动；规则名 `body_off_pavement`
  （已登记进 `ARBITRATION_RULES / RULE_WORST_LEVEL / _REASON_TO_RULE`）。
  时钟语义：**累计确认时间**，`unknown` 暂停（不清零、不确认），单 tick 上限
  `BODY_COV_MAX_STEP_S=1.0`（防一条 6.5 s 的慢 tick 伪造 6.5 s 证据）；
  1.0 s 降级爬行（≤1.5 m/s），3.0 s 最小风险。
- **实测抓出的假正例（重要）**：`scripts/m5_body_cov_probe.py` 在 Tech 上把车
  放在**确认在铺装内**的位姿（`787.77, 732.59`，取自 `linesfix1.json` 走过的点），
  旧阈值（`observed_cells ≥ 4`）读到的却是 `off_road`：因为前视/鱼眼最近可见地面
  在**车心前方 2.4 m**，车身矩形 72 格里只有 6 格被观测，且这 6 格正好在盲区
  边缘（2–3 m：观测 16 格、可行驶 0 格）。据此新增
  `BODY_COV_MIN_OBS_FRAC=0.3`：观测比例不足即 `unknown`。修正后同一帧读
  `unknown`，不再假装"车不在铺装上"。

### Tech 正反例（`--cam both`，主摄 ∪ 鱼眼，四次摆位，日志 `logs/goal_20260921/body_cov/`）

| 位姿（来源） | 车身矩形（车心） | +2 m | +3 m | +4 m | +6 m |
| --- | --- | --- | --- | --- | --- |
| `in` 铺装内（`linesfix1.json` t=29.7 走过） | `unknown`（6/72 观测） | `on_road` 0.556 | `on_road` 0.704 | `on_road` 0.758 | `on_road` 1.000 |
| `edge` 临界（`models2_nodqn.json` t=17.5） | `unknown`（6/72） | `on_road` 0.556 | `on_road` 0.704 | `on_road` 0.758 | `on_road` 1.000 |
| `off` 事故点（同文件 t=45.2，`787.785, 730.668`） | `unknown`（6/72） | `off_road` 0.306 | `off_road` 0.278 | `off_road` 0.258 | `off_road` 0.194 |
| `blind` 无观测（同位姿，不投影） | `unknown` | `unknown` | `unknown` | `unknown` | `unknown` |

- 分带证据（`in`，主摄+鱼眼）：`−2..+2 m` 观测 **0** 格；`+2..+3 m` 观测 16 /
  可行驶 **0**；`+3..+4 m` 24/20；`+4..+6 m` 40/40。最近观测格 = 车心前 **2.4 m**。
- **读法（诚实）**：同一个函数、同一份真实栅格、同一个矩形，只把样本沿车头平移，
  就能在铺装内读 `on_road`（0.56→1.00）、在事故点读 `off_road`（0.19→0.31）——
  说明**测量本身是对的**；而**车身矩形本身今天不可观测**（最近观测 2.4 m），
  所以门在真车上**永远不会开火**，输出是 `unknown` 而不是假 `off_road`。
  要让"车身覆盖保护"真正可用，前置条件是**能观测车身的近场通道**（交接单 P1-1），
  这一点 §4c 的四模式实验已经给了同样的结论。
- 闭环一次（`BEAMNG_BODY_COVERAGE_GATE=1`，15 帧，`logs/goal_20260921/p13_gate_on.json`）：
  `body_cov_status=unknown` × 15、`observed_frac=0.1`、`body_cov_low_s=0.0`，
  `effective_rule` 从未出现 `body_off_pavement`（该轮真实规则是
  `obstacle_very_close`/`body_crosses_boundary`）——即**开关打开也不会误刹**。
- 测试：`tests/test_body_coverage.py`（15 例：三态语义、UNKNOWN 不判、薄样本
  必须 `unknown`、`observed_frac` 必须发布、覆盖足够时仍能判 `off_road`、
  时钟暂停/上限/陈旧冻结、薄样本在监控层不得触发刹车）。

## 5. 回归状态

- `pytest tests/ -o addopts='' -q`：**2018 passed**（新增 `test_stop_digest.py`、
  `test_lateral_contract.py`、`test_reference_stability.py`、`test_body_coverage.py`、
  栈级与循环级守卫，并更新受 `lane_dev_m=None` 影响的断言）。
- `scripts/m5_offline_validate.py`：**ALL PASS**。
- `git diff --check`：干净。全部改动**未提交**。

## 6. 未完成（按交接单顺序，明确列为待办）

| 项 | 状态 |
| --- | --- |
| P1-1 前鱼眼接入 + 0–4 m 覆盖实验 | **输入实验完成**（§4c，n=2/模式）；鱼眼视角**微调未做**（交接单第 4 步）——§4d 把它确认为车身覆盖门的**前置条件** |
| P1-2 车道参考跨帧稳定门 | **代码+回归完成**；首次 A/C 对照 n=2/臂（见 §4b），未达"每臂≥5 次" |
| P1-3 无边界时车身 drivable 覆盖保护 | **完成**（§4d）：归因（车道门先开火，不是可行驶证据空）+ 四态测量 + 覆盖下限（`BODY_COV_MIN_OBS_FRAC=0.3`）+ 单测 15 例 + Tech 正反例四摆位 + 一次闭环（开关打开不误刹）。**结论：车身本身今天不可观测（最近 2.4 m），门默认关且输出 `unknown`；可用化前置 = P1-1 的近场通道** |
| P1-6 性能分解（ring 是否饿死 range/object） | **未开始**（分段遥测已就位，需一次实车跑次取数） |
| P1-7 学习模块严格 A/B | **按交接单要求暂缓**（当前只完成参与度实测） |
| 阶段 1 四臂 A/B（A/B/C/D × ≥5 交错） | **未开始**（需要 C/D 两个行为开关先实现） |
| 阶段 2 后处理四臂**闭环**指标 | 离线半已完成（§3）；闭环未做 |
| 阶段 3 跨视角标线（每视角 20–40 帧人工） | **未开始**（需人工标注投入） |
| 阶段 4 性能与调度修复 | **未开始** |
| 阶段 5 重新评估学习模块 | **未开始** |
| 阶段 6 corridor/escape hatch 标定 | **未开始** |
| Round6 正式报告 | 本文件为阶段 0 进展，不是最终报告 |

## 7. 交接单 §9 禁止事项的自查

- 未 `reset --hard` / `clean`；未覆盖既有工作树改动。
- 未使用"导航线 + 固定偏移"做横向控制或压线判定（`lane_dev_m` 改为 None 是**收紧**，
  且 `edge_over` 明确标注"仅作度量"）。
- 未把 `road_off=0`／缺边界读作"未出铺装"（契约里写明这是两种含义不可区分，
  须先读 `lat_left/lat_right`）。
- 未把 UNKNOWN 当 PASS：`lateral_digest`/`stop_digest` 缺列一律 `UNKNOWN`。
- 本轮只改统计/契约/后处理阶段化，**没有**同时动相机、planner、安全阈值与学习模型。
- 未把 4×4 多视角样例写成 200 帧能力已验证（Round5 报告已按此更正）。
- 未用单次 `damage=0` 证明安全。

## 8. 建议的下一步（交接单顺序不变）

1. **P1-2 参考跨帧稳定门**（不依赖新硬件，风险最低，直接针对"参考翻转"）
2. **P1-1 前鱼眼接入**（先做固定 drivable-band 上界实验，再双相机对照）
3. **P1-3 车身 drivable 覆盖保护**（单测 + Tech 正反例）
4. **P1-6 性能分解**（一次实车跑次即出数）
5. 然后才进入阶段 1 的四臂 A/B（每臂 ≥5 次交错），最后才是阶段 5 的模型重评。
