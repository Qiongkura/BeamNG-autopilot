# 阶段 A 实施记录（按 `FINAL_INTEGRATED_PLAN_20260922.md` T01/T02/T04/T12）

日期：2026-09-22
分支：`fix/round3-hardening-20260921`，HEAD `21c32ab`，**改动全部未提交**（工作树含用户既有未提交改动，本文只登记本轮新增/修改的文件）。
验证层级：**确定性测试 + 既有实录遥测重算**。本轮**没有**启动游戏、没有跑闭环、没有训练，
所以本文的所有结论都只到「组件/契约」层级（计划 §8.1 的第 1、2 类证据），
不含「实车行为」证据。

## 0. 一句话结论

计划 §2.2/§2.3/§2.5/§1.6 里那些被标为 S1/S2 的缺陷，**四条基准链缺陷已在代码里修掉并由反例测试钉住**：
参考发布唯一化与授权语义（T02）、末端命令不得越过当前权限（T04）、缺测不得获得 PASS 与停车口径（T01）、
实验工具的四处量错（T12）。`pytest tests/ -o addopts='' -q` → **2366 passed**（本轮开始前 2018）。

**验收台账（复核入口）**：T07/T08/T09 的"做到哪了"按计划 §7.1 的三态口径汇总在
`docs/ACCEPTANCE_T07_T08_T09_20260924.md`：**T07 = UNKNOWN（场景齐、精度类证据缺）**、
**T08 = FAIL（"减少错误关联"实测 0 处净减少）**、**T09 = PARTIAL/UNKNOWN（横向量测实车全缺、场景以单元级为主）**，
三项**都不接线**。该文件也列出了需要复核者裁定的三处口径问题。

## 1. T02 唯一参考发布 + 稳定器授权接线（计划 §2.2 / §3.3）

**新反例**：`tests/test_reference_contract.py`（18 例，单元 14 + 栈级 4）。写完后先跑出
11 例失败，再动代码——不是事后补的测试。

**代码改动**

| 落点 | 改动 |
| --- | --- |
| `lane/pairing.py` | `LaneFrame` 新增 `inferred`（至少一侧不是本帧量测）与 `obs_seq`（本帧观测序号），并加 `two_sided_measured` 属性；`_centre_line_own_lane`（中心漆线，`paired=True` 但 `right=None`）与单侧镜像标 `inferred=True` |
| `lane/lidar.py` | 单侧 LiDAR 边（`paired=False`、中心靠宽度先验）同样标 `inferred=True` |
| `lane/reference.py` | `LaneReference` 新增 `two_sided / fresh_obs / inferred / geom_id / publishable_sensor`；`scene_ref` 改为「已接受的感知参考都能进 Scene，只有 BEV 全路面中心永不进」（不再由 `frame_used` 暗中决定）；**撤销分支同时清掉撤销前的 `_single_vision` / `_divider`**（禁止用否决前的缓存布尔复活来源）；返回前加**fail-closed 兜底**：`src=sensor` 而中心为空/无效 → 一律降为 `unavailable` 并把几何字段一起清空；`select_lane_reference` 新增 `tick_id` 参数用于判 `fresh_obs` |
| `fsd_stack.py` | 新建 `_publish_reference_geometry()`：slew 限幅后的几何**写回同一个已接受对象**（各消费者同一版本）；`_sensor_lane()` 给本帧新观测盖 `obs_seq`（融合 hold/coast 保留旧序号）；稳定器改为 `paired=lane_ref_out.two_sided`、`fresh=lane_ref_out.fresh_obs`；发布 `lane_ref_geom_id / scene_ref_geom_id / lane_ref_two_sided / lane_ref_fresh_obs / lane_ref_inferred`，并在 Scene 与控制器几何不一致时置 `scene_ref_geom_mismatch` |
| `tests/test_fsd_drive_pipeline.py` | 夹具支持**逐 tick** 的 `ref_authority` 列表（T04 反例需要权限在 tick 之间下降） |

**反例覆盖（计划 §2.2 的四条 + §3.3 不变量）**

1. 构造式中心线（`paired=True, right=None`）永远拿不到 full 权限；真双侧才能；
2. hold/回放（`obs_seq` 属旧 tick）不得自我晋升：连续 20 次仍 `limited`；
3. 被路面门撤销的候选不得以 `sensor` 发布空中心（`publishable_sensor` + 栈级兜底）；
4. slew 之后 **planner Scene 与控制器几何 id 相同**（并断言限幅确实生效：请求 −6.0 m 被拒）；
5. 已接受的感知参考不再依赖 `frame_used`；BEV 全路面中心仍被挡在 Scene 之外。

## 2. T04 末端命令约束（计划 §2.3）

**先复现，后修**。计划给的组件反例在改前原样重现：上一输出 `0.55`、本帧限额 `0.15`、`dt=0.1`
→ 整形器输出 **0.49**。修完后同一条链路在**生产驾驶循环**里也复现了越权：第 1 tick 全权限发出
`steer=0.40`，第 2 tick 权限降到 `0.15`，**发到线上的仍是 0.40**。

**代码改动**

| 落点 | 改动 |
| --- | --- |
| `control/steering.py` | `update(..., cap=)`：请求与输出都夹到当前权限；权限**变小**时立刻把内部 `value` 拉回区间并重设 `rate`（否则下一步又从区间外起步）；新增 `force_state()`（外部否决后的状态同步）与 `capped` 计数（进入 `digest()`）；`cap=None/NaN` 视为无界，`force=True` 仍拥有该 tick |
| `fsd_drive.py` | 每 tick 计算一次 `_authority_cap / _authority_src`（mode 帽 ∩ ref-stability 限权）；整形器带 `cap=`；**整形之后再复核方向性约束**（away-clamp 命中即写回整形器状态）；子步重算转向时 `cap=min(曲率帽, _authority_cap)` 并同样复核；终点对准分支的 ±0.4 也受同一权限约束并同步状态；**发送前最后一道**：非硬停且超权限 → 夹回、标记 `_steer_final_clamped`；遥测新增 `steer_authority_cap / steer_authority_src / steer_final_clamped / steer_shaper{capped}` |

**反例（`tests/test_terminal_constraint.py`，12 例）**：计划场景在**线缆层**闭合（逐 tick 断言实际
`send` 参数 ≤ 该 tick 公布的权限，并断言第 2 tick 的 `capped ≥ 1`）；正例对照（同一需求在全权限下
必须大于限权值，防止「一律夹死」也能通过）；状态重基、符号不翻转、无帽行为不变、NaN 帽、force 语义。

## 3. T01 测量契约与评分（计划 §2.5 / §1.4）

| 缺陷 | 修法 | 证据 |
| --- | --- | --- |
| 横向缺列仍可能 PASS | `assess_run` 新增 `lat_left_frames/lat_right_frames/lat_left_coverage/lat_right_coverage`；`score_run` 中**每一侧各自**判定：该侧无样本 → 对应越界检查列入 `unknown`，**不再以缺列的 0 获得 no-crossing** | 撞车那次实录 `lat_left/right` 全缺 → 两项越界检查现在都是 UNKNOWN（此前是「0 越界」） |
| 单侧缺测被写成「两侧都没越界」 | 分侧判定与分侧覆盖：只缺一侧时另一侧照常给结论 | `tests/test_measurement_contract.py` |
| `by_reason[*].s` 被当成停车秒数 | 明确 `s`/`active_s` = 原因**存在**时长；新增 `stop_frames/stop_s/stop_longest_s`（按原因的**真停车**秒数与最长连续段），`spec.by_reason_fields` 自描述 | `linesfix1`：`no_drivable_path` 存在 25.89 s，**真停车 22.92 s**，最长连续停车 9.54 s——此前报告里的 25.9 s 是前者 |
| 契约把 `lat_left/right` 写成四角 | `telemetry_contract.LATERAL_FIELD_SPECS` 与 `docs/LATERAL_FIELD_CONTRACT.md` 改为**车辆中心点**并指向四角字段 `body_lat_*`；`lane_dev_m` 单位明确为米 | `tests/test_measurement_contract.py` 钉住文字 |
| 分割四臂的「像素臂」不等价 | 脚本新增 `legacy_pixel_road_constraint()`（按 `git show HEAD` 原文：fill holes → dilate(7) → 逐像素相与）；`constrain_line_to_road(elongated_frac=None)` 才是**关掉**第二档（`0.0` 是「全部放行」）；IoU 消费 `label != 255` 有效区；阶段不一致 **exit 3** | `tests/test_seg_road_constraint.py`（+2 例） |

> 由此 **Round6 报告里「旧像素约束把线删光（0.0011→0）」那句结论作废**：它不是旧线上算法的等价复现，
> 需用新的 `pixel` 臂重跑后才能引用（计划 §1.4-19）。

## 4. T12 实验工具修正（计划 §1.6）

- `m5_phase1_ab.py`：`_span` 改为**统一区间规则**（每个命中帧拥有到「下一帧」的间隔），
  不再把两段停车之间的行驶时间算成停车；新增 `--goal/--map/--seg-model` 透传，run 记录里带完整命令。
- `m5_body_cov_probe.py`：车身矩形改用**生产常量**（`vehicle_body.HALF_LENGTH_M/HALF_WIDTH_M`，
  2.2/0.9），不再自称一致却用 2.3/0.95；`--cam main/front_main` 与 `--cam front_fisheye` 各自单臂
  （修掉 `args.role` 未定义与「front_main 落进双相机分支」）；退出码现在**真的判正反例**
  （in/edge 前方应 `on_road`、off 事故点应 `off_road`、blind 必须 `unknown`），不再只判 blind。
- `m5_perf_profile.py`：门禁结论进入 `prof["gate"]` 与**退出码**（超目标 `exit 2`，未测得 UNKNOWN 仍 0，
  提示读 `gate.meets`）。实测 phase-1 臂次：`tick p95 446 ms vs 150 ms → NOT MET, exit 2`。
- `m5_perf_decompose.py`：`stale_owner` 明确为「最大 age 项」而非 SafetyMonitor 触发来源；缺 `tick_ms`
  子项不再 KeyError。
- 测试：`tests/test_tool_contract.py`（9 例）。

## 5. T10（并行项）划分协议名实不符

**已复核的缺陷**（计划 §1.5-20）：`dataset_split.leak_check` 在 `plan.groups_val` 非空时**跳过组纯度检查**，
而默认的时间尾切分**恰好**会把同一组同时放进 `groups_train` 与 `groups_val` ⇒ 共享组永远不会被报出来。
本轮内存反例（10 帧单组）改前输出 `leak=False, leaked_groups=[]`。

**修法**：两项检查各自独立，不再有条件短路——`leaked_frames`（同一帧在两侧＝硬泄漏）、
`leaked_groups` + `shared_groups`（逐组两侧帧数）、`leak = 任一成立`。
`m5_train_seg.py` 的入口把两个数字分开打印，并把「组被时间尾切分」明确命名为
**temporal-tail 开发验证**而不是「泄漏」或「无泄漏」（组隔离要用 `holdout_groups`）。
测试更新为分别钉住两种协议：时间尾切分 → 帧不重叠但组共享必须被报出；全组 holdout → 才是组纯净。

## 6. 实车（Tech）复核：新契约在真实闭环里成立

三条命令（`--attach --lane-mode sensor --strict --seconds 14 --speed 6`，起点 `779.5, 734.63`，
`BEAMNG_REF_STABILITY=1`）写成 `logs/goal_20260921/stageA_live{1,2,3}.json`：

| 检查 | 结果 |
| --- | --- |
| 规划 Scene 与控制器用**同一几何**（`lane_ref_geom_id == scene_ref_geom_id`，或两者都为空） | **21/21 帧**（其中 9 帧本 tick 没有接受的参考 ⇒ 两侧都为空）；**0 帧越界不一致**，`scene_ref_geom_mismatch` 未出现 |
| 实发转向 ≤ 该 tick 公布的权限 | **0 次越权**（三跑一致）；观测到的权限档：`0.15`（稳定门限权，16 帧）、`0.55`（mode 帽）、`1.0` |
| 稳定门授权语义 | `ref_authority`：limited 19 / full 2；`lane_ref_two_sided`：0 有 18 帧、1 有 3 帧（本路是单侧中心漆线为主）；`lane_ref_fresh_obs`：1 有 16 帧、0 有 5 帧（hold 可见） |
| 车身覆盖（P1-3） | `body_cov_status=unknown` 且 `body_cov_checked=1`（近场仍看不见车身，与前轮实测一致） |

**这次实车跑还抓出我自己遥测里的一个缺陷**：第一版 `geom_id` 在「参考完全不存在」时也会返回一个哈希
（对空字段做摘要），于是 16/20 帧出现「一个 id 有值、另一个为空」的假不一致。已改为
`_reference_geometry_id` 在 `center is None` 时返回 `None`，并把「两个 id 必须相等或同时为空」
写成显式判定（不一致即置 `scene_ref_geom_mismatch`）。修正后 21/21 帧成立。

**诚实边界**：这些是短程（14 s、约 20 帧、车基本停走）的闭环证据，只证明「契约在真实循环里成立」，
**不证明**驾驶行为变好（计划 §8.1 第 3 类证据需要更长的对照与独立安全真值）。另外 `steer_shaper.capped`
在三次实车跑里始终为 0——即「权限中途下降导致输出被裁」那条路径本轮只在确定性测试里被触发过。

## 6b. T03 源身份 / 局部证据 / 预测身份（本轮追加）

**代码**

| 落点 | 改动 |
| --- | --- |
| `vision/line_evidence.py` | `update()` 新增 `source_id / source_seq / capture_t`：投票键从「处理时刻」改为**真实观测**（每格记录 `(epoch, capture_t)`），因此①重放同一帧（处理时间变了）不再加票、②两份内容相同但确属两次曝光的帧仍算两票、③两台相机在同一曝光时刻不会把一次目击算两遍；每源**序号水位**：重复序号→`rejected_duplicate`，小幅回退→`rejected_out_of_order`，大幅回退（>2²⁰）→**计数回绕**，开新 epoch 但**保留历史**。新增 `source_events()`（刷新请求 vs 真实新增源证据、按源计数、水位）、`support_digest(points, role)`（候选自身几何的 current / history-only 支持与年龄）、`local_bands()`（按车前 0–10 / 10–25 m 分带的 supported/current/最老年龄）。`fuse_with_confidence()` 追加 `yellow_mask` 与**按证据年龄**切分的 `added_pixels_current / added_pixels_history`（阈值 `ADDED_FRESH_S=1.0`）与黄色先验单独计数。逐格 6 s 裁剪、去重、衰减语义**未改动**。 |
| `vision/hydra.py` | `FrameContext` 新增 `seq`（该相机自己的帧序号）。 |
| `vision/heads/semantic.py` | 融合调用传入 `source_id=ctx.role`、`source_seq=ctx.seq`、`capture_t=ctx.timestamp`、`yellow_mask=_yellow_prior`。 |
| `fsd_stack.py` | 主 `FrameContext` 带 `seq=tick`；`semantic_to_meta` 透传新增字段；新增 `lane_left_support / lane_right_support / lane_ref_support`（带 `role` 自述）与 `line_evidence_bands`。 |
| `fsd_drive.py` | hist 记录 `line_added_current_px / line_added_history_px / line_added_yellow_px / line_evidence_events / lane_*_support / line_evidence_bands`。 |

**回归**：`tests/test_line_evidence.py` 新增 9 例（+15→24），覆盖计划 T03 的每一条反例：同源不同处理时间、
相同内容两次真曝光、乱序与回绕序号、跨相机同一曝光瞬时、远处刷新不掩盖近处过期（分带）、
按年龄的 added 切分、黄色先验单列、候选支持摘要分 current/history-only、无几何时返回 `None`。

**实车复核**（`logs/goal_20260921/stageA_t03_live{,2}.json`，各 14 s）：

- `line_evidence_events` 每帧都在：42–44 次真实观测、28k+ 格投票、`by_source={'front_main': …}`、水位随 tick 前进；
  干净跑次里重复/乱序计数为 0（负例由确定性测试覆盖）。
- `added_pixels_*` 的年龄切分**有真实内容**：例如一帧 `current=380 / history=135`，另一帧 `27 / 158`
  ——说明"刚刚还在跟的漆"和"记忆里的漆"确实分开了。
- `line_evidence_bands`：近带 359 支持/160 current（最老 5.24 s），远带 646 支持/147 current（最老 6.30 s）
  ——**远处刷新、近处更旧**这一情形在全局比例里看不出来，分带能看见。
- `lane_left/right_support` 本次 **0 帧有值**：该路段没有发布双侧边界（参考是单侧中心漆线构造的），
  所以这两项只在有配对边界时才出现，属预期而非缺测被当成 0。
- `lane_ref_support` 报的是**车道中心**，其自身支持期望接近 0（中心是相对漆线偏移出来的）——
  已在字段里加 `role` 自述，避免把 0 读成"参考没有证据"。

**未做（T03 余项）**：真实**曝光时刻**仍不可得（`head_source_clock_basis` 记的是取帧返回时刻，本文件已如实保留该
命名）；`pose_id / ground_model_id`、人工注入（annotation 真值）与地图假设的贡献分列尚未进入契约；
异步结果迟到（`async_adopted` 的局部新鲜度）仍只在 `head_sched/consumed` 里，未与证据身份打通。

## 6c. T05 几何基准（本轮追加，详见 `docs/GEOMETRY_BASELINE.md`）

**唯一基准**：新增 `beamng_autopilot/geometry.py`——车辆原点（高于路面 `EGO_GROUND_GAP_M=0.17 m`）、
路面平面 `ego_ground_z()`、`projection_ground_z()`、姿态模型标签 `pose_label()`（`yaw_only` / `quat_6dof`）、
`resolution_distance_m()`（`d = fx·w/px`）、`nearest_ground_distance_m()`（解析盲区）。主视、鱼眼、标线、
铺装、近场与遥测**都通过它取值**；每 tick 发布 `pose_label / ground_model / ground_z / ego_ground_gap_m`，
实车实测 **19/19 帧 `pose_label=quat_6dof`**（此前主链只有 yaw，姿态只是"库里有"）。

**姿态进主链**：`_back_project_many` 新增 `rotation`（车身四元数）与 `pitch_rad`（纯相机俯仰，两者分开作用）；
`rotation is None` 时不传第三个参数（duck-typed 相机模型不受影响）；标线世界化的白线/黄线两条分支都传实测姿态。

**独立 oracle 与误差分布**（`scripts/m5_geometry_audit.py`）：oracle 按第一性原理独立实现并在真实 8 个环相机上
与被测投影对比——**p50 0.0000 m，最大 0.0142 m**（可使用带 = 命中距离 ≤60 m）；贴地平线的行单独报为
**不可用**（narrow 在 +2° 时最大 0.36 m，更靠近地平线可达数百米）。写 oracle 的过程查出它自己四个错
（挂载高度当相机高度、横向符号取反、地面平面当 z=0、返回轴序读反），**这正是"oracle 必须独立"的实证**。

**坡道建模误差**：8° 坡上主视米制纵距误差 p50 0.905 m / max 2.026 m（0° 时为 0）⇒「世界坐标 + 水平面」
不等于坡道已解决，坡道 ODD 必须显式声明。

**分辨率-检测距离预算**：320×240 下 0.1 m 标线在 ≥2 px 内只到 **9.4 m**（主视）/3.5 m（鱼眼），
640×480 才到 18.8 m；解析盲区主视 4.30 m、鱼眼 3.06 m。⇒ 远场必须靠时序证据，鱼眼的物理盲区
**不会因为域内微调消失**。

**命名**：`fisheye` 与 `fuse` 是**同一条代码路径**（主视已在同一栅格内），现在遥测报规范化模式 + `nearfield_alias`
与说明，不再让读者以为比较了两种融合算法。

**刻意保留的旧取值**：路面平面开关 `BEAMNG_GEOM_GROUND_PLANE` **默认 0**。当时的 n=1 配对显示
`lane_drivable.frac` p50 从 0.75 落到 0.50；**该对照已在阶段 C 用每臂 8 次、ABBA 交错重做**
（`docs/STAGE_C_GROUND_PLANE.md`），结果 **A 0.791 vs B 1.000**——方向相反且被噪声吞没，
**原 n=1 说法已撤回**；里程/停车/`no_drivable_path` 分布重合，唯一稳定差异 `lane_sensor_rate` 0.649→0.400 对 B 不利。
结论仍是默认保持关闭，但理由从"等对照"改成"对照做了、无增益"。

**回归**：`tests/test_geometry_baseline.py`（22 例，含逐相机 oracle 一致性与姿态进链）+ `test_camera_ring.py`
（+3：挂载几何与轴约定）+ `test_nearfield_input.py`（+2：别名与物理盲区）+ `test_body_coverage.py`
（+2：不同车型 footprint，实测 1.8 m 车 0.50 / 2.6 m 车 0.33）。

## 6d. T06 影子横向状态后验（本轮追加，阶段 B 的第一项）

**落点与分工**（计划要求「放在 lane 子包并保持与已有 tracker 分工明确」）：新增
`beamng_autopilot/lane/shadow_state.py`——`tracking.py` 管「帧可用/新鲜」，`stability.py` 管「参考配不配
有转向权限」，本模块只回答第三件事：**车相对已接受参考在哪里、这个误差变化多快、知道得有多准**。

**状态与模型**（先把简单的一版写清楚，不上 IMM）：
`x = [e, e_dot, theta, theta_dot]`，`e` = 自车相对参考的**有符号米制横向偏移**（+ = 车在参考左侧），
`theta` = 相对参考**切线**的航向差。预测用低侧滑运动学：`e_dot = v·sin(theta)`，
`theta_dot = yaw_rate − v·κ/(1+e·κ)`；`yaw_rate` 是**车辆实测**转动率（信号来源逐帧标注
`supplied` / `finite_difference` / `none`），**转向命令是另一个量，绝不参与**。量测为 (e, theta)，
单侧推断帧把宽度先验的不确定度按 `hypot(σ_e, 0.5·σ_width)` 传播；HOLD 帧（`fresh_obs=False`）**不做量测更新**，
只预测并标注原因。

**身份门**：身份 = `lane_src|ref_side`；身份变化 → 复位并记录事件；同一身份内量测跳变 > 1.5 m →
按「换了车道/换了边界」复位，**保留跳变事件而不是把它滤平**（计划原话：身份更换时不跨两条车道求导）。

**诚实的可观测性发现**：只用量测 (e, theta) 时，**速率状态 `e_dot/theta_dot` 本身弱可观**——协方差里它们的
方差在我们的时间尺度内保持很大（实车末帧 `P[1,1]=2.36`、`P[3,3]=1.41`）。这正是计划要求「输出协方差」而不是
「输出一个更平滑的曲线」的原因：**滤波器自己报出了它不知道的部分**。

**实车**（`logs/goal_20260921/t06_shadow_live.json`，37 帧，20 s）：37/37 帧有影子遥测；
`mode` 分布 measured 14 / predicted 11 / init 7 / reset 5；`σ_e` p50 **0.151 m**；末帧
`e=0.078 m, theta=0.104 rad, σ_e=0.176, σ_theta=0.070, age=1.15 s, 一致性残差 −0.0098 m/s`；
**身份事件 8 次**（`sensor|right` ↔ `perception-unavailable|unknown`）——§4b/§4c 观察到的参考抖动
在状态后验里同样可见。**耗时**：`lane_shadow` 的计算含在 tick 内（纯 numpy 的 4 状态更新，未单独计时，
尚未做性能分解——列入余项）。

**回归**：`tests/test_shadow_state.py`（18 例）：量测符号约定（e/theta/κ）、协方差随量测收缩/随 HOLD 增长并
满足对称正定、HOLD 不算量测、身份变化复位并记录、跳变事件保留、单侧宽度不确定度传播、低侧滑一致性残差
（构造 0.5 m/s 横移 + 0.3 rad 航向差 → 残差 ≈ 0.204 m/s）、**响应延迟被测量而不是假定**（阶跃后 2 拍内进入
0.1 m 内，且明确写"滤波器必须先动起来"）、以及**影子只读**（control 子包不得 import 本模块、估计器没有
任何执行器输出、驾驶循环只把它写进遥测）。

**没有声称的东西**：本轮**没有**做「滤波 vs 基线」的误差对比——真值横向偏移需要**独立真值**（计划 §7），
现有实车跑只有感知自己的参考，因此「偏移/速率误差、错误接受、重新捕获时间」这几项**未测**，也**不声称收益**。
按计划「无收益则保留简单方案」，本模块保持 shadow：驾驶循环不读它的任何输出去改转向。

## 6e. T07 真实道路边界与局部拓扑（本轮追加，评估为主，详见 `docs/BOUNDARY_EVIDENCE.md`）

**前置能力核查（计划要求"满足条件才复现滑束"）**：实机核查 `beamngpy` 的 LiDAR 只返回
`type / pointCloud / colours`——**没有逐束 ID、垂直角、扫描时间**；原始点云一帧 ~414k 点；
且 **attach 后第一次 poll 返回 0 点**（不预热就会静默记下空点云）。结论：Zhang 的滑束法与
`δxy/δz/nv` 公式**无法复现**，按计划走退化方案（沿面高度突变 + 局部采样尺度），并把缺口写进代码与文档。

**三类证据分开发布**（`lane/boundary_evidence.py`）：`curb_candidates`（LiDAR 沿面突变）、
`pavement_edges`（语义掩码边沿，复用 `lane/pavement`）、`obstacle_entities`（障碍框），各自带 provenance 与
阈值；`associate()` **只报告**重合（`merged: False`、几何不变）。证据结构里**没有**任何
`drivable/permission/authority/crossable` 字段（测试钉住）——按计划"粗分割不得成为通行证"。

**几何自适应阈值**：竖直束间距 `d·tan(3.36°)` 在 5 m 就有 0.29 m > 路缘 0.15 m，所以**基于束间距的阈值
在几米外必然看不见路缘**（第一版实测 0 命中，已改）；改成 `clip(k×实测沿面邻距, 0.05, 0.6×路缘高)`，
上限锁死在半个路缘高——阈值不允许超过它要找的信号。检测器用 range×bearing 距离图像找台阶。

**臂对照**（`scripts/m5_boundary_arms.py` + `m5_capture_boundary.py` 实采）：
合成 0.15 m 路缘上 **fixed 0 个 vs adaptive 4 个**；实采一帧（59207 点）**fixed 1326 vs adaptive 4790**。
**没有独立真值 ⇒ 只报计数与阈值，不报精度**（脚本结尾明写）。同帧 `pavement_edge` 为 0（该帧 abstain）、
`obstacle_entities` 为 0 是**采集脚本没记障碍列表**（已知缺口）。

**诚实边界**：单帧真实点云给 4790 个候选 ⇒ **检测器选择性远不够**，现在**不能**当控制输入；要可用至少还需
与铺装边沿关联 + 时间持续性过滤，这两件本轮**没有**做。验收矩阵（坡道/护栏/草边/分合流/路口）**未做**。

**回归**：`tests/test_boundary_evidence.py`（14 例）。

> **更新（§6k）**：本节把"加时间持续性过滤"当作补齐路径；多帧实采序列（8 帧）实测表明
> 持续性过滤对低选择性前端**无效**（候选的 24% 是反复重现的静止物体），真正有效的是
> "边界是车旁一条细曲线"的几何约束。本节其余结论（能力缺口、三类证据分开、几何阈值）不变。

## 6f. T08 地图辅助候选关联（本轮追加，影子，详见 `docs/MAP_ASSOCIATION.md`）

**输入只用真实字段**：`RoadRuleView` 的链路身份（`n1->n2`）、弦方向、`lanes` 字符串（**只当数量**，
绝不当宽度/偏移）、曲率（`inRadius/outRadius`）、单行/可驾驶性；感知侧用**带身份**的标线候选
（id/side/kind/bearing/span/confidence）。**输出只有** 关联分数、`hypothesis_id`、实际用到的 `map_fields`、
逐条 `conflicts`（点名地图字段 vs 感知量）与 `abstain` 原因——**没有任何横向几何**（`AssociationResult`
里不存在 `center/offset/lateral/target/authority/drivable/crossable`；模块无横向目标函数；分数不依赖自车横向位置，
三条测试分别钉住字段名、签名、源码）。

**可证伪的弃权/冲突**：无链路→弃权且**不改分数**（空行为：无地图时 B 臂 ≡ A 臂）；hold 候选→弃权归零；
分岔（多条候选链路）→**弃权而不替地图选路**；车道数与实测宽度矛盾→冲突；方向软(>25°)/硬(>45°)；
错误地图 180°→硬冲突。A/B 对照直接算 `false_acceptance_risk`＝"带冲突仍被改分的候选数"，
B 臂只重打分、不创建/删除/移动几何（测试钉住）。

**实车一帧跑出两个发现**（`logs/goal_20260921/map_assoc_probe{,2}.json`）：
1. **地图 `rightHandDrive: false` 与道路矛盾**：这段路是右侧通行，第一版据此把 **4 个真实候选里的 3 个**
   标成 `side_mismatch`——先验在砍真实观测。修法：侧别先验**默认关闭**，未信任时该字段也不出现在
   `map_fields`；修正后同帧只剩方向类冲突（2/4）。
2. **链路方向是弦不是局部切线**：实测候选与弦差 33–35°，说明方向先验应与局部切线比较；
   本轮**只登记不修**（需要链路几何采样）。

**顺手修的 bug**：`RoadRuleView` 的 `bool(rhd)` 会把字符串 `"false"` 读成 True；改为显式 `_as_bool()`
（未知值 → UNKNOWN）。`tests/test_map_association.py` 27 例（含"标志未核验前不得因它降分"）。

**未做**：未接线（不改参考权限）；无真值身份标注 ⇒ **未测关联正确率**，不引用配对率数字；
分岔/宽路/急弯/单侧线/过期地图的完整矩阵只做了单元级合成反例 + 实车一帧。

> **已被 §6m 取代（2026-09-24）**：上面"未测关联正确率"这一条不再成立——逐候选的**引擎漆线确认**已接上，
> 四点实测见 §6m；"分岔/宽路/急弯/单侧线/过期地图矩阵只做单元级"也不再成立（急弯/路口已实车）。

## 6g. T09 保持机制联合条件 + 有界风险预测（本轮追加，可读遥测 + 仅收缩门）

**先核实现状**（计划点名的三处"不能过度解读"）：`PathHold` 的 `PATH_HOLD_MAX_LAT_M = 2.5` 是"自车到保持路径**最近点**的距离"，
不是累计行驶距离；`PATH_HOLD_MIN_AHEAD_M = 4.0` 是**剩余弧长**，不是航位推算误差界；而保持寿命以 `offered_at`
（最近一次 **offer**）计时，**不是**最近一次**真实观测**——反复 offer 会把寿命刷新。这四条联合条件
（时间 / 累计行驶 / 航向与横向不确定度 / 剩余可停车空间）此前都没有。

**新增 `planning/hold_audit.py`**：`audit_hold()` 输出四条联合条件与 `satisfied / failed / unknown`：
观测年龄（`offered_at` 不参与，签名里都没有这个参数，测试钉住）、**自车实际行驶距离**（调用方按"上次新鲜观测"累计）、
影子状态的 `sigma_lat/sigma_theta`、以及"剩余弧长 vs 保守停车距离"。**未测到的条件不算 satisfied**（`unknown` 与
`failed` 分开）。门 `BEAMNG_HOLD_JOINT_GATE` **默认关**，且**只能拒绝**（关掉保持这一 tick），
本轮**不延长**任何窗口（grace 0.30 / max 0.80 未动，有测试钉住）。

**新增 `lane/lateral_risk.py`**：有符号间隙（从**车身**量起，不是车心）、**首次越界**距离与时间、
保守停车距离、证据新鲜度区间。三条硬规矩：**缺边界 → UNKNOWN**（不是 0 间隙，也不除以近零接近率）、
**不趋近 → 没有有限越界时间**（平行行驶不产生 TLC）、**公式不是安全证明**（停车距离复用项目既有的
`RISK_BRAKE_DECEL_MPS2` 与 `stop_distance_m`，并注明该减速度是**声明常量**而非本车实测；
坡度/附着/制动建立时间都在适用域之外）。风险输出无任何执行器字段（测试钉住）。

**实车**（`logs/goal_20260921/t09_risk_live{,2}.json`，各 36 tick，**两次运行必须分开引用**）：
- `lateral_risk` 36/36 tick，但**该路段没有发布双侧边界** ⇒ `gap_left_m/gap_right_m/cross_distance_m/cross_time_s`
  全为 **UNKNOWN**（`unknown=['evidence_interval','first_crossing','gap_left','gap_right']` 36/36，
  `evidence.state='UNKNOWN'`）——计划要求的"缺边界输出 UNKNOWN"是**实车跑出来的**，不是断言出来的；
  同时也说明这条路上的横向风险**今天根本算不出来**，得先有双边边界。
  **能算出来的那一项有数字**：`stopping_margin_m` 36/36 非空，run1 **0.500–1.045 m**、run2 **0.500–1.892 m**，
  理由字段一致写着"flat ground, dry asphalt, no brake build-up; deceleration is a declared constant"。
- `hold_audit` 36/36 tick，两次都 `enforced=False`（门默认关，行为未变）、`lifetime_source='observation'` 36/36
  （寿命按真实观测计时，不按 offer）。失败原因**两次不同，正是修复前后的对照**：
  - **run1（修复前）**：`satisfied` 8/36；`failed` 22/36 是 `sigma_lat: 1.00 m > 0.60 m` + `sigma_theta: 1.000 rad`
    ——把"尚未测量"的初值协方差当成"measured 且超限"；`sigma_lat_m` 非空 33/36（0.096–1.000）。
  - **run2（修复后）**：`satisfied` 7/36；`failed` 20/36 变成 `remaining path unknown`，而 `sigma_lat/sigma_theta`
    进入 **`unknown`**（非空只剩 7/36，0.081–0.291）——**未测量 → unknown，不再伪造成超限失败**。
  - 两次的 `travelled_since_obs_m` 分别 ≤0.378 m / 0 m，`observation_age_s` ≤1.69 s / 0 s；
    `remaining_arc_m` 分别 11/36、16/36 非空（7.99–12.0 / 6.26–12.0 m）→"该 tick 没有可用路径"是**缺测**，不是距离超限。

**回归**：`tests/test_lateral_risk.py` 29 例（含不趋近/缺边界/零速/大横摆/已在界外/未测条件不得 satisfied）+
`test_path_hold.py` +3（窗口数值未动、门默认关、期满后重复请求仍被拒）。

## 6h. T11 前级单变量消融（本轮追加，详见 `docs/FRONTEND_ABLATION.md`）

**设计**：只有**前级**在变——数据（人工标注 20 帧，line 占 2.5%）、后处理（生产链：闭运算→路面约束→形状过滤）、
度量（valid 区 `label != 255`）全部固定；四臂 `model`（生产学习前级）/`abs`（既有绝对阈值颜色候选）/
`tophat_otsu`（Top-Hat+Otsu）/`ycbcr_pct`（文献分位 Y0.97+Cb0.02，**未在本项目标定**）。
脚本**不训练、不调阈值、不改任何默认**（测试钉住：不改 env/config、默认走生产解析器）。

**结果（n=20，中位）**：`model` raw/final IoU **0.567/0.567**、误删真线 1882 px、**保留假线 380 px**；
`ycbcr_pct` 0.201→0.271、2523/2994；`tophat_otsu` 0.058→**0.169**、3681/2110；`abs` 0.025→0.025、
0/**182618**。

**按失败类型的结论**：① 学习前级领先且**假线最少**（经典臂是它的 5–8 倍）；② **后处理对经典臂的增益远大于
对学习臂**（tophat +11 点、ycbcr +7 点、model 近 0），代价是后处理耗时高一个量级（41–42 ms vs 2.5 ms，
因为要处理多得多的连通域）；③ **`abs` 臂是一个具体缺陷**：把整帧判成线（假线 18 万 px），后处理砍不掉——
这是计划优先序 2"明确光照失败"的实测例子；④ 学习前级是**召回受限**（漏真线 1882 px > 假线 380 px）。

**数据可用性（先查后做）**：第一版误用 `run_20260815_010127`（每帧仅 ~20 个真线像素，IoU 被假线支配，
model 臂 0.0046）→ 扫描 111 个标注目录后换成人工标注批；自动标注批 `run_tech_ann_*_y`（line 占 10.4%）
**明确排除**（含注入/自动 palette 成分）。

**未做**：训练类消融（弱特征增强/多尺度/轻量骨干/存在性分支）需先有标注与损失定义；世界系曲线几何误差
（缺可见段标注）；身份与错误接受（无真值）；闭环端到端时延。
`tests/test_frontend_ablation.py` 12 例。

## 6i. 阶段 C 第一项：路面平面统一配对 A/B（已完成，结论=不采纳）

选定项：`BEAMNG_GEOM_GROUND_PLANE`（T05 影子验证过的那一项）。方法：`scripts/m5_stage_c_ab.py`
**每臂 10 次、轮内 ABBA 交错**（20 次全成、0 重复文件）、同起点/同路线/同 CLI（`--lane-mode sensor --strict`）、
每跑一文件（含命令、hist 路径、commit 与 dirty=113）、`stageC_summary.json` 汇总。

**结论：无可测增益**——里程 p50 A 2.305 vs B 2.78（区间几乎完全重叠）、停车 17.46 vs 16.79、
`no_drivable_path` 9.16 vs 10.19；唯一稳定差异是 `lane_sensor_rate` 0.635→0.400（对 B 不利）。
**撤回**此前 n=1 的"0.75→0.50"说法：n=8 后为 A 0.791 vs B 1.000（方向相反、被噪声吞没）。
默认**保持关闭**，符合"没证明增益的变更不进入下一阶段"。

**记录里点名的自身缺陷**：① `off_pavement_frames` 与 `body_cross_frames` 在 20 次跑里**帧集合完全相同**，
且与 `lat_left` 有值帧几乎一致——它们不是三个独立检查，而是"感知边界撤销"这一个信号的三种记账
（抽查帧：`road_off=1.604`、`lat_left=None`）；② 20/20 `damage=0` **不作安全证据**（车几乎没动，里程 0.6–5.7 m）；
③ 本 ODD 是停走主导（里程 1–5.7 m/20 s），本轮只能就"证据门更松/更紧"发言。
工具自身也修了一处：ABBA 会让同一臂在一轮内出现两次，原先按 (arm, round) 命名导致**后一次覆盖前一次的遥测文件**
（16 次里 8 个文件被覆盖）——已改为带全局序号，并以修正后的命名**重跑**得到干净记录
（`logs/goal_20260921/stageC_plane/`，旧的留在 `stageC_plane_badtag/` 并附 README 说明）。

## 6j. 阶段 C 余项：更长对照 + 压线/出铺装的独立真值核对（已完成可做部分）

**更长对照**（`--goal 860 726`、45 s、每臂 10 次 ABBA 交错、20 次全成、每跑独立 vis 目录）：
里程 p50 A **8.50** vs B **5.87**、`no_drivable_path` A **19.29** vs B **22.39**、`lane_sensor_rate` 0.503 vs 0.451
——**B 在主进度指标上更差**（区间仍重叠）；damage 20/20=0 **不作安全证据**（停车占 33–42 s/45 s）。
结论与短路线一致：**无增益，默认保持关闭**。

**核对发现并修掉的度量缺陷**：压线/出铺装计数**没有"边界已发布"门**。20 次长跑共 107 个被标帧；
按严格门（只认 `lat_left`/`lat_right` 已发布）重算，**A 臂 23 帧有边界 + 13 帧无边界**（36% 是
"没测到边界却声称压线"），B 臂 70 + 1。第一版门误把 `body_lat_*` 当边界（该字段在被标帧上恒等于 `road_off`），
用**已录制跑次回算**时当场暴露，已收紧。

**人工核对 3 帧**：有边界发布的两帧（`B4_19@10`、`B1_07@50`）**成立**（左漆线确实从车下穿过，
且 `lat_left=0.123` 与车宽半宽 0.9 m 自洽）；无边界发布的一帧（`A0_01@25`）**不成立**
（车在铺装内、漆线在左约 2 m，`line_lat=+2.06`）。⇒ **UNKNOWN 不能当"越界"**，与"UNKNOWN≠PASS"同构。

**核对样本已放大到 24 帧**（20 被标 + 4 对照）：新增 `--vis-on-flag` / `--vis-control-every`（只渲染被标帧与
对照帧）+ **文件名带时间戳** + `vis_index.json` 连接键；核对时按最近邻时间配对（35 帧全部匹配，|Δt| ≤ 0.016 s）。
判读结果：**有边界发布的声明 16/17 成立**（误报 ≈ 6%，唯一不成立的是临界帧）、**无边界发布的声明 3 帧
无法用项目自身数据判定**（`road_off` 5.9–12.3 m 但 `lat_left=None`）、**4 个对照帧未见漏报**（裁剪区只覆盖车身左侧，
对漏报只是部分证据）。过程中修掉两个真问题：`--vis 1` 逐帧导出会把循环拖死（20 分钟无输出，已停止）；
**图名的循环计数 ≠ 遥测行号**（实测图 `line_lat=2.627` vs 行 `2.052`），按序号配对会得到错误的真值表——
改用时间戳索引后才成立；捕获决策已抽成 `fsd_drive.vis_should_write()` 并有决策矩阵测试。
详见 `docs/STAGE_C_TRUTH.md`。

**逐帧米制真值测量已落地**（`scripts/m5_truth_measure.py`，用影子记录的**原始 RGB+位姿**而非叠加图）：
量"漆线内侧边缘"与"车身左前角"的**米制余量**，判 `cross/clear/ambiguous`。**4 段采集 330 帧**（含新采的 60 s 段 `seg_C1`）：
**FP = 25/247 = 10.1%、FN = 9/231 = 3.9%、TP = 0**（有边界声明无一被证实；换段后比率一致）；
两个方向各留一帧独立验证
（`tr2_B3` 帧 31 = 误报样例、`tr2_B4` 帧 37 = 漏报样例）。**过程中撤回上一轮的 24 帧目视结论**：
那些图是叠加图，而且**前视相机在画面下缘看到自车引擎盖**（盖上还有亮条纹）——第一版测量把命中点画回图上时，
命中的是**引擎盖边缘**而不是漆线，说明目视读到的"漆线下穿"很可能是画在盖子上的模型掩码。
修法：行带移到引擎盖之外（5–9 m）+ 行上限 0.66 高度，并加**序号配对**（时间配对会错位：影子 `t` 从 11.36 起、
遥测从 0.71 起，错位版本给出过 54% 的假漏报率）。`ambiguous` 70 帧（29%）单独计数、不并入比率。

**又踩到两条产物纪律**：`seg_C2`（从无铺装的事故点位姿起跑）**采集超时失败**（exit 124、无遥测），**未纳入**统计；
汇总时 `glob` 把**修复前**的产物也扫了进去，一度把 FN 污染成 23/257 —— 已把无效产物移入
`truth_measure_INVALID/` 并附 README，汇总只统计有效产物。

**仍未做**：双边边界直道仍未找到
（扫描到的 100% 配对运行全是 **map 模式**，按计划不能作横向真值）。

## 6k. T07 余项：多帧实采序列上的选择性与持续性（本轮追加，详见 `docs/BOUNDARY_EVIDENCE.md` §5）

计划 T07 的余项之一是"把持续性过滤放到多帧真实点云上量测"。本轮把采集脚本扩成**序列模式**
（`m5_capture_boundary.py --frames K --step-m S`，证据级别 `placed_step_replay`：真实点云/位姿/掩码，
运动是摆位不是行驶），采到 8 帧 italy 山路段序列（步长 1.5 m，实际 1.42 m/帧，每帧 ~5.9 万点 + 语义掩码），
新增离线指标工具 `scripts/m5_boundary_sequence_metrics.py`。同一批真实帧上的单变量对照：

| 每帧 p50 | 自适应高度台阶臂 | 线一致性臂 |
| --- | --- | --- |
| 候选点数 | 4845 | **10** |
| 横向残差（PCA） | 2.04 m | **0.025 m** |
| 世界格重现（3 s 窗，0.5 m 格） | 存活 1156 格 | 存活 10 点（全部） |
| 掩码边沿命中率 | 15.8% | **61.3%** |
| 到掩码边沿像素距离 | 15.2 px | **5.5 px** |

三条结论：

1. **否定结论**：持续性过滤**救不了**低选择性前端——4845 个候选里 1156 个世界格在 3 s 内重现，
   静止的石头/草丛/墙根都会重现。§6e 曾把"时间持续性过滤"当作补齐路径，实测否掉：它是必要条件，
   不是充分条件。
2. **采纳**：`lane/boundary_evidence.py::line_consistent_candidates`（"边界是车旁一条细曲线"）在**同一批帧**上
   把候选收缩到 0.2%、残差缩小 80 倍、掩码边沿命中率提到 61.3%。按**链**发布（前向跨度、横向偏移、链长），
   多条链各自带身份，**不做平均**。22 例回归（`tests/test_boundary_evidence.py`）把偏移在自车系测量、
   散点被拒、车身下方不算边界、横向跳变断链、两线只发布较长者、相对角 30° 如实不发布全部钉死。
3. **否决第二个臂**：范围自适应薄度容差（按束间距放宽"薄"）在 8 帧里有 4 帧**零发布**、第 8 帧把多条线
   混成一个偏移。固定 0.35 m 的代价是可用距离 ~3–9 m，已写进代码注释。

**双边直道样本（本轮补齐长期缺口，详见 `docs/BOUNDARY_EVIDENCE.md` §5b）**：游戏重启后在 italy 默认出生点
`(729.63, 763.91)`（历史双边路段的 x,y；朝向按四次探测定为 205.6°，25.6°/45°/130.9° 会在 ~7 m 内驶出铺装）
采集 6 帧 × 1.0 m（全程铺装占比 0.417–0.471）。结果两条：**修**——链发布原先按"最长链"选中了车左 18.9–19.7 m 的
远墙并当成车道边界，现改为按到自车路径的距离取相关链（`LINE_LAT_MAX_M = 8.0` m，越带链按身份记入 `distant_chains`）；
**不修**——把薄度容差 0.35→0.6/0.9 m 对该路段发布率仍是 0.0（卡点不在阈值），而山区段同一放宽会用精度换覆盖
（发布率 0.50→0.75 但边沿命中 0.61→0.47、残差 0.025→0.068 m），故只登记为候选改动。
原始台阶候选在该路段两侧都强一致（左 0.654 @4.0 px、右 0.541 @6.4 px），但**链层一条都没发布**——
所以"双边"目前是原始证据层面的结论，不是参考层面的。

**验收矩阵两行（本轮补齐，详见 `docs/BOUNDARY_EVIDENCE.md` §5c）**：采集工具新增 `--annotations`——逐帧数**引擎标注**
类别（`GUARD_RAIL`/`GRASS`/`NATURE`/`SIDEWALK`），所以"这算什么路段"是测量值而不是操作者描述。
**护栏行**（`787.77, 732.59`，6 帧，`GUARD_RAIL` 1449 px/帧、6/6）：台阶候选偏向护栏一侧（右 3386 vs 左 320），
链臂 3/6 帧发布且**发布线投影 100% 落在掩码边沿**；**软自然边行**（`729.63, 763.91`，6 帧，
`GRASS` 733 + `NATURE` 9745、`GUARD_RAIL` 仅 277 远景）：原始候选与掩码边沿一致性更高（0.622 @4.12 px）
但**链臂 0/6 发布**。两行合读给出机制：**链判据筛的是硬线性结构（护栏/路缘/墙脚），软边过不了 3 个 1 m 分箱
横向跨度 ≤0.35 m 这一关**，这同时解释了 §5b 宽路 0 发布。人行道路缘（`SIDEWALK`）两行均为 0，该行仍未测得。

**漏检与限制**：8 帧中 4 帧未发布线（该处道路相对摆位朝向转出 >22°，超过 1 m 分箱 + 0.4 m 链阶梯的容差）；
掩码是另一路传感器的一致性检查**不是真值**；该段无独立几何真值，故仍**没有精度声称**。
线一致性只在离线工具中使用，**驾驶路径未接线**（要接需按计划 §8.2 做单因子闭环 A/B）。

## 6l. T10 余项：曲线标注 schema + 时间切片工具（本轮追加，详见 `docs/ANNOTATION_SCHEMA.md`）

计划 T10 登记了两项：标注 schema 需要曲线 ID/角色/可见段/遮挡/起止/属性/unknown，且**中心线、像素漆面、
推断延伸必须分开保存**；`scripts/m5_timeslice_annotate.py` 被列为"提议，当前不存在"。本轮两项都落地：

- `beamng_autopilot/labeling/curve_schema.py`：记录锚在首帧、曲线跨帧只出现一次；
  `source ∈ {pixel_paint, centre_line, inferred_extension}`，**推断段必须带 `derived_from`、量测段不得带**；
  被遮挡段必须 `visible=False`；`unknown` 是显式字段名列表；像素类别钉死 `0/1/2/255`，
  未定义值如实报告、255 计为 ignore 而非 background；写读都先校验。
  回归 19 例（每条规则一个反例）。
- `scripts/m5_timeslice_annotate.py`：按固定图像行取**对比度**（max−median）堆时间切片
  （第一版用行均值，被背景梯度主导且丢掉线的列位置，实测后改掉）；控制点按时间线性插值、
  两端不外推、**断点既不跨越插值也切断输出段**；落回帧做窗口内亮/暗极值 snap，产物一律
  `inferred_extension` 带来源；`spot_check` 报告逐帧行误差 p50/max/容差内占比，单侧帧记 UNKNOWN。
  回归 21 例（含"线在窗口外时切片没有对比度""断点切段""CLI 拒绝无 rgb 的 episode"）。

**本轮追加（身份与跨视角）**：采集侧记录 `map_name`/`source_id`/逐帧 `t_wall` 与**跨视角共享的 `exposure` 计数**；
`dataset_split.frame_refs_from_meta()` 成为唯一索引入口（缺什么报什么，缺钟时 `t_is_index=True` 而不假装是秒）；
`cross_view_groups()`/`cross_view_leak()` 查"同一曝光的多视角是否被切到两侧"，**没有曝光计数时报 `checked=False`**；
`m5_train_seg.py` 的 by-map-scene 分支接入并打印身份回退与跨视角结论（过滤后索引取自保序序列）。
回归 `tests/test_dataset_split.py` 新增 7 例。

**泄漏审计（本轮追加）**：`dataset_split.split_audit()` 把五类陈述分开报——帧重叠、组重叠、
**复制样本**（`same_path` > `same_exposure_view` > `same_wall_clock`，按强度去重）、
**时间邻近**（≤ `--split-gap-s`，默认 0.5 s，且**排除**已判为同一样本的对）、跨视角同曝光；
每段自带 `checked` 与原因。训练入口 by-map-scene 分支打印这五段并分"严重/注意"两级。
旧采集（无 `meta.json`）实测得到的是**"未检查"而非"干净"**；新采集由 `frames[].path` 提供最强证据。
回归 7 例。

**锁定留出集（本轮追加）**：`scripts/m5_freeze_holdout.py` 从 run 目录冻结清单，`--dev-runs` 里的 run
**拒绝冻结**，`--holdout-all` 把整跑作为留出侧；身份是 `<采集目录>/<视角>`（第一版用角色目录名，
两次采集 digest 撞车，已修 + 回归）。排除清单 = `eval_v8..v13` 用过的 13 个 run + 全部 `manual_*`；
`pseudo_us_map*` 因标签是伪标签不可作测试集。新采两份引擎标注留出集并冻结：
`frozen_holdout_town.json`（39 帧，城镇段，**几乎无标线**，digest `8736cd945940f3e3`）、
`frozen_holdout_lines.json`（39 帧，有标线，digest `1a087769a5842f88`）。首次锁定集评估：
`v13b` line IoU 0.1101 / asphalt 0.9403；`v13b_dice` line IoU **0.0000** / asphalt 0.9523——
"总准确率高"与"标线可用"在这份从未参与开发的集上是分开的两件事。

**候选提取修正与上游缺陷（本轮追加，详见 `docs/FRONTEND_ABLATION.md` §5.3）**：给候选并集的经典 CV 臂加上
黄色臂已有的门（细长 + ≥50% 在路面上），学习掩码支撑的候选原样保留（第一版对全部候选施门，
在城市场景一次删光 100 个掩码支撑候选，已按实测改正）。效果：路外候选 176→138、直线段匹配/零假设
信噪比 2.1×→4.8×；**但候选集对引擎漆线的覆盖率仍 ≈0**。反例（城市路口 f0）：引擎漆线 2662 px、
模型线掩码 18 986 px（precision 0.130，**左半幅 0.000**、右半幅 0.264）、候选像素 247 px 中 **0 落在漆线上**；
概率阈值扫描（0.50/0.70/0.85/0.95）precision 0.207/0.265/0.129/0.126——**不是阈值问题**。
因此登记为**上游感知缺陷（线通道精度）**，T08 的关联 A/B 仍不开跑。

**线通道精度修复：实施、复测、不采纳（本轮追加，详见 `docs/FRONTEND_ABLATION.md` §5.5）**：
`refine_line_mask`（结构判据 + **极性无关**的局部路面离群判据；缺参照时保留并记 unknown）已实现并有单测；
判据分解显示**结构判据在路缘场景损失 42.8% 漆线**（真漆线 42.8% 落在模型路面掩码之外）、
**外观判据在 `frozen_lines` 上把 recall 打到 0**；候选路径 A/B 更决定：直线段匹配 0.132→0.000。
故**默认保持关闭**，结论是"需要模型/数据层修复而非后处理"。两份既有留出集因参与该决策**已作废**，
新冻结 `frozen_holdout_street2.json`（digest `214fc491c98f4724`，线类很稀）。
**"有漆线且从未用于决策"的留出集：本轮已补**（详见 §5.6）：用引擎标注沿四个未用方向的路线搜索 + 每 3 m 细扫，
在一条从未采集过的路上找到约 23 m 的有漆线窗口（line_px 761→2250→529→0），
从 `(704.8, 703.1)` 以 `--step-m 1.0` 采 20 帧（步进 1.05 m/步、总 19.8 m、line_px p50 **2081**），
冻结为 `frozen_holdout_lines2.json`（digest **`a0f229e0a5a4531f`**）。首次数值：
`v13b` **precision 0.175 / recall 0.924**——"线通道精度不足"在第三个独立场景复现，
这份集即数据层修复的评测基线。顺带修了采集器的静止陷阱（`--step` 不摆位 → 20 帧同一位姿；
新增 `--step-m` 并加契约测试）。

**线通道数据层修复：受控微调 + 前后对比（本轮首个正向结果，详见 `docs/FRONTEND_ABLATION.md` §5.7）**：
以 `v13b` 为初值、158 帧同源标注做 3 轮微调（固定配方，~32 s）：冻结留出集上
**precision 0.175 → 0.423、recall 0.924 → 0.928、IoU 0.172 → 0.410**；候选层
`on_engine_line` **0 → 9**、匹配率是零假设的 **8 倍**、匹配者角色一致 0.9375——T08 的
"候选↔真实标线对应关系"在**有漆线路段首次部分成立**。边界（跨场景表）：
城市路口也改善（0.156/0.921 → 0.236/0.968），但**城郊引道无变化、路缘位姿反而退化到 0**——
小样本微调学会了"训练分布里那类有漆线路面"而非"线"这个概念，**因此不接线**，
下一步要的是**场景多样化数据**而不是加轮数；内部 line IoU 0.533 高于留出集 0.410（部分泛化）、该留出集**已因本次决策消耗**
（digest `a0f229e0a5a4531f` 转开发资料，下次需找新窗口再冻结）、微调检查点**未接线**（生产仍是默认模型，
接线需 T07/T08/T09 全链验收）。

**场景多样化数据 + 重训（本轮追加，详见 `docs/FRONTEND_ABLATION.md` §5.8）**：沿六个新方向用引擎标注搜到
20 个有漆线位姿，采得 `diverse_wide`（p50 1569）与 `diverse_town`（p50 3547）两组可用数据
（`curve`/`plain` 太薄未入训练；另有两处因"直线摆步离开铺装"采集失败，工具缺口已登记）。
用 188 帧重训（固定配方）后**上一轮的两处退化消失**：路缘位姿 0.000/0.000 → **0.554/0.411**、
城郊引道 0.000/0.000 → **0.575/0.631**；新冻结独立留出集 `frozen_holdout_wide2.json`
（digest `5af57693878615cf`，训练窗口外 ~50 m）上 **precision 0.265 → 0.597、recall 0.942 → 0.987**。
**仍不接线**：同一街道走廊内的未见路段不算"未见场景"，接线需 T07/T08/T09 全链验收 + 闭环 A/B；
下一步登记跨地图/跨街区测试。另更正：三个 `ident_probe_*` 采集是**静止 20 帧**（同视角多次渲染）。

**本轮交接证据（计划 §8.4 八项）**：已按规划组装为 `docs/ROUND8_HANDOVER_20260924.md`——
工作树/哈希/开关/清单（HEAD `55f41594`、已跟踪改动 0、22 个行为开关全未设、
`ft_t11` sha256 `a4c6e07e1b53b935`、冻结帧 digest `5af57693878615cf`）、分字段覆盖与 UNKNOWN、
独立几何与生产一致性（`production_mismatch=0`）、感知层 TP/漏检、性能与 deadline、
以及"假设→是否激活→输入→实测→支持/推翻→未知"的逐条表。
**明确空白**：第 3 类（Tech 闭环）证据本轮**没有**——驾驶层的碰撞/压线/出铺装/停车/推进/deadline
全部未测；`ft_t11` 未被任何驾驶路径引用（`grep` 可复核），生产默认仍是 `seg_model/best.pt`。
另按计划 §8.3 模板实跑了 `pytest tests/`（2328 passed）、`m5_offline_validate.py`（**ALL PASS**）、
`m5_seg_stage_eval.py --runs <冻结run> --model <ckpt> --arms`（两模型 raw→final 全链路）。

**路网引导沿路采集（本轮追加，详见 `docs/FRONTEND_ABLATION.md` §5.10）**：采集器新增 `--follow-road`
（沿路网放置 + 新朝向 + **不在路网时先吸附**；实测该图路网含无铺装小径，因此"在路网上"≠"在铺装上"，
必须配路面校验）。在 east_coast 已知好路反向采到 2×30 帧（`line_px` p50 **6537/6968**，沿弯道跟住未崩溃）。
用 248 帧重训（`ft_t12`）：**本图** `wide2` precision 0.597→**0.747**、IoU 0.592→**0.727**；
但**跨图留出集没有改善**（0.303→0.218、IoU 0.192→0.149）——一条路的反向增量不足以修复跨图召回缺口
（0.32–0.35 vs 本图 0.94–0.99）。下一步：多条路/多区域/多光照的跨图数据。

**跨图多样数据 + 专项评估（本轮追加，详见 `docs/FRONTEND_ABLATION.md` §5.11）**：按"路网引导+路面校验"采到
east_coast 同路 2×30 帧（`line_px` p50 6537/6968）与 **gridmap_v2** 30 帧；运行规程为**一次会话只采一条路**
（远距/跨节点传送会让 east_coast 会话崩溃，沿路 1 m 步进可上百帧；gridmap 稳定）。冻结
`frozen_holdout_gridmap.json`（digest `89a604352743414b`），并给冻结工具加 `--min-frames` 守卫
（本轮一次失败采集只产出 1 帧却被工具冻结）。四模型 × 三场景结果：跨图召回靠加数据从 0.322 拉回 **0.454**
但精度掉到 0.197（IoU 0.160 < 仅 italy 的 0.192）→ **跨图召回缺口仍是结构性**；gridmap 的 0/0 是
**标签密度 0.017%** 导致的数据问题而非模型失败；**本图**`wide2` 仍保持 precision 0.747 / IoU 0.727。
**范围判断**：跨图为自加研究项，**不再阻塞接线决策**（接线看本图 T07/T08/T09 全链）。

**地图先验 A/B + 方向约定修复（本轮追加，详见 `docs/MAP_ASSOCIATION.md` §5）**：实车跑 A/B 时发现
"近乎直线的链路上两个候选都判 179°/154° 硬冲突"——根因是**候选方向没有朝向规范**
（`atan2(world[-1]-world[0])` 在提取器不保证近→远顺序时是符号硬币）。已用**自车朝向**规范
（不碰地图/横向几何）+ 2 条单测；重跑后 wide2 变为**无冲突 + 26° 软**、lines2 变为 26°/41°/42° 软
+ 一个 53° 硬（`rank changed=True`，仅降权、无新增接受）。地图探针同时补 `--model` 以与身份工作同检查点。

**方向先验单因子对照 + 假接受分布（本轮追加，详见 `docs/MAP_ASSOCIATION.md` §6）**：多帧摆位（12 帧 × 2 m）
并逐帧抓引擎标注，使每个候选都能被独立确认。结果：**弧切线被实测否掉**（该地图 `inRadius/outRadius`
是 3.50 m，直路上"切线"从 4° 扫到 166°；候选方向偏差中位数 弦 17.87° vs 切线 ~80°），
故**弦保持默认**、切线降为显式开关；切线臂各项都更差（放行 41 vs 19、假接受 90.3% vs 75.5%）。
弦臂的假接受分布：49 候选里 **未被拒且未被漆线确认 37（75.5%）**、**假拒绝 0** → 方向判据是**很松的放行门**，
只能降权不能当接受门。

**未做**：GUI 未接（工具文件驱动），因此**"分钟/有效标签"本轮没有数字**，不得引用提速倍数；
真实连续片段上的完整标注与质量统计未做；旧 `logs/` 采集没有新字段，训练入口对其如实打印回退提示
（新字段只对之后的采集生效）。

## 6m. T08 余项：链路折线切线 + 方向×漆线联合接受门（本轮追加，详见 `docs/MAP_ASSOCIATION.md` §7）

上一条登记的两件余项都做完了，并且**修掉了一个方法论错误**（§6 那次"弦 vs 切线"是两次独立进程、
候选集不同，32 vs 43，不能当配对对照）：

- **数据源换成地图折线**：不再用 `inRadius/outRadius`（实测 3.50 m，不是道路弧半径），
  改用 `RoadNetwork.nearby_polylines`（路由用的同一张图）在 ±10 m **节点弧长**窗口内的局部弦、
  方向朝远离车一侧；窗口只含单节点时回退到"最近节点的相邻节点"（回归：直道 0.00°、弯道 16.70°、无图 None）。
- **配对三臂**（同进程、同一帧同一候选集）：`chord` / `tangent_arc` / `tangent_roadnet`，参考来源由
  `direction_reference_rad()` 统一裁决——**只有真的用了切线才标切线**；每臂各自报方向偏差中位数、
  漆线确认、假接受率与联合门计数。
- **联合接受门**（方向判据 ∧ `on_line_frac ≥ 0.5`）与"仅漆线门"并列成单因子表，
  丢因分 `dropped_paint_confirmed` / `dropped_road_only` / `dropped_off_road`（四桶之和 = 判数）。

**实车三次重复（`wide2`，`ft_t12`，12 帧 × 2 m，逐帧引擎标注）**：候选 29/31/27；方向偏差中位数
弦 15.87/22.78/18.26°、折线 **13.91/23.04/14.78°**、弧切线 89.11/90.95/77.70°。结论：

1. **折线切线可用但无实质收益**：三次里弦与折线的**接受判定完全相同**（hard 0、soft 逐帧一致、假接受率同值）
   → 直路段上"改用局部切线"是**零增量**，价值只能在弯道/长链路上体现（未测）。
2. **弧切线再次被否**：中位 78–91°、`<45°` 只占 16–30%，**7/10 个漆线确认候选被它硬拒**，
   它那份更低的假接受率是**用 70% 真阳性换的**。
3. **联合门 = 仅漆线门**：三次都恰好通过 10 个候选、丢掉 0 个漆线确认候选，与"仅漆线门"逐个相同
   （方向否掉 0 个）→ 可信参考下**方向判据对接受门零贡献**；只在被否的弧参考下两门才分歧（10 → 3）。
4. **运行间波动必须随数字引用**：同摆位同检查点同路径，候选数 29/31/27、逐帧中位数最大差 ~20°，
   故绝对值只当量级、比较必须同进程配对。

**路口复测：第一次判出"折线切线有害"，也是 T08 验收指标的第一个实数（本轮追加，详见 `docs/MAP_ASSOCIATION.md` §8.3）**：
取 T07 路口行同一处（`(722.0, 760.3)`，航向 235.6°，12 帧 × 2 m，逐帧引擎标注）：

| 臂 | 候选 | 中位 \|Δ\| | `<45°` | 落漆线 | 硬冲突 | 硬冲突落在哪 | 方向反向 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **弦** | 55 | **25.88°** | **0.855** | 3 | 8 | **8/8 全在路外** | 13/55 = 0.236 |
| 弧切线 | 55 | 48.00° | 0.491 | 3 | 25 | 1 个落漆线（假拒绝） | 21/55 |
| 折线切线 | 55 | 49.43° | 0.418 | 3 | 32 | **3/3 落漆线候选全被硬拒** | 0/55 |

1. **折线切线的反例**：它取"离自车最近的折线"，在路口会锁到**另一条路**上，把本车自己车道的漆线判成方向不符
   （3/3 漆线确认候选被硬拒）→ **不能当方向参考**；要用必须先解决折线身份问题（未做）。
   至此三处结论：直路/转角"无害也无用"、**路口"有害"** → **弦保持唯一参考，弧与折线都不采纳**。
2. **T08 验收指标"减少错误关联"的实测值 = 0 处净减少**：弦臂 8 个硬拒绝**全在路外候选**上，
   而路外候选本来就被道路存在性判据拒 → 地图先验的拒绝力在这处**与既有判据重复**。
   往好处说它也**从未增加放行**（只减不加、假拒绝 0）。→ T08 结论由"未测"变为"**测了，无增益**"，仍不接线。
3. **轴语义在这里第一次真起作用，但只是降噪**：弦臂 23.6% 的候选方向是反的（轴一致），
   按旧有向判会全变硬冲突；逐候选核对这 13 个里 **0 个落漆线** → 改轴**没救回任何真阳性**，去掉的是 13 条伪硬冲突。

**弯道复测 + 方向判据改为轴比对（本轮追加，详见 `docs/MAP_ASSOCIATION.md` §8）**：

- **判据改轴**：标线没有箭头，按"有向"判实际在测"地图节点顺序 + 自车朝向"。现按轴比对（mod 180），
  反向作为并列诊断（`sense_delta_deg` + `direction_sense:flipped|aligned`）**不罚分**；
  探针每臂统计 `direction_sense.flipped/measured`。**撤回**上一轮前"五点摸底"得到的
  "地图节点顺序不可信"（那次把车朝向 0° 传过去，比的是两个朝向约定不同的量）。
  真实证据是三次沿路行驶运行：弦 0/33、0/48、0/59 反向 → **车头朝路时该改动是恒等变换**。
  wide2 第 4 次重复（轴判据）：33 候选 / 落漆线 **10** / 弦中位 **18.87°**（前三次 15.87–22.78 区间内）
  / 联合门 10/0/3/20 / 假拒绝 0 → **无回归**。
- **弯道**：取 96°/99° 两个转角，从转角前 15 m 沿路起步各 12 帧 × 2 m。折线仍只比弦好
  0.2–0.6°（27.04 vs 26.68、14.93 vs 14.30）、判定基本不变；弧切线**第三次被否**（中位 39–57°，
  且方向反向占 5–20%，弦臂 0%）。**§6 的"长链路上弦偏 30–45°"在本图不可复现**——本图每个转角
  **换链路**（`DR343_207→208` 过转角变 `208→209`），弦本身已是局部段；折线价值需要的路网条件
  （链路长/节点疏）本图未找到，登记为未测。
- **弯道的假接受不可用**：两转角处引擎把 **39% 画面**判为"线"（斑块状公交站/路口区，`road` 类为 0），
  落漆线候选 0；其中一处地图规则把自车挂到 `wp_busStop_09_a_2` 公交站航点链路。→ 只用于方向对照。

## 8. 提交切分与逐提交验证（本轮收尾）

工作树按 `scripts/check_commit_scope.py` 的模块定义（`beamng_autopilot/<子包或顶层模块>` / `scripts` / `tests` /
`docs` / 顶层文件，测试跟随它验证的模块）重提为 **21 个提交**，`--last 21` → `OK - 每个提交只动一个模块`。

**验证方法**：把每个提交单独 checkout 到自己的 worktree（junction `weights/`、`logs/`、`data/`、`.yolo/`
使离线测试可用），在其中运行 `pytest -o addopts='' -q`。结果（通过数随模块递增）：

| # | 提交主题 | 通过 |
| --- | --- | --- |
| 1 | geometry（T05） | 1580 |
| 2 | obstacle_risk | 1588 |
| 3 | occupancy | 1588 |
| 4 | planning（T09） | 1638 |
| 5 | safety_monitor（T09） | 1703 |
| 6 | telemetry_contract | 1713 |
| 7 | eval（T03） | 1733 |
| 8 | lane（T06/T07/T08） | 1876 |
| 9 | vision（T10） | 1910 |
| 10 | control（T04） | 1910 |
| 11 | rl | 1915 |
| 12 | beamng_autopilot_tech | 1932 |
| 13 | fsd_stack | 1951 |
| 14 | fsd_drive（T02/T04/T06） | 2016 |
| 15 | run_manifest | 2016 |
| 16 | runtime | 2016 |
| 17 | traffic | 2016 |
| 18 | labeling（T10） | 2035 |
| 19 | scripts | 2272 |
| 20 | docs | 2282 |
| 21 | README | 2282 |

**验证环境造成的 3 项排除**（都在主工作树 HEAD 上通过，且排除项本身有据）：

- `test_benchmark_manifest::test_manifest_records_the_commit_and_dirty_state`：读的是 worktree 的 git 状态；
- `test_fsd_closed_loop::test_fsd_closed_loop_recovers_from_a_body_crossing`：在**改动前基线 21c32ab** 的
  worktree 上同样失败（已实测），属 worktree 环境而非本轮改动；
- `test_phase0_contracts::test_seg_eval_ignores_255_in_metrics`：需要只存在于主工作树 `logs/`（gitignore）里的
  训练检查点，worktree 里没有该文件。

**该过程抓到的真问题（不是文档问题）**：

1. **分层倒置**：`planning/hold_audit.py` 曾从 `lane.lateral_risk` 取停止余量模型——规划模块依赖 lane 包，
   在中间提交上表现为真实 `ImportError`。修法：模型移到 `obstacle_risk`（它已持有 `RISK_*` 与
   `stop_distance_m`），lane 侧改为委托，并新增回归测试"planning 不得导入 lane"（`tests/test_planning.py`）。
2. **测试与模块错配**：5 个测试文件验证的是驾驶回路/工具（`fsd_drive`、`m5_run_metrics`、
   `m5_seg_stage_eval`、测试夹具 `test_fsd_drive_pipeline`），却与更早的模块同提交。修法：较早提交里提交
   **去掉这些测试的中间版本**（它们失效正是该提交自己引入的 API 变更所致），在落地对应模块的提交里提交完整版本。
3. **oracle 脚本归属**：几何/障碍的测试需要后来才提交的 oracle 脚本（`m5_geometry_audit.py` 等），
   故测试改随其验证的脚本提交。

复现脚本（未纳入版本库，属工作区临时件）：`.workbuddy-ai/tmp_commit_split.sh`、`.workbuddy-ai/tmp_verify_commits.sh`、
`.workbuddy-ai/make_test_variants.py`、结果 `.workbuddy-ai/commit_verify7.txt`。

## 7. 回归与自查

- `pytest tests/ -o addopts='' -q` → **2222 passed**（本轮新增 200 例：reference 18、terminal 12、
  measurement 9、tool 9、seg 2、dataset_split 1、line_evidence 9、geometry 22、camera_ring 3、
  nearfield 2、body_coverage 2、shadow_state 18、boundary_evidence 14、map_association 27、
  lateral_risk 29、path_hold +3、frontend_ablation 12，其余为受影响断言的更新）。
- `scripts/m5_offline_validate.py` → **ALL PASS**。
- 所有改动只新增字段/接口与收紧约束；未 `reset`/`clean`，未动 `logs/`、`weights/` 既有产物。
- §9 禁止事项自查：未用「导航线 + 固定偏移」做横向控制（T02 让 Scene 只消费**已接受的感知参考**，
  BEV 全路面中心仍被拒）；未把缺测当通过（T01 反而把撞车那次的两项越界降为 UNKNOWN）；
  未同时改相机、后处理、planner、安全阈值与学习模型后比较里程（本轮没有闭环里程结论）。

## 7. 未完成（阶段 A 余项，按计划顺序）

| 项 | 状态 |
| --- | --- |
| T02/T04 的 **Tech 闭环复核** | **已完成短程**（见 §6）；仍需更长对照与独立安全真值 |
| T03 源身份 / 局部证据 / 预测身份 | **本轮完成主链**（见 §6b）：源 ID/序号/采集时刻、投票键改为真实观测、事件计数、按年龄的 added 切分、黄色先验单列、候选支持摘要与分带年龄；**余项**：真实曝光时刻、`pose_id/ground_model_id`、注入/地图假设贡献分列、异步迟到与证据身份未打通 |
| T05 平地几何基准与近场可观测性 | **本轮完成**（见 §6c 与 `docs/GEOMETRY_BASELINE.md`）：唯一基准 + 姿态进主链 + 独立 oracle（p50 0.0000 m）+ 分辨率预算 + 坡道/地平线不可用带；**余项**：地面平面取值的配对 A/B、`pose_id/ground_model_id` 进快照契约、历史证据跨地面重投影的高度信息、鱼眼畸变标定 |
| T10 余项（真实 map/episode/time ID、跨视角同曝光归组、冻结测试集） | **本轮完成前两项**（见 §6l 与 `docs/ANNOTATION_SCHEMA.md`）：采集侧记录 `map_name`/`source_id`/`t_wall`/跨视角共享 `exposure`，索引入口 `frame_refs_from_meta()` 缺什么报什么，`cross_view_leak()` 三值口径（不可检查≠无泄漏）；**冻结留出集未产生**，GUI/切片标注量与金帧复制检测未做 |
| 阶段 2–6（四臂重跑、场景矩阵、影子能力） | T06（§6d）、T07 第一版（§6e）、T08（§6f）、T09（§6g）、T11 第一版（§6h）已落地；**阶段 C 第一项已完成**（§6i，每臂 8 次交错，结论=不采纳）；阶段 C 余项（更长的对照路线、独立真值的压线/出铺装）已完成可做部分；**T07/T10 余项本轮补齐可做部分**（§6k、§6l）；**T08 余项已补齐**（§6m：折线切线 + 三臂配对 + 方向×漆线联合门；余项=折线切线在弯道/长链路上的收益未测）；阶段 D 未开始 |

**阶段 A 退出门对照**（计划 §6）：

- 「所有已知窄反例被回归固定」→ 本轮新增 51 例反例/契约测试，覆盖 T01/T02/T04/T10/T12 的已核缺陷；
- 「正例仍可工作」→ 每处修复都配了正例对照（真双侧才有 full 权限、全权限不被夹死、测到的越界仍然 FAIL、
  目标达成时性能门禁 exit 0、组 holdout 时 `leak=False`）；
- 「最终横向缺测不获 PASS」→ 撞车实录的两项越界检查已从「0 越界」变为 UNKNOWN；
- 「安全约束覆盖实际主/子步命令」→ 夹具层断言实际 `send` 参数 ≤ 该 tick 公布的权限（含子步与终点分支），
  实车短跑 3 次 0 越权；
- 「结果可追溯到工作树、模型与原始输入」→ **部分达标**：实车跑产物路径已登记（`stageA_live{1,2,3}.json`），
  但改动仍未提交，且没有更长的闭环对照。
