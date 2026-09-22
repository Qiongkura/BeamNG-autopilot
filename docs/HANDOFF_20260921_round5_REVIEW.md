# 交接执行单：Round5 审查后的修复与验证计划

> **历史交接单，后续行动建议已被取代（2026-09-22）。** 请从 [统一终版实施方案](FINAL_INTEGRATED_PLAN_20260922.md) 接手。当前工作树已实现本单中的部分能力，也发现了新的接线与验收问题；本文保留历史证据，不作为当前待办清单。现行 AGENTS.md 未因此改变。

日期：2026-09-21  
来源：`docs/REPORT_20260921_round5.md` 及其遥测、代码、测试和运行产物审查

> 给下一位开发者：本文件回答三个问题——**哪里有问题、应该改哪里、先做什么以及什么结果才算完成**。  
> 本文件是执行计划，不代表 Round5 的问题已经解决。

---

## 0. 总结结论

**项目当前不能宣称已修复。** 车辆仍有停车、压线和出铺装风险；曾发生真实碰撞。

当前已经确认的事实：

- `models2_nodqn` 发生过碰撞，`damage_total` 从 0 增至 **86.2445**。
- 旧的“0 压线 / 0 出铺装”结论存在测量盲区；缺少边界时的 0 不能作为通过。
- `front_main` 最近可见地面约 **3.55 m**，`front_fisheye` 约 **2.22 m**；近场可行驶证据不足是结构性问题。
- 游戏 annotation 对车道标线提供的监督几乎为 0；标线监督主要依赖人工标注。
- `constrain_line_to_road` 已改为连通域判定，但其闭环驾驶收益尚未证明。
- DQN、BC、E2E、横向残差 RL 均没有通过严格、同条件、同输入的城市场景收益验证。
- Round5 记录的 `pytest` 和离线验证通过，只能证明离线回归通过，不能证明实车目标完成。

当前不能直接写成已证实的结论：

- “后处理修复已经解决实车识别问题”；
- “金帧微调模型整体优于 hand 模型”；
- “标线只标一次即可跨视角迁移”；
- “参考翻转是撞车的唯一直接原因”；
- “修复后 `damage=0` 就代表安全”；
- “DQN 已被证明是负价值”；
- “横向 RL 在当前城市场景可用”。

---

## 1. 接手前必须知道的代码状态

- 分支：`fix/round3-hardening-20260921`
- HEAD：`21c32ab`
- Round5 改动：**未提交**。
- 工作树有大量未提交/未跟踪改动。接手时不得 `reset --hard`、`clean` 或覆盖无关修改。
- 实车必须使用 BeamNG.tech；纯逻辑改动至少运行 pytest 和 `m5_offline_validate.py`。
- 实车运行前确认没有第二个控车进程，且 Tech 图形质量不是 `Lowest`。

已有但不能重复“假设已解决”的实现：

| 已有实现 | 代码位置 | 当前判断 |
|---|---|---|
| 连通域式标线约束 | `beamng_autopilot/vision/segmentation.py::constrain_line_to_road` | 离线方向合理，闭环收益未证实 |
| 中线分隔线脱困 | `beamng_autopilot/lane/reference.py::own_lane_beside_divider` | 已接线，仍需实车验证是否稳定脱困 |
| 接受车道的 drivable 门槛 | `beamng_autopilot/lane/reference.py::_drivable_fraction` | 能防止驶向路肩，但可能把漏检变成停车 |
| 越转越偏夹紧 | `beamng_autopilot/fsd_drive.py::clamp_steer_away_from_road` | 有单测，不能替代完整事故复现 |
| 障碍风险分级 | `beamng_autopilot/obstacle_risk.py` | 需统计误停，不要继续盲目加规则 |
| 结构化 corridor feasibility | `beamng_autopilot/planning/corridor_feasibility.py` | 几何原语，尚未标定为驾驶安全门 |
| 多视角 annotation 采集 | `beamng_autopilot_tech/providers.py`、`scripts/m5_collect_seg_ring.py` | 已证明能采集 road，未证明能提升驾驶 |

---

## 2. P0：先统一证据口径，不要先继续调模型

### P0-1 统一“停车”的定义

**问题**：报告中的 `57/75、62/78、53/64` 等数字无法由当前遥测字段唯一复现。一个跑次同时可能得到：`speed < 0.3`、`speed <= 0.1`、`final_stop=True`、`emergency=1`、`reason=no drivable path` 等不同计数。

**需要检查/修改的位置**：

- `beamng_autopilot/eval.py`
- `scripts/m5_fsd_benchmark.py`
- `scripts/m5_coverage_index.py`
- `scripts/m5_run_metrics.py`
- 报告生成脚本或后续 Round6 报告

**要求**：统一并明确：

1. `settle_s` 规则；
2. 采样帧的时间戳和帧间隔；
3. 停车阈值；
4. 硬停车和低速蠕行是否分开。

每个跑次至少输出：

- `raw_frames`；
- `settled_frames`；
- `speed_lt_0_3_frames`；
- `final_stop_frames`；
- `emergency_frames`；
- `stall_seconds`；
- `longest_continuous_stall_seconds`；
- 按 `reason/effective_rule` 分类的累计秒数。

**通过标准**：报告任意一个停车数字，都能由一条命令和一个 JSON 直接重算；缺列时输出 `UNKNOWN`，不能输出 0。

### P0-2 统一横向字段契约

**问题**：`ego_lat_route_m`、`lat_route_m`、`line_lat` 的参考对象、坐标系和正负方向没有统一说明；报告中“实线 +0.91 m”和探针中的 `-0.368 m` 也没有解释是否属于不同帧/不同坐标系。

**需要检查/修改的位置**：

- `scripts/m5_lateral_scene_probe.py`
- `beamng_autopilot/lane/reference.py`
- `beamng_autopilot/lane/pairing.py`
- `beamng_autopilot/fsd_drive.py`
- `beamng_autopilot/eval.py`
- `scripts/m5_lane_metrics.py`

**要求**：给每个横向字段写清：

- 参考对象：车辆、路线、标线或车道中心；
- 坐标系：车体、路线或标线局部坐标；
- 正负方向；
- 单位；
- 覆盖条件；
- `None`、`UNKNOWN` 和 `0` 的含义。

建议字段说明固定为：

```text
字段 = reference + frame + sign + unit + coverage
```

**通过标准**：只看一帧遥测和字段契约，就能判断车辆相对标线在哪一侧。

### P0-3 统一分割评估阶段

**问题**：报告把 raw、形态学、约束层和完整 `Segmenter` 输出的 IoU 混在一起。特别要解释报告的 `0.358/0.372` 与现有 `eval_base.json/eval_v1.json` 中完整 pipeline 指标的差异。

**需要检查/修改的位置**：

- `beamng_autopilot/vision/segmentation.py`
- `scripts/m5_eval_seg.py`
- `tests/test_seg_road_constraint.py`
- `logs/goal_20260921/eval_base.json`
- `logs/goal_20260921/eval_v1.json`

**要求**：分开报告：

1. network raw；
2. morphology 后；
3. `constrain_line_to_road` 后；
4. shape filtering 后；
5. 完整 `Segmenter.predict()` 后。

每一项都标注：数据集、帧数、类别映射、忽略值、global IoU 或 mean-frame IoU、生成命令和产物路径。

**通过标准**：每个数字都能由明确的中间产物复现；不能把“后处理扫描改善”写成“完整模型已经改善”。

---

## 3. P1：按优先级修改/验证的模块

### P1-1 近场可行驶证据：最高行为优先级

**现象**：主前视相机 0–3 m 基本没有 BEV 可行驶格；strict 模式进入 `no_drivable_path` 或 `perception lane unavailable`。

**改动/接线位置**：

- `beamng_autopilot/vision/ring.py`
- `beamng_autopilot_tech/providers.py`
- `beamng_autopilot/runtime.py`
- `beamng_autopilot/fsd_stack.py`
- `beamng_autopilot/fsd_drive.py`
- `beamng_autopilot/vision/projection.py`
- `beamng_autopilot/occupancy.py`

**建议**：

1. 先接入 `front_fisheye`，只改相机输入，不同时改 planner、安全阈值、DQN 和模型结构。
2. 先做固定 drivable-band 上界实验，确认“有近场证据时 planner/safety 是否仍停车”。
3. 再做主前视、前鱼眼、双相机融合三组输入对照。
4. 只有确认覆盖改善后，才做鱼眼视角微调。
5. road annotation 可以自动采集；line 仍必须人工留出验证。

**必须记录**：

- 车头前 0–4 m drivable evidence coverage；
- `no_drivable_path` 帧数和秒数；
- `lane_src`、`lane_paired`、`lane_from`；
- `body_cross_current`、`painted_body_cross`；
- damage；
- 停车秒数；
- 是否到达 goal。

### P1-2 车道参考跨帧稳定

**现象**：同一路段在 paired、本车道、对向车道、整路融合、单侧镜像之间切换；侧向门控拒绝错误参考后可能形成停车自锁。

**改动位置**：

- `beamng_autopilot/lane/reference.py`
- `beamng_autopilot/lane/pairing.py`
- `beamng_autopilot/lane/fusion.py`
- `beamng_autopilot/planning/hysteresis.py`
- `beamng_autopilot/fsd_drive.py`

**实现方向**：

1. 保存上一帧已验证参考的侧别、来源和中心横向带。
2. 侧别与中心带连续 2–3 tick 一致后，才恢复正常转向权限。
3. 未 paired 或单侧镜像参考只能小幅修正，禁止满锁。
4. 车身已越界时，只允许向可信车道收敛的缓行。
5. 旧参考不能在没有新鲜感知证据时升级为新合法参考。
6. 任何稳定器都不能违反 `AGENTS.md`：不得使用导航线加固定偏移作为横向控制。

**新增/修改测试位置**：

- `tests/test_lane_reference.py`
- `tests/test_lane_pairing.py`
- `tests/test_fsd_drive_pipeline.py`
- 必要时新增参考稳定器专用测试。

**验收指标**：参考侧别翻转次数、中心跳变 P50/P95/max、`lane_paired`、`pair_paired`、`lane_from`、`steer_away_clamped`、越界和 damage。

### P1-3 无边界时的出铺装保护

**问题**：车道边界不可用时，旧的 `road_off=0` 可能只是“没有测量”；车辆已经在铺装面外，却没有进入控制层保护。

**改动位置**：

- `beamng_autopilot/fsd_drive.py::_perception_off_road_m`
- `beamng_autopilot/safety_monitor.py`
- `beamng_autopilot/occupancy.py`
- `beamng_autopilot/vision/projection.py`

**实现方向**：使用感知 drivable mask 对车身 footprint 做覆盖率评估。连续 N 个新鲜 tick 低于阈值时，进入最小风险停车。

必须区分：

1. 有边界且明确越界；
2. 无边界但车身 drivable 覆盖不足；
3. 没有有效观测；
4. 观测到路面但有障碍。

`UNKNOWN` 不能当成 `on_road`，也不能用地图边界或导航线补齐横向事实。

先写确定性单测，再用 Tech 做正反例：车在铺装内、车在路肩、无观测、边界丢失。

### P1-4 标线后处理闭环验证

当前代码已经修改 `constrain_line_to_road`，下一步不是继续调阈值，而是做四臂 A/B：

1. 无约束；
2. 像素式约束；
3. 连通域 `0.5`；
4. 两级 `0.5/0.25`。

固定同一批 RGB、同一 checkpoint、同一数据集。

**离线指标**：raw/full line IoU、假线像素、连通域数量、配对率、正确车道中心率、参考翻转次数。

**闭环指标**：`lane_paired`、`lane_from`、`lane_drivable`、`line_lat`、`no_drivable_path`、停车秒数、越界、damage。

只有离线和闭环都改善，才能把“后处理是实车直接原因”升级为已证实结论。

### P1-5 路面门槛的误停问题

**问题**：`lane/reference.py::_drivable_fraction` 当前要求中心采样点至少 60% 位于 observed & drivable。它能阻止驶向路肩，但模型中景截断时可能把“模型漏检”读成“非铺装”，从而撤回真实车道。

**不要做**：不要直接把 60% 阈值调低来换取里程。

**应该做**：

- 记录 `not enough observed samples`、`centre off observed pavement`、障碍占用、真实非铺装等不同原因；
- 用前鱼眼实验区分“没有路”与“没有看到路”；
- 任何未知状态仍然 fail-closed；
- 报告撤回车道导致的停车秒数。

### P1-6 性能与调度

**重点位置**：

- `beamng_autopilot_tech/providers.py`
- `beamng_autopilot/vision/ring.py`
- `beamng_autopilot/fsd_stack.py`
- `beamng_autopilot/fsd_drive.py`
- `scripts/m5_perf_profile.py`
- `scripts/m5_sched_metrics.py`

历史遥测显示 tick p95 明显高于目标，ring 可能占主要耗时；预算调度可能饿死 range/object，继而触发 stale 和停车。

先跑现有分段遥测，不要预先指定优化对象。必须报告：

- ring、semantic、segmentation、range、object、plan 的 p50/p95/p99；
- budget skip 率；
- range/object age；
- command gap；
- watchdog；
- 实际被控制逻辑消费的结果年龄；
- stale 的具体触发模态。

目标是回答：**停车是感知本身不可用，还是感知太慢导致被调度器饿死。**

### P1-7 DQN/E2E/RL 暂缓重训

当前只完成“接入/被拒/出现动作”的实测，不足以判定收益。

先完成感知和参考稳定性验证，再做严格同会话 A/B：

- DQN on/off；
- 其他输入完全一致；
- 至少多次交错运行；
- 同时看速度、停车秒数、越界、damage、到达目标，而不是只看里程。

E2E 当前候选被拒，应先解决域/视角/标定契约，不要用放宽校验强行放行。

---

## 4. 下一位必须按顺序执行的工作流

### 阶段 0：只修统计和字段契约

不改变驾驶行为。完成：

- 停车统计统一；
- 横向字段统一；
- 分割评估阶段统一；
- 缺测统一输出 `UNKNOWN`。

### 阶段 1：固定条件四臂 A/B

每臂至少 5 次，交错运行，建议 ABBA 或 Latin square。固定：地图、车辆、起点、goal、代码版本、checkpoint、速度、安全阈值、所有学习开关、连接方式。

四个臂：

- **A**：当前 `front_main`；
- **B**：只接入 `front_fisheye`；
- **C**：A + 参考跨帧稳定门；
- **D**：B + 参考跨帧稳定门。

若鱼眼尚不能提供可信 line，先使用固定 drivable-band 做上界实验，不能把上界注入当成最终方案。

### 阶段 2：后处理四臂对照

固定 RGB 和 checkpoint，只切换后处理判据；完成离线指标和短实车闭环指标。

### 阶段 3：跨视角标线验证

每个视角人工标注至少 20–40 帧，按视角独立留出。报告 line IoU 之外的：近场召回、配对率、车道中心误差、错误接受率和跨帧稳定性。

### 阶段 4：性能分解和调度修复

确认 ring、range/object、stale 的因果关系后再决定缓存、异步、采样频率或模型推理优化。

### 阶段 5：重新评估学习模块

在感知输入和控制口径稳定后，重新做 DQN、E2E、横向 RL 的严格 A/B。

### 阶段 6：corridor/escape hatch 标定

结构化 `corridor_feasibility` 目前只是几何原语。必须通过 A–H 场景集和成对场景实验后，才能作为控制放行门。

---

## 5. 下一轮实验的统一记录要求

每个 run 必须保存：

- commit / 工作树状态 / 配置 / 环境变量；
- 地图、车辆、起点、goal、运行时长、speed；
- 控车进程独占证据；
- 各字段覆盖率、缺列和 UNKNOWN 数量；
- head 刷新序号、source 序号和实际消费证据；
- `lane_src`、`lane_from`、`lane_paired`、`pair_paired`、参考翻转；
- `lane_drivable` 各原因；
- `road_surface`、`road_checked`、`road_lost_s`；
- `body_cross_current`、`painted_body_cross`、`edge_over`；
- damage；
- 停车按原因的帧数、累计秒数、最长连续时长；
- ring/head/range/object/plan 性能分解；
- 是否到达 goal。

**任何字段没有覆盖时，报告 `UNKNOWN`，不要写 0。**

---

## 6. 建议验收门槛

### 感知与参考

- 0–4 m 有可解释的 drivable evidence；
- 有效车道参考覆盖率单独报告；
- paired、single-side、divider fallback 分开统计；
- 参考侧别和中心变化可解释；
- `lane_drivable` 通过、撤回、样本不足分开统计。

### 安全

- `damage_total=0`；
- `body_cross_current=0`；
- `painted_body_cross=0`；
- 出铺装证据有覆盖，否则只能是 UNKNOWN；
- 感知不可用时不得继续加速；
- 不得用导航线 + 固定偏移替代感知横向边界。

### 行驶质量

- 停车同时报告帧数和秒数；
- `no_drivable_path`、障碍停车、path hold、感知不可用分别统计；
- 不能以大量 fail-closed 停车换取“零碰撞”；
- 不能用单次短跑 `damage=0` 验收。

### 性能

- tick p50/p95/p99；
- command gap p95/max；
- range/object age；
- budget skip 与 stale 的对应关系；
- 消费结果的实际年龄。

---

## 7. 常用复现命令

### 单帧横向探针

```powershell
.venv\Scripts\python.exe scripts\m5_lateral_scene_probe.py --runtime tech --attach `
  --teleport 779.7 735.6 -13 --goal 868.3 744.9 --ticks 2 `
  --json logs\goal_20260921\scene.json
```

### Tech 实车基线

```powershell
.venv\Scripts\python.exe scripts\m5_fsd_drive.py --runtime tech --attach `
  --seconds 45 --speed 5 `
  --teleport 779.7 735.6 -13 --goal 868.3 744.9 `
  --strict --lane-mode sensor `
  --seg-model logs\m5_seg\seg_model_hand\best_task.pt `
  --no-signal --no-shadow `
  --out logs\goal_20260921\run.json
```

### 多视角采集

```powershell
.venv\Scripts\python.exe scripts\m5_collect_seg_ring.py --runtime tech --attach `
  --frames 200 --roles front_main front_fisheye pillar_left pillar_right
```

### 分割评估

```powershell
.venv\Scripts\python.exe scripts\m5_eval_seg.py `
  --runs logs\m5_seg\manual_mountain_labeled `
         logs\m5_seg\manual_review_batch_labeled `
  --model logs\m5_seg\seg_model_lines_v1\best.pt --json out.json
```

### 离线回归

```powershell
.venv\Scripts\python.exe -m pytest tests\ -o addopts='' -q
.venv\Scripts\python.exe scripts\m5_offline_validate.py
```

---

## 8. 下一位开发者第一批任务清单

- [ ] 统一停车统计口径，并补齐可复现汇总命令。
- [ ] 统一 `line_lat`、`lat_route_m`、`ego_lat_route_m` 字段契约和符号说明。
- [ ] 拆分并复现 raw / constraint / full pipeline 分割指标。
- [ ] 不改其他变量，完成主前视 vs 前鱼眼对照。
- [ ] 完成参考稳定门开/关对照，记录侧别翻转和中心跳变。
- [ ] 完成后处理四臂离线 + 短实车闭环对照。
- [ ] 为无车道边界场景补充感知车身 drivable 覆盖保护的单测和 Tech 正反例。
- [ ] 跑性能分解，确认 ring 是否饿死 range/object。
- [ ] 统计障碍停车原因的累计秒数和最长连续时长。
- [ ] 以上完成前，不要优先重训 DQN/E2E/RL。
- [ ] 最后更新 Round6 报告，所有数字绑定命令和产物。

---

## 9. 明确禁止事项

- 不要 `git reset --hard`、`git clean` 或覆盖已有工作树改动。
- 不要使用“导航线/地图线 + 固定偏移”做横向控制或压线判断。
- 不要把 `road_off=0`、缺边界或缺列当作“未出铺装”。
- 不要把 UNKNOWN 当 PASS。
- 不要同时修改相机、后处理、planner、安全阈值和学习模型后比较里程。
- 不要把 4×4 多视角样例写成 200 帧能力已经验证。
- 不要在没有跨视角 line 留出集时宣称“标线只标一次即可迁移”。
- 不要用单次 `damage=0` 证明安全。

---

## 10. 完成定义

下一轮完成必须同时满足：

1. 停车、横向、分割指标均可独立复现；
2. 近场、参考稳定性、后处理三类假设都有固定条件 A/B 证据；
3. `damage=0` 同时有横向覆盖和出铺装覆盖证据；
4. 停车原因按秒数分类；
5. 性能瓶颈和 stale 来源已定位；
6. 未测项明确为 UNKNOWN；
7. 报告不再把“采集能力已验证”写成“驾驶收益已验证”。

**交接结论：先统一证据口径，再做分层 A/B；确认主瓶颈后再改行为，最后才重新评估学习模型。**
