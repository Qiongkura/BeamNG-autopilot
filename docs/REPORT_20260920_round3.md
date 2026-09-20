# 2026-09-20 第三轮汇报

按评审第六节格式（原文存于 `docs/REPORT_FORMAT.md`，出处
`BeamNG_下一阶段执行计划_20260920.md:217-225`）。

每项写：原假设 → 是否真正激活 → 可复现输入 → 实测结果 → 支持/推翻的结论 →
未知与下一步。八项必附见文末。

---

## 0. 范围与基线

| 项 | 值 |
|---|---|
| 分支 / HEAD | `main` @ `dd3cf1e` |
| 领先 `origin/main` | 17 个提交（`68e0285..dd3cf1e`） |
| 工作树 | 干净（3 项为备份目录，未跟踪） |
| 分析语料 | `logs/fsd_benchmark/town_*.json`，**131 个跑次 / 26202 帧** |
| 本轮新增实车跑次 | **0**（唯一一次重跑游戏进程崩溃，随后叫停） |
| 本轮产出 run manifest | **0 个**（`logs/fsd_benchmark/*.manifest.json` 不存在） |
| 单测 | 收集 1585，**1584 通过 / 1 失败** |

**关于"本轮实车零验证"** —— 这不是措辞保守。本轮唯一一次实车尝试中
`BeamNG.tech.x64.exe`(pid 25852) 已死，控制器 CPU 归 0、socket 停在
`SYN_SENT`；之后未再重启游戏。所以凡是需要"新跑次"才能得到的数字，本报告
一律标 **未测**，不回填旧语料冒充。

---

## 1. lane：SIDE gate 在 strict 下必须常开

* **原假设**：`strict_lane` 同时关掉了两件事 —— map 先验当参考（这是 strict
  的本意："感知领航，map 不得导航"），和 SIDE gate 这道安全网（不是）。关掉
  后者，未配对的感知中心（双向路 = 整路走廊中心 = 路中线）就直达规划器。
* **是否真正激活**：是。commit `175a8b0`，`left_max_m = -0.2 if strict_lane
  else ...`，SIDE gate 与 `strict_lane` 解耦，永远运行；strict 下阈值收紧到
  -0.2，使 `off=0`（中心正好压在路中线）也被拒。
* **可复现输入**：`town_1789890448.json`（`--strict --runtime tech
  --scenarios town --goal 868.3 744.9`，166 帧，本语料中最后一条）。
* **实测结果**：
  - 该跑次 `line_lat` mean **+1.181 m**、停车率 **74%**（75/166 帧）—— 骑在
    分隔线右侧，与假设一致。
  - 全语料 `lane_reject=side` 仅 **42 / 26202 帧**；`lane_src` 分布：
    `perception-unavailable` 15006（57.3%）、`sensor` 7745（29.6%）、
    `map_lane` 2649（10.1%）、`bev/route` 600、`corridor` 202。
  - **关键反证**：在离线闭环 fixture 上逐版本对比，修复前与修复后
    `lateral_reference` 的 ref median y **都是 0.060**，
    `best_path` median y **都是 0.060**。几何一字未变。
* **支持/推翻**：**部分推翻**。gate 确实拦住了 `lane_src`（sensor →
  perception-unavailable），但下游 `lateral_reference()` 退回
  `lane_envelope.center`，把同一条 y=0.060 的中线原封不动送回规划器。
  **该修复只改了一个标签，车走的是同一条几何。**
* **未知与下一步**：感知层为何输出 0.060（而非配对算出的 -1.77）未修，见 §15。

---

## 2. planning：envelope 的 center 不得把 gate 的拒件偷运回来

* **原假设**：`lateral_reference()` 中 envelope 的 fallback 排在 strict 检查
  **之前**，导致 gate 刚拒绝的车道换个标签回来当转向参考。
* **是否真正激活**：是。commit `218b3cd`，strict 检查提到 envelope 之前；
  envelope 仍供硬边界，其 center 不再是转向参考。
* **可复现输入**：`tests/test_fsd_closed_loop.py` 的 `_LaneWorldSemantic`
  fixture —— markings 在 road lat **0**（黄）与 **-3.5**（白），车道中心应为
  **-1.75**。
* **实测结果**：修复后 `lateral_reference` 返回 `src = none`，不再退回
  envelope。全量单测 1584 通过 / 1 失败。
* **支持/推翻**：**支持**（绕路已堵死，安全方向正确）。
* **未知与下一步**：代价是 strict 下感知不可信 → 无路径 → fail-closed 停车。
  这正是 `test_fsd_closed_loop_recovers_from_a_body_crossing` 现在失败的原因
  —— 车停住，爬不出跨线。**可恢复性目前为零**。

---

## 3. run_manifest：别把自家 launcher 当成对手

* **原假设**：Windows 下 `.venv\Scripts\python.exe` 是 shim，把真解释器作为
  **子进程** spawn，shim 携带相同命令行成为本进程父进程。只排除
  `os.getpid()` 会把自家 launcher 报成"第二个控制器"。
* **是否真正激活**：是。`own_process_lineage()` 排除祖先；从同一 shell 起的
  兄弟进程仍上报（那确实抢车）。
* **可复现输入**：实车尝试 —— pid 36168 是本进程父进程，被误报；单测
  `tests/test_run_manifest.py`。
* **实测结果**：单测通过。实车层面 `logs/fsd_benchmark/*.manifest.json` =
  **0 个**，即 manifest 从未在实车上产出过。
* **支持/推翻**：**支持（离线）**；**实车未验证**。
* **未知与下一步**：跑一次实车产出 manifest，才能证明独占检查在真环境下
  不误报也不漏报。

---

## 4. telemetry_contract：每个 head 每帧端到端追溯

* **原假设**：每 head 每帧一条记录，统一单调时钟（source_t / eligible_t /
  dispatch_t / finish_t / publish_t），`source_seq` / `result_seq` 使"控制侧
  消费了过期版本"可被检出。
* **是否真正激活**：代码在（commit `af8f1a6`），单测在。
* **可复现输入**：全语料 131 跑次。
* **实测结果**：帧级 `source_seq` / `result_seq` 覆盖 **0 / 131 跑次**。
  该契约是本轮新写，旧语料从未记录过。
* **支持/推翻**：**无法判定 —— 语料零覆盖**。
* **未知与下一步**：新跑一次并确认 `result_seq` 单调、与 `source_seq` 的差值
  有界。在此之前"消费了过期版本"这类故障形状不可测。

---

## 5. fsd_stack：保活下限（range_schedule）

* **原假设**：保活触发条件是"超预算 **且** 超过保活下限"的合取，而 8 臂 A/B
  从未同时产生两半，所以"它没触发过"永远无法变成"它不可能触发"。
  把分支顺序抽成纯函数即可测。
* **是否真正激活**：是。`range_schedule()` 从 tick 体内原样抽出（commit
  `898f9d2`）。
* **可复现输入**：`tests/test_schedule_floor.py`；语料侧 `head_sched`。
* **实测结果**：`head_sched` 覆盖 **25 / 131 跑次**、4203 帧、12609 条 head
  记录。状态分布：`ran` 10454、`not_due` 1995、`budget_deferred` 156、
  **`keepalive_forced` 4**（0.03%）。
* **支持/推翻**：**部分支持**。函数可测了（单测通过），但语料里强制刷新只
  触发 4 次 —— "能不能触发"有了弱证据，"触发后是否改善"仍无数据。
* **未知与下一步**：需要一次 `STALE_RANGE_S` 附近的长跑，统计强制刷新前后
  age 的变化。

---

## 6. fsd_drive：final_target_speed 硬约束被抬高

* **原假设**：旧的 `target_sm = min(_lt.reference, plan_speed)` 没有与
  `target` 取 min，导致更低的 target 被重新抬回去。
* **是否真正激活**：是（commit `49ffec6`）。
* **可复现输入**：`tests/test_final_target.py`。
* **实测结果**：单测通过。语料侧无 `final_target_speed` 列，**无法回溯验证**。
* **支持/推翻**：**支持（单测层）**；语料层未测。
* **未知与下一步**：新跑次需记录该列，并统计"抬升事件"次数。

---

## 7. planning：走廊可行性三态 + ClearanceGuard

* **原假设**：走廊可行性应返回三态（连续 / 行间连通 / 几何可进入 / 与可行驶
  层相交），UNKNOWN 不得打开逃生口；ClearanceGuard 取窗口最小、跳变锁存、
  网格缺失时 FREEZE 而非归零。
* **是否真正激活**：是，默认关闭（`BEAMNG_CORRIDOR_FEASIBILITY`）。
* **可复现输入**：`tests/test_corridor_feasibility.py`、
  `tests/test_clearance_guard.py`。
* **实测结果**：单测通过。语料侧 `corridor_open` 仅 **35 / 131 跑次**
  （5502 帧）；`clear_guard` / `clear_src` 仅 **16 / 131**（2725 帧）；
  `fwd_clear_guarded` **16 / 131**（1198 帧）。
* **支持/推翻**：**支持（单测层）**；语料覆盖 12–27%，不足以做跨跑次结论。
* **未知与下一步**：`corridor_free_band` 仍只返回 bool，不暴露自由带的**位置
  与宽度**，所以"还能不能切进去"今天无法回答 —— 这是"能绕开"和"绕不开"之间
  唯一的分界线。

---

## 8. planner：接触包络速度上限（**故意未接线**）

* **原假设**：`contact_envelope_speed_mps` 可作为逃生口的限速器。
* **是否真正激活**：**否** —— 数学已实现并有单测，但**未接到逃生口**。
* **可复现输入**：`tests/test_obstacle_risk.py`；两个逃生口场景。
* **实测结果**：碰撞帧 3.30 → 2.72 m/s 确实生效，但两个逃生口场景在
  3.750 m、碰撞帧在 3.789 m，**相差 4 cm**。接上去会静默回退一个已测的修复。
* **支持/推翻**：**推翻**（在该判据下）。
* **未知与下一步**：真正区分"能绕开"的是自车速度 / 接近率或横向逃逸可行性，
  不是 `closest`。需要带速度维度的场景集。

---

## 9. safety_monitor：仲裁链可观测 + road_checked 不再伪装健康

* **原假设**：`road_checked` 的初版自己犯了它要修的错 —— 三次跑次里 4 个
  "unknown" 帧全是 dataclass 默认值（检查压根没跑），与"读过且无证据"逐字节
  相同，于是 `road_lost_s=0.0` 读作路面健康。
* **是否真正激活**：是（commit `8738e7f`），加滞回
  `ROAD_RECOVER_CONFIRM_S`，网格缺失时 FREEZE。
* **可复现输入**：语料 `road_surface` / `road_lost_s` 列。
* **实测结果**：`road_surface` 覆盖 **3 / 131 跑次**、**528 帧**；取值
  `on_road` 524、`unknown` 4；`road_lost_s > 0` 的帧 **0**。（`unknown` 4 帧的
  成因已定位：无 drivable path ×3、path hold ×1。）
* **支持/推翻**：**支持**（默认值伪装已消除）。
* **未知与下一步**：见 §附5 的灵敏度上限 —— 它是"band 缺失"检测器，不是横向
  安全网，不得当作后者使用。

---

## 10. damage：观测碰撞而不是推断碰撞

* **原假设**：此前 131 个 town 跑次里只有 25 个有碰撞证据，其余分析无法区分
  "开得干净"和"我们没看"。
* **是否真正激活**：是（commit `b5dbc91`），damage.py + connector + tech
  provider 接线。
* **可复现输入**：语料 `damage_total`。
* **实测结果**：`damage_total` 覆盖 **25 / 131 跑次**、4203 帧；其中
  **14 个跑次 damage_total > 0**；`collision_count` 合计 **28 次**。
* **支持/推翻**：**支持**（碰撞现在可观测）。
* **未知与下一步**：`closest_obs_m = 999.0` 是"无障碍"哨兵值，留进统计会变成
  中位数 —— 任何统计脚本必须先剔除。

---

## 11. eval：三态评分，缺列不臆造结论

* **原假设**：评分需 pass / fail / unknown，无法判定的跑次不得算作通过。
* **是否真正激活**：是（commit `14a1f2d`）。
* **可复现输入**：全 131 跑次跑 `assess_run` + `score_run`。
* **实测结果**：**PASS 0 / FAIL 131 / UNKNOWN 0**（跑次级）。但**门级**
  UNKNOWN：`no_collision` 在 **106 / 131** 跑次无法测量（无 damage 通道）。
  失败门：`no_stall` 131、`no_centre_crossing` 78、`no_reversing` 74、
  `on_road` 59、`no_edge_crossing` 56、`no_collision` 14。
* **支持/推翻**：**强烈支持** —— 旧口径下这 106 个跑次会因为没有碰撞证据而
  **静默通过**，占语料 81%。
* **未知与下一步**：`route_dist` 仅 35/131 有，同样规则适用：缺失 = unknown，
  不是 no_route_config。

---

## 12. benchmark：写 manifest + 污染端口 fail-closed

* **原假设**：两个控制器共用一个端口会用两次 teleport 驱动同一辆车，结果不
  属于任何一次实验；旧的 mtime 比较只在两次跑到秒级重叠时才抓得到。
* **是否真正激活**：是（commit `b978bef`），默认 fail-closed，退出码 3，除非
  传 `--allow-contaminated`。`_Print_row` / `--score` 同步改三态。
* **可复现输入**：`tests/test_benchmark_manifest.py`；实车未跑。
* **实测结果**：单测通过；**产出 manifest 0 个**。
* **支持/推翻**：**支持（离线）**；实车未验证。
* **未知与下一步**：实车跑一次，确认 manifest 落盘且独占检查不误报。

---

## 13. 六个诊断脚本的结论

### 13.1 m5_shadow_replay

* **原假设**：逃生口从未被"咨询"，它只是在附和。
* **是否真正激活**：是。
* **可复现输入**：`town_1789886413`。
* **实测结果**：`corridor_open` 在 **160/160 帧**全为 True，从无 False，
  包括 `closest = 1.206 m` 那一帧。
* **支持/推翻**：**支持** —— 逃生口不是被否决，是从来没有反对过。
* **未知与下一步**：`corridor_open` 只 35/131 覆盖，需在更宽语料上复核。

### 13.2 m5_perf_profile

* **原假设**：瓶颈在 object head，把 object 改异步能救。
* **是否真正激活**：是。
* **可复现输入**：131 跑次 tick_ms / frame_ms。
* **实测结果**：跨 25 个有 `head_compute_ms` 的跑次取中位数 —— 语义
  **301.0 ms** / object **7.5 ms** / traffic **1.5 ms**。**把 object 改异步
  救不了瓶颈**（7.5 ms 只占 tick total p50 177.9 ms 的 4%）。详见 §附6。
* ⚠️ 早期草稿引用过 "257.9 ms / 7.3 ms"，那是**单个跑次**的数字。评审明确
  要求：单个跑次是例子，不是可引用的数。已改为语料中位数。
* **支持/推翻**：**推翻**。
* **未知与下一步**：ring 是最大项，需拆到子阶段。

### 13.3 m5_boundary_coverage

* **原假设**：边界感知不足可用下游阈值调。
* **是否真正激活**：是。
* **实测结果**：`lat_left` 覆盖 **3427/26202 = 13.1%**、`lat_right`
  **3262/26202 = 12.4%**。
* **支持/推翻**：**推翻** —— 这是传感器覆盖问题，下游阈值调不动。
* **未知与下一步**：先提覆盖，再谈阈值。

### 13.4 m5_stall_attribution

* **原假设**：no_stall 的主要失败是速度下限。
* **实测结果**：`no_executable_path` **32.7%**（最长 118.7 s） vs
  `unjustified` **0.9%**（最长 6.3 s）。详见 §附7。
* **支持/推翻**：**推翻** —— 速度下限是错的仪器。

### 13.5 m5_scenario_set

* **原假设**：n=4 的阻断臂 A/B 能分辨 12 m 差异。
* **实测结果**：组内里程差异 3.0×（30.9–92.9 m），pooled sd 23.7 m，d=0.51 →
  约需 **61 臂/条件**。
* **支持/推翻**：**推翻**。

### 13.6 m5_sched_metrics

* 多臂调度对比 + mean 行，随 §5 使用。

---

## 14. docs / AGENTS.md：把评审格式固化进仓库

* **原假设**：该格式只存在于外部 Codex 输出目录，仓库内仅一句压缩版，
  目录清了规范就丢了。
* **是否真正激活**：是（commit `ebc9a14` + `dd3cf1e`）。
* **实测结果**：`docs/REPORT_FORMAT.md` 存原文；`AGENTS.md` 新增"汇报格式
  （评审硬性要求，每轮必填）"，与提交纪律同级。
* **支持/推翻**：**支持**。

---

## 15. 本轮最重要的未修项（感知层）

* **原假设**：参考层 gate 能挡住骑中线。
* **实测**：fixture markings 在 lat 0 / -3.5 → 车道中心应 -1.75；配对算出的
  车辆坐标中心 **-0.58**（世界 ≈ **-1.77**）是对的；但
  `lane_envelope.center` 输出到世界坐标是 **0.060**。
* **结论**：**感知层自己产出的车道中心就落在分隔线上**。参考层只能拒，不能
  让它变对。
* **下一步**：查 `lane_envelope.center` 的构造（世界坐标转换 / 走廊中心定义）。

---

# 八项必附

## 附1. commit / config / run 清单及无控制权冲突证据

**commit 清单**（`origin/main..main`，17 个）：

```
175a8b0 lane: keep the SIDE gate on under strict perception
adcd59e run_manifest: P0 run manifest, and stop calling our own launcher a rival
af8f1a6 telemetry_contract: end-to-end traceability for every head, every frame
898f9d2 fsd_stack: per-head scheduling telemetry, and a keepalive that is testable
49ffec6 fsd_drive: consumption side, and a hard limit that was being raised
1ce3e9b planning: corridor feasibility as three states, and a clearance guard
50490d7 planner: contact envelope speed cap, measured before it is wired
8738e7f safety_monitor: observable arbitration, and no default that reads as healthy
b5dbc91 damage: observe collisions instead of inferring them
14a1f2d eval: three-state scoring, and never invent a verdict from a missing column
b978bef scripts: benchmark writes a manifest, and refuses a contaminated port
4f381da scripts: six diagnostic tools, each of which moved a conclusion
3bac45b docs: evidence index, round reports, and the strict centre-line review
218b3cd planning: the envelope centre must not smuggle the lane gate's rejects back in
251a624 docs: the gate was cosmetic until the envelope fallback was moved
ebc9a14 AGENTS.md: write down the report format the review requires
dd3cf1e docs: keep the full report-format requirement, not a one-line summary
```

**config**：`lane_mode=sensor`（119/131 跑次首帧）、`map`（12/131）；
`kind`：`lane_center` 31、`arc` 3、`hold_heading` 1（今日 35 跑次）。
本轮新增配置项 `BEAMNG_CORRIDOR_FEASIBILITY`（默认关闭）、
`ROAD_RECOVER_CONFIRM_S`。

**run 清单**：本轮新增跑次 **0**；分析所用 131 跑次为历史语料
（`town_1788602082` … `town_1789890448`），其中今日（Sep 20）35 个。

**无控制权冲突证据**：

| 检查 | 结果 |
|---|---|
| BeamNG 进程 | **无**（`tasklist` 无匹配） |
| 端口 64257 | **无占用**（`netstat` 无匹配） |
| 同端口第二控制器 | **无**（无实车运行） |
| manifest 独占检查实车输出 | **无**（0 个 manifest） |

⚠️ 这条证据的**强度很弱**：它证明的是"现在没有冲突"，不是"运行时不会冲突"。
真证据需要一次带 manifest 的实车跑次。

## 附2. 测量覆盖率、缺列情况、UNKNOWN 数量

全语料 131 跑次 / 26202 帧。覆盖分四档：

| 档 | 列 | 覆盖 |
|---|---|---|
| 全量 | `reason` `lane_src` `lane_reject` `road_off` `speed` `tick_ms` `frame_ms` `t` `mon_target` 等 ~55 列 | 131/131 跑次，26202 帧 |
| 高 | `best_bear` 79.5%、`fwd_clear` 67.0%、`signal_conf` 62.5%、`closest_obs` 58.6% | 全 131 跑次但帧级有缺 |
| 中 | `rear_clear` 43.0%（119 跑次）、`lane_bear` 38.0%（129）、`plc_desired` 39.0%（127） | — |
| 低 | `corridor_open` `route_dist` `closest_obs_m` `min_ttc` 等 | **35/131**，5502 帧 |
| 很低 | `range_sched` `head_sched` `damage_total` | **25/131**，4203 帧 |
| 极低 | `clear_guard` `clear_src` 16/131；`fwd_clear_guarded` 16/131（1198 帧）；`body_road_off` 5/131（1036 帧）；`road_surface` `road_lost_s` **3/131（528 帧）** | — |
| 孤例 | `mode_why` `drive_mode` `long_plan` `steer_blend` | **1/131** |

**UNKNOWN 数量**：

* 跑次级 verdict UNKNOWN：**0**（全部 131 为 FAIL）
* **门级 UNKNOWN：`no_collision` 在 106/131 跑次未测量**（无 damage 通道）
* `road_surface` 帧级 unknown：**4 / 528**（0.76%），成因：无 drivable path
  ×3、path hold ×1
* `source_seq` / `result_seq` 覆盖率：**0 / 131** → 该维度全 UNKNOWN

## 附3. 强制刷新触发与实际新源结果消费证据

| 指标 | 值 |
|---|---|
| `head_sched` 覆盖 | 25/131 跑次、4203 帧、**12609 条 head 记录** |
| `ran` | 10454 |
| `not_due` | 1995 |
| `budget_deferred` | 156 |
| **`keepalive_forced`** | **4（0.03%）** |
| 帧级 `source_seq` / `result_seq` | **0 跑次 —— 无数据** |

**结论：强制刷新的"触发"有弱证据（4 次），"触发后是否真的消费了新源结果"
证据为零。** 端到端追溯契约（`telemetry_contract`）在本语料里从未被记录过，
因为它是本轮新写的。此项**未测**，须新跑次补。

## 附4. 几何正反例与最终控制链证据

**反例（几何错误链，已定位）**：

```
fixture markings:  road lat 0 (黄) / -3.5 (白)   → 车道中心应 = -1.75
配对中心(车辆坐标)  -0.58                        → 世界 ≈ -1.77   ✓
lane_envelope.center(世界)  0.060                → 路中线          ✗
best_path median y          0.060                → 路中线          ✗
```

**正例（gate 确实生效）**：

* `lane_src=perception-unavailable` **15006 帧（57.3%）** —— 不可信感知被拦下
* `lane_reject=side` 42 帧、`heading` 1185、`corner` 866
* 修复后 `lateral_reference` 返回 `src = none`，不再退回 envelope

**最终控制链证据（末端）**：

| 跑次 | 帧 | `line_lat` mean | 停车率 |
|---|---|---|---|
| `town_1789890448`（--strict） | 166 | **+1.181** | **74%** |
| `town_1789890286` | 180 | +1.323 | 61% |
| `town_1789890111` | 182 | +0.638 | 56% |

全语料 `line_lat`：n=24969，mean **+0.091**、p50 **-0.088**、min -7.688、
max 8.116；**|lat| < 0.3 m 的帧 3546（14.2%）**。
按模式拆：`map`（1795 帧）mean +0.857；`sensor`（23174 帧）mean +0.031。
今日 35 跑次 `line_lat` mean **+0.475**（明显差于全语料）。

⚠️ **注意**：修复前后 ref / best_path 的 median y **都是 0.060** —— 控制链
末端几何未改变，只改了来源标签。这是 §1 判为"部分推翻"的直接依据。

## 附5. 路面门真阳性、漏检与误停

| 指标 | 值 |
|---|---|
| 覆盖 | **3/131 跑次、528 帧** |
| `road_surface` 取值 | `on_road` 524、`unknown` 4 |
| `road_lost_s > 0` 的帧 | **0** |
| **真阳性**（判定为路面丢失且确实丢失） | **0** |
| **误停**（门触发导致停车） | **0** |
| **漏检** | **≥1，已知** |

**已知漏检**：`road_off = 0.596 m` 持续 **146 帧**，全程 `road_surface` 仍为
`on_road`。另有 `road_off` 0.765(19)、0.721(18)、1.486(17)、2.011(7)、
2.56(6) 等帧。`body_road_off` 采样 1036 帧，其中 **105 帧 > 0（10.1%）**。

**灵敏度上限 ≈ 1.45 m**：车在铺装外 0.60 m 待了 146 帧，门全程无反应。
它是"band 缺失"检测器，**不是**"我们压在线上"的检测器，不得当横向安全网用。

## 附6. 性能分解与 deadline 违反

跨 131 跑次（取各跑次分位数的中位数）：

| 阶段 | p50 | p95 | max |
|---|---|---|---|
| ring | **78.6 ms** | 156.2 ms | 26248.5 ms |
| range | 10.1 ms | 229.1 ms | 3210.3 ms |
| plan | 34.8 ms | 61.9 ms | 3998.9 ms |
| **total** | **177.9 ms** | **409.8 ms** | 26301.6 ms |

**deadline 违反（>150 ms）**：

* **17311 / 26202 帧 = 66.1%**
* **零违反的跑次：0 / 131**（没有一跑次达标）

**实际控制间隔**：p50 **0.535 s** → **1.87 Hz**（标称 15 Hz）；p95 0.848 s；
max 0.928 s。

**按 head 分解**（跨 25 个有该列的跑次，取 p50 的中位数）：

| head | p50 |
|---|---|
| semantic | **301.0 ms** |
| object | 7.5 ms |
| traffic | 1.5 ms |

语义是唯一的大项；object **7.5 ms** 只占 tick total p50（177.9 ms）的
**4%** —— **把 object 改异步救不了瓶颈**。这个结论与早期草稿引用的单跑次
数字（257.9 / 7.3）方向一致，但量级必须以语料中位数为准。

## 附7. 碰撞、压线、出铺装与按原因分类的停车时长

**碰撞**：

| 指标 | 值 |
|---|---|
| `collision_count` 可测跑次 | 25 / 131 |
| 碰撞总数 | **28** |
| `damage_total > 0` 的跑次 | **14** |
| `no_collision` 违反 | 14 跑次 |
| `no_collision` 未测量（UNKNOWN） | **106 跑次** |

**压线 / 出铺装**（门违反计数）：

| 门 | 违反跑次 |
|---|---|
| `no_centre_crossing`（压中线） | **78 / 131** |
| `no_edge_crossing`（压边线） | **56 / 131** |
| `on_road`（出铺装） | **59 / 131** |
| `no_reversing` | **74 / 131** |
| `no_stall` | **131 / 131** |

**停车时长按原因分类**（全语料，停止帧 14809 / 26202 = 56.5%）：

| 类别 | 次数 | 停止帧 | 时长 | 占比 | 最长单次 |
|---|---|---|---|---|---|
| obstacle_or_boundary | 516 | 5419 | **3475.4 s** | 36.6% | 117.6 s |
| no_executable_path | 227 | 4847 | **2615.4 s** | 32.7% | **118.7 s** |
| unknown | 343 | 3846 | 1864.6 s | 26.0% | 90.5 s |
| near_goal | 4 | 557 | 296.2 s | 3.8% | 98.5 s |
| **unjustified** | 33 | 140 | **79.2 s** | **0.9%** | 6.3 s |

**关键读法**：真正"没理由停"的只有 0.9%、最长 6.3 s；而
`no_executable_path` 占 32.7%、最长 118.7 s。**速度下限是针对 0.9% 那一项的
仪器，用在 32.7% 上是错的。**

## 附8. 通过项、失败项、尚未测试项

### 通过项

| 项 | 证据 |
|---|---|
| 单测 | **1584 / 1585 通过**（1 失败） |
| SIDE gate 与 strict 解耦 | 4 个回归测试 |
| envelope 绕路堵死 | `lateral_reference` 返回 `src=none` |
| `own_process_lineage` 消除误报 | `tests/test_run_manifest.py` |
| eval 三态评分不再把缺列算通过 | 106 个跑次现标 UNKNOWN 而非 PASS |
| damage 碰撞可观测 | 28 次碰撞被记录（此前推断） |
| 评审格式固化 | `AGENTS.md` + `docs/REPORT_FORMAT.md` |
| `contact_envelope_speed_mps` 数学正确 | 单测；碰撞帧 3.30 → 2.72 m/s |

### 失败项

| 项 | 失败形态 |
|---|---|
| **所有 131 跑次** | verdict = **FAIL**（PASS 0） |
| `no_stall` | 131/131 违反；停止帧占 56.5% |
| `no_centre_crossing` | 78/131 |
| `no_reversing` | 74/131 |
| `on_road` | 59/131 |
| `no_edge_crossing` | 56/131 |
| `no_collision` | 14/131 违反（+106 未测量） |
| `test_fsd_closed_loop_recovers_from_a_body_crossing` | strict 下 fail-closed 无路径，车停住，爬不出跨线 |
| 性能 deadline | 66.1% 帧超 150 ms；**0/131 跑次达标**；实际 1.87 Hz vs 标称 15 Hz |
| 边界覆盖 | `lat_left` 13.1% / `lat_right` 12.4% |
| `corridor_open` 恒真 | 160/160 帧全 True，包括 `closest=1.206 m` |

### 尚未测试项（**明确列出，不含糊**）

| 项 | 为何未测 | 要拿到需做什么 |
|---|---|---|
| **本轮全部实车结论** | 游戏进程崩溃后叫停，**新增跑次 0** | 重启游戏跑一次 |
| run manifest 实车输出 | 0 个 manifest | 带 manifest 的实车跑次 |
| 端口独占检查真环境行为 | 无实车 | 同上（+ 人为起第二控制器做反例） |
| `source_seq` / `result_seq` 消费证据 | 语料 **0 覆盖**，契约本轮新写 | 新跑次记录并校验单调性 |
| 强制刷新后的效果 | 仅 4 次触发，无前后对比 | `STALE_RANGE_S` 附近长跑 |
| `final_target_speed` 抬升事件 | 语料无该列 | 新跑次记录该列 |
| `road_surface` 门真阳性 | 覆盖仅 3/131、528 帧，真阳性 0 | 专门构造路面丢失场景 |
| 走廊可行性三态实车行为 | 默认关闭 + 覆盖 12–27% | 打开开关跑并采样 |
| `contact_envelope_speed_mps` 接线效果 | **故意未接线**（会回退已测修复） | 先找到区分"能绕开"的正确判据 |
| n=4 A/B 的分辨力 | 已知需要约 61 臂/条件 | 按 `m5_scenario_set` 的 AB/BA 协议跑 |
| 感知层 `envelope.center` 偏移 | 未修 | 查世界坐标转换 / 走廊中心定义 |
