# 证据索引（2026-09-20 冻结）

对应《BeamNG-autopilot 下一阶段执行计划（第二轮后修订）》的 **P0：冻结基线与实验契约**。

用途：本文件把每一条被引用的结论，绑到它**实际来自哪个文件、覆盖到哪些跑次、哪些跑次
根本没法支撑它**。报告里出现过的两处错误读数（「保活是空处理」、把 `range_age` 的界
写成 2.0 s）都是因为跳过了这一步。

复现：`.venv\Scripts\python.exe scripts\m5_coverage_index.py <run JSON...>`

---

## 1. 运行清单（P0-1）

| 项 | 值 |
| --- | --- |
| 记录点 | `docs/ROUND_REPORT_20260920_round2.md` 写就时的 HEAD `66f0c1e` |
| 分支 | `main`，本地领先 `origin/main`（未 push） |
| 工作树 | **不干净**：未跟踪目录 `.workbuddy-ai/`（工具目录，非仓库代码）。**保留，不清理、不重置历史** |
| 本次冻结后新增提交 | `run_manifest.py`（清单模块）、`scripts/m5_coverage_index.py`、`scripts/m5_fsd_benchmark.py` 接线 |
| 清单模块 | `beamng_autopilot/run_manifest.py`，已接线到 `scripts/m5_fsd_benchmark.py` |

**注意**：清单模块是**本次才接线的**，所以上面所有已记录的跑次（town / mountain / A/B）
**都没有 manifest**。它们的「同配置」仍然只能靠 `ab3_clean.txt` 这类手写记录支撑，
不是自动产出。**下次跑次起才有。**

### 1.1 开关解析值

清单记录**全部**声明过的 `BEAMNG_*`（17 个行为开关 + 9 个环境变量），未设的记 `None`
——**`None` ≠ `"0"`**：前者是「模块默认生效」，后者是「显式关闭」，两者的行为可能不同。
另会捕获任何未声明的 `BEAMNG_*`。

历史跑次的开关状态**没有记录**（只有 `ab3_clean.txt` 里手写的
「off 臂 / on 臂」），所以任何「九组含 async」这类口径冲突，**在现有证据里无法仲裁**
——这正是方案 P3 第 2 条要求解决的。

### 1.2 控制权独占性（P0-2）

判据：`run_manifest.exclusivity()` —— 问操作系统「有没有另一个像控制器的进程」，
匹配 `m5_fsd_drive` / `m5_fsd_benchmark` / `m5_drive_test` / `m5_e2e_test` / `fsd_drive`。
**不再用日志 mtime 比对**：那种判据只在两个跑次恰好重叠到同一秒时才成立（2026-09-20
就是这么漏掉 `town_1789889000/_9006/_9171/_9172` 的，四个跑次全废）。

基准入口现在是 **fail-closed**：检出第二个控制器时默认拒绝开跑（退出码 3），
要跑得显式加 `--allow-contaminated`，而冲突无论如何都会写进 manifest。

**现有跑次的独占性是「事后推断」**，不是当时确认的。按方案要求，
不能把日志同秒落盘当成并发污染的唯一证据。

---

## 2. 测量覆盖率（P0-4 / P0-5）

口径：`logs/fsd_benchmark/town_*.json` 共 **131 个跑次 / 26202 帧**。

| 列 | 覆盖 | 状态 |
| --- | --- | --- |
| `tick_ms` / `tick_wall_ms` / `frame_ms` / `budget_s` / `budget_skips` | 131/131 | 全跑次可用 |
| `fwd_clear` | 131/131 | 全跑次可用 |
| `mon_target` / `target_sm` / `reason` / `level` / `emergency` / `source` | 131/131 | 全跑次可用 |
| `road_off` | 131/131 | 全跑次可用 |
| `head_age_s` / `freshness` | 100/131 | 31 个跑次**无法支撑** |
| `corridor_open` / `closest_obs_m` / `path_occ_frac` | 35/131 | 96 个跑次**无法支撑** |
| `damage_total` | 25/131 | **106 个跑次对碰撞是 UNKNOWN** |
| `range_sched` / `range_worker` / `head_worker` / `head_sched` / `head_errors` / `errors` | 25/131 | 106 个跑次**无法支撑** |
| `fwd_clear_guarded` / `clear_guard` / `clear_src` | 16/131 | 115 个跑次**无法支撑** |
| `road_surface` / `road_lost_s` | 3/131 | 只有路面门那三轮 |
| **`road_checked`** | **0/131** | **一次都没被记录过** |

### 2.1 直接后果

1. **`road_checked` 从未落盘。** 它是本轮为「区分『读了但没证据』和『根本没读』」
   而加的（`d28d1cf` + `b8f44dd`），但**没有任何一个跑次带着它**——三轮路面门跑次
   （`town_1789890111/_286/_448`）有 `road_surface` 却**没有** `road_checked`，
   说明它们跑在 `b8f44dd` 之前。**所以「新遥测修好了那个假象」这件事本身尚未验证。**
2. **碰撞观测只覆盖 25/131 跑次。** 其余 106 个跑次的 `collision_count` 必须按
   **UNKNOWN** 处理，不能当 0 读。`score_run` 现在就是这样做的（`ef1704c`）。
3. **调度遥测只覆盖 25/131。** 任何跨全部历史的「饥饿率」统计都必须限定在这 25 个
   跑次（§2B-3 的 r = 0.895 就是限定在带 `range_sched` 的 22 个跑次上算的）。

---

## 3. 路面门（`BEAMNG_ROAD_SURFACE_GATE=1`）三轮读数

`town_1789890111` / `_1789890286` / `_1789890448`，共 **528 帧**：

| 读数 | 值 |
| --- | --- |
| `state == on_road` | 524 |
| `state == off_road` | **0** |
| `state == unknown` | 4 |
| `road_checked` 列缺失 | **528 / 528** |
| `damage_total` 通道 | 3 / 3 有 |
| 门开火 | **0** |

**4 个 `unknown` 帧的逐帧归因**（这是「unknown 不是『读了但没证据』」的证据）：

| 跑次 | idx | t (s) | `reason` | `road_lost_s` |
| --- | --- | --- | --- | --- |
| `town_1789890111` | 4 | 3.376 | `no drivable path` | 0.0 |
| `town_1789890286` | 99 | 67.062 | `no drivable path` | 0.0 |
| `town_1789890286` | 100 | 67.713 | `no drivable path` | 0.0 |
| `town_1789890448` | 135 | 99.826 | `path hold (creep)` | 0.0 |

四个全是**判据提前 return**（3× 无可行驶路径、1× 路径保持爬行），发布的是 dataclass
默认值：`state=unknown` + `road_lost_s=0.0`。**`road_lost_s=0.0` 读起来就是「路面正常」，
这正是方案 P0-5 禁止的「把默认值当健康」。**

**能得出的结论只有一条**：这 528 帧里没有一次门触发导致的停车。
**不能**说它「能在危险时正确停车」——三轮零触发不是验收（方案 P4 放行条件）。

**已知漏检**：第 2 轮车压出铺装面 **0.60 m 持续 60 帧**，`road_surface` 全程 `on_road`。
判据读的是**前方 2–12 m** 的 drivable 带，不是车身是否还在铺装面内；灵敏度上限 ≈ 1.45 m。
**它是「带丢了」检测器，不是「压线/出路沿」检测器。**

---

## 4. 保活 A/B（`BEAMNG_SCHED_KEEPALIVE`）8 轮干净跑次

跑次清单：`logs/fsd_benchmark/ab3_clean.txt`（含 4 个作废跑次的说明）。
**8 个跑次都没有 `road_surface`**（路面门还没落地），`damage_total` 8/8 有。

两个阈值分属两个模块，**不要混**：

* **保活界**：`fsd_stack.py` 的 `RANGE_KEEPALIVE_S = 1.0` / `OBJECT_KEEPALIVE_S = 2.0`。
* **新鲜度契约线**：`safety_monitor.py` 的 `STALE_RANGE_S = 2.0`。

| 臂 | 逐轮 `range_age` max (s) | `range_age ≥ 1.0 s` 帧 | defer 帧 |
| --- | --- | --- | --- |
| OFF | 1.337 / 0.605 / **2.232** / 1.335 | 5 / 0 / 11 / 3 = **19** | 19 |
| ON | 1.322 / 1.276 / 1.199 / 1.231 | 1 / 3 / 1 / 2 = **7** | 7 |

| 读数 | OFF | ON |
| --- | --- | --- |
| `range_forced_rate` | 0.0% | 0.0% |
| `keepalive_forced` 帧数 | **0** | **0** |
| defer 被 `_starved` 挡下 | **0** | **0** |
| `object_age` max | 1.70 s | 1.65 s（界 2.0 s，**没到**） |
| ON 臂每次 defer 的 age | — | **0.608–0.677 s** |
| OFF 臂最坏 defer | **1.552 s → 下一帧 2.232 s** | — |

### 结论（说准）

保活是**合取**干预：只有「(a) range 判定那一刻已超预算」**且**「(b) 被复用的扫描已过界」
同时成立才动手。8 轮里 **(b) 出现过 26 帧，(a) 在这些帧上一次都没成立**
（`_over` 在 `fsd_stack.py:984` 量的是 tick 起点到 range 判定点，ring 单独只有
253–420 ms，没过 450 ms 预算）→ 扫描照常执行、state 记 `scanned`、**没有任何一次 defer
被挡下**。两臂跑的确实是同一套逻辑，`travel` / `collision_count` / `off_road` 的差都是噪声。

**但输入不是无害的**：OFF 臂 `town_1789888682` 有一次 defer 在 age **1.552 s**，
把下一帧推到 **2.232 s**，越过 `STALE_RANGE_S = 2.0` —— **正是这个下限要防的形状**。

所以：**这组实验证明处理「没有开火」，不能证明它「不会开火」。** 要判它，必须构造
让 (a)、(b) 同时成立的**可控超预算测试**（假时钟 + 可控耗时），不是再加 town 重复次数。

---

## 5. 评分口径（P0-3）

`beamng_autopilot/eval.py`（`ef1704c`）：

* 三态：`PASS` / `FAIL` / `UNKNOWN`。缺测的检查进 `unknown` 列表，**不放行**，
  也**不伪装成 FAIL**（不伪造为「已发生碰撞」）。
* `no_collision` 读 `collision_count`；读不到 → `UNKNOWN`。
* `on_road` 读 `off_road_frames`；无源 → `None` → `UNKNOWN`。
* 越界/停车同时报**时长、暴露时间与最大幅度**（`off_road_s` / `off_road_frac` /
  `off_road_longest_s` / `off_road_episodes` / `off_road_max_m`；`stall_s` /
  `stall_events` / `stall_longest_s`），不只统计帧数。
* `collision_count` 是 0/1 量纲（不是计数）；事件计数用 `collision_episodes`
  （间隔 > `merge_s` = 1.0 s 才算新事件）。

基准入口的 `--score` 汇总现在打印 `N PASS / N FAIL / N UNKNOWN`，
**UNKNOWN 不释放门禁**。

---

## 6. 尚未测试 / 无法回答（必须保留为未解决项）

| 问题 | 状态 | 为什么答不了 |
| --- | --- | --- |
| 保活是否有效？ | **未测** | 强制分支从未执行（§4）；需可控超预算测试 |
| 路面门能否在危险时正确停车？ | **未测** | 三轮零触发；需注入序列 + 独立真值（P4） |
| 横向 0.6 m 级贴边为什么漏检？ | 已定位为**结构限制** | 判据读前方带，不是车身越界传感器（§3） |
| 环/相机 I/O 的耗时各占多少？ | **未拆** | 只有 `tick_ms.ring` 一个总数（P3） |
| A4 异步开关在实车上做了什么？ | **未测** | 从未在实车上量过；且 `ring.grab_ring()` 仍同步（P3） |
| 「九组含 async」的口径冲突 | **无法仲裁** | 历史跑次没记录开关解析值（§1.1） |
| 逃生舱该用什么量区分绕得过去/绕不过去？ | **待决策** | `closest` 在两组场景只差 4 cm（§3 第 5 项） |
| 现有跑次的控制权独占性 | **事后推断** | 当时没问操作系统（§1.2） |
| `no_stall` 为什么两臂全 FAIL？ | **未分类** | 需按停止时长分类（P6） |

---

## 7. 复现命令

```pwsh
# 覆盖度索引（131 个 town 跑次）
.venv\Scripts\python.exe scripts\m5_coverage_index.py

# 只算路面门三轮
.venv\Scripts\python.exe scripts\m5_coverage_index.py `
  logs\fsd_benchmark\town_1789890111.json `
  logs\fsd_benchmark\town_1789890286.json `
  logs\fsd_benchmark\town_1789890448.json

# 落一份 JSON 供别的脚本消费
.venv\Scripts\python.exe scripts\m5_coverage_index.py `
  --json logs\fsd_benchmark\coverage_index.json

# 跑次前：manifest 与独占检查（接在基准入口里，默认 fail-closed）
.venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --attach --runtime tech `
  --scenarios town --strict --goal 868.3 744.9
```

`logs/` 是运行产物（gitignore），本文件引用的原始跑次都在
`logs/fsd_benchmark/` 下。


---

## 追加：第三轮新增证据（P1–P6）

复现命令（全部只读、不需要游戏，Windows 下用 .venv\Scripts\python.exe）：

```console
# 性能关键路径（全量 town）
python scripts/m5_perf_profile.py logs/fsd_benchmark/town_*.json --json logs/_prof.json

# 边界/近场覆盖（全量 town）
python scripts/m5_boundary_coverage.py logs/fsd_benchmark/town_*.json --json logs/_cov.json

# stall 归因（全量 town）
python scripts/m5_stall_attribution.py logs/fsd_benchmark/town_*.json --json logs/_stall.json

# 逃生舱影子回放（单跑次，标量级）
python scripts/m5_shadow_replay.py logs/fsd_benchmark/town_1789886413.json

# 闭环场景集与放行门
python scripts/m5_scenario_set.py
python scripts/m5_scenario_set.py --pairs 8 --seed 1
```

### 结论 → 证据 → 覆盖 → 可复现性

| 结论 | 来源 | 覆盖 | UNKNOWN / 缺口 |
| --- | --- | --- | --- |
| 保活会开火（超预算 **且** 过界） | `tests/test_schedule_floor.py` | 确定性，无需跑次 | 实车上从未自然触发 |
| `corridor_open` 在 160/160 帧为 True | `m5_shadow_replay.py` @ `town_1789886413` | 1 跑次 / 160 帧 | 无栅格，无法重建 P2.1 输入 |
| tick p95 目标 150 ms：0/131 达标 | `m5_perf_profile.py` 全量 | 131 跑次 | ring 内部无法再分（外部模块） |
| semantic 301 ms vs object 7.5 ms | 同上 | **仅 25 跑次**带 `head_sched` | 106 跑次无 head 计时 |
| `road_checked` 从未落盘 | `m5_boundary_coverage.py` 全量 | **0 / 131 跑次**有任一 checked 帧 | 该修复本身仍未验证 |
| 边界覆盖 3.0% / 2.2% | 同上 | 131 跑次 | 12/131 达到 50% 门槛 |
| 无谓 stall 只占停止帧 0.9% | `m5_stall_attribution.py` 全量 | 131 跑次 | 26% 因 `route_dist` 缺列归 unknown（96/131 无此列） |
| no_executable_path 占 32.7%、最长 118.7 s | 同上 | 131 跑次 | 规划器为何给不出路径未诊断 |
| 硬约束曾被后续链抬高 | `tests/test_final_target.py` | 确定性纯函数 | 实车影响量未测（开关未开） |

### 本轮仍未解决（不列为已验证）

1. P2 横向模型未标定（`min_turn_radius=5.5` / `max_lateral_speed=2.0` 是运动学占位值）。
2. ring 内部不可插桩 → 无法定位采集 / 预处理 / 后处理。
3. 传感器覆盖是硬门槛（边界 2.2–3.0%），换传感器配置前近场横向验收不可能真过。
4. 本轮**零实车跑次**；1571 个用例全是离线 / 单元测试。
5. 决策项 5 仍挂着（逃生舱用什么量区分绕得过 / 绕不过）。
