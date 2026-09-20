# 第三阶段报告：P1–P6（2026-09-20）

上一轮停在 P0 完成、P1 只有词汇表。这一轮把评审列的 P1–P6 全部做完，
**默认开关一个没动**，所有行为变更都在显式开关后面。

验证：`pytest tests/` 零失败；`m5_offline_validate.py` ALL PASS；
每个提交 `check_commit_scope.py --last 1` OK。

---

## 1. P1 端到端遥测与确定性故障测试

### 1.1 保活有没有用——现在有答案了

`range_schedule` 从 tick 主体里抽成纯函数（分支与顺序不变），
`_keepalive_expired` / `_budget_defers` 支持注入保活界（模块开关在
import 时就固化了，不注入没法测）。

**结论：保活会开火**，条件是超预算 **且** 过界同时成立 —— 输出
`("scan", "keepalive_forced")`。8 轮 A/B 一次都没产生过这个格子。
另外证明了 forced 之后 age 归零（确实采用了新源结果），以及关掉保活时
同一序列会无限 defer、age 只增不减 —— 四行代码复现了 151/151 的饥饿形状。

### 1.2 生产侧与消费侧

`fsd_stack` 每个 head 每帧产出 `source_t / eligible_t / dispatch_t /
finish_t / publish_t`（单一单调墙钟）与 `source_seq / result_seq`。
`fsd_drive` 记 `cmd_seq / cmd_t` 与每帧消费的 `result_seq / source_seq`
和**命令发出时刻**的 age。

两处判断值得记：`error` 记 `finish_t`（抛异常是**结束**了，留空会被读成
还在跑）；`async_idle_no_output` 是 `not_dispatched` 而非 `not_produced`
（源帧存在，是调度决定不提交）。

### 1.3 仲裁链可观测性

`_evaluate_core` 命中第一条规则就 return，所以 `reason` 只说一条规则。
现在每条 verdict 带 `effective_rule / rules_evaluated / rules_unevaluated /
masked_hard_rules`。**「scattered obstacle」会掩盖后面的硬停规则** ——
它是减速，却排在路面规则和车身越界规则（两者都是停车）之前。

### 1.4 road_checked 假象

grid 缺失时 reader 根本没跑，但旧代码仍把 `road_checked` 写成 True。
现在返回三元组，缺失时 False。

---

## 2. P2 修逃生舱

### 2.1 结构化可行性原语

`planning/corridor_feasibility.py` 返回 FEASIBLE / INFEASIBLE / UNKNOWN：

* **连续性** —— 极大自由区间，不是空格子计数（左 3 格 + 右 3 格 ≠ 6 格通道）
* **行间连通** —— 相邻行区间必须重叠；左右交替的空隙是两条不同的带
* **可进入性** —— `max_lateral_shift_m` 取自行车模型几何上限（s²/2R）与
  横向速度上限的较小者。**静止时几何上限起作用**：基于时间的模型会返回
  无穷大，那是"有空隙所以我能过"换了种说法
* **可行驶性** —— `drivable` 存在时求交；形状不匹配记 UNKNOWN 而非忽略
* UNKNOWN 不开逃生舱 —— 旧 bool 在 grid 缺失时返回 True

### 2.2 正反例与影子回放

先核查了 `town_1789886413` 有什么：**160 帧全是标量，没有栅格、没有
障碍层、没有候选轨迹**。用 `closest_obs_m` 重建 BEV 是编造输入，所以影子
回放改为复核旧判据实际做了什么：

**`corridor_open` 在 160/160 帧都是 True —— 一次都没说过 False**，包括
`closest_obs_m = 1.206 m` 那一帧。也就是说这个门根本没有分辨力，任何
归因到它的里程差都没有意义。

顺带发现 `closest_obs_m = 999.0` 是「未检测到障碍」的哨兵值，留在统计里
会**直接成为中位数**（原中位数 999 → 修正后 3.99）。

### 2.3 接线与最终仲裁

`corridor_feasibility` 已接进 occupancy 分支，默认关闭
（`BEAMNG_CORRIDOR_FEASIBILITY=1`）—— 横向模型是运动学，P2.2 还没标定。

审计后续链时发现一处**真实的硬约束被抬高**：
`target_sm = min(_lt.reference, plan_speed)` 只跟 plan 取了 min，没跟
`target`（= `min(verd.target_speed, plan_speed, args.speed)`）取。
监视器说 3.3 m/s、plan 允许 6.0 时，整形后的参考值会成为下发目标。
已改为 `final_target_speed(reference, plan_speed, hard_cap)`。

---

## 3. P3 性能：先量，再决定改什么

`scripts/m5_perf_profile.py`，**全量 131 个 town 跑次**（括号内为有该列的跑次数）：

| 量 | 全量中位数 | 备注 |
| --- | --- | --- |
| tick p50 | 177.9 ms | (131) |
| tick p95 | 409.8 ms | (131) |
| ring p50 | 78.6 ms | (131) |
| **semantic p50** | **301.0 ms** | (25) |
| object p50 | 7.5 ms | (25) |
| 控制间隔 p50 | 0.535 s（**1.9 Hz**） | (131) |

**达到 150 ms p95 目标的跑次：0 / 131。**

> 一处自我更正：单跑次 `town_1789890111` 上 ring p50 读数是 280 ms，
> 我一度据此写「ring 占大头」。全量中位数只有 **78.6 ms** —— 那个跑次是
> 偏高的个体。**跨跑次中位数才是能引用的数**，单跑次只能当例子。

这张表直接回答了 A4 的问题：最大单项是**同步 semantic（301 ms）**，而 A4
只让 `object` 异步（7.5 ms）—— 动不了阻塞项。

实测控制率 1.9 Hz，名义 substeps 是 15 Hz。名义值不等于达成值，所以加了
`substep_digest`（expected / executed / shortfall）和 `watchdog_verdict`
（基于**命令之间**的墙钟间隔，不依赖会阻塞的感知循环；默认关闭，阈值
1.5 s 高于实测最大间隔 0.801 s）。

`ring` 内部无法再分（外部模块，未插桩），脚本明确标注这是选择优化对象的
阻塞项，而不是猜一个。

---

## 4. P4 无边界与近场

**计时器被两种"假恢复"清零**：

1. 一帧 ON 就清零。带子是**感知**带且读的是前方 2–12 m，天生会闪。
   实测：ON 每 3 秒出现一次的序列，无滞回时峰值只到 2 s（永远够不到 8 s
   停车线），加滞回后越过 8 s。
2. 没有 grid 的帧也清零。那根本不是读数，现在改为**冻结**：发布上次读数
   并标 `road_checked=False`。

新增 `blind_drive_distance_m`：路面阈值写的是**秒**，但风险是**距离** ——
5 m/s 持续 8 s 就是 **40 m** 无路面证据的行驶，这个数字从没被验证过是可
接受的盲驶距离。

近场覆盖实测（`m5_boundary_coverage.py`，**全量 131 个 town 跑次**）：

| 量 | 全量中位数 / 计数 |
| --- | --- |
| lat_left 覆盖率 | 3.0%（最高 100%） |
| lat_right 覆盖率 | 2.2%（最高 100%） |
| 支持侧向安全声明的跑次（≥50% 门槛） | **12 / 131** |
| 最长双边缺失 | 中位数 164 帧，最大 226 帧 |
| **有任何 `road_checked=True` 帧的跑次** | **0 / 131** |

最后一行是上一轮证据索引里那条发现的**全量确认**：`road_checked` 至今
一次都没落过盘，所以「新遥测修好了默认值当健康」这件事**依然未被验证**。

覆盖率远低于门槛 → **记为传感器覆盖阻塞项**，不是靠下游阈值能调出来的。

---

## 5. P5 确定性闭环场景集 A–H

`scripts/m5_scenario_set.py` 定义 A–H，每个场景写明**要回答的问题**、
**必须观测到的列**、**放行判据**。

* D（已知碰撞前序）要求 `throttle` 和 `brake`，不只 `mon_target` ——
  比较目标速度不等于比较车做了什么
* C（空隙存在但不可达/不合法）要求 `corridor_state` / `corridor_reason`
* 随机化 **AB/BA 成对顺序**（种子化、成对平衡），效应按**对内差**估计，
  并同时报 spread；效应小于 spread 时明说无法区分噪声
* 统计单位是**跑次**，相邻帧不是独立样本
* 排除跑次必须声明理由，未声明直接 FAIL；无有效跑次是 UNKNOWN 且不放行
* 零碰撞只属于该测试集，脚本每次都把这句话打出来

---

## 6. P6 stall 归因（最后一步）

`scripts/m5_stall_attribution.py` 先分类再谈优化。**全量 131 个 town 跑次**，
停止帧占比中位数 **55.4%**：

| 类别 | 停止次数 | 占停止帧 | 最长单次 |
| --- | --- | --- | --- |
| obstacle_or_boundary | 516 | 36.6% | 117.6 s |
| **no_executable_path** | 227 | **32.7%** | **118.7 s** |
| unknown（`route_dist` 缺列） | 343 | 26.0% | 90.5 s |
| near_goal | 4 | 3.8% | 98.5 s |
| **unjustified（待优化）** | 33 | **0.9%** | 6.3 s |

**结论直接推翻了「no_stall = 无谓停车」这个默认理解**：无谓停车只占停止
时间的 0.9%，而且从没超过 6.3 s。真正的大头是 **no_executable_path
（32.7%，单次最长 118.7 s）** —— 那是规划器给不出路径，**速度地板对它
毫无作用**。

两处保守规则：

* 监视器要求的停车只有在**说出保护对象**时才算 `obstacle_or_boundary`，
  否则 `unknown` —— 把所有 degraded 停止都归成"障碍"会让问题看起来已经解决。
* `route_dist` **缺列**记 `unknown`，不算 `no_route_config`。131 个跑次里
  只有 35 个带这一列；按缺列判定会凭空造出 26% 的"无路线配置"结论。

---

## 7. 仍未解决（诚实清单）

1. **P2 的横向模型没有标定**。`min_turn_radius=5.5` / `max_lateral_speed=2.0`
   是运动学占位值。接线已完成但默认关闭，需要 P5 场景 C 的成对试跑。
2. **ring 内部不可插桩**，所以「semantic 具体是采集还是预处理慢」不知道。
3. **传感器覆盖是硬门槛**：lat_left/lat_right 只有 0.6–5.5% 覆盖，在换
   传感器配置之前，任何近场横向验收都不可能真正通过。
4. **1285 个用例全是离线/单元测试**，本轮没有任何一次实车跑次。
5. **决策项 5 依然挂着**：逃生舱用什么量区分绕得过去与绕不过去
   （`closest` 在两组场景只差 4 cm）。现在有了 `corridor_feasibility` 的
   几何答案，但标定仍未做。
