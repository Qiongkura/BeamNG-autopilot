# T14 完整开发方案：W0 冻结 + P0 缺口（G01–G06）执行报告（2026-09-25）

方案：`docs/T14_COMPLETE_DEVELOPMENT_PLAN_20260925.md`（v1.0，核查基线 `0189509`）
本轮执行：**W0 冻结现状与实验协议** + **G01–G06 六个 P0 缺口** + W1 §6.2 部分标签语义
提交范围：`0322abb` → `2e4abe8`（11 个提交，按模块拆分）
门禁：`RESULT: pytest=PASS offline_validate=PASS`（**2652 passed**，
`logs/experiments/t14_w0_20260925/gate_w1.log`）

---

## 1. W0：冻结现状与实验协议（已完成）

| 方案要求 | 落地物 |
| --- | --- |
| git commit / 未提交差异清单 / 版本 / GPU / 相机分辨率 / 模型与后处理哈希 | `logs/experiments/t14_w0_20260925/baseline_freeze.json`（commit `0189509b94e1`、dirty 2 项**标记为用户工作**、Python/Torch/CUDA/OpenCV、GPU、536×403） |
| 显式目录清单（禁止递归扫描） | 只读 `--model` / `--run-dir` 点名的项；四个数据目录各带组/可用帧/被拒帧/覆盖/来源 rank |
| 生产模型与研究基线分别命名并存指纹 | `production` = `logs/m5_seg/seg_model/best.pt`（sha16 `b16a734d67f3c0de`）、`research_agentline`（`df197943a985de8e`）、`research_width2b`（`492e381cd4c4c76a`） |
| 冻结指标定义与覆盖要求 | `beamng_autopilot/experiments/protocol.py` + `docs/metric_protocol_v1.json`（**hash 36b5e5181ef5fff1**）+ 人读版 `docs/EXPERIMENT_PROTOCOL_20260925.md` |
| 读不到的东西记 UNKNOWN | 快照里每项失败都带 `status: UNKNOWN` + `error`；`will_update` 为空（W0 只读） |
| 控制权归属 | 快照写明 `game_session_started: false` / `vehicle_control: none` |

协议 v1 的要点：11 个指标各带**分母与未知规则**；6 类场景覆盖要求（各 20 帧、≥2 独立组、
≥6 路段组、主视角 front_main）；来源资格表（verified 可晋级 / agent、pseudo 只可训练与
测量 / unreliable、absent 全不可）。

## 2. P0 缺口逐条（先反例 → 再修 → 再验证）

### G01 停止条件只打印，不阻止启动
* **反例**（新增）：预置 2 轮"rejected 且无收益"历史 + `max_rounds_without_gain=2`
  → 实测入口**仍去调用采集**。
* **修法**：停止条件接成启动硬门（资源门之后、dry-run 之前），命中时打印
  "本轮不采集、不训练"、写 `stopped_before_start` 事件、返回 **rc=8**；
  顺带修 G08 相邻项：`rounds` 内部 `should_stop` 原来传 `gpu_minutes_today=0.0`
  （预算停止条件永不触发）→ 改读机器级账本。
* **验证**：`tests/test_autoloop_entry.py` 8 项（新反例 + 原预算/资源门用例）。

### G02 锁非原子、6 小时后可被抢
* **反例/正例**（新增 5 项）：并发只有一个成功；**活着但心跳旧的长任务不被抢**
  （旧实现 6 小时无条件接管）；死掉的持有者可回收；PID 复用（同号不同创建时间）
  按"不是本人"处理；心跳/释放尊重所有权。
* **修法**：`MachineLease`——`O_CREAT|O_EXCL` 原子创建、pid + **进程创建时间**
  （PowerShell 7 `Get-Process.StartTime`）校验、心跳（临时文件 + `os.replace`）、
  强抢只能显式 `force` 且记 `recovery_reason`；`run` 与 `rounds` 都持租约
  （同进程重复获取 `already_held`，不自锁）。

### G03 来源资格由字符串决定（agent 未自动列入研究来源）
* **反例**：`--paint-source X=agent_revision` 不带 `--research-arm` 原来能绕过；
  在命令行写 `human_revision` 能把 agent 数据升格。
* **修法**：`experiments/credentials.py` 读**帧所在目录自己**的凭证（self → parent），
  `protocol.effective_source()` **凭证优先**（命令行只能降不能升；无凭证时 verified
  声明一律不认）；`resolve_paint_sources()` 的结论写进判定文件；`rounds` / `evaluate`
  / `replay` 共用同一函数（G10）。
* **验证**：凭证 5 项 + 接线 4 项（无 flag 也不能晋级、声明 human 被降级、
  整轮记录资格且 replay 相同、evaluate 同样降级）。

### G04 路外假线率的分母含未知区（可被稀释）
* **反例**：向未知区添加预测，已知区假阳不变，但旧口径的比例**变小**。
* **修法**：`pred_line_known_px` / `pred_line_unknown_px` 分开计数；
  任务指标分母只用有效区；老产物缺该计数记 UNKNOWN；输出
  `offroad_ratio_denominator` 写明口径；"只在未知区预测"与"完全没预测"分开报。

### G05 判定只看 IoU（任务主指标没接线）
* **反例**：`rounds` 只送 IoU，而判定要求 PRIMARY_ORDER 内有改善
  → **任何候选都晋不了级**。
* **修法**：逐 seed 采集两臂任务主指标（身份率 / 漏线 / 精度 / 路外假线 / 延迟，
  与硬门同定义），`task_pairings()` 按 seed 组装（缺测不进配对、不补 0；
  假线与延迟按"越小越好"比较），与 IoU 代理一起进 `decide`。
* **验证**：三种判定例——主指标改善 → `shadow_candidate`；**只有 IoU 改善 →
  `rejected`**；主指标缺测 → `needs_evidence`。

### G06 游戏归属只看 PID 差集；枚举失败不阻断计时
* **修法**：`game_procs()` 带创建时间；`close_started_game(launched_after=…)` 只关
  **本次任务启动之后创建**的进程，拿不到创建时间或创建早于启动（PID 复用）→
  只报 `unverified` 不关；`timing_precondition()`：游戏在跑 → 延迟不可引用，
  探测失败（UNKNOWN）→ 同样不可引用，命中时硬门 p95 记 None + 原因入
  `timing_suspect`。
* **顺带修掉一个真缺陷**：ticks→unix 的偏移用 `datetime` 相减把符号弄反，
  于是"创建早于启动"永远判不出来（危险方向：把所有新 pid 都当自己的关掉）。

## 3. W1 §6.2 部分标签语义（方案点名 5 条测试）

| 方案要求 | 状态 |
| --- | --- |
| 区域改 UNKNOWN 后不贡献对应监督 | ✅ 测试：改未知区内的预测，损失必须不变 |
| 弱标签未标区域不提供错误负例 | ✅ **修掉一个语义漏洞**：`line_supervision_flags` 用 `usable` 后，弱来源的**零标线帧会被当成负例**；现在三态分开（不可用→屏蔽；已核验→零标线帧可当负例；弱/agent→只保留正例） |
| 已确认的无线帧可合法贡献负例 | ✅ `human_revision` 下零标线帧监督 |
| 没有复核的零标线帧不能被误认成无线 | ✅ 同上（弱来源屏蔽） |
| 全 ignore 帧不产生 NaN | ✅ **修掉真缺陷**：`F.cross_entropy` 对零有效像素返回 NaN（会污染整步权重）；两处加"零损失但保留计算图"护栏 |

## 4. 八项状态（方案 §18）

| # | 项 | 状态 |
| --- | --- | --- |
| 1 | commit/config/run 与控制权 | 11 个提交（`0322abb`…`2e4abe8`）；快照 `baseline_freeze.json` 记录 commit/dirty/模型指纹；**无控车任务**（`vehicle_control: none`） |
| 2 | 覆盖与 UNKNOWN | 协议 v1 定义分母与未知规则；冻结快照列出四个数据目录的可用帧与来源 rank；新指标 `pred_unknown_line_px` 单独上报 |
| 3 | 刷新与消费 | **未测**（本轮无驾驶闭环） |
| 4 | 几何与控制 | **未测**（同上） |
| 5 | 路面门 | **未测**（需 Tech 闭环） |
| 6 | 性能 | 离线分割链指标口径已修（分母只算有效区）；独占计时前提已加；**完整驾驶 tick deadline 未测** |
| 7 | 驾驶结果 | **未测**（碰撞/压线/出铺装/停车时长均需闭环） |
| 8 | 状态结论 | R1 部分推进（后台防线加固）；**R2/R3 未达成**；本轮无候选晋级（研究来源限制 + 任务硬门） |

## 5. 未做与下一步（按方案优先级）

1. **W1 §6.3 首批人工评价集**（6 类场景 × 20 帧）：需要人工时间；当前 640 帧
   只是**生成**，人工确认数为 0，禁止写成"640 帧人工真值"。
2. **W2 数据版本与空间隔离**：同曝光多视角/同地点重采的空间缓冲、最终集封存与
   访问账本（含"失败后不得对同一最终集调参"）。
3. **W5 剩余**：G09（候选匹配的覆盖/左右角色/参考缺失口径）、统计方法（成对 t 区间
   取代 `2sd/√n`）、分场景与最差场景报告、最终集消费账本。
4. **W3 剩余**：`max_wall_minutes` 真实消费、用户活动信号（未接入时必须显示"未接入"）、
   每个 seed/epoch 边界的可恢复停止、8 小时常驻验收。
5. **W4/W6/W7**：E0–E3 实验顺序（E0 可信基线、E1 复验 Tversky 2.0）、看板全链时间线、
   Tech 影子与短程闭环（前置 R2）。

**不得据本轮声称**：R2/R3 达成；任何候选已晋级；驾驶安全；"640 帧是人工真值"。
