# T16 驾驶层根因实验：预注册（2026-10-06，先定后跑）

**背景**：限定场景 Tech 驾驶验收（`docs/T16_TECH_DRIVING_ACCEPTANCE_RESULT_20261006.md`）
town ×4 全部 FAIL：`no_stall`（stall_frac 0.93–1.0）、`no_centre_crossing`
（**车体**压线 12–24 帧）、`reached_goal`（离目标 76.5–88.8 m）。本实验按
"**根因 → 单因子 → 预注册 → 实测**"推进，目标是把这两个失败项**定量归因**，
并逐个验证修复因子。**不放宽任何硬门、不动安全阈值、不同时改两个因子。**

---

## 1. 离线根因（先用已录遥测做，不开游戏；4 次验收轮 808 个 settled 帧）

| 观测 | 数值 | 含义 |
|---|---|---|
| `reason=no drivable path` | 481/808（59.5%） | 停车的第一大原因：**规划器没产出路径** |
| 其中 `lane_drivable.frac < 0.6`（铺装门撤销） | 284/808（35%） | 撤销后无感知车道 → 严格模式停 |
| 其中**有**合格车道但仍无路径 | 202 帧 | 车道在，规划器仍拒绝 |
| `reason=planned vehicle body crosses lane boundary` | 176 帧 | 全部为**已配对**车道；`lane_side_off_m` 中位 **-2.15 m**（限 -0.20），`lane_dev_m` 中位 0.82 m，`lane_bear` 中位 -10.9°，**速度中位 0.001 m/s** → **静止死锁**：车停在车道左缘，任何路径的车体扫掠都越界 → 拒绝 → 更停 |
| 有界 PATH_HOLD | 4 次运行 **24/40/17/24 次 verified offers，仅 2/2/1/1 帧被复用**；safe-stop 帧 180/163/207/183 | 监控器**反复给出**已验证轨迹，驾驶层几乎不复用 |

**代码级根因（第一版判断已被代码复核推翻，记录在案）**：我先按"`no_drivable_path`
分支无条件把裁定改成 `minimal_risk/0` 并 return"推断 hold 被覆盖；**复核代码后推翻**：
`_evaluate_core` 在服务 hold 时会 `path = served[0]`（用 held path 顶替缺失的规划路径），
所以该分支在 hold 已服务时**不会触发**，裁定是 `degraded`（`drivable=True`），驾驶层的
消费条件也就能通过。**真正的缺口**是：481 个"无路径"帧里 `path_hold_active` **全为 0**
——即监控器**根本没服务**（`_serve_hold` 返回 None），而遥测里**没有任何字段**说明
被哪一条有界复检挡住。因此本实验的第一步是**补诊断并实测拒绝原因**（只读字段，
不改行为），再据此设计修复因子。

**已排除的候选**：`HOLD_JOINT_GATE`（`BEAMNG_HOLD_JOINT_GATE`，默认 OFF）——
481 帧的 `hold_audit.enforced` 为 False，它没有参与拒绝。

## 2. 实验步骤与候选单因子（各带一个开关，默认关；同一提交内可 A/B）

**步骤 0（先做，只读诊断）**：`SafetyVerdict.hold_refuse_reason` +
`PathHold.request(refuse_out=…)` + 遥测字段 `hold_refuse_reason`：
记录"无路径 tick 为什么没复用 hold"（过期 / 漂移 / 前向弧不足 / 车体越界 /
held path 越界 / 占据阻塞 / 未 offer）。**跑 town ×2 收直方图**，据此确定修复点。

| 因子 | 开关 | 改什么 | 不改什么 |
|---|---|---|---|
| **F-A 停车/爬行策略**（步骤 0 之后按实测原因定） | `BEAMNG_HOLD_*`（名字随实测定，默认关） | 只针对**实测到的主导拒绝原因**做一处改动（候选：把 hold 窗口从"offer 年龄 ≤0.8 s"改为 T09 契约的"**观测年龄 ≤0.8 s 且自观测以来行驶 ≤12 m**"——两者都是有界条件，后者正是 `hold_audit` 已经测的量） | 宽限/上限语义、每次复检（车体/边界/占据）、障碍风险层、超时清除——全部不变 |
| **F-B 横向参考质量**（F-A 之后做） | `BEAMNG_PAVEMENT_RECENTRE=1` | 铺装门撤销后，用**观测到的可行驶带**把车道中心重定中（侧向限位内），对**新中心**重跑侧向门+铺装门，过了才接受（来源仍是 perception） | 铺装门本身（阈值/语义）不变；不引入地图横向参考 |

**为什么 F-A 不是"放宽安全"**：它是把**已实现、已单测、有界**的机制从"被覆盖"
恢复成"生效"；蠕行窗口 ≤0.8 s 且每次服务都按当前场景复检，障碍风险层仍在
（已有测试 `test_risk_layer_constrains_a_served_path_hold` 钉住"hold 不能朝
接触带障碍开"）。安全由**实测非劣性**把关（见 §3）。

## 3. 判定规则（先定后跑）

- **场景/次数**：`town`（注册表默认 = goal + sensor + strict）× **每臂 4 次**，
  **交替执行**（off,on,off,on,…）以摊平时间漂移；同一提交、同一协议（v8）。
- **硬门**：仓库既有 `eval.score_run`（`has_frames/no_reversing/no_centre_crossing/
  no_edge_crossing/no_stall/on_road/no_collision/reached_goal`）。**每臂 4/4 才 PASS**；
  UNKNOWN 不释放门。
- **安全非劣性（必须不更差）**：对 off/on 两臂取**最大值**比较
  `collision_count`、`off_road_frames`、`cross_centre_frames`、`body_cross_centre_frames`、
  `cross_right_frames`、`body_cross_right_frames`。**任一项 on 更差 → 该因子否决**
  （回滚默认，如实记录）。
- **可用率**：`stall_frac`、`travelled_m`、`goal_dist_m`、hold 复用帧数
  （`[fsd-drive] path hold: …` 行）报**中位与范围**。
- **不当作通过**：单次 `damage=0`；部分口径缺测；"更接近目标"不等于达门。

## 4. 复现命令

```pwsh
# 交替 8 次（off/on 各 4）
foreach ($i in 1..4) {
  .venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --scenarios town --runtime tech --allow-contaminated
  $env:BEAMNG_HOLD_CREEP_ON_NOPATH="1"
  .venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --scenarios town --runtime tech --allow-contaminated
  Remove-Item Env:BEAMNG_HOLD_CREEP_ON_NOPATH
}
```

## 5. 结果

见 `docs/T16_DRIVING_ROOTCAUSE_RESULT_20261006.md`（跑完写；含逐次表 + 判定 + 是否否决）。

## 6. 步骤 0 实测（诊断轮，2 次，提交 `c978b90`）

town ×2（注册表默认），每次 50 / 30 次 verified offer，**只复用 5 / 3 帧**；
无路径帧 115 / 113，其拒绝原因直方图：

| 拒绝原因 | run1 (115) | run2 (113) |
|---|---|---|
| **no hold offered**（根本没有可服务的 hold） | **86** | **86** |
| current body crosses a boundary | 9 | 8 |
| held path now crosses a boundary | 5 | 4 |
| expired（明确报年龄，1.0–3.0 s > 0.80 s） | ~13 | ~13 |
| 其它 | ~2 | ~2 |

**读法（重要）**：停车的主导原因**不是**"hold 被某条复检挡住"，而是
**"根本没有 hold 可用"**——`no hold offered` 占 ~75%，其成因是 hold 窗口
（offer 年龄 ≤ `FSD_PATH_HOLD_MAX_S` = 0.80 s）远短于规划器两次可行驶输出之间的间隔
（本场景下可行驶 tick 仅约 12–25%）。次要项：车体/held path 越界（约 8–12%，
与"车停在车道左缘"一致）、明确过期（约 11%）。

**据此的修复候选（尚未实现、尚未 A/B，故本实验只到"归因"这一步）**：
把 hold 窗口从"**offer 年龄** ≤0.8 s"改为 T09 契约的"**观测年龄 ≤0.8 s 且自观测
以来行驶 ≤12 m**"（这两个量 `hold_audit` 已经在测、且都是有界条件），
使"感知新鲜但规划器暂无路径"的时段可以继续有界蠕行，而不是每次 offer 后
0.8 秒就永久清除。**安全读数按 §3：安全项取最大值、4/4 才 PASS、更差即否决。**

**未完成（接手须知）**：F-A 已实现并实测（见下）；F-B 尚未实现/未实测。
驾驶验收结论不变：**0/4 未过门**。

## 7. F-A 实测结果（2026-10-06，8 次交替 A/B @ `761ce83`）：**否决**

| 项 | off（4 次） | on（4 次） |
|---|---|---|
| 硬门 PASS | 0/4 | 0/4 |
| 安全最大值（coll/off/C/bC/R/bR） | 0/0/0/**29**/**0**/2 | 0/0/0/**30**/**1**/1 |
| stall_frac 中位 | 0.986 | 0.958 |
| 行进中位 | 3.15 m | 8.5 m |
| 离目标中位 | 86.8 m | 81.8 m |
| hold 复用帧（逐次） | 4/0/2/1 | 12/8/16/6 |

**判定**：机制生效（复用 6×、行进 2.7×）但**硬门仍 0/4**，且按 §3 的安全
非劣性规则 bC 29→30、R 0→1 **更差** → **F-A 否决**，开关保持默认关。
完整逐次数据与读法：`docs/T16_DRIVING_ROOTCAUSE_RESULT_20261006.md`。

## 8. F-C（横向放置）：路径没有落在感知车道中心上（2026-10-06 新增，先测量后修）

**为什么加这条**：F-A 否决、F-B 被证据否定之后，按"根因 → 单因子"继续追。
12 次 town 运行、2443 个 settled 帧的实测：

| 观测 | 数值 |
|---|---|
| 有感知车道（`lane_sel=sensor`）的帧 | 999 |
| 其中**路径到车道参考的中位距离**（`lane_dev_m`，无符号） | **0.773 m**（p90 1.05） |
| 其中距离 **>0.5 m** 的帧占比 | **62%** |
| `plc_active`（把路径拉向感知车道中心的校正器） | **0 帧为真**（999 帧全关） |
| 死锁帧（512 帧）的 `lane_dev_m` / `first_cross_m` | 0.86 m / 中位 **2.50 m** |

`painted_line_correction_active()` 在 `lane_src_sel == "sensor"` 时**直接返回 False**
（设计假设："感知车道已是参考 ⇒ 路径本来就跑在它上面"）。实测显示该假设不成立：
**有感知车道的帧里 62% 的路径离车道中心 >0.5 m**，而唯一能把路径拉回中心的机制
恰好在这时被关掉。这条 0.77 m 的横向偏差与两个失败门直接相连：
车道宽 4.06 m（半宽 2.03 m）+ 充气车体（11° 歪斜时横向半径 ~1.6 m）⇒ 车体扫掠越界
（`no_centre_crossing` 每次不过）与静止死锁（越界点 2.5 m < 4 m 硬门）。

**先测量、后修（本因子的第一步，只读诊断）**：0.77 m 的偏差有两种互斥解释，必须先分辨：
1. **规划器自己的路径就偏**（`best` 相对 `out.lane_ref` 偏）→ 修规划器的横向放置；
2. **两层用的参考不一致**（规划器的 `out.lane_ref` 与监控器 `scene.lane_ref` 不同）→
   修参考传递（历史上已修过同类：重建 Scene 曾把整路中心喂进安全横向检查）。
新增只读字段：`plan_dev_m`（`best` vs `out.lane_ref`）、`ref_gap_m`
（`out.lane_ref` vs `scene.lane_ref`），跑 town ×2 收直方图后定修法。

**判定规则同 §3**（硬门 4/4、安全项最大值非劣、可用率中位与范围、UNKNOWN 不释放）。

## 9. F-D（选项①：修横向参考——让驾驶栈用**已交付的感知臂**，2026-10-06 新增）

**为什么是这条**：三条驾驶层因子（F-A/F-B/F-C）都改不动硬门，残余阻塞点已定位为
**路径可用率只有 35%**（805 个 settled 帧里 284 帧有路径；`plan_blocked=no_perception_lane`
是主要 blocker）。路径可用率的上游是**感知车道可用率**（配对可用 + 未被铺装门撤销）。
选项①"修横向参考（感知侧）"的**最直接、单因子**实现是：**驾驶栈目前用的不是本方案
交付的感知臂**——`town` 注册表钉的是 `logs/m5_seg/seg_model_hand/best_task.pt`
（2026-09-11 的人工标注小模型），而 T16 全流程交付的感知臂是
`logs/experiments/t16_negdose6x_20261001/round0/seed42/checkpoint_last.pt`
（R2 过门、一次性最终确认已消费的那个候选）。

**候选不是按分数挑的**（避免在评价集上选型）：它就是本方案记录在案的**交付候选**
（规则 = 项目默认 seed 42，按索引选，见 `docs/T16_FINAL_CONFIRM_20261006.md` §1）。
本因子只回答一个问题：**把交付的感知臂装进驾驶栈，路径可用率与硬门会不会变好。**

**单因子**：`seg_model`（town 场景），CLI 覆盖（`--seg-model`；注册表只在 CLI 未给时
才用自己的 pin，`scenario_args` 的既有语义）。其它一切不动（相机/规划/安全阈值/协议）。

| 臂 | 命令差异 |
|---|---|
| off（对照） | 注册表默认 pin = `logs/m5_seg/seg_model_hand/best_task.pt` |
| on（交付臂） | 追加 `--seg-model logs/experiments/t16_negdose6x_20261001/round0/seed42/checkpoint_last.pt` |

**臂分类**：从每次运行的 manifest 的 `run.effective.town.seg_model` 读（不按运行顺序）。

**兼容性前置检查（已做，离线）**：两个 checkpoint 都能被驾驶栈的 `Segmenter` 加载并
推理；在无漆线帧上交付臂输出 0 个线像素（正确），人工臂输出少量（假线）。

**判定规则（同 §3，另加本因子主指标）**：
1. 硬门 4/4 才 PASS；UNKNOWN 不释放门；
2. 安全项取两臂最大值比较，**任一项 on 更差即否决**；
3. **主指标（本因子）**：**路径可用率**——settled 帧中"规划器产出路径"的占比
   （遥测口径：`lane_dev_m` 有值 = 有路径；另报 `lane_sel=="sensor"` 占比、
   `source!="none"` 占比、`plan_blocked` 直方图）；报中位与范围；
4. 可用率（stall_frac / travelled / goal_dist）报中位与范围。

## 10. F-E（选项①续：铺装门只回答"在不在**观测到的可行驶面**上"）2026-10-06

**证据（2 次 town 诊断轮，415 settled 帧；新增的只读分级字段）**：

| 观测 | run 1 | run 2 |
|---|---|---|
| 被铺装门撤销的帧 | 88/210（42%） | 114/205（56%） |
| 撤销帧的 `frac` 中位 | 0.40 | 0.46 |
| 撤销帧 `n_obstacle`（观测样本中被判障碍）中位 | **4**（共 7） | **7**（共 13） |
| 撤销帧 `n_off_mask`（观测到但**不在可行驶掩码**上）中位 | **0** | **0** |
| **去掉 obstacle 项后 ≥0.6（会被接受）** | **88/88** | **114/114** |
| 撤销帧的 `n_off_mask` 分布 | 0→78、1→10 | 0→114 |

**读法**：这个门名为"中心必须落在传感器观测到的可行驶面上"，但实测**202/202 的撤销
全部由 obstacle 项触发**，`n_off_mask` 几乎恒为 0——即**没有任何一帧是因为"中心不在
可行驶掩码上"被撤销的**。而 `obstacle` 层是**BEV 头的预测**（`fsd_stack.py:1565`：
`grid.obstacle[:] = (out.bev >= 0.6)`），于是该门在实车链上退化成
"BEV 头在车道中心误报障碍 → 撤掉整条车道" → `plan_blocked=no_perception_lane`
（实测 119–175 帧/次）→ 无路径 → 停车。这正是路径可用率只有 ~20–35% 的直接机制。

**单因子（默认关）**：`BEAMNG_LANE_PAVEMENT_GATE_NO_OCC=1` ——
**车道接受**用的铺装门只按"观测到的可行驶掩码"判定（去掉 obstacle 合取项）；
`obstacle` 证据**保留在诊断字段里**，且**所有路径级检查一字不动**
（`_path_occupied_fraction`/`path_blocked`、车体扫掠/边界、障碍风险层）。

**为什么这不是放宽安全门（写进记录供复核）**：
1. 门自己的 docstring 定义的任务就是"中心落在观测可行驶面上"，drivable 项才是它的
   实现；obstacle 项是额外合取，实测它在本链上由**模型预测**（BEAMV 头）而非观测几何触发；
2. 障碍**没有被忽略**：路径级检查用**更好的几何**（实际路径而非参考折线）在做同一件事，
   且这四条检查全在（规划器占位 → path_blocked、车体扫掠、风险层 TTC/接触带）；
3. 变更在开关后、默认关；按 §3 判定（硬门 4/4、安全最大值非劣、UNKNOWN 不释放门），
   **主指标 = 路径可用率**（本次预期直接上升，因为撤销机制被移除）。

**实现位置**：`beamng_autopilot/lane/reference.py` 的铺装门调用点传
`ignore_obstacle=<开关>`；`_drivable_fraction` 增加该参数（默认 False → 其它调用点
行为不变，含 divider 回退）。`BEAMNG_LANE_PAVEMENT_GATE_NO_OCC` 登记进 run_manifest。

## 11. F-F（选项②：静止 + 当前车体不越界时的**有界居中回正**）2026-10-06 深夜

**为什么现在做它**：F-E 把路径可用率推起来了（11/58/70 → 52/107/93/70 帧、行进
6.85→14.1 m），但硬门仍 0/4，`stall_frac` 仍 0.92–0.96 —— **瓶颈已转移为"有路径但被
4 m 车体扫掠硬门拒绝 + 歪斜停车位姿"**（结果文档 §8）。死锁几何实测（12 次 town、
512 帧）：**96% 的车是静止的、当前车体 512/512 都在道内、越界点中位 2.50 m
（509/512 落在 2–4 m，正好卡在 4 m 硬门内侧）**。即：一辆歪着停在离车道中心 0.86 m
处的车，**每一条路径都被拒绝，永远无法回正** → 静态死锁。

**单因子**：`BEAMNG_STATIONARY_RECENTRE=1`（默认关）。打开后：
**静止（≤0.3 m/s）+ 当前车体不越界 + 越界点 ≥2.0 m + 感知车道参考（REF_SENSOR）+
传感器/规划新鲜** 时，不立即停车，而是**蠕行**（`min_risk_speed × 0.5`）。
当前车体越界、越界点 <2.0 m、无感知参考、传感器过期 → **原样硬停**；
车速一旦 >0.3 m/s 例外立即失效 → **不能带着越界扫掠行驶**；障碍风险层照常生效。

**基线（两臂相同）**：`BEAMNG_LANE_PAVEMENT_GATE_NO_OCC=1`（F-E 开）——因为
"有路径可被拒绝"是这条例外能起作用的前提；**F-E 自身的采纳与否仍待计划层**
（结果文档 §8），本实验的标注是 **F-F | F-E=1**。

**判定规则（本文件 §3 的读法 + 一条**更好规定**的安全读法）**：
1. 硬门 4/4 才 PASS；UNKNOWN 不释放门；
2. 可用率：`stall_frac` / `travelled_m` / `goal_dist_m` 报中位与范围；
3. **安全主读法（本次起改为按行进量归一）**：`body_cross_centre_frames` 与
   `cross_centre_frames` **每 100 m 行进**的最大值（两臂比较）。理由：这两个量是
   **帧计数**，而两臂行进量可以差 2 倍（F-E：6.85 vs 14.1 m），计数直接比会**把
   "车动得多"读成"更不安全"**——本会话已记录两次（F-C 的 bR 0→2、F-E 的 bC 23→66）；
   原始最大值**同时照报**，供复核。
4. 采纳条件：硬门有改善 **且** 归一化安全率不更差；否则否决（默认关，代码保留）。

**性质声明**：这是**新的预注册**（用更好规定的规则），不是对 F-C/F-E 结果的事后改判；
那两次判定按当时的规则保持不变。
