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
