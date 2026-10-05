# T16 限定场景 Tech 驾驶验收：预注册（2026-10-06，先定后跑）

**性质**：方案 §5「签收」的第三件（"可靠参考上的 R2、独立最终确认、**限定场景 Tech R3**"）
的**驾驶侧验收**。R2 已过（v8 生效 + mean 判门，2026-10-05），一次性最终确认已消费
（`docs/T16_FINAL_CONFIRM_20261006.md`）。本文件只定驾驶验收怎么判，**跑之前**写下。

**前置**：本项目铁律——横向参考**只来自感知**（漆线 + LiDAR 走廊），地图只回答"去哪"。
`town` 场景注册表条目本身就是这条规范的固化：`lane_mode=sensor`、`strict=True`、
goal=(868.3, 744.9)（road-graph 点）、`seg_model=logs/m5_seg/seg_model_hand/best_task.pt`。
即"goal、sensor、strict"三项**不需要额外开关**，注册表默认即是；唯一控制者由
`m5_fsd_benchmark` 的排他审计（manifest `exclusivity`）保证。

## 1. 设计（单因子 = 无因子：这是验收，不是实验）

| 项 | 值 |
|---|---|
| 场景 | `town`（限定场景；`require_goal=True`） |
| 配置 | **注册表默认**（sensor + strict + 固定 goal + 固定 seg_model）；相机/规划/安全阈值一律不动 |
| 协议 | v8（生产默认，HEAD 生效版本；manifest 记录 `BEAMNG_PROTOCOL`） |
| 重复 | **4 次**（驾驶非确定；§9 已实测 run 间方差大于配置差，单次不作数） |
| 唯一控制者 | manifest `exclusivity.ok` 必须为真；若为启动器误报，用 `--allow-contaminated` 并在 manifest 留痕（记录在结果里） |
| 采样 | 每帧遥测 `logs/fsd_benchmark/town_<ts>.json`；记分卡 `scorecard_<ts>.json` |

## 2. 判定规则（先定后跑，不放宽）

**硬门（`beamng_autopilot/eval.score_run`，仓库既有，不改）**：
`has_frames`、`no_reversing`、`no_centre_crossing`（参考线**与车体**两条都算）、
`no_edge_crossing`、`no_stall`、`on_road`、`no_collision`、`reached_goal`（goal 场景）。

**读法**：

1. **PASS 只在 4/4 次全部 PASS 时给出**——门不由多数票清掉；一次 FAIL 即 FAIL；
2. **UNKNOWN 不释放门**（`lat_left/lat_right` 整段缺失 = 未测，不是"没压线"）；
3. 安全项报**最大值**（最差那次），可用率报**中位数与范围**；
4. **单次 `damage=0` 不是安全证明**：碰撞/出铺装/压线必须 4 次全测到且全过才算；
5. 缺测、ABORT、无遥测 → UNKNOWN，如实记录，不当通过。

**若 FAIL**：如实报告并停在此处（不调阈值、不改注册表、不挑场景重跑）；
失败项按 §9 的已知归因（感知配对可用率 ↔ minimal-risk 停车）单列。

## 3. 复现命令

```pwsh
# 4 次 × town（注册表默认 = goal + sensor + strict）
foreach ($i in 1..4) {
  .venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --scenarios town `
      --runtime tech --allow-contaminated
}
# 记分（含 goal 检查）
.venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --score <4 个 town_*.json>
```

`--runtime tech` 与注册表默认的其余项不冲突；`--allow-contaminated` 仅在排他审计
误报时使用，manifest 会记账。

## 4. 执行记录

见 `docs/T16_TECH_DRIVING_ACCEPTANCE_RESULT_20261006.md`（跑完写）。

## 5. 过程中发现并修掉的两个自身缺陷（先修后跑，记录在案）

1. **驾驶收尾不关游戏**（提交 `b681cd1`）：驾驶侧只 `conn.close()`（关连接），
   不关进程。实测：第一次验收轮 run 1 跑完后机器上留下 9 个 BeamNG 实例，
   而且它们**持有 stdout 管道**，把 `... | tail -8` 的无人值守循环**卡死**
   （run 1 结束后 12 分钟没有任何输出）。修法：复用采集侧
   `close_started_game`（差集 + 创建时间所有权校验，`--attach` 不碰）。
   与 2026-09-25 采集侧同款缺陷、同一纪律。
2. **manifest 只记 CLI base，不记生效策略**（提交 `60e2a5e`）：town 的
   sensor/strict/goal/seg_model 写在**场景注册表**里，manifest 却记 CLI 默认值
   （`lane_mode=map`、`strict=false`、`goal=null`）——记录与实跑不符，验收
   就无法自证"同一配置"。修法：`run.effective[场景]` 记 `scenario_args()`
   解析后的策略；`run.protocol` 记生效协议版本（v8 生效后 env 未设也要查得到）。

**影响**：这两处修复都不在驾驶控制路径上（收尾 + 记录），但为保证**同一提交**
的 4 次验收，第一轮（`2afe733`）的数据只作过程记录，正式验收 4 次全部在
`60e2a5e`（修复后）重跑。
