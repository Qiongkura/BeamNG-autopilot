# T18：传感器回放正确性专轮结果（2026-10-11）

计划：`docs/T18_SENSOR_REPLAY_CORRECTNESS_PLAN_20261009.md`  
固定配置：`docs/T18_TECH_REPLAY_CONFIG_20261009.json` + 重试配置 `docs/T18_TECH_REPLAY_RETRY_CONFIG_20261009.json`

## 结论摘要

本轮代码与确定性回归已完成；T18 的同源回放与 CPU 比较已有合格证据。Tech 首次 live 尝试因曝光跨度不足被判无效，
按预注册原样保留；一次重试合格。**本轮支持的是同一冻结输入上的 CPU 处理输出一致性与阶段计时记录修正，
不支持独立路面/漆线真值、安全性、执行 ACK、驾驶质量或模型晋级。**

## 1. 代码与冻结源

| 项 | 结果 |
|---|---|
| T17 源快照 | `logs/experiments/t18_reference_20260910/providers_9b10da96.py`（sha256 `50bc24fb57eb2039a6aea05dccd685036cc37728f528748ae99c5b19a8ce8a18`） |
| 当前源快照 | `logs/experiments/t18_codefreeze_after_gate_20261009.json`；代码 hash 在 T18 配置中绑定 |
| 修改范围 | Tech provider 的 source metadata/阶段时间、range replay、segmentation raw API 的 argmax/road 保留、geometry probe/资格入口 |
| 默认行为 | 经典 CV、Steam、控制/安全硬约束、默认异步开关与形状阈值不改；捕获默认关闭 |

## 2. 同源回放与 CPU 比较

输入来自第一次 live 尝试的**第一份合格 warmup capture**，文件摘要写入 replay manifest，未使用 pickle：

* `logs/experiments/t18_geometry_replay_20261009/pose_00/range_replay/`
* `cloud.npy`、`sample.json`、`manifest.json`
* 输入点数：423,670；障碍数：80；ray hit 数：66；输入清单 SHA-256 在 `cpu_comparison.json`
* reference provider：`providers_9b10da96.py`；sha256 绑定

两轮、每条件每轮 10 次，顺序 `[reference,current]` / `[current,reference]`，另加每轮 warmup 2 次：

| 条件 | p95（最佳轮） | 原始值保留 |
|---|---:|---|
| reference | 172.647 ms | 20 次原始 elapsed 全保留 |
| current | 132.420 ms | 20 次原始 elapsed 全保留 |

* `outputs_equal = true`
* `compared_calls = 48`
* 实测范围是**同源输入、fresh-tracker 单帧 CPU processing**；不是完整历史 runtime，也不是多帧 tracker 比较。
* `independent_safety_truth = UNKNOWN`，没有把 CPU 处理加速写成安全或驾驶收益。

## 3. Tech live 资格与失败原因

### 首次尝试（保留，不丢弃）

`logs/experiments/t18_geometry_replay_20261009/pose_00/`：

* drive elapsed：26.359 s；遥测第一帧 `t=7.146`、最后 `t=25.354`，曝光跨度 **18.208 s**；
* 预注册最低曝光要求为 24 s，因此 `qualified=false`，原因是 `telemetry exposure too short`；
* 首次尝试的 replay capture、telemetry、runtime audit 全部保留；不参与等曝光 live 比较。

### 重试（合格）

`logs/experiments/t18_geometry_replay_retry_20261009/pose_00/`：

* `qualified=true`，同一模型、地图、起点、goal、strict/sensor、25 s/150 s、geometry probe、capture 开关；
* cleanup：`closed=true`、`failed=[]`、`unverified=[]`、`still_running=[]`；
* `source_poll_times_strictly_increasing=true`；range `scanned` 40 帧；capture artifact 已验证；
* geometry/replay 结果：生产安全评估回放与记录一致；没有把 body coverage UNKNOWN 读作安全通过。

## 4. UNKNOWN 与边界

| 项 | 状态 |
|---|---|
| 真实 sensor capture time | UNKNOWN；本轮记录的是 `poll_start_proxy` |
| 独立铺装真值 | UNKNOWN；不能从 `road_surface` 或 `road_off` 推导 |
| 独立真实漆线真值 | UNKNOWN |
| 车体盲区/完整 body coverage | UNKNOWN |
| actuator ACK | UNKNOWN；`cmd_seq` join 不等于执行确认 |
| 实际安全结论 | 未测；geometry probe 只验证生产 predicate 与回放 predicate 一致 |
| 模型/驾驶晋级 | 未做；本轮不训练、不消费封存最终集、不改生产默认异步开关 |

## 5. 验证记录

* 定向回归：T18 range replay / Tech split / geometry probe / pose qualification / drive pipeline 全部通过；
* 全量 `pytest tests/`：本轮运行完成后记录在 session 输出；
* `git diff --check`：应在提交前复跑；
* 结果与证据在 `logs/experiments/`，不提交 logs；本文件与脚本提交。

## 6. 下一步

T18 达成了计划中的 replay correctness 与同源 CPU 比较，但未关闭独立真值、执行 ACK、真实 capture time、
多次配对闭环和学习收益。下一阶段不能把 132.420 ms 的 p95 直接转成驾驶安全结论；应先补真实 source capture timestamp
或继续保持 UNKNOWN，再设计 T17/T18 的第二次配对 Tech 运行。
