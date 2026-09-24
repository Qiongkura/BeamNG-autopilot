# 第四轮升级报告：硬约束、几何与证据链（2026-09-21）

依据 `ROUND_REPORT_20260920_round3.md`，同时核对 `REPORT_20260920_round3.md`、
`README.md`、`AGENTS.md` 与当前调用链。汇报遵循 `REPORT_FORMAT.md`。

本轮是代码与离线验证升级，**不是 Tech 驾驶验收通过**。没有启动游戏、训练或替换模型。
默认实验开关保持不变；修复既有安全缺陷的代码会默认生效，不能沿用旧报告中
“所有行为变更都在开关后面”的描述。

## 0. 基线与范围

| 项目 | 记录 |
| --- | --- |
| 开始时分支 / HEAD | `main` / `21c32ab` |
| 本轮工作分支 | `fix/round3-hardening-20260921` |
| 本轮提交 / 推送 | 无，改动保留在工作区 |
| 既有未跟踪目录 | `.git.bak-20260920/`、`.workbuddy-ai/`、`rescue-20260920/`，未修改 |
| 修改前 pytest | 1585 个用例，1584 通过、1 失败 |
| 修改前失败项 | `test_fsd_closed_loop_recovers_from_a_body_crossing` |
| 修改前深度离线验证 | `RESULT: ALL PASS` |
| 本轮 Tech 实际驾驶跑次 | 0 |
| 本轮重算历史集合 | 131 个 town 跑次，26,202 帧，只读 |

## 1. 最终控制链

- **原假设**：`final_target_speed` 已经保证所有速度整形都不抬高硬上限。
- **是否真正激活**：旧代码只在默认关闭的 LONG_PLAN 分支调用；默认 ramp 只与
  `plan_speed` 取最小值。`painted_body_cross` 的停车标志还会被后续净空结果覆盖。
  DQN 限速也会被随后恢复的 `plan_sm` 覆盖。
- **可复现输入**：上一目标 6 m/s，计划允许 6 m/s，监控器要求 3.3、1 或 0 m/s；
  已触发硬停但后续出现净空或终点对齐请求。
- **实测结果**：默认 ramp 与 LONG_PLAN 都在速度控制器入口统一受最终 target 约束；
  净空硬停与原停车标志取逻辑或；已被监控器拦停的 PATH_HOLD 不再重新进入仲裁；
  最终踏板发送前重申硬停，DQN 上限不再被计划平滑抹去。接受 PLC 修改路径后使用
  该路径重新评估的裁决。
- **支持或推翻**：推翻“只修一个整形分支就修好最终约束”的假设。修复目标是
  已授权的速度和硬停最终到达控制发送入口，而不是只检查一个数学辅助函数。
- **未知与下一步**：发送回执不是车辆实际执行确认；制动距离、轮胎附着与控制延迟
  仍需要 Tech 场景 D 验证，不能据离线踏板值宣称碰撞已避免。

新增运行记录包含 `hard_cap`、`final_target_speed`、`final_stop`、`dqn_cap`。

## 2. 仲裁与路面丢失

- **原假设**：scattered obstacle 等软减速不会影响后续硬停；路面恢复滞回已生效。
- **是否真正激活**：同一当前车身越界场景，没有杂波时停车，加入杂波后旧代码可返回
  3.3 m/s。旧 road timer 虽保留 lost_s，但响应又受单帧 raw ON 短路，确认窗口内
  仍可放行。旧 stale、hold、road degraded 分支也可能跳过后续硬检查。
- **可复现输入**：杂波与当前/近端规划越界同时成立；路面丢失 9 s 后仅一帧 ON；
  ON 确认中插入缺失或陈旧网格；hold 期间出现路面或障碍硬停。
- **实测结果**：软规则累计取更低上限，继续执行适用的硬检查；合法正向收敛恢复仍为
  degraded 缓行，不通过放宽车身规则恢复。路面观测不依赖是否已有规划路径；
  缺失/陈旧读数不确认恢复，ON 必须满足确认窗口才解除丢失响应。
- **支持或推翻**：支持复合故障下硬规则不得被软规则遮蔽。`rules_evaluated` 现在来自
  实际访问，不再按 reason 在固定列表中的位置猜测。
- **未知与下一步**：2–12 m 路面带仍不是车身脚下铺装或近场横向安全真值。
  本轮没有改变 4 s/8 s 等标定阈值，也没有启用默认关闭的路面响应开关。

`fsd_drive` 现在记录 `effective_rule`、`rules_evaluated`、`rules_unevaluated`、
`masked_hard_rules` 和结构化 corridor 结果。旧 bool 走廊明确标注
`method=legacy_bool, enabled=False`，不能冒充新几何证据。

## 3. 配对几何与无路径处理

- **原假设**：详细第三轮报告 §15 认为 `lane_envelope.center` 的世界坐标转换错误
  导致闭环恢复失败。
- **是否真正激活**：检查发现 envelope 只复制几何。测试 fixture 反而使用车辆当前
  heading 旋转“固定世界道路”，因此 yaw=12° 时标线已经不在 world y=0/-3.5。
  另有真实数学错误：配对最近点投影把线段长度当成方向向量进行点积。
- **可复现输入**：固定 y=0 与 y=-3.5 的标线，改变车辆平移和旋转；独立标量最近点
  oracle、稀疏线段、重复顶点、弯曲线段；原 60 tick 压线恢复用例。
- **实测结果**：点积修正为实际线段向量；固定世界 fixture 后 pair 中心位于 -1.75 m，
  envelope 与 pair 坐标精确一致且内存独立。原恢复断言未放宽，通过且不需恢复 strict
  envelope fallback。额外测量：前进 15.0 m，冻结 0 tick，穿线深度初始 0.137739 m、
  最大 0.153446 m、最终 0 m，最大单步增量 0.022137 m。
- **支持或推翻**：推翻“该 fixture 证明 envelope 转换错误”的归因；支持修复配对投影。
  只证明最终收敛，**未证明每帧单调降低穿线深度**。
- **未知与下一步**：真实城镇感知质量与完整候选拒绝原因仍需进一步记录，不能将
  `no_executable_path` 统一当作无谓停车并提高速度地板。

另修两个直接影响规划可靠性的缺陷：

1. `_path_body_collision` 在无网格或无足迹采样时统一返回三元组，避免短前向路径
   评分因解包异常中断。
2. strict 模式所有候选被拒后，自动生成的 hold 路径也必须重新通过同一约束层，
   且只能读取已接受的感知参考。缺 drivable 证据不能通过“走廊没障碍”重新获得路径。
   非 strict 的旧兜底策略保留。

## 4. 走廊必要几何与真实接线

- **原假设**：原三态函数已经充分排除近场墙、断开的窄通道与未知证据。
- **是否真正激活**：旧网格比例跳过区会漏掉鼻外 x=2.75 m 的墙；两个 2.5 m 通道
  相邻只重叠 0.5 m 时仍能判可行；wrapper 还读取生产 Scene 并不存在的
  `speed_mps/closest_obs_m/bev_age_s` 属性。
- **可复现输入**：40/60/120 行网格中的鼻外墙、车辆宽度内缩后不连通的区间、
  近端突然横移、陈旧或 NaN 年龄、缺路面/观测证据、请求超出网格视野；真实 Scene
  通过 monitor.evaluate 传入速度与 BEV 年龄。
- **实测结果**：从权威前保险杠位置逐行扫描，以车身中心可行区判断连通，逐行限制
  横移预算；缺证据、非法数值、数组形状不符或视野不足返回 UNKNOWN。wrapper 显式
  接收当前 ego speed、计算的障碍距离与 snapshot BEV 年龄。
- **支持或推翻**：支持用连续几何替代空格子计数；推翻“存在原语且 stub 单测通过就代表
  生产链已正确接线”。
- **未知与下一步**：`FEASIBLE` **仅是几何必要条件**。未给出经过动态验证的候选轨迹，
  不证明车道合法性、转向响应或制动能力。`min_turn_radius=5.5` 与
  `max_lateral_speed=2.0` 仍未标定，`BEAMNG_CORRIDOR_FEASIBILITY` 仍默认关闭。

## 5. 感知性能与命令追溯

- **原假设**：object 异步能消除主要延迟；发布年龄可代表控制消费时的输入年龄。
- **是否真正激活**：strict semantic 仍同步，未发现重复 UNet predict。发现标线提取
  逐像素重建相机姿态，可复用已有批投影。旧消费年龄等于 cmd_t-publish_t，漏掉推理
  延迟；缓存/异步结果还可能被标为新提交任务的来源。
- **可复现输入**：固定 403×536 合成 mask、相机、位姿，原 HEAD 与修改版交错运行
  3 轮 × 10 对；受控延迟发布、缓存复用、异步采用后同 tick 新提交；一段无子步的
  慢 tick 与单调时钟下的命令发送。
- **实测结果**：同两条 marking 的 world/pixels/kind/color/confidence 精确一致。
  完整提取器 p50 **6.402 → 2.589 ms**，p95 **7.710 → 3.134 ms**；轮内中位范围
  分别为 6.321–6.490 ms 和 2.546–2.710 ms。消费 age 以 source_t 为起点，另列
  publish_age_s；缓存/采用结果保留自己的来源。所有驾驶 tick 和子步共用发送回执。
- **支持或推翻**：支持该受控标线提取热点的等价降耗；不能由此推断 semantic 总耗时
  改善同样比例，更不能宣称整栈 p95 达到 150 ms。
- **未知与下一步**：新增 `perception_ms/semantic_ms/segmentation_ms` 分解，为未来
  Tech 跑次定位真正阻塞项；相机内部 RPC/读回/重试仍未拆分。

时间口径必须区分：

- head trace 与 `cmd_t` 仍使用 `time.time()`，标为 `wall_time`，不是单调时钟。
- `source_t` 表示 `acquire_return` 边界；provider 未给曝光时间，不能称传感器曝光真值。
- deadline 使用独立单调钟；`cmd_gap_s` 是发送回执间隔，`cmd_send_ms` 是发送调用耗时，
  `watchdog_gap_s` 是发送前检查值。发送失败不刷新回执。
- 默认关闭的 Python deadline 响应现在也在主 tick 发送前执行，不依赖是否有余裕跑子步。
  **Python 阻塞期间仍依靠既有游戏侧看门狗**，不声称本轮实现独立的 1.5 s 控制线程。
- 子步实际发送内容、序号和来源年龄记录在 `substep_commands`。shortfall 使用该帧实际
  墙钟区间与该帧执行数量，不把累计子步数当本帧数量。
- 性能脚本优先使用实际命令回执；只有旧 `t` 的数据明确标为帧间隔 proxy。

## 6. 统计与验收门禁

- **原假设**：边界覆盖脚本正确识别 runtime 的 checked 标志，A–H gate 会核验要求列。
- **是否真正激活**：runtime 写 0/1，脚本用 `is True`；最长双缺错误取两侧最长单缺；
  gate 只看调用方 metrics，缺 frames 仍可 PASS，字符串 `"False"` 可被当真。
- **可复现输入**：`road_checked=[1,0,true,false]`，两侧交错缺失，多个长短不同的双缺段；
  A–H 仅 metrics、D 缺 throttle、无效或部分数值、字符串 flag、F/E 明确记录不可用状态。
- **实测结果**：识别真实二进制编码；双缺按同一连续段的实际时间与速度计算。门禁复用
  PASS/FAIL/UNKNOWN，证据不足不能放行，明确的测量违规保持 FAIL。预期不可用与数值
  读数分开统计；不从 null 或缺字段编造安全状态。
- **支持或推翻**：支持先核对证据覆盖再使用结论。只读重算历史 131 跑次发现
  **26,202 帧全部缺 road_checked 字段**，所以该历史集合的 0/131 是未插桩；类型缺陷
  在这组历史数据中没有激活，不能把此次修复当作覆盖率已经提高。
  修正“双边同时缺失”的口径后，同一历史集合最长双缺段的跨跑次中位数为
  **125 帧**、最大 **225 帧**，不是旧报告的 164/226 帧；左右覆盖率中位数仍为
  3.03%/2.24%，达到既有侧向声明覆盖门槛的仍只有 12/131 跑次。
- **未知与下一步**：A–H 门禁仍核验显式输入的指标，不替代独立真值；H 中无解释的 TTC
  null 保持 UNKNOWN。必须获得真实运行记录后才能放行。

## 八项必附

### 1. Commit / config / run 与控制权

本轮无 commit、无 push；所有代码基于 `21c32ab`，在上述工作分支保留未提交修改。
若提交，必须按 AGENTS 的模块边界拆分，不能把整个工作树一次提交。

`BEAMNG_ASYNC_HEADS`、`BEAMNG_CORRIDOR_FEASIBILITY`、`BEAMNG_ROAD_SURFACE_GATE`、
`BEAMNG_CTRL_WATCHDOG`、`BEAMNG_LONG_PLAN`、`BEAMNG_DRIVE_MODES`、
`BEAMNG_STEER_BLEND` 的默认值未改；控制子步默认配置未改。

本轮没有向游戏发送控制。只读进程检查未发现 BeamNG 游戏进程；离线控制测试使用假连接。
此证据只覆盖“本轮没有控制权争用”，不证明未来 Tech 独占检查一定正确。

### 2. 测量覆盖、缺列与 UNKNOWN

历史集合 131 跑次 / 26,202 帧，`road_checked` 缺失 26,202 帧。
真实新增跑次为 0，因此新增最终速度、source trace、性能分段等真实覆盖均未验证。
新脚本逐列报告 measured/unavailable/unknown 与 missing-column，不回填旧日志。

### 3. 强制刷新与消费

既有 `test_schedule_floor.py` 覆盖超预算且过保活界、冷启动及缓存场景；新增测试区分
原结果来源与当前调度尝试，并验证消费 age 包含推理延迟。完整回归中统一执行。
这些是确定性离线证据，不是新增 Tech 强制刷新或车辆执行证据。

### 4. 几何正反例与末端控制

正例：固定世界车道中心、pair/envelope 精确复制、受验证的 strict hold、连通且可达
的已观测路面带。反例：鼻外墙、车宽区间断连、近端不可进入、未知/陈旧证据、无路面
strict 兜底、NaN 和坏形状。末端控制测试见 `test_final_target.py`、
`test_control_watchdog.py` 及本轮驾驶循环集成测试。

### 5. 路面门真阳性、漏检与误停

离线测试证实：超过停车时长能停车；未确认的单帧 ON 不解除；缺失/陈旧读数不重置恢复；
健康 ON 输入不触发丢失停车。**Tech 真阳性、漏检率和误停率本轮未测**。
既有前视相机看不到车身脚下的覆盖缺口未解决，不接入已被历史数据否证的足迹覆盖指标。

### 6. 性能与 deadline

本轮仅有上述合成标线提取微基准，不能与历史 301 ms semantic 或 409.8 ms tick p95
直接相减。新计时各层嵌套，不相加重复计算。真实整栈 deadline 违反率本轮未知。

### 7. 碰撞、压线、出铺装与停车

新增实际驾驶 0 跑次，碰撞、压线、出铺装及分类停车时长均未测。历史
`no_executable_path=32.7%` / `unjustified=0.9%` 不因代码已改就成为新运行成绩。
恢复 fixture 最终压线深度为 0，但峰值有增加，不称逐帧单调恢复。

### 8. 验证状态与保留限制

| 验证 | 最终结果 |
| --- | --- |
| 完整 `pytest tests/ -o addopts='' -q` | **1913 passed in 42.77s**，0 失败 |
| `scripts/m5_offline_validate.py` | **RESULT: ALL PASS** |
| 生产驾驶循环假连接集成测试 | **10 passed**；最终收紧 DQN 激活断言后再次通过，3.22s |
| `git diff --check` | 通过；README 只有既有 CRLF/LF 规范化提示 |
| 本轮新增用例数 | 相对基线净增 **328** 个 |
| Tech 实际驾驶与 A–H 验收 | **未执行，不宣称通过** |

`test_fsd_drive_pipeline.py` 直接执行生产 `FSDriveSession.run`，使用固定感知输入、
可控裁决和假连接，真正运行仲裁、速度整形、净空检查、爬坡/终点分支、最终发送与
JSON 落盘。验证包括：默认低限速、LONG_PLAN、DQN 主导的 0.6 m/s 上限、已有油门
后硬停、压线与净空共同输入、无子步余裕的 deadline 制动、终点对齐油门不得覆盖硬停、
实际激活爬坡辅助后下一帧停车，以及保护命令回执不被后续主命令覆盖。

这些测试不加载模型、不连接游戏、不写项目 logs；fake 控制输入不是车辆动力学验证。
独立差异复核提出的“保护发送回执遗漏”已修复：单独记录 `protective_commands`，
性能脚本将其纳入实际命令间隔。未发现其他本轮新增的阻断性控制问题。

尚未验收或仍保留的缺口：

1. Tech A–H 场景、成对 A/B、真实碰撞/横向违规/停车原因和控制频率。
2. 转弯半径、横移速度、路面时长与近场覆盖标定；新几何开关仍关闭。
3. 既有 async 生命周期：reset 后在途旧任务、缓存清理与 close 的 worker 回收未在本轮重构。
4. 既有同步 `head_age_s` 仍从完成时刻起算；本轮修正的是 source trace 与消费年龄，
   不宣称生产安全新鲜度已经全面切换到曝光时间。
5. async range 调度与保活路径统一、所有候选拒绝原因遥测尚未补齐。
6. semantic 推理异常沿用既有回退；模型质量、错误输出健康判据与部署选择没有重新验证。

## 可复现命令

```pwsh
.venv\Scripts\python.exe -m pytest tests/ -o addopts='' -q
.venv\Scripts\python.exe scripts\m5_offline_validate.py
$runs = @(Get-ChildItem logs\fsd_benchmark\town_*.json | Select-Object -ExpandProperty FullName)
.venv\Scripts\python.exe scripts\m5_boundary_coverage.py @runs
.venv\Scripts\python.exe scripts\m5_perf_profile.py @runs
git diff --check
```

历史诊断命令默认只读并输出到终端；未指定输出文件，不修改原始运行日志。
