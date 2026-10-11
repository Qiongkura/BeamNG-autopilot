# T17 开发、离线门与 Tech 诊断

日期：2026-10-09。目标继承 T16：推进可验证的研究系统；本批先完成代码，
再过离线门，最后实测。不把基础设施完成、参数更新或低速零损伤写成论文安全结论。

## 1. 代码范围

- `vision/lanes.py`：新增 component ROI 开关，组件局部像素顺序、抽样、投影、
  分类和 GT 丢弃诊断与全图扫描相同。`segmentation.py` 的学习式白线提取接入；
  经典 CV 入口、黄色来源策略和全部形状阈值保持原样。
- `runtime.py`／Tech provider：逐样本 RangeSample metadata；拆分锁等待、
  LiDAR poll、解包/有限值过滤、Lua poll、CPU 准备、聚类/融合，保留各源错误。
- `fsd_stack.py`：同步扫描和异步 fetch 的源代理时间提前至调用开始，复用样本
  保留 metadata；记录 age_basis=poll_start_proxy，真实 sensor_capture_t=None。
- `safety_monitor.py`：默认关闭的安全几何拷贝，记录实际 scene、感知边界、
  输入或 hold 路径、生产膨胀尺寸及容差；捕获失败仅影响诊断，不影响安全规则。
- `geometry_probe.py`／driver／回放入口：最终指令旁保存安全评估拷贝；分别
  回放物理车体/最终路径与生产安全检查，检查生产 verdict 是否与回放一致。
  没有检查记录或没有边界覆盖时保留 UNKNOWN，safe_to_execute 始终不被回放认证。
- 新增 16 个确定性回归例子：6 个 ROI 等价、1 个逐样本 timing/错误隔离、
  4 个 range 时间/复用、5 个安全几何/不同路径/hold/UNKNOWN/关闭开关。

开发计划见 `T17_PERCEPTION_CADENCE_GEOMETRY_PLAN_20261009.md`；实测预注册见
`T17_TECH_DIAGNOSTIC_CONFIG_20261009.json`。

## 2. 结论链

### A. 学习式标线提取

原假设：反复扫描整幅 label 图可消除 → 激活：学习式白标线入口明确传
component_roi=True，经典入口仍走默认 → 可复现输入：不同颜色/地面高度、
斜线、贴边、邻近组件、宽 blob、短组件及 GT 丢弃计数 → 离线结果：相关 193
测试通过，ROI 与原算法 pixels/world 数组逐位相等，类型/置信度/诊断一致 →
支持计算范围可缩小且这些例子的输出不变 → 真正的总 markings 加速尚待实测，
不能用算法复杂度代替时延结果。

### B. 距离感知计时和源时间

原假设：处理完成时间会把 IO/排队/处理延迟抹掉 → 激活：同步、every-n
强制刷新和异步 fetch 均在调用开始记录代理时间 → 可复现输入：假时钟注入
0.7s 扫描、0.5s IO + 0.7s 排队；LiDAR 失败但 Lua 成功 → 离线结果：新样本
age 保留 0.7s／1.2s；逐样本阶段时间和 LiDAR 错误不被 Lua 成功覆盖 → 支持
原源时间记录存在低估，修正后包含该延迟 → 真实传感器 capture time 仍 UNKNOWN，
新的代理时间不是对实际数据新鲜度的完整证明；实测 IO/CPU 瓶颈尚待分解。

### C. 安全几何统计口径

原假设：T16 的 production current/planned 与 physical/final-path 数量不同，
可能来自 margin 和路径阶段 → 激活：probe 开关同时启用生产检查点的独立拷贝 →
可复现输入：物理车体未跨线但膨胀车体跨线；评估路径跨线但最终控制路径不跨；
envelope、hold、无边界及篡改 verdict → 离线结果：正反例区分上述输入，回放
匹配生产 predicate，篡改被检出，无边界虽 predicate=False 但 evidence=UNKNOWN →
支持记录同一套输入才能检查实现是否一致 → T16 没有这套拷贝，不能补猜旧帧；
生产 predicate 匹配仍不等于独立真值或执行安全。

## 3. 验证记录（整批代码后）

- 定向回归：`193 passed in 11.78s`。首次定向门检出 JSON list 未恢复 ndarray
  及测试 envelope 缺 age_s，已在离线阶段修好；修复后整批收齐才进入全量门。
- `git diff --check` rc=0；CRLF 提示不是失败。没有提交/暂存/推送。
- 当前完整源码快照：`9b10da96f91b5613e0e840f328bdf5c4392a2f9fdcf8b8df6fb6ccb054d89582`，
  371 文件，`logs/experiments/t17_codefreeze_20261009.json`。范围为库/Tech/scripts，
  不包含 docs/tests/weights；此范围限制与 T16 一致。
- 全量门：`gate_20261009_202029.log`，**3351 passed in 533.15s**；深度离线
  RESULT: ALL PASS，入口 rc=0。门后完整源码快照仍相同，另存
  `t17_codefreeze_after_gate_20261009.json`。
- Tech 同步及现有全局异步开关诊断依次完成，均 rc=0、qualified=true。实验资格
  不是安全资格。代码均冻结在同一摘要，未与门/训练并发，未在测量中改代码。

## 4. 八项审计

### 1）commit / config / run 与控制权

HEAD `aedd2617e09e51bfe661a9b49357cda8e2b97ae7` + dirty workspace，源码为上节
371 文件快照。两轮完整分割模型摘要均为
`19f54e5363e86323834a9f885fe930a0e7996029d4b7cef8633e3edf1ef3a36e`。Tech、italy、
strict、sensor、带 goal、25s、wall 150s、production_alignment、geometry probe。

| 项 | 同步 | 现有全局异步开关 |
|---|---|---|
| run | t17_geometry_20261009 | t17_geometry_async_20261009 |
| session config hash | daf9be78fc183a5e1b3cd179ac5f1989e1f067f03f010188fc2dd3329a8e585f | c66066522f74ce68bd47f32e246e3df75505f21772fa95c2e0755abba653c39e |
| 实际起点 XYZ | 779.43569,734.55967,179.75442 | 779.44145,734.58660,179.75583 |
| 实际 heading rad | −0.22994 | −0.22986 |
| 环境因子 | ASYNC 默认 OFF | BEAMNG_ASYNC_HEADS=1 |

异步预注册另见 `T17_TECH_ASYNC_DIAGNOSTIC_CONFIG_20261009.json`；session hash
覆盖入口参数，环境因子由该文件及实际 worker/state 激活证明。该开关影响 range
与 eligible 非语义头，strict semantic 仍同步，**不是 range-only 因子**。实际位姿
是感知放置的结果；每臂只有一次，运动轨迹/实际帧不配对，不能宣称独立因果加速。
其余 E2E/BC/DQN 模型有加载日志，本轮没有逐一保存其权重摘要，复现范围受限。

开发前、全量门后、两次实验间和收尾后相关进程枚举为空；测量中仅本次 driver
及其 Tech 子树。本任务未并发回归/训练，不能据此保证整台电脑无其他负载。内部
及外层 owned cleanup 均 closed=true、unverified/still_running 为空。内部有终止
调用 failed PID（同步 4 个、异步 3 个），记录原样保留，不冒称全部亲自杀死；
最终验证和外层枚举均无残留。前后源码一致，两套原始目录保留，没有改默认开关。

### 2）覆盖、缺列和 UNKNOWN

同步 30 帧，t=1.290..25.615，曝光 24.325s；drive 27.031s，整体 128.047s。
异步 40 帧，t=0.839..25.660，曝光 24.821s；drive 27.438s，整体 111.688s。
两臂均沿用要求 span≥24s 等资格门，qualified=true，没有降低曝光标准。
12 项基本控制/位姿/新增诊断字段缺项均 0，geometry/timing/send joins 70/70。

| UNKNOWN 项 | 同步 | 异步 |
|---|---:|---:|
| body_cov（当前车体铺装覆盖） | 30/30 | 40/40 |
| physical 当前车体不能给出完整负结论 | 26/30 | 36/40 |
| 膨胀 safety 当前车体不能给出完整负结论 | 25/30 | 25/40 |
| sensor_capture_t | 30/30 | 40/40 |
| 独立铺装/真实漆线真值、执行 ACK | 未测 | 未测 |

当前 crossing witness 为阳性时允许部分边界覆盖；predicate=False 且覆盖不足
仍是 UNKNOWN。这里的 physical 是**物理尺寸车体对感知边界**，不是独立真实漆线
真值。真正 sensor capture time 未得，源代理时间不能代替它。

### 3）强制刷新与实际消费

假时钟强制扫描和慢 fetch 例子通过。实测 keep-alive OFF，强制 requests/新源
强制 command joins 均 0：分支未激活，不能计通过。普通同步 range scanned=30/30；
异步 range async_adopted=40/40，两臂 poll-start 逐帧严格递增。异步 range worker
提交/完成从 19/18 增至 58/57，timeouts/errors=0；该 metadata 来自被融合的样本。
object worker 的累计 timeout=1（首帧和末帧都为 1，测量区间没有新增），原样保留。
没有执行 ACK，不能把源消费和命令发送当执行证明。

### 4）几何正反例及最终控制链

16 个新增回归覆盖 ROI、源时间、逐源错误、margin、不同路径、envelope/hold、
无边界、篡改和默认关闭。两个真实 run 的 production safety predicates 回放
**70/70 匹配，mismatch=0**；每帧独立拷贝实际评估 scene/path/尺寸/容差。

| 项 | 同步 | 异步 |
|---|---:|---:|
| 物理尺寸当前车体 witness | 4 | 4 |
| 膨胀 production current predicate | 5 | 15 |
| production 评估路径 planned predicate | 7 | 6 |
| 最终控制路径 planned witness | 0 | 0 |
| 指令 seq/time/throttle/brake/steer joins | 30/30 | 40/40 |
| 真正 hold 帧 | 1 | 0 |

上述是不同输入/阶段的数量，不混为一项。发送收据匹配没有独立 actuator ACK；
最终路径零 witness 不等于执行安全，也不否定安全评估路径的 crossing。旧 T16
缺这套检查点输入，不能用本次输入补猜旧帧。

### 5）路面门真阳性、漏检与误停

未改路面策略。独立 TP/FN/误停分母 UNKNOWN；不能由 reason 推导误停，也不能
把 road_off/edge_over 当独立铺装真值。body_cov 70/70 UNKNOWN 反映车体盲区尚未补齐。

### 6）性能分解及 deadline

单位 ms；每臂一次 integrated 诊断，n=30/40，以下列缺项 0，p50/p95 不做单次
模型属性，也不将两臂中较小值当质量门的重复测量。测量与 CPU 回归/训练串行。

| 项（p50 / p95） | 同步 | 异步 |
|---|---:|---:|
| tick wall | 615.80 / 706.54 | 407.30 / 515.21 |
| range 主循环阶段 | 309.50 / 342.69 | 114.20 / 125.13 |
| sample fetch | 85.89 / 94.87 | 93.51 / 103.22 |
| sample CPU process | 205.82 / 238.15 | 289.05 / 346.62 |
| prepare | 76.62 / 91.51 | 136.89 / 205.06 |
| fusion | 127.69 / 157.61 | 150.12 / 194.75 |
| LiDAR poll | 8.56 / 19.29 | 11.88 / 18.99 |
| unpack | 21.97 / 25.36 | 21.80 / 25.66 |
| Lua poll | 54.92 / 59.08 | 60.33 / 65.96 |
| semantic prediction | 28.45 / 31.12 | 28.23 / 52.35 |
| semantic markings | 95.23 / 173.13 | 149.01 / 163.18 |
| 安全检查点拷贝 | 0.0358 / 0.0431 | 0.0350 / 0.0661 |
| 发送点 geometry probe | 0 / 16 | 15 / 16 |

锁等待 p50=0.0055/0.0040ms，源点数 p50=416121/420633.5。range proxy age
p50/p95=0.308/0.340s、0.757/0.930s（async max=0.976s）；异步移出 CPU 工作
同时增加源延迟，不能只报主循环降时。主循环 deadline **30/30、33/40 违反**。
control interval p50=0.813/0.602s，max=1.265/0.844s。E2E 首轮 cold spike
408.3/130.6ms 原样保留。geometry timer 的 0/16ms 分布含计时粒度影响。
与 T16 的 markings 中位数差异仅描述性，不宣称 ROI 单独导致总时延变化。

### 7）碰撞、压线、铺装及停车时长

damage 数值 30/30、40/40，max-min=0；这不是完整碰撞次数真值。两臂对感知边界
的物理车体 crossing=4、painted flag=4，strict 安全验收不通过；真实漆线独立
真值仍 UNKNOWN。路程=10.013/2.022m，max speed=3.009/1.869m/s；reverse_active=0，
signed min=−0.066/−0.011m/s（漂移原样保留）。road_off>0 各 4 帧，max=0.320/0.775m；
edge_over=0，不证明没有离开铺装。

v<0.5m/s 的左采样相邻区间停车估算：同步 **20.320/24.325s**，异步
**23.599/24.821s**。首帧前/末帧后未观测尾段不计，不把 reason 当独立原因真值。

| 当时 reason 对应停车时长（s） | 同步 | 异步 |
|---|---:|---:|
| no drivable path | 6.157 | 15.542 |
| obstacle contact risk | 0.818 | 1.254 |
| planned body crossing | 3.146 | 2.712 |
| obstacle very close | 5.808 | 2.747 |
| path hold creep | 0.802 | 0 |
| lane boundary recovery | 1.662 | 0 |
| planned boundary ahead | 0.968 | 0 |
| obstacle stopping distance | 0 | 1.344 |
| UNKNOWN "-" | 0.959 | 0 |

### 8）通过、失败和尚未测试

通过：193 定向、3351 全量及深度离线；两臂实验资格、owned cleanup、源码一致、
生产几何回放和发送 joins。失败：strict 的感知边界 crossing、主循环 deadline；
异步没有证明驾驶质量改善，不能根据主循环更快改生产默认。未测：强制刷新真实
消费、独立路面/漆线真值、执行 ACK、真实 GPU 中断恢复、学习收益、多个配对重复。
本轮没有训练/晋级，更好模型仍未证明；不新增人工标注任务。

## 5. 实测后的结论链

距离瓶颈假设 → timing 70/70 真激活，普通新源逐帧更新；async worker 真启用 →
上述固定 request/source/model/config 和单独环境因子 → 同步约 86ms fetch、206ms
CPU；异步主循环 range 114ms，但 CPU 289ms、源 age 0.757s，deadline 33/40 →
支持该条件下 IO/CPU 可被区分，异步可移出部分主循环成本 → 不支持默认晋级，
单次且轨迹不同不能建立 range-only 因果或驾驶安全收益；下一批优化 Tech 点云
CPU 的重复处理及学习式标线后处理，并补实际车体覆盖/独立真值。

几何口径假设 → probe 保存实际安全输入和发送点输入 → 两个 run 70 帧回放 →
production 5/7 与 15/6 predicates 均匹配；physical/final-path 各为 4/0 witnesses →
支持不同口径在新运行中能被解释，未发现这 70 帧 predicate 实现不一致 → 不追认
缺输入的 T16 帧；缺覆盖/真值/执行 ACK 不转成安全通过。

证据：两个 run 根的 `geometry_audit.json`、`runtime_audit_v2.json`，及 pose_00
的 request/result/session/telemetry/driver.log。v1 派生 audit 用错 outer cleanup
字段，产生 null；v2 修为 result.outer_cleanup。v1 文件保留，不回写原始运行。
所有实验结束后相关进程为空，最终源码复核仍为上节摘要（371 文件、complete=true），
另存 `logs/experiments/t17_codefreeze_after_tech_20261009.json`。最后仅补 README/docs
实测说明，驾驶/训练源码未改；`git diff --check` rc=0。
