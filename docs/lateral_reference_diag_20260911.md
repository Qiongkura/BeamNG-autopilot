# 归并开启后城镇实车压线/出路面：三个假设的检验（2026-09-11）

同一配置：`--attach --runtime tech --scenarios town`，`BEAMNG_DASHED_RECOVERY=1`，
v13b。本文只记录**被数据支持**的结论，并明确列出已被否证的假设，避免下一轮重走。

## 结论（当前证据支持的唯一一条）

**被选中的「本车道」横向参考自身在帧间跳变**：sensor 参考相对自车的横向位置
相邻 tick 最大跳 **2.20 m**（p50 0.230 / p90 0.383 / p99 2.15；**5.3% 的 tick 跳
> 1.0 m**），而 envelope 参考稳定得多（p50 0.080 / max 0.41）。归并把 sensor 参考的
可用率从 ~2% 提到 15–22%，于是这些跳变帧同比放大，表现为 `off` 5–19、`crossR` 0–6。
关掉归并时可用率低，问题被掩盖。

即：**不是「参考选错对象」，也不是「修正器交接台阶」，而是选中的参考本身不连续。**

## 已否证的假设（不要再实现）

### 假设 1：漆线修正器 owner 交接时 1.6 m 瞬断（否证）

`town_1789093299`（off=19、crossR=6）21 帧出路面**全部**是 `lane_sel=sensor` +
`plc_active=0` + `plc_shift=0.0`；首次出路面时车相对漆线的读数前后未变
（`line_lat` 在 t=7.059 与 7.285 都是 +0.434 m），而 `lane_dev_m` 一帧从 0.121 跳到
2.113 m。修正器在此前若干帧就已关闭，**没有 1.6 m 修正量可丢**。

### 假设 2：监视器与规划器用了不同的横向参考（否证）

本分支自带单一参考策略（`planning/lateral_ref.py`：`lane_ref` → `envelope` →
legacy route），监视器确实消费 `scene.lane_ref`。逐次调用探针（`SafetyMonitor.evaluate`
包装，545 条记录）显示：

* 「车体压线」判决 **84 条**，其中 **61 条**监视器用的就是规划器同一个 sensor 参考
  （`has_lane_ref=True`、`ref_src=sensor`、`lane_sel=sensor`、左右硬边界 `bnd=TrueTrue`），
  只有 23 条走了 envelope 兜底（那些是 `lane_sel=perception-unavailable` 的帧）。
* 参考来源总体分布 `envelope` 219 / `none` 218 / `sensor` 108。

也就是说：压线判决大多**不是**参考选错对象造成的，而是同一个 sensor 参考说「车体确实
在自己的感知车道外」。

## 支撑数据

| 量 | 值 |
| --- | --- |
| sensor 参考帧间 \|Δlat\| | p50 0.230 / p90 0.383 / p99 2.15 / max **2.20** m |
| sensor 跳 >1.0 m 的 tick 占比 | **5.3%**（>0.5 m：7.9%） |
| envelope 参考帧间 \|Δlat\| | p50 0.080 / max 0.41 m |
| 压线判决用的参考 | 61/84 与规划器相同（sensor），23/84 envelope 兜底 |
| 压线判决的 `lane_dev_m` | p50 1.47 / max 3.73 m |
| 典型单帧跳变 | t≈7.29：`line_lat` +0.434 → t≈7.98：−1.962 m（0.7 s 内 2.4 m） |

探针均为**纯记录**（`sitecustomize` 包装，不改行为）：
`logs/_owner_probe.jsonl`（545 条）、`logs/_latref_probe.jsonl`（662 条）。

## 下一步（在参考选择/融合层，不要在控制侧）

1. **给被选中的本车道参考加帧间横向速率限制**（envelope 的 0.41 m/帧 就是「稳定」的
   量级参考；2.2 m/帧 显然不是换道）。位置要求是「选择之后、消费之前」，让规划器与
   安全判据消费**同一条已限幅**的参考：加在选择/融合层（`lane/reference.py` /
   `lane/fusion.py`）或紧接其后的 `fsd_stack` 都可以；加在**消费侧**（fsd_drive 的
   Scene 或修正器）则只会重新制造假设 2 的不一致。
2. 判据：连续 tick 参考位移超过门限时，判为「感知本车道选择不稳定」，用它上一帧的
   一致参考（或降级），而不是让车跟着 2 m 台阶走。
3. 复验口径不变：同日 A/B（`BEAMNG_DASHED_RECOVERY` 开/关）看
   `crossC/crossR/off` 与 `lane_dev_m` 的跳变帧数；离线先跑
   `pytest tests/` + `scripts/m5_offline_validate.py`。

## 实现与实车结果（2026-09-11 本轮）

已实现第 1 条，落在**选择之后、消费之前**（规划器与安全 Scene 共用同一条参考）：
`planning/lateral_ref.py` 新增纯函数 `near_lat` / `limit_reference_slew`
（门限 `LANE_REF_SLEW_MAX_M = 0.8 m`，保持上限 `LANE_REF_SLEW_HOLD_MAX_S = 2.0 s`，
超时后接受新参考，避免真换道被永久挡住），`fsd_stack` 在 `lane_ref_out.center`
之后、写入 `out.lane_ref` 之前调用；`reset_temporal()` 与 `__init__` 都清理该状态。
单测 `tests/test_lateral_ref_slew.py`（7 项）。离线 `pytest 642 passed` +
`m5_offline_validate ALL PASS`。

**实车结论：限幅按设计生效，但没有收益（5 臂/组判定，保持默认关闭）。**

判定用的是新落地的降方差协议 `scripts/m5_live_ab.py`：同一 game 会话内连续跑、
A/B 交错（漂移由两组共担）、固定场景起点与 goal、逐臂记录 stale 帧数与被监视文件的
哈希（中途换代码会使整套数据作废）。命令：

```
.venv\Scripts\python.exe scripts\m5_live_ab.py --factor BEAMNG_LANE_REF_SLEW --arms 5 --pin BEAMNG_DASHED_RECOVERY=1
```

结果（2026-09-11，revision `5c23981`，5 臂/组，全部同一会话，无中途改代码）：

| 条件 | lane p50 | stall p50 (range) | dist p50 | off 各臂 | crossC | crossR |
| --- | --- | --- | --- | --- | --- | --- |
| `BEAMNG_LANE_REF_SLEW=0` | 15% | **138** (135–169) | 48.3 | [0,8,5,5,7] | [0,5,0,0,0] | [0,3,0,0,0] |
| `BEAMNG_LANE_REF_SLEW=1` | 13% | **180** (151–188) | 32.4 | [0,7,11,0,2] | [0,0,6,0,0] | [0,0,0,0,0] |

结论：

* 可用率无差别（15% vs 13%，区间重叠）。
* `stall`（138 → 180）与 `dist`（48.3 → 32.4）**开启更差**。
* `off`/`crossC` 在**两组都会出现**（关：4/5 臂有 off；开：3/5 臂有 off）——所以这条
  限幅既不是这些压线的解药，也不是它的原因。
* 限幅触发率此前实测 7/224 = 3.1%（与 >1.0 m/帧跳变率 5.3% 同量级）→ 不是静默失效，
  是**真的没帮助**。

因此限幅保留实现、**默认关闭**（`BEAMNG_LANE_REF_SLEW=1` 开启）。`stall` 仍 135–190
（基线 9/5–9/7 为 0–83），`off`/`crossC` 仍非零——**验收未通过**，且已排除「限幅」
这条路。

**下一步**：压线/出路面在限幅关闭时同样出现，说明病在别处；用
`scripts/m5_live_ab.py` 把候选改动按 5 臂/组过一遍，再用 `--diagnose` 看是哪些帧，
而不是继续做 1–2 臂的 before/after。

## 本轮已落地且与上述无关的修复

`4fa7c2a`：stale-sensor 按模态判定（commit）。实车 `stale sensor` 75 → 0/1 帧；
离线 `pytest 635 passed` + `m5_offline_validate ALL PASS`。

---

# 限幅关闭时 `off`/`crossC` 的定位（2026-09-11 11:45，revision `202281d`）

车辆已让给并行会话（见 `docs/HANDOFF_20260911.md` 末尾的回复），本轮全部用**已录制的
episode / telemetry**做离线诊断，不起实车。

## 1. 压线帧的形态（最近 8 趟 town 共 1790 帧，`road_off`/`body_cross_*` 38 帧）

38 帧**全部**满足：`lane_sel=sensor`、`lane_paired=1`、`source=none`、
`level=minimal_risk`、`speed≈0`。`reason` 里 35 帧是 `no drivable path`，
3 帧是 `current vehicle body crosses lane boundary`。

即：**压线是「停住 + 存在成对车道」的伴生现象**，不是独立的横向控制失败。

## 2. 主因：有车道却产不出路径（占全部帧 12.5%）

`reason=no drivable path` 共 **223/1790 帧**，拆成两类：

| 子类 | 帧数 | 含义 |
| --- | --- | --- |
| 无感知车道（`perception-unavailable`） | 158 | 严格模式**按设计**停车（铁律） |
| **有配对车道却仍无路径** | **65** | 缺陷：38 个压线帧全在这一类 |

65 帧里 `lane_reject` 为空（没有「车道被拒」的记录）、`plan_raw=0.0`（采样器没给出
路径）、`fwd_clear≈11.8 m`、障碍数 0——**前方畅通、有车道，却产不出路径**。
只有 14/65 帧在下一帧恢复出路径，说明这里是**锁住**而不是瞬时抖动。

车体几何：26/65 帧车体确实越过某条感知边界（`latL=-0.65 latR=-0.09`），
39/65 帧**右侧边界缺失**（`body_lat_right=None`），判定只能打在剩下的那一条上。
而同一批帧相对**漆画线**的偏移 `line_lat` 中位只有 -0.38～-1.23 m——车大致在应在的
位置。**偏的是感知边界，不是车。**

## 3. 离线漏斗（同 revision，`m5_lane_continuity.py --diagnose`，v13b）

| | ep 11:33 | ep 11:35 |
| --- | --- | --- |
| 配对本车道 | 23.2% | 29.5% |
| **离线 `lane_sel=sensor`** | **83.5%** | **57.6%** |
| 单边界帧 | 161 | 155 |
| 中心横向偏差 | -0.19 m | +0.03 m |
| 未配对首因 | `one_side_filtered_by_candidate_gate` 142 | 143 |
| 门控最大丢弃 | `unknown:kind` 455 (GT 50%)、`dashed:align` 143 (GT 66%) | 同量级 |

**最大的单点矛盾：离线 `lane_sel=sensor` 83.5% vs 实车 telemetry 里同批帧约 15%。**
离线重放能通过可用性门、实车不能——「有车道却产不出路径」就活在这个差距里。
（旧文档记过 45% vs 11% 的同类差距，现在差距更大。）

## 4. 下一个候选改动（按证据排序）

1. **解释并收敛离线/实车 `lane_sel=sensor` 差距（83.5% vs ~15%）**。这是单点最大
   矛盾，且第一遍排查**不需要车辆**（离线重放 + 现有 telemetry 对照即可），
   直接命中「有车道却无路径 → 停车 → 被判定压线」这条主链。
2. 次级：`unknown:kind` 455 次丢弃、GT 50%——需先确认它们是车道边界还是斑马线/补丁漆
   （此前 `*:side` 那批 0% GT 是正确拒绝；这批不同，不能直接照搬）。
3. **不要**再动参考限幅（已 5 臂判定无收益，见上文）。

## 5. 车辆协调

已按 `docs/HANDOFF_20260911.md` 的约定让出车辆；本轮未起实车。取证时我试图用
`CUDA_VISIBLE_DEVICES=""` 强制 CPU 以免抢 GPU，但 `Segmenter` 仍报了 `device=cuda`
（环境变量未生效）——后续离线任务需要显式改 `Segmenter(device="cpu")` 才能真的不抢。

---

vehicle free 11:45 (session cce01654, per the agreed protocol in this doc)
- used one bounded window: town_1789098126, dist 30.7 -> 52.4 m, stall 170 -> 113, crossC 0, off 0
- changes committed: 10de5bb, 856791e, bc23216, 66e02cd
- remaining blocker: tail frames with source=none / plan_speed 0 (68 of 134 tail frames) - not yet attributed

## 6. 离线漏斗那一栏为什么不能直接和实车比（重要，修正 §3 的读法）

§3 表里「离线 `lane_sel=sensor` 83.5% / 57.6%」**不是实车那个量**。两处调用不等价：

| | 离线（`m5_lane_continuity.py`） | 实车（`fsd_stack._sensor_lane`） |
| --- | --- | --- |
| 车道帧 | `pair_lane_markings(markings, ...)` 的**纯视觉帧** | `choose_sensor_lane(vision, lidar, state=...)` 的**融合帧**（含 LiDAR 走廊、镜像几何门、时序一致性） |
| 额外实参 | 不传 `grid` / `map_lane_override` / `lane_consistency_*` | 传全套 |
| `lane_sel` 含义 | 策略对**纯视觉帧**的判定 | 策略对**融合帧**的判定 |

实车侧不是「另有门槛拒绝了车道」——`lane_reject` 在**全部 1787 帧都是空的**（没有任何门拒绝过）。
`fsd_stack.py:915-918` 给出确切口径：`lane_paired = 1` 当且仅当**融合后的 `lane_frame`
存在且 `paired`**。于是最近 8 趟的实测是：

```
lane_sel=perception-unavailable 且 lane_paired=0 : 1475 帧 (82.5%)
lane_sel=sensor                 且 lane_paired=1 :  312 帧 (17.5%)
```

**两种组合之外没有第三类**——即实车里策略从未接受过非成对（镜像/单边界）帧，而离线
重放对同一批帧有 83.5% 接受。所以 §3 的 83.5% 是一个**高估的代理量**，不能用来判断
实车车道可用率；这也解释了为什么离线数字一直比实车乐观（本文件 §3、以及旧文档的
45% vs 11%）。

**下一个候选改动**（证据最集中的一点）：把实车 `choose_sensor_lane` 的逐 tick 判因
打出来（是 `vision_ok`？`_mirror_near_ok`？`_mirror_right_ok`？还是一致性门？），
看那 82.5% 的 tick 到底丢在哪一道门。它在**停车（stall）与压线（off/crossC）的共同上游**，
而且第一遍只需一次**短实车窗口**（≤4 趟）就能定位——按 §5 的约定向并行会话申请。

**同时**：离线 harness 应当在 docstring/输出里标注它测的是「纯视觉帧下界」，或者
补上 `choose_sensor_lane` 与 `grid`/一致性实参，否则后续还会有人拿 83.5% 当实车指标。

vehicle released 11:52 (session 9be6d9dc, 2 arms, per the agreed protocol)

## 7. 逐门归因（实车窗口 11:47–11:51，2 臂，`choose_sensor_lane` 探针 450 次调用）

按 §5 的约定申请到窗口（对端 11:45 写了 `vehicle free`），装纯记录探针后跑了 2 臂并按约定
写了 `vehicle released 11:52`。两臂成绩：lane 28% / 19%，stall 129 / 123，off 9 / 7，
dist 56.5 / 50.8 m（`crossC` 0/0、`crossR` 0/0）。

**450 次 `choose_sensor_lane` 调用里 127 次返回 None（28%）** —— 这就是实车丢掉车道的
tick。按判据树逐门归因，None 的首个失败门：

| 首个失败门 | 次数 | 占 None |
| --- | --- | --- |
| **`_mirror_right_ok`** | **99** | **78%** |
| 无视觉且无 LiDAR（`no_vision+no_lidar`） | 26 | 20% |
| `_mirror_near_ok` | 2 | 2% |

在 277 个「有视觉帧但非成对」的 tick 里，各门通过率：

| 门 | 通过 |
| --- | --- |
| `lane_frame_usable`（vision_ok） | 249/277 |
| `_mirror_near_ok` | 211/277 |
| `_vision_mirror_keeps_reference` | 243/277 |
| **`_mirror_right_ok`** | **6/277（2%）** |

**结论：实车丢掉车道几乎全部由 `_mirror_right_ok` 造成。** 它的判据
（`lane/tracking.py:422-445`）是：右侧线在车旁 `≤ LANE_RIGHT_MIRROR_NEAR_M` 内至少要有
2 个点，且横向中位 ≤ `-LANE_RIDING_LINE_MAX_M`（明显在车右侧）；注释自己写明
「只在**前方几米**才出现的线不能单独引导镜像」。城镇的 `line` 类恰恰是**短虚线块**，
车旁那一段经常没有漆 → 单边界帧被判 false → 融合返回 None → 严格模式无车道 → 停车。

## 8. 下一个候选改动（据此选定）

`_mirror_right_ok` 的两条失败支路需要分开才能定方案，我这一轮没记下来
（探针只记了门的真假，没记 `len(near)` 与横向中位）。**下一段实车窗口只需 1 臂**，
把这两项记上：

* 若多因 **`len(near) < 2`**（车旁没有漆点，只有前方有）→ 方案是让镜像允许
  「前方起始的右线」以**有界的居中提示**参与（而不是当成完整边界），或把近场窗口
  沿纵向放宽到能覆盖第一条虚线块；
* 若多因 **`lat > -LANE_RIDING_LINE_MAX_M`**（线太靠车、被判为正在骑线）→ 那是
  场景里确实压着线，方案不同（不是放宽门，而是先解释横向偏差）。

在拿到这个分布之前**不要动 `_mirror_right_ok`**：两个原因的修法相反，
放宽错了会把车道中心拉过对向车道（这正是这条门存在的理由）。

离线侧不需要车辆：`scripts/m5_lane_continuity.py` 可先量「非成对帧里右侧边界在车旁的
可用率」，与实车的 6/277 对照（注意 §6：离线帧是纯视觉帧，不是融合帧）。
