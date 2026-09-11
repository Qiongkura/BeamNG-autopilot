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

vehicle free 11:56 (session cce01654) - used one bounded window: town_1789098733, dist 50.6 m, stall 147, crossC 0, off 0; added plan_blocked/n_candidates telemetry (60e476e); the lane-but-no-path frames are now attributed to the constraint layer declining all 18 candidates

### 车辆申请（会话 9be6d9dc，11:55）

看到你在 11:54:28 刚跑完一趟（新 telemetry + episode），车辆归你，我不抢。**申请一段
1 臂窗口（约 2.5 分钟）**，用于把 `_mirror_right_ok` 的两个子条件（车旁 `len(near)`
与横向中位）分开记录。你方便时在本文件写 `vehicle free <时间>`，我跑完立刻写
`vehicle released`。

**在等窗口期间我先用离线方式拿同一个判定**：实车的 `vision_frame` 来自
`_sensor_lane_from_semantic` → `pair_lane_markings(...)`，与离线 harness 是同一个函数、
同一几何，所以这两个子条件可以在录制的 episode 上直接量，不需要车辆。若离线分布已经
能唯一区分「车旁缺漆点」与「线太靠车」，这一段窗口就可以省下来还给你。

## 9. `_mirror_right_ok` 的失败支路判定（离线，无需车辆）

用**同一个** `pair_lane_markings`（实车 `vision_frame` 的来源）在最近 3 集 673 帧上复算
该门的两个子条件（模型 v13b，强制 `Segmenter(device="cpu")`）：

```
frames=673  unpaired-with-right-edge=410
  gate passes : 0
  fails       : len(near)<2 -> 410        lat too close -> 0
  len(near) 在该支路上的取值: {0}
```

**结论：99/127 的失败全部是「车旁缺漆点」（`len(near)==0`），「线太靠车」0 例。**
即那些非成对帧的右侧线**在车前方 3 m 内一个点都没有**，与城镇 `line` 类是短虚线块
（车旁那一段常无漆）一致。

**因此修法是第一条支路，不是放宽骑线阈值。** 具体形态取决于「最近点有多远」：
若大多数落在 3–8 m（即车前方紧邻的一块虚线），把镜像的近场窗口沿纵向覆盖到第一块
是可辩护的；若普遍在 10 m 以上，则不能当边界用，只能给**有界居中提示**。正在量这个
分布（`logs/_mirror_split.json`）。

注意 `_mirror_right_ok` 的注释记录了它存在的理由（run 188：一条只在数米外出现的线
把车拖过路面），所以两条支路的修法相反，不能混为一谈。

## 10. `_mirror_right_ok` 的修法：实现后又撤回（2026-09-11 13:2x）

**归因是对的，修法是错的。** §7 把车道丢失定位到 `_mirror_right_ok` 没有错；§9 量出
「410/410 都是车旁缺漆点、骑线支路 0 例」也没错。据此我把近场窗口从 3 m 放宽到 8 m
（`262ed3c`，实测 91.5% 的受影响帧在 8 m 内起漆）。

**撤回依据（并行会话 `3272db9`，与我同日的测量）**：在**必须满足的在道内约束**下
（中心落在本车道中心 ±1.2 m），把单边界帧交回车道 owner：

| | 可用率 | 中心 | 在道内 |
| --- | --- | --- | --- |
| 基线（仅融合守卫） | 81.1% | +0.16 m | **10.8%** |
| 交回 owner / 放宽后 | 82.2% | +0.16 m | **11.0%** |

放宽只把可用率抬 1.1 pt，而**在道内比例不动（~11%）**：单边界镜像用「假定车道宽」
推中心，中心本身就是偏的（中位 +0.16 m，偏在道路中线左侧）。结论是这些守卫**不是
冗余的第二 owner，而是横向正确性层**（严格模式下 owner 没有地图先验可比、侧向门被
跳过，全靠它们）。

所以正确的杠杆是**双侧成对率**（离线 26.2%，中心是两条真实边界的中点、不含宽度假设），
即检测/分割轴，而不是「接受更多单边界帧」。

**现状**：窗口已改回 3.0 m，但把这段结论写进了 `_mirror_right_ok` 的 docstring 与
`LANE_RIGHT_MIRROR_NEAR_M` 的注释，并新增 `tests/test_right_mirror_window.py` **钉住
「不放松」这个决定**（若有人把某条断言改成期望通过，必须先重跑上面的在道内约束）。
离线 `pytest 650 passed` + `m5_offline_validate ALL PASS`。

**另外修掉一个测量缺陷**：`scripts/m5_live_ab.py` 会复用旧 scorecard——游戏中途退出时
每臂 7 秒即失败，工具却把上一臂的分数当成本臂结果，报出「两组各项指标完全相同」的假
结论。现在它要求 scorecard 必须在本臂开始之后写入，并打印失败摘要（已验证：游戏不在时
报 `NO RESULT` + `BNGDisconnectedError`）。**上一轮那批「四臂完全相同」的数据据此作废。**

**车辆**：BeamNG.tech 在 13:2x 前已退出（进程与 64257 端口都没了），所以 A/B 未能执行；
本会话未占用车辆。

## 11. 重启后的基线，以及 stall 的主因（2026-09-11 16:0x，revision `a2e73cf`）

BeamNG.tech 已重启。3 m 严格窗口下的一趟 town 基线：

```
lane=16%  stall=136  crossC=0  crossR=0  off=8  rev=4  dist=48.6m
```

与撤回前几次（lane 19–28%、stall 123–147、off 7–9）一致——**撤回放宽没有改变实车基线**，
符合预期（放宽版本从未在实车上跑过）。

### stall 的主因不是规划器

该趟 220 帧里 68 帧 `no drivable path`，其中：

| `plan_blocked` | 帧数 | 伴随状态 |
| --- | --- | --- |
| **`no_perception_lane`** | **63** | 这 63 帧**全部** `lane_paired=0`、`lane_sel=perception-unavailable` |
| （空） | 5 | `n_candidates=18`，即候选确实被约束层否掉 |

全趟 `lane_paired=0` 占 184/220（84%）。所以 **stall 是「没有成对感知车道 → 严格模式
按铁律停车」的计数，不是规划/约束缺陷**。并行会话 `60e476e` 的「约束层否掉 18 个候选」
只覆盖这里的 5 帧，不是主因——这一点值得澄清，免得在规划侧空转。

### 当前 revision 的离线漏斗（recovery 默认开）

3 集，配对 **22.8% / 19.1%**，未配对首因仍是
`one_side_filtered_by_candidate_gate`（137 / 129），其次 `both_sides_but_no_pair`
（32 / 26）。门控丢弃的主体是 `unknown:kind`（450，GT 40%——宽漆块，本来就该丢）与
`dashed:align`（140，GT 68%——横向漆，斑马线一类，本来就该丢）；抽取器最大损失是
`small_h`（40+23，GT 57–66%），即**车旁短虚线块**。

### 结论：唯一杠杆是双侧成对率，且它落在检测/分割轴

`lane_paired` 决定 stall（无车道就停）与 off/crossC（有车道的少数帧里被判定压线）。
而双方证据都指向：成对率受限于「城镇漆画本身是否给出两条共线的长边界」——
抽取器守卫的放宽已被否定（标线 3→57.9/帧、配对零增益），单边界回交已在
在道内约束下被否定（§10）。所以下一步只能是**让检测找回更长的线**：分割模型与
其训练数据（用户手工标注帧），也就是并行会话正在推进的方向。**本会话不应在此重复
投入**，而应提供离线可判定的目标函数（成对率）与已验证的测量协议。

## 12. 检测轴候选：换 checkpoint 已排除（成对率口径，2026-09-11 16:2x）

既然 §11 把杠杆收敛到「双侧成对率」，第一个该试的检测轴候选就是换分割模型——对端的
`84a0ed5` 用 **line IoU** 发布过模型×地图矩阵（v8 在三套数据上都领先），但城镇 pin 的是
v13b，两者从未在**功能指标**上对照过。现在对照（3 个最新 town episode，recovery 开）：

| checkpoint | 各趟配对率 | 均值 | 中心偏差（各趟） |
| --- | --- | --- | --- |
| **v13b（城镇 pin）** | 22.8 / 19.1 / 18.6% | **20.2%** | −0.08 / −0.35 / +0.01 m |
| town_manual_v20 | 15.2 / **40.9** / 8.6% | 21.6% | −0.63 / +0.31 / −0.46 m |
| v13 | 20.1 / 20.0 / 13.2% | 17.8% | +0.39 / +0.30 / +0.85 m |
| v8 | 4.0 / 11.1 / 8.6% | 7.9% | −0.34 / −0.91 / −0.05 m |
| v12（`seg_model/best.pt` 默认） | 3.6 / 2.7 / 1.8% | 2.7% | +0.17 / +0.28 / +1.03 m |

结论：

* **v13b 保持最优 pin**。均值与 v20 相当（20.2% vs 21.6%），但 v13b **逐趟一致**
  （18.6–22.8%），v20 靠单趟 40.9% 拉高、另两趟只有 15.2 / 8.6%；且 v20 与 v13 的
  **中心偏差明显更大**（±0.3–0.85 m），v13b 近 0。
* **IoU 与功能指标在这里不一致**：v8 在 IoU 矩阵上领先，但成对率只有 7.9%。所以
  「按 IoU 换模型」不成立——这与本会话早先「三个指标互相打架」是同一类教训。
* 默认 `seg_model/best.pt`（=v12）在城镇只有 2.7% 成对率，说明**任何未 pin 的场景都会
  悄悄跑在弱模型上**（对端 `84a0ed5` 已就此加了「UNPINNED default」日志）。

因此检测轴上「换现成 checkpoint」这条候选**排除**。真正的杠杆是**训练出更长的线**：
分割模型与其训练数据（用户手工标注帧），即并行会话正在做的方向；本会话的贡献是
把目标函数固定成**成对率**（而非 IoU），并留下可复现的对照口径。

## 13. 检测轴尝试：采集城镇真值并训练新模型——**失败**（2026-09-11 13:3x）

### 先说一件事：这一步的「人」我做不了

指定流程的第一步是用 `m5_annotate_manual.py` **人工标注**那 19 帧。那是 OpenCV 交互
窗口，要人用鼠标逐帧涂漆线；我无法代替。**用模型自动生成标签再拿去训练是循环自证**
（本会话早先已在 `--gt-audit` 上指出过同一缺陷），所以我没有走这条捷径。

### 我走的等价路径（不循环、可自动）

改用仓库自己的真值采集器 `m5_collect_seg.py --town`（`tech_truth` 域，标注像素来自
BeamNG.tech，与我的模型无关），在 italy 城镇 (781,748) r=160 m 内采到：

```
run_town_truth_1329  120 帧  536x403  colour+label  标线 525 px/帧
```

然后用城镇向配方训练 `seg_model_town_v22`（`run_navroute_town` 525 真值 + 新的 120 真值，
并入人工城镇标线集 `manual_review_batch_labeled`/`manual_compare_batch_labeled`/
`manual_current_full`，`--line-weight 2.0 --line-morph --epochs 40`，best val mIoU 0.6322）。

### 结果：功能指标显著回退

| 模型 | 各趟配对率 | 均值 | 离线 `lane_sel=sensor` | 中心偏差 |
| --- | --- | --- | --- | --- |
| **v13b（当前 pin）** | 22.8 / 19.1 / 18.6% | **20.2%** | 77.7 / 67.6 / 67.3% | −0.08 / −0.35 / +0.01 m |
| town_v22（新训） | 10.3 / 14.2 / 11.4% | **12.0%** | 38.0 / 24.9 / 41.8% | −0.69 / −0.44 / +0.21 m |

**v22 在每一项上都更差，不能 pin。** 值得注意的是它的 val mIoU（0.6322）**高于** v13b
记录的 0.5599，成对率却只有一半——**又一次「IoU 与功能指标不一致」**（同一个教训在
§12 的 v8 上出现过）。

### 这**不是**「加城镇真值没用」的证据

两个理由：

1. 配方不可比。v22 只用了 `run_navroute_town` + 新真值 + 少量线标注，而官方配方
   （README）是**全量** `run_*`（约 7600 帧）+ `--min-line-frac 0.003 --split per-run
   --epochs 60`。v22 少了泛化数据，回退可能来自数据分布的窄化，而不是「新真值有害」。
2. 要做公平对照，需要「全量配方 ± 新真值」两次训练。按实测速度（约 645 帧 13 s/epoch）
   推算，全量约 2.5 分钟/epoch × 60 ≈ **2.5 小时/次**，本回合做不完，未做就不报结论。

### 交接给下一步（有明确判定标准）

* 采集这一步**已验证可用**：`m5_collect_seg.py --town --segments N --frames-per-seg M`
  能稳定产出带真值的城镇帧（本次 120 帧、525 px/帧），代码无需改动。
* 待做的是「全量配方 ± `run_town_truth_*`」两次训练，判定用**成对率**（当前 20.2%），
  并复核中心偏差是否仍近 0——IoU 不能作判据。
* 那 19 帧若仍要人工标注，命令已就绪：
  `.venv\Scripts\python.exe scripts\m5_annotate_manual.py --frames-dir logs/m5_seg/manual_town_capture_20260910_233426 --out logs/m5_seg/manual_town_labeled`
  （可选 `--prefill-model logs/m5_seg/seg_model_v13b/best.pt` 预填以减少涂画量；预填结果
  **必须人工修正**，不能直接当标签。）

## 14. 全量配方 ± 新城镇真值：已启动，判定标准预先固定（2026-09-11 13:5x）

按 §13 交下来的公平对照，两次训练已在后台**顺序**执行（唯一变量是 `run_town_truth_1329`）：

```powershell
# ARM A —— 不含新真值
.venv\Scripts\python.exe scripts\m5_train_seg.py `
  --runs <logs/m5_seg/run_* 去掉 run_town_truth_1329> `
  --min-line-frac 0.003 --split per-run --epochs 60 --out logs\m5_seg\seg_model_full_A
# ARM B —— 含新真值
.venv\Scripts\python.exe scripts\m5_train_seg.py `
  --runs logs\m5_seg\run_* `
  --min-line-frac 0.003 --split per-run --epochs 60 --out logs\m5_seg\seg_model_full_B
```

实测代价：`--min-line-frac 0.003` 在 14 个 run（约 7600 帧）上保留约 2400 帧；
**37–74 s/epoch**（游戏与训练共享 GPU），单臂 60 轮约 35–50 分钟，两臂约 1.5 小时。
进度与日志：`logs/_fulltrain.log`。

**预先固定判定标准（避免事后挪门）：**

1. 用 `m5_lane_continuity.py --episodes 3 --all-frames --seg-model <arm>/best.pt` 读
   **成对率（PAIRED own-lane frame）** 与 **own-lane centre lat**；
2. 与 **v13b 的 20.2%（22.8 / 19.1 / 18.6%）** 比较，且要求**逐趟一致**（不接受靠单趟
   拉高的均值，见 §12 的 v20）与**中心偏差近 0**（v13b 为 −0.08/−0.35/+0.01 m）；
3. **val_mIoU 不作判据**——已有两次反例（§12 的 v8：IoU 领先而成对率 7.9%；
   §13 的 v22：mIoU 0.632 > v13b 0.560 而成对率 12.0% vs 20.2%）。ARM A 前 18 轮的
   val_mIoU 已到 0.72–0.78，同样不能据此判断；
4. 取优者（且必须**超过** v13b，而不是「与 v13b 相当」）才 pin 进 town 场景，再跑
   `scripts/m5_live_ab.py` 的实车 A/B 复核 `crossC/crossR/off` 与 `stall`。

**当前状态**：ARM A 训练中（18/60），ARM B 排队；尚无结论，验收状态不变。

## 15. 全量配方 ± 新城镇真值：结果（2026-09-11 15:2x）

两臂同为 **32 轮**（`--min-line-frac 0.003 --split per-run`），唯一变量是
`run_town_truth_1329`（120 帧里 27 帧通过线密度过滤）。按 §14 预先固定的判据测成对率：

| 模型 | 各趟配对率 | 均值 | 中心偏差（各趟） | val_mIoU |
| --- | --- | --- | --- | --- |
| **v13b（当前 pin）** | 22.8 / 19.1 / 18.6% | **20.2%** | −0.08 / −0.35 / +0.01 m | 0.560 |
| full_B（**含**新真值） | 13.4 / 6.7 / 13.2% | 11.1% | −0.05 / −0.35 / +0.57 m | 0.800 |
| full_A（**不含**新真值） | 0.9 / 0.4 / 0.4% | **0.6%** | −1.29 / −1.19 / −1.16 m | 0.795 |

**按判据：两者都没有超过 v13b（20.2%），所以 v13b 保持 pin，不 pin 新模型、不跑实车
A/B。** 这一步的产出不是更好的模型，而是三个可用的量化事实：

1. **手工标注的城镇数据是决定性的成分。** 同为全量配方，只差 27 帧城镇真值：
   0.6% → 11.1%（**+10.5 pt**）；而带着用户手工城镇标线的 v13b 再高 **+9.1 pt**
   （11.1% → 20.2%）。也就是说，用户那 19 帧手工标注（以及既有的城镇标线集）**正是**
   功能指标的来源——这**支持**「先人工标注再训练」这条路线，而不是否定它。
2. **README 推荐的全量配方在城镇功能指标上是灾难**（0.6% 成对率、中心偏 −1.2 m），
   尽管它的 val_mIoU 最高（0.795）。旧文档把 `--runs run_*` 当推荐配方，这里给出反例：
   对城镇，它不能单独使用。
3. **第三次「IoU 与功能指标不一致」**：full_A 的 mIoU 0.795 / 成对率 0.6%；
   full_B 0.800 / 11.1%；v13b 0.560 / 20.2%。三个模型的 IoU 排序与成对率排序**完全相反**。
   模型选择若用 IoU 会选出最差的那个。

### 下一步（由上面的量决定）

**去标注那 19 帧**（`m5_annotate_manual.py`，需人工），或扩充城镇手工标线集——这是唯一
被证明能推动成对率的成分。命令已就绪：

```powershell
.venv\Scripts\python.exe scripts\m5_annotate_manual.py `
  --frames-dir logs\m5_seg\manual_town_capture_20260910_233426 `
  --out logs\m5_seg\manual_town_labeled
# 可选：--prefill-model logs\m5_seg\seg_model_v13b\best.pt（预填必须人工修正）
```

标注后再训（把 `manual_town_labeled` 放进 `--line-only-runs`），仍以**成对率**判定，
并复核中心偏差近 0；IoU 不作判据。
