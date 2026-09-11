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
