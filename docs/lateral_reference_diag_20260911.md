# 横向参考台阶：诊断（2026-09-11）

**结论先说**：归并开启后城镇实车的出路面/压线**不是**漆线修正器
（`PaintedLineLateralCorrector`）在 owner 切换时瞬断造成的——出路面那几帧修正器
早已关闭且 shift 为 0。证据指向：**规划器选中的感知车道帧与 Scene 里供安全判定
用的感知参考不是同一个对象**。规划器按 `lane_src_sel=sensor` 的帧走，监视器却按
`lateral_reference(scene)` / `_scene_boundaries(scene)`（Scene 的
`lane_left/lane_right` 或 `lane_envelope`）判「车体压线」，
两者不一致就 fail-closed（`minimal_risk` + `reason=current vehicle body crosses
lane boundary`）。因此本文件记录诊断，并**撤回**「给修正器加交接限幅」这条计划。

## 数据来源

`logs/fsd_benchmark/town_1789093299.json`（2026-09-11，`BEAMNG_DASHED_RECOVERY=1`，
v13b，221 帧，off=19、crossR=6）。出路面/右侧压线共 21 帧。

## 时间线（首次出路面）

| t (s) | lane_sel | source | level | plc_active | plc_shift | lane_dev_m | line_lat | speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.059 | sensor | fsd | safe | 0 | 0.0 | 0.121 | +0.434 | 2.83 |
| **7.285** | sensor | **rule** | **minimal_risk** | 0 | 0.0 | **2.113** | +0.434 | 3.35 |
| 7.979 | sensor | rule | minimal_risk | 0 | 0.0 | 2.297 | −1.962 | 1.60 |
| 8.629 | perception-unavail | fsd | safe | 1 | 1.2 | 1.292 | −1.962 | 0.50 |

读法：

* 7.285 这一帧 `plc_active=0`、`plc_shift=0.0`，且**它之前若干帧也是如此**——修正器
  没有在一帧内丢掉 1.6 m 的修正量，不存在「修正器瞬断」台阶。
* 车的**漆线横向读数没变**（`line_lat` 7.059 与 7.285 都是 +0.434 m），说明车相对漆画线
  并没有横向跳变。
* 变的只有 `lane_dev_m`（0.121 → 2.113 m）与 `source`（fsd → rule）：**判定所用的
  车道参考跳了**，于是监视器判 `minimal_risk`，车被 fail-closed 到 `rule` 兜底路径。
* 21 帧出路面**全部**是 `lane_sel=sensor` + `plc_active=0` + `source=rule` +
  `level=minimal_risk`，`lane_dev_m` 区间 1.5–5.7 m。

## 由此排除与由此指向

排除：

1. 修正器 owner 交接瞬断（本文件上表）——`plc_shift` 在事件前已是 0。
2. 修正器饱和顶到 `PLC_MAX_SHIFT_M=1.6`（全程 p50=1.60）本身不是病因：饱和发生在
   `perception-unavailable` 段（地图先验与漆线中心差 ~1.6 m，属已知偏差），
   而出路面发生在 `sensor` 段。

指向：

* **同一 tick 的横向参考不唯一**。规划器在 `lane_sel=sensor` 时用感知车道帧；监视器的
  `body_pose_crosses_lane` / `lane_dev_m` 走 `lateral_reference(scene)` /
  `_scene_boundaries(scene)`，即 Scene 的 `lane_left/lane_right` 或 `lane_envelope`。
  两者不一致时，监视器把「车体在自己的感知车道内」判成压线，直接 fail-closed 并把车按
  rule 路径带离。
* 归并把感知车道帧可用率从 ~2% 提到 16–17%，于是这类「两个参考不一致」的帧数同比放大，
  表现为 off 5–19、crossR 0–6；关掉归并时可用率低，反而看不到。

### 探针证据（`lateral_reference` 逐次调用，662 条）

用纯记录探针（`sitecustomize` 包装 `safety_monitor.lateral_reference`，不改行为）在
同配置下再跑一趟（`town_1789094003`，归并开启，lane 22%、off 8）：

* 参考来源分布：`envelope` **502** 次 / `sensor` **160** 次——监视器的参考多数时候是
  **envelope**，与规划器记录的 `lane_sel=sensor` 并不是同一个对象。
* 该参考相对自车的横向位置在 **−2.07 … +1.40 m** 之间摆动（p1 −1.63 / p50 −0.29 /
  p99 +0.83）。
* 相邻两次调用之间最大跳变 **1.72 m**（p90 0.044）。这与出路面帧 `lane_dev_m`
  单帧从 0.121 跳到 2.113 的量级一致：跳的是**参考**，不是车。

## 下一步（不要重走修正器交接）

1. 让监视器的车道交叉/偏离判据**消费规划器同一次选中的那个 lane reference**
   （`lane_src_sel` 与 lane frame 一起进 `Scene`），而不是各自从 envelope 再推一次；
   这才是分支名 `lane-reference-single-owner` 的字面要求。
2. 加一致性守卫时要区分「车真的压线」与「感知帧与 envelope 互相矛盾」：后者应当
   降级为「参考不可信、退回上一帧一致参考」，而不是把车甩给 rule 路径。
3. 复验仍用同日 A/B：`.venv\Scripts\python.exe scripts\m5_fsd_benchmark.py --attach
   --runtime tech --scenarios town`，`BEAMNG_DASHED_RECOVERY` 开/关各跑，看
   `crossC/crossR/off` 与 `lane_dev_m` 跳变帧数。

## 本轮另外解决的事（已提交）

`4fa7c2a`：stale-sensor 按模态判定。实车 `stale sensor` 75 → 0/1 帧，
`snapshot_age`（≈控制周期，p50 0.58 / max 0.92 s）不再顶着 0.8 s 的模态阈值误判。
