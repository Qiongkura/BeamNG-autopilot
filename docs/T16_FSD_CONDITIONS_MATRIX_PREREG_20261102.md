# T16 FSD 条件矩阵预注册（2026-11-02，③ 整车主线第一役）

## 动机与定位

标线识别研究已收口（`T16_LINE_RECOGNITION_CLOSEOUT_20261102.md`），按双指标
计划第 4 步转 ③ 整车主线。真车能力的第一问不是"峰值指标多高"，而是
**"什么条件下还能开、什么条件下 fail-close"**——环境条件（时刻×天气）是
其中最基本的一维。本预注册在跑批前冻结矩阵、口径与判读规则。

## 矩阵（20 轮，全部固定场景）

场景 = `m5_fsd_benchmark.SCENARIOS["mountain"]`（italy hairpin 起点
(729.6, 763.9, 45.0)，goal (616.2, 894.5)，seg_model 沿用 benchmark 的 v8
pin），`lane_mode=sensor + strict=True`（唯一 FSD 合规路径），60 s/轮，
speed 6.0，`--max-wall-s 300`。每格一个独立引擎会话（自启自关）。

| 维度 | 值 |
| --- | --- |
| 时刻（`--tod`，EnvironmentApi.set_tod） | dawn 06:30 / noon 12:00 / dusk 18:30 / night 21:30 |
| 天气（`--weather-preset`） | clear = 引擎默认（不调预设）；rain / fog = 预设名，跑批前由引擎探测（`logs/probe_weather_presets.py`，`extensions.core_weather.getPresets()`）解析后经 CLI 传入 |
| 重复 | 12 格各 1 轮（r0）+ 8 个严酷格 1 重复（r1）：night×{clear,rain,fog}、dusk×{rain,fog}、noon×{rain,fog}、dawn×fog |

**失败预案（预注册，不临时改）**：
- 预设名缺失 → 该格**不跑**，聚合标 UNKNOWN（不静默降级为晴天格）。
- 环境应用失败（session `environment.applied=False`）→ 该轮 UNKNOWN。
- 引擎侧读回时刻与请求不一致 → UNKNOWN（防预设覆盖 TOD 的静默错格）。
  接线顺序已按 `core_weather.lua` 证据固定为 **weather → tod**
  （preset 的 diff 表可携带 timeOfDay；先设时刻会被覆盖），应用后
  `get_tod()` 读回存证。
- 放置失败（strict+sensor 布不上感知车道）→ 该轮 UNKNOWN，计为
  **可用性=0** 的事实证据，不重试到通过为止（每格至多重试一次，见下）。
- 每格允许 1 次协议重试（BNGDisconnectedError 类瞬断）；重试仍失败 →
  UNKNOWN。**绝不为了出表重跑到好看为止**——重试轮全部记录。

## 指标（口径冻结）

每轮由 `m5_fsd_conditions.py --score` 聚合（复用 `beamng_autopilot.eval`
的 `assess_run`/`score_run`，settle_s=8.0）：

1. **车道可用性**（主指标）：`lane_sensor_rate`（感知车道占比）、
   `lane_paired_rate`（双侧真实检测占比）——夜间/雨雾下预期显著下降，
   这是"真车条件鲁棒性"的直接读数。
2. **安全等级分布**：safe / degraded / minimal_risk 帧占比。
3. **纪律计数**：cross_centre / cross_right / off_road / reversing /
   stuck / emergency 帧（settled 口径）。
4. **判定**：`score_run` PASS/FAIL/UNKNOWN；格级聚合取**最差轮**，任何
   UNKNOWN 轮或轮次不全 → 整格 UNKNOWN（不降门）。
5. **已知边界**：车道穿越计数（车身级）当前存在 UNKNOWN 清零任务——
   本矩阵如实报 assess 口径计数，不声称碰撞=0（collision 通道无证据时
   score_run 自身会给 UNKNOWN）。

## 判读规则（跑批前冻结）

- **主问题**：哪些格还能保持 `lane_sensor_rate > 0` 并 PASS/FAIL（在开），
  哪些格 fail-close 到原地（minimal_risk 主导）或放置失败（不能开）。
- clear 各时刻格为基线；rain/fog 相对同时刻 clear 的可用性差值即
  天气代价；night/dusk 相对 noon 的差值即光照代价。
- 重复格两轮一致（差值 < 0.10 可用性）才允许报"该格稳定"；不一致如实
  报告并标方差（引擎侧天气/交通随机性）。
- 不因结果调整门限、不删格、不换场景、不换模型（v8 pin 全程不动）。

## 预注册修订案 A1（2026-11-02，任何有效格跑出之前）

- **seg_model pin 变更**：原 pin（benchmark 的 `seg_model_v8_backup`）在
  2026-10-29~11-01 的 `logs/m5_seg` 外部删除事件中灭失（部署默认
  `seg_model/best.pt` 同灭，见 `PROJECT_STATUS_RECHECK` 删除事件记录）。
  改 pin **`seg_model_v28_sslA/best.pt`**（SSL +7.5% 水位模型、种子 42；
  fold A 留出=早期 utah，**训练集含 italy**——矩阵场景在 italy）。
  20 格全程单一固定 pin，不换模型。首轮 dawn_clear_r0（无分割模型放置
  失败刹车保持）记为 **VOID 轮**，证据保留，不计入矩阵。
- **预设名解析结果**（引擎探测 `logs/probe_weather_run.log`）：
  库存 7 个（sunny_evening/sunny/cloudy_evening/sunny_noon/rainy/
  foggy_night/foggy_morning）；映射 rain→`rainy`、fog→`foggy_morning`
  （单一 fog 预设保证天气维与时刻维独立）、clear→引擎默认。
- **顺序问题的引擎证据**：7 预设中 6 个携带 timeOfDay 并覆盖显式 TOD
  （after_preset 实测 22:48/17:57/06:08…），weather→tod 顺序修复被
  实测验证（after_tod_again 全部回到请求值）；矩阵按修订后接线跑。

## 纪束

产物只落 `logs/fsd_conditions/`（遥测 + session 审计 + batch_manifest +
conditions_table）；探针/诊断帧不进任何训练池；不降门、不消费最终集、
无人工标注、不晋级模型。矩阵结果只用于刻画条件边界与排下一批
（250 s 长跑 + 安全计数可信化）的优先级。

## 预注册修订案 A2：执行阻断（2026-11-04）

条件矩阵在 A1 修订后的首轮断点续跑就被**放置阶段**阻断：dawn 四格与
noon_clear 全部 `rc=3`（感知零线/单侧），驾驶格 0 个。三场景探测定因：
mountain 无配对标线（line_px=0）、town 仅单侧（mirror_right/mirror_left 被
严格门拒）、italy 认证站 0.15m 细线远场对拒绝（far_kind）；历史 town 配对
率 43–60% 依赖的部署域模型已在删除事件中灭失。按预注册"不降门、不换合规
路径"的纪束：**矩阵在本资产下不可执行**，全部格以 UNKNOWN 归档，不重跑、
不改门、不改场景凑指标。详细证据与解阻路径见
`T16_FSD_CONDITIONS_BLOCKED_20261104.md`。
