# T16 条件矩阵结果：执行被放置阶段阻断（12/12 UNKNOWN，2026-11-04）

## 执行记录（断点续跑）

- `dawn_clear_r0` / `dawn_rain_r0`：已有 session 证据，按预注册**不重跑**
  （`dawn_rain` 的 `tod_after=09:56` 是天气过渡动画修复前的记录，聚合器已
  正确判 UNKNOWN）。
- 本轮实际执行：`dawn_fog_r0`、`dawn_fog_r1`、`noon_clear_r0` → 全部
  `rc=3`（放置失败，刹车保持到墙钟上限）；`noon_rain_r0` 启动后被场景诊断
  中止（矩阵无法产出驾驶格）。
- **环境接线本身正常**：`applied_tod` 分别为 06:29:59 / 06:29:59 / 12:00:00，
  与请求一致（weather→tod 顺序与读回存证按设计工作）。

## 条件×表现表（12 格全 UNKNOWN）

| 格 | 轮次(UNK) | 状态 | placement_rc | 环境应用 | 读回时刻 |
| --- | --- | --- | --- | --- | --- |
| dawn_clear | 1(1) | UNKNOWN | 3 | ✓ | 06:29:59 |
| dawn_rain | 1(1) | UNKNOWN | 3 | ✓ | 09:56:26（修复前记录） |
| dawn_fog | 2(2) | UNKNOWN | 3 | ✓ | 06:29:59 |
| noon_clear | 1(1) | UNKNOWN | 3 | ✓ | 12:00:00 |
| noon_rain/fog | 0 | UNKNOWN | — | — | 未执行（中止） |
| dusk/night 六格 | 0 | UNKNOWN | — | — | 未执行（中止） |

**表中没有任何 PASS/FAIL**：阻断发生在感知放置，驾驶指标（车道可用率、
越线、出界、卡死）全部无数据。这是 fail-closed 的正确行为，不是降级。

## 阻断根因（三场景放置探测）

| 场景 | 模型 | line_px | 配对模式 | 结果 |
| --- | --- | --- | --- | --- |
| mountain hairpin（benchmark） | sslA | 0 | no_marks | 场景无配对标线，放置不可能 |
| town（benchmark） | sslA | ~4400 | mirror_right | 仅检测到右单侧，严格门拒 |
| town（benchmark） | v29_ssl2A | ~4500 | mirror_left | 仅检测到左单侧，严格门拒 |
| italy 认证站 off30 a1s0 | sslA | ~9300 | far_kind | 0.15m 细线远场对拒绝 |

**历史对照**：已灭失的 `seg_model_hand` 在 town 基准轮 `lane_paired_rate`
43–60%、120 s 内行驶 208 帧；现存池模型（71 族线池训练）在 town 配对率 **0%**。
→ 2026-10-29~11-01 的 `logs/m5_seg` 外部删除事件不仅删掉了模型文件，
还使**部署域感知能力**出现可复现回归：池模型是线裁剪研究资产，不能替代
town 域部署模型。

## 判读与后续

1. **条件矩阵在当前感知资产下不可执行**：矩阵测的是条件对驾驶的影响，
   但放置阶段就 fail-closed，无法进入驾驶响应。把全 UNKNOWN 表当作结论
   如实归档，不为了出表降门或换合规路径。
2. 解阻需要先恢复**部署域感知**，三条合法路径：
   - 恢复/重训 town 域模型（需要数据授权或重建标注管线；人工标注约束不变）；
   - 预注册"近场薄线/单侧配对鲁棒性"研究臂（涉及安全门，需单独预注册）；
   - 换问题：paved/corridor 车道源下的条件测试（需新预注册，非本次矩阵）。
3. 本轮不改门、不重跑已记录失败格、不消费最终集；mountain 12 格 UNKNOWN
   与三场景探测 session 全部保留。

## 产物

`logs/fsd_conditions/conditions_table.json`（12 格 UNKNOWN）、
`dawn_*_r*.json.session.json`（放置失败审计，含环境读回）、
`_town_probe*.json.session.json`、`_italy_probe.json.session.json`（场景诊断）。
