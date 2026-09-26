# 本轮报告（S4 看板 / S5 有效因子 / T10 / T15 / 审计后重测）

日期：2026-09-26 深夜；HEAD `5231b91`；工作树（已跟踪文件）干净。
方案：`docs/T14_DEVELOPMENT_EXECUTION_PLAN_V2_20260926.md`（v2.1）。
本轮并行 4 个子 agent（看板 S4、独立复核 T01–T16、损失方向 T15、内容冲突 T01/T02），
主 agent 独占评价/循环入口并集成；复核者与实现者分开。

## 1. commit / config / run / 控制权

| 项 | 值 |
|---|---|
| HEAD | `5231b91`（本轮 18 个提交，`check_commit_scope.py --last 16` = OK，每提交一个模块） |
| dirty | 已跟踪文件 0 改动；未跟踪的 `.git.bak-20260920/`、`.workbuddy-ai/`、`rescue-20260920/` 是既有用户工作，未动 |
| 本轮命令 | 见 §2/§3 与各 doc；回归 `scripts/dev_validate.ps1`；标定见 `T14_AUDITED_TRUTH_REMEASURE_20260926.md` |
| 数据/协议/模型哈希 | 协议 v5 `275cddb5834ba549`；阈值 v4 `a02e4db5b46feeaa`；production `b16a734d67f3c0de`、候选 last `aa3c615c101b3fbf`（sha16，写入标定 JSON） |
| 实际设备 | `cuda`（探针读回 Segmenter 属性，不是命令行声明值） |
| 控制权 | 本轮**未**启动游戏、未调用 ControlBridge、未跑训练；无控制权争用。唯一写入者：主 agent（入口）+ 各子 agent（登记文件） |

## 2. 覆盖 / 缺列 / UNKNOWN

* **测量清单**（`reviewed_full`）：136 帧 = 136 唯一图，0 拒绝、0 冲突（manifest 内容
  去重 + 标签/几何身份判定的结果）。
* **union 审计**（`reviewed` + `reviewed_full`）：159 输入 → 114 唯一 / 45 拒绝 /
  **24 个内容冲突组**。根因：早先的预览包把同一源帧复制进两个槽位、且两次人工标注
  不同（打包缺陷，已在 `m5_review_pack.py` 池内去重修掉）。预览包是被取代的输入，
  不进测量分母；冲突逐条留在 `union_audit.conflicts`（不是"不扫就没有冲突"）。
* 逐通道覆盖：`paint_valid=136`、`pavement_valid=27`、`road_type_frames=27`、
  8 个采集组（与上一轮一致；本轮未新增采集）。
* **N/A 与 UNKNOWN 分开**：两个确认真无线场景（P_frames=0，共 58 帧）记
  `not_applicable`（不进有线门，也不算通过）；有线但 R=0 记 `unknown`（不是 0 分）。
* 训练关系：评价集 24 帧（`ring_20260923_135342`）与候选训练数据确定 in-sample；
  其余 112 帧与四个权重声明的训练目录无内容重叠。生产模型 13 个训练目录**没有
  地图身份** → 组键比对是键空间差异，**不能**读作"无泄漏"（UNKNOWN，未变）。
* 探针缺测：缺相机 → 逐帧 UNKNOWN（T09）；缺 meta / 探针异常的 run 逐条进
  `eval_run_errors`（T11，本轮补上）；缺计数不按 0 累加。

## 3. 刷新与消费

* 本轮没有新采集。审计后的唯一清单被**标定**（`m5_identity_calibration.py`）与
  **自动循环**（`identity_metrics(frames_by_dir=...)`）消费，二者现在同一套计数实现、
  同一份清单；证据：`tests/test_scene_counts_wiring.py`（含"两个字节相同目录不把 C
  翻倍"）与真实标定运行。
* 判定文件新增入口字段：`git_commit`/`git_dirty`/`device`/`data_counts`/
  `scene_applicability`/`eval_run_errors`/`factor_activity`/`round_outcome`，
  看板直读（§4 页面检查）。
* 驾驶时强制刷新与最终消费：**未测**（本轮无驾驶）。

## 4. 几何与最终控制

* 未改控制链、未改候选提取/投影/角色代码。确定性几何正反例：`m5_offline_validate.py`
  ALL PASS（bend-wall / twitch-scene 共 10 项）；T15 合成 FP/FN 与 ignore 用例通过。
* 候选来源与角色：探针逐帧记 `candidate_sources`，覆盖率分母 C 与角色分母 L 分开；
  未做新的几何实验。
* 最终横向参考/控制命令：未测（无驾驶）。

## 5. 路面门

未测（本轮无驾驶、无路面门实验）。相关：负例诊断（`negative_line`）现在按 T10 强制
资格——只有 verified 档位的全零标签才算合格负例，非 verified 记 `unverified_frames`
并排除；58 帧负例诊断的旧结论（生产 0 误报、候选 last 在土路 28/28 误报）保留，
但已注明"生产训练过土路素材"这一混淆因素。

## 6. 性能 / deadline

* 全量回归 **2944 passed in 544.20s**（`logs/experiments/gate_20260926_222702.log`），
  离线验证 `RESULT: ALL PASS`，入口 `RESULT: pytest=PASS offline_validate=PASS`（rc=0）。
* 本轮未做受控计时复测、未训练；驱动 tick deadline 未测。旧 p95 值不作为本轮结论。

## 7. 驾驶事件

未驾驶：碰撞、压线、倒车、出铺装、分类停车时长**均未测**，不填 0。

## 8. 状态

| 级别 | 状态 | 说明 |
|---|---|---|
| D1 评价链修复 | **通过**（上一轮签收，本轮补 T10/T11 缺口并重测） | 计数契约、适用性、版本与重放、逐场景下限在真实路径生效 |
| D2 学习循环可解释 | **部分** | S4 看板（七面板 + 页面实检）、S5 因子有效性与单因归因已落地并有测试；S6 E2 已跑一轮且**因子真实激活**（见下），但只有 3 个 seed、证据不足，无受控的改善结论 |
| R2 离线候选 | 未开始 | 身份率 0.3115/0.3280 远低于 0.60，无可晋级候选；最终集未消费 |
| R3 Tech 候选 | 未开始 | 依赖 R2 |

### S6 E2 第一轮（研究臂，因子真实激活）

`logs/experiments/t14_e2_direction_v3_20260926`：FP/FN 方向（alpha/beta
0.3/0.7 → 0.5/0.5，比值 0.43→1.0，和不变）。两臂各 20 帧、**实测步数各 120**
（等步数成立），`factor_activity.active=True`，命令逐字 diff 只有这两个参数。
配对结果：line_recall +0.0117（CI ±0.114，需 62 seed）、line_iou −0.0102、
line_precision −0.0127、**身份率 −0.0219（3/3 seed 全负）**、时延不变 →
按冻结规则判 **"证据不足"**；唯一一致的方向是身份率下降，**支持保留默认 recall 偏向**。
判定 `research_only=True`（agent 弱标签不得晋级），`round_outcome=qualification_failure`。

### 本轮踩到并修复的两个真实缺陷（都在我自己的接线里）

1. **唯一清单键不匹配**（E2 前两轮）：`dev_frames_by_dir` 的键先是帧文件路径、
   再是未规范化的相对/绝对混用 → `frames=[]` → 探针拒测 → **身份率静默 UNKNOWN**
   （判定里 `counts` 全 0 而 `eval_run_errors` 为空）。修复：键统一为"目录的
   规范化绝对路径"，查不到就记 `eval_run_errors`（缺测可见）；两轮失败产物保留在
   `logs/experiments/t14_e2_direction_20260926`、`..._v2_...` 作为证据。
2. **白名单漂移**：把 `line_tversky_alpha` 加进 `TRAINER_FLAG_FACTORS` 后，
   `proposer.APPLICABLE_KEYS` 与测试里手抄的线键表没跟上 → 全量回归失败
   （`test_the_proposer_only_emits_keys_the_trainer_can_apply`）。修复：三处同步，
   并把测试改成从 autoloop 现场派生（不再手抄）。

* **候选去留**：两个权重都不满足身份门（0.60），不晋级；也不降低门槛。
* **下一轮输入**：S6 定向实验。E2 第一轮已跑（方向 = 证据不足，见上）；
  E1（困难负例）**受阻**：按 T10"已确认负例"必须是 verified 档位，而现有 verified
  负例都在开发/评价集里（E1 明文禁止入训）→ 需要评价集之外的新人工确认负例，
  或书面授权研究臂版本。线通道的可晋级训练标签同样缺（现在只有 agent 弱标签，
  只能记 `research_only`）。
* **停止理由**：无（未触发停止条件）。

## 9. 本轮并行子 agent 交付与独立复核

| 子 agent | 交付 | 结果 |
|---|---|---|
| D2（S4 看板） | `scripts/m5_seg_dashboard.py` 七面板 + `tests/test_dashboard_v5_panels.py`（8 项） | 34 passed；主 agent 用浏览器实检两页（UNKNOWN 渲染成文字、来源字段逐行可追） |
| F2（独立复核 T01–T16） | `docs/T14_T01_T16_REVIEW_20260926.md` | 8 covered / 8 partial / 0 missing；点名 3 处真实缺陷，其中 2 处本轮修复（逐场景计数键空间、去重不进评价路径），第 3 处（标定另抄公式）本轮统一到 `candidate_metrics` |
| G（T15 损失方向） | `tests/test_seg_losses_direction.py`（15 项）+ `docs/T15_SEG_LOSS_DIRECTION_20260926.md` | 推翻"权重 2 压 FP"；实测方向由 `beta/alpha` 决定、总方向由 CE 类别权重主导 |
| H（T01/T02 冲突） | `beamng_autopilot/experiments/manifest.py` + `tests/test_manifest_conflicts.py`（11 项） | 82 passed；真实包审计发现 24 个冲突组（见 §2） |
| J（T09/T11 探针） | `scripts/m5_marking_identity_probe.py` + `tests/test_marking_identity_probe_visibility.py`（12 项） | 29 passed；P_frames 恢复单一口径，缺相机/失败帧逐条可见 |
| 旧批次看板 agent | 只读复核（收到停写指令后转 reviewer） | 报出 3 处并发补丁（1 处 NameError 由 D2 修复） |

复核者与实现者分开：F2/旧看板 agent 只读复核，缺口回派给实现者（本轮由主 agent
作为集成者修复 autoloop/标定两处）。

## 10. 未测 / 已知残留

* 一致性验证：按负责人决定不做（"未做一致性验证"作为事实保留）。
* 逐场景候选覆盖/身份/角色的**门控**仍是只上报：逐场景候选门槛尚未标定
  （方案 §S3.4 要求接入，但标定证据不足时不得随意加门；记录为缺口）。
* 旧判定文件（pre-v5）不可按新分母重判：`legacy_replay_note` 明确说明，不补 0。
* 预览包 22 张冲突图的重新标注（可选）：它们不进测量分母；若要恢复，需人工确认
  哪份标签为准，本轮不替负责人决定。
* 时延复测、驾驶验收、最终集确认：未做（见 §6/§7/R2）。
