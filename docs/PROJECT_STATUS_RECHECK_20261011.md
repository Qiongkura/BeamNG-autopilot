# 当前项目状态复查（2026-10-11，T18 与 E5 扩展收口）

## 已完成

- 论文三轮审查修正、20 条参考文献题录核验、证据归档（531 文件 + SHA-256）。
- T17 感知节拍/几何诊断：代码回归与两臂 Tech 诊断完成；deadline、独立真值、ACK 等限制保留。
- T18 传感器回放正确性：首轮无效保留、同协议重试合格、48 次同源输出一致、CPU p95 对比完成；定向/全量测试通过。
- E5 基线 3×3 资格入口：9/9 尝试，1 completed、8 unplaceable；生产放置偏差与 exposure qualification 均记录。
- E5 扩展：先前 `production_alignment` 批次 2 qualified/2 exposure-invalid；随后用 `fixed_actual_pose` 契约重跑，Italy 更大偏移 4/4 在曝光前因 2.57–3.08m 漂移被拒（rc=4），mountain 第二路段 4/4 放置截止（rc=3）；结果见 `docs/T16_E5_EXTENSION_RESULT_20261011.md`。
- 收尾后游戏进程为 0；纸面数字审计最新批次 129 项通过（2026-10-11 扩展，含 E5 起点A/B 回查；早先批次为 115 项）、图 QA 0、LaTeX 静态 0。

## 当前不能声称

- E5 扩展没有产生第二路段合格驾驶证据；unplaceable 不是驾驶失败/安全失败。
- 大偏移批次的两个 qualified 仍不能与基线做等曝光因果比较（另两项 exposure-invalid）。
- T18 的 CPU p95 改善不是驾驶安全、执行 ACK 或模型晋级证据。
- 第二地图 E4 仍被逐地图 `line-class` 真值通道阻断。

## 尚待用户决定/专轮

- 公开上传本地证据归档到 Zenodo（当前只生成 `logs/paper_release/` 本地 zip）。
- ORCID、目标期刊、中文版是否投稿、邮箱是否放首页。
- 驾驶正确性两个 P0 已修复并通过全量门冻结（pytest 3404 + 深度离线 ALL PASS，gate_20261011_171740）：只读诊断归还 `if scene is None` 回退、诊断移到选择后只读；最终发送前新增时效重判（发送年龄=发布年龄+单调时长下界，>1.5 s 按既有硬停收尾）。冻结后实机验证 1 轮：44/44 帧几何链 cmd_seq 连接一致、发送门无过期油门；运行曝光不足 ~1.1 s 记为 invalid 原样保留。见 `T16_SCENE_AND_SEND_GATE_FIX_20261011.md`、`T16_SCENE_SEND_GATE_LIVE_PREREG/RESULT_20261011.md`。批次随后补齐阶段一第 3 项（曝光前真实帧预热+状态清理、`send_chain_ms` 后链计时）并重新冻结（f64d6bd，pytest 3406 + 深度离线 ALL PASS）；第二轮合格曝光验证：发送门新鲜帧静默、47/47 几何链连接、后链 p50 63 ms。
- 起点位姿 A/B 已按预注册完成（O→B→B→O，各臂 2 次）：3/4 合格（B×2、O×1），1 次 O 臂重复 exposure-invalid 原样保留。三次合格运行等曝光（24.61–24.79 s）；B 臂重复净行进 3.944 vs 1.573 m（2.5 倍），唯一合格 O 观察（3.484 m、20 帧被接受路径）落在 B 重复区间内——会话级变异大于起点位姿差异，无因果分离可声称。见 `docs/T16_E5_AB_RESULT_20261011.md`。此前 +5 m 反向标定观察（0.0209 m/0.087°）作为契约可实现证据保留；偏移方向为沿航向（实际前移 7.85 m、+2.19°），不是路面横向。
- 阶段二推进（2026-10-11 下午）：停车归因（91/91 复算一致、五类归因、B1=0 无误停，`62e78bd`）；`no_path_mechanism` 派生字段（`ed1c3b8`）；回合级线证据占比入审计 + 配对配比试点预注册（`eb1c66d`）。试点结果：去掉 80% 无线帧池在两个配对种子上 dev line IoU +0.041/+0.039（方向支持稀释假说），ArmA 复现 R3 baseline 0.480≈0.464；2 种子探索性，模型未晋级，未改任何门。见 `T16_STALL_ATTRIBUTION_RESULT_20261011.md`、`T16_DOSE_PILOT_PREREG/RESULT_20261011.md`。
- 负例剂量正式配对完成（2026-10-11 晚）：4 档剂量 × 5 配对种子 = 20 训练 + 20 评测，0 失败。d100−d20 配对差 **5/5 种子为正（+0.087±0.116）**且臂内方差最小（0.049）；中间剂量方向被种子噪声掩盖。稀释假说正式复现并加强；不改门、模型未晋级。见 `T16_DOSE_SWEEP_PREREG/RESULT_20261011.md`。
- 第二线帧池族采集（预注册 a5cab37）：8 站采集→导出准入 3 池 9 帧全无线负例，含线站点被三道冻结门全部隔离（投影残差/标注覆盖/外观一致率）；回查证实 R2 也从未准入含线池——**新线真值供给被「生成线→标注验证」链结构性阻断**（与 E4 同类），解链需代码侧对齐修正而非降门。正控制未获得，如实记录。见 `T16_LINEPOOL_COLLECTION_RESULT_20261011.md`。
- 线帧池阻断的三轮离线诊断（191033a/696fbe4/9b2397c）：相机基架假设否定、径向尺度否定、残差=链弧长单调函数 → **根因=生成器弯/坡道站点真值链与贴花沿弧分离**。浮点深度标定帧已实现冻结（f845c9a，门 3415 绿）；仪表化重采集两次因 Tech 启动握手失败推迟（环境级，非代码）。
- 采集器相机姿态链已修复（26c1835，pytest 3417 + 深度离线 ALL PASS）：保存创建时车体系 `mount_dir_local`，逐帧用当前车辆基架正向变换到世界系；旧版每帧重瞄 `site.dir` 会在坡/弯道吞掉挂载俯仰/侧倾，形成伪投影残差。Tech 重采集验证待启动环境恢复。
- Tech 恢复后窄化验证（2026-10-12，端口 50000）：修复后的相机局部光轴采集成功，known_line 覆盖/外观均 1.0，`_depthcal.npz` 生成；但冻结投影门仍 `after_px_median=1.271>1.0`，未获准入。浮点深度原始 buffer 已落盘但通用 near/far 反解背景饱和（真值点 affine RMSE 4.15m），暂不能定量链-贴花分离；跨池族配比仍锁定。见 `T16_LINEPOOL_COLLECTION_FINAL_20261012.md`。
- non-shared depth 编码校准已完成：known4 53 个投影 truth points、每五点 holdout；5–10m p95 0.221m、10–15m p95 0.636m 支持，15m 外 p95 2.27–6.60m 拒绝；known3 无 depthcal，保持 UNKNOWN。工具与报告见 `scripts/m5_depth_calibration_audit.py` / `docs/T16_DEPTH_CALIBRATION_RESULT_20261012.md`。这不是链-贴花修复或安全证明。
- 多站点/多距离链-贴花验证（预注册 5ce3688，2026-10-12）：known_line/material_mix/slope_curve 三站投影冻结门 3/3 通过，导出准入 3/3、9 engine_verified 线帧；depth 支持范围按站点收敛为 known_line 5–15m、material_mix 5–10m、slope_curve UNKNOWN（holdout不足），未外推。旧池15帧+新池9帧审计 group/content/exposure overlap 全 0，第二线帧池族正式解锁后续跨池族配比实验。见 `T16_DEPTH_SEPARATION_MULTISITE_RESULT_20261012.md`。
- 第二线帧池已正式入库并完成跨池配对（预注册 8fb2bc7，2026-10-13）：human 子集 vs engine 多站点池各 9 帧、5 配对种子、60 步，20 作业 0 失败；E−H line IoU 5/5 正向，mean +0.1003±0.069，paired verdict=candidate_better；输入审计 group/content/exposure overlap 全 0。边界：不解释为 provenance 普遍优越、不晋级模型/驾驶。见 `T16_CROSSPOOL_RESULT_20261013.md`。
- 跨池第二独立重复（预注册 da231ca，2026-10-13）完成：10训练/10评测、UNKNOWN=0；E−H line IoU −0.0574±0.2356、正向2/5、paired verdict=inconclusive；E seed48/51 被 collapse diagnostic 标出，排除后诊断均值 +0.0972 但不替换原始 verdict。第一轮 +0.1003/5正向未被复现，池族优势不能稳定声称；下一步应扩大池族/样本或步预算扫描。见 `T16_CROSSPOOL_REPEAT_RESULT_20261013.md`。
- 跨池步预算扫描（预注册 cf74188，2026-10-13）完成：30/60/120 steps × H/E × 5 seeds，共30训练+30评测、0失败/UNKNOWN。E−H line IoU：30步 −0.0957±0.1354（1/5）、60步 −0.0622±0.2297（2/5 collapse2）、120步 −0.2139±0.1340（0/5，champion_better）；结果支持小池/初始化/步预算交互主导，禁止稳定 engine 池优势主张。见 `T16_CROSSPOOL_BUDGET_RESULT_20261013.md`。
- 通道资格掩码训练支持已完成并冻结（b4b0a41）：LINE/ROAD 的 UNKNOWN、N-A、unverified 从损失与 IoU 分母移除，SHOULDER/ROLE 状态保留并审计；混合 measured/unverified batch 的区域损失支持逐样本 mask；全量 pytest 与深度离线均 PASS。见 `docs/T16_CHANNEL_QUALIFICATION_MASK_RESULT_20261013.md`。
- 真实通道凭证小训练回归完成（2026-10-13）：engine_verified annotation.json/truth_provenance 实测输入，2 steps CPU；hist/checkpoint 均记录 LINE=unverified、ROAD=measured、SHOULDER=unknown、ROLE=unverified，6/6 帧 masked，line IoU 分母未伪造；loss/梯度有限且训练正常结束。见 `T16_CHANNEL_MASK_SMOKE_RESULT_20261013.md`。
- 跨池 240-step 独立预算重复（预注册 6276f8c）完成：10训练/10评测；line IoU E−H −0.4106±0.0426、recall −0.4214±0.0447，均 0/5 正向、paired verdict=champion_better；precision 5/5 UNKNOWN（无有效正例分母）。结果进一步否定稳定 engine 池优势，见 `T16_CROSSPOOL_BUDGET240_RESULT_20261013.md`。
- 第三个独立池 H2 跨池配对（预注册 ac38003，2026-10-13）：10训练/10评测0失败；H2 human 臂5种子 line IoU全0，E−H均值+0.0119±0.0164、正向2/5、inconclusive；recall +0.0123±0.0171、2/5；precision 5/5 UNKNOWN。不能解释为来源质量排序，进一步说明小池/内容/预算交互主导。见 `T16_CROSSPOOL_H2_RESULT_20261013.md`。
- 更大真实凭证池掩码回归（预注册 a5d85da，2026-10-13）：3个engine-verified场景族、24帧原始/240训练消费、5 seeds、5训练+5评测0失败；每个hist/checkpoint均记录 ROAD measured 240、LINE unverified 240、SHOULDER unknown 240、ROLE unverified 240、masked 240；line IoU/recall全0，precision 5/5 UNKNOWN，未伪造差值或collapse。见 `T16_LARGEPOOL_MASK_REGRESSION_RESULT_20261013.md`。
- LINE=measured 独立真值池已打通（2026-10-13）：三受控站点/9帧冻结门通过；修复 exporter 使 `label_source=engine_verified` 且 `truth_provenance.labels.channel_valid_area` 写入 LINE/ROAD measured、SHOULDER/ROLE unknown；真实小训练 2 steps 验证 hist/checkpoint 为 LINE/ROAD measured、masked=0，loss/梯度有限。见 `T16_LINE_MEASURED_POOL_RESULT_20261013.md`。
- 第二个 LINE=measured 独立池族训练回归（预注册 859319b，2026-10-13）：实际 3 帧新池，5 seeds/5训练+5评测0失败；每 seed LINE/ROAD measured 180、SHOULDER/ROLE unknown 180、masked=0；IoU mean 0.2660±0.0767、UNKNOWN 0/5、collapse 0。池太小，结果仅证明资格接线可消费 LINE=measured，不作收益/泛化主张。见 `T16_LINE_MEASURED_LARGE2_SMOKE_RESULT_20261013.md`。
- 第三 LINE=measured 大池族采集完成（2026-10-13）：5 anchors/5站点/15帧目标，3站冻结准入/9帧 engine_verified，LINE/ROAD measured、SHOULDER/ROLE unknown；2站隔离原因保留，数据审计 rejected=0 且 group/content/exposure overlap=0。见 `T16_LINE_MEASURED_LARGE3_RESULT_20261013.md`。
- 第三 LINE=measured 池配对训练完成（预注册 4c49880，2026-10-13）：E3 engine 9帧 vs H human 9帧，5 seeds、10作业0失败/UNKNOWN；E3−H IoU +0.1140±0.0644、recall +0.1364±0.0859，均5/5正向、collapse=0；precision −0.0149±0.0639、2/5正向、inconclusive。证明第三池可进入后续配对研究，不作来源普遍优势主张。见 `T16_CROSSPOOL_E3_RESULT_20261013.md`。
