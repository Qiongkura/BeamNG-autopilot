# 当前项目状态复查（2026-10-11，T18 与 E5 扩展收口）

## 已完成

- 论文三轮审查修正、20 条参考文献题录核验、证据归档（531 文件 + SHA-256）。
- T17 感知节拍/几何诊断：代码回归与两臂 Tech 诊断完成；deadline、独立真值、ACK 等限制保留。
- T18 传感器回放正确性：首轮无效保留、同协议重试合格、48 次同源输出一致、CPU p95 对比完成；定向/全量测试通过。
- E5 基线 3×3 资格入口：9/9 尝试，1 completed、8 unplaceable；生产放置偏差与 exposure qualification 均记录。
- E5 扩展：先前 `production_alignment` 批次 2 qualified/2 exposure-invalid；随后用 `fixed_actual_pose` 契约重跑，Italy 更大偏移 4/4 在曝光前因 2.57–3.08m 漂移被拒（rc=4），mountain 第二路段 4/4 放置截止（rc=3）；结果见 `docs/T16_E5_EXTENSION_RESULT_20261011.md`。
- 收尾后游戏进程为 0；纸面数字审计最新批次 **261 项通过**（2026-10-18 扩展，含 ECA3 资格与配对；批次演进 115 → 129（10-11）→ 240 → 257 → 261）、图 QA 0、LaTeX 静态 0。审计计数目前只存在于提交信息与本文档，尚无落盘产物。

## 当前不能声称

- E5 扩展没有产生第二路段合格驾驶证据；unplaceable 不是驾驶失败/安全失败。
- 大偏移批次的两个 qualified 仍不能与基线做等曝光因果比较（另两项 exposure-invalid）。
- T18 的 CPU p95 改善不是驾驶安全、执行 ACK 或模型晋级证据。
- 逐地图 `line-class` 真值通道阻断已解除（west 2026-10-13 准入、east ECA3 2026-10-18 准入），但**地图资格≠学习收益/泛化**：east ECA3 池（2站/6帧）首个配对数据点为负（E3−H IoU −0.2684，5/5 负、champion_better，排除 2 个崩坏对后仍 −0.1690），与 west 小池多数方向一致；**来源排序/泛化仍未建立**——三图全部是小池、同初始化单预算对照。

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
- 更大 LINE=measured 合并池配对完成（预注册 3d64dd0，2026-10-13）：E-large 18帧 vs H 9帧，5 seeds/10作业0失败；IoU E-H +0.0245±0.1177、recall +0.0279±0.1376、precision +0.0037±0.0316，均2/5正向、collapse/UNKNOWN=0、全部inconclusive。池扩大后第三池+0.114优势消失，继续支持池内容/种子/预算交互而非来源主效应。见 `T16_CROSSPOOL_LARGE_RESULT_20261013.md`。
- 第四 LINE=measured 独立池族采集完成（预注册 d7ba5bf/corrected offset100，2026-10-13）：5 anchors中4站采集、2站冻结准入/6 engine_verified帧，2站因 projection 1.54/1.29px 隔离；审计6帧trainable、overlap全0。见 `T16_LINE_MEASURED_LARGE4_RESULT_20261013.md`。
- 第四 LINE=measured 池配对训练完成（预注册 788dcfb，2026-10-13）：H9 vs E4 6帧，5 seeds/10作业0失败/UNKNOWN；E4−H IoU −0.2222±0.2925（1/5，inconclusive），recall −0.1959±0.3420（2/5），precision −0.4295±0.3911（0/5）；E4 seed49/51 collapse，排除后诊断均值不替换原始裁决。资格通过≠学习收益，模型未晋级。见 `T16_CROSSPOOL_E4_RESULT_20261013.md`。
- 第五 LINE=measured 池扩展尝试收口（预注册 902442e/修订，2026-10-13）：offset140无anchor、offset130五链均过短、offset110采集挂起，均保留失败原因；本批次无新增准入池，已停止盲扫。下一扩展应改不同地图/道路档案或先扩展anchor规划容量。见 `T16_LINE_MEASURED_LARGE5_RESULT_20261013.md`。
- west_coast_usa 第五地图尝试（预注册 c59bafe，2026-10-13）：3 known_line站点/9帧全部隔离，coverage/annotation line/appearance均无证据，准入0；启动与清理通过，未降门、未入训练。与E4地图线通道阻断一致，需材质/渲染对齐修复。见 `T16_LINE_MEASURED_WCU2_RESULT_20261013.md`。
- west_coast 高程对齐修复（88f1bde）已通过纯逻辑/全量 pytest；重采集仍 0 annotation line pixels、3/3隔离，说明高程不是唯一阻断，地图材质/渲染可见性链仍需排查。
- west_coast palette 已确认 API 映射可用（SOLID_LINE/DASHED_LINE存在），但运行证据显示 road-chain mid.z≈144.7、ground/vehicle/camera z≈213.6，约69m高程坐标不一致；ground对齐初修后仍未消除，下一步需场景加载后重建/平移受控道路对象。
- west_coast 修复后冻结准入已成功（2026-10-13）：干净pytest通过后，修复入口获得 1站/3帧 `engine_verified` LINE/ROAD measured，2站隔离（coverage 0.7308、projection 1.15px），审计3帧trainable、overlap全0、cleanup归零。此前地图全隔离问题已被 `Scenario(map_name)` 硬编码修复解除。见 `T16_LINE_MEASURED_WCU_FINAL_RESULT_20261013.md`。
- west_coast LINE measured配对训练完成（预注册 aa16cdf，2026-10-13）：W3帧 vs H9帧，5 seeds/10作业0失败/UNKNOWN；W−H IoU −0.1709±0.1497、recall −0.1753±0.1430均0/5正向、inconclusive；precision −0.0590±0.2680、1/5正向，W seed49 collapse。准入链成功不等于收益，模型未晋级。见 `T16_WEST_COAST_CROSSPOOL_RESULT_20261013.md`。
- west_coast 第二批扩展完成（预注册 6e20299，2026-10-13）：offset3/5 anchors采集，2站/6帧 `engine_verified` LINE/ROAD measured 准入（projection median 0.619/0.488、coverage 1.0、appearance 1.0/0.9048），3站隔离原因保留；审计6帧trainable、overlap全0、cleanup归零。见 `T16_LINE_MEASURED_WCU3_RESULT_20261013.md`。
- west_coast 第二批 LINE measured 配对训练完成（2026-10-13）：W3 6帧 vs H 9帧，5 seeds/10作业0失败；W3−H IoU −0.2374±0.1786、recall −0.2499±0.1808 均0/5正向、champion_better；precision −0.1263±0.3158、3/5正向、inconclusive；W3 seed49/51 collapse。模型未晋级、硬门未改。见 `T16_WCU3_CROSSPOOL_RESULT_20261013.md`。
- west_coast 场景加载后ground重测/Scenario重建验证完成（b6034b0，2026-10-13）：3站采集，2站/6帧冻结准入 engine_verified LINE/ROAD measured，1站 projection 1.15px隔离；审计6帧trainable、overlap全0、cleanup归零。已解除全地图线通道失效问题。见 `T16_LINE_MEASURED_WCU_RELOAD_RESULT_20261013.md`。
- west_coast修复后准入池配对完成（预注册 98630f7，2026-10-13）：Wfix6帧 vs H9帧、5 seeds/10作业0失败/UNKNOWN；IoU −0.2211±0.0614、recall −0.2035±0.0459、precision −0.2382±0.1866，三项均0/5正向、champion_better、collapse=0。准入通过不等于学习收益，模型未晋级。见 `T16_WCU_RELOAD_CROSSPOOL_RESULT_20261013.md`。
- west_coast 第三批扩展完成（预注册 6e20299，2026-10-13）：offset6/5 anchors采集，3站/9帧 engine_verified LINE/ROAD measured 准入，2站 appearance 0.771/0.452 隔离；审计9帧trainable、overlap全0、cleanup归零。见 `T16_LINE_MEASURED_WCU4_RESULT_20261013.md`。
- west_coast 第三批9帧配对训练完成（预注册 f45eeb1，2026-10-13）：W4 vs H，5 seeds/10作业0失败/UNKNOWN；IoU −0.1072±0.0960、recall −0.0844±0.1158（均1/5正向、inconclusive），precision −0.1600±0.0850（0/5，champion_better），collapse=0。模型未晋级、硬门未改。见 `T16_WCU4_CROSSPOOL_RESULT_20261013.md`。
- west_coast 第四批扩展完成（预注册 a63c437，2026-10-13）：offset9/5 anchors采集，3站/9帧 engine_verified LINE/ROAD measured 准入，2站 appearance 0.446/0.488（含1站投影残差 1.56px）隔离；审计9帧trainable、overlap全0、与 WCU4 跨池内容零重叠、cleanup归零。见 `T16_LINE_MEASURED_WCU5_RESULT_20261013.md`。
- west_coast 第四批9帧配对训练完成（预注册 2ee89cf，2026-10-13）：W5 vs H，5 seeds/10作业0失败/UNKNOWN；IoU −0.1345±0.0568、precision −0.2094±0.1562（均0/5正向、champion_better），recall −0.0817±0.1022（1/5、inconclusive），collapse=0。与 WCU3/WCU4 方向一致，模型未晋级、硬门未改。见 `T16_WCU5_CROSSPOOL_RESULT_20261013.md`。
- `run_scene` asset 块地图名硬编码修复（22cc12f，2026-10-13）：定位到 `m5_controlled_scenes.py` batch 元数据 `"asset": {"map": "italy"}` 硬编码——west_coast 批次证据的 `verification.*.stats.asset.map` 误记为 italy。已改 `str(map_name)` 并加回归断言。影响边界：冻结 `truth_provenance.asset.map` 由导出侧 `--map` 显式写入，全部为正确 west_coast_usa，准入与训练不受影响；误记只存在于原始采集证据的嵌套 stats 中，原样保留不回写。
- west_coast 第五批扩展完成（预注册 91739bc，2026-10-13）：offset12/5 anchors采集，批次验收 5/5 合格，冻结 gate 仅 1站/3帧（a2s0）准入，4站隔离（3站投影残差 1.07/1.56/1.60px 边缘超限、1站 annotation 线通道失效 coverage 0.13/0.0）；**采集目标 ≥2站/≥6帧未达成，不降门**；审计3帧trainable、overlap全0、与全部既往导出池+H 跨池零重叠（81帧81唯一）；asset.map 修复生效（全部证据 west_coast_usa）。见 `T16_LINE_MEASURED_WCU6_RESULT_20261013.md`。
- west_coast 第五批3帧配对训练完成（预注册 WCU6-CROSSPOOL，2026-10-13）：W6 vs H，5 seeds/10作业0失败/UNKNOWN、collapse=0；IoU +0.1161±0.1195（4/5正向）、recall +0.1229±0.1372（4/5）、precision +0.0195±0.0467（3/5），三项均 **inconclusive**——六池中首个均值方向为正，但五种子下不构成 candidate_better。六批聚合 30 配对 25 负、3 champion_better、从未 candidate_better：单池方向可整体翻转，小池/种子/预算交互主导的结论不变。模型未晋级、硬门未改。见 `T16_WCU6_CROSSPOOL_RESULT_20261013.md`。
- west_coast 聚合池36帧配对完成（预注册 T16_WCU_POOL_PREREG，2026-10-13）：WALL（12站/6批全部已准入帧聚合）vs H，5 seeds/10作业0失败/UNKNOWN；IoU −0.1918±0.0952（0/5、champion_better、1 collapse seed50排除均值 −0.1747）、recall −0.1601±0.1113（0/5）、precision −0.3480±0.1895（0/5），三项全 champion_better——**池扩大到 4 倍没有恢复任何学习优势**。复现性发现：H 臂同种子跨运行非逐位一致（seed48–51 IoU 最大差 0.0356、recall 0.0375、precision 0.0094，GPU 非确定性），跨实验比较带约 0.036 IoU 噪声下限；WCU6 初稿「H 臂完全一致」已撤回更正。模型未晋级、硬门未改。见 `T16_WCU_POOL_RESULT_20261013.md`。
- east_coast_usa 第三图 LINE=measured 扩展（2026-10-18）：ECA1/ECA2 两批同锚点族均 0/5 全隔离（ECA1 初判"z 埋地 28m"**已撤回**——ECA2 五站 ground 重读 delta 全≈0，54.2 实为 ego 出生点地形读数、与站点分处两地）。诊断（`m5_decal_annotation_probe.py` 5 场景变量分离，诊断输出不进训练池）：z 抬升/renderPriority/插入序/宽度/over_objects 全无效，**根因=annotation pass 的 alpha 测试剔除**——east 线材质 opacityFactor 0.7×bcf alpha 0.8–0.9→最终 alpha≤0.56 对 alphaRef=255 全像素剔（west 线材质无 opacity 缩放所以正常）；唯一可写替代 line_yellow_double（alphaTest=True/alphaRef=0），宽度扫描 0.40m 三门全过（appearance 1.00×2）、0.15m 挂、0.50m 回落。见 `T16_LINE_MEASURED_ECA2_RESULT_20261018.md`、`T16_LINE_MEASURED_ECA1_RESULT_20261018.md`（更正节）。
- east 线材质/线宽生成器修复入库（863c828）：MAP_PROFILES east 白/黄→line_yellow_double@0.40m（west/italy 合同宽度 0.15m 不变），冻结门一字不改；回归测试含对真实材质资产的 annotation 可写性断言（alphaTest/alphaRef 准入规则）。
- east_coast 第三批重采完成（预注册 T16_LINE_MEASURED_ECA3_PREREG，2026-10-18）：唯一变更因子=线材质/线宽，同锚点族同门；**2站/6帧 engine_verified 准入（east 首次达标，目标 ≥2站/≥6帧达成）**——a0s0 coverage 0.9889/proj median 0.598px/paint 0.9941，a4s0 1.0/0.587px/0.9941；3站隔离原因逐实例保留（a1 三门全挂、a2 right 实例 coverage 0.4216、a3 paint 0.0536 被标签中毒守门拦截）；审计6帧trainable、overlap全0、cleanup归零。不降门、不消费最终集、模型未晋级；配对训练另行预注册。见 `T16_LINE_MEASURED_ECA3_RESULT_20261018.md`。
- east_coast 首批 6 帧配对训练完成（预注册 T16_ECA3_CROSSPOOL_PREREG，2026-10-18）：E3（ECA3 池 2站/6帧）vs H（9帧 human），5 seeds/10 作业 0 失败/UNKNOWN=0；IoU −0.2684±0.1617（0/5、champion_better）、recall −0.2846±0.1645（0/5、champion_better）、precision −0.3281±0.5059（3/5、inconclusive）；collapse=2 对（均 E3 臂，seed47 IoU 0.0013、seed49 三项全 0.0 完全退化），排除后 IoU −0.1690/recall −0.1853 仍为负。H 臂跨运行（vs WCU6 同种子）IoU 最大差 0.0190，在 0.0356 噪声下限内。预注册内跨池审计：105 帧 105 唯一内容摘要、0 重复组。**第三图首个学习数据点不支持任何来源优势**；模型未晋级、硬门未改、不消费最终集。见 `T16_ECA3_CROSSPOOL_RESULT_20261018.md`。
- utah 资格采集第一批（预注册 T16_LINE_MEASURED_UTAH_PREREG，2026-10-18）：annotation 可写性修复生效（5 站全写入线类像素 3434/799/2948/1350/5425，line_yellow_double@0.40m），但冻结导出门 **0/5 全隔离、0 帧**（目标 ≥2站/≥6帧未达成）——a0s0 left 实例 paint 0/85（annotation 写进、beauty 全程无漆的站点局部遮挡案例，RGB 均值 ~(106,98,74) 对 right 条纹色 (185,161,124)）、a1/a2 投影 2.70/1.66px 挂、a3 right coverage 0+14.35px、a4 right 0.5472+4.27px（逐帧 1.6→6.6px 随步进漂移，需原地 vs 步进对照定因）；首次启动挂起（游戏 t=4.85s 卡死、collector 阻塞 MapLoaded ack）杀进程同命令重试成功，attempt1 日志保留；不降门、无帧入池、模型未晋级。见 `T16_LINE_MEASURED_UTAH_RESULT_20261018.md`。
- utah a0s0 left 线无漆定因完成（2026-10-18，纯离线已采集帧）：**无几何遮挡物**——世界系直线阴影边界浅斜穿过走廊、左线整段处于阴影，annotation 无光照直写调色板照常写线、beauty 同几何阴影渲染 0.56–0.66×（仅环境光，(106,98,74)=(185,161,124) 阴影化）；`paint_like` 绝对亮度判据对阴影不具照度不变性故 left 0/85。已排除覆盖贴花（footprint 逐行等宽）、近距遮挡物（depth 同）、ego 阴影/视角 mip（暗区世界系固定且随步进退去）、条纹材质差异（路面同倍率变暗、条纹亮暗随边界相对位置逐帧变化）。门不改、a0s0 维持隔离。见 `T16_UTAH_A0S0_OCCLUSION_ATTRIB_20261018.md`。
- utah 投影残差两项定因完成（2026-10-18，纯离线已采集帧）：**a4s0 步进漂移**=我方两贴花渲染范围世界固定止于 stn≈15（链 4→33），stn 16–33 真值点为假残差、服务像素随步进由地图自有车道漆续线切换到中带/邻带（f0 0.3–6.9px → f1 8.9–12.2 → f2 15.7–17.9），left 全程 ~1px 自证；位姿链（2.00m/≤21mm/≤1°）、相机标定、链-贴花横移、埋地（表面 z−链 z ±0.12m）、mip 截断（截断站位不随步进平移）、阴影型写入失败均排除。**a3s0 14px**=右贴花对象级双 pass 全缺（annotation R=0 像素、beauty 土色 (151–163,113–124,93–103)），右链真值点全程量到左带 12–96px 假残差，pooled 14px≈(left ~1px+right ~27px)/2；paint 84/84 为松黄色判据在红土 (170,134,107, r−b=63≥60) 假阳性。projection 与 coverage 两门失败同根因，门未改、0 帧入池、模型未晋级。见 `T16_UTAH_PROJECTION_ATTRIB_20261018.md`。
- utah 贴花截断根因判决 + 修复 + 重采挂起（2026-10-29）：引擎探针两轮（预注册 `T16_UTAH_DECAL_EXTENT_PROBE_PREREG_20261029`）变量分离判决——**单 Scenario 10 条贴花路会让后创建的路渲染截断**（单站 2 路全跨渲染 22-34m bin solid 11/11；同几何放进批次 10 路结构后 idx9 的 22-34 bin 全灭、beauty 同步塌陷；线位/坡度/节点 z 提升/优先级/多站资源位置均排除，lift 序列反向有害、显式高优先级更差），与 east ECA3 "a2 right coverage 0.4216" 同签名。修复=非 GROUND_ALIGN_MAPS 地图每站一个 Scenario（scripts 53850c1，回归测试 41 通过，冻结门一字未改；GROUND_ALIGN_MAPS 加载后 ground 重检依赖全量 Scenario、保持原流程另行迁移）。重采（预注册 `T16_LINE_MEASURED_UTAH2_PREREG_20261029`，同协议同门）attempt1 首个每站 Scenario 加载即遭模拟器断连（BNGDisconnectedError，无帧、无 collect 目录，证据=会话日志 traceback；与 2026-10-18 批次 attempt1 启动挂起同属首启不稳定族，协议允许杀进程同命令重试一次）——**重试待执行**。探针产物 diag_decal_probe2_utah_20261029_*（probe_report.json 落盘）不进训练池。见 `T16_UTAH_DECAL_EXTENT_PROBE_RESULT_20261029.md`。
- utah 重采达标 2站/6帧 + 外部证据删除事件（2026-11-01）：attempt1 断连后同命令重试成功（首启不稳定族第二次复现），**批次验收 5/5 全合格**（10-18 为 0/5 全隔离；覆盖 0.87–1.0、标注线像素 777–5921/站），截断机制在生产批次消失；**冻结导出门准入 2 站/6 帧 engine_verified（utah 首次达到 ≥2站/≥6帧 目标）**——a3s0 0.398px/paint 1.0（修复前右覆盖 0+14.35px）、a4s0 0.936px（修复前逐帧 1.6→6.6px 漂移）；隔离 3 站原因逐站保留（a0s0=已定因阴影 median 1.05px+paint 左 0/86、a1s0 2.86px、a2s0 1.76px——后两站为纯投影几何新前沿）。**外部删除事件**：10-29 夜至 11-01 间 `logs/m5_seg` 整目录与 `logs/experiments` 历史产物被外部删除（无保留清单、非空间触发、非采集器所为）——v13b 部署模型、全部历史池帧、10-29 探针报告损失；幸存 git 全部 docs/代码/测试、bc_best.pt、presence_model、v13b holdout 评测（**line_iou_near=0.116 为标线识别幸存绝对基线**）。对策：本批证据即时哈希归档（`logs/evidence_archives/t16_utah2_20261101_evidence.zip`，35 项 complete）+ 保留清单加固。见 `T16_LINE_MEASURED_UTAH2_RESULT_20261101.md`。
- 采集扩量收口（2026-11-01，用户指令：批次 3 后停采）：批次 2（offset 5）导出 **3 站/9 帧**（median 0.592px，a0s0/a3s0/a4s0）；批次 3（offset 10，最后一轮）批次 4/5、**导出 0 帧**（median 2.058px，锚点组几何更难，原因逐站保留）。**认证池最终规模 3 站点族/15 帧 engine_verified**，三批证据全部哈希归档（evidence_archives/ 3 件 105 项）。下阶段标线数据两靶点：投影前沿（1.1–14px）与左实例部分覆盖缺失，均不降门；重训待用户指令（管线就绪：m5_train_seg.py + m5_eval_seg.py 同口径 holdout，基线 line_iou 0.116）。
- 线分割认证池首训（2026-11-02，预注册 T16_LINE_SEG_RETRAIN_PREREG_20261102）：4 族/12 帧训练、留一族（offset5-a4）同口径评测——内部相邻帧 val_lineIoU 峰值 0.773（ep24）→ 同族实际 0.2975 → **跨族留出 0.0000**（mIoU 0.2734，模型输出零线像素，数据线像素占比 0.038 正常）。现有光度/几何增广全覆盖仍跨族归零 → **数据广度（站点族/地图/光照）是绑定约束，增广非瓶颈**；对 v13b 0.116 为非同集对照、不可声称超越；模型不晋级，权重+评测已哈希归档。见 `T16_LINE_SEG_RETRAIN_RESULT_20261102.md`。
- 数据族扩量 + 多族重训（2026-11-02，预注册 T16_LINE_FAMILY_EXPAND_PREREG / T16_LINE_SEG_MULTIFAMILY_PREREG）：east 迁移每站 Scenario 后批 0 帧（右实例 paint 系统 0.48–0.80，阴影/光照族嫌疑，原因保留）；utah offset15/20/25/30 四批导出 **2+3+4+1 站**，认证池 5→**15 族/45 帧**（每批即时哈希归档，15 件归档全 complete）。3 折分组留出重训：跨族 line IoU **0.0000 → 宏平均 0.2792**（A 0.3225 / B 0.2319 / C 0.2832，各 15 帧留出全未见），与首训同族 0.2975 同量级、v13b 0.116 的 2.4×（外部参照）；**数据广度杠杆实证成立**。距"顶尖"仍有 0.68–0.77 留出缺口、折间方差大（池仍小）；模型不晋级。见 `T16_LINE_SEG_MULTIFAMILY_RESULT_20261102.md`。
- east paint 定因 + 回收 + 20 族池宏平均（2026-11-02）：east 右实例 paint 偏低定因为**阴影/光照族**（右线 RGB (98–105,99–103,60–76) 对左线 122–148，lloose 判据绝对亮度下限不具照度不变性，与 utah a0s0 同族机制）；回收策略=受光锚点，east offset10 批导出 **4 站/12 帧**（受光站 paint 0.9645–1.0）。池扩至 **20 族/60 帧**（16 utah + 4 east），3 折宏平均 line IoU **0.2792 → 0.3238（+16%）**（A 0.3543 / B 0.2392 / C 0.3778，跨图 utah→east 迁移成立）；目标达成，模型不晋级。见 `T16_LINE_SEG_POOL20_RESULT_20261102.md`。
- 扩量饱和实测（2026-11-02 续）：utah offset40 **锚点簇耗尽**（4 锚道路链 14–38m 布站失败、1 站隔离，失败证据归档）；east offset15 仅 1 站准入（受光站从 4 降到 1）。池 21 族/63 帧，3 折宏平均 **0.3228 ≈ 0.3238（−0.001 持平）**——扩量边际=噪声级，模型侧/数据侧均入平台期；"顶尖"下一步需换杠杆（多尺度增广代码臂 / 大批量 east 20–60 批 / 标签课程侧）。不晋级。见 `T16_LINE_SEG_POOL21_RESULT_20261102.md`。
- 多尺度增广代码臂完成（2026-11-02 续，预注册 T16_LINE_SEG_AUGSCALE_PREREG）：新增 --aug-scale（缩放抖动+裁/边缘复制回原尺寸，43 测试通过）；21 族池固定 3 折——scale0.15 宏平均 0.3348（名义 +0.012 但折间方差 0.19–0.57 不可靠）、scale0.30 0.2358（−0.087 负收益）。**训练侧杠杆全部探明**（数值旋钮+代码臂均无可迁移增益），平台期 0.323±0.01 保持；模型不晋级。见 `T16_LINE_SEG_AUGSCALE_RESULT_20261102.md`。
- east 大批量扩量 + 30 族池（2026-11-02 续）：offset 20–55 八批连续采（每批即采即导出即归档）新增 **9 站/27 帧**（off45 单批 3 站），认证池 **30 族/90 帧**（16 utah + 14 east）。**严格配对**（留出族集合与 0.3228 时相同、仅训练集 21→30 族）：宏平均 **0.3228 → 0.3850（+19%）**，fold C +0.172；30 族平衡重分 3 折（早期 utah/后期 utah/全部 east）宏平均 **0.5177**（0.3072/0.6493/0.5967）。**数据侧杠杆实证成立**；最弱折仍 0.31（旧稀疏族），距"顶尖"仍有缺口；模型不晋级。见 `T16_LINE_SEG_POOL30_RESULT_20261102.md`。
- 远端扩量转负实测（2026-11-02 续）：east offset60-90 续采——60 批 0 站、65 批 2 站、70 批 1 站、75 批 0、80+ **锚点耗尽**（布站失败）；utah 早期族补强（off0/5 重采 frames4/step3）加 20 帧。池 **33 族/119 帧**。平衡 3 折（早期 utah/后期 utah/east）：0.2803/0.5437/0.5946，**宏平均 0.4729 < 0.5177（−0.045）**——远端/同分布扩量收益转负（新族与既有 east 族高度同质、补强帧扩大早期折留出面）。继续提拉需地图多样性（west/italy 线通道需先修）或训练/标签侧根本改动。模型不晋级。见 `T16_LINE_SEG_POOL33_RESULT_20261102.md`。
- west 第三图线通道开启 + 三图泛化（2026-11-02 续）：每站 Scenario 下 west offset20 批 **4 站/12 帧 engine_verified**（0.62–0.83px，paint 0.988–1.0）——历史全隔离的第三图首次打通；池 **37 族/131 帧**（utah 16 + east 17 + west 4）。4 折：A 早期 utah 0.2795 / B 后期 utah 0.5898 / C east+west 0.5638 / **W west 单独留出 0.4799（训练零 west）**——**跨图迁移（第三图）成立**，同图扩量饱和后跨图泛化仍达同图量级；3 折宏平均 0.4777（折组成随池变，非严格配对）；最弱折仍早期 utah。模型不晋级。见 `T16_LINE_SEG_WEST3MAP_RESULT_20261102.md`。
- west 扩至 18 族：度量集变难、宏平均回落（2026-11-02 续）：west offset25–55 七批新增 **14 族/42 帧**（全池 56 族/169 帧：utah 21/east 17/west 18）。固定 4 折：A 0.2676 / B 0.6542 / C 0.4398 / **W west 单留出 0.3362（37 族池为 0.4799——4 族时是易子集，18 族回归真实难度）**；3 折宏平均 **0.4539**（37 族 0.4777、33 族 0.4729、30 族平衡 0.5177——随池组成波动）。**数据侧扩量边际已全部释放；本 recipe 水平被池组成主导**，继续提拉需质变杠杆（数量级采集/线专用预训练/自监督/真实相机域），非剩余 session 预算可证。模型不晋级。见 `T16_LINE_SEG_WEST_EXPAND_RESULT_20261102.md`。
- 数量级扩量（残余锚点 13 批）：west 新增 13 族/39 帧（east/utah 残余 0，**三图锚点库存用尽**），全池 **69 族/208 帧**（utah 21+east 17+west 31）。固定 4 折：A 0.2877 / B 0.6321 / C 0.4433 / **W west 单留出 0.4449（+0.109）**；3 折宏平均 **0.4544**（持平）。数量级扩量主要补足 west 族多样性；宏平均仍被最弱折（早期 utah 0.29）拉低。后续需新地图（italy 通道修）或训练侧质变（预训练/自监督）。模型不晋级。见 `T16_LINE_SEG_POOL69_RESULT_20261102.md`。
- italy 第四图通道验证（2026-11-02 续）：每站 Scenario 下 offset20 批导出 0/5（最接近 a2s0 median 1.00px）、offset30 批 **2 站/6 帧 engine_verified**（a1s0 0.991px/paint 0.982、a4s0 0.56px/paint 1.0）——**第四图通道可用但产出稀疏**（0.15m 细线覆盖门敏感）。池 **71 族/214 帧（四图）**。71 族固定 4 折：A 0.2334 / B 0.6817 / C 0.4284 / W 0.4319，**3 折宏平均 0.4478**（69 族 0.4544，−0.007 噪声内）。**平台期定格：3 折宏平均 0.448–0.454（56/69/71 族池）**；四图锚点库存全部用尽；A（早期 utah）持续最弱 0.23–0.29。模型不晋级。见 `T16_LINE_SEG_ITALY4MAP_RESULT_20261102.md`。
- 训练侧质变臂（balance-runs）结果（2026-11-02 续）：71 族固定 4 折下 A 0.2274 / B 0.6733 / C 0.4386 / W 0.4395，**3 折宏平均 0.4464 ≈ 0.4478（噪声内）**——族等量采样（v9 线专用课程方向）无迁移增益。**训练侧杠杆全部证伪（8 臂）**；数据侧四图锚点库存用尽（214 帧饱和）。本 recipe 跨族平台 = **3 折宏平均 0.446–0.454**、最弱折 0.23–0.29；剩余需多 session 工程（线专用预训练/自监督或真实相机域）。模型不晋级。见 `T16_LINE_SEG_BALANCE_RESULT_20261102.md`。
- **标线识别研究收口（2026-11-02）**：用户未就多 session 方向答复，按最佳判断接受 0.45 平台并收口。最终状态：71 族/214 帧四图池、跨族 line IoU 轨迹 0.0→0.45、固定 4 折 A 0.2334/B 0.6817/C 0.4284/W 0.4319（3 折宏平均 0.446–0.454）、训练侧 8 臂全部证伪、四图锚点耗尽、模型从未晋级。方法学结论与四条可续接路径（SSL 预训练/真实相机域/新地图/转整车主线）固化于 `T16_LINE_RECOGNITION_CLOSEOUT_20261102.md`；全部证据归档、工作树干净。
- **SSL 预训练突破（2026-11-02 续）**：新建 `scripts/m5_ssl_pretrain.py`（SegUNet 编码器去噪自编码预训练，--init 严格可加载）；各折训练帧无标签预训练 20ep + 同配置微调，固定 4 折配对：A 0.2643（+0.031）/ B 0.7457（+0.064）/ C 0.4261（−0.002）/ W 0.4588（+0.027），**3 折宏平均 0.4478 → 0.4787（+7%）**——**训练侧 8 臂证伪后的首个正迁移**（3/4 折提升，超出配对噪声带），机制=去噪重建迫使编码器学光照/纹理不变特征。预注册确认步骤（重复种子）待做；模型不晋级。见 `T16_LINE_SEG_SSL_RESULT_20261102.md`。
- **SSL 重复种子复核确认（2026-11-02 续）**：同一固定 4 折、配对设计（种子 42/43 各跑基线 vs SSL）——基线 0.4478/0.4776 对 SSL 0.4787/0.5148，**SSL 效应 +0.031/+0.037（平均 +0.034，+7.5%）**，2/2 种子复现；逐折 A +0.03~+0.05、B +0.06~+0.09、C −0.00~−0.02、W +0.03。**训练侧首个可复现正迁移成立**（utah 折收益最大；C 组合折不敏感）。基线自身种子方差 +0.030，配对设计为唯一可信口径。模型不晋级。见 `T16_LINE_SEG_SSL_CONFIRM_RESULT_20261102.md`。
- SSL 超参扫描（2026-11-02 续）：种子 43 筛选 e40 0.4846 / s10(σ0.10) 0.5494 / s02 0.4615（对水位 0.5148）；s10 种子-42 确认为 0.4721（−0.007）——**σ0.10 优势未复现**（折 A 种子间波动 0.26–0.53，超参空间被种子噪声主导）。**σ0.05/20ep 仍为最佳 SSL 配置（配对 +7.5%，2/2 种子）**；继续提拉需更大无标签帧集或容量/架构级改动。模型不晋级。见 `T16_LINE_SEG_SSL_SWEEP_RESULT_20261102.md`。
- SSL 无标签帧扩量（2026-11-02 续）：SSL 预训练集 162→596–628 帧（全部记录帧按折排除留出家族，646 条记录帧、50 批次映射，避免转导泄漏；脚本加 --frames-file）——**无提升**：种子 42 宏平均 0.4763（−0.002）、种子 43 0.5235（+0.009），均值 +0.003 在噪声内。**SSL 收益在同分布无标签帧上已饱和（~160 帧即够）**；已确认 SSL 收益（σ0.05/20ep 配对 +7.5%）保持不变。剩余未测变量=编码器容量/架构（width 未贯通训练/评测脚本）。模型不晋级。见 `T16_LINE_SEG_SSL_BIGFRAMES_RESULT_20261102.md`。
- 容量臂 + 训练侧搜索空间收口（2026-11-02 续）：SSL 脚本补 --width（编码器容量与微调一致，44 测试通过）；容量臂 w1.25 宏平均 0.5193/0.4681（均值 −0.003 噪声内）、w1.50 0.4580/0.4709（均值 −0.032 负）。**训练侧 14 个变量全部证伪**（监督 8 + SSL 超参 3 + SSL 帧扩量 + 容量 2）；唯一确认收益 = SSL σ0.05/20ep（配对 +7.5%，2/2 种子）。最高水位 **0.4787/0.5148（两种子）**，模型不晋级。剩余需新资源/新工程：真实相机域数据、线专用架构、或 SSL 进部署链路。见 `T16_LINE_SEG_WIDTH_RESULT_20261102.md`。
- 条带架构臂 + 转整车主线（2026-11-02 续）：`--arch strip`（非对称 1×k/k×1，参数 +6%，arch_args 贯通，45 测试通过）固定 4 折两种子——宏平均 0.4628（−0.016）/0.4722（−0.043），**两种子皆负**。**训练/架构侧 15 变量全部证伪**，最高水位 SSL +7.5%（0.4787/0.5148）保持；按双指标计划第 4 步转 ③ 整车主线（TODO 接线 TOD/天气 → 条件矩阵战役 → 安全计数可信化）。条带架构代码保留（checkpoint 契约兼容），真实域数据（②）到来时可重评。模型不晋级。见 `T16_LINE_SEG_STRIP_RESULT_20261102.md`。

## 工程整改（2026-10-20，评审后的 P0）

本轮没有跑 Tech、没有训练、没有改任何驾驶行为；只解掉两个**会挡住后续实测**的
工程问题，并按模块分三个提交入库（`scripts` / `tests` / `experiments`）。

1. **磁盘解封（资源门恢复）**。实测 I: 盘 free 16.14 GB < `min_free_disk_gb=20`
   （`experiments/collection.py`），所有 Tech 采集会在前置检查被拒。
   新增 `scripts/m5_logs_retention.py`（默认 dry-run、manifest 先落盘、路径越界
   整批拒绝、只删 `.pt/.pth/.ckpt`）：保留 `seg_model*` 部署目录、`best*`、
   含 `final_set_seal.json` 的封存子树、以及文档点名当 `--init`/交付候选/
   确认集的三个目录（`t14_e1_promotable_20260927`、`t16_negdose6x_20261001`、
   `t14_e0_20260925`）。实测 `--apply --min-free-gb 30` 删 3301 个 / 13.86 GB，
   **free 16.14 → 30.07 GB**；受保护权重逐个复核仍在。9 项单测。
2. **进程探测（`tasklist` 在本机被拒）**。实测 `tasklist` 返回
   `ERROR: Access denied`（rc=1），导致：单实例锁断电恢复永不生效（`_alive`
   把已退出进程判成活着）、采集前置检查永远"不确定"、采集收尾所有权证明永远
   给不出（测试里的 `owned game cleanup unproven` 就是这个）。改用 PowerShell 7
   的 `Get-Process`：`collection._process_table()` 一张表喂
   `game_running_probe`/`game_pids`，`controller._alive` 用
   `Get-Process -Id` 明确判定"已退出"，仅当连 PowerShell 都不可用时才退回
   tasklist 的保守语义。实测 `_alive(self)=True`、子进程 terminate 后
   `_alive=False`、`_process_table()` 枚举 365 个进程。
3. **测试可复现性**。同一份代码：普通 PowerShell 下 `pytest tests/` 7 失败，
   设 `PYTHONUTF8=1` 后 0 失败——根因是控制台代码页（子进程 GBK 打印 "↔" 直接
   UnicodeEncodeError；父进程 `text=True` 按 GBK 解码）。`tests/conftest.py`
   统一 UTF-8（环境变量 + Popen 文本模式默认编码）；同一缺陷还让
   `scripts/check_commit_scope.py` 遇中文提交标题崩在 `.strip()`（**推送前
   自查实际从未跑通**），四处 git 调用统一 `encoding="utf-8"` 并补 5 项链路测试。

**实测门**：`pytest tests/` = **3448 passed / 0 failed**（510 s，含本轮新增 17 项）；
`scripts/m5_offline_validate.py` = ALL PASS；`check_commit_scope.py --last 3` = OK。
**未验证**：Tech 实机、GPU 训练、冻结评审门、PDF 重编译——本轮一项都没跑。

## 论文补正与结构收敛（2026-10-20 续）

**论文（docs/paper）**：四处只改表述、不改数字的修正 + 主张清单补录，
依据全部来自 10-11～10-18 的新证据。

1. **69 m 方向写反**：正文 V-C 与局限原写「道路链高于地形约 69 m」，实测
   `mid.z`=144.678 对地面/相机 z≈213.6 → 链在**下方**。中英文源同改，四份
   生成物（`main.tex`/`main_zh.tex`）重建，正文 PDF 重出（英文 20 页、中文
   18 页，较原 19/17 各 +1）。
2. **归因收缩**：原「三处修复后六批准入」→「地图名传播修复解除隔离；高程
   不一致本身未消除（仅对齐重采仍隔离 3/3 站）」。
3. **utah 段更新**：不再是「未探测地图」；五站 annotation 均可写（3,434/799/
   2,948/1,350/5,425），0/5 原因为对象级贴花缺失 + 渲染范围止于 stn≈15 +
   阴影敏感判据。
4. **C5 谓词限定**：补实测——阴影中真漆 (106,98,74) 被同一判据剔除、干红土
   r−b=63 被判作漆，故结论只能是「不满足*这个*谓词」。
5. **CLAIMS 补录 C19–C39**（21 条，状态一律「未复算」），并加「C7 与 C19
   不同轴」「C5 语义限定」两条口径；`submission/` 源包重建 + CLAIMS 同步。
   新增 5 项结构性守卫（编号连续/六列/状态口径/副本逐字节一致/69 m 符号回归）。

**结构收敛**：

- **技术债棘轮**入库并进入 `dev_validate.ps1`：当前基线 = `except Exception`
  **334** / 静默 **63** / `print` 163 / 最长函数 3430 行；总量只许减少，
  `silent_except` 与单函数长度逐文件只许减少。读文件用 `utf-8-sig`（仓库里有
  带 BOM 的源文件）。**已开始还债**（79 → 63）：`run_manifest` 收窄 5 处
  （父链/进程表；坏掉的 psutil 现在会冒出去，不再让"第二控制器"证据悄悄失效）、
  `planning/selector` 一处收窄 + 纵向剖面失败改写入 `meta["speed_profile_error"]`、
  `connector` 只吞 `ImportError`、`experiments/metrics` 两处收窄；`fsd_drive` 8 处**安全相关**静默点收窄为 `(TypeError, ValueError, IndexError)`（越界最差值、逐帧 `line_lat`、车身越线计数、弯道速度帽），纯诊断/清理的 19 处不动。
  **未动**的静默点（`fsd_drive` 27 处等）都在热路径里，需逐处读上下文，留待专轮。
- **`select_lane_reference` 拆分**（四步纯搬移，零语义变更）：信任顺序的关键
  判据逐块抽成可单测的模块级函数——`_lane_gate_verdict`（航向/转角/侧向三维门，
  拒绝理由与阈值字面量不变）、`_sensor_lane_within_map`（一致性门，近场点不足
  时返回 None 而非误判）、`_lane_width_from_map_edges`（最近点距离 + 合同宽度
  兜底）、`_trusted_single_paint`（单边漆线在 strict 下的信任判据）。
  主函数 **571 → 374 行**（−197，−35%），信任顺序的每一级都有独立可测函数：三维门、一致性门、边线宽度、单边漆线、铺装判据/几何、分隔线兜底、几何形状门、硬边界判据、meta 装配、BEV 整路中心、末道 fail-closed；新增 23 项单测，基线逐次收紧。**剩余未拆**：最终 `LaneReference` 构造（声明式，13 个字段）与 map-lane 构建，二者共享多个本地量，收益低于风险。
- **`FSDStack.tick` 首切片**（2026-10-20 起）：位姿/地面（`_pose_and_ground`）
  与参考发布（`_publish_lane_reference`）抽出，**1394 → 1378 行**；同时用 AST
  打出 tick 的 60+ 条顶层语句骨架（行号 + 首行），后续切片按它定位，不必重读
  1400 行。tick 的其余部分以长编排为主、共享局部量多，抽取收益低于 lane。- **文档索引**：`docs/INDEX.md` 224 篇，含预注册↔结果配对（82）与「无正文
  入口（28）」小节——孤儿文档至少有一条可达路径；生成器带 `--check` 门禁。
- **lane 去重**：`fusion.py` 复用 `tracking.py` 的 5 个逐字相同 helper。

**门**：`pytest tests/` = **3480 passed / 0 failed**；深度离线 ALL PASS；
`check_commit_scope` OK；`m5_paper_texcheck` 4/4 OK；`m5_paper_number_audit`
261 项 OK；`m5_docs_index --check` OK。
**未验证**：LaTeX 实际编译（本机无 TeX）、补充材料 PDF（源未改）、Tech 实机、
GPU 训练。

## 评审整改收口（2026-10-20）

**P0 四项全部完成**（每项都有实测证据，见上文）：① 磁盘解封（free 16.14 → 30.07 GB，
资源门恢复）；② pytest 在默认 shell 与 `PYTHONUTF8=1` 下都可复现（3448 → 3503 passed，
0 failed）；③ README 验收口径 + 论文四处修正 + 主张清单补录；④ FSD 入口默认
`sensor`+`strict`，旧地图参考需显式 `--allow-legacy-map-lane`
且该轮结果不得用于验收。

**P1 结构项完成的**：`select_lane_reference` **571 → 374 行**（11 个可测判据）、
`FSDStack.tick` 首切片 1394 → 1378、技术债棘轮入库并进 `dev_validate`（静默异常
**79 → 59**、`except Exception` **351 → 332**）、`docs/INDEX.md`（224 篇 / 28 无正文入口）、
论文源包与 CLAIMS 同步、`lane/fusion` 去重。

**刻意不做（有理由，不是遗漏）**：
1. `FSDriveSession.run`（3430 行）**不拆**——它是顺序编排，60+ 语句共享十几个局部量，
   抽纯函数要传 10+ 参数、返回 5+ 值；无人值守下改错的风险高于可读性收益。
2. `FSDStack.tick` 不再继续硬拆，同样原因（已在上一节记录）。
3. `fsd_drive` 剩余 19 处静默点**保留**：despawn / watchdog 心跳 / e2e reset /
   stack.close 等纯诊断与清理路径，吞掉不会让安全读数变松。
4. `autopilot.py` 的 18 处（legacy 规则栈）不动：按 AGENTS 它只是兼容兜底。

**待你决定**：C11「四次运行未观察到压线」的口径（10-11 后另有 5 次合格运行，
压线帧计数 UNKNOWN）、Zenodo/ORCID/目标期刊、以及**分支 ahead 84 未推送**。



## ③ 整车主线：条件矩阵战役（2026-11-02 深夜–11-03 凌晨）

- **git 历史 squash（2026-11-02）**：161 个细粒度提交按「日×模块」squash 为
  27 个单模块汇总提交 + 1 个索引更新（96ca452），最终树逐字节不变，
  `--force-with-lease` 推送完成；旧哈希经 annotated tag
  `history/pre-squash-20261102` 完整可达（`git rev-list --count HEAD`=831）。
- **环境接线两处引擎侧实锤 + 修复**：① `core_weather.lua` 证据：天气预设
  diff 表可携带 timeOfDay，先设时刻再上天气会被覆盖——接线固定为
  **weather→tod** 并加 `get_tod()` 读回存证（条件格=引擎侧可验证）；
  ② 矩阵实测（dawn_rain 请求 06:30 读回 09:56）：预设过渡是**动画**
  （time 秒内逐帧插值回写），紧跟的 set_tod 被插值覆盖——应用后等过渡
  结束再设时刻。两处均有单测钉住（52 测试通过）。
- **环境探测（看门狗版，75 s）**：天气预设库存 7 个（sunny_evening/sunny/
  cloudy_evening/sunny_noon/rainy/foggy_night/foggy_morning）；矩阵映射
  rain→rainy、fog→foggy_morning、clear→引擎默认。教训入库：beamngpy
  套接字无超时（RPC 必须看门狗）；引擎子进程持有 stdout 管道（输出必须
  落文件，绝不过管道）。
- **seg_model pin 修订（预注册修订案 A1）**：benchmark 的 v8_backup 与
  部署默认 seg_model/best.pt 均在 2026-10-29~11-01 外部删除事件中灭失；
  改 pin **seg_model_v28_sslA**（SSL +7.5% 水位、fold A 训练集含 italy），
  20 格单一固定 pin；无模型的首轮 dawn_clear 记 VOID 轮。
- **条件矩阵部分轮（中断于收尾，可断点续跑）**：dawn 三格全部
  **fail-close 于放置**（06:30 黑暗→感知零线像素→刹车保持→UNKNOWN，
  与 east 阴影定因同族的照度门机制）；dawn_fog 被中断（missing_session）。
  聚合器验证端到端：12 格全 UNKNOWN、行级证据完整（placement_rc=3 +
  tod_after 读回），**放置失败格不再静默消失**（从 batch_manifest 播种）。
  部分证据已哈希归档
  `logs/evidence_archives/t16_fsd_conditions_partial_20261103.zip`
  （complete: True）。
- **明日续跑**：`--run` 断点续跑剩余 17 格（dawn_fog 起），noon/dusk/night
  才是矩阵的决定性格（noon_clear 隔离"模型 vs 光照"变量）；矩阵表出来后
  进 250 s 长跑 + 车道穿越计数 UNKNOWN 清零。全部提交按模块过
  check_commit_scope 并已推送（tip a03a94c 后续 3 提交）。
- **线识别推理侧杠杆（2026-11-04）**：新增 `--tta hflip`（镜像概率平均）与
  `--tta ens`（同折双种子概率平均，`--model2`）；固定 4 折（留出集与 SSL
  复核一致，8 模型训练∩留出=0）实测：hflip TTA **证伪**（宏平均 0.4787→
  0.4576，3/4 折负）；**双种子 SSL 集成确认正迁移**：宏平均 0.5148→
  **0.5392（+4.7%）**，最弱折 A 0.2643→**0.5048**（超过该折最强成员
  0.4288，也高于逐折 oracle 0.5335）；基线对集成对照无 fold A 跳升
  （0.2802），排除"普通种子多样性"解释。代价 2× 推理；模型不晋级，部署
  接线需单独预注册。见 `T16_LINE_SEG_ENSEMBLE_RESULT_20261104.md`。
- **条件矩阵执行阻断（2026-11-04）**：断点续跑首轮即被放置阶段阻断——dawn
  四格 + noon_clear 全 `rc=3`；三场景探测定因：mountain 无标线（line_px=0）、
  town 仅单侧（mirror_right/mirror_left 被严格门拒）、italy 认证站 0.15m 细线
  远场拒绝（far_kind）。历史 town 配对率 43–60% 依赖的 `seg_model_hand` 已在
  删除事件中灭失，现存 71 族池模型 town 配对率 0%——**部署域感知回归**被
  定位。按预注册纪束矩阵标 12/12 UNKNOWN（不降门、不改场景凑指标），
  解阻见 `T16_FSD_CONDITIONS_BLOCKED_20261104.md`。
- **配对鲁棒性研究臂预注册（2026-11-04）**：条件矩阵阻断后，针对
  `far_kind` / `thin_near` / `mirror` 三个拒绝门冻结可证伪假设（H1 far
  thin 几何自洽、H2 细线近场稀疏、H3 镜像与路面中心一致）；四臂（F1/F2/
  M1 + 条件组合 C1）全部研究开关**默认等于现行行为**，先离线探针
  （认证场景真值：true_pair_rate/phantom_rate），判据写死：true_pair
  ≥+10pp 且 phantom 增幅 ≤+2pp（复现 ≥1 场景），否则证伪。不降门、不晋级、
  不新增标注。见 `T16_LANE_PAIRING_ROBUSTNESS_PREREG_20261104.md`。
- **配对鲁棒性基线 + 前置条件（2026-11-04）**：探针跑 3 认证场景 × 3 重复 ×
  20 tick。认证 0.15m 细线候选召回 **0.00**（italy `thin:side` 丢弃率 38%，
  候选为 ±4.2m 路缘实线）；italy 配对门从未触发；east 17/17 配对为
  `centre_line_own_lane`（宽误差 0.099m、中心差 2.06m，参考系不一致）；
  west 三轮 0/0/11 不稳定、真配对时匹配 GT。**S1 不可评估**，原假设
  "配对门是绑定约束"证伪——绑定约束在更上游（薄线检测/collect 丢弃 +
  场景车道语义不对齐）。F1/F2/M1 三臂延后，需先造配对专用场景；预注册
  修订案 A1 已记录。模型不晋级。
- **配对专用场景（2026-11-04）**：探针改为场景模式（自管引擎 + 运行时构建
  right_lane 贴花 Scenario + drive --attach），GT 用构建记录真实节点。首轮
  3 场景基线：west 17/17 true、召回 1.00；east true 19.7/20 但召回 0.70、
  16/20 为 centre_line_own_lane；italy 召回 0.375。**A2 硬门未过 → 三臂
  不执行**；修订案 A1 将场景集改为宽路子集（west 2.5/6.0/7.5m 半宽），
  判据不变。见 `T16_PAIRING_SCENE_BASELINE_20261104.md`。
- **配对三臂执行结果（2026-11-04）**：F1/F2 在 right_lane 的 east/west 场景
  完成 3 重复（18 轮）。基线 true_rate=1.000（饱和），F1 east −7.5pp、
  F2 east −2.2pp、west 0pp，phantom 0——**S1 不适用（天花板）**，三臂无增益、
  不晋级；入选场景排除了目标失效模式（单侧/跨中心线/far-thin），不能外推。
  M1 未实现（缺路面中心输入）。下一步需难场景或转"薄线检测召回"上游课题。
  见 `T16_LANE_PAIRING_ROBUSTNESS_RESULT_20261104.md`。
- **薄线召回结果（2026-11-04）**：east/west × 贴边余量 0.05/0.6/1.0 × 2 重复
  （12 轮）。难场景定位 **east/m1.0**（true_rate 0.326、`thin:span` 6.5/轮）；
  east 全余量下**右线（黄色）候选召回≈0**，配对靠 mirror/centre_line + 宽度先验
  得到正确中心（m0.05/0.6 true=1.0 属"先验正确"，非双侧检测）；west 在
  0.05/0.6 为真双侧 `pair`。按预注册 D3（span 为主因）→ F1/F2 **不复测**、
  不跑空臂；下一课题为黄色右线/`thin:span` 候选层修复（需新预注册）。
  见 `T16_THIN_LINE_RECALL_RESULT_20261104.md`。
- **黄线/span 候选层修复预注册（2026-11-04）**：针对 east 右线（黄线）候选
  召回≈0 与 east/m1.0 的 `thin:span` 6.5/轮，冻结三条可证伪假设（H1 黄线
  外观阈值、H2 thin 最小跨度/碎段合并、H3 坡度站 CLAHE 归一）与四臂
  （Y1/S1/I1/C1，研究开关默认 off），主判据：east/m1.0 右线召回 ≥0.6、
  true_rate ≥0.8 且 phantom ≤0.05，west 控制不回归 >5pp；三臂全败则记录
  H0（损失在语义掩码侧）。不训练、不改默认门、不新增标注。见
  `T16_YELLOW_LINE_SPAN_PREREG_20261104.md`。
- **黄线/span/CLAHE 三臂结果（2026-11-04）**：east/m1.0 + east/m0.6 + west/m0.6
  × 3 臂 × 2 重复（24 轮）。右线召回在所有臂/轮 ≈0（最大 +0.08），east_hard
  的 true_rate 被三臂从 0.91 降到 0.50/0.40/0.03（phantom 升高）；west 控制
  零变化。**R1/R2 失败、R3 通过、H0 成立**：损失在语义掩码/检测层，不在
  候选层；三臂不晋级、默认关。另记录 east/m1.0 配对结果**双稳态**
  （两次批 0.326 vs 0.91、单轮 0.00–1.00），后续难场景实验须先做 3+ 重复
  基线稳定性检查。见 `T16_YELLOW_LINE_SPAN_RESULT_20261104.md`。
- **掩码侧右线诊断（2026-11-04）**：离线池帧（east 9 帧/west 3 帧）比较
  raw/post 线掩码与 engine GT（按漆色拆黄/白）。**HA 被否定**：raw 黄线召回
  east 0.942/west 0.927，post 0.727/0.889——模型能检出黄线；HB 仅部分
  （east 后处理删除 22.5%，但 post 仍 0.73）；HC 不成立（碎段 63/21.6px）。
  east 白 GT 仅 56px/帧（annotation 覆盖极少），白召回数字不可解读。
  结论：live 右线候选召回≈0 发生在**掩码之后**（mask→marking 或 live 536×403
  域差异）。下一步二选一：live 帧回放 detect_lines，或 144 原生 vs 536 上采样
  域对照（均需新预注册）。见 `T16_MASK_SIDE_DIAG_RESULT_20261104.md`。
- **重采样域对照（2026-11-04）**：同一池帧按 native/cubic/lanczos/roundtrip
  输入比较。east 触发 D2：raw 黄召回 0.97 但 **post 从 0.727 掉到 0.103
  （−62.5pp）**；west 变化 ≤5pp。机制=后处理在输入分辨率上跑、核大小像素
  固定——池域 native 后处理在 144、live/预上采样在 536，细黄线在 536 被
  删掉。**live 右线候选≈0 的直接机制找到**（场景相关，west 不回归）。
  下一步：后处理分辨率归研究臂（先把掩码缩到规范分辨率跑后处理再放大），
  离线验证 post 黄召回 0.10→≥0.6 后再谈 live。见
  `T16_RESAMPLE_DIAG_RESULT_20261104.md`。
- **后处理分辨率归结果（2026-11-04）**：P1（W=144）east 恢复（0.103→0.629）
  但 west 跌到 0.679（native 被缩小）→ R2 失败；P2（W=192=训练域宽）双场景
  全过：east cubic post 黄召回 **0.103→0.688**、west **0.843→0.863**，native
  与 raw 零变化（R1/R2/R3 全过）。这是候选中修复链首个成立的确定性修复：
  后处理固定像素核在 536 输入下删除细黄线，缩到 192 跑后处理再放大即可
  恢复。下一步：live 接线（默认 None，加 `--seg-post-norm-width`）复测右线
  候选/true_rate + 时延与安全审计。见 `T16_POST_NORM_RESULT_20261104.md`。
- **post-norm live 复测（2026-11-04）**：`--seg-post-norm-width` 接线完成
  （默认 None），east 难场景开/关 × 2 重复。**L1 失败**（右候选召回
  0.00→0.15，目标 0.6）；L2 失败（true_rate 双稳态干扰，off 0.83/0.05、
  on 0.28/0.17，phantom 上升）；**L3 通过**（无崩溃、未放置保持刹车
  rc=3、安全门语义不变）；**L4 通过**（p50 18.9→8.0ms，−10.9ms）。
  结论：离线 post 恢复（0.103→0.688）不迁移到 live——剩余丢失点在
  mask→marking 提取或 live 536 原生域；开关不接部署。下一步：live 帧
  回放 detect_lines（探针加 --save-frames）。见
  `T16_POST_NORM_LIVE_RESULT_20261104.md`。
- **live 帧回放结果（2026-11-04）**：`--save-frames` 采集 + `m5_marking_replay`
  回放（east 19 帧/west 24 帧）。east **H-M0 18/19（掩码缺右黄线）**，
  投影距离验证左线 2.7–6.8px、右线 16–50px（非投影假象）；west **ok 14/24、
  H-M1 10/24（掩码有、无 marking）**。两场景失败层不同：east 在语义掩码
  （live 536 原生域 vs 训练 144 上采样域），west 在 mask→marking。
  下一步：输入域匹配臂（live 先缩 144 再送模型），离线用已保存帧验证
  east H-M0 ≤0.3、ok ≥0.5，west ok 不降 >10pp。见
  `T16_MARKING_REPLAY_RESULT_20261104.md`。
- **输入域匹配臂结果 + 场景渲染发现（2026-11-04）**：east `--input-width 192`
  回放 P-In1/P-In2 失败（H-M0 17/19、ok 5.3%，与 native 无差别）→ H-In 证伪。
  逐帧像素核查发现 **live 帧右黄线 GT 位置是沥青灰（左线橙漆可见）**，
  ground delta=0，引擎日志报 east 场景引用不存在的 italy 线材质
  （`ptSetMaterial failed`）——**右侧参考贴花很可能未渲染**。因此此前的
  live 右线指标（H-M0、post-norm L1、右候选≈0）不能归因于感知；需先做
  场景渲染校验门（GT 投影点附近必须有漆色像素）再谈感知结论。离线池域
  结论（mask diag/resample/post-norm P1/P2）不受影响。见
  `T16_INPUT_DOMAIN_RESULT_20261104.md`。
- **无标线路段筛选结果（2026-11-04）**：east a1/a2、utah a3 以 `known_no_line`
  采集（不放生成线），GT 左右两侧 paint_ratio 全部 **1.0**（加此前 east a0/
  west a3，五个锚点三张图全失败）→ **S1 失败，live 合成贴花参考不可建立**。
  V1/V2/M1 不执行；此前 live 右线指标继续标记不可归因。结论：环境/场景
  设计限制（现有锚点均带地图标线），需新地图或程序化无标线路段（外部
  资源）；本环境内标线识别的 live 路径到此为止，可继续的有效工作为
  离线 SSL 集成部署化评估与整车主线条件矩阵。见
  `T16_MARKING_FREE_REF_RESULT_20261104.md`。
- **SSL 双种子集成部署化评估（2026-11-04）**：新增部署组件
  `EnsembleSegmenter`（概率平均→同一后处理；fail-closed）与评测入口
  `--ensemble-deploy`。固定 4 折逐折**精确复现**既有记录（A 0.5048/B 0.6897/
  C 0.4232/W 0.4312），宏平均 **0.539246**（对 0.5392 差 +0.00005，R-Repro
  通过）。时延 p50 18.76→29.00ms（**1.55×**，后处理只跑一次）；S-FailClosed/
  S-Post/S-Default 单测全过。组件未接 drive（需单独接线预注册）；模型不晋级。
  见 `T16_ENSEMBLE_DEPLOY_RESULT_20261104.md`。
- **集成接 drive 部署接线（2026-11-04）**：`--seg-model2` → `EnsembleSegmenter`
  （默认单模型不变）。D1–D4 全过：时延 p50 18.76→29.00ms（1.55×），live 4 轮
  无 watchdog/timeout；显存单 1.7MB/集成 3.4MB（≪200MB）；fail-closed 负面
  注入（坏 checkpoint）→ 感知 disabled → perception-unavailable → 刹停；
  放置可用性 paired 率 1.00 vs 1.00、placed 1/2 vs 1/2（同分，站点本身双稳态）。
  集成仍非默认；live 贴花 GT 环境阻塞未变。见
  `T16_ENSEMBLE_DRIVE_RESULT_20261104.md`。
- **条件矩阵收口（2026-11-04）**：修订案 A3 改 west a3s0 后，dawn 4 格 +
  noon_clear 全部 rc=3（12 格 UNKNOWN）——无路线探针的 1/2 成功不迁移到
  带路线驾驶链路；加上 mountain（无标线）与 town（单侧，单模型+集成均
  失败），**三场景均无法产出驾驶格**。矩阵按预注册 UNKNOWN 收口，不重跑
  凑表；解阻需部署域感知数据或新场景（外部）。见
  `T16_FSD_CONDITIONS_CLOSED_20261104.md`。
- **三模型集成结果（2026-11-04）**：等权（SSL42+SSL43+基线43）固定 4 折宏平均
  0.549925（+0.0107）但 W −0.0103、B −0.0097 触发预注册逐折一致性门
  （R2 ≤0.005）→ **不采用**；A 折 +0.0402 说明基线种子对最弱折有互补价值，
  未来可预注册加权/专用策略。参考水位仍为双模型 **0.539246**；模型不晋级。
  见 `T16_ENSEMBLE3_RESULT_20261104.md`。
- **折 A 错误归因（2026-11-04）**：双模型集成在折 A 50 帧上逐帧近场 IoU 均值
  0.451；按 run：a0s0 0.164（线 691px）、a2s0 0.210（1053px）、a3s0 0.562
  （1195px）、a4s0 0.594（1367px）。**IoU 与 GT 线像素密度单调相关，与亮度
  无单调关系**——折 A 的弱是"线稀疏/短/断"，不是暗。对补数据的含义：优先
  长而连续、两侧清晰的锚点，而非单纯增站点。本分析不改模型/门。
  见 `T16_FOLDA_ERROR_ATTRIB_20261104.md`。
- **加权集成结果（2026-11-04，已获用户授权执行）**：W1(0.4/0.4/0.2) 宏平均
  0.546702（+0.0075）、W2(0.45/0.45/0.10) 0.548378（+0.0091）——A 折回收
  +0.026~+0.028、B 几乎不回退（W2 −0.0015），但 **W 仍退步 0.005~0.007** 且
  宏增益未达 +0.010 门槛 → R1/R2 未过，**不采用**；参考水位仍为双模型
  **0.539246**，模型不晋级、默认不变。见
  `T16_ENSEMBLE_WEIGHTED_RESULT_20261104.md`。
