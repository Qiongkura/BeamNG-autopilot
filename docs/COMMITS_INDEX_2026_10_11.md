# 提交索引（2026-10-18 ～ 2026-11-02，160 个提交）

本索引为**可读导航**：提交历史不重写（一模块一提交，文档哈希引用保持有效）。
每日本节末给出当日最后一个提交（annotated tag `day-<日期>` 指向它）。

## 2026-10-18（共 32 个）

模块分布：docs×20、scripts×12

- **docs**（20 个）：
  - `6486a07` docs: west_coast第四批LINE measured扩展结果（3站9帧准入、2站隔离原因保留）
  - `2ee89cf` docs: west_coast第四批LINE measured配对训练预注册（9帧W5、5种子）
  - `2e1b628` docs: west_coast第四批配对结果、状态与论文第二地图章节同步（PDF重导出）
  - `91739bc` docs: west_coast第五批独立LINE measured采集预注册（offset12）
  - `a039027` docs: west_coast第五批LINE measured配对训练预注册（W6 3帧、目标未达成证据保留）
  - `f38cde2` docs: west_coast第五批采集/准入结果（1站3帧准入、目标未达成如实记录）
  - …（其余 14 个见 `git log --oneline 2026-10-18 -- 模块路径`）
- **scripts**（12 个）：
  - `22cc12f` scripts: 修复run_scene asset块地图名硬编码italy（west_coast证据误记已定位）
  - `38bc773` scripts: east_coast_usa 地图档案（全嵌套复扫补正第三图材质）+ 回归断言
  - `3eaabf5` scripts: 地面重对齐泛化到实测 z 偏差地图集合（east_coast 埋地根因修复）
  - `33ca92e` scripts: 稿件数字审计扩展至 east 第三图资格（annotation alpha 定因/宽度扫描/ECA1-3 准入，240 项）
  - `c67f0c2` scripts: 稿件数字审计扩展至 east ECA3 配对结果（E3−H 三项/collapse/复现性，257 项）
  - `3450599` scripts: utah 地图档案（line_yellow_double@0.40m 探针实测）+ annotation 可写性回归测试
  - …（其余 6 个见 `git log --oneline 2026-10-18 -- 模块路径`）

当日最后提交：`a7a3a9e`（tag `day-2026-10-18`）

## 2026-10-20（共 58 个）

模块分布：scripts×19、docs×13、lane×9、README×3、tests×2、experiments×2、fsd_realism×1、docs(paper)×1、fsd_stack×1、run_manifest×1、planning×1、connector×1、fsd_drive×1、control×1、watchdog×1、vision/seg×1

- **scripts**（19 个）：
  - `7b12bfe` scripts: logs 权重保留策略（默认 dry-run、证据不动、点名权重与封存集保留）
  - `e41186f` scripts: git 调用统一 UTF-8 解码（提交范围检查器遇中文标题直接崩）
  - `7c1f861` scripts: FSD 入口默认改为 sensor+strict，旧的地图横向参考需显式逃生口
  - `5567a8f` scripts: 新增 docs 索引生成器（类型/日期/配对/引用，可 --check 门禁）
  - `b8fe429` scripts: 技术债棘轮（静默 except / print / god-function），并入 dev_validate
  - `d4078f4` scripts: 索引增加「无正文入口」小节（28 篇孤儿至少有一条可达路径）
  - …（其余 13 个见 `git log --oneline 2026-10-20 -- 模块路径`）
- **docs**（13 个）：
  - `f037592` docs: 状态文档补 2026-10-20 工程整改节，审计计数 129→261
  - `c105378` docs: 状态文档补「论文补正与结构收敛（2026-10-20 续）」
  - `3738312` docs: 状态文档同步 select_lane_reference 拆分最终状态（571 → 445）
  - `9510033` docs: 状态文档把拆分数字更新到本轮（571 → 432）
  - `feb102a` docs: 状态文档记 lane 拆分收敛（571 → 374，−35%）
  - `864d8f6` docs: 状态文档记 FSDStack.tick 首切片（1394 → 1378）
  - …（其余 7 个见 `git log --oneline 2026-10-20 -- 模块路径`）
- **lane**（9 个）：
  - `38125e4` lane: fusion 复用 tracking 的 5 个 helper，删掉逐字复制
  - `101d68a` lane: 把三维门（航向/转角/侧向）抽成可单测的 _lane_gate_verdict
  - `c7cf2fe` lane: 一致性门抽成 _sensor_lane_within_map（纯搬移）
  - `5c26ac8` lane: 地图边线宽度抽成 _lane_width_from_map_edges（纯搬移）
  - `d603d9f` lane: 单边漆线信任判据抽成 _trusted_single_paint（纯搬移）
  - `68d2d39` lane: meta 装配与 BEV 兜底抽成纯函数（_reference_meta / _bev_whole_road_center）
  - …（其余 3 个见 `git log --oneline 2026-10-20 -- 模块路径`）
- **README**（3 个）：
  - `fbf053c` README: 加「2026-10-20 验收口径」块，修正与最新证据冲突的四处表述
  - `27a9138` README: 验收口径块链到 docs/INDEX.md（224 篇报告的唯一入口）
  - …（其余 1 个见 `git log --oneline 2026-10-20 -- 模块路径`）
- **tests**（2 个）：
  - `0b8a59d` tests: 子进程统一 UTF-8，pytest 不再依赖 shell 代码页
  - `5dc8570` tests: 主张清单结构性守卫（编号/列数/状态口径/副本一致/69 m 符号回归）
- **experiments**（2 个）：
  - `8f84009` experiments: 进程探测改用 Get-Process（tasklist 在受限令牌下 Access denied）
  - `bdf86fa` experiments: metrics 的两处静默异常收窄
- **fsd_realism**（1 个）：
  - `288efdc` fsd_realism: 新增 lane_mode_is_fsd_compliant（合规 = sensor + strict）
- **docs(paper)**（1 个）：
  - `1eeb736` docs(paper): 修正 69 m 方向、归因与 utah 段，补录 CLAIMS C19–C39
- **fsd_stack**（1 个）：
  - `e6e919d` fsd_stack: tick 首切片——位姿/地面与参考发布抽成 _pose_and_ground / _publish_lane_reference
- **run_manifest**（1 个）：
  - `15dd303` run_manifest: 收窄 5 处 except Exception（父链与进程表解析）
- **planning**（1 个）：
  - `2330076` planning: selector 的两处静默异常——一处收窄、一处改成可诊断
- **connector**（1 个）：
  - `1562aa8` connector: set_active_map 的登记失败只吞 ImportError
- **fsd_drive**（1 个）：
  - `4f98ac7` fsd_drive: 8 处安全相关静默异常收窄（越界读数、压线计数、速度帽）
- **control**（1 个）：
  - `0ec9fad` control: 交接失败后的"停车也失败"必须留痕（原来是 pass）
- **watchdog**（1 个）：
  - `d636c13` watchdog: disarm 失败必须留痕（原来是 pass）
- **vision/seg**（1 个）：
  - `6fcc71a` vision: 黄色掩码与分类标签两处静默异常收窄

当日最后提交：`775ca04`（tag `day-2026-10-20`）

## 2026-10-29（共 6 个）

模块分布：scripts×3、docs×3

- **scripts**（3 个）：
  - `144c38c` scripts: annotation 探针 utah 化（--site-index/--material + utah lift/priority 变体组，诊断用）
  - `70f0c32` scripts: 探针第二轮 utah2 组（批次线位 ±1.8 单站 + 全 5 站×2 线批次结构复现 utfull）
  - `53850c1` scripts: 每站 Scenario 采集模式（utah 探针判决：单 Scenario 10 路后建贴花渲染截断；GROUND_ALIGN_MAPS 保持原流程）+ 回归测试
- **docs**（3 个）：
  - `d044f9f` docs: utah 贴花渲染截断引擎诊断预注册（lift/优先级变量矩阵 + 判读判据 + 纪束；诊断输出不进池）
  - `0870ff9` docs: utah 贴花截断探针结果（单 Scenario 多路对象截断判决）+ 重采预注册（每站 Scenario，冻结门不改）
  - `2c744b5` docs: 状态同步——utah 贴花截断判决/修复入库/重采 attempt1 断连证据保留（重试待执行）

当日最后提交：`2c744b5`（tag `day-2026-10-29`）

## 2026-11-01（共 3 个）

模块分布：docs×2、scripts×1

- **docs**（2 个）：
  - `7d85130` docs: 采集收口——批次 2 达标 3站/9帧、批次 3 困难锚点 0 帧（原因保留）；认证池定格 3 族/15 帧；停采
  - …（其余 1 个见 `git log --oneline 2026-11-01 -- 模块路径`）
- **scripts**（1 个）：
  - `84a6a82` scripts: 保留清单点名 engine_verified 采集/导出/证据档目录（2026-11-01 外部删除事件后加固）

当日最后提交：`7d85130`（tag `day-2026-11-01`）

## 2026-11-02（共 61 个）

模块分布：docs×47、scripts×10、tests×2、vision/seg×1、fsd_drive×1

- **docs**（47 个）：
  - `d418818` docs: 线分割重训预注册（认证池 15 帧、留一族划分、对 v13b 0.116 外部参照）
  - `8dc9c3d` docs: 线分割认证池首训结果——内部 val 0.77/同族 0.30/跨族留出 0.0 泛化坍缩；增广非瓶颈，数据广度是绑定约束；不晋级
  - `3603505` docs: 数据族扩量预注册（utah offset15 + east offset5，每站 Scenario 全图统一，冻结门不改）
  - `d20f597` docs: 多族线分割重训预注册（3 折分组留出、宏平均 line IoU、对照首训跨族 0.0）
  - `269c788` docs: 多族线分割结果——跨族 line IoU 0.0000→宏平均 0.2792（15 族/45 帧，3 折分组留出）；不晋级
  - `4384517` docs: 线分割训练侧消融预注册（epochs/line-weight/oversample 五臂，fold-A 固定划分）
  - …（其余 41 个见 `git log --oneline 2026-11-02 -- 模块路径`）
- **scripts**（10 个）：
  - `eb6384a` scripts: 每站 Scenario 全图统一（east 迁移）+ ground 重检下沉每站循环；静态断言同步
  - `60f7295` scripts: 修复 weak-line-oversample 3 元帧解包 bug + 回归测试
  - `fdb92ff` scripts: SSL 预训练支持 --frames-file（全量记录帧扩量用）
  - …（其余 7 个见 `git log --oneline 2026-11-02 -- 模块路径`）
- **tests**（2 个）：
  - `919a1a4` tests: 条带架构 + arch_args 贯通回归测试
  - `92be282` tests: TOD/天气接线回归
- **vision/seg**（1 个）：
  - `3d6eb7d` vision: SegUNet 条带架构变体（arch strip，非对称 1×k/k×1）+ Segmenter 从 arch_args 读 arch
- **fsd_drive**（1 个）：
  - `9a5b6ae` fsd_drive: 会话启动应用 TOD/天气（EnvironmentApi）并写入运行记录，失败不静默

当日最后提交：`92be282`（tag `day-2026-11-02`）
