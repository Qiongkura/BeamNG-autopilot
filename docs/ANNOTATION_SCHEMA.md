# 标注 schema 与时间切片（T10）

计划 T10 要求标注能表达像素类别之外的东西（曲线 ID、左右/分隔角色、可见段、遮挡、起止、属性、unknown），
并要求**中心线、像素漆面、推断延伸分开保存**；同时把"按固定图像行堆时间切片、人工控制点、时间方向插值、
再映射回帧"的工具登记为**提议**（`scripts/m5_timeslice_annotate.py`，本轮落地）。

## 1. 曲线 schema（`beamng_autopilot/labeling/curve_schema.py`）

一个记录 = 一次标注批次，锚在 `frame_index`（其段覆盖的第一帧）；一条曲线跨 12 帧只出现**一次**，
段内带逐帧点，而不是 12 条几乎相同的记录。

| 字段 | 含义 | 强制规则 |
| --- | --- | --- |
| `curve_id` | 曲线身份（同一记录内唯一） | 重复即报错 |
| `role` | `left` / `right` / `divider` / `centre` / `unknown` | 其它值报错 |
| `attributes` | 仅 `colour` / `style` / `width_m` | 其它键报错 |
| `unknown` | **显式**未标注的字段名 | 不是可标注字段即报错 |
| `segments[].source` | `pixel_paint` / `centre_line` / `inferred_extension` | 其它值报错 |
| `segments[].derived_from` | 推断段所依据的观测 ID | **推断段必须有；量测段必须没有** |
| `segments[].visible` / `occlusion` | 可见性与遮挡（含成因与起止帧） | 被遮挡的段必须是 `visible=False`，两者不能同真 |
| `segments[].points[].frame` | 逐帧点 | 必须落在该段的帧范围内 |

像素类别固定为 `0=background, 1=road, 2=line, 255=ignore`（`CLS_*`）。
`pixel_value_report()` 对掩码里**未定义的值**如实报告（`unknown_values`），不重贴标签；
`255` 计为 `ignore` 而**不是** background——未标注像素不能被度量当成真负例。

写盘/读盘都先校验：`dump_jsonl()` 拒绝写非法记录，`load_jsonl()` 拒绝读非法行；
未知关键字走 `TypeError` 而不是静默默认（所以字段增删是显式事件）。

**这套规则针对的是一个具体失效模式**：把插值出来的曲线当量测保存，之后任何"真值"统计都会把它算成观测。
因此"推断必须带来源、量测不得声称推断"由 schema 直接拒绝，而不是靠注释提醒。

回归：`tests/test_curve_schema.py`（19 例）——每条强制规则一个反例、正例（含遮挡段与推断段）各一、
JSONL 往返、未知值报告、空掩码不造 0 线、未知关键字 TypeError。

## 2. 时间切片标注工具（`scripts/m5_timeslice_annotate.py`）

**切片**：`build_timeslice(frames, rows, u0, u1)` 每格回答一个问题——
"这一帧里，漆线是否在这条图像行、这个列窗口内穿过？" 统计量默认是行内**对比度**（max − median），
不是均值：均值被背景梯度主导，且丢掉了线的列位置（第一版就是这么错的，实测后改掉）。
`u0/u1` 窗口**就是工具的假设**，它的正确性由逐帧人工抽检衡量，工具自己不宣称。

**插值**：`interpolate_rows(clicks, breaks=...)` 在控制点之间按时间线性插值；
两端不外推；**断点不被跨越**（断点两侧没有控制点的帧一律 UNKNOWN，直接不输出），
控制点落在断点帧本身则保留。

**落回帧**：`snap_to_stroke()` 在预测行附近按亮度（`avoid=True` 时按暗）在窗口内取极值作为该帧的列，
因此产物一律是 `inferred_extension` 且带 `derived_from`（引用控制点）；
断点同时切断**输出段**（`runs`）——身份/可见性边界不只在插值时生效。

**抽检**：`spot_check(predicted, manual)` 输出逐帧行误差的 p50 / max / 容差内占比，
只出现在一侧的帧记为 **UNKNOWN**（不算分、也不当作命中）。

**限制（写进工具输出，不靠文档记忆）**：只适用于连续、可跟踪身份；换道/遮挡/身份变化处必须给断点；
产物是**辅助标签**而非可靠稠密真值；工具**不承诺**提速倍数——提速要用"分钟/有效标签"实测，
本轮**没有**这个数字（见 §3）。

回归：`tests/test_timeslice_annotate.py`（21 例）——线性插值/不外推/断点不跨越/断点帧控制点保留、
切片形状与列窗口（线在窗口外时没有对比度）、亮暗两种 snap、窗口为空时如实报 `reason`、
产物 schema 合法且**全部**是推断段带来源、断点把记录切成两段、JSONL 往返、
抽检统计与单侧 UNKNOWN、CLI 拒绝没有 `rgb` 的 episode、CLI 落盘标签与报告（含 `limits`）。

## 3. 真实身份与跨视角同曝光（本轮落地）

计划 T10 要求"使用真实 map/episode/time/source ID，不只 run basename"，以及"跨视角同一曝光、金帧复制与
相邻片段必须一起归组"。本轮：

- **采集侧记录身份**：`m5_collect_seg_ring.py` 每次 `grab_ring_labels()` 记一个 `exposure` 计数，
  **同一曝光的每个视角共享它**，并逐帧写 `view`/`t_wall`/`pos`/`line_pixels`；`meta.json` 增 `map_name`、
  `source_id` 与 `frames` 列表。`m5_collect_seg.py` 同样写入 `map_name`/`source_id` 与逐帧 `t_wall`。
- **索引侧读身份**：`dataset_split.frame_refs_from_meta(meta, run=...)` 是唯一的索引构造入口，
  身份取自记录；**缺什么就报什么**（无 `map_name`/`source_id`/`t_wall`/`exposure` 各有独立提示），
  没有钟时 `t` 是帧号并置 `t_is_index=True`，绝不假装是秒。
- **跨视角归组**：`cross_view_groups()` 按 `(map/source_id, exposure)` 聚合多视角帧；
  `cross_view_leak(plan, refs)` 报告同一曝光是否被切到两侧，**没有曝光计数时报 `checked=False`
  并给出原因**——"没检查"不会被写成"没有泄漏"（三值口径）。
- **训练入口**：`m5_train_seg.py` 的 by-map-scene 分支改用该入口，`min_line_frac` 过滤后**索引仍取自保序序列**
  （meta 的帧号不知道过滤），并打印身份回退条数与跨视角检查结论。

回归：`tests/test_dataset_split.py` 新增 7 例（身份取自记录而非目录名、缺 ID 逐条报告、
缺钟置 `t_is_index` 且不猜秒、多视角同曝光成组、无计数不成组、同一曝光跨侧判泄漏、
不可检查不得报成干净）。

## 4. 未做（明确登记）

- **GUI 未接**：时间切片工具是文件驱动的（clicks/spot-check 走 JSON），人工点击界面尚未接入；
  因此"分钟/有效标签"和真实误差分布**本轮没有测量**，不得引用任何提速倍数。
- **真实标注量**：没有在真实连续片段上完成一轮完整时间切片标注，故没有标注质量统计。
- **金帧复制与相邻片段**：`cross_view_leak` 现在能查"同一曝光的多视角"，但**复制帧/相邻片段的检测**
  （同一张图被复制到两个 run、或相邻片段被切开）仍未做。
- **冻结测试集**：`freeze_testset`/`check_frozen_testset` 已存在并可用，但本轮没有产生新的锁定留出集。
- **采集数据未重跑**：新增的 meta 字段只对**之后**的采集生效；已有 `logs/` 采集没有 `t_wall`/`exposure`，
  对这些旧数据训练入口会如实打印回退提示。

