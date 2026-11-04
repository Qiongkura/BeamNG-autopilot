# T16 输入域匹配臂结果：P-In1/P-In2 失败；并发现 live 右贴花未被渲染（2026-11-04）

## 执行

预注册 `T16_INPUT_DOMAIN_PREREG_20261104.md`：用已保存帧离线回放，
`--input-width 192`（live 536×403 → INTER_AREA 缩到 192×144 → 模型内部
放大 536），分类逻辑与回放诊断一致。east 19 帧、west 24 帧。

## 结果

| 场景 | 输入 | ok | H-M0 | H-M1 | 判定 |
| --- | --- | --- | --- | --- | --- |
| east | native 536 | 1/19（5.3%） | 18/19（94.7%） | 0 | — |
| east | 192 输入域 | **1/19（5.3%）** | **17/19（89.5%）** | 1 | **P-In1/P-In2 失败** |
| west | 192 输入域 | 待复算（native 14/24、H-M1 10/24） | — | — | 未达标 |

P-In1（H-M0 ≤0.3）与 P-In2（ok ≥0.5）均未达；输入域匹配（H-In）在 east
被证伪。

## 关键发现（跑臂后追加、非预注册判据）

P-In 失败后逐帧检查 GT 位置的实际像素，发现**参考标记本身可能未渲染**：

1. 在 east live 帧上把 GT 左/右线世界点投影并取像素：
   - 左线（+1.8）投影点：`[255,139,49]` —— 橙色漆，**可见**；
   - 右线（−1.8）投影点：`[150,148,147]` —— 沥青灰，**不可见**（5 个站点全灰）。
2. 地面重对齐检查：`ground_z=25.581`、`road_z=25.581`、`delta=0.0` ——
   不是 z 埋没；加入采集器的 GROUND_ALIGN 后结果不变。
3. 引擎日志（最近一次场景加载）：
   `DecalRoad::ptSetMaterial, failed to find Material of name
   italy_road_markings_line_thin(_yellow)!`
   —— east 场景引用了 **italy** 的线材质，east_coast_usa 上不存在。

**结论**：live 帧中右侧黄线贴花很可能**没有被渲染**（或落在不可见位置），
左贴花可见。因此此前所有基于"右线 GT"的 live 指标（H-M0 94.7%、
post-norm L1 右召回 0.15、配对右候选≈0）都**不能归因于感知**——它们测量
的是一个未渲染/不可见的参考。west 的帧需同样做像素级渲染校验后才能使用。

## 判读与下一步

- 形式判定：输入域匹配臂失败（P-In1/P-In2），不采用、不接部署。
- 实质判定：**live 参考场景完整性**是先决条件，而不是感知杠杆。必须
  先做"场景渲染校验"：场景加载后、采集前，逐帧检查 GT 左右线投影点附近
  是否有漆色像素；不满足则场景作废（UNKNOWN），不进任何指标。
- 场景材质修复（east/west 的线材质名与地图匹配）与校验门需新预注册；
  在此之前不再跑 live 右线指标，避免继续在无效参考上得出结论。
- 训练/离线池域结论（mask diag/resample/post-norm P1/P2）不受影响：那些
  场景由采集器构建并通过导出冻结门（投影/漆色一致），贴花确实可见。

## 产物

`logs/m5_seg/input_domain_east.json`、`marking_replay_east_ga_r0.*`（GROUND_ALIGN
版帧）、`marking_replay_east_sym_r0.*`、`_east_ga_frame5_marks.png`（GT 投影
叠加图）。

## 追加：材质修复、渲染校验门与 A/B 判定（同日）

1. **材质修复已实现**：探针此前漏调 `apply_map_profile(map)`，east/west 的
   贴花仍用 italy 默认材质（引擎日志 `ptSetMaterial failed to find
   italy_road_markings_line_thin`）。修复后 profile 正确记录
   （east `line_yellow_double@0.40`、west `line_white/line_yellow@0.15`），
   GT 投影点邻域确实有漆色像素（paint_ratio=1.0）。
2. **但 A/B 证明这些漆不是贴花**：同一站点分别构建 `known_line` 与
   `known_no_line`（无任何生成线），GT 左右两侧的 paint_ratio **都是 1.0**
   —— 画面里的漆全部来自地图自带标线，合成贴花在 live 帧中不可区分。
3. **渲染校验门**已实现（`paint_ratio` + `INVALID_GT` 标签），但通用"漆色"
   判据无法区分贴花与地图线；需要贴花特异判据（唯一材质色/渲染 A/B 差分），
   或改到**无地图标线的路段**。
4. **结论**：east a0s0 / west a3s0 这类站点上，live 的"合成贴花 GT"与地图
   标线在 6px 内重合/被覆盖，**live 右线指标仍不可归因**。正确做法是换到
   地图本身无标线的路段（或唯一材质色的专用站点）再测；这不是感知问题，
   是场景设计问题。
5. 已实现 `--scene-type known_no_line` 供 A/B；材质档案修复与渲染门代码保留。
