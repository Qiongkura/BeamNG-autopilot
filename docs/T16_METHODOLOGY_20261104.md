# T16 标线识别与整车主线方法学（2026-11-04，Phase 0 固化）

## 1. 评测协议（固定，不可随结果改）

- **固定 4 折**：A 早期 utah / B 后期 utah / C east+west+italy / W west 单留出；
  同族帧绝不跨训练/评测。
- **配对种子**：同一实验跑种子 42/43，只有配对方向一致的增益才采信；
  单种子单跑不足采信（基线自身种子方差可达 +0.03）。
- **指标**：`line IoU(近场)`（下半画面）；宏平均 = A/B/C（W 单独报告）。
- **聚合**：全局累加 inter/union（iou_from_accum），未出现类记 0，不做逐帧
  均值虚高。
- **数据隔离**：训练 runs ∩ 留出 runs = 0（逐模型核验，见 8 模型核验记录）。

## 2. 数据质量门（冻结）

导出认证池必须同时满足：覆盖率 ≥0.8、投影中位误差 ≤1px、最大 ≤2px、
漆色一致率 ≥0.9、真值来源 engine_verified。不满足的站点隔离并保留失败
证据；门限不因结果调整。

## 3. live 参考校验门（2026-11-04 新增）

在 live 帧进入任何指标前，必须验证"合成贴花确实渲染在 GT 位置"：
1. 场景加载时应用 `apply_map_profile(map)`（否则用错地图材质 → 贴花不可见）；
2. GROUND_ALIGN_MAPS（east/west）加载后地面重对齐（z 差 >2m 必须重建）；
3. GT 左右线投影点邻域必须有漆色像素（paint_ratio ≥0.5）；
4. A/B 对照：`known_line` vs `known_no_line` 的 GT 像素差必须显著；
   若二者相同，说明是地图自带标线，参考无效 → 场景作废（INVALID_GT）。
本会话实测：五个锚点三张图全部失败（GT 两侧漆色 1.0），live 贴花参考
不可建立——这是环境阻塞，不是感知结论。

## 4. 诊断链（分层定位）

按"从上游到下游"逐层量化，禁止跳层归因：
1. 掩码侧：raw/post 线掩码在 GT 漆色上的召回（`m5_mask_diag.py`）；
2. 重采样：native vs cubic/lanczos/roundtrip（同帧对照）；
3. 后处理分辨率：`post_norm_width`（训练域宽度 192；P2 离线 0.103→0.688）；
4. 候选层：collect_drops 按 kind（thin:span / thin:side / align / kind）；
5. mask→marking：live 帧回放 `detect_lines` + `pair_lane_markings`，
   逐帧互斥分类 H-M0（掩码缺）/ H-M1（无 marking）/ H-M2（collect 门）/
   H-M3（配对层）/ ok；
6. 配对门：F1 far_thin / F2 thin_near / M1 mirror（研究开关，默认 off）。

## 5. 部署门（集成）

- D1 时延：p50 单 18.76ms → 集成 29.00ms（1.55×），live 无 watchdog/超时；
- D2 显存：双模型增量 +1.7MB（总 3.4MB，≪200MB）；
- D3 fail-closed：成员构建/预测失败 → 感知禁用 → 刹停；禁止静默退单模型；
- D4 放置可用性：paired 率与放置成功率不下降（同场景同条件对照）。
- S-*：加权默认等权、非法权重拒绝、默认路径单模型不变（单测钉住）。

## 6. 负结果登记（不重复投入）

训练/架构侧（15 项）：lw1/lw4、80ep、难例过采样、scale0.15/0.30、
族等量采样、capacity 1.25/1.50、strip 架构、SSL 轮数/噪声扫描、
SSL 无标签帧扩量、hflip TTA。
数据/场景侧：mountain 无标线、town 单侧（单模型+集成）、west 带路线链路、
hw6 宽路（thin:span）、五锚点 live 贴花参考被地图标线污染。
集成侧：等权三模型（宏 +0.0107 但 W/B 退步，一致性门失败）。

## 7. 规则

不降门、不消费最终集、不新增人工标注、不晋级未过门槛的模型；
每批实验预注册；失败/隔离证据保留；全部产物哈希归档。
