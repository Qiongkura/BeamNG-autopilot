# utah 资格采集预注册（2026-10-18，MAP_PROFILE 探测 + annotation 可写材质建档后）

* 前置：utah MAP_PROFILE 探测完成（Utah.zip 全嵌套复扫 + `m5_decal_annotation_probe.py`
  同款探针 A/B 两运行实测 + east/utah 受控对照 + 跨运行投影校准）；定因与 east 同族：
  annotation pass 的 alpha 测试剔除（line_white opacityFactor 0.7 × bcf alpha 0.902、
  line_yellow 0.7 × 1.0 对 alphaRef=255 全像素剔，实测近距 bin solid 0/3、0/4
  对 line_yellow_double 4/4，两运行一致）；唯一可写实线 line_yellow_double
  （alphaTest=True/alphaRef=0，annotation 全写）；档案已入库
  （scripts 提交 3450599：MAP_PROFILES utah + annotation 可写性回归测试）；
* **本轮唯一变更因子**：地图（east_coast_usa → utah）+ 该图档案的线材质/线宽
  （line_yellow_double @0.40 m，白/黄角色同材质——与 east 同款决策）。其余与 ECA3
  逐项一致：Tech RPC、`--anchors 5 --sites 1 --scenes known_line
  --frames 3 --step-m 2 --anchor-offset 0`、线位约定 symmetric（默认）、
  同一采集/导出代码路径；
* **冻结准入一字不改**：projection median≤1px、coverage≥0.8、
  appearance≥0.9、engine_verified truth_provenance；
* 线宽 0.40 m 是**场景规格适配**（宽度扫描实测：0.20–0.30 m 双条纹亚像素无漆、
  0.50 m 远距回落，0.40 m 为唯一"真值点处仍见漆"点；utah 与 east 共享同一
  line_yellow_double 贴花资产（`.link` 同 `/assets/materials/decalroad/lines/
  line_yellow_double/`），故同取 east 已验证的 0.40 m；导出门 paint_like（loose）
  30/30×2 全宽饱和不判别宽度——已记录）——预注册在案、真值几何（中心线
  class=2 点链）不变、**不是降门**；west/italy 保持 0.15 m 合同宽度不变；
* 报告自证键：`map_profile.line_width=0.4`、line_instances.material=
  line_yellow_double、teleport_alignment 逐站记录（utah 探针站点 road z 与
  地形实测 delta 0.02 m，无 z 缺陷，GROUND_ALIGN_MAPS 保持
  {west_coast_usa, east_coast_usa} 不动；批次站点若现 z 偏差按失败证据保留
  并停批分析，不预防性修）；
* 已知风险（掩码侧证据保留，不改门）：严格外观掩码 paint_like_mask 对 utah
  偏暗渲染欠覆盖（条纹 RGB 180–186/158–163/117–123 贴 LINE_WHITE_MIN=185 与
  r−b 阈值下沿，探针 strict 0.53–0.87 对 east 1.00）；冻结 appearance 门用
  导出 loose 判据（实测 30/30×2）；批次实例若挂冻结门 → 隔离 + 证据保留，
  不降门；
* 目标 ≥2站、≥6帧；锚点耗尽、道路链失败、隔离、UNKNOWN、启动/cleanup失败
  全部保留；探针/诊断产物（diag_utah_sites_20261018、
  diag_utah_probe_20261018_a/b）不进训练池；
  **不降门、不消费最终集、不新增人工标注、不提升模型**；
* annotation 可写性准入规则（后续新地图选材必须先过）：材质须
  `alphaTest=True` 或 `alphaRef<255`，且条纹纹理在所选线宽下居中且
  appearance 门实测通过——档案进 MAP_PROFILES 前用
  `m5_decal_annotation_probe.py` 同款探针实测，不靠猜
  （utah/east 两图已按此规则执行）。
