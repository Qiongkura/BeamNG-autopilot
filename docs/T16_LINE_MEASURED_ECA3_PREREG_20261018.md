# east_coast 第三批 LINE=measured 采集预注册（2026-10-18，annotation 可写材质修复后重采）

* 前置：ECA2 0/5 全隔离（`T16_LINE_MEASURED_ECA2_RESULT_20261018.md`），
  定因为 annotation pass 的 alpha 测试剔除（east 线材质 opacityFactor 0.7 ×
  bcf alpha 0.8–0.9 对 alphaRef=255 全像素剔；z 埋地论断已撤回）；修复已入库
  （MAP_PROFILES east 线材质/线宽 → line_yellow_double @0.40 m + 回归测试，
  含对真实材质资产的 annotation 可写性断言）；
* **本轮唯一变更因子**：线材质/线宽（line_white/line_yellow @0.15 m →
  line_yellow_double @0.40 m，白/黄角色同材质）。其余与 ECA2 逐项一致：
  east_coast_usa、Tech RPC、`--anchors 5 --sites 1 --scenes known_line
  --frames 3 --step-m 2 --anchor-offset 0`、线位约定 symmetric（默认）、
  同锚点族、同一采集/导出代码路径；
* **冻结准入一字不改**：projection median≤1px、coverage≥0.8、
  appearance≥0.9、engine_verified truth_provenance；
* 线宽 0.40 m 是**场景规格适配**（east 船运材质集无"0.15 m 可写且过 appearance
  门"的材质，宽度扫描实测 0.40 m 为唯一全过点：appearance 1.00×2、
  coverage 1.0、投影 0.54/0.63 px；0.15 m appearance 0.43–0.67 挂、
  0.50 m 回落）——预注册在案、真值几何（中心线 class=2 点链）不变、
  **不是降门**；west/italy 保持 0.15 m 合同宽度不变；
* 报告自证键：`map_profile.line_width=0.4`、line_instances.material=
  line_yellow_double（白/黄角色均然）、ground_alignment measured；
* 目标 ≥2站、≥6帧；锚点耗尽、道路链失败、隔离、UNKNOWN、启动/cleanup失败
  全部保留；**不降门、不消费最终集、不新增人工标注、不提升模型**；
* annotation 可写性准入规则（后续新地图选材必须先过）：材质须
  `alphaTest=True` 或 `alphaRef<255`，且条纹纹理在所选线宽下居中且
  appearance 门实测通过——档案进 MAP_PROFILES 前用
  `m5_decal_annotation_probe.py` 同款探针实测，不靠猜。
