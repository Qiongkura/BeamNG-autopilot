# east_coast 第二批 LINE=measured 采集结果 + annotation 丢失定因（2026-10-18）

预注册：`docs/T16_LINE_MEASURED_ECA2_PREREG_20261018.md`。
证据：`logs/m5_seg/collect_e5line_eca2_20261018/`（scene_report.json + 帧）。

## 批次结果（0/5，全部隔离保留）

配置与 ECA1 同锚点族：east_coast_usa、`--anchors 5 --sites 1 --scenes
known_line --frames 3 --step-m 2 --anchor-offset 0`、线位约定 symmetric、
生成器 script_sha16 `2f8d851996449c5a`（含地面重对齐修复；线材质仍为
line_white/line_yellow @0.15 m）。冻结准入一字不改。

* **0/5 站合格、5 站全部隔离**（coverage 门）：
  * `known_line_a0s0`：coverage 0.3185
  * `known_line_a1s0`：coverage 0.0
  * `known_line_a2s0`：coverage 0.0
  * `known_line_a3s0`：coverage 0.0667
  * `known_line_a4s0`：coverage 0.0
* 隔离理由逐站记录（"line evidence does not cover the declared chain"），
  不引用对无关像素算的投影残差（acceptance.before_px n=0）；
* 目标 ≥2站/≥6帧**未达成**；不降门、不补采替代；全部证据保留。

## 地面重对齐实测（z 埋地论断的否定证据）

五站 `ground_alignment` 全部 `status=measured`，delta_m =
+0.0195 / −0.00016 / −0.00050 / −0.00009 / −0.00043 —— **全部 ≈0**，
包括 ECA1 引用的 `mid.z=26.07` 站（ground 重读 26.072 m）。ECA1 的
"26.07 对地形 54.2"读数与同址重读矛盾，**"生成路链整体埋入地下 ~28 m"撤回**
（详见 `T16_LINE_MEASURED_ECA1_RESULT_20261018.md` 更正节）。54.2 的来源
已定位：采集启动时初始 ego 出生点的 `ensure_above_ground` 读数
（`ego at z=-5.0 below ground 54.2; lifting`）——那是出生点的地形，不是
站点的地形，两处被误并成一个"偏差"。对齐后仍 0/5、RGB 漆线可见 →
annotation 丢失另有机制。

## 定因：annotation pass 的 alpha 测试剔除（探针 5 场景变量分离）

诊断工具：`scripts/m5_decal_annotation_probe.py`（同站点并排测试线，逐条
只改一个变量；输出 `logs/m5_seg/diag_*`，**诊断专用，不进训练池**）。

逐项判决（全部无效）：

| 变量 | 取值 | 判决 |
| --- | --- | --- |
| z 抬升 | 0 / 0.08 / 0.25 / 0.50 m（含失败侧负侧全档） | SOLID 命中不变 |
| renderPriority | 0 / 50（beamngpy `render_priority=idx` 改写已 monkey-patch 还原） | 逐像素一致 |
| 插入序 / width 0.15–0.8 m / over_objects | 各对照 | 无效 |
| 材质 | line_white / line_yellow / line_yellow_double / line_yellow_double_discontinue / line_parking1 / line_parking2 | **决定性** |

机制（正向可复现）：

* BeamNG annotation pass 对贴花按材质 alpha **测试**取像素：`alphaTest`
  缺席时按 `alphaRef` 阈值（255 = 最终 alpha<255 全剔）；
* east 的 line_white/line_yellow 家族：`opacityFactor 0.7 × baseColorFactor
  alpha 0.8–0.9` → 最终 alpha ≤ 0.56 → **annotation 全像素被剔**，落回
  覆盖带 ASPHALT；RGB pass 走 translucent 混合 → 漆可见（ECA 帧里
  paint-like 沿真值曲线的来源）；
* west 的 line_white 无 opacity 缩放（核心 alpha=1.0）→ annotation 正常
  （真值像素 19/30 命中）——采集/投影/真值管线在 west 早已全链路通过；
* east 的 line_yellow_double（alphaTest=True/alphaRef=0）annotation 全写
  （探针 solid2px 口径：0.40 m 两次运行 30/30 真值点窗口内 SOLID，0.15 m
  29/30）——同一条生成链在 east 也能写进 annotation，坐实"材质 alpha 态"
  是唯一杠杆。

材质普查（east `art/road/main.materials.json` 13 个 SOLID_LINE）：可写仅
alphaTest=True 或 alphaRef<255 的 4 个（line_yellow_double、
line_yellow_double_discontinue、line_parking1/2）；其中 parking 两材质
条纹在纹理内偏心（annotation 落真值外 5–8 px，投影门挂）、discontinue 为
虚线（沿链 appearance 缺口）。**唯一合格：line_yellow_double**，宽度扫描
（0.20/0.25/0.30/0.40/0.50 m，两次运行）给出冻结门判决：

| 宽度 | appearance（≥0.9） | coverage@6（≥0.8） | 投影 mean（≤1 px） |
| --- | --- | --- | --- |
| 0.20 | 0.07 / 0.03 ✗ | 1.0 ✓ | 1.08 ✗ |
| 0.25 | 0.43 / 0.43 ✗ | 1.0 ✓ | 1.04 ✗ |
| 0.30 | 0.73 / 0.77 ✗ | 0.97 ✓ | 1.25 ✗ |
| **0.40** | **1.00 / 1.00 ✓** | **1.0 ✓** | **0.54 / 0.63 ✓** |
| 0.50 | 0.67 / 0.63 ✗ | 0.97 ✓ | 2.21 ✗ |

（0.40 m 另在材质场景单独测得 appearance 0.90 / coverage 1.0 / 投影
0.55 px——三测三过；0.50 回落是双条纹 gap 回到真值点，"越宽越好"不成立。）

## 处置与后续

* 生成器修复（已入库）：MAP_PROFILES east 线材质/线宽 →
  line_yellow_double @0.40 m（白/黄角色同材质；west/italy 保持 0.15 m 合同
  宽度），`apply_map_profile` 支持档案 `line_width` 与 `--line-width` 覆盖，
  回归测试含对真实材质资产的 annotation 可写性断言；
* **冻结门一字不改**（projection median≤1px、coverage≥0.8、appearance≥0.9、
  engine_verified）；0.40 m 宽度是场景规格适配（east 船运材质集所限），
  预注册在案，不是降门；真值几何（中心线 class=2 点链）不变；
* ECA3 同锚点族重采：`docs/T16_LINE_MEASURED_ECA3_PREREG_20261018.md`。
