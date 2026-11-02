# east_coast 第一批 LINE=measured 扩展结果（2026-10-18）

预注册：`docs/T16_LINE_MEASURED_ECA1_PREREG_20261018.md`。

* east_coast_usa、offset=0、5 anchors、5 known_line 站点、3帧/站；
* 材质档案生效（铺装=AsphaltRoad_variation_two_way 白=line_white
  黄=line_yellow 碎石=DirtRoad_variation_01 蓝=无）；
* 采集批次验收：**0/5 站合格、5 站全部隔离**：
  * `a0s0`：coverage 0.3203；外观 checked=0（166 点全遮挡）；annotation 线像素 1942
  * `a1s0`：coverage 0.0；外观 0/140；annotation 线像素 21
  * `a2s0`：coverage 0.0；外观 14/81；annotation 线像素 38
  * `a3s0`：coverage 0.1889；外观 1/47；annotation 线像素 1997
  * `a4s0`：coverage 0.0；外观 6/170；annotation 线像素 44
* 帧证据自证：`label_px` 全 background（road 0 / line 0）、
  `depth_cal m_max≈3.1e6 m`（射线逸出）；采集期一次瞬时 RPC
  "Error while receiving data" 自动恢复，批次完整；
* 目标 ≥2站/≥6帧**未达成（0 站）**；不降门、不补采替代；冻结导出门未进入
  （批次验收已全隔离），全部证据保留于 `collect_e5line_eca1_20261018`。

## 根因定位（ECA1 后诊断）——**已更正（2026-10-18，ECA2 + 探针定因）**

> **撤回**：ECA1 后把 0/5 归因为"生成路链整体埋入地下"（road-network z 低于
> 地形 ~28 m）是**错的**。上面对 0/5 各项测量（coverage/外观/annotation 线像素）
> 如实保留，只更正因果解释。

更正依据（证据链）：

1. ECA2 按"地面重对齐"修复重采后 **ground_alignment 五站全 measured、
   delta≈0**（+0.020 / −0.0002 / −0.0005 / −0.0001 / −0.0004 m）——包括
   ECA1 引用的 `mid.z=26.07` 站点，其 ground 重读为 26.072 m。ECA1 的
   "26.07 对地形 54.2"读数与同址重读矛盾：54.2 实为**初始 ego 出生点**的
   `ensure_above_ground` 读数（`ego at z=-5.0 below ground 54.2; lifting`），
   与站点分处两地、被误并成一个"偏差"，28 m 偏差不成立；
2. ECA2 对齐后仍 **0/5**（coverage 0.32/0/0/0.07/0），且逐帧 RGB 漆线
   paint-like 沿真值曲线可见——"埋地致线不可见"的推断不成立；
3. 探针（`scripts/m5_decal_annotation_probe.py`，5 场景变量分离，诊断输出
   仅存 `logs/m5_seg/diag_*`，不进训练池）逐项判决：z 抬升 ≤0.5 m、
   renderPriority 0/50、插入序、宽度 0.15–0.8 m、over_objects 全部无效；
   **真正机制是 annotation pass 的 alpha 测试剔除**：east 的
   line_white/line_yellow 家族 `opacityFactor 0.7 × baseColorFactor alpha
   0.8–0.9` → 像素最终 alpha ≤ 0.56，配 `alphaRef=255`（alphaTest 缺席时的
   阈值语义）全像素被 annotation pass 剔除；RGB pass 走 translucent 混合，
   漆仍可见。west_coast 的 line_white 无 opacity 缩放（核心 alpha=1.0）所以
   annotation 正常（实测真值像素 19/30 命中）——管线本身是好的；
4. east 实测可替代材质只有 `line_yellow_double`（alphaTest=True/alphaRef=0，
   该图自己的车道线就是它）：0.40 m 宽下冻结门三门全过（appearance 1.00、
   coverage 1.0、投影 mean 0.54/0.63 px，两次运行），0.15 m 下双条纹亚像素
   appearance 0.43–0.67 挂，0.50 m 回落（条纹间 gap 回到真值点）。

处置：生成器 MAP_PROFILES 按 annotation 可写性重选 east 线材质/线宽
（line_yellow_double @0.40 m，west/italy 合同宽度 0.15 m 不变），冻结门
一字不改；`GROUND_ALIGN_MAPS` 保留 east（对齐是实测读数驱动的恒等/微调，
五站 delta 记录在案），但注释同步更正（z 偏差不再是 0/5 的解释）。ECA3
按修复后代码重采：`docs/T16_LINE_MEASURED_ECA3_PREREG_20261018.md`。
诊断全过程见 `docs/T16_LINE_MEASURED_ECA2_RESULT_20261018.md`。
