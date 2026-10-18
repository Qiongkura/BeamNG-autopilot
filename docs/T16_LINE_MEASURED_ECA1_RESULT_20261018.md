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

## 根因定位（ECA1 后诊断，证据链）

生成站点 `mid.z = 26.07` 来自 road-network `middle.z`，而实测地形为 z ≈ 54.2
（`ensure_above_ground` 读数；同锚点 ground 查询一致）——**road-network z 低于
真实地形约 28 m，生成的 DecalRoad 路链整体埋入地下**：线不可见（外观近零）、
annotation 全背景、深度射线穿地逸出、投影校验 mismatch。与 west_coast 首批
失败同根因类（road-chain 高于地形 ~69 m 的镜像变体）。修复前的地面重对齐
（`_align_sites_to_ground`）只对 west_coast 开启，east_coast 未覆盖。

修复：`GROUND_ALIGN_MAPS = {west_coast_usa, east_coast_usa}`（进集合需该地图
实测 z 偏差证据，italy 实测一致不进），初始对齐与 post-load 复测/重建两处
gate 同步泛化（含回归断言）。ECA2 采集按修复后代码重跑（见预注册）。
