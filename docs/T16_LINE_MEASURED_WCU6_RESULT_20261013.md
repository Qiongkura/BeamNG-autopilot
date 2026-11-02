# west_coast 第五批 LINE=measured 扩展结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_WCU6_PREREG_20261013.md`（commit `91739bc`）。

* offset=12、5 anchors、5 known_line 站点、3帧/站；
* 采集批次验收：5/5 站 eligible、coverage 0.936–1.0、标注线像素 3–2606/站、
  标定后投影 5/5 ≤2px（mean 1.278、median 1.409、max 1.484）；
  采集过程中出现一次瞬时 RPC "Error while receiving data"（a0 与 a1 之间），
  自动恢复、批次完整（`scene_report.json` errors=[]），原样保留；
* 冻结导出准入（truth_provenance / engine_verified）：**1站/3帧** 导出
  （`known_line_a2s0`：coverage 1.0，标定后 median 0.968 px，paint 1.0，
  certified 153/180 点，truth_contract v1）；
* **4站隔离**，原因原样保留：
  * `known_line_a0s0`：标定后投影残差 median 1.073 px > 1.0（几何未证实；
    paint 两侧 1.0——仅几何门边缘超限）；
  * `known_line_a1s0`：投影残差 median 1.565 px > 1.0 且 paint appearance
    agreement 0.750 < 0.9（168 投影真值点；left 69/84、right 57/84）；
  * `known_line_a3s0`：线实例 coverage left 0.1282、right 0.0 < 0.8
    （annotation 通道失效，全站线类像素 3）且投影残差 median 1.372 px；
  * `known_line_a4s0`：标定后投影残差 median 1.599 px > 1.0（paint 两侧 1.0）；
* **采集目标 ≥2 站/≥6 帧未达成**（实际 1 站/3 帧）：不降门、不补采替代、
  不改预注册口径；缺口如实记录；
* 数据集审计：`dataset_id ff5eefe9…71cd`，3帧、3 trainable、rejected=0、
  group/content/exposure overlap=0、内容摘要 3帧3唯一；UNKNOWN x3
  （road px 0 < 200，ROAD 通道未证实，按资格掩码处理）；
* 本批使用修复后的 `run_scene` asset 地图名（22cc12f）：全部证据
  `asset.map = west_coast_usa`（此前批次该字段误记 italy，仅证据元数据，
  不改写旧证据）；
* cleanup 完成，采集后无残留 BeamNG 进程。

采集批次验收与冻结导出门是两道独立门：本批 5/5 站过批次验收，但冻结门
（投影 median ≤1px + coverage ≥0.8 + appearance ≥0.9）再隔离 4 站——
其中 3 站仅输在投影几何边缘（1.07/1.56/1.60 px），1 站 annotation 线通道失效。
隔离站点不入训练；导出池保持 1 站 3 帧独立 LINE=measured 站点（a2，与既往
a2 站点索引相同但 offset=12 为独立采集）。

跨池独立性证据（`crosspool_overlap_audit.json`）：全部既往导出池 + H 人工臂
共 81 帧、81 个唯一内容摘要、content/exposure/group overlap=0、0 重复组——
WCU6 与任何既往训练/配对用帧无内容复制。
