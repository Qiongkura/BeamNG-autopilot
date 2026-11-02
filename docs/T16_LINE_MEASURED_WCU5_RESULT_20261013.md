# west_coast 第四批 LINE=measured 扩展结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_WCU5_PREREG_20261013.md`（commit `a63c437`）。

* offset=9、5 anchors、5 known_line 站点、15帧目标；
* 采集批次验收：5/5 站 eligible、coverage 0.920–1.0、标注线像素 1176–2826/站、
  标定后投影 5/5 ≤2px（mean 0.934、median 0.751、max 1.832）；
* 冻结导出准入（truth_provenance / engine_verified）：**3站/9帧** 导出
  （`known_line_a0s0`、`a2s0`、`a3s0`），LINE/ROAD measured；
* **2站隔离**，原因原样保留：
  * `known_line_a1s0`：paint appearance agreement 0.446 < 0.9
    （168 投影真值点；near_left 0/84、near_right 75/84）——标注通道忽略遮挡，
    标签标线处 RGB 无漆面证据；
  * `known_line_a4s0`：标定后投影残差 median 1.56 px > 1.0（几何未证实）
    且 paint appearance agreement 0.488 < 0.9（near_left 56/84、near_right 26/84）；
* 数据集审计：`dataset_id 868a4155…08eb`，9帧、9 trainable、rejected=0、
  group/content/exposure overlap=0、内容摘要 9帧9唯一；UNKNOWN x3
  （road px 0 < 200，ROAD 通道未证实，按资格掩码处理）；
* cleanup 完成，`game_pids() = set()`。

采集批次验收与冻结导出门是两道独立门：本批 5/5 站过批次验收，但外观一致性门
再隔离 2 站。隔离站点不降门、不入训练；导出池保持 3 站 9 帧独立
LINE=measured 站点（a0/a2/a3，与 WCU4 的站点索引相同但 offset=9 为独立采集）。

跨池独立性证据（`crosspool_overlap_audit.json`）：WCU4∪WCU5 共 18 帧、18 个唯一
内容摘要、content/exposure/group overlap=0、0 重复组——两批为独立采集而非复制。
