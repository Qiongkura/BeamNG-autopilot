# west_coast_usa 修复后冻结准入结果（2026-10-13）

本轮使用干净环境和修复后的 `d5c461e` 入口；完整 pytest 通过后重采。
Tech RPC 50000，3 anchors × 1 known_line × 3 frames。

## 结果

* 3 站采集侧投影/覆盖/外观初筛均完成；
* 冻结 truth export：**1 站/3 帧 engine_verified 准入**；
* 凭证通道：LINE=`measured`、ROAD=`measured`、SHOULDER/ROLE=`unknown`；
* 输入审计：3 帧、3 trainable、rejected=0、group/content/exposure overlap=0；
* 另 2 站隔离且原因原样保留：
  * near_right coverage 0.7308 < 0.8；
  * projection median 1.15px > 1.0px；
* 收尾 owned cleanup 通过，BeamNG 进程归零。

## 结论

west_coast_usa 在修复 `Scenario(map_name)` 跟随请求地图、ground alignment 证据和运行时
palette 保留后，已经获得首个真正 `LINE=measured` 的独立池（3 帧）。这证明先前
“3/3 全无线证据”主要来自场景地图传播/坐标链错误，而不是材质档案本身无法工作。
隔离站点仍按冻结门处理，未降门、未纳入训练。
