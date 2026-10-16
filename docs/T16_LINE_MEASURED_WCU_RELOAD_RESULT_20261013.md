# west_coast 场景加载后重建验证结果（2026-10-13）

使用 commit `b6034b0` 的新逻辑：场景加载后再次测量 `ground_z_at`，若与 road-chain
z 偏差超过 2m，则按最终地面重建同一请求地图的 Scenario/DecalRoad/truth chain，再采集。

## 结果

* 3 个 known_line 站点重新采集；
* 3/3 采集侧投影/覆盖/外观初筛完成；
* 冻结导出准入：**2 站/6 帧 engine_verified**，LINE/ROAD measured；
* 数据审计：6 帧 trainable、rejected=0、group/content/exposure overlap=0；
* 第三站因 projection median 1.15px >1.0px 隔离，原因原样保留；
* 收尾 cleanup 归零。

这确认“场景加载后按请求地图重建 + ground 对齐”链已经能让 west_coast 生成池通过
冻结 truth_provenance 准入；剩余隔离是单站几何门失败，不是全地图线通道失效。
