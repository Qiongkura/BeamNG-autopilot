# west_coast_usa LINE=measured 采集结果（2026-10-13）

预注册：`T16_LINE_MEASURED_WCU2_PREREG_20261013.md`（commit `c59bafe`）。

* 3 anchors、3 known_line 站点、9 帧目标；道路锚点与启动完成；
* 三站全部隔离：coverage=None、annotation line pixels=0、appearance checked=0、
  projection 校准无可用证据；
* 准入 0/3、LINE=measured 池 0 帧；无数据进入训练；
* 失败原因保留为地图/材质/渲染可见性阻断，未降低准入门；
* 收尾 owned cleanup 完成，进程归零。

结论：`west_coast_usa` 的受控入口和道路锚点可运行，但当前地图档案材质在实际
渲染/annotation 中没有提供可验证线证据，因此不能作为 LINE=measured 池。该负结果
与 E4 地图线通道阻断一致；需要地图材质/相机渲染对齐的独立修复后再试，不把它当
成训练池或跨地图收益证据。

## 6. 高程对齐修复尝试（2026-10-13）

代码提交 `88f1bde` 增加 west_coast_usa 的 `ground_z_at` 对齐与 teleport_alignment 记录，
因为此前 road-network `mid.z` 与实际场景高度存在明显不一致。纯逻辑回归和全量 pytest
通过；修复后重新采集仍出现 0 annotation line pixels、3/3 隔离，说明高程对齐不是唯一
阻断，当前地图的 line material/渲染可见性链仍需独立排查。没有降低准入门，也没有把
这批数据送入训练。

## 7. 详细排查结论

WCU4 的运行时调色板已保存：`SOLID_LINE=[255,196,128]`、`DASHED_LINE=[196,196,255]`，
annotation API 映射本身可用；raw annotation 仍没有 line 类。进一步检查发现：

* scene road-chain `mid.z=144.678`；
* `ground_z_at`/teleport 后实际车辆 z≈213.6；
* camera readback z≈213.6；
* z 差约 69m，生成的 DecalRoad 与 truth chain 在实际相机/地面下方。

因此 west_coast 当前首要阻断是 **road-network 高程与加载地图地面坐标不一致**，不是
单纯 palette 名称；ground 对齐修复已加入，但首次重试仍未消除这个不一致，后续要在
场景加载后再次测量并重建/平移受控道路对象，不能只换材质名。
