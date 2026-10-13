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
