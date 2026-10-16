# west_coast 第三批 LINE=measured 扩展结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_WCU4_PREREG_20261013.md`（commit `c8814c2`）。

* offset=6、5 anchors、5 known_line站点、15帧目标；
* 冻结导出准入：3站/9帧 `engine_verified`，LINE/ROAD measured；
* 3站投影/覆盖门通过，appearance也通过；
* 2站因 appearance 0.771/0.452 隔离，原因原样保留；
* 数据审计：9帧、9 trainable、rejected=0、group/content/exposure overlap=0；
* cleanup 完成，进程归零。

这批将 west_coast 新准入池扩大到更多独立站点；隔离站点不降门、不入训练。
