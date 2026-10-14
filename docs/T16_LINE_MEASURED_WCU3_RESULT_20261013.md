# west_coast 第二批 LINE=measured 扩展结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_WCU3_PREREG_20261013.md`（commit `6e20299`）。

* offset=3、5 anchors×1 known_line×3 frames；5站采集；
* 冻结导出准入：2站/6帧 `engine_verified`，LINE/ROAD measured；
* 准入站点：a0 after_px_median 0.619、coverage 1.0、appearance 1.0；
  a1 after_px_median 0.488、coverage 1.0、appearance 0.9048；
* 隔离 3 站原因原样保留：appearance 0.762；coverage 0.6994+projection 3.64px；
  appearance 0.782；
* 输入审计：6 帧、6 trainable、rejected=0、group/content/exposure overlap=0；
* Tech 收尾 cleanup 通过，进程归零。

结果：west_coast 第二批新增 6 帧 LINE=measured，冻结准入和 truth_provenance 完成；
隔离站点不入训练，不降门。
