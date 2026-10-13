# 第四 LINE=measured 独立池族结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_LARGE4_PREREG_20261013.md`，含 offset120 失败修订与
offset100 重试。最终使用 offset100 的 5-anchor known_line 采集：

* 5 anchors 中 1 个道路链过短失败，4 个站点采集；
* 冻结采集门 4/4 通过（projection/覆盖/外观初筛）；
* 导出冻结准入：2 站、6 帧 `engine_verified` LINE/ROAD measured；
* 2 站因 projection median 1.54px / 1.29px 隔离，原因原样保留；
* 准入池数据审计：6 frames、6 trainable、rejected=0、group/content/exposure overlap=0。

这轮形成了新的独立 LINE=measured 池族（6 帧），超过前一轮 3 帧；仍是小池，不能
外推学习收益。