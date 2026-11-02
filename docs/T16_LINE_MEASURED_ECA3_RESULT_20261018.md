# east_coast 第三批 LINE=measured 采集结果（2026-10-18，annotation 可写材质修复后）

预注册：`docs/T16_LINE_MEASURED_ECA3_PREREG_20261018.md`（唯一变更因子：
east 线材质/线宽 line_white/line_yellow@0.15m → line_yellow_double@0.40m）。
证据：`logs/m5_seg/collect_e5line_eca3_20261018/`（采集包）、
`logs/experiments/t16_linepool_eca3_20261018/`（导出+审计）。

* east_coast_usa、offset=0、5 anchors、5 known_line 站点、3帧/站、
  线位约定 symmetric，与 ECA1/ECA2 逐项一致；
* **冻结导出准入：2站/6帧 `engine_verified`，LINE/ROAD measured**
  （目标 ≥2站/≥6帧**达成**——east 地图首次准入）：
  * `known_line_a0s0`：coverage 0.9889（逐实例 left 0.9885 / right 0.9889）、
    after_px_median 0.598（mean 0.609）、paint_agreement 0.9941、
    认证点 167/180（丢弃：出画 11、半径内无线 2）
  * `known_line_a4s0`：coverage 1.0（left 0.9881 / right 1.0）、
    after_px_median 0.587（mean 0.637）、paint_agreement 0.9941、
    认证点 159/180（丢弃：出画 11、半径内无线 10）
* 隔离 3 站原因原样保留（逐实例门 + 投影门 + 漆面一致门，三类都触发过）：
  * `known_line_a1s0`：逐实例 coverage left 0.4166 / right 0.2151；
    投影 10.73px；paint 0.030——三门全挂；
  * `known_line_a2s0`：left 实例 coverage 0.9889 过、**right 0.4216 挂**
    （逐实例要求每条线都达标→整站隔离）；投影 3.985px；paint 0.8512；
  * `known_line_a3s0`：coverage 0.9889、投影 0.714px 均过，
    **paint_agreement 0.0536 挂**——annotation 有线类像素（5503）但真值点
    RGB 无漆，正是标签中毒守门（annotation pass 不做深度/遮挡测试）要拦的
    情形，隔离并保留证据；
* 采集时批验收（较宽口径）4/5 合格、1 隔离；**导出冻结门更严**
  （逐实例 coverage + 投影 + 漆面），最终 2/5——两套口径的差异如实并列，
  不用批验收替代导出门；
* 输入审计（`m5_seg_dataset_audit`，只读）：6 帧、6 trainable、rejected=0、
  `line_masked_px=0`、group/content/exposure overlap 全 0；凭证
  `engine_verified`（verified）生效；
* ground_alignment 5/5 measured（delta ≈0–0.02 m）；Tech 收尾 cleanup 通过
  （closed=True、进程归零）。

结果：east_coast_usa 第三图资格打通——ECA1/ECA2 的 0/5 是 east 线材质
annotation 不可写所致（定因与宽度扫描见
`T16_LINE_MEASURED_ECA2_RESULT_20261018.md`），换 annotation 可写材质后
同锚点族同门重采 2站/6帧准入。**不降门、不消费最终集、不新增人工标注、
模型未晋级**；隔离 3 站全部保留。本池 6 帧可进入后续 LINE=measured 配对
研究（另行预注册），不作收益/泛化主张。
