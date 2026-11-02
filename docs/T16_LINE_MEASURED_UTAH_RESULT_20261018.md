# utah 资格采集结果（第一批，2026-10-18）

预注册：`docs/T16_LINE_MEASURED_UTAH_PREREG_20261018.md`（唯一变更因子：
地图 east_coast_usa → utah + 线材质/线宽按 utah 档案 line_yellow_double
@0.40 m，其余与 ECA3 逐项一致）。
证据：`logs/m5_seg/collect_e5line_utah1_20261018/`（采集包 + 启动挂起日志
attempt1）、`logs/experiments/t16_linepool_utah1_20261018/`（导出报告）。

* 配置与 ECA3 逐项一致：Tech RPC、`--anchors 5 --sites 1 --scenes
  known_line --frames 3 --step-m 2 --anchor-offset 0`、线位约定 symmetric；
  生成器 script_sha16 `b1bdd538ca6fb6d4`；报告自证键
  `map_profile.line_width=0.4`、白/黄均 line_yellow_double；5 站全部生成、
  15 帧采集、cleanup closed=True、进程归零；
* **annotation 可写性修复在 utah 成立**：5 站全部写入 annotation 线类像素
  （3434/799/2948/1350/5425）——ECA1/ECA2 的"annotation 全丢"模式未复现，
  line_yellow_double@0.40 m 档案生效；
* **冻结导出准入：0/5 全隔离、0 帧（目标 ≥2站/≥6帧未达成）**。逐站理由
  （冻结门一字不改：逐实例 coverage≥0.8、projection median≤1px 且
  mean≤2px、paint_agreement≥0.9、engine_verified）：
  * `known_line_a0s0`：coverage 1.0（left/right 均 1.0）、投影 median
    0.829px 过，**paint 0.485 挂**（left 0/85、right 82/84）；逐帧 label
    像素侧证据：left 真值点 RGB 均值 ~(106,98,74) 三帧全程无漆
    （0/204–206），right 呈条纹色（max (195,169,142)、frame2 均值
    (185,161,124)）——annotation（深度盲）写得进、beauty 无漆的站点局部
    遮挡/覆盖案例，导出门正确拒绝（标签中毒守门同一机制）；
  * `known_line_a1s0`：paint 1.0、coverage 0.9189/0.8558 均过，
    **投影 mean 2.666px >2.0、median 2.696 >1.0 挂**；
  * `known_line_a2s0`：paint 1.0、coverage 0.9872/0.9506 均过，
    **投影 median 1.66px >1.0 挂**（mean 1.669 <2.0）；
  * `known_line_a3s0`：**right 实例 coverage 0（annotation 不载该线）+
    投影 14.35px 挂**（left 0.9778 过、paint 1.0）；批次验收即隔离
    （coverage 0.495）；
  * `known_line_a4s0`：**right 实例 coverage 0.5472 挂 + 投影 4.266px 挂**；
* 采集时批验收（较宽口径）4/5 合格、1 隔离；**导出冻结门最终 0/5**——
  两套口径的差异如实并列，不用批验收替代导出门；
* 首次启动挂起证据保留（同命令重试后成批，非采集内容失败）：attempt1
  游戏启动卡死——engine 日志 t=4.85s 写到音频初始化行即截断、CPU 约 15%
  （后台节流渲染）、内存平稳、无 WER 崩溃/挂起事件、窗口无模态对话框；
  collector 阻塞于 `LoadScenario → ack("MapLoaded")` 20+ 分钟。杀本次启动
  进程后**同一命令重试一次成功**。日志原样保留
  `collect_e5line_utah1_20261018.attempt1_hang.log`；
* 投影残差模式（定因线索，本批次内不改任何东西）：a3s0 三帧均 ~14px
  （系统性几何错位）、a4s0 沿 2m 步进逐帧 [1.634, 4.555, 6.61] 递增、
  a2s0 [1.446, 1.66, 1.901] 缓增、a1s0 平稳 2.3–3.0px、a0s0 平稳
  0.79–1.02px；全部 curve_deg=0（非链曲率近似误差），slope −3.28%…6.94%；
  批次校准后 mean 2.37 / median 2.167（ECA3 同口径 0.59/0.59）。a4s0 即
  探针站点（mid 722.41,347.19,143.80）：diag 单帧校准 0.892px 与步进序列
  首帧 1.634 同量级、后两帧漂移至 6.6px——需"原地多帧 vs 步进序列"对照
  探针分离相机位姿误差与链几何误差；
* **不降门、不消费最终集、不新增人工标注、不提升模型**；5 站隔离证据
  全部保留；导出 0 帧，无输入审计对象（无帧进入任何训练/开发池）；
  探针/诊断产物（diag_utah_*）不进训练池；
* 后续（另行安排，修复前不重采）：a0s0 left 线遮挡物定因（beauty 覆盖物
  是什么）；投影残差步进漂移定因（原地 vs 步进对照）；a3s0 系统性 ~14px
  错位定因。重采仍按同一预注册协议与冻结门执行。
