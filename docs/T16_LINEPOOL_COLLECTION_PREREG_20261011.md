# 第二线帧池族采集预注册（阶段二第 5 项，2026-10-11）

状态：**预注册，先于本轮任何采集提交**。依据：剂量扫描
([T16_DOSE_SWEEP_RESULT_20261011.md](T16_DOSE_SWEEP_RESULT_20261011.md) §5) 记录的
操作缺口——线任务配比实验当前只有 **1 个线帧池族**（e2_it3/front_fisheye，15 帧）。
本轮用既有受控场景生成器采一个**新线帧池族**（正控制：有线可见），同轮产出的
无线池如实保留为负例。

## 1. 采集配置（冻结）

* 入口：`scripts/m5_controlled_scenes.py`（审计过的受控生成入口）→
  `scripts/m5_auto_truth_export.py`（独立真值导出，冻结门不放宽）；
* 地图 italy；`--anchors 1 --sites 8 --frames 3 --step-m 2.0`；
  `--anchor-offset 24`（R2 用 12——错开站点避免与 e2_it3 池空间重叠；
  采集后仍以审计的空间冲突检查为准，重叠池隔离）；
  `--line-convention tiers --attempts 2`（attempts 经由 qualify 重试）；
  `--collect-timeout 600`；
* 输出：`logs/m5_seg/collect_e5line_20261011/`，导出
  `logs/experiments/t16_linepool_20261011/auto_truth/`。

## 2. 准入与记录规则（冻结）

* 准入沿用 `qualify_export` 冻结门（内容哈希、通道凭证、隔离）；**首轮 invalid
  原样保留**，重试另记；UNKNOWN（SHOULDER/ROLE/遮挡）原样保留；
* 每个准入池记录 `line_evidence`（帧数、line_px、线帧占比）——本轮的
  正控制判据：**至少一个准入池 line_px > 0**；若全部无线，则本轮如实记为
  「未获得线帧正控制」，不伪造；
* 新池与既有训练池/开发包的空间与组隔离以 `m5_seg_dataset_audit.py` 复核为准；
  空间冲突的池隔离，不进配比实验。

## 3. 用途与边界

* 采集到的线帧池供下一轮**跨池族配比实验**（第二线帧族 ≠ e2_it3），不给本轮
  任何训练结论；不晋级模型；不消费封存最终集；不新增人工标注；
* GPU 分钟与游戏进程照纪律记录，收尾归零。
