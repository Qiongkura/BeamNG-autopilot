# 第三 LINE=measured 大池族采集预注册（2026-10-13）

状态：预注册先于本轮采集。前两次扩展分别获得 9 帧和 3 帧；本轮改为多道路
anchor 设计，避免单条 60m 链无法布多个站点。

## 冻结参数

* Tech RPC `50000`；Italy；`m5_controlled_scenes.py`；
  `--anchors 3 --sites 2 --frames 3 --step-m 2 --anchor-offset 90`
  `--line-convention tiers`；场景类型轮转 `known_line material_mix slope_curve`；
* 目标：至少 4 个站点、12 个帧；anchor/站点布置失败原样保留，不把失败当准入失败；
* 输出：`logs/m5_seg/collect_e5line_large3_20261013/`，导出到
  `logs/experiments/t16_linepool_large3_20261013/`；
* 冻结准入：projection median ≤1px、coverage ≥0.8、appearance ≥0.9、
  `truth_provenance.report.verified=true`；任何隔离/UNKNOWN 保留；
* 采集结束必须 owned cleanup closed=true、无残留 Tech 进程；
* 不消费最终集、不新增人工标注、不改硬门。

如果目标站点数未达成，结果仍按实际站点数报告，不更改参数中途补采。
