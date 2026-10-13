# 第二个 LINE=measured 大池族预注册（2026-10-13）

状态：预注册先于采集。上一轮已获得 3 站/9 帧 LINE=measured 池；本轮使用不同
anchor offset=60、5 个站点、3 帧/站，目标建立第二个独立 engine-verified 大池族。

* Tech RPC `50000`；Italy；`m5_controlled_scenes.py`；`--anchors 1 --sites 5`
  `--frames 3 --step-m 2 --anchor-offset 60 --line-convention tiers`；
* 输出：`logs/m5_seg/collect_e5line_large2_20261013/`，导出到
  `logs/experiments/t16_linepool_large2_20261013/`；
* 站点类型使用 `known_line material_mix slope_curve` 循环；
* 冻结准入：投影 ≤1px 中位、覆盖 ≥0.8、外观 ≥0.9、truth_provenance verified；
  任一站失败隔离，不降门；
* 目标：至少 3 个站点、9 个 LINE=measured 帧，否则负结果原样保留；
* 收尾 zero process；不消费最终集、不新增人工标注。

## 修订 A（2026-10-13）

原 `--sites 5` 在实际 anchor offset=60 的道路链上被生成器拒绝：链长 60m，小于
5站所需的最小间距约束（30m×4）。该失败原样保留。为完成同一预注册因子的可执行
验证，**只把站点数改为 3**（同 offset、同 frames/step/scene types/冻结门不变），
目标调整为至少 3 站、9 帧；不把道路布站失败读成数据准入结果。

## 修订 B（2026-10-13）

修订 A 的 `--sites 3` 在该 offset 的单条道路链上仍被拒绝（实际只走到 2 站，
链长 60m；失败原样保留）。本次改为 **`--anchors 2 --sites 2`**：每条独立道路链
只布 2 站，目标 4 站/12 帧；offset、场景循环、frames/step、冻结准入门全部不变。
跨 anchor 的组/content/exposure overlap 仍由 dataset audit 检查，任一冲突隔离。

## 修订 C（2026-10-13）

实际 anchor0 道路链不足，anchor1 仅 `known_line_a1s0` 通过冻结准入并导出 3 帧
LINE=measured；`material_mix_a1s1` 由覆盖/残差/外观门隔离。目标“大池”未达成，
失败原样保留。为不丢失独立性证据，后续训练回归将把这 **3 帧新 anchor pool** 作为
新的独立 engine-verified 臂；结论限定为新池小样本资格回归，不称为大池收益。
