# T16 试车场静止条件实验：首格 dawn/clear 失败，批跑中断（2026-11-05）

## 状态

按用户要求收尾，本批**未完成**。预注册见
`T16_PROVING_GROUND_CONDITIONS_PREREG_20261105.md`（12 格，PASS/FAIL/
UNKNOWN 门冻结）。已跑/中断情况：

| 格 | 结果 | 证据 |
| --- | --- | --- |
| `gridmap_dawn_clear` | **FAIL** | ticks 12，`no_markings` 12/12，true 0，cert_both 0.0；环境读回 `tod=06:30 applied_tod=06:29:59` |
| `gridmap_noon_clear` | **UNKNOWN** | 探针进程存活约 10 分钟、引擎进程已退出且未产出 jsonl/报告；人工停止；无环境读回 |
| 其余 10 格 | **UNKNOWN** | 未运行 |

聚合（`logs/m5_seg/proving_conditions/conditions_table.json`）：
`FAIL 1 / UNKNOWN 11`；按预注册，UNKNOWN 不计 PASS。

## dawn/clear 判读

- 环境应用有效（读回 06:29:59），不是 UNKNOWN；
- 画面帧（`logs/m5_seg/_dawn_clear_frame5.png`）显示 decal 线在黎明光照
  下呈暗红色、与浅色地面/阴影对比很低；
- 分割→候选链路 **12/12 帧无任何 marking**（不是配对门问题），
  说明"黎明 + 该材质/曝光"下感知整体失效；
- 这是真实的鲁棒性负结果：`no_markings` 发生在候选层之前，属于
  输入域/光照问题，不能用配对参数修。

## noon/clear 中断

- 该格探针进程在引擎退出后仍未返回（无 `jsonl`/报告），人工停止；
  具体卡点未确认（疑似 `conn.open(launch=True)` 等待/重试路径），
  记 UNKNOWN，不当作任何结论。
- runner 目前对"引擎启动/连接挂死"没有硬看门狗；下次跑批前需要加
  每格墙钟上限（含启动阶段）并保留失败日志。

## 判定与下一步

- 本批不构成条件矩阵结论；门限未改、未降级、未消费最终集；
- 重启时：
  1. 给 runner 加硬超时（每格 ≤ 8 分钟，超时杀进程并记 UNKNOWN）；
  2. 先补 `noon_clear`（正午对照），再按天气分批跑 rain/fog；
  3. 黎明格可加一档更亮的 dawn（如 07:30）作为对照，但**不改**原格门；
- 驾驶格仍搁置（试车场无可行驶参考，见 6.2）。

## 产物

`logs/m5_seg/proving_conditions/gridmap_dawn_clear.*`、
`conditions_table.json`、`logs/m5_seg/proving_conditions_clear.log`、
`logs/m5_seg/_dawn_clear_frame5.png`。
