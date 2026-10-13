# 第二个 LINE=measured 独立池族训练回归结果（2026-10-13）

预注册：`docs/T16_LINE_MEASURED_LARGE2_SMOKE_PREREG_20261013.md`（commit `859319b`）。
采集修订与结果：`T16_LINE_MEASURED_LARGE2_PREREG_20261013.md`；实际新池规模为
3 帧（known_line anchor1），material_mix 被准入门隔离，anchor0 因道路链长失败。

## 1. 训练与资格统计

* 5 seeds `{47..51}`、60 steps、相同初始化、相同 8 开发包、相同硬门；
* 5 训练 + 5 评测全部完成，0 失败；
* 每个 seed `train_hist`/checkpoint 记录：
  * `frames=180`（3 帧按 per-run train/val 消费）；
  * `masked_frames=0`；
  * `ROAD=measured:180`、`LINE=measured:180`；
  * `SHOULDER=unknown:180`、`ROLE=unknown:180`。

## 2. 开发集逐种子读数

| seed | line IoU | recall | precision |
|---:|---:|---:|---:|
| 47 | 0.3487 | 0.4214 | 0.6688 |
| 48 | 0.1466 | 0.1478 | 0.9479 |
| 49 | 0.2984 | 0.3036 | 0.9460 |
| 50 | 0.2841 | 0.3678 | 0.5551 |
| 51 | 0.2522 | 0.2926 | 0.6462 |
| mean±std | 0.2660±0.0767 | 0.3067±0.1008 | 0.7526±0.1770 |

UNKNOWN：IoU/recall/precision 均为 0/5；训练失败 0；collapse：按既有规则臂中位数
0.2841 的 0.5 阈值，最低 seed48=0.1466（略高于 0.1421），因此 **0 个正式 collapse**。

## 3. 结论边界

* 证明了第二个独立 LINE=measured 池可以被资格掩码训练器实际消费：LINE/ROAD 进入
  损失与评测分母，SHOULDER/ROLE 保持 UNKNOWN；
* 3 帧池过小，seed48 的低 IoU 显示高方差；不做收益/泛化/来源排序主张；
* 这不是跨池因果对照，也不晋级模型；下一步需要更大 LINE=measured 池或更多独立站点，
  而不是用这 3 帧配方宣称学习收益。
