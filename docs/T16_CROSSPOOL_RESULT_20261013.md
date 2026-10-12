# 第二线帧池族跨池配对结果（2026-10-13）

预注册：`docs/T16_CROSSPOOL_PREREG_20261012.md`（commit `8fb2bc7`）。
实验目录：`logs/experiments/t16_crosspool_20261012/`。

## 1. 输入资格

* 旧 human 池：确定性抽样 9 帧；新 engine 池：多站点验证后准入 9 帧；
* 输入审计：18 帧，两个 `map/source_id` 组，`group_overlap=0`、
  `content_overlap=0`、`exposure_overlap=0`，checked=true；
* 训练/评测：2 臂 × 5 配对种子 × 60 steps，20 作业全部完成，0 失败；
  评测为同一 8 个 reviewed_full 开发包、112 帧、同一 line IoU 口径。

## 2. 逐种子结果

| seed | H human IoU | E engine IoU | E−H |
|---:|---:|---:|---:|
| 42 | 0.2959 | 0.4044 | +0.1085 |
| 43 | 0.4835 | 0.4879 | +0.0044 |
| 44 | 0.3530 | 0.4686 | +0.1156 |
| 45 | 0.2296 | 0.4254 | +0.1958 |
| 46 | 0.3381 | 0.4155 | +0.0773 |
| **mean±std** | **0.3400±0.098** | **0.4403±0.032** | **+0.1003±0.069** |

配对比较（复用 `PairedResult`，含崩坏种子诊断）：`n=5`、95% paired-t halfwidth
0.0859、`verdict=candidate_better`；E−H **5/5 正向**；两臂均无 collapse seed，
因此排除崩坏后的均值仍为 +0.1003。

次级：recall E−H mean **+0.0830±0.0699，5/5 正向**；precision E−H
mean **+0.1140±0.1301，4/5 正向**（seed43 轻微负差）。

## 3. 解释边界

* 这证明的是：在匹配 9 帧预算、同初始化/步预算、同开发评测下，**新 engine-verified
  多站点池族在本次试点上优于旧 human 池子集**；不能把它解释为 engine truth 普遍
  优于人工标注，也不能拆分 provenance、场景多样性和帧内容的贡献；
* 样本池小、单地图族、5 seeds，属于探索性配对证据；不晋级驾驶模型、不改任何硬门；
* 新 engine 池的 depth 证据按站点有效范围限制（known/material 支持 5–15/5–10m，
  slope depth UNKNOWN）；训练准入来自完整投影/覆盖/外观冻结门，而不是把 UNKNOWN
  深度升级成 PASS；
* 结果解锁“第二线帧池族已入库、跨池族配比可继续”，不解锁安全、泛化或自动学习
  普遍收益主张。
