# 第四 LINE=measured 独立池族预注册（2026-10-13）

状态：预注册先于采集。基于前一批 5-anchor offset=90 仅 3/5 站通过，
本轮切换到新的 `anchor-offset=120`，只采 known_line，避免 material/scene 因子混入。

* Tech RPC 50000；Italy；`--anchors 5 --sites 1 --scenes known_line`；
  `--frames 3 --step-m 2 --anchor-offset 120 --line-convention tiers`；
* 目标：≥3 个独立站点、≥9 个帧；道路链布站失败/投影/覆盖/外观隔离原样保留；
* 导出准入必须 `engine_verified` 且 channels LINE/ROAD=`measured`；
* 数据审计 group/content/exposure overlap=0；
* 后续训练固定：H human 9 帧、E4 新池、同初始化、5 seeds `{47..51}`、60 steps、
  同一 8 个开发包、硬门；逐种子记录 IoU/recall/precision、collapse、UNKNOWN；
* 不降门、不消费最终集、不新增人工标注、不晋级模型。

## 修订 A/B（2026-10-13）

A 版 offset=120：5 个 anchor 中 4 个道路链 <30m 失败，1 个站 coverage 0.742/0.598
被冻结准入隔离；失败与隔离均保留。B 版改用 **offset=100**，仍为 5 anchors×1
known_line×3 frames，目标至少 3 站；其余配置/准入门不变。
