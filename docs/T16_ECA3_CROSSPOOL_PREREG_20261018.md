# east_coast 首批 LINE=measured 配对训练预注册（2026-10-18）

* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧；
* E3 east_coast：`t16_linepool_eca3_20261018/auto_truth/m5auto_a0_known_line_a0s0`
  + `m5auto_a4_known_line_a4s0`，2 站 6 帧（engine_verified；ECA3 冻结导出门准入，
  目标 ≥2站/≥6帧达成，见 `T16_LINE_MEASURED_ECA3_RESULT_20261018.md`）；
* 跨池审计：全部导出池（26 个 auto_truth + 3 个 depthcheck_auto_truth 家族）
  + H + H2 共 **105 帧 105 唯一内容摘要、0 重复组**
  （`logs/experiments/t16_eca3_crosspool_20261018/preflight_pool_audit.json`，
  `m5_seg_dataset_audit --digest-scan` 只读）；
* 冻结协议（与 WCU/WCU3/Wfix/WCU4/WCU5/WCU6 配对逐项一致）：
  * 初始化 `logs/experiments/t14_e1_promotable_20260927/baseline/seed42/checkpoint_last.pt`
    （优化器从头；init_source=random）；
  * 5 seeds `{47..51}`、`--total-steps 60`、`--batch 4`、`--lr 1e-3`、
    `--val-frac 0.2`、`--split per-run`、`--sampler quota`、`--epochs 3`
    （步预算为主停止条件，epochs 由步预算推导）；
  * 评测：`m5_seg_eval_matrix` 同一 8 个开发包（review_pack_20260926
    reviewed_full：collect_dirt / collect_plain2 / diverse_curve / diverse_plain /
    plain / t13_corner / t13_junction2 / town），硬门与指标定义冻结；
* 记录逐种子 line IoU/recall/precision、E3−H 配对差、collapse 诊断、UNKNOWN；
* 不改硬门、不消费最终集、不新增人工标注、不晋级模型。
