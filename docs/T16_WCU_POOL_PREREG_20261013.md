# west_coast 聚合池（36帧）配对训练预注册（2026-10-13）

* 问题：六个小池的配对方向不稳定（30 配对 25 负、3 champion_better，
  但第六批均值整体翻正且 4/5 正向）——论文假设「小池内容/种子/预算交互主导」。
  本实验把**全部已准入** west_coast LINE=measured 帧聚合成单池 WALL
  （12 站/6 批/36 帧，批间内容零重叠已审计），直接检验：池扩大到 4 倍后
  方差与方向是否稳定；
* WALL：WCU(a0) ∪ WCU3b(a0,a1) ∪ Wfix(a0,a1) ∪ WCU4(a0,a2,a3)
  ∪ WCU5(a0,a2,a3) ∪ WCU6(a2)，12 个 run 目录、36 帧、全部 engine_verified；
  聚合池审计 `dataset_id 2373a91d…57e4`：36 trainable、0 rejected、
  group/content/exposure overlap=0、36 帧 36 唯一摘要；
* H human：`t16_crosspool_old_subset_20261012/front_main`，9 帧（不变）；
* 初始化、5 seeds `{47..51}`、60 steps、per-run split、batch/lr、8开发包和硬门
  全冻结（与六批小池配对同一协议）；
* 记录逐种子 IoU/recall/precision、WALL−H、collapse、UNKNOWN，
  并与六批小池的均值/标准差/正向数对照（方差是否随池大小收缩）；
* 不改硬门、不消费最终集、不晋级模型；
* 聚合只使用已准入证据，不新采集、不降门、不改写任何旧批次记录。
