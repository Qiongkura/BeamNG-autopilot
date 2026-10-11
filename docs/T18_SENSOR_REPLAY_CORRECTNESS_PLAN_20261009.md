# T18：同源传感器回放、Tech CPU 成本与预测一致性

2026-10-09。上一轮 T17 是真实进展：3351 全量门和两个 Tech run，70 帧生产
安全判定回放匹配；strict crossing/deadline 仍失败，车体铺装覆盖仍 UNKNOWN。
本批遵照「所有代码完成后再实测」，不凭单次跑得更远晋级模型或开关。

## 1. 需要解决的具体问题

1. Tech process 先对远场 corridor 降采样并转为 Python 点列表，随后只要已有
   payload.ray_hits 就整份丢掉。删除这一无消费计算；Lua 逻辑与保底路径不改。
2. tracker 用 process 完成时刻估速度，异步排队/CPU 时延混入观测间隔。改用已有
   poll-start proxy，并显式记录 fallback；真实 sensor capture time 依然 UNKNOWN。
3. predict_proba 文档承诺普通 argmax，但用插值后的概率 >= 比较，平局双真且
   边界与普通 predict 不一致。直接复用 _argmax_masks，概率图继续线性插值；
   缺 road/line 概率通道明确报错。不得重算或调参已消费的封存集。
4. detect_lines 自预测丢掉 road；外部 road mask 未校验形状。保留同次预测的
   road（显式 road 仍优先），外部尺寸不符时报错；不添加第二次推理或覆盖外部 line。
5. 只有真实同源数据才能验证 CPU 优化的耗时和输出。新增显式 --capture-range-replay
   诊断，保存第一份合格 warmup 距离输入及姿态、源错误、ego 尺寸、完整 source hash。
   原始数组无 pickle，文件摘要核对；完整标记最后写，部分产物不算成功。

## 2. 文件、复用和影响

- Tech providers：复用原下采样、近场地面过滤、LiDAR 聚类及融合，guard 只跳过
  后续不消费的 corridor 计算。追加 tracker time basis 和 source metadata。
- vision/segmentation：复用 _argmax_masks，原概率图和 postprocess 保持；只修
  raw API 的承诺及 detect_lines 参数门。新增原尺寸/缩放/平局/缺通道/同次 road 例子。
- experiments/range_replay.py：原始 payload 保存/读取/摘要校验、同一输入的串行
  比较。旧 reference 是修改前当前 providers.py 的冻结字节，不从 HEAD 猜用户工作。
  回放只处理输入，不连接游戏；保存全部重复时延、取较小 p95 并核对逐项几何。
- fsd_drive、入口及 pose_sweep：显式开关启用 Tech 捕获，并在 warmup 结束前
  停止捕获；保存是否捕获成功的诊断，资格检查须验证 artifact 和 capture 阶段。
  默认关闭；无输出路径/非 Tech 拒绝，不能把记录开销混入行驶计时。
- scripts：薄的回放评估入口；tests：同源输出、无命中保底、近场障碍、处理延迟
  下速度估计、文件损坏/部分输出、捕获默认关闭及资格门。

经典 CV、Steam、控制/硬约束、驾驶模型、默认异步开关和后处理形状阈值不改。
概率 raw API 边界修正不是旧实验追认；with_probs 掩码仍走同一原有后处理。

## 3. 验证顺序及证明范围

1. 修改前冻结 T17 source=9b10da96f91b5613e0e840f328bdf5c4392a2f9fdcf8b8df6fb6ccb054d89582
   的 providers.py，保存完整字节 hash；不修改用户其他改动。
2. 整批完成后跑定向、全量 pytest 和深度离线；冻结新源 hash。开发阶段仅
   确定性逻辑测试，正式时间比较在门结束且无游戏/训练后单独跑。
3. 新目录 Tech strict + sensor + goal，同 T17 固定模型，25s/150s，默认同步。
   捕获只在 warmup；资格按原曝光标准及新增 capture 真正完成检查，UNKNOWN 不通过。
4. 游戏关闭后同一 cloud/Lua/pose/尺寸输入，旧 reference 与新 process 串行重复
   （两个 round，每 round 10 次，warmup 2），几何/ray_hits 逐项比较，保留原时延。
   fresh tracker 单帧比较不验证真实多帧速度；后者用确定性时间正反例另证。
5. 报告包含因果链及八项审计；源/模型/输入/参考 byte hash、缺列/UNKNOWN、刷新
   实际消费、最终指令、路面真值、性能/deadline、碰撞/压线/停车和未测试项均单列。

本批证明范围是 CPU 无消费计算和 API/时间正确性；不能替代独立路面/漆线真值、
车体盲区、执行 ACK、多个配对闭环和真实学习收益。下一阶段这些仍须开发与验证。
