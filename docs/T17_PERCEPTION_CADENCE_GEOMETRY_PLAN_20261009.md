# T17：感知时延与安全几何证据（2026-10-09）

## 本批目标及顺序

继承 T16 的实测失败：31/31 tick 超预算，range p50 304.4 ms，markings p50
160.83 ms；生产压线标志与物理车体/最终控制路径回放口径不同。上述数字只描述
T16 的单次诊断，不能用作新版本性能或安全通过证据。

遵照用户要求，先完成以下全部代码及单测，再运行全量离线门，最后启动 Tech
实测。实测期间不运行回归、训练、论文渲染。保持当前驾驶模型、阈值、strict
规则、默认异步开关；本批不宣称训练质量提高，不需要人工标注。

## 代码、复用和数据流

1. `vision/lanes.py` 增加学习式提取专用的组件 ROI 选项；`vision/segmentation.py`
   的学习式白标线提取接入（黄臂来源策略保持原样）。复用已有连通域 stats，在组件包围框内查找像素，保留
   原来的像素顺序、抽样、投影、分类和阈值。经典 CV 入口保持原默认路径。
2. `runtime.py` 的 `RangeSample` 增加可选逐样本 metadata；Tech provider 将
   锁等待、LiDAR poll、Lua poll、解包过滤、CPU 准备、聚类与融合耗时及各源错误
   写入样本，避免用共享实例计时属性。输出几何和源融合策略保持一致。
3. `fsd_stack.py` 的同步扫描/异步 fetch 在调用开始记录源代理时间，包含 IO 和
   处理等待；复用的样本保留 metadata。遥测明确这是 poll-start proxy，传感器
   实际 capture timestamp 仍 UNKNOWN；不把 poll 代理冒充真采集时间。
4. `geometry_probe.py` 增加安全评估输入的拷贝与回放，复用 constraints 的同一
   感知边界选择、vehicle_body 的同一车体几何。`safety_monitor.py` 在实际检查
   点记录场景位姿、感知边界、实际输入或 hold 路径、0.25m 膨胀和 0.05m 容差。
   `fsd_drive.py` 仅在已有 `--probe-geometry` 开关启用该拷贝，再随最终指令归档。
   旧 physical/final-control 回放保留；缺安全记录的旧帧为 UNKNOWN，不能补猜。
5. `tests/` 增加不同掩码/海拔的 ROI 等价、假时钟证明慢 poll/处理不会刷新成
   age=0、逐源失败 metadata、margin/不同路径/hold/缺边界回放与独立拷贝反例。

## 验证及通过条件

- 开发阶段仅确定性单测；整批收齐后 `scripts/dev_validate.ps1`（pytest + 深度
  离线）。任何失败先修代码，全部收齐并重新过门后才实测。
- 保存 source/config/model hash，生成新目录，启动前/结束后枚举相关进程；既有
  T16 记录和用户改动均不覆盖，不新增控制进程并发。
- Tech strict + sensor + goal + geometry probe，用有限墙钟和显式 owned cleanup；
  对比阶段耗时、deadline、实际曝光时长，曝光不足仍判未合格。
- 安全评估回放匹配生产 current/planned/distance/index/side；物理车体及最终
  指令回放分别统计。无压线 witness 不等于可安全执行；执行 ACK、独立铺装真值
  和盲区依然单列 UNKNOWN。
- 汇报按 REPORT_FORMAT 的因果链及八项审计。代码未提交时明确 dirty source；
  不以测试通过或损伤增量为零替代 strict 驾驶通过。

## 风险和未知

ROI 优化是否降低总 markings 时延需 quiet 实测；距离瓶颈是 IO 还是 CPU 尚未
分解；较诚实的 age 可能更早触发 stale 停车；前视相机车体盲区本批不伪造补齐。
默认关闭安全几何拷贝；开启后额外耗时需计入 probe 开销。
