# 停车原因生产输入回放：归因结果（2026-10-11）

任务来源：[复核方案](PROJECT_NEXT_DEVELOPMENT_REVIEW_20261011.md) 阶段二第 4 项（P1：
建立停车原因的生产输入回放——合格驾驶记录约 92.6% 观测时间低速/停车，需区分合理
停车与误停）。本轮只做**解释**，不改任何阈值与行为。

## 1. 数据与工具

* 输入：`--probe-geometry` 捕获的两轮生产链（`e5_postfix_sendgate_20261011` 44 帧、
  `e5_postfix_sendgate2_20261011` 47 帧，共 **91 帧**，同一 B 臂锚点、同一协议）。
* 判定复算：`scripts/m5_geometry_replay.py`（生产安全谓词离线复算）——**91/91 帧一致、
  0 失配、0 命令断链**（T18 27/27 之后扩展到完整驾驶链）。
* 时间段分类：既有 `scripts/m5_stall_attribution.py`——两轮停车段全部归入其
  `no_executable_path` 类（停止帧 86.4% / 97.9%）。
* 几何确认（新）：`scripts/m5_stall_geometry_check.py`——对每帧从捕获栅格重建占用，
  复用生产 cell→world 转换与监视器自己的前方走廊测试，对「无可执行路径」帧回答
  「是真无路，还是候选全被拒/走廊空闲却被拒」。

## 2. 逐帧归因（91 帧）

| 类 | 帧（轮1+轮2） | 机制 |
|---|---|---|
| `C_perception_missing` | 23（15+8） | 感知车道丢失 → strict 规划器以 `strict_no_perception_lane` 拒绝全部候选；记录原因却是 "no drivable path"。走廊不可算（无 lane_ref，UNKNOWN 如实保留） |
| `A2_candidates_all_rejected` | 20（7+13） | **车道可用**，规划器生成 14 个候选后**全部内部拒绝**：累计 lane_cross **191**、body_cross **103**、no_forward_progress 3、blind_path 3 |
| `B2_rejected_blocked_ahead` | 28（20+8） | 规划器给出路径，安全监视器拒绝（24 planned body-cross、2 contact risk、2 current body-cross） |
| `D1_accepted_slow` | 19（5+14） | 路径被接受但车速 ≈0（≤0.02 m/s） |
| `D2_accepted_moving` | 1（0+1） | 接受且移动 |
| `B1_rejected_corridor_free` | **0** | — |

## 3. 关键发现

1. **没有一帧是「几何空闲却被拒」**（B1=0）：全部 48 个拒绝帧（A2+B2）前方走廊
   10 m 窗口内都有障碍侵入（1.0–5.6 m，p50 ≈ 3.1 m）。在本轮捕获的驾驶里，
   **不存在「误停」 population**；停车是保守机制在真实近场证据下的输出。
2. **记录原因 "no drivable path" 混合了三种机制**：感知丢失（C）、候选全被
   lane_cross/body_cross 拒绝（A2）、以及安全门拒绝已选出路径（B2，各自有专名）。
   这正是复核要求「不让一个布尔值混用两种含义」的实证；遥测原因字段的细化记录
   是后续代码批次的工作，本轮不改。
3. **两帧 C 类带爬行油门**（0.13/0.41）：`path hold (creep)`——有界保持窗口复用
   此前验证轨迹的既名机制，非矛盾；归因表逐帧保留。
4. **机制链与论文死锁解剖一致**：车体偏离车道中心 → 候选扇面系统性扫掠越界
   （lane_cross/body_cross 计数）→ 偶尔选出的路径再被安全门拒绝 → 净行进趋零。

## 4. 边界

* 单锚点、单路线、单地图、两轮 91 帧；结论限定为本轮捕获的驾驶，不外推为
  「系统从不误停」。
* 走廊「前方有侵入」不是「不可通过」的证明，也不是「必须停车」的证明——两类
  几何陈述按预注册分开记录；改阈值之前需要阶段二的配对实验与独立近场真值。
* 本轮零代码行为变更；`m5_stall_geometry_check.py` 为只读分析入口。
