# 地图辅助候选关联（T08，影子）

日期：2026-09-22
状态：**可证伪的影子对照已跑通；未接线、未改任何权限。**

## 1. 输入只有真实存在的字段

| 输入 | 来源 | 说明 |
| --- | --- | --- |
| 链路身份 `link_id` | `connector.read_current_road_rule` 的 `n1->n2` | 地图里真实存在的节点对 |
| 链路方向 | 同上 `inPos → outPos` | **弦方向**（见 §4 的发现） |
| 车道数先验 `lanes_hint` | 同上 `lanes` 字符串（`-+`、`++--`） | **只当作"数量"**，绝不当宽度或偏移（计划 §1.7 明确禁止用 legacy `traffic.py` 的均分偏移） |
| 弯道先验 | 同上 `inRadius/outRadius` | 平均曲率 |
| 单行/可驾驶性/右侧通行 | 同上 | 侧别先验**默认不信任**（见 §3） |
| 感知候选**带身份** | 语义标线（kind / side / bearing / span / confidence），未来接 `lane/pairing` 候选 | 身份 = 候选自己的 id + 侧别 + 类型 |

**输出**：每个候选的 `association_score`、`hypothesis_id`（用的是哪条链路）、`map_fields`（**实际用到**的字段列表）、
`conflicts`（每条都点名"哪个地图字段 vs 哪个感知量"）、`abstain`（弃权原因）。**没有任何横向几何输出**——
`AssociationResult` 里不存在 `center/offset/lateral/target/authority/drivable/crossable` 字段，
模块也没有任何返回横向目标的函数（三条测试分别钉住字段名、签名与源码）。
分数也不依赖自车横向位置（签名与源码里都不出现 `pos/line_lat/lane_dev`，同输入同输出）。

## 2. 可证伪的弃权与冲突（计划列出的场景逐条落地）

| 场景 | 行为 |
| --- | --- |
| 无地图链路 | `abstain="no map link under the ego"`，**不改分数**（空行为：无地图时 B 臂必须等价于 A 臂） |
| 无量测（hold） | `abstain="candidate is not a fresh observation"`，分数归零 |
| 分岔/路口（多条候选链路） | `abstain="ambiguous: N plausible map links"`——**不替地图选一条路** |
| 车道数与实测宽度矛盾 | `conflict="width_contradicts_map: measured 7.00 m vs map lanes=1 implying 3.0-3.8 m"` |
| 候选朝向与链路方向矛盾 | 软（>25°）降分 / 硬（>45°）判冲突 |
| 弯道先验不符 | 目前只记录（软），未参与打分（见 §5） |
| 错误地图（方向反 180°） | 硬冲突 ✓ |

**A/B 对照的可证伪数字**（`compare_arms`）：改了几个候选的分数、其中几个**在带冲突的情况下**仍然被改动——
后者就是 false-acceptance 风险，代码直接把它算成 `false_acceptance_risk` 而不是靠论述。
B 臂只能重打分：不创建、不删除、不移动任何几何（有测试钉住候选顺序集合不变）。

## 3. 实车跑出来的两个发现（这才是影子对照的价值）

实车一帧（`logs/goal_20260921/map_assoc_probe{,2}.json`，italy，`--attach`）：

- **地图的右侧通行标志与道路矛盾**：原始 Lua 返回 `rightHandDrive: false`，而这段路是右侧通行
  （项目既有的横向记录把本车道放在中漆线右侧，AGENTS.md 也要求靠右）。第一版把这个标志当作侧别先验，
  结果 **4 个真实候选里有 3 个被标 `side_mismatch`**——先验在拿真实观测开刀。修法：**侧别先验默认关闭**
  （`TRUST_RIGHT_HAND_DRIVE = False`），只有在有人核对该等级的标志后才打开；未信任时该字段也不出现在
  `map_fields` 里，避免读者以为用了它。
- **链路方向是"弦"而不是局部切线**：`inPos→outPos` 是整条链路的弦，实测有候选与它差 **33–35°**
  （该链路较长/该处有弯），于是被标 `direction_loose`。这说明方向先验应该拿**局部切线**比，而不是弦；
  当前的 25°/45° 阈值对弦比较偏紧。**本轮只登记，不修**（修它需要链路几何采样，属于下一轮）。

修完侧别先验后的同一帧：4 个候选里 2 个带冲突（1 个 160° 反向硬冲突、1 个 33° 软冲突），
`rank_changed=True`、`false_acceptance_risk=2`——**这两个数字是"若接线会发生什么"，不是"已经发生"**。

## 4. 第二次实车（本轮追加）：地图先验没有改变排序

同一天稍后在同一段路重跑探针（`logs/goal_20260921/map_assoc_probe3.json`，车位
`(796.9, 730.1)`、heading −0.27 rad，即 T07 序列采集结束时车辆所在处）：

```
link=DR343_204->DR343_205 lanes=2 dir=174.07 oneway=False rht=False
perception candidates: 6
  mk0..mk3 score=0.625 hyp=DR343_204->DR343_205 conflicts=[]
  mk4      score=0.500 conflicts=['direction_mismatch: 47 deg off (hard 45)']
  mk5      score=0.500 conflicts=['direction_mismatch: 160 deg off (hard 45)']
rank changed=False   score changes=4   with conflict=0
shadow only: no geometry is created, moved or authorised
```

三件可核对的事：

1. **先验可用且不越权**：地图给出链路 ID、车道数、方向与单行标志，6 个候选拿到同一个假设 ID；
   `rank changed=False`——本次先验**没有**改变候选排序，因此也没有引入新的接受。
2. **冲突是显式输出**：mk4 与链路方向差 47°（软/硬边界 45°）、mk5 差 160°（反向），二者记为冲突并降到
   0.500，其余 4 个保持 0.625。**160° 那条是值得单独看的真实观测**：要么候选指向对向车道，
   要么候选身份判错——两种解释都指向 T08 的下一项（真值确认的候选身份）。
3. **`rht=False` 再次出现**，与 §3 一致：该标志与这段路的实际右侧通行矛盾，故仍未用作侧别先验。

**仍然不能声称的**：这不是关联正确率。没有真值确认的候选身份，所以"减少错误关联"与
"没有增加 false acceptance"都**未被测量**；本节能说的是：真实链路上影子对照能跑、能给出假设 ID 与
冲突原因，且本次没有改变排序。

## 5. 顺手修掉的一个真 bug

`traffic.RoadRuleView` 原先用 `bool(rhd)` 解析 JSON 布尔：如果哪天标志以字符串 `"false"` 到达，
`bool("false") is True`——布尔被反转。现改为显式 `_as_bool()`（`true/1/yes/on` 与 `false/0/no/off/""`），
未知值返回 `None`（UNKNOWN 而不是猜）。这类"布尔当数值读"的错误本计划已经反复抓到，故一并修。

## 6. 未做（明确登记）

- **未接线**：影子模块不进 `fsd_stack`，不影响参考选择、不给任何许可（计划要求"改变现行参考选择权限的接线
  必须作为明确行为变更验证"）。
- **人工/真值确认的候选身份**：验收要求"在真值确认的身份上减少错误关联"，本轮没有真值标注，
  因此**未测关联正确率**，也**不引用**任何配对率数字。
- **分岔/宽路/急弯/单侧线/过期地图**的完整矩阵：只有单元级合成反例（分岔弃权、错误方向拒绝、宽度矛盾）+ 两次实车单帧
  （§3、§4）；矩阵仍缺。
- 弯道先验参与打分、方向先验改用局部切线、多条链路枚举（真正的 roadnet 分支查询）未做。
