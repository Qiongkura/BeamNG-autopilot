# T16 Order 2：五类受控场景与自动真值实机验收（2026-09-27）

方案：`docs/T16_AUTONOMOUS_LEARNING_ROADMAP_20260927.md` §4.1/§4.2（路线 B）。
前置：`docs/T16_ORDER0_FREEZE_20260927.md`（契约）、
`docs/T16_ORDER1_2_REPORT_20260927.md` §5（原生地图线通道 0 覆盖 → 必须走路线 B）。

本轮把"程序化道路 + 生成侧真值"落成**可复现生成入口**，并在独占 Tech 上把
五类场景全部实机跑通。产物在 `logs/experiments/t16_scenes_20260927/`
（logs 不提交）；入口与库改动全部入库。

## 1. 可复现生成入口

`scripts/m5_controlled_scenes.py`（新增，纯逻辑测试 `tests/test_controlled_scenes.py` 6 例）：

- **材质用地图自己的 DecalRoad 材质**（从 `levels/italy` 的 items.level.json 统计
  得到，1333 处 `italy_road_markings_line_thin`、180 处 `_yellow`、624 处
  `italy_asphalt_overlay_light`）——生成场景与地图同材质，不引入域外外观。
- 五类场景（一次场景加载，站点沿道路链自适应布站）：

  | 场景 | 生成物 | 真值要点 |
  |---|---|---|
  | known_line | 白线（左）+ 黄线（右） | 2 条线实例，role=left/right |
  | known_no_line | 不生成线 | `line_generated=False`（对照） |
  | occluded_line | 白线 + 遮挡车（pickup 真车模型，压在线上 8 m 处） | 被挡点由深度判遮挡 |
  | slope_curve | 白线 + 黄线（选在弯/坡最大的站点） | 记录实测曲率/坡度 |
  | material_mix | 白线 + 土肩/碎石干扰带 + 蓝线 | 干扰材质不得被判成线 |

- 几何是**真实可见表面**：线是 0.15 m 宽的 DecalRoad 贴花（与地图画线方式一致），
  不是"突出路面的粗立方体"；`interpolate=False` + 1.0 m 节点间距，让"节点连线
  = 渲染路径"（Catmull-Rom 插值会让渲染线在节点间鼓出，实测导致 2 px 以上偏差）。
- 真值来自生成侧定义：每条线的节点链 + 宽度 + role；铺装边界来自 road network；
  路面点取行驶车道中心；站点 z 用**沿链 z 剖面**（坡道上 34 m 内可差 4 m）。
- 凭证形态逐场景落盘（`scene_<name>.json`）：生成器版本/脚本 sha、材质、线实例、
  站点、曲率/坡度、两种证据的完整 stats、相机标定。

## 2. 实机结果（italy，BeamNG.tech v0.38.5 + BeamNGpy 1.35.1，5 站点 10 帧）

```
known_line     线像素 467–496 | 到最近线像素 mean 1.9–4.2 px  ≤2px 0.30–0.68
known_no_line  线像素 0       | （对照：LINE 通道 not_applicable）
occluded_line  线像素 266–334 | mean 2.4–5.5 px | 深度判遮挡 5 处
slope_curve    线像素 533–534 | mean 5.9–6.6 px
material_mix   线像素 639–733 | mean 2.8–3.5 px
```

**跑通的部分**（可复现、有证据）：

1. 五类场景全部生成并采到帧；每个场景的线实例、role、材质、站点几何都落盘；
2. **生成的线贴花被 annotation 标成线类**（`annotation` 线类像素 467–1372，
   类色 = SOLID_LINE (255,196,128)）——这是路线 B 的关键使能，对照原生地图
   同一区域线类像素为 0；
3. `known_no_line` 的 LINE 通道判 `not_applicable`，依据是**生成器声明**
   （`line_generated=False`、`line_texture_embedded=False`）+ 帧内无线像素，
   不是"模型没预测线"；
4. 遮挡场景里深度判据生效（5–14 处 occluded，depth 已按真值点几何标定成米）；
5. 每次实机跑都自动记录"相机方向 vs 传感器读回"的角度差与深度标定系数。

**未通过的部分**（判定诚实失败，按方案 §4.3 应隔离并修生成器，不扩量）：

- 生成侧真值与**渲染结果**的像素级一致性只有 2–7 px（平地站点最好 1.9 px，
  11.76% 坡站点 5.9–6.6 px），2 px 半径内的点占比 0.02–0.68 → 多数帧
  `PROJECTION_MISMATCH`。原因按已测证据排序：(a) 贴花按地形投影渲染，横向
  有路拱/超高，我的弦线模型没含横向坡度；(b) 车辆/传感器异步（已加驻车制动，
  但 poll 到的帧仍可能早于状态读取）；(c) 亚像素与抗锯齿地板（拟合最优残差
  0.7–2.0 px，说明模型本身已接近，差的是渲染细节）。
- `occluded_line` 的深度标定被遮挡物污染（标定用了全部真值点，其中一部分在
  卡车后面）→ 遮挡判定不可靠；需要"无遮挡子集标定"。
- 外观证据只覆盖白/暖白/黄：material_mix 的蓝线按设计不判"像线"（记录在案）。

## 3. 本轮顺带修掉的三个真缺陷（都有实机证据）

1. **相机方向约定**：`Camera(dir=r)` 的实际世界朝向是 `R_vehicle·R_z(+90°)·r`
   （受控探针 `camera_convention.json`：传 `(1,0,0)` 得到车体左向、`(0,-1,0)`
   才是车头前向）。按世界方向直接传会让相机朝侧后方。
2. **`cam.get_direction()` 不含车辆俯仰/侧倾**：坡道上与渲染光轴差 2–6°
   （11.5% 坡站点需 -6° 修正、平地 -2°），且随帧间姿态变化。改为**用车辆状态
   `pos/dir/up` + 挂载偏移独立重建相机位姿**后，基线距离从 4.9–21.5 px 降到
   2.0–7.8 px。
3. **深度缓冲语义**：uint8、非 NDC、非米制；用真值点几何拟合出
   `true_m ≈ a·raw + b`（实测 a≈0.52、b≈1.25），遮挡判据因此可用；
   全 0 缓冲是"未渲染帧"，不是"深度标定错"。

## 4. 仍未做（下一步）

1. 把像素级一致性做到 2 px 内：给线节点加**横向坡度**（用 road network 的
   left/right 高程差），或用渲染回读做一次静态标定并落盘；
2. 遮挡场景的**无遮挡子集**深度标定；
3. 小批次验收通过后，再按 §4.4 扩到数百帧有效样本池（路段/材质/曲率/光照），
   训练/开发/最终按场景族划分；
4. 把 `verify_batch`/`write_truth_credentials` 接进采集链，并加
   "`checked>0` 且几何一致性达阈值才允许写 `engine_verified`"门槛。
