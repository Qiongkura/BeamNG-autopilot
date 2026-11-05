# T16 试车场静止条件实验：完整 12 格结果（2026-11-05）

## 状态

预注册见 `T16_PROVING_GROUND_CONDITIONS_PREREG_20261105.md`（12 格、
PASS/FAIL/UNKNOWN 门冻结）。本表为**修复后**的完整批跑（填充率门
`minAreaRect` 修正、条件 runner 进程树清理/引擎等待均已合入）：

- **PASS 2 / FAIL 10 / UNKNOWN 0**；
- 12 格环境读回全部有效（`applied_tod` 与请求一致），失败均为真实
  感知失败，不是环境未应用。

| 格 | 结果 | ticks | true | modes | cert_both | 环境读回 |
| --- | --- | --- | --- | --- | --- | --- |
| dawn_clear | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 06:29:59 |
| dawn_rain | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 06:29:59 |
| dawn_fog | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 06:29:59 |
| **noon_clear** | **PASS** | 12 | **12** | `pair` 12 | **1.0** | 12:00:00 |
| noon_rain | FAIL | 12 | 0 | `none` 12 | 0.0 | 12:00:00 |
| **noon_fog** | **PASS** | 12 | **12** | `pair` 12 | **1.0** | 12:00:00 |
| dusk_clear | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 18:30:00 |
| dusk_rain | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 18:30:00 |
| dusk_fog | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 18:30:00 |
| night_clear | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 21:30:00 |
| night_rain | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 21:30:00 |
| night_fog | FAIL | 12 | 0 | `no_markings` 12 | 0.0 | 21:30:00 |

聚合：`logs/m5_seg/proving_conditions/conditions_table.json`
（`counts {'FAIL': 10, 'PASS': 2}`）。

## 判读

1. **感知链在正午清晰/雾天可用**：`noon_clear` 与 `noon_fog` 都是
   12/12 双侧 `pair`、GT 左右线全命中、cert_both=1.0。说明分割→候选→
   配对→真值投影这条链本身在受控场地是通的；雾天的漫射光没有破坏
   橙色标线与浅色地面的对比。
2. **低照度是硬边界**：dawn/dusk/night × 三种天气共 9 格全部
   `no_markings` 12/12。黎明帧（`_dawn_clear_frame5.png`）显示贴花线
   呈暗红色、与地面/阴影对比极低；候选层之前的学习掩码就没有线。
   这不是配对门/宽度门能修的，属于**输入域/曝光问题**。
3. **正午雨天是另一种失败**：`noon_rain` 的 `modes` 是 `none` 而不是
   `no_markings`，逐帧只有 1 条 `thin:align` 丢弃、`cands=[]`、
   cert 0.0。画面（`_noon_rain_frame5.png`）整体被雨水预设冲淡成米色，
   标线仍可见但对模型不再构成可提取的线。雨天需要专门的训练数据，
   不是把阈值放宽能解决的。
4. **修复没有把失败格"修成"通过**：门限、配置、模型 pin 全部保持
   预注册值；低照度/雨天格没有重跑换参数。

## 与前一版（中断批）的关系

前一版在试车场驾驶修复前运行，且 noon 格因探针挂死记 UNKNOWN。
本版在以下修复之后重跑全部 12 格：

- `too_sparse` 填充率门改用最小外接旋转矩形（`e1a82f2`）；
- 条件 runner 单格超时杀进程树 + 每格启动前等待引擎退出
  （`05b6448`、后续 taskkill 解码修复）；
- 试车场驾驶修复（`44d0451` 等）不改变静止采集的感知配置。

前一版证据保留在 `logs/m5_seg/proving_conditions_pre_fix_20261105/`。

## 结论与下一步

- 结论：**当前模型在试车场只在正午（晴/雾）可用；低照度与雨天不可用。**
  这是数据覆盖问题，与配对逻辑无关。
- 下一步（按 ROADMAP Phase 1/2）：
  1. 在 dawn/dusk/night/rain 条件下采集带 `engine_verified` 真值的
     训练帧（同一 decal 场景可程序化扩量），扩池后重跑本 12 格；
  2. 若扩池后低照度仍为 0，再考虑输入域归一（曝光/白平衡）或
     低照度增广，而不是改配对门；
  3. 正午 clear/fog 两格可作为后续驾驶条件实验的基准格。

## 产物

- `logs/m5_seg/proving_conditions/gridmap_<tod>_<weather>.{json,jsonl,run.log,frames/}`
- `logs/m5_seg/proving_conditions/conditions_table.json`
- `logs/m5_seg/proving_conditions_{clear2,clear3,rain,fog}.log`
- 证据帧：`logs/m5_seg/_dawn_clear_frame5.png`、
  `logs/m5_seg/_noon_rain_frame5.png`
