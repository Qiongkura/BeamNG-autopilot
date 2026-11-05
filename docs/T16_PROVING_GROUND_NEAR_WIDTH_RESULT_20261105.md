# T16 试车场宽度修复：近场口径让 12/12 真值达标，放置与车道中心稳定（2026-11-05）

## 症状

试车场（GridMap 纯地面 + decal 双线，`BEAMNG_CV_WHITE=0`）live 探针：
配对模式 12/12、cert_both 0.833，但 `width_m=4.88`（GT 3.96）→
true 0 / ambiguous 11。驾驶放置虽能过，但车道宽度/中心不可信。

## 分层定位（同一帧，离线可复现）

1. **候选合并把地面接缝并入左线**：`merge_close_candidates` 默认半径
   1.0m；画面底部白色混凝土接缝（0.3m 短碎段，横向 +0.95m）与左线
   （+1.8m）中位差 0.85m < 1.0m → 被并入，左线变成 178 点折线，近场
   世界 x 从 500.2 漂到 499.2（GT 499.3）。关掉合并（`BEAMNG_LINE_MERGE=0`）
   后左线恢复 141 点、x 498.7–499.5。
2. **世界宽度取全线中位数**：`_pair_world_geometry` 返回的宽度被
   `median(dw[vg])` 覆盖，包含 40m 远的低分辨率尾段。实测同一对线：

| 车体系 s | 宽度中位 |
| --- | --- |
| 0–5m | 3.79m |
| 5–10m | 3.84m |
| 10–15m | 4.73m |
| 15–25m | 4.59m |
| 25–45m | **5.99m** |
| 全线 | **4.95m** |

远场几像素回投误差被放大到数米；代码自己的策略是"宽度/中心在车旁读"
（`_best_vision_pair` docstring），世界宽度覆盖违背了这条。

## 修复（研究开关，默认不变）

- `lane/constants.py`：`LANE_PAIR_WIDTH_NEAR_M = 10.0`；
- `lane/pairing.py`：`BEAMNG_PAIR_WIDTH_NEAR=1` 时世界宽度只取近场
  站点；若 10m 内不足 3 站（粗站点步长），退化为**最近的 3 个有效站**，
  绝不回到全线中位数；debug 记 `pair_width_near_m/n`；
- 默认关闭，生产路径逐字节不变；单测 `tests/test_pair_width_near.py`
  钉住"近场平行、远场外漂"线对的开/关差异与粗步长回退。

## 复测（live）

配置：`proving_lane`（无铺装）、材质 `line_yellow_double`、
`BEAMNG_CV_WHITE=0 BEAMNG_LINE_MERGE=0 BEAMNG_PAIR_WIDTH_NEAR=1`。

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 配对模式 | pair 12/12 | **pair 12/12** |
| true / ambiguous | 2 / 10 | **12 / 0** |
| cert 左/右/双边 | 1.0 / 1.0 / 1.0 | **1.0 / 1.0 / 1.0** |
| `width_m` | 4.56–4.76 | **3.765–3.95（近场 3 站）** |

严格+sensor 放置复测（纯地面，无铺装）：**placed=True**；车道偏差
`painted line lateral mean=+0.09m p50=+0.10m min=−0.37 max=+0.13`
（0 帧在左线左侧），比铺装版的 mean −0.27/min −4.85 稳定得多。

## 判定

- 试车场真值链**宽度/中心/配对三项全部与 GT 对齐**，可作为受控参考；
- 该配置是**试车场诊断配置**（三个开关均非生产默认），任何条件实验
  必须先预注册并记录这套开关；
- 试车场仍**不能开**：34/38 帧安全停车（空 GridMap 无路线/可行驶证据），
  与车无关（直接控制 0→10 m/s 已验证）。条件矩阵仍不启动。

## 下一步

1. 给试车场可行驶参考（GridMap AI 城市道路：`zone_AI_city/ai_roads`
   有 260m 直路、宽 7.6m，坐标见下）→ 复测能开；
   - 候选直路：`y=-612`，x 从 180 到 -80，dir=(-1,0)，z=100；
   - `y=-788`，x 从 -80 到 180，dir=(+1,0)。
2. 或在试车场实现直线导航路线（不走地图 roadnet）。
3. 预注册"静止放置+配对"条件实验（TOD×天气，不依赖驾驶）。

## 产物

`logs/m5_seg/gridmap_nearw2_r0.{json,jsonl,frames/}`（探针）、
`logs/m5_seg/gridmap_plain_drive.{log,jsonl}`（放置）、
`logs/m5_e2e/shadow_fsd_1793854502_20261105_125529.npz`（38 帧）。
