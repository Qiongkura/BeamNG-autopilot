# T16 live 帧回放结果：east=掩码缺失（H-M0），west=mask→marking（H-M1）（2026-11-04）

## 执行

- 采集：`--save-frames` 保存 live 536×403 帧 + pos/heading + CameraModel；
  east a0s0（余量 1.0）19 帧、west a3s0（余量 0.6）24 帧。
- 回放：`scripts/m5_marking_replay.py` 逐帧
  `seg.predict → detect_lines → pair_lane_markings`，按 H-M0..M3/ok 分类。

## 结果

| 场景 | 帧数 | ok | H-M0（掩码缺） | H-M1（无 marking） | H-M2（collect 拒） | H-M3（配对层） |
| --- | --- | --- | --- | --- | --- | --- |
| east | 19 | 1（5.3%） | **18（94.7%）** | 0 | 0 | 0 |
| west | 24 | 14（58.3%） | 0 | **10（41.7%）** | 0 | 0 |

## 关键验证：H-M0 不是投影误差

对 east 前 4 帧，把 GT 右线/左线世界点投影到图像，测其到 line 掩码的最近距离：

| 帧 | 左线（白，GT） | 右线（黄，GT） |
| --- | --- | --- |
| 00001–00004 | **2.7–6.8 px** | **16–50 px** |

左线投影落在掩码上（2.7–6.8px），右线投影离掩码 16–50px——**掩码确实
没有右侧黄线**，H-M0 成立，不是标定/投影假象。

## 判读

1. **east（难场景）**：损失在**语义掩码侧**——live 536×403 下模型不检出
   右侧黄线。而池域同场景（144×192 输入、内部上采样到 536）raw 黄召回
   0.94。差异指向**输入域**：训练/池域是"144 原生 → INTER_AREA 放大到
   536"，live 是"原生 536 直接送入"；模型没见过原生 536 的锐利黄线。
2. **west（控制）**：掩码有黄线，但 41.7% 帧没有形成 marking（H-M1）——
   这是 mask→marking 提取层的问题，与 east 不同层。
3. 之前候选层三臂（黄线阈值/span/CLAHE）在 live 上无效，原因也清楚了：
   east 的掩码本身没有黄线，候选层再调也拿不到；west 的 marking 提取
   才是它的瓶颈。
4. 与 post-norm 结果一致：后处理尺度能救"掩码里有但被删"的情况（离线
   0.103→0.688），但救不了"掩码里没有"（east live）。

## 下一步（需新预注册）

**输入域匹配（D 臂）**：live 536×403 先按 **INTER_AREA 缩到 144×192** 再
送入模型（模型内部再放大到 536，等价训练域）。离线用已保存的 19 帧验证：
- 判据 P-In1：east 的 H-M0 比例从 94.7% 降到 ≤0.3；
- 判据 P-In2：east 的 ok 比例 ≥0.5；
- 控制：west ok 比例不下降 >10pp；
- 时延：预计与 native 接近（多一次 536→144 缩小）。
若成立，再讨论 live 接线（仍默认 None，需预注册）。

## 产物

`scripts/m5_marking_replay.py`、`logs/m5_seg/marking_replay_east_result.json`、
`marking_replay_west_result.json`、两套 `.frames/` 与探针 JSON。
