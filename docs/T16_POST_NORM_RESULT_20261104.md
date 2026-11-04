# T16 后处理分辨率归结果：P2（宽度 192）通过 R1/R2/R3（2026-11-04）

## 执行

预注册 `T16_POST_NORM_PREREG_20261104.md` + 修订案 A1。离线只读：池帧
east 9 帧 / west 3 帧，模型 pin `seg_model_v28_sslA`；
`Segmenter(post_norm_width=W)` 把帧与掩码缩到宽度 W 跑原后处理，再放大回
输入分辨率；对比 cubic/lanczos 预上采样（模拟 live 536 输入）与 native。

## 结果

### P1（W=144，预注册原臂）——R1 过、R2 失败

| 场景 | 条件 | post 黄召回 | 判定 |
| --- | --- | --- | --- |
| east | native | 0.606 | —（native 被 144 缩小；无归一时 0.727） |
| east | cubic | **0.629** | R1（≥0.6）通过 |
| west | cubic | **0.679** | **R2（≥0.84）失败**（无归一时 0.843） |

原因：池帧原生宽是 **192**；归一 144 把 native 自身也缩到 144，west 正例
被拖低。P1 失败如实保留，不删除。

### P2（W=192 = 原生/训练域宽，修订案 A1）——**全部通过**

| 场景 | 条件 | post 黄召回 | 相对无归一 |
| --- | --- | --- | --- |
| east | native | 0.727 | 不变（0.727） |
| east | cubic | **0.688** | **0.103 → 0.688（+58.5pp）** |
| east | lanczos | 0.690 | 0.103 → 0.690 |
| west | native | 0.889 | 不变（0.889） |
| west | cubic | **0.863** | 0.843 → 0.863 |
| west | lanczos | 0.864 | 0.842 → 0.864 |

## 判定（预注册 R1–R3）

- **R1 通过**：east cubic post 黄召回 0.688（目标 ≥0.6；无归一 0.103）。
- **R2 通过**：west cubic 0.863（目标 ≥0.84；无归一 0.843）。
- **R3 通过**：raw 黄召回与无归一完全一致（0.973/0.993），归一不改变模型输出。
- **结论**：**后处理分辨率归（W=192）是候选层修复链里第一个成立的确定性
  修复**——把 536 输入下的黄线存活从 10% 拉回 69%，同时 native 与控制场景
  零回归。

## 机制复核

后处理（形态学/路面约束/外观门）的核大小与跨度为像素固定值；把输入缩到
训练域宽度（192）再跑，等价于让后处理在"它被调参时的尺度"上工作。west
仍有 0.889→0.863 的残余差（−2.6pp），是预上采样锐度带来的轻微域差异，
在容许范围内。

## 下一步（需新预注册）

1. **live 接线**：给 `m5_fsd_drive`/FSDStack 增加 `--seg-post-norm-width`
   （默认 None = 现行行为），在 east 站 live 放置探针复测右线候选召回与
   true_rate；需同时报告单帧时延变化（多一次 resize，预计 <1ms）与安全
   审计（fail-closed 行为不变）。
2. 若 live 复测通过，再把该开关纳入部署候选；仍不训练、不晋级模型。

## 产物

`logs/m5_seg/post_norm_diag_{east,west}.json`（W=144）、
`post_norm_diag_{east,west}_w192.json`（W=192）；实现见
`beamng_autopilot/vision/segmentation.py`（`post_norm_width`）与
`scripts/m5_mask_diag.py --post-norm-width`。
