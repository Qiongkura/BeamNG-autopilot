# T16 post-norm live 复测：L1/L2 失败，L3/L4 通过（2026-11-04）

## 接线

`--seg-post-norm-width` 已透传 drive/FSDStack 的三个 Segmenter 构造
（默认 None = 现行行为）；探针 `--post-norm-width` 记录并在报告中留痕。
判据在读取结果前冻结（预注册补充 A2：L1–L4）。

## 结果（east 难场景 m1.0，开/关 × 2 重复 × 20 tick）

| 臂 | 轮 | 右召回 | 左召回 | 配对 | true | phantom | true_rate | 模式 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | r0 | 0.00 | 0.85 | 18 | 15 | 1 | 0.83 | mirror_left 16 |
| off | r1 | 0.00 | 0.00 | 20 | 1 | 5 | 0.05 | mirror_left 16 |
| **on** | r0 | **0.16** | 0.05 | 18 | 5 | 3 | 0.28 | mirror_left 15, pair 2 |
| **on** | r1 | **0.15** | 0.10 | 18 | 3 | 8 | 0.17 | mirror_left 16, pair 1 |

## 判定（L1–L4）

- **L1 失败**：右线候选召回 0.15–0.16，目标 ≥0.6（有提升但远不够）。
- **L2 失败**：true_rate on 0.28/0.17 vs off 0.83/0.05（off 两轮本身双稳态，
  均值 0.44 vs 0.22）；phantom 上升。
- **L3 通过（fail-closed 审计）**：4 轮均无异常/崩溃；未放置时保持刹车到
  墙钟（rc=3）；开关只影响后处理尺度，安全门语义未变。
- **L4 通过**：离线微基准 p50 **18.9 → 8.0 ms（−10.9ms）**——后处理在 192
  上比 536 更便宜；多一次 resize 的成本被形态学/连通域的节省覆盖。

## 判读

1. **离线修复真实、live 不迁移**：post 黄线召回 0.103→0.688（离线）只换来
   live 右候选 0.00→0.15。说明 live 的丢失点还在**mask→marking 提取**或
   **live 536 原生域**，不是后处理分辨率单独能解决的。
2. **collect 丢弃已变化**：on 臂 `thin:span` 2/6→1/2、出现 `solid:side/align`
   ——黄线 marking 可能被归类为 solid 后在 side/align 被拒；也可能根本没
   形成 marking。
3. **双稳态再次出现**：off r0/r1 = 0.83/0.05，单轮不可信；L2 的比较被方差
   污染，不能据此断言"开启更差"，但也不能声称有效。
4. **开关不接部署**：默认 None 保持不变；代码保留（离线修复与诊断资产）。

## 下一步（需新预注册）

**live 帧回放 + mask→marking 诊断**：给探针加 `--save-frames`（保存
536×403 live 帧与当帧的安全/放置元数据），离线回放 `detect_lines`，统计
黄线 marking 是否生成、kind/span/align/side 与投影横向，定位 masking→
marking 的具体拒绝点；west 作不回归控制。仍不训练、不改默认门。

## 产物

`logs/m5_seg/post_norm_live_table.json`、`pairing_live_pn_{off,on}_r{0,1}.json/jsonl`、
`logs/m5_seg/post_norm_latency.json`。
