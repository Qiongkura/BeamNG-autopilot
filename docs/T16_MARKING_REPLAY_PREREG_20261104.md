# T16 live 帧回放 + mask→marking 诊断 · 预注册（2026-11-04）

## 1. 背景

post-norm live 复测：右线候选召回 0.00→0.15（L1 失败），离线 post 修复
（0.103→0.688）不迁移；collect 丢弃结构从 `thin:span` 变为出现
`solid:side/align`。剩余丢失点被夹在 **语义掩码 → marking 提取** 或
**live 536 原生域** 之间，需要 live 帧回放才能钉死。

## 2. 采集（探针新增 `--save-frames`）

- `m5_pairing_probe.py --save-frames`：把每 tick 的以下内容写入
  `<out>.frames/`（npz，每 tick 一个）：
  - `frame`：live 536×403 RGB 原始帧；
  - `pos`（3,）、`heading`；
  - `cam`：当帧相机模型（object，npz allow_pickle）；
  - `t`：放置阶段相对时间。
- 同时保留现有 pairing JSONL（配对决策/拒绝计数），**帧与 tick 一一对应**。
- 采集场景：east a0s0（难场景 m1.0）+ west a3s0（控制 m0.6），各 1 轮 20 tick；
  post-norm 关（现行行为），模型 pin `seg_model_v28_sslA`。

## 3. 离线回放与判定

对每个保存帧离线执行：

1. `seg.predict(frame)` → road/line 掩码；
2. `seg.detect_lines(frame, cam, pos, heading, line_mask=line, road_mask=road,
   debug=dbg)` → markings 列表；
3. `pair_lane_markings(markings, pos, heading, debug=dbg2)` → 配对与
   `collect_drops`/`collect_stats`。

**GT**：认证场景的右线（黄线）在当帧 ego 体系里的横向位置（场景构建记录
的真实节点）。对每个 marking 计算其近场横向与 GT 的差。

**分类（每帧只归一类，互斥）**：
- **H-M0（掩码侧丢失）**：line 掩码在 GT 黄线附近（±0.6m 投影带）没有像素；
- **H-M1（mask→marking 丢失）**：掩码有像素，但没有生成任何一侧 marking
  落在 GT 黄线 ±0.6m 内；
- **H-M2（collect 门拒绝）**：marking 命中 GT，但被 collect 以
  `side`/`align`/`span`/`kind` 拒绝（读 collect_drops/stats 的 key 与横向）；
- **H-M3（配对层）**：marking 成为候选，但配对未选它（拒绝计数/模式）；
- **ok**：黄线成为候选并被配对使用。

## 4. 判据（跑前冻结）

- **D-M**：统计 east 20 tick 中缺失帧的归属；若某一类 ≥2/3，则该层即修复
  对象：
  - H-M0/H-M1 → 掩码→marking 提取器（新预注册，可能涉及检测阈值/连通域）；
  - H-M2 → collect 门（side/align/span 的按 kind 阈值，研究开关）；
  - H-M3 → 配对层（已有 F1/F2 资产可复用）。
- **对照**：west 同类统计；若 west 的 ok 比例 ≥0.8，则 east 的失败是站点
  相关，报告差异原因（投影/分类/掩码）。
- 不允许把 H-M0 与 H-M1 合并掩盖；每帧只归一类并保留原始证据。

## 5. 纪束与交付

- 采集与回放均只读；不训练、不改默认门、不新增标注、不消费最终集；
- 交付：本预注册、`--save-frames` 实现 + 单测、回放工具
  `m5_marking_replay.py` + 单测、`T16_MARKING_REPLAY_RESULT_*.md`、
  npz/JSON 证据 zip。
