# 评审：无标线场景评价这轮开发（2026-09-26）

**评审对象**：astra 的未提交改动（`beamng_autopilot/experiments/negative_scenes.py`、
`tests/test_negative_scenes.py`、`scripts/m5_seg_eval_matrix.py` 的设备修复、
`scripts/dev_validate.ps1` 的编码修复、对 `docs/W1_VERIFIED_TRUTH_136_20260926.md`
的四处更正、`docs/NEGATIVE_SCENE_EVAL_DEVELOPMENT_20260926.md`）。

**独立复核结论**：**这轮开发可以采纳**，但先处理 P1 两条（结论口径）。

| 复核项 | 我的独立验证 | 结果 |
| --- | --- | --- |
| 门禁 2829 通过 | 我自己跑 `scripts/dev_validate.ps1`（`gate_20260926_181117.log`） | **2829 passed / ALL PASS** ✓ |
| 58 帧负例数字 | 独立复算（`verified_truth_eval_136.json`） | 与他们的表**逐项吻合** ✓ |
| 设备参数不生效 | 读 `Segmenter.__init__`（接受 `device`，`self.device` 存在） | **真 bug，修复正确** ✓ |
| 编码修复 | 读 diff（`PYTHONUTF8`/`PYTHONIOENCODING` 保存-设置-恢复） | 合理 ✓ |
| 对我报告的四处更正 | 逐条核对 | **四处都准确** ✓（"CPU 推理"确实不成立、`best.pt`≠`checkpoint_last.pt`、8 个是**采集组**、砾石场景只有候选画线） |
| 未替换生产模型/未动运行产物 | `git status` + 权重哈希 | 未替换 ✓ |

---

## P1（改变结论口径，建议先处理）

### P1-1 评价集混入了**训练用过的组**（24/136 帧）

`italy/ring_20260923_135342`（town）既是 E0/agentline 与验收轮（`soak4h`）的训练组，
又出现在这 136 帧评价集里 → 这 24 帧对候选是 **in-sample**。他们已在我报告里加了提示，
但没有量化影响。我算好了（`verified_truth_eval_leakfree.json`）：

| 模型 | 全集 136 帧 line_iou | **去训练组 112 帧** | 全集 recall | **去训练组 recall** |
| --- | --- | --- | --- | --- |
| 生产模型 | 0.2868 | **0.2437** | 0.3431 | **0.2968** |
| 研究候选（seed43/last） | 0.3741 | **0.3424** | 0.5175 | **0.5404** |

**含义**：town 组对两个模型都更容易 → 混入会**同时抬高**两者的 line_iou（生产 +0.043、
候选 +0.032）。**建议**：报告与后续引用一律用**去泄漏的 112 帧口径**（或明确标注哪 24 帧
是 in-sample 并单独列）；候选的 recall 不受影响（去掉后反而更高 0.540）。

### P1-2 生产模型的"土路零误报"很可能是**它训练过土路**

从 checkpoint 的 `train_args` 读出：生产模型训练数据含 `logs/m5_seg/dirt_road_labeled`
（土路手涂素材），共 **1883 帧 / 60 epoch**；候选只在 **20 帧城镇**上训练。所以

* "生产模型在 58 帧负例上零误报"**成立**（实测），但**解释**要改成"**见过土路材质**的
  模型不在这里画线"，而不是"生产模型感知更好"；
* `dirt_road_labeled` **无地图身份、无位姿**，无法核对它是否与新采土路（`123638`）
  同路段 → "是否同路段"是**未知**，不能当已排除。

**建议**：在负例报告里写明两者训练数据差异（这是**非受控对照**），并把它列为
"下一轮要做的受控实验"：同一训练预算下，含土路素材 vs 不含，看负例误报是否变化。

## P2（口径与可维护性）

### P2-1 新指标没进协议哈希

`negative_line_*` 在 `experiments/protocol.py` 里**没有定义**（`grep` 为 0）。
诊断指标不进门可以接受，但方案 §10.2 要求指标定义（分母、有效区域、UNKNOWN 规则）
与协议一起版本化；将来若要做门（哪怕软门）必须先标定再升协议。
**建议**：要么加进 `METRIC_DEFINITIONS` 并升协议版本（v4），要么在协议文档里显式写
"诊断指标，不进哈希，不得用于晋级"。

### P2-2 指标的资格前提没有在数据流里强制

`negative_line_counts` 依赖"标注可信"，但评估矩阵对**任何**目录都会算它——engine 档
（rank `unreliable`）的"标签里没有线"**不等于**"真的没有线"（方案 §6.2 的原文：
"0 line px is NOT evidence of no paint"）。在那种数据上会产出**假的 clean negative**。
**建议**：`negative_line` 带上 `label_rank`（或 `requires_verified`），并在非 verified 数据上
把 `status` 标成不可用于结论（例如 `unverified_labels`），避免以后被误读成"零误报"。

## P3（表达与接线）

* **P3-1** `false_positive_pixel_fraction` 的分母是**整帧像素**（`eligible_px = lab.size`），
  与 `offroad_false_frac_of_pred`（分母=预测标线像素）**不可比**；两个数字并列时要写明，
  避免混读。
* **P3-2** "一个噪声像素即算假线帧"已在报告披露 ✓；建议在字段名或文档里也标出
  （`false_positive_frames` 是**严格**口径）。
* **P3-3** 新统计目前只在评估矩阵 JSON 里；`_pixel_eval` 会算出来但判定文件的
  `hard_gate` 与看板都不带（他们已披露）。建议下一步把它写进判定文件与看板"逐场景"表，
  并明确标"诊断，不进门"。

## 其他

* 我已清掉自己遗留的三个 `_patch_*.py` 临时文件（评审前工作区里还有）。
* 他们没有提交/推送、没有替换生产模型、没有动 `logs/` 运行产物 ✓ 与声明一致。
* 提交建议：可以提交；但**先**把 P1-1（112 帧口径）与 P2-1（协议标注）落进文档，
  再按模块拆提交（`experiments` 库 + 其测试 / `scripts` 设备与编码修复 / `docs`）。
