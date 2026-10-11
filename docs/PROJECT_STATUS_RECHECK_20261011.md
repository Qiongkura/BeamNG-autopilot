# 当前项目状态复查（2026-10-11，T18 与 E5 扩展收口）

## 已完成

- 论文三轮审查修正、20 条参考文献题录核验、证据归档（531 文件 + SHA-256）。
- T17 感知节拍/几何诊断：代码回归与两臂 Tech 诊断完成；deadline、独立真值、ACK 等限制保留。
- T18 传感器回放正确性：首轮无效保留、同协议重试合格、48 次同源输出一致、CPU p95 对比完成；定向/全量测试通过。
- E5 基线 3×3 资格入口：9/9 尝试，1 completed、8 unplaceable；生产放置偏差与 exposure qualification 均记录。
- E5 扩展：先前 `production_alignment` 批次 2 qualified/2 exposure-invalid；随后用 `fixed_actual_pose` 契约重跑，Italy 更大偏移 4/4 在曝光前因 2.57–3.08m 漂移被拒（rc=4），mountain 第二路段 4/4 放置截止（rc=3）；结果见 `docs/T16_E5_EXTENSION_RESULT_20261011.md`。
- 收尾后游戏进程为 0；当前纸面数字审计 115 项通过、图 QA 0、LaTeX 静态 0。

## 当前不能声称

- E5 扩展没有产生第二路段合格驾驶证据；unplaceable 不是驾驶失败/安全失败。
- 大偏移批次的两个 qualified 仍不能与基线做等曝光因果比较（另两项 exposure-invalid）。
- T18 的 CPU p95 改善不是驾驶安全、执行 ACK 或模型晋级证据。
- 第二地图 E4 仍被逐地图 `line-class` 真值通道阻断。

## 尚待用户决定/专轮

- 公开上传本地证据归档到 Zenodo（当前只生成 `logs/paper_release/` 本地 zip）。
- ORCID、目标期刊、中文版是否投稿、邮箱是否放首页。
- 更大偏移/第二路段已用固定实际位姿契约阻止漂移后曝光：大偏移 4/4 rc=4、mountain 4/4 rc=3。随后先做 +5m production_alignment 反向标定，再用实测 actual_pose 作为 fixed_actual_pose anchor，获得 1/1 qualified 等曝光偏移观察（位置误差 0.0209m、航向 0.087°、26.203s、6.284m、22 个有路径帧）。它证明契约可实现和偏移点可测，但不是相对 baseline 的严格因果 A/B。
