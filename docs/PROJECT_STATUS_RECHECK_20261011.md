# 当前项目状态复查（2026-10-11，T18 与 E5 扩展收口）

## 已完成

- 论文三轮审查修正、20 条参考文献题录核验、证据归档（531 文件 + SHA-256）。
- T17 感知节拍/几何诊断：代码回归与两臂 Tech 诊断完成；deadline、独立真值、ACK 等限制保留。
- T18 传感器回放正确性：首轮无效保留、同协议重试合格、48 次同源输出一致、CPU p95 对比完成；定向/全量测试通过。
- E5 基线 3×3 资格入口：9/9 尝试，1 completed、8 unplaceable；生产放置偏差与 exposure qualification 均记录。
- E5 扩展：Italy 更大偏移 4/4 尝试（2 qualified、2 exposure-invalid）；mountain 第二路段 4/4 unplaceable；结果见 `docs/T16_E5_EXTENSION_RESULT_20261011.md`。
- 收尾后游戏进程为 0；当前纸面数字审计 115 项通过、图 QA 0、LaTeX 静态 0。

## 当前不能声称

- E5 扩展没有产生第二路段合格驾驶证据；unplaceable 不是驾驶失败/安全失败。
- 大偏移批次的两个 qualified 仍不能与基线做等曝光因果比较（另两项 exposure-invalid）。
- T18 的 CPU p95 改善不是驾驶安全、执行 ACK 或模型晋级证据。
- 第二地图 E4 仍被逐地图 `line-class` 真值通道阻断。

## 尚待用户决定/专轮

- 公开上传本地证据归档到 Zenodo（当前只生成 `logs/paper_release/` 本地 zip）。
- ORCID、目标期刊、中文版是否投稿、邮箱是否放首页。
- 更大偏移/第二路段若要形成等曝光驾驶因果证据，需要先修复生产放置契约与 Tech 启动/端口流程；当前结果只作为边界与失败原因记录。
