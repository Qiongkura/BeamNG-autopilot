# 当前项目状态复查（2026-10-10，E5 新资格入口之后）

## 结论

项目已完成论文证据链的主要收口：20 条参考文献逐条核验、E1/E2/E3-lite、E4 第二地图负结果、
7 张满栏正文图、投稿源包、数字审计 115 项、图 QA 0 条、LaTeX 静态检查 0 问题。
本次把 E5 的**新资格入口**跑完：

- `e5_pose_sweep_qualified_20261009`：9 个请求位姿，1 个 `completed`、8 个 `unplaceable`；
- `e5_pose_sweep_remaining_20261010`：原先剩余 6 个请求位姿，6/6 `unplaceable`；
- 所有请求都有 `request/result/session` 证据；端口轮换、等待、退避和 owned cleanup 生效；
- 无游戏残留（最终 `game_pids() = set()`）；
- 结论是**放置/感知车道可用率失败**，不是“固定真实位姿下 8 个偏差点都完成了等曝光驾驶”。

## E5 当前可用证据

| 批次 | 结果 | 解释 |
|---|---|---|
| `e5_pose_sweep_qualified_20261009` | 1/9 completed、8/9 unplaceable | 唯一 completed 的名义基线：46 帧、15 帧有路径、26 帧 sensor lane、行进 0.56 m；实际生产放置偏离请求约 1.05 m，因此不是 `fixed_actual_pose` |
| `e5_pose_sweep_remaining_20261010` | 6/6 unplaceable | 全部 `placement_rc=3`，没有等曝光驾驶测量；使用端口等待/轮换/退避的新入口 |

旧的 `e5_pose_sweep_20261007` 九份原始输出继续保留，但不计为合格矩阵。

## 已修复的工具问题

1. `fsd_drive` 的整轮墙钟上限已做进驱动内部（`--max-wall-s`），放置超时走 `return 3` 与正常 `finally`；
   不再使用外部 `timeout` 杀驱动，避免孤儿 BeamNG 实例、CEF/端口污染。
2. 新 `pose_sweep.py` 入口：
   - 逐位姿记录 request/session/result；
   - 配置、源码、checkpoint、时长、退出码、实际放置位姿、控制序列和 owned cleanup 都资格化；
   - 端口等待、端口轮换、失败退避；
   - `actual_port` 写入证据。
3. `allow_unplaced` 不能绕过已过期的放置墙钟；新增回归测试。

## 当前验证

- 全量 pytest：通过（清理残留游戏后重跑）；
- `test_pose_probe_qualification.py` + `test_fsd_wall_cap.py`：通过；
- 数字审计：115 项 OK；
- 图内文字 QA：0 条重叠；
- LaTeX 静态检查：0 问题（本机没有 TeX 引擎，未声称实际编译通过）；
- 四份 PDF：正文英文/中文、补充英文/中文均重新生成；
- 证据归档：531 文件 + SHA-256 清单（`logs/paper_release/`）。

## 仍未完成

- E5 更大偏移、第二路段/第二地图的扩展；第二地图 E4 已证明当前引擎逐地图 line-class 标注通道是阻断点；
- 公开归档（Zenodo 或上传证据 zip）；当前论文如实写“代码/稿件源公开，记录产物公开归档尚不可用”；
- ORCID、目标期刊、中文版是否投稿、邮箱是否上首页；
- 其它未提交工作树改动需由后续专轮统一收口，不能在本快照中覆盖或还原。
