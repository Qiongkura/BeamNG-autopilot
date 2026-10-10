# E5 扩展结果（2026-10-11）

## 1. Italy 更大位姿偏移

入口：`scripts/m5_pose_sweep.ps1` → `scripts/m5_pose_sweep.py`，资格化结果目录：
`logs/experiments/e5_pose_sweep_large_20261011/`。

请求矩阵：横向偏移 `{+1.5,+2.0}` m × 航向偏移 `{0,+10}`°；其它配置固定为
Italy、strict + sensor、`seg_model_hand`、25 s、6 m/s、墙钟上限 150 s、`production_alignment`。

| pose | status | qualified | 原因 |
|---|---|---:|---|
| d=+1.5 m, Δψ=0° | invalid | 否 | telemetry exposure too short |
| d=+1.5 m, Δψ=+10° | completed | 是 | — |
| d=+2.0 m, Δψ=0° | invalid | 否 | telemetry exposure too short |
| d=+2.0 m, Δψ=+10° | completed | 是 | — |

**解释边界**：两个 `completed` 结果通过资格入口的配置/退出/清理检查；两个 `invalid` 结果保留，
不能当作驾驶成功或失败。这个批次没有产生足够的固定实际位姿等曝光对照，因此不支持把「更大偏移」
的路径可用率与基线做因果比较。

## 2. Mountain 第二路段

入口同上；结果目录：`logs/experiments/e5_pose_sweep_mountain_20261011/`。
请求矩阵：anchor `(729.6,763.9,45°)`、goal `(616.2,894.5)`，横向 `{0,+0.5}` m × 航向 `{0,+5}`°，
其它配置与 Italy 扩展相同。

| pose 数量 | completed | qualified | unplaceable | 启动失败 |
|---:|---:|---:|---:|---:|
| 4 | 0 | 0 | 4 | 0 |

四个位姿均为 `placement_rc=3` / `placement deadline; no equal-exposure drive measurement`，没有进入等曝光
驾驶测量；这只能报告为「第二路段未获得放置」，不能读成第二路段的驾驶质量或安全结论。

## 3. 清理与复现

* 两批使用资格化 runner：request/session/result、实际放置位姿、退出码、资格原因与 owned cleanup 均保留；
* 所有 UNKNOWN 与失败原因原样保留；
* 收尾后 `game_pids() = set()`；
* 两批都不改模型、不放宽 strict、不倒车、不消费封存最终集、不新增人工标注。

## 4. 结论

E5 扩展已经**尝试完成**，但没有产生第二路段的合格驾驶证据：Italy 更大偏移批次有 2 个 qualified、2 个
exposure-invalid；Mountain 4/4 unplaceable。论文因此只写「扩展边界与失败原因」，不把这些结果包装成泛化或安全提升。
