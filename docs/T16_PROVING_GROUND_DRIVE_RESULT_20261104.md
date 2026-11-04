# T16 试车场接 drive 首测：场景可加载，严格放置仍失败（2026-11-04）

## 执行

`m5_fsd_drive --scenario-json logs/m5_seg/gridmap_scene.json --map gridmap_v2`
（材质覆盖 `line_yellow_double`、`AsphaltRoad_variation_01`、
`m_dirt_road_gravels`；`BEAMNG_THIN_SPAN=relax`），无 teleport/goal，
`--max-wall-s 90/150`，两轮。

## 结果

- **场景加载成功**：`[fsd-drive] scenario-json loaded: gridmap_scene
  profile=line_yellow_double/line_yellow_double`；
  实际位姿 (501.10, 500.00, 100.14)、heading 1.5708——正是车道中心，
  说明 Scenario ego 正确生成并被驱动会话接管。
- **严格+sensor 放置失败**：`placed=False, placement_rc=3,
  termination=placement_failure`；放置诊断 `marks=3, line_px≈6440,
  pair_mode=none`（探针在 S1 下最好也只到 `mirror_left`）。
- 放置确认需要**双侧（paired）**帧；镜像单侧不被接受。

## 判定

1. `--scenario-json` 接线完成且可用（场景/材质/自车位姿全部正确）；
2. **试车场仍不可放置**：右线碎段（0.92–1.09m）导致只有单侧候选，
   镜像配对不满足严格放置确认；
3. 因此**条件矩阵仍不能在试车场启动**——先修右线候选（见下）。

## 下一步（Phase 2 内，工程项）

二选一（均无需外部数据）：
- **取景/几何调整**：把 ego 稍微偏离车道中心（如 -0.3m）或调整 decal 车道
  位置，让右线远离画面右下角裁切区，增加可检跨度；
- **候选共线碎段合并**：在 `_collect_candidates` 对同侧、横向差 ≤0.15m、
  纵向间隙 ≤2m 的 thin 碎段做合并后重算 span（S1 预注册里提到但未实现）。

修好后用同一 `--scenario-json` 入口复测 placed；通过后再启条件矩阵。

## 产物

`logs/m5_seg/gridmap_scene.json`、
`logs/m5_seg/gridmap_drive_probe/gm_probe2.log`（首测日志）、
`.../gm_probe2.session.json`（会话审计）。
