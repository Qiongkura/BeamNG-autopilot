# AI 转接文档（2026-11-07）

> 交接对象：接手本项目的下一个 AI。
> 本文档是**唯一权威入口**，读完即可安全动手；引用的事实全部有文件/命令可核对。
> 阅读顺序：§1 纪束 → §2 环境 → §3 现状 → §6 阻塞 → §10 下一步。

---

## 1. 纪束（先读，违反=事故）

来源：`AGENTS.md` + 用户多次强调。**没有例外。**

1. **不降门**：评测门限不因结果调整；不因"接近门槛"而采纳。
2. **不消费最终集**：留出集/最终集不得进入训练或调参。
3. **不新增人工标注**。
4. **不晋级模型**：未过预注册门槛的 checkpoint 不改默认、不发布；部署模型缺失时 fail-closed（感知禁用），**不得静默回退到研究 checkpoint**。
5. **每批实验先预注册**：改协议/改门限/新实验前先写 `docs/T16_*_PREREG_*.md`，冻结后再跑；修订案（A1/A2…）写进原文档。
6. **一个模块一个提交**：`scripts/check_commit_scope.py` 必须过（模块=beamng_autopilot/<子包>、scripts、tests、docs、顶层文件；tests 可随被测模块）。
7. **技术债棘轮**：`scripts/m5_debt_meter.py --check` 必须过（except_exception / silent_except / prints / max_func_lines 不得超基线）；基线更新必须单独提交并注明理由，**不得用 wrapper 绕过计数**（曾被抓：`_log_drive` 用 sys.stdout.write 躲 print 计数，已回退）。
8. **probe/诊断输出绝不进训练池**。
9. **失败/隔离证据保留**：不得删除失败日志、quarantine、UNKNOWN 记录。
10. **感知铁律**：FSD 入口不得用"地图线+偏移"做横向参考（`--allow-legacy-map-lane` 除外，且不算合规运行）。
11. **不动运行时产物**：训练产物、遥测、证据归档不被提交或清理。
12. **报告必须如实**：测试失败就报失败；没跑过模拟器就不说"已复现"；UNKNOWN 不是 PASS。

## 2. 环境与入口

| 项 | 值 |
| --- | --- |
| 仓库 | `I:\projects\beamng-autopilot`，分支 `fix/round3-hardening-20260921`，HEAD `390fe84`，与 origin 同步 |
| Python | `.venv\Scripts\python.exe`（3.10.11；系统 Python 在 `I:\python`，别混用） |
| BeamNG | Tech 0.38.5，home `G:\BeamNG.tech.v0.38.5.0`；RPC 端口 **47000**（2026-11-07 从 64257 迁移，见 §7）；**单实例**（并发=事故） |
| 地图包 | `G:\BeamNG.tech.v0.38.5.0\content\levels\*.zip`（gridmap_v2 / Utah / east / west / italy） |
| shell | 用户要求 **pwsh7**：`pwsh -NoProfile -NonInteractive -Command "..."`；UTF-8 需 `[Console]::OutputEncoding=[Text.Encoding]::UTF8` + `$env:PYTHONIOENCODING='utf-8'` |
| 杀引擎 | `taskkill /PID <pid> /T /F`（GBK 输出，代码里必须 `errors="replace"`）；进程名 `BeamNG.tech.exe` / `BeamNG.tech.x64.exe`（一个会话 7-9 个进程，正常） |
| pytest | 全量 ~10-15 分钟；`-p no:cacheprovider`；**偶发 flaky**（见 §9） |
| git | 一直有 `unable to find all commit-graph files` 警告——无害，但没修过 |

## 3. 项目现状（一句话 + 证据水位）

**一句话**：感知/驾驶链路在 BeamNG 里端到端可跑，GridMap 无标线试车场单场景单次行驶 29.4m；但 **(a) 标线识别的部署默认模型缺失（fail-closed 中），(b) 低照度/雨天 10/12 格不可用，(c) 采集扩池被"右线渲染偏移 0.8m"阻塞在导出真值，(d) 多项采集/评分状态门不足以支撑结论**。

### 3.1 感知（分割+配对）

- 认证池：71 族 / 214 帧 / 4 图（utah 21、east 17、west 31、italy 2），engine_verified。
- 固定 4 折：A=早期 utah、B=后期 utah、C=east+west+italy、W=west。
- 最高水位：`seg_model_v28_ssl{A,B,C,W}`（种子42）+ `v29_ssl2{A,B,C,W}`（种子43）集成，宏平均 **0.539246**（A 0.5048 / B 0.6897 / C 0.4232 / W 0.4312）。
- 集成部署组件已就绪（`EnsembleSegmenter`、D1-D4 过），但**保持显式开关 `--seg-model2`，未默认化**（需 live 审计+用户确认）。
- **部署默认 `logs/m5_seg/seg_model/best.pt` 已灭失**（2026-10-29~11-01 外部删除事件）。现在 `default_model_path()` 返回 None → `Segmenter()` 抛错 → 感知禁用（fail-closed，这是**有意的**）。跑任何要感知的东西必须 `--seg-model logs/m5_seg/seg_model_v28_sslA/best.pt`。
- 训练侧 15 个变量已证伪（lw/epoch/过采样/容量/strip/SSL 轮数/TTA 等），**别再调参**，瓶颈是数据。
- 负结果清单见 `docs/T16_METHODOLOGY_20261104.md` §6。

### 3.2 驾驶（GridMap 试车场）

- 场景：`logs/m5_seg/gridmap_scene.json`（mid 500,500 / dir (0,1) / proving_lane=仅两条橙线、无铺装）。
- 单次行驶 **29.44m / vmax 7.01 / 3 stops / 0 reject=side**（`logs/m5_seg/gridmap_routeref_probe.log`）。仅单场景单次，**不能当稳定能力**。
- 已修的三处驾驶根因（都过了 A/B）：stuck 判定用 `prev_brk`（原始 controller brk 恒 0）；`_climb_clearance_ok` 接受 +inf 净空；`--scenario-json` 用场景 `mid/dir` 生成道路中线 route_ref（否则 strict 侧向门 reject=side）。

### 3.3 条件矩阵（12 格，TOD×天气）

- 结果 **PASS 2（noon_clear、noon_fog）/ FAIL 10**（`logs/m5_seg/proving_conditions/conditions_table.json`）。
- **⚠ 结论降级**：评分器只查 `tod_after` 非空，**没比较请求时刻与读回**；天气无独立读回。所以"环境全部有效/失败均为真实感知失败"是过强表述。低照度 9 格 `no_markings` 12/12 与 noon_rain `none` 大概率真实，但需按 §6.4 重核。
- `cert_line_recall` 的真实含义：**候选横向位置落在 GT ±0.6m 内的 tick 比例**，不是逐像素认证。

### 3.4 采集扩池（方向 2，当前主战场）

- 预注册：`docs/T16_PROVING_CAPTURE_PREREG_20261105.md`（门禁冻结：覆盖率≥0.8、投影残差中位≤1px/最大≤2px、漆色≥0.9、环境读回有效）。
- **站点总数是 40**：S1/S2/S3 各 12 格（训练）+ P 站 4 格（留出：noon_clear/noon_rain/dusk_clear/night_clear）。**文档/帮助里写的"36 格"是错的**（已在审查中确认），以 40 为准。
- 已完成 **4/40**：S1 的 dawn_clear(499帧)、dawn_rain(453)、noon_clear(437)、noon_rain(51)。其余 36 格未跑。
- **导出被阻塞**：`m5_proving_export.py` pilot **51/51 帧被拒**——右线渲染位置比构建节点（`probe.built.json`：left 299.1 / right 302.7，宽 0.15m）向外偏约 **0.8m**；左线残差 0.55px、右线中位 10.4px。根因未定（详见 §6.1 与 `docs/T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md`）。
- **在 A1 对照实验定论前，导出器不得用于训练。**

## 4. 关键文件地图

```
beamng_autopilot/
  fsd_drive.py            # 5700+ 行巨石：FSDriveSession.run ~3400 行（技术债最大项）
                          # 关键点：_counts_as_stuck(675) / _climb_clearance_ok(668) /
                          # _scenario_road_reference(433) / _setup_scenario_and_env(2436) /
                          # drive循环帧保存 _save_probe_frame(419, 调用点在 drive 循环 frames+=1 后)
  lane/pairing.py         # 配对：_best_vision_pair / _resolve_pair_world_geometry /
                          #   BEAMNG_THIN_MERGE(默认关) / BEAMNG_PAIR_WIDTH_NEAR(默认关)
  lane/constants.py       # LANE_* 常量（LANE_PAIR_WIDTH_NEAR_M=10）
  lane/reference.py       # _lane_gate_verdict：strict 侧向门（side/corner/heading）
  vision/segmentation.py  # Segmenter/EnsembleSegmenter；default_model_path()=fail-closed；
                          #   BEAMNG_CV_WHITE(默认1)/BEAMNG_CAND_CLAHE/BEAMNG_YELLOW_GATE
  vision/lanes.py         # _mask_to_markings + _component_fill_ok(minAreaRect 斜线修正)
  vision/heads/semantic.py# 逐 tick 感知头（yellow prior 融合、evidence、概率门）
  pinned_weights.json     # 权重清单（无代码消费；joint_default=seg_model/best.pt）
scripts/
  m5_fsd_drive.py         # drive CLI（--scenario-json/--mat-*/--seg-model/--out/--drive-*）
  m5_pairing_probe.py     # 配对探针（场景模式、--save-frames、--drive-seconds、probe.built.json）
  m5_controlled_scenes.py # decal 场景生成器（SCENES/MAP_PROFILES/build_scenario/verify_batch）
  m5_proving_capture.py   # 采集 runner（--run-all 40格、--run-site、capture.json 状态门）
  m5_proving_export.py    # 导出器（实验性，被 §6.1 阻塞）
  m5_proving_conditions.py# 12 格静止条件 runner/评分（--plan/--run/--score）
  m5_debt_meter.py        # 技术债棘轮（scripts/debt_baseline.json）
  check_commit_scope.py   # 单模块提交检查（提交后必跑）
  check_t16_regression.py # T16 回归 7 组（deploy/post_norm/diagnostics/conditions/scene/probe/drive）
  m5_offline_validate.py  # 离线规划/避障回归
  m5_eval_seg.py / m5_train_seg.py  # 评测/训练（--model 必填）
docs/
  ROADMAP_20261104.md     # Phase 0-5 总计划
  T16_HANDOFF_20261104.md # 交接（§6 试车场系列，含 6.1-6.5）
  T16_METHODOLOGY_20261104.md # 方法学金标准
  T16_PROVING_CAPTURE_PREREG_20261105.md  # 扩池预注册（本实验的宪法）
  T16_PROVING_GROUND_CONDITIONS_RESULT_20261105.md  # 12 格结果
  T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md      # 导出阻塞+A1 修订案
logs/
  m5_seg/proving_capture/<site>/<cell>/   # 采集帧+capture.json+probe.built.json
  m5_seg/proving_sites/                   # 4 个站点 JSON
  m5_seg/proving_pool/                    # 导出（当前 51/51 拒）
  m5_seg/proving_conditions/              # 12 格表
  m5_e2e/shadow_fsd_*.npz                 # 驾驶 shadow episode（rgb/bev/轨迹）
```

## 5. 常用命令

```bash
# 门禁（每次收尾/推送前全跑）
.venv/Scripts/python.exe scripts/m5_debt_meter.py --check
.venv/Scripts/python.exe scripts/check_commit_scope.py            # 或 --last N
.venv/Scripts/python.exe scripts/check_t16_regression.py          # 7 组，~3 分钟
.venv/Scripts/python.exe scripts/m5_offline_validate.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider         # ~10-15 分钟，偶发 flaky

# 试车场驾驶（必须显式 seg-model；proving 三开关）
$env:BEAMNG_CV_WHITE='0'; $env:BEAMNG_LINE_MERGE='0'; $env:BEAMNG_PAIR_WIDTH_NEAR='1'
.venv/Scripts/python.exe scripts/m5_fsd_drive.py --runtime tech --map gridmap_v2 `
  --scenario-json logs/m5_seg/gridmap_scene.json --scenario-line-convention right_lane `
  --scenario-edge-margin 0.6 --mat-white line_yellow_double --mat-yellow line_yellow_double `
  --mat-pave m_road_asphalt_rough_01 --mat-gravel m_dirt_road_gravels `
  --seg-model logs/m5_seg/seg_model_v28_sslA/best.pt --seconds 20 --max-wall-s 150 `
  --out logs/m5_seg/<name>.json

# 采集（扩池）
.venv/Scripts/python.exe scripts/m5_proving_capture.py --build-sites
.venv/Scripts/python.exe scripts/m5_proving_capture.py --run-site --site S1   # 12 格
.venv/Scripts/python.exe scripts/m5_proving_capture.py --run-all              # 40 格

# 条件矩阵
.venv/Scripts/python.exe scripts/m5_proving_conditions.py --plan / --run / --score

# 提交（单模块）
git add <模块文件> [<tests/...>]
.venv/Scripts/python.exe scripts/check_commit_scope.py HEAD~1..HEAD
git push origin HEAD
```

## 6. 当前阻塞清单（按优先级；全部有证据）

### 6.1 [已解除] 右线渲染偏移 ~0.8m，导出真值不可用
- A1 定论（2026-11-07，commit 7a6308c/8fe305d）：右线渲染**固定外偏 -0.85m**、渲染跟随节点；左线对齐（+0.05 ≈ 量化）；只画左线时右走廊 0/1200 命中（无幽灵线）。
- 导出器按结论重写（beb1419）：标签=渲染漆色连通域（世界 δ 归属最近线；厚度 2×p90 折线距离）、built 绑定几何、溯源 SHA-256、stale/quarantine 保留；pilot **1/51 入池**（旧 0/51），门禁零调整。
- 详细证据与产物见 `docs/T16_PROVING_EXPORT_PILOT_BLOCKER_20261105.md`。

### 6.2 [已闭环 1/2/4/5] 导出器/采集状态门不可信（审查确认的缺陷）
审查（2026-11-06）确认的问题，修复前**所有 `status=ok` 都要打折**：
1. ~~capture/probe 不传播子进程失败~~ → attempt 编排按 rc/超时/帧清单/归属判 EVIDENCE_READY（3591c22）；
2. ~~环境门不比较请求 vs 读回时刻、天气无读回~~ → tod 容差比对 + 引擎侧天气读回（3591c22）；
3. ~~重试残留帧计入帧数~~ → attempt 独占目录 + 逐帧 SHA-256 清单核对（3591c22）；
4. ~~INCOMPLETE 格被跳过无法恢复；超时部分输出丢失~~ → 自动重试/`--reclaim`，历史 attempt 全保留（3591c22/ca7fd92）；
5. ~~导出器不读 `probe.built.json`、color 口径、无溯源~~ → 已按 A1 重写（beb1419）。
6. 2026-11-07 深夜补修两处挂死：探针非守护线程卡退出（os._exit）、
   编排侧管道被引擎继承导致 `subprocess.run` 超时后二次 drain 永久挂死
   （输出落 `probe.log` + 超时 `taskkill /T /F` 整树）。S1/dusk_clear
   实跑 EVIDENCE_READY（27/27 帧、无超时、引擎干净）。

### 6.3 [P1] strict 场景参考线安全边界
- 场景**构建/加载失败**后代码继续运行，但 route_ref 仍可能从 JSON 生成并参与 strict 侧向门（`fsd_drive.py` `_setup_scenario_and_env` 的 except 后继续 + `_scenario_road_reference`）。
- 参考线只在初始化生成一次（后10m/前80m），**行驶超出窗口后回退成过自车直线**（`local_route` fallback）→ 可能再次 reject=side。
- 修法：参考线绑定"场景加载成功"标志；窗口按场景长度裁剪；失败/耗尽时 fail-closed。加负例测试。

### 6.4 [P1] 条件矩阵可信度重核
- 评分加"请求时刻 vs 读回（容差 ~1min）"比较；不一致 → UNKNOWN；
- 天气加独立读回证据（引擎侧 weather 状态），没有就 UNKNOWN；
- 重算 12 格并更新 `conditions_table.json` 与结果文档；已有 dawn/dusk/night 的 `no_markings` 大概率维持，但要重算而不是沿用。

### 6.5 [P2] 测试隔离
- 全量 pytest 偶发失败：`test_acquire_port_rotates_around_a_busy_port`（端口）、`test_seg_resume::test_resume_equals_uninterrupted_on_the_deterministic_path`（状态残留）；单独跑都过。
- 修法：端口用随机/释放等待，tmp 目录隔离；目标=连续 3 次全量全绿。

### 6.6 [P2] 其他
- `git commit-graph` 警告未修；`logs_paper_archive.tar.gz` 未跟踪未审计；
- `EnsembleSegmenter.last_reset_errors` 无消费方（诊断需主动读）；
- `fsd_drive.py` 5700 行巨石待拆（PlacementManager/纵向状态机/安全仲裁）；
- 采集格数 40 vs 36 文档错误（§3.4）。

## 7. 环境与代码的坑（踩过一次就别再踩）

- **BeamNG 并发**：上一实例没死透就开下一个 → RPC 挂死。启动前必查进程；超时清理用 `taskkill /T /F`（GBK 输出必须 `errors="replace"`，否则 UnicodeDecodeError）。
- **RPC 端口必须低于 49152**：64257 落进 Hyper-V/WSL/HNS 动态保留段（约 62000-64449，每次开机洗牌），保留期内引擎绑定报 WSAEACCES → tcom 起不来 → 采集全失败（表现为"启动冻结"，实际是控制台在等 GE-Lua> 而日志未刷盘）。现用 47000。
- **探针/子进程输出绝不走管道**：引擎会继承探针的 stdout 句柄，管道模式下父进程超时后二次 drain 等 EOF 会永久挂死；一律 `stdout=文件`，超时按进程树强杀。
- **探针报告落盘后必须 `os._exit(0)`**：beamngpy/相机非守护线程会让解释器卡在退出阶段，编排侧会把成功采集误判超时。
- **pwsh7 输出**：中文需显式 UTF-8（见 §2）；pytest 管道缓冲会让中途看不到输出。
- **heredoc 反斜杠**：Git Bash heredoc 里写 `\n` 会被吞，改 Python 补丁文件或 pwsh。
- **时间**：`set_weather_preset` 是动画，会回写 TOD——固定顺序 weather→sleep(2)→tod，读回留证。
- **pytest 全量**：~10-15 分钟；跑之前想清楚要不要全量（多数时候定向+T16 回归足够）。
- **日志目录被 gitignore**：提交不携带证据；引用结果时给 `logs/...` 路径并确认文件还在（本会话发生过证据目录被删）。

## 8. 模型与数据资产

| 资产 | 路径 | 状态 |
| --- | --- | --- |
| SSL 单模型（fold 专家） | `logs/m5_seg/seg_model_v28_ssl{A,B,C,W}/best.pt` | 在 |
| SSL2（种子43） | `logs/m5_seg/seg_model_v29_ssl2{A,B,C,W}/best.pt` | 在 |
| 部署默认 | `logs/m5_seg/seg_model/best.pt` | **缺失**（fail-closed） |
| 旧部署 | `logs/m5_seg/seg_model_v8_backup/` | 已灭失（别引用） |
| 认证池 | `logs/m5_seg/`（71 族/214 帧，批次哈希在 evidence_archives） | 在 |
| 试车场采集 | `logs/m5_seg/proving_capture/S1/*`（4 格，437-499 帧） | 在 |

## 9. 测试现状

- 全量：~2100+ 用例，10-15 分钟。三次全量 = 两种单测偶发失败（§6.5），第三次全绿。
- 定向快速：`tests/test_proving_capture.py`、`tests/test_proving_conditions.py`、`tests/test_proving_export.py`、`tests/test_lane_fill_gate.py`、`tests/test_pair_width_near.py`、`tests/test_fsd_drive_helpers.py`。
- T16 回归 7 组是**发布门**；全量 pytest 是背景健康检查。

## 10. 下一步（建议执行顺序，全部来自已预注册/已审查的结论）

1. ~~**A1 对照采集** → 定右线偏移根因~~（✅ 2026-11-07）。
2. ~~按 A1 结论修 `m5_proving_export.py`~~（✅ beb1419，pilot 1/51 入池）。
3. ~~修采集状态门（§6.2 的 1/2/4）~~（✅ 3591c22/ca7fd92；深夜补修退出/管道挂死）。
4. **重跑已完成的 4 格（旧门产物）+ 跑完剩余 36 格**（`m5_proving_capture.py --run-all --reclaim`；每格先查引擎集合为空）→ 导出。
   - 2026-11-07 23:07 首跑：clear 格 5/5 全过（27 帧/格），fog/rain 格 9/9 因 **TOD 被天气动画回写**（读回 14:2x–14:5x）判 INCOMPLETE。续跑前先修 `_apply_environment` 的等待/轮询（见 blocker 文档"批次首跑观察"）。
5. strict 场景参考 fail-closed（§6.3）+ 负例测试。
6. **2 seed 混练**（v28_sslA 起步，原池:新池=1:1）→ 重跑 12 格 + 原 4 折 + P 折（晋级门见预注册 §5：12 格 ≥8 PASS、低照度 ≥5、宏平均 ≥0.534246、P 折 +0.05）。
7. 全绿后才谈：集成默认化（需用户确认）、默认模型归位、`BEAMNG_PAIR_WIDTH_NEAR` 工程化。

## 11. 与用户的协作约定

- 用户中文交流；要求"如实、有据、不糊弄"；**质疑和反驳是常态，按证据答**。
- 用户偏好 pwsh7；嫌慢就少跑全量、多用定向+门禁。
- 长任务放后台 + 定期 `Get-Content -Tail` 轮询；收尾时停批/杀引擎。
- 用户会随时说"收尾"——把在跑的批停干净、门禁过一遍、文档/提交/推送，然后交短报告。
