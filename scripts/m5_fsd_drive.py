"""FSD-mode live driving: FSDStack planning -> safety monitor -> control.

This is the optional *real-driving* path of the FSD-style stack: instead
of only recording shadow data, it drives the car with the layered
planner's chosen trajectory, arbitrated every frame by the safety
monitor (which can degrade to a stop when the path is blocked, sensors
go stale, or the trajectory leaves the lane).  It is a separate entry
point from ``m5_autopilot.py`` so the proven rule autopilot (94.6%
route result) is never touched.

Usage::
    .venv\\Scripts\\python.exe scripts\\m5_fsd_drive.py --runtime tech \\
        --attach --seconds 30 --speed 8
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.fsd_drive import run
from beamng_autopilot.fsd_realism import lane_mode_is_fsd_compliant


def build_parser() -> argparse.ArgumentParser:
    """CLI 定义（抽出来是为了让"默认值是否合规"可被单测钉住）。

    2026-10-20：``--lane-mode`` 默认从 ``map`` 改成 ``sensor``、``--strict``
    默认开启。理由见 AGENTS.md 铁律——**FSD 入口不得用导航线/地图线 + 偏移做
    横向参考**；旧默认等于每个不带参数的 FSD 运行都在用被禁的参考。需要诊断
    旧行为时显式加 ``--allow-legacy-map-lane``，那一轮不算 FSD 合规运行。
    """
    ap = argparse.ArgumentParser(description="FSD-mode live driving")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="auto")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default=None,
                    help="BeamNG map/level to drive on (default italy; "
                         "with --attach this forces a scenario load to "
                         "switch the running instance to the map)")
    ap.add_argument("--tod", type=str, default=None, metavar="HH:MM",
                    help="时刻（条件矩阵用）：夜间/黄昏鲁棒性；EnvironmentApi.set_tod")
    ap.add_argument("--weather-preset", type=str, default=None,
                    help="天气预设名（art/weather/defaults.json 中的名字）")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--max-wall-s", type=float, default=None,
                    help="整轮墙钟上限（秒）：放置阶段超限即放弃并正常收尾；默认 None=不限"
                         "（strict+sensor 的放置循环按设计不退出，扫位姿时必须设上限）")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=3)
    # 536x403 = segmentation training resolution: live frames feed the
    # UNet upscaled, and far-line pixels lost at 400x300 cannot be
    # recovered by upscaling (town far-line misses 2026-09-06)
    ap.add_argument("--cam-w", type=int, default=536)
    ap.add_argument("--cam-h", type=int, default=403)
    ap.add_argument("--seg-model", type=str, default=None,
                    help="segmentation checkpoint override (default deployed model)")
    ap.add_argument("--scenario-json", type=str, default=None,
                    help="Phase 2 试车场：放置前用采集器构建 decal 车道场景"
                         "（试车场场景规格 JSON；默认关）")
    ap.add_argument("--scenario-line-convention", default="right_lane")
    ap.add_argument("--scenario-edge-margin", type=float, default=0.6)
    ap.add_argument("--mat-white", default=None)
    ap.add_argument("--mat-yellow", default=None)
    ap.add_argument("--mat-pave", default=None)
    ap.add_argument("--mat-gravel", default=None)
    ap.add_argument("--seg-model2", type=str, default=None,
                    help="部署集成（研究候选，默认关）：与 --seg-model 组成"
                         "EnsembleSegmenter（概率平均）；只给 --seg-model "
                         "时行为与现行完全一致")
    ap.add_argument("--line-seg-model", type=str, default=None,
                    help="optional separate painted-line checkpoint; road mask "
                         "still comes from --seg-model, while line extraction "
                         "uses this checkpoint (recommended for the paved-" 
                         "shoulder fine-tune until its line head is revalidated)")
    ap.add_argument("--teleport", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "YAW_DEG"),
                    help="teleport to an open stretch before driving")
    ap.add_argument("--out", type=str, default=None,
                    help="path for per-frame JSON telemetry export")
    ap.add_argument("--pose-contract", choices=("production_alignment", "fixed_actual_pose"),
                    default="production_alignment",
                    help="placement contract; fixed_actual_pose rejects pose drift before exposure")
    ap.add_argument("--position-tolerance-m", type=float, default=0.10,
                    help="fixed_actual_pose position tolerance in metres")
    ap.add_argument("--heading-tolerance-deg", type=float, default=1.0,
                    help="fixed_actual_pose heading tolerance in degrees")
    ap.add_argument("--probe-geometry", action="store_true",
                    help="record exact sensor boundaries, candidates, control path and grid; "
                         "diagnostic overhead is reported separately")
    ap.add_argument("--seg-post-norm-width", type=int, default=None,
                    help="研究开关（默认 None=现行行为）：后处理在固定宽度"
                         "（如 192）运行再放大，修复 536 输入下细黄线被像素核"
                         "删除；见 T16_POST_NORM_RESULT_20261104")
    ap.add_argument("--pairing-research", choices=("off", "far_thin",
                                                   "thin_near"),
                    default="off",
                    help="研究臂开关（默认 off = 现行行为）：far_thin 放行"
                         "几何自洽的远场薄线对；thin_near 放宽薄线近场计数。"
                         "仅用于预注册实验，不改变默认门限")
    ap.add_argument("--pairing-probe-frames-dir", type=str, default=None,
                    help="回放诊断（默认关闭）：放置阶段逐 tick 保存 live 帧"
                         "+位姿+相机模型到该目录（npz），供离线 detect_lines 回放")
    ap.add_argument("--pairing-probe-out", type=str, default=None,
                    help="研究探针（默认关闭）：放置阶段逐 tick 把配对决策、"
                         "拒绝计数、ego 位姿写入 JSONL；只读记录，不参与控制。"
                         "供 scripts/m5_pairing_probe.py 使用")
    ap.add_argument("--lane-mode", choices=("map", "auto", "sensor"),
                    default="sensor",
                    help="lane-keep reference policy: sensor = perception-led "
                         "(default, the only FSD-compliant mode), auto/map = "
                         "legacy map-prior lane (requires "
                         "--allow-legacy-map-lane; AGENTS.md forbids it on FSD "
                         "entries)")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="FSD realism mode (docs/fsd_realism.md, default ON): "
                         "the map lane may NEVER steer or label a frame; no "
                         "paired perception lane -> no-lane degradation. "
                         "--no-strict re-enables the legacy rule stack and "
                         "needs --allow-legacy-map-lane")
    ap.add_argument("--allow-legacy-map-lane", action="store_true",
                    help="诊断用逃生口：允许 --lane-mode map/auto 或 --no-strict。"
                         "AGENTS.md 铁律禁止 FSD 入口用导航线/地图线做横向参考；"
                         "开了它这一轮**不算 FSD 合规运行**，结果不得用于验收")
    ap.add_argument("--allow-unplaced", action="store_true",
                    help="collection mode: on cross-map lane-placement "
                         "failure keep driving unplaced instead of aborting")
    ap.add_argument("--corridor-lane", action="store_true",
                    help="REFUTED LIVE, kept only as the recorded negative "
                         "result: the corridor's right edge is the ROAD edge "
                         "(spills ~1.9 m past the lane line on town), so "
                         "the candidate pointed 1.24 m off the lane centre "
                         "on 52.7%% of frames (town_1789142315).  Do not "
                         "enable without a redesign anchored to painted "
                         "lines.")
    ap.add_argument("--paved-lane", action="store_true",
                    help="OPT-IN, not proven: use the paved-boundary lane "
                         "candidate on a paved road with no usable marking "
                         "(keep right against the observed paved edge).  "
                         "Default OFF: live 2026-09-18 with the deployed "
                         "model the car rode the lane line and hit the "
                         "guardrail, because the road mask classifies the "
                         "gravel shoulder as pavement.  Needs a trustworthy "
                         "pavement edge (fine-tuned model) + live A/B.")
    ap.add_argument("--e2e-model", type=str, default=None,
                    help="trained E2ENetTorch checkpoint to rank as the "
                         "neural planning candidate (default: "
                         "logs/m5_e2e/best_temporal.pt)")
    ap.add_argument("--no-e2e", action="store_true",
                    help="disable the E2E neural planning candidate")
    ap.add_argument("--bc-model", type=str, default=None,
                    help="trained DAVE-2 BC steering checkpoint to rank "
                         "as a neural candidate (default: "
                         "logs/m3_bc/bc_tech_smallgrid.pt)")
    ap.add_argument("--no-bc", action="store_true",
                    help="disable the DAVE-2 BC steering candidate")
    ap.add_argument("--dqn-model", type=str, default=None,
                    help="trained SB3 DQN decision policy (default: "
                         "logs/m4_dqn/dqn_decision.zip)")
    ap.add_argument("--no-dqn", action="store_true",
                    help="disable the DQN decision layer")
    ap.add_argument("--goal", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="set an in-game navigation route to this goal "
                         "before driving (else reuse the active nav route)")
    ap.add_argument("--no-signal", action="store_true",
                    help="disable the traffic-light stop (a confident "
                         "vision red or a game-side red signal normally "
                         "stops the car)")
    ap.add_argument("--ring", choices=("front", "all"), default="front",
                    help="camera ring to poll: front = front_main only "
                         "(the only frame the live tick consumes), "
                         "all = full 8-camera surround ring")
    ap.add_argument("--traffic", type=int, default=0, metavar="N",
                    help="park N NPC vehicles along the route (YOLO / "
                         "obstacle-fusion verification; best-effort)")
    ap.add_argument("--vis-on-flag", action="store_true",
                    help="render ONLY the frames where perception claims a "
                         "crossing / off-pavement (plus controls, see "
                         "--vis-control-every); use with --vis-dir")
    ap.add_argument("--vis-control-every", type=int, default=0,
                    metavar="N",
                    help="with --vis-on-flag: also render every N-th "
                         "UNFLAGGED frame as a control (0 = none)")
    ap.add_argument("--vis-dir", type=str, default=None,
                    help="where --vis writes its frames (default "
                         "logs/m5_vis/live; give each run its own folder)")
    ap.add_argument("--vis", type=int, default=0, metavar="N",
                    help="render a live lane-recognition overlay every N "
                         "ticks into logs/m5_vis/live/ (0 = off): camera "
                         "frame + road/line masks + world markings + the "
                         "accepted lane reference and its hard boundaries "
                         "+ BEV panel + HUD")
    ap.add_argument("--no-shadow", action="store_true",
                    help="disable shadow-episode recording during the drive")
    ap.add_argument("--capture-range-replay", action="store_true",
                    help="Tech-only: save one raw range input during warmup; requires --out")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if not lane_mode_is_fsd_compliant(args.lane_mode, args.strict):
        if not args.allow_legacy_map_lane:
            ap.error(
                f"--lane-mode {args.lane_mode} / --strict={args.strict} 不是 FSD "
                "合规路径：AGENTS.md 铁律禁止 FSD 入口用导航线/地图线 + 偏移做横向"
                "参考。默认已是 sensor+strict；确需诊断旧行为请显式加 "
                "--allow-legacy-map-lane（那一轮不得用于验收）。")
        print("[warn] 本轮走 legacy 地图横向参考（--allow-legacy-map-lane）："
              "结果不得用于 FSD 验收", file=sys.stderr)
    if args.capture_range_replay and (args.runtime != "tech" or not args.out):
        ap.error("--capture-range-replay requires --runtime tech and --out")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
