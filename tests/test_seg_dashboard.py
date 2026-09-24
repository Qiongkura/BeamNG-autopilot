"""T14 学习看板验收测试（离线：无网络、无 GPU、无游戏）。

覆盖方案"可视化验收案例"的四条（`docs/T14_BACKGROUND_MODEL_ITERATION_PLAN_
20260924.md` §"可视化验收案例"）：

1. 导入 T13 六个 ``train_hist.json``：按 seed 显示 3 个 epoch 与开发集
   line IoU，冻结集最终结果单独成节，训练曲线不得被标成安全成绩。
2. 模拟第 2 epoch 断电恢复：重放日志里同一 epoch 只画一次，并报出去重丢
   弃的点数（否则曲线会在重启处出现假跳变）。
3. 四个确定性样例（标线真值缺失但 RGB 有白线 / 路外假线 / 土肩被预测成路 /
   预测全背景）：每个都带证据等级、真值有效区、FP/FN 或 UNKNOWN 与淘汰
   理由；缺标线真值的帧不得出现算出来的 IoU（也绝不能出现伪造的百分比）。
4. 推理 p95 超限且闭环未跑：晋级给 ``rejected``/``needs_evidence``，闭环
   字段写"未测"。

另外固定三条渲染纪律：HTML 不引用任何外部资源（``http://`` / ``https://``
/ ``<script src``）、``checked: false`` 渲染成"未检查"、证据包里未测字段带
``"status": "未测"``。

测试全部走进程内 ``main([...])``（不起子进程、不连游戏），用 ``tmp_path``
造小样本，秒级完成。
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m5_seg_dashboard as dash
from beamng_autopilot.experiments.events import Event, EventLog, metric
from beamng_autopilot.experiments.manifest import (
    DatasetManifest,
    FrameRecord,
)

ARMS = ("armA", "armB")
SEEDS = (42, 43, 44)


# ---------------------------------------------------------------------------
# 造样本
# ---------------------------------------------------------------------------
def _write_t13_hists(root: Path) -> Path:
    """六个 ``train_hist.json`` 的小样本（逐 seed 数值不同，便于断言）。"""
    for arm in ARMS:
        for seed in SEEDS:
            d = root / f"{arm}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            blob = {
                "epoch": [0, 1, 2],
                "train_loss": [round(2.2 - 0.01 * seed - 0.2 * e, 4)
                               for e in range(3)],
                "val_miou": [round(0.70 + 0.001 * seed + 0.01 * e, 4)
                             for e in range(3)],
                "val_acc": [0.96, 0.96, 0.96],
                "val_line_iou": [round(0.20 + 0.001 * seed + 0.02 * e, 4)
                                 for e in range(3)],
            }
            (d / "train_hist.json").write_text(
                json.dumps(blob), encoding="utf-8")
    return root


def _write_t13_hist_with_gap(root: Path) -> Path:
    """一个 epoch 的 val_line_iou 缺失：曲线必须断开并写原因。"""
    d = root / "armA_seed42"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_hist.json").write_text(json.dumps({
        "epoch": [0, 1, 2],
        "train_loss": [2.1, 1.9, 1.7],
        "val_line_iou": [0.25, None, 0.33],
    }), encoding="utf-8")
    return root


def _eval_matrix(*, p95: float = 13.5, n_seeds: int = 3) -> dict:
    """两臂 × seed 的评估矩阵；frozen 与 dev 数值不同以便分辨口径。"""
    model = {}
    for arm_idx, arm in enumerate(ARMS):
        for seed in SEEDS[:n_seeds]:
            model[f"{arm}_seed{seed}/best.pt"] = {
                "model": f"logs/m5_seg/{arm}_seed{seed}/best.pt",
                "sha256_16": f"{arm}{seed}"[:16].ljust(16, "0"),
                "n_frames": 25,
                "line_precision": 0.60 + 0.01 * arm_idx,
                "line_recall": 0.95 + 0.001 * seed,
                "line_iou": 0.45 + 0.01 * arm_idx,
                "tp_px": 10000, "fp_px": 2000, "fn_px": 500,
                "offroad_false_line_px": 300 + 10 * arm_idx,
                "offroad_false_frac_of_pred": 0.02 + 0.001 * arm_idx,
                "pred_line_px": 12000, "gt_line_px": 10500,
                "inference_ms_p50": 11.0 + arm_idx,
                "inference_ms_p95": p95,
            }
    frozen = {}
    for arm in ARMS:
        frozen[f"{arm}_seed42/best.pt"] = {
            "model": f"logs/m5_seg/{arm}_seed42/best.pt",
            "n_frames": 50, "line_precision": 0.4929, "line_recall": 0.8181,
            "line_iou": 0.4929, "tp_px": 95741, "fp_px": 77229, "fn_px": 21284,
            "offroad_false_line_px": 6252, "offroad_false_frac_of_pred": 0.0361,
            "pred_line_px": 172970, "inference_ms_p50": 11.31,
            "inference_ms_p95": p95,
        }
    return {"frozen": frozen, "dev": model}


def _write_ident(dirpath: Path, *, arms: tuple[str, ...] = ("armA42",),
                 seeds: tuple[int, ...] = (42,)) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    for tag in arms:
        (dirpath / f"ident_t13_testA2_{tag}.json").write_text(json.dumps({
            "run": "t13_testA2/front_main",
            "view": "front_main",
            "summary": {
                "frames": 25, "frames_with_engine_line": 25,
                "engine_lines_total": 41,
                "candidates_total": 120, "candidates_matched": 17,
                "match_rate": 0.1417, "match_rate_null": 0.025,
                "roles_agreeing": 99, "role_agreement_rate": 0.824,
                "candidates_on_engine_line": 7, "candidates_on_road_only": 21,
                "candidates_off_road": 92,
                "candidate_paint_recall_p50": 0.0058,
            },
        }), encoding="utf-8")
    return dirpath


def _quality(*, road_valid: bool, paint_valid: bool, paint_px: int,
             paint_reason: str = "") -> dict:
    return {
        "road": {"valid": road_valid, "reason": "" if road_valid else "road px low",
                 "pixels": 12000},
        "paint": {"valid": paint_valid,
                  "reason": paint_reason or ("paint truth from human_revision"
                                             if paint_valid else
                                             "engine annotation's paint class "
                                             "is unreliable on this map "
                                             "(paint renders as ASPHALT)"),
                  "pixels": paint_px},
        "pavement": {"valid": False,
                     "reason": "pavement vs shoulder needs its own verified "
                               "labels", "pixels": 12000},
        "unknown_reason": "" if paint_valid else
        f"paint truth unusable: {paint_reason or 'engine annotation'}",
        "line_masked_px": 0 if paint_valid else paint_px,
        "frame_unknown_frac": 0.0,
        "label_sha256_16": "0" * 16,
        "notes": [],
    }


def _write_manifest(path: Path) -> Path:
    """真清单样本：train/dev/final + 被拒帧；曝光信息缺失以触发"未检查"。"""
    records = [
        FrameRecord(path="logs/m5_seg/a/frame_0000.npz", run="a",
                    view="front_main", group="italy/ring_a", map_name="italy",
                    source_id="ring_a", t_wall=1.0, exposure=None,
                    content_sha16="a" * 16, label_sha16="b" * 16, road_px=12000,
                    line_px=1500, quality=_quality(road_valid=True,
                                                   paint_valid=False,
                                                   paint_px=1500),
                    split="train"),
        FrameRecord(path="logs/m5_seg/a/frame_0001.npz", run="a",
                    view="front_main", group="italy/ring_a", map_name="italy",
                    source_id="ring_a", t_wall=2.0, exposure=None,
                    content_sha16="c" * 16, label_sha16="d" * 16, road_px=11000,
                    line_px=0, quality=_quality(road_valid=True,
                                                paint_valid=False, paint_px=0),
                    split="train"),
        FrameRecord(path="logs/m5_seg/b/frame_0000.npz", run="b",
                    view="front_main", group="italy/ring_b", map_name="italy",
                    source_id="ring_b", t_wall=3.0, exposure=None,
                    content_sha16="e" * 16, label_sha16="f" * 16, road_px=9000,
                    line_px=800,
                    quality=_quality(road_valid=True, paint_valid=True,
                                     paint_px=800,
                                     paint_reason="human_revision"),
                    split="dev"),
        FrameRecord(path="logs/m5_seg/c/frame_0000.npz", run="c",
                    view="front_main", group="italy/ring_c", map_name="italy",
                    source_id="ring_c", t_wall=4.0, exposure=None,
                    content_sha16="1" * 16, label_sha16="2" * 16, road_px=9500,
                    line_px=700,
                    quality=_quality(road_valid=True, paint_valid=True,
                                     paint_px=700,
                                     paint_reason="human_revision"),
                    split="final"),
        FrameRecord(path="logs/m5_seg/d/frame_0000.npz", run="d",
                    view="front_main", group="italy/ring_d", map_name="",
                    source_id="", t_wall=None, exposure=None,
                    content_sha16="3" * 16, label_sha16="4" * 16, road_px=0,
                    line_px=0, quality=_quality(road_valid=False,
                                                paint_valid=False, paint_px=0),
                    split="none",
                    reject_reason="no map identity in the recording: the "
                                  "group would be a directory name only"),
    ]
    mf = DatasetManifest(dataset_id="deadbeefdeadbeef", created="2026-09-24T00:00:00Z",
                         records=records, groups={"italy/ring_a": "train"})
    mf.save(path)
    return path


CASES = [
    {
        "case_id": "paint_truth_missing_rgb_white",
        "label": "标线真值缺失但 RGB 有白线",
        "level": "dev",
        "frame_sha16": "aa11bb22cc33dd44",
        "quality": _quality(road_valid=True, paint_valid=False, paint_px=1500,
                            paint_reason="engine annotation renders paint as "
                                         "ASPHALT while the RGB has 1500 "
                                         "white-line px"),
        "metrics": {
            "line_iou": {"value": 0.10, "unit": "ratio",
                         "missing": "paint truth unusable: the line channel is "
                                    "ignored for this frame"},
            "offroad_false_line_px": {"value": 0.0, "unit": "px"},
        },
        "decision": "rejected",
        "reasons": ["缺标线真值：不得把可见漆线当负例，也不得报漂亮的 IoU"],
    },
    {
        "case_id": "offroad_false_lines",
        "label": "路外假线",
        "level": "dev",
        "frame_sha16": "bb22cc33dd44ee55",
        "quality": _quality(road_valid=True, paint_valid=True,
                            paint_px=900, paint_reason="human_revision"),
        "metrics": {
            "offroad_false_line_px": {"value": 1250, "unit": "px",
                                      "numerator": 5, "denominator": 25},
            "missed_true_line_px": {"value": 40, "unit": "px"},
            "line_iou": {"value": 0.31, "unit": "ratio"},
        },
        "decision": "rejected",
        "reasons": ["路外假线像素超出门槛：候选在路肩画线"],
    },
    {
        "case_id": "shoulder_as_road",
        "label": "土肩被预测成路",
        "level": "dev",
        "frame_sha16": "cc33dd44ee55ff66",
        "quality": _quality(road_valid=True, paint_valid=True, paint_px=400,
                            paint_reason="human_revision"),
        "metrics": {
            "shoulder_as_road_px": {"value": 8000, "unit": "px",
                                    "numerator": 2, "denominator": 25},
            "line_iou": {"value": 0.28, "unit": "ratio"},
        },
        "decision": "rejected",
        "reasons": ["有铺装时土肩不得算道路（AGENTS 硬性约束）"],
    },
    {
        "case_id": "all_background",
        "label": "预测全背景",
        "level": "dev",
        "frame_sha16": "dd44ee55ff667700",
        "quality": _quality(road_valid=True, paint_valid=True, paint_px=600,
                            paint_reason="human_revision"),
        "metrics": {
            "pred_line_px": {"value": 0, "unit": "px"},
            "missed_true_line_px": {"value": 600, "unit": "px"},
            "line_iou": {"value": 0.0, "unit": "ratio"},
        },
        "decision": "rejected",
        "reasons": ["全背景预测：漏掉全部真线，line IoU 0"],
    },
]


def _write_tasks(dirpath: Path) -> Path:
    """一个真实 T13 相关帧/人物的任务结果 + 四个确定性样例。"""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "t13_testA2_frame0012.json").write_text(json.dumps({
        "frame": "logs/m5_seg/t13_testA2/front_main/frame_0012.npz",
        "frame_sha16": "aa11bb22cc33dd44",
        "level": "final",
        "engine_grader": {"grader": "engine", "gt_line_px": 2400,
                          "engine_lines_total": 2},
        "surface_grader": {"surface": "dirt", "paved_frac": 0.12},
        "traffic_persona": {"personas": 1, "vehicles": 2},
        "hazards": {"hazards": 1},
        "score_delta": {"value": -0.25, "unit": "", "numerator": 3,
                        "denominator": 25},
        "case": CASES[0],
    }), encoding="utf-8")
    for i, case in enumerate(CASES[1:], start=1):
        (dirpath / f"case_{i}.json").write_text(json.dumps({
            "frame": f"logs/m5_seg/t13_testB/front_main/frame_{i:04d}.npz",
            "level": "dev",
            "case": case,
        }), encoding="utf-8")
    return dirpath


def _render(tmp_path: Path, *args: str, out_name: str = "dash.html") -> str:
    out = tmp_path / out_name
    argv = ["render", "--out", str(out), *args]
    assert dash.main(argv) == 0
    return out.read_text(encoding="utf-8")


def _section(html: str, heading: str) -> str:
    """取某个 ``<h2>`` 到下一个 ``<h2>`` 之间的 HTML。"""
    m = re.search(r"<h2>" + re.escape(heading) + r"</h2>", html)
    assert m, f"section {heading!r} not found"
    nxt = re.search(r"<h2>", html[m.end():])
    return html[m.end(): m.end() + (nxt.start() if nxt else len(html))]


def _text(html: str) -> str:
    """去标签后的可读文本（断言文案用；数值断言仍用原始 HTML）。"""
    import html as _h
    return _h.unescape(re.sub(r"<[^>]+>", " ", html))


def _svg_for(html: str, title: str) -> str:
    hits = [b for b in re.findall(r"<svg .*?</svg>", html, re.S)
            if f"<title>{title}</title>" in b]
    assert hits, f"no svg titled {title!r}"
    return hits[0]


# ---------------------------------------------------------------------------
# (a) T13 六个 train_hist.json + 单独的冻结集分节
# ---------------------------------------------------------------------------
def test_t13_import_per_seed_series_and_frozen_section(tmp_path):
    root = _write_t13_hists(tmp_path / "seg_t13_data_20260924")
    evalp = tmp_path / "06_eval_matrix.json"
    evalp.write_text(json.dumps(_eval_matrix()), encoding="utf-8")
    html = _render(tmp_path, "--t13-import", str(root), "--eval", str(evalp))
    curves = _section(html, "学习曲线")
    # 每个 seed 自己的序列，3 个 epoch，值印在点上
    for arm in ARMS:
        for seed in SEEDS:
            block = _svg_for(curves, f"{arm}_seed{seed} · train_loss")
            assert block.count("<circle") == 3
            assert block.count('class="val"') == 3
            assert f"{arm}_seed{seed} · val_line_iou" in curves
    first = _svg_for(curves, "armA_seed42 · val_line_iou")
    assert "0.242" in first              # 0.20 + 0.001*42 的第一个 epoch 值
    # 曲线只能标训练中 / 开发集，不能出现最终集标签
    assert "最终集一次确认" not in curves
    assert "训练中" in curves and "开发集" in curves
    assert "不是安全成绩" in _text(curves)
    # 冻结集最终结果单独成节，且用的是 --eval 的 frozen 数字
    final = _section(html, "最终集一次确认")
    assert "49.3%" in final              # line_precision 0.4929 → 49.3%
    assert "81.8%" in final              # line_recall 0.8181（不写成 0）
    assert "50" in final                 # n_frames
    assert "冻结集" in _text(final)


def test_t13_import_reports_missing_epoch(tmp_path):
    root = _write_t13_hist_with_gap(tmp_path / "sub" / "t13")
    html = _render(tmp_path, "--t13-import", str(root))
    curves = _section(html, "学习曲线")
    assert "val_line_iou missing" in _text(curves)
    assert "epoch 1" in _text(curves)


# ---------------------------------------------------------------------------
# (b) 重放：同一 epoch 写两次只画一个点，并报出去重数
# ---------------------------------------------------------------------------
def test_replayed_epoch_renders_once_and_counts_duplicates(tmp_path):
    log = EventLog(tmp_path / "run_t14")
    base = dict(run_id="run_t14", candidate_id="armB_seed42",
                dataset_id="ds_abc", config_hash="cfg1234", seed=42)
    log.append(Event(phase="queued", status="ok", ts="2026-09-24T10:00:00Z",
                     **base))
    log.append(Event(phase="auditing", status="ok",
                     ts="2026-09-24T10:01:00Z", **base))
    log.append(Event(phase="training", status="ok", epoch=0,
                     ts="2026-09-24T10:02:00Z",
                     metrics={"train_loss": metric(2.0),
                              "val_line_iou": metric(0.2)}, **base))
    log.append(Event(phase="training", status="ok", epoch=1,
                     ts="2026-09-24T10:03:00Z",
                     metrics={"train_loss": metric(1.5),
                              "val_line_iou": metric(0.3)}, **base))
    # 断电后重启：同一 (phase,status,epoch,step) 再写一遍
    log.append(Event(phase="training", status="ok", epoch=1, step=None,
                     ts="2026-09-24T10:04:00Z",
                     metrics={"train_loss": metric(1.5),
                              "val_line_iou": metric(0.3)}, **base))
    html = _render(tmp_path, "--events", str(log.path))
    overview = _text(_section(html, "运行总览"))
    assert "重启重复点" in overview
    assert re.search(r"重启重复点\s*1\b", overview), overview[:400]
    assert "去重" in overview
    block = _svg_for(_section(html, "学习曲线"), "seed 42 · train_loss")
    assert block.count("<circle") == 2          # 一个 epoch 一个点
    assert "1.5" in block and "2" in block
    # run 的身份在重启前后一致
    assert "run_t14" in overview and "ds_abc" in overview


# ---------------------------------------------------------------------------
# (c) 四个确定性样例
# ---------------------------------------------------------------------------
def test_deterministic_cases_render_unknown_not_fake_numbers(tmp_path):
    tasks = _write_tasks(tmp_path / "tasks")
    html = _render(tmp_path, "--tasks", str(tasks))
    body = _text(_section(html, "候选任务结果"))
    raw = _section(html, "候选任务结果")
    for case in CASES:
        assert case["label"] in body
        row = next(r for r in re.findall(r"<tr>.*?</tr>", raw, re.S)
                   if case["label"] in r)
        assert 'class="lvl dev"' in row, case["label"]   # 每个样例都带等级
    for level in ("开发集", "最终集一次确认"):
        assert level in body
    # 任务结果的五个分项按固定顺序出现
    labels = [r["label"] for r in dash._task_rows(
        {"tasks": dash._task_state(_write_tasks(tmp_path / "tasks2")),
         "ident": {"readable": False}})]
    assert labels == list(dash.TASK_ROWS)
    assert "grader: engine" in body and "surface: dirt" in body
    # 缺标线真值的帧：写 UNKNOWN（原因），不写算出来的 IoU
    assert "line_iou" in body
    assert "paint truth unusable" in body
    assert "未测" in body
    assert "10.0%" not in html
    assert "UNKNOWN" in body
    # 真值有效区 + FP/FN 都要有
    assert "road" in body and "paint" in body and "pavement" in body
    assert "offroad_false_line_px" in body and "1250" in body
    assert "FP:" in body and "FN:" in body
    # 淘汰理由
    assert "缺标线真值" in body


# ---------------------------------------------------------------------------
# (d) p95 超限 + 闭环未跑
# ---------------------------------------------------------------------------
def test_over_limit_p95_and_closed_loop_untested(tmp_path):
    evalp = tmp_path / "eval.json"
    evalp.write_text(json.dumps(_eval_matrix(p95=61.5)), encoding="utf-8")
    pack = tmp_path / "pack.json"
    html = _render(tmp_path, "--eval", str(evalp), "--pack", str(pack))
    text = _text(html)
    assert "61.5" in text
    assert "rejected" in text
    assert "inference_ms_p95" in text
    # 闭环没有事件：写"未测"和固定原因，不写 0 事故
    assert "no Tech closed loop in this run" in text
    assert "未测" in text
    resources = _section(html, "资源与闭环")
    closed = resources[resources.find("闭环字段"):]
    for field in ("碰撞", "压线", "出铺装", "停车", "deadline", "新源消费"):
        assert re.search(rf"<td>{field}</td><td>.*?未测", closed, re.S), field
    assert "0 accidents" not in html
    blob = json.loads(pack.read_text(encoding="utf-8"))
    assert blob["report"]["closed loop"]["status"] == "未测"
    assert "no Tech closed loop in this run" in \
        blob["report"]["closed loop"]["reason"]
    raw = pack.read_text(encoding="utf-8")
    assert '"status": "未测"' in raw          # 未测项只带状态与原因
    assert blob["report"]["performance/deadline"]["full_tick_deadline"][
        "status"] == "未测"
    # 已测到的时延仍要导出成数值
    assert any(m["metric"] == "inference_ms_p95" for m in blob["measured"])


def test_pack_never_puts_a_zero_where_nothing_was_measured(tmp_path):
    """没有事件流/清单时，n_events 与丢弃重复点不能是 0，只能是未测。"""
    evalp = tmp_path / "eval.json"
    evalp.write_text(json.dumps(_eval_matrix()), encoding="utf-8")
    pack = tmp_path / "pack.json"
    _render(tmp_path, "--eval", str(evalp), "--pack", str(pack))
    blob = json.loads(pack.read_text(encoding="utf-8"))
    commit = blob["report"]["commit/config/run"]
    assert commit["n_events"]["status"] == "未测"
    assert commit["dropped_duplicates"]["status"] == "未测"
    assert commit["run_id"]["status"] == "未测"
    cov = blob["report"]["coverage/UNKNOWN"]
    assert cov["dataset_id"]["status"] == "未测"
    assert cov["n_by_split"]["status"] == "未测"
    assert cov["paint_truth_unusable_frames"]["status"] == "未测"


# ---------------------------------------------------------------------------
# 渲染纪律
# ---------------------------------------------------------------------------
def test_html_is_self_contained_and_no_external_resources(tmp_path):
    root = _write_t13_hists(tmp_path / "t13")
    evalp = tmp_path / "eval.json"
    evalp.write_text(json.dumps(_eval_matrix()), encoding="utf-8")
    html = _render(tmp_path, "--t13-import", str(root), "--eval", str(evalp),
                   "--manifest", str(_write_manifest(tmp_path / "ds.json")))
    for needle in ("http://", "https://", "<script src", "<link ", "@import"):
        assert needle not in html, f"external resource reference: {needle}"
    for heading in ("运行总览", "学习曲线", "固定探针图像", "候选任务结果",
                    "实验对比", "数据与标签", "资源与闭环"):
        assert f"<h2>{heading}</h2>" in html


def test_manifest_checked_false_renders_unchecked(tmp_path):
    manifest = _write_manifest(tmp_path / "ds.json")
    html = _render(tmp_path, "--manifest", str(manifest))
    body = _text(_section(html, "数据与标签"))
    assert "未检查" in body
    assert "同曝光跨视角重叠" in body
    # 有 Tech annotation 但无可靠标线真值：单独计数（样本里是 1 帧）
    assert "有 Tech annotation 但无可靠标线真值" in body
    assert re.search(r"有 Tech annotation 但无可靠标线真值", body)
    # 被拒帧与原因
    assert "no map identity in the recording" in body
    assert "组重叠" in body


def test_empty_manifest_is_unmeasured_not_a_row_of_zeros(tmp_path):
    """清单里 0 帧不是成功：帧数/覆盖/被拒/审计都写未测，不写 0。"""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": 1, "dataset_id": "x",
                                 "created": "2026-09-24T00:00:00Z",
                                 "groups": {}, "notes": [], "records": []}),
                     encoding="utf-8")
    pack = tmp_path / "pack.json"
    html = _render(tmp_path, "--manifest", str(empty), "--pack", str(pack))
    body = _text(_section(html, "数据与标签"))
    assert "清单里没有 records" in body
    assert "未测" in body
    blob = json.loads(pack.read_text(encoding="utf-8"))
    cov = blob["report"]["coverage/UNKNOWN"]
    for key in ("n_by_split", "n_rejected", "paint_truth_unusable_frames"):
        assert cov[key]["status"] == "未测", key


def test_unreadable_input_renders_not_readable(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    html = _render(tmp_path, "--eval", str(broken), "--manifest", str(broken),
                   "--events", str(tmp_path / "missing" / "events.jsonl"))
    text = _text(html)
    assert text.count("not readable") >= 3
    assert "<h2>实验对比</h2>" in html        # 渲染不中断


def _write_probe(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "epoch0002_front.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (dirpath / "epoch0002_front.json").write_text(json.dumps({
        "frame_sha16": "aa11bb22cc33dd44", "label_source": "human_revision",
        "model_sha16": "700ee0487dba1d95",
        "postprocess": "raw->morph->road constraint",
        "created": "2026-09-24T12:00:00Z"}), encoding="utf-8")
    (dirpath / "epoch0004_front.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return dirpath


def test_probes_need_sidecar_and_relative_existing_file(tmp_path):
    probes = _write_probe(tmp_path / "probes")
    html = _render(tmp_path, "--probes", str(probes))
    body = _section(html, "固定探针图像")
    assert body.count("<img") == 1
    assert 'src="probes/epoch0002_front.png"' in body
    assert "无法验证" in _text(body)
    assert "missing data" in _text(body) or "无法验证" in _text(body)


def test_probe_referenced_across_drives_is_not_a_broken_image(tmp_path,
                                                              monkeypatch):
    """跨盘符时相对路径算不出来：只写 missing 文本，不输出 <img>。"""
    probes = _write_probe(tmp_path / "probes")
    def _boom(*_a, **_k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")
    monkeypatch.setattr(dash.os.path, "relpath", _boom)
    html = _render(tmp_path, "--probes", str(probes))
    body = _section(html, "固定探针图像")
    assert "<img" not in body
    assert "missing: " in _text(body)


def test_pack_zip_when_not_json(tmp_path):
    evalp = tmp_path / "eval.json"
    evalp.write_text(json.dumps(_eval_matrix(p95=61.5)), encoding="utf-8")
    pack = tmp_path / "pack.zip"
    _render(tmp_path, "--eval", str(evalp), "--pack", str(pack))
    with zipfile.ZipFile(pack) as zf:
        names = zf.namelist()
        assert "evidence.json" in names
        blob = json.loads(zf.read("evidence.json").decode("utf-8"))
    assert blob["report"]["closed loop"]["status"] == "未测"


def test_watch_once_renders_and_exits(tmp_path):
    log = EventLog(tmp_path / "run_watch")
    log.append(Event(run_id="run_watch", candidate_id="c", dataset_id="d",
                     config_hash="h", seed=42, phase="queued", status="ok"))
    out = tmp_path / "watch.html"
    assert dash.main(["watch", "--events", str(log.path), "--out", str(out),
                      "--every", "1", "--once"]) == 0
    assert out.exists()
    assert "运行总览" in out.read_text(encoding="utf-8")


def test_run_dir_merges_event_streams_and_decisions(tmp_path):
    """`--run-dir`：合并 run 级 + 每候选事件流，并读 decision_*.json。

    `rounds` 按设计写多条事件流（共用一条流时第一个候选的 rejected 是终态，
    第二个候选就进不了 evaluating）；看板此前只读单条流，于是自治循环的
    "成对差值 / 硬门 UNKNOWN / 淘汰理由"在页面上根本看不到。
    """
    from beamng_autopilot.experiments.gates import Thresholds

    run = tmp_path / "run"
    (run / "candidates" / "cand-r0").mkdir(parents=True)

    def mk(phase, status, cand):
        return Event(run_id="r", candidate_id=cand, dataset_id="",
                     config_hash=Thresholds().config_hash, seed=0,
                     phase=phase, status=status)

    log = EventLog(run)
    log.append(mk("queued", "start", "loop"))
    log.append(mk("auditing", "started", "loop"))
    clog = EventLog(run / "candidates" / "cand-r0")
    for ph, st in (("queued", "start"), ("auditing", "ok"),
                   ("training", "done"), ("evaluating", "evaluated"),
                   ("rejected", "decided")):
        clog.append(mk(ph, st, "cand-r0"))
    (run / "decision_cand-r0.json").write_text(json.dumps({
        "candidate_id": "cand-r0",
        "pairings": {"road_iou": {"metric": "road_iou", "n": 3,
                                 "champion": [0.42, 0.42, 0.42],
                                 "candidate": [0.50, 0.43, 0.55],
                                 "deltas": [0.08, 0.01, 0.13],
                                 "mean_delta": 0.0733, "ci95_halfwidth": 0.09,
                                 "verdict": "inconclusive"}},
        "hard_gate": {"line_recall": None, "inference_ms_p95": 18.5},
        "decision": {"decision": "rejected",
                     "reasons": ["line_recall: UNKNOWN (hard gate needs a "
                                 "measurement)", "road_iou: inconclusive"]},
        "steps_by_arm": {"baseline": {"42": 15}, "candidate": {"42": 15}},
        "equal_steps_requested": True, "max_train_frames": 20,
    }, ensure_ascii=False), encoding="utf-8")

    st = dash.merge_event_streams(run)
    assert st["readable"] and st["n_events"] == 7, st
    assert st["last"].phase == "rejected"
    assert len(st["streams"]) == 2, "run 级 + 候选各一条"

    ctx = dash.build_context(type("A", (), {
        "out": str(tmp_path / "d.html"), "run_dir": str(run),
        "events": None, "manifest": None, "eval": None, "ident": None,
        "t13_import": None, "probes": None, "tasks": None,
        "champion": "baseline", "pack": None, "decisions": None})())
    html = dash.render_html(ctx)
    assert "判定与成对比较" in html
    assert "road_iou" in html and "0.0733" in html
    assert "未测" in html, "硬门里的 null 必须渲染成未测"
    assert "inconclusive" in html and "rejected" in html and "cand-r0" in html
    # 外部资源禁令在看板上仍然成立
    assert "http://" not in html and "https://" not in html
