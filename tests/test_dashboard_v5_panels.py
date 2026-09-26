"""S4/T16 七面板验收测试（离线：无 GPU、无模型、无游戏、无网络）。

方案 v2 §S4 的数据流是「原始帧/计数 → 评价结果 → 判定 JSON → 看板」，T16 的
验收是「判定文件到看板，包含 UNKNOWN/N/A/旧协议/坏场景：页面实际显示一致；
不补零、不吞坏场景、可追溯原帧」。本文件用 ``tmp_path`` 造合成 run 目录，
钉住四类确定性反例：

* v5 判定（counts/scene_counts/per_scene/hard_by_seed/negative_line）：
  七个面板都渲染，且每个值都标出**来源字段**（可追溯）；
* 一个 ``unknown`` 适用性场景：渲染成"未测（UNKNOWN）"，**不能**出现比率
  0（R=0 是"没有可判的样本"，不是 0 分）；
* 一个 ``not_applicable``（确认真无线）场景：与 UNKNOWN 分开显示，且它不
  参与最差场景；
* 一个坏场景（line_recall 很低）：坏场景不能被 champion/candidate 的池化
  读数吞掉，硬门违反里要能直接看到它；
* 一个 pre-v5 旧判定（没有 counts/scene_counts/negative_line）：新口径字段
  必须写"旧协议未记录"（原因来自 ``gates.LEGACY_REPLAY_NOTE``），不补 0；
* 没有判定文件时：七个面板都写"无数据 + 原因"，不画任何数字。

另有一条渲染纪律：面板只引用**相对本 HTML 真实存在**的文件（原帧/叠图），
不存在的路径写 missing，不伪造可点击链接。

测试全部走进程内 ``dash.main(["render", ...])``，不训练、不连游戏。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m5_seg_dashboard as dash
from beamng_autopilot.experiments.events import Event, EventLog, metric

FRAME_NAME = "frame_0001.npz"
OVERLAY_NAMES = ("truth_0001.png", "production_0001.png", "candidate_0001.png")


# ---------------------------------------------------------------------------
# 造样本：一个合成 run 目录（判定 + 准入门产物 + 事件流 + 训练历史 + 原帧）
# ---------------------------------------------------------------------------
def _v5_decision(frame: Path, overlays: dict) -> dict:
    """v5 判定文件：坏场景 + UNKNOWN 场景 + 不适用场景 + 负例诊断。"""
    return {
        "candidate_id": "cand-v5",
        "git_commit": "abc1234def5678",
        "git_dirty": ["scripts/x.py"],
        "dataset_id": "ds_v5_train_0001",
        "device": "cuda:0 (test-card)",
        "protocol": {"version": "t14-protocol-v5", "hash": "protohash0001",
                     "thresholds": {"line_recall_min": 0.7}},
        "thresholds": {"config_hash": "thr0001",
                       "source": "logs/thresholds_v4.json"},
        "model_path": "logs/m5_seg/cand/seed42/best.pt",
        "model_sha16": "deadbeefcafe0001",
        "epochs": 12,
        "candidate_epochs": 12,
        "data_counts": {"generated": 640, "reviewed": 136},
        "n_train_frames_by_arm": {"baseline": {"42": 20},
                                  "candidate": {"42": 24}},
        "steps_by_arm": {"baseline": {"42": 15}, "candidate": {"42": 18}},
        "factor": {"add_runs": ["logs/m5_seg/x"]},
        "applied_flags": ["add_runs"],
        "skipped_factors": {},
        "data_factor_note": {"applied": ["add_runs"], "not_applied": []},
        "eval_checkpoint": "checkpoint_last.pt",
        # v5 计数契约：先累加整数、再算比率（micro）
        "counts": {"P_frames": 2, "C": 10, "R": 8, "M": 6, "L": 4, "A": 3,
                   "C_outside_P": 10},
        "scene_counts": {
            "italy/ring_a": {"P_frames": 1, "C": 8, "R": 8, "M": 6, "L": 4,
                             "A": 3, "C_outside_P": 0},
            "italy/ring_b": {"P_frames": 1, "C": 2, "R": 0, "M": 0, "L": 0,
                             "A": 0, "C_outside_P": 0},
            "italy/ring_negative": {"P_frames": 0, "C": 0, "R": 0, "M": 0,
                                    "L": 0, "A": 0, "C_outside_P": 10},
            "italy/ring_unverified": {"P_frames": 0, "C": 0, "R": 0, "M": 0,
                                      "L": 0, "A": 0, "C_outside_P": 3},
        },
        "scene_applicability": {
            "italy/ring_a": {"status": "measured", "why": ""},
            "italy/ring_b": {
                "status": "unknown",
                "why": ("line truth exists but no candidate has a usable "
                        "reference (R=0)")},
            "italy/ring_negative": {
                "status": "not_applicable",
                "why": "confirmed line-free scene"},
            "italy/ring_unverified": {
                "status": "unverified_labels",
                "why": ("label rank 'agent' is not verified: counts are "
                        "diagnostic only")},
        },
        "scene_candidates": {
            "italy/ring_a": {"candidate_reference_coverage": 1.0,
                             "candidate_identity_rate": 0.75,
                             "left_right_role_agreement": 0.75,
                             "n_candidates": 8},
            "italy/ring_b": {"candidate_reference_coverage": 0.0,
                             "candidate_identity_rate": None,
                             "left_right_role_agreement": None,
                             "n_candidates": 2},
        },
        "per_scene": {
            "italy/ring_a": {"line_recall": 0.91, "line_precision": 0.88,
                             "offroad_false_ratio": 0.01,
                             "inference_ms_p95": 18.0},
            "italy/ring_b": {"line_recall": 0.12, "line_precision": 0.20,
                             "offroad_false_ratio": 0.55,
                             "inference_ms_p95": 19.0},
            "italy/ring_negative": {"line_recall": None},
        },
        "hard_by_seed": {
            "42": {"candidate_identity_rate": 0.75, "line_recall": 0.90,
                   "line_precision": 0.85, "offroad_false_ratio": 0.02,
                   "inference_ms_p95": 18.0},
            "43": {"candidate_identity_rate": 0.10, "line_recall": 0.20,
                   "line_precision": 0.15, "offroad_false_ratio": 0.90,
                   "inference_ms_p95": 19.0},
            # seed 44：身份率/漏线未测（UNKNOWN），line_precision 是实测 0
            "44": {"candidate_identity_rate": None, "line_recall": None,
                   "line_precision": 0.0, "offroad_false_ratio": 0.02,
                   "inference_ms_p95": 17.5},
        },
        # 同 seed 两臂（形状二：每 seed 一张指标表）
        "champ_by_seed": {
            "42": {"road_iou": 0.9069, "line_recall": 0.90},
            "43": {"road_iou": 0.4200, "line_recall": 0.20},
        },
        "cand_by_seed": {
            "42": {"road_iou": 0.8944, "line_recall": 0.88},
            "43": {"road_iou": 0.4100, "line_recall": 0.19},
        },
        "hard_gate": {"line_recall": None, "line_precision": 0.85,
                      "candidate_identity_rate": 0.75,
                      "inference_ms_p95": 18.4},
        "hard_gate_violations": ["scene italy/ring_b: line_recall 0.12 < 0.7"],
        "missing_metrics": ["line_precision: UNKNOWN (hard gate needs a "
                            "measurement)"],
        "negative_line": {
            "frames": 43, "eligible_frames": 40, "clean_frames": 38,
            "false_positive_frames": 2, "false_positive_px": 120,
            "eligible_px": 400000, "positive_frames": 0, "unknown_frames": 0,
            "empty_frames": 0, "unverified_frames": 3,
            "unverified_pred_line_px": 9, "status": "measured",
            "false_positive_frame_rate": 0.05,
            "false_positive_pixel_fraction": 0.0003,
            "excluded_frames": 3,
            "excluded_reason": ("3 frame(s) excluded: label rank is not "
                                "verified, so an all-zero label does not "
                                "confirm 'no line'")},
        "decision": {"decision": "rejected",
                     "reasons": ["scene italy/ring_b: line_recall 0.12 < 0.7",
                                 "no primary metric shows a credible "
                                 "improvement over the champion"]},
        "research_only": True,
        "r2_confirmed": False,
        "worst_frames_by_seed": {"42": [{
            "frame": str(frame), "iou": 0.12, "gt_px": 5000,
            "mask": "line truth 5000 px", "candidate": "3 candidates",
            "projection": "1 matched", "role": "left"}]},
        "overlays": {str(frame): {
            "truth": overlays["truth"], "production": overlays["production"],
            "candidate": overlays["candidate"]}},
        "resources": {"gpu": "RTX-TEST 24 GB", "cpu": "16 cores",
                      "mem": "32 GB", "disk": "100 GB free"},
        "stage_durations": {"training": 120.0, "evaluating": 30.0},
        "timing_repeats": [{"seed": 42, "p50": [8.1, 8.4],
                            "p95": [18.0, 18.4], "used": 18.0}],
        "timing_suspect": [],
    }


def _legacy_decision() -> dict:
    """pre-v5 判定：没有 counts/scene_counts/negative_line（不许补 0）。"""
    return {
        "candidate_id": "cand-legacy",
        "pairings": {"road_iou": {
            "metric": "road_iou", "n": 2,
            "champion": [0.42, 0.44], "candidate": [0.50, 0.43],
            "mean_delta": 0.035, "ci95_halfwidth": 0.09,
            "verdict": "inconclusive"}},
        "hard_gate": {"line_recall": None, "inference_ms_p95": 22.7},
        "decision": {"decision": "needs_evidence",
                     "reasons": ["line_recall: not measured"]},
        "hard_by_seed": {"42": {"line_recall": 0.9, "line_precision": 0.6}},
        "worst_frames_by_seed": {"42": [{"frame": "logs/m5_seg/gone.npz",
                                         "iou": 0.2, "gt_px": 100}]},
        "per_scene": {"italy/ring_a": {"line_recall": 0.9}},
        "steps_by_arm": {"baseline": {"42": 15}, "candidate": {"42": 15}},
    }


def _write_run(tmp_path: Path, *, name: str = "run_v5",
               negative_line: bool = True) -> Path:
    """合成 run 目录：v5 判定 + 准入门产物 + 事件流 + 训练历史 + 原帧/叠图。"""
    run = tmp_path / name
    (run / "round0" / "seed42").mkdir(parents=True)
    # 真 checkpoint（可被 torch.load(weights_only=True) 读回）：避免用一个坏文件
    # 触发 torch 的错误文案（那会把文档 URL 带进页面，污染"无外部资源"断言）。
    import torch
    torch.save({"state_dict": {}, "dataset_id": "ds_v5_train_0001",
                "git_commit": "abc1234def5678",
                "train_args": {"arch": "SegUNet", "n_params": 1, "batch": 4,
                               "epochs": 12, "lr": 0.001,
                               "device_name": "cuda:0 (test-card)"}},
               run / "round0" / "seed42" / "checkpoint_last.pt")
    (run / "frames").mkdir()
    frame = run / "frames" / FRAME_NAME
    frame.write_bytes(b"npz-bytes")
    (run / "overlays").mkdir()
    overlays = {}
    for key, fname in (("truth", OVERLAY_NAMES[0]),
                       ("production", OVERLAY_NAMES[1]),
                       ("candidate", OVERLAY_NAMES[2])):
        p = run / "overlays" / fname
        p.write_bytes(b"png")
        overlays[key] = str(p)
    # 训练历史（train vs dev loss 的直读来源）
    (run / "armA_seed42").mkdir()
    (run / "armA_seed42" / "train_hist.json").write_text(json.dumps({
        "epoch": [0, 1, 2],
        "train_loss": [1.9, 1.5, 1.2],
        "val_line_iou": [0.21, 0.30, 0.41],
    }), encoding="utf-8")
    dec = _v5_decision(frame, overlays)
    if not negative_line:
        dec.pop("negative_line")
    (run / "decision_v5.json").write_text(
        json.dumps(dec, ensure_ascii=False), encoding="utf-8")
    # 数据准入门产物：去重/冲突/已见组/空间 UNKNOWN
    (run / "rounds_dataset.json").write_text(json.dumps({
        "dataset_id": "ds_v5_train", "dev_dataset_id": "ds_v5_dev",
        "train_groups": ["italy/ring_a"], "dev_groups": ["italy/ring_b"],
        "group_overlap": [],
        "coverage": {"n_frames": 44, "trainable_frames": 44,
                     "paint_valid_frames": 0},
        "spatial": {"buffer_m": 50.0, "violations": [],
                    "n_pairs_checked": 120, "min_distance_m": 61.5,
                    "n_missing_position": 2},
        "exposure_leak": [],
        "rejected": [{"path": "logs/m5_seg/a/frame_0009.npz",
                      "reason": "byte-identical to frame_0001.npz in a"}],
    }, ensure_ascii=False), encoding="utf-8")
    # 事件流：一个资源指标 + 一个暂停原因（阶段耗时/暂停原因面板的来源）
    log = EventLog(run)
    base = dict(run_id=name, candidate_id="cand-v5",
                dataset_id="ds_v5_train_0001", config_hash="cfg", seed=42)
    log.append(Event(phase="training", status="running", epoch=0, step=7,
                     ts="2026-09-26T10:00:00Z",
                     metrics={"train_loss": metric(1.5),
                              "val_line_iou": metric(0.30),
                              "gpu_mem_gb": metric(11.0)}, **base))
    log.append(Event(phase="paused", status="paused", seq=9,
                     ts="2026-09-26T10:05:00Z",
                     note="user is using the machine: paused by configuration",
                     **base))
    return run


def _write_legacy_run(tmp_path: Path) -> Path:
    run = tmp_path / "run_legacy"
    run.mkdir()
    (run / "decision_legacy.json").write_text(
        json.dumps(_legacy_decision(), ensure_ascii=False), encoding="utf-8")
    return run


def _write_eval(tmp_path: Path, run: Path) -> Path:
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"dev": {"armA_seed42/best.pt": {
        "model": "logs/m5_seg/armA_seed42/best.pt",
        "sha256_16": "aa11bb22cc33dd44", "n_frames": 25,
        "road_iou": 0.44, "inference_ms_p95": 18.0}}}), encoding="utf-8")
    return p


def _render(tmp_path: Path, run: Path, *args: str,
            view: str = "full") -> str:
    out = tmp_path / "dash.html"
    argv = ["render", "--out", str(out), "--run-dir", str(run),
            "--view", view, *args]
    assert dash.main(argv) == 0
    return out.read_text(encoding="utf-8")


def _section(html: str, heading: str) -> str:
    m = re.search(r"<h2>" + re.escape(heading) + r"</h2>", html)
    assert m, f"section {heading!r} not found"
    nxt = re.search(r"<h2>", html[m.end():])
    return html[m.end(): m.end() + (nxt.start() if nxt else len(html))]


def _v5_section(html: str) -> str:
    return _section(html, "方案 v2 判定看板（T16 七面板）")


def _text(html: str) -> str:
    import html as _h
    return _h.unescape(re.sub(r"<[^>]+>", " ", html))


# ---------------------------------------------------------------------------
# 1) 七个面板都在，并且每个值可追溯回来源字段
# ---------------------------------------------------------------------------
def test_v5_panels_render_and_every_value_names_its_source(tmp_path):
    run = _write_run(tmp_path)
    html = _render(tmp_path, run, "--eval", str(_write_eval(tmp_path, run)))
    v5 = _v5_section(html)
    text = _text(v5)
    for panel in ("本轮身份", "数据入口", "学习过程", "比较结果", "无线诊断",
                  "错误定位", "资源与判定"):
        assert panel in text, panel
    # 面板 1：身份
    for needle in ("abc1234def5678", "个未提交路径", "ds_v5_train_0001",
                   "t14-protocol-v5", "protohash0001", "thr0001",
                   "logs/m5_seg/cand/seed42/best.pt", "deadbeefcafe0001",
                   "cuda:0 (test-card)"):
        assert needle in text, needle
    # 面板 2：数据入口四计数 + 去重/冲突/rank/已见组/空间
    for needle in ("640", "136", "seed 42", "24", "25 帧",
                   "去重隔离 1 帧（byte-identical）", "组重叠",
                   "italy/ring_a", "帧没有位置", "research_only"):
        assert needle in text, needle
    # 面板 3：学习过程
    for needle in ("epoch 12", "train_loss", "val_line_iou", "add_runs",
                   "checkpoint_last.pt", "120.0"):
        assert needle in text, needle
    assert "来源: decision_v5.json: train_hist" not in text  # 来源写的是文件名+指标
    # 面板 4：同 seed 两臂、micro/macro、C/R/M/L/A
    for needle in ("同 seed 两臂", "micro candidate_reference_coverage",
                   "macro candidate_reference_coverage", "80.00%", "75.00%",
                   "50.00%", "P_frames", "C_outside_P",
                   "每场景一个单位", "n_units=2"):
        assert needle in text, needle
    # 面板 5：无线诊断（含 T10 的档位不可信排除量）
    for needle in ("诊断，不直接晋级", "5.00%", "（2/40）", "0.03%",
                   "（120/400000）", "档位不可信 3"):
        assert needle in text, needle
    # 面板 6：错误定位（可点击原帧 + 掩码/候选/投影/角色）
    assert f'href="run_v5/frames/{FRAME_NAME}"' in v5, v5[:2000]
    for needle in ("line truth 5000 px", "3 candidates", "1 matched", "left"):
        assert needle in text, needle
    # 面板 7：资源与判定（硬门与研究结论分开；计时重复/可信度可见）
    for needle in ("RTX-TEST 24 GB", "硬门（晋级门", "研究结论（不直接晋级）",
                   "user is using the machine", "计时重复（p50/p95 各测两次）",
                   "计时可信度", "8.1", "18.4", "采用"):
        assert needle in text, needle
    # 可追溯性：每个面板都能看到"来源:"字样
    assert text.count("来源:") >= 20, text.count("来源:")


# ---------------------------------------------------------------------------
# 2) UNKNOWN 不能渲染成 0；not_applicable 与 UNKNOWN 分开
# ---------------------------------------------------------------------------
def test_unknown_applicability_is_not_a_zero_and_na_is_separate(tmp_path):
    run = _write_run(tmp_path)
    html = _render(tmp_path, run, "--eval", str(_write_eval(tmp_path, run)))
    v5 = _v5_section(html)
    text = _text(v5)

    def _row(scene: str) -> str:
        m = re.search(r'<tr><td class="mono">' + re.escape(scene)
                      + r"</td><td>(.*?)</td>", v5, re.S)
        assert m, v5[:3000]
        return m.group(1)

    # 四个适用性取值都要在页面上可区分（协议原文 + 人读解释）
    assert "measured" in _row("italy/ring_a")
    assert "实测" in _row("italy/ring_a")
    # ring_b：R=0 -> UNKNOWN，必须写"未测"并说明不是 0，且那一格不能有比率
    cell_b = _row("italy/ring_b")
    assert "unknown" in cell_b and "未测" in cell_b and "不是 0" in cell_b, \
        cell_b
    assert "%" not in cell_b, "UNKNOWN 那一格不能出现任何比率（更不许 0）"
    # not_applicable：确认真无线，单独一档，不是未测也不是通过
    cell_n = _row("italy/ring_negative")
    assert "not_applicable" in cell_n and "不适用" in cell_n, cell_n
    assert "未测" not in cell_n, "不适用不许写成未测"
    assert "通过" not in cell_n or "不算通过" in cell_n, cell_n
    assert "%" not in cell_n, "不适用那一格没有比率"
    # unverified_labels：档位不可信 -> 只许诊断，不得当结论
    cell_u = _row("italy/ring_unverified")
    assert "unverified_labels" in cell_u and "仅诊断" in cell_u, cell_u
    assert "%" not in cell_u
    # 未知场景不参与最差场景（否则 0/未测会被当成"最差"）
    worst = re.search(r"最差场景.*?</p>", v5, re.S)
    assert worst, v5[:3000]
    assert "italy/ring_b" in _text(worst.group(0))
    assert "italy/ring_negative" not in _text(worst.group(0))
    # hard_gate 里的 null 渲染成"未测（UNKNOWN）"，不是 0
    assert "硬门输入未测（UNKNOWN）" in text, text[:3000]
    # 逐 seed：未测写"该 seed 未测（不是 0）"，实测 0 才是 0
    assert "该 seed 未测（不是 0）" in text, "UNKNOWN 的 seed 值不能写成 0"
    assert "0.00%" in text, "实测 0（line_precision）应照实显示"


def test_missing_negative_line_in_a_v5_decision_is_no_data_not_zero(tmp_path):
    """v5 判定但没写 negative_line：是"无数据"，不是 0 假线率。"""
    run = _write_run(tmp_path, negative_line=False)
    html = _render(tmp_path, run, "--eval", str(_write_eval(tmp_path, run)))
    v5 = _v5_section(html)
    neg = v5.split("<h3>无线诊断</h3>", 1)[1].split("<h3>错误定位", 1)[0]
    text = _text(neg)
    assert "无数据" in text and "negative_line" in text, text
    assert "%" not in text, "没有记录时不许出现任何假比率"


# ---------------------------------------------------------------------------
# 3) 坏场景不能被池化读数吞掉
# ---------------------------------------------------------------------------
def test_a_bad_scene_is_visible_next_to_good_pooled_numbers(tmp_path):
    run = _write_run(tmp_path)
    html = _render(tmp_path, run, "--eval", str(_write_eval(tmp_path, run)))
    v5 = _v5_section(html)
    text = _text(v5)
    # 池化/micro 读数很好看（80% 覆盖、75% 身份），坏场景仍必须同时可见
    assert "80.00%" in text and "75.00%" in text
    assert "italy/ring_b" in text
    assert "12.00%" in text, "坏场景的 line_recall 0.12 必须原样显示"
    assert "scene italy/ring_b: line_recall 0.12 < 0.7" in text
    # 逐 seed 也得能看见坏 seed 43
    assert "seed 43" in text
    assert "90.00%" in text and "20.00%" in text


# ---------------------------------------------------------------------------
# 4) 旧协议：新计数写"旧协议未记录"，绝不补 0
# ---------------------------------------------------------------------------
def test_legacy_decision_shows_old_protocol_not_zeros(tmp_path):
    run = _write_legacy_run(tmp_path)
    html = _render(tmp_path, run)
    v5 = _v5_section(html)
    text = _text(v5)
    assert "旧协议未记录" in text
    # 计数面板：不渲染任何 C/R/M/L/A 数值表，只写旧协议未记录 + 原因
    cmp_block = v5.split("<h3>比较结果</h3>", 1)[1].split(
        "<h3>无线诊断</h3>", 1)[0]
    cmp_text = _text(cmp_block)
    assert "旧协议未记录" in cmp_text, cmp_text
    assert "cannot be re-judged under the new denominators" in text, text
    assert "C_outside_P" not in cmp_text, "旧记录不许出现新计数字段"
    # 负例诊断：旧记录同样写旧协议未记录
    neg = _text(v5.split("<h3>无线诊断</h3>", 1)[1].split(
        "<h3>错误定位", 1)[0])
    assert "旧协议未记录" in neg, neg
    # 旧记录里真实存在的字段照常显示（配对数/硬门 null/最差帧）
    assert "road_iou" in text and "0.035" in text
    assert "硬门输入未测（UNKNOWN）" in text
    assert "logs/m5_seg/gone.npz" in text
    assert "missing: logs/m5_seg/gone.npz" in text, "不存在的原帧不能用链接假装"


# ---------------------------------------------------------------------------
# 5) 没有判定文件：七个面板都写原因
# ---------------------------------------------------------------------------
def test_no_decision_file_renders_reasons_not_numbers(tmp_path):
    empty = tmp_path / "run_empty"
    empty.mkdir()
    html = _render(tmp_path, empty)
    v5 = _v5_section(html)
    text = _text(v5)
    for panel in ("本轮身份", "数据入口", "学习过程", "比较结果", "无线诊断",
                  "错误定位", "资源与判定"):
        assert panel in text, panel
    assert "无数据" in text and "没有判定文件" in text
    assert "旧协议未记录：" not in text, "没有判定文件时不许把记录说成旧协议"


# ---------------------------------------------------------------------------
# 6) 默认（compact）视图也必须带七个面板
# ---------------------------------------------------------------------------
def test_compact_default_view_carries_the_v5_panels(tmp_path):
    run = _write_run(tmp_path)
    out = tmp_path / "compact.html"
    assert dash.main(["render", "--out", str(out), "--run-dir", str(run),
                      "--eval", str(_write_eval(tmp_path, run))]) == 0
    html = out.read_text(encoding="utf-8")
    text = _text(html)
    assert "方案 v2 判定看板（T16 七面板）" in text
    for panel in ("本轮身份", "数据入口", "学习过程", "比较结果", "无线诊断",
                  "错误定位", "资源与判定"):
        assert panel in text, panel
    assert "80.00%" in text and "诊断，不直接晋级" in text


# ---------------------------------------------------------------------------
# 7) 看板自包含：不得引用任何外部资源（离线验收环境）
# ---------------------------------------------------------------------------
def test_v5_page_is_self_contained_and_references_no_external_resources(
        tmp_path):
    run = _write_run(tmp_path)
    html = _render(tmp_path, run, "--eval", str(_write_eval(tmp_path, run)))
    for needle in ("http://", "https://", "<script src", "<link ", "@import"):
        assert needle not in html, f"external resource reference: {needle}"
