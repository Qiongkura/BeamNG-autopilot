"""固定探针图的验收：可测量分档、哈希清单、UNKNOWN 灰显、目录一致性。

方案对探针图的要求是"图像和标签哈希固定、无真值区域灰显 UNKNOWN、预测颜色
不覆盖原图细节、每轮保存数量有上限"，这些都要能被测到，而不是靠看图。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tool():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_probe_images", ROOT / "scripts" / "m5_seg_probe_images.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_probe_images"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frame(tmp_path, name: str, *, line_px: int = 12, unknown: bool = False,
           base: int = 60) -> Path:
    """写一帧：line_px 控制标线像素数，unknown=True 时留一块 255 区域。"""
    d = tmp_path / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    colour = np.full((24, 32, 3), base, np.uint8)
    label = np.zeros((24, 32), np.uint8)
    label[8:20, :] = 1
    if line_px:
        label[14, :line_px] = 2
    if unknown:
        label[0:4, :] = 255
    np.savez(d / "frame_00000.npz", colour=colour, label=label)
    (tmp_path / name / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "label_source": "beamng_annotation (road dense; line class is NOT "
                        "provided by the game - see module doc)",
        "frames": [{"i": 0, "view": "front_main", "exposure": 0,
                    "t_wall": 1.0, "path": "front_main/frame_00000.npz"}],
    }), encoding="utf-8")
    return d


class TestSceneBands:
    def test_scene_is_decided_by_a_measurable_rule_with_its_number(self):
        tool = _tool()
        name, why = tool.scene_of(0.0001)
        assert name == "none" and "0.00010" in why.replace("0.0001", "0.00010")
        assert tool.scene_of(0.003)[0] == "faint"
        assert tool.scene_of(0.02)[0] == "dense"
        # 判据必须写进说明：别人要能复核"为什么这帧算退化线"
        for frac in (0.0, 0.001, 0.5):
            assert "line_frac=" in tool.scene_of(frac)[1]


class TestFrameMetrics:
    def test_unknown_pixels_never_count_as_errors(self):
        tool = _tool()
        label = np.full((6, 6), 255, np.uint8)
        label[0, :2] = 1                      # 已知路面
        pred = np.zeros((6, 6), bool)
        pred[3:, :] = True                    # 只在 UNKNOWN 区"预测"
        m = tool.frame_metrics(pred, label)
        assert m["fp_px"] == 0 and m["fn_px"] == 0,             "UNKNOWN 区的预测不算错（真值未知）"
        assert m["unknown_px"] == 34
        assert m["iou"] is None, "没有可比像素时 IoU 是 None，不是 0"
        # 但把线画在**已知路面**上就是假阳性——这条不能被 UNKNOWN 规则吞掉
        pred2 = np.zeros((6, 6), bool)
        pred2[0, 0] = True
        assert tool.frame_metrics(pred2, label)["fp_px"] == 1

    def test_offroad_false_line_is_reported_separately(self):
        tool = _tool()
        label = np.zeros((5, 5), np.uint8)
        label[0, 0] = 2
        pred = np.zeros((5, 5), bool)
        pred[0, 0] = True
        pred[4, 4] = True                     # 画在背景上
        m = tool.frame_metrics(pred, label)
        assert m["tp_px"] == 1 and m["offroad_false_line_px"] == 1
        assert m["precision"] == 0.5 and m["recall"] == 1.0


class TestPanel:
    def test_panel_draws_unknown_grey_and_prediction_as_outline(self, tmp_path):
        tool = _tool()
        colour = np.full((20, 30, 3), 80, np.uint8)
        label = np.zeros((20, 30), np.uint8)
        label[5:15, :] = 1
        label[10, :8] = 2
        label[0:3, :] = 255                   # UNKNOWN 区
        pred = np.zeros((20, 30), bool)
        pred[8:13, :8] = True                 # 2 维块：内部像素才是"内部"
        fig = tool.panel(colour, label, pred, title="t")
        # 直接从 figure 取像素：UNKNOWN 区必须是灰的（205），不是白也不是黑
        ax = fig.axes[2]                      # 可信标签面板
        img = ax.images[0].get_array()
        assert tuple(img[1, 1]) == (205, 205, 205), "UNKNOWN 要灰显"
        err = fig.axes[1].images[0].get_array()
        assert tuple(err[1, 1]) == (205, 205, 205), "误差图的 UNKNOWN 同样灰显"
        assert tuple(err[10, 0]) == (40, 170, 70), "命中为绿"
        # 预测只描边：内部像素不应被涂成纯红
        rgb_ax = fig.axes[0].images[0].get_array()
        inside = rgb_ax[10, 3]                # 块内部
        assert not (inside[0] > 240 and inside[1] < 30 and inside[2] < 30), \
            "预测不得铺满色块覆盖原图细节（只描边界）"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_edges_mark_the_mask_boundary(self):
        tool = _tool()
        m = np.zeros((5, 5), bool)
        m[1:4, 1:4] = True                    # 3x3 块：有真正的内部
        e = tool._edges(m)
        assert e[1, 1] and e[2, 1] and e[1, 2], "边界像素是边界"
        assert not e[2, 2], "内部像素不是边界"
        assert not e[0, 1] and not e[4, 2], "掩码外的像素不算边界"


class TestSelectionAndManifest:
    def test_selection_rotates_scenes_and_reports_missing_bands(self,
                                                               tmp_path):
        tool = _tool()
        a = _frame(tmp_path, "coll_none", line_px=0)
        b = _frame(tmp_path, "coll_dense", line_px=20, base=90)
        picked = tool.select_frames([a, b], max_frames=4)
        scenes = [p["scene"] for p in picked]
        assert "none" in scenes and "dense" in scenes
        assert len(picked) <= 4, "每轮保存数量必须有上限"

    def test_manifest_locks_hashes_and_matches_the_directory(self, tmp_path):
        tool = _tool()
        run = tmp_path / "exp"
        _frame(tmp_path, "coll_a", line_px=4)
        _frame(tmp_path, "coll_b", line_px=20, base=120)
        mf = tool.build(run, Path("logs/m5_seg/seg_model/best.pt"),
                        runs=[tmp_path / "coll_a" / "front_main",
                              tmp_path / "coll_b" / "front_main"],
                        max_frames=2, seg=_FakeSeg(), stages=False)
        d = run / "probes" / mf["checkpoint_id"]
        files = sorted(p.name for p in d.glob("probe_*.png"))
        assert files == [e["probe"] for e in mf["frames"]], \
            "目录里的图必须与清单一一对应"
        for e in mf["frames"]:
            assert e["frame_sha16"] and e["label_sha16"]
            assert e["label_source"], "标签来源必须记录"
            assert "metrics" in e and e["scene_why"]
        assert mf["max_frames"] == 2 and mf["n_frames"] <= 2
        assert mf["model_sha16"] and mf["postprocess"]["line_source"]
        assert "不复制" in mf["note"] or "not copied" in mf["note"]

    def test_a_second_run_cleans_stale_probe_images(self, tmp_path):
        tool = _tool()
        run = tmp_path / "exp"
        _frame(tmp_path, "coll_a", line_px=4)
        _frame(tmp_path, "coll_b", line_px=20, base=120)
        runs = [tmp_path / "coll_a" / "front_main",
                tmp_path / "coll_b" / "front_main"]
        model = Path("logs/m5_seg/seg_model/best.pt")
        first = tool.build(run, model, runs=runs, max_frames=2,
                           seg=_FakeSeg(), stages=False)
        d = run / "probes" / first["checkpoint_id"]
        assert len(list(d.glob("probe_*.png"))) >= 2
        again = tool.build(run, model, runs=[runs[1]], max_frames=1,
                           seg=_FakeSeg(), stages=False)
        files = sorted(p.name for p in d.glob("probe_*.png"))
        assert files == [e["probe"] for e in again["frames"]], \
            "重写同一 checkpoint 目录后不得留下清单没记录的旧图"


class _FakeSeg:
    """假推理：把"真值是标线"的像素预测成标线（用于验证管线而不是模型）。

    只实现 ``predict_with_probs``：这些用例用 ``build(stages=False)``，因为
    它们验证的是哈希清单与目录一致性，不是阶段对照。
    """

    def predict_with_probs(self, colour):
        lab = np.zeros(colour.shape[:2], bool)
        lab[10:15, :] = True
        return lab, lab, None


class _StageSeg:
    """假 segmenter：按固定规则产出四个阶段的掩码，用于验证管线而非模型。

    ``break_shape_filter`` 让 ``predict()`` 与 ``after_shape_filter`` 不一致，
    用来验证"两条链漂移必须报出来"。
    """

    route_is_dirt = False

    def __init__(self, *, break_shape_filter: bool = False, px: int = 6):
        self.break_shape_filter = break_shape_filter
        self.px = px

    def _infer_logits(self, colour):
        return np.zeros(colour.shape[:2], np.float32)

    def _argmax_masks(self, logits, colour):
        road = np.zeros(colour.shape[:2], bool)
        road[2:8, :] = True
        line = np.zeros(colour.shape[:2], bool)
        line[4, : self.px] = True
        return road, line

    def _morph_close_line(self, mask):
        m = np.asarray(mask, bool).copy()
        m[5, : self.px] = m[4, : self.px]        # 模拟闭运算加粗
        return m

    def predict(self, colour):
        """部署链：raw → morph → filter（与手工链相同），可选故意漂移。

        漂移要**确定性地**构造：早期版本只是加了一段孤立像素，结果被真实的
        ``filter_line_shape`` 当成碎块滤掉，两条链又一致了，于是交叉检查没抓到
        ——现在直接返回空掩码，保证与手工链不同。
        """
        from beamng_autopilot.vision.segmentation import filter_line_shape
        road, line = self._argmax_masks(None, colour)
        if self.break_shape_filter:
            # 漂移要用"过滤链不可能产出"的掩码：全 1 不可能是 12 px 细线的过滤
            # 结果。（先前用全零，结果与过滤后的空掩码相等 —— 断言反而失败。）
            return road, np.ones_like(line)
        return road, filter_line_shape(self._morph_close_line(line))


class TestPostprocessStages:
    """同一帧的四阶段对照：复用 stage_eval 的实现，并带一致性交叉检查。"""

    def test_stage_masks_follow_the_evaluator_path(self):
        tool = _tool()
        colour = np.full((10, 12, 3), 70, np.uint8)
        stages, ok = tool.stage_masks(_StageSeg(), colour)
        assert list(stages) == list(tool.POSTPROCESS_STAGES)
        assert ok is True, "两条链一致时交叉检查应为 True"
        # morph 比 raw 多一行：阶段差异必须能看出来
        d = tool.stage_diffs(stages)
        assert d["raw_argmax->after_morph_close"] == 6
        assert d["raw_argmax->full_predict"] >= 6

    def test_a_drifting_full_predict_is_reported_not_hidden(self):
        tool = _tool()
        colour = np.full((10, 12, 3), 70, np.uint8)
        stages, ok = tool.stage_masks(_StageSeg(break_shape_filter=True), colour)
        assert ok is False, "full_predict 与 after_shape_filter 不一致必须报 False"
        assert tool.stage_diffs(stages)["after_shape_filter->full_predict"] > 0
        # 反向：两条链一致时必须报 True（否则这个交叉检查只会永远报警）
        stages_ok, ok2 = tool.stage_masks(_StageSeg(), colour)
        assert ok2 is True and tool.stage_diffs(stages_ok)[
            "after_shape_filter->full_predict"] == 0

    def test_the_manifest_carries_stages_diffs_and_the_cross_check(self, tmp_path):
        tool = _tool()
        run = tmp_path / "exp"
        _frame(tmp_path, "coll_a", line_px=8)
        mf = tool.build(run, Path("logs/m5_seg/seg_model/best.pt"),
                        runs=[tmp_path / "coll_a" / "front_main"],
                        max_frames=1, seg=_StageSeg())
        entry = mf["frames"][0]
        assert set(entry["stages"]) == set(tool.POSTPROCESS_STAGES)
        for name, m in entry["stages"].items():
            assert "iou" in m, f"{name} 缺指标"
        assert entry["full_predict_matches_shape_filter"] is True
        assert entry["stage_diffs"]["raw_argmax->full_predict"] >= 0
        pp = mf["postprocess"]
        assert pp["stages_compared"] == list(tool.POSTPROCESS_STAGES)
        assert "m5_seg_stage_eval" in pp["shared_with"], \
            "清单要写明阶段函数与阶段评估器共用"

    def test_the_panel_draws_both_rows_when_stages_are_given(self):
        tool = _tool()
        colour = np.full((20, 30, 3), 80, np.uint8)
        label = np.zeros((20, 30), np.uint8)
        label[5:15, :] = 1
        label[10, :8] = 2
        pred = np.zeros((20, 30), bool)
        pred[10, :8] = True
        stages = {name: pred.copy() for name in tool.POSTPROCESS_STAGES}
        fig = tool.panel(colour, label, pred, title="t", stages=stages)
        assert len(fig.axes) == 8, "上排四张 + 下排四个阶段"
        titles = [ax.get_title() for ax in fig.axes]
        assert any("raw argmax" in t and "px=" in t for t in titles)
        import matplotlib.pyplot as plt
        plt.close(fig)
