"""分割训练数据工具（load_frames 密度过滤 / per-run 验证划分）纯逻辑回归。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_TRAIN_PATH = (Path(__file__).resolve().parent.parent /
               "scripts" / "m5_train_seg.py")
_spec = importlib.util.spec_from_file_location("m5_train_seg", _TRAIN_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def _write_frame(run_dir: Path, name: str, line_px: int) -> None:
    """写一张 8x8 合成帧：line 像素数可指定，其余为路面。"""
    colour = np.full((8, 8, 3), 90, dtype=np.uint8)
    label = np.ones((8, 8), dtype=np.uint8)
    label.ravel()[:line_px] = 2
    np.savez(run_dir / name, colour=colour, label=label)


def test_load_frames_filters_sparse(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    for i in range(3):
        _write_frame(run_a, f"frame_{i:05d}.npz", line_px=8)   # 8/64
    for i in range(2):
        _write_frame(run_b, f"frame_{i:05d}.npz", line_px=1)   # 1/64

    frames, per_run = _mod.load_frames([run_a, run_b], min_line_frac=0.05)
    assert len(frames) == 3                       # 稀疏 run_b 全部被过滤
    # keys are path-unique (basenames collide across ring collections)
    ka, kb = _mod._run_key(run_a), _mod._run_key(run_b)
    assert per_run[ka]["kept"] == 3
    assert per_run[kb]["kept"] == 0
    assert per_run[kb]["line_px_frac"] == pytest.approx(1 / 64, abs=1e-6)


def test_split_frames_per_run_keeps_each_tail():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {
        "a": {"kept": 10, "start": 0, "end": 10},
        "b": {"kept": 10, "start": 10, "end": 20},
    }
    tr, va = _mod.split_frames(frames, per_run, "per-run", 0.2)
    assert len(tr) == 16 and len(va) == 4
    # 每个 run 各取尾部 20%：a 的最后 2 帧与 b 的最后 2 帧进验证
    assert va == [frames[8], frames[9], frames[18], frames[19]]
    assert tr == frames[:8] + frames[10:18]


def test_split_frames_tail_global():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(10)]
    tr, va = _mod.split_frames(frames, {}, "tail", 0.2)
    assert len(tr) == 8 and len(va) == 2
    assert va == frames[8:]


def test_iou_from_accum_no_inflation_for_absent_class():
    ious = _mod.iou_from_accum(
        np.array([10.0, 5.0, 0.0]), np.array([20.0, 10.0, 0.0]))
    assert ious[0] == pytest.approx(0.5)
    assert ious[1] == pytest.approx(0.5)
    assert ious[2] == 0.0          # 未出现类别：实数 0，而不是虚高 1.0


def test_balanced_indices_each_run_equal():
    rng = np.random.default_rng(7)
    idx = _mod.balanced_indices([(0, 5), (5, 15)], rng)
    assert len(idx) == 20                       # 每 run 补齐到最长 10 帧
    assert sum(0 <= j < 5 for j in idx) == 10   # run a：5 帧循环补齐到 10
    assert sum(5 <= j < 15 for j in idx) == 10  # run b：自身 10 帧
    assert len(np.unique(idx)) == 15            # 全部来源帧都被覆盖


def test_balanced_indices_deterministic():
    a = _mod.balanced_indices([(0, 5), (5, 15)],
                              np.random.default_rng(3))
    b = _mod.balanced_indices([(0, 5), (5, 15)],
                              np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_train_run_bounds_matches_per_run_split():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {"a": {"kept": 10, "start": 0, "end": 10},
               "b": {"kept": 10, "start": 10, "end": 20}}
    tr, _ = _mod.split_frames(frames, per_run, "per-run", 0.2)
    bounds = _mod.train_run_bounds(frames, per_run, "per-run", 0.2)
    assert bounds == [(0, 8), (8, 16)]
    assert len(tr) == 16


def test_train_run_bounds_tail_clips_last_run():
    frames = [(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8))
              for _ in range(20)]
    per_run = {"a": {"kept": 10, "start": 0, "end": 10},
               "b": {"kept": 10, "start": 10, "end": 20}}
    tr, va = _mod.split_frames(frames, per_run, "tail", 0.2)
    bounds = _mod.train_run_bounds(frames, per_run, "tail", 0.2)
    assert bounds == [(0, 10), (10, 16)]  # 全局尾部 4 帧进验证，b 被截断
    assert len(tr) == 16 and len(va) == 4


def test_default_model_path_follows_logs_dir(tmp_path, monkeypatch) -> None:
    """Which checkpoint is deployed must be resolvable, not assumed.

    An unpinned run (the mountain scenario) uses whatever this returns, so
    the resolution has to be a pure function of ``config.LOGS_DIR`` - and
    the choice is now logged, because the deployed default and the v13b/v8
    specialists disagree strongly per map.
    """
    from beamng_autopilot import config
    from beamng_autopilot.vision.segmentation import default_model_path

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    assert default_model_path() is None

    p = tmp_path / "m5_seg" / "seg_model" / "best.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"stub")
    assert default_model_path() == p


def test_segmenter_exposes_the_loaded_checkpoint(tmp_path) -> None:
    """The loaded path is recorded on the instance for attribution."""
    import torch
    from beamng_autopilot.vision.segmentation import Segmenter, SegUNet

    ckpt = tmp_path / "best.pt"
    model = SegUNet(n_classes=3)
    torch.save({"state_dict": model.state_dict(), "n_classes": 3,
                "class_names": ["background", "asphalt", "line"]},
               ckpt)
    seg = Segmenter(model_path=ckpt, device="cpu", use_half=False)
    assert seg.model_path == ckpt


class TestEvalMatrixMath:
    """像素层指标的数学是纯函数：先把它钉死，模型加载不在单测范围里。"""

    def _tool(self):
        import importlib.util
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "m5_seg_eval_matrix", root / "scripts" / "m5_seg_eval_matrix.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m5_seg_eval_matrix"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_line_metrics_count_offroad_and_missed_separately(self):
        import numpy as np
        tool = self._tool()
        label = np.zeros((4, 4), np.uint8)
        label[0, :] = 1                      # 路面
        label[1, 0] = 2                      # 真值标线
        label[2, 0] = 2                      # 真值标线（会漏）
        pred = np.zeros((4, 4), bool)
        pred[1, 0] = True                    # 命中
        pred[3, :2] = True                   # 画在背景上（路外假线）
        m = tool.line_pixel_metrics(pred, label)
        assert m["tp_px"] == 1 and m["fn_px"] == 1
        assert m["missed_true_line_px"] == 1
        assert m["offroad_false_line_px"] == 2
        assert m["pred_line_px"] == 3

    def test_ignore_pixels_never_count_as_errors(self):
        import numpy as np
        tool = self._tool()
        label = np.full((3, 3), 255, np.uint8)     # 整帧 unknown
        pred = np.ones((3, 3), bool)
        m = tool.line_pixel_metrics(pred, label)
        assert m["fp_px"] == 0 and m["fn_px"] == 0 and m["known_px"] == 0

    def test_totals_report_none_instead_of_zero_without_denominator(self):
        tool = self._tool()
        acc = {"tp_px": 0, "fp_px": 0, "fn_px": 0, "pred_line_px": 0,
               "gt_line_px": 0, "known_px": 100}
        out = tool.totals_to_metrics(acc, n_frames=1, ms=[3.0])
        assert out["line_precision"] is None and out["line_recall"] is None
        assert "no predicted line pixels" in out["line_precision_missing"]
        assert out["inference_ms_p50"] == 3.0

    def test_totals_are_global_not_average_of_frames(self):
        tool = self._tool()
        acc = {}
        tool.accumulate(acc, {"tp_px": 9, "fp_px": 1, "fn_px": 0,
                              "pred_line_px": 10, "gt_line_px": 9,
                              "known_px": 100})
        tool.accumulate(acc, {"tp_px": 1, "fp_px": 9, "fn_px": 0,
                              "pred_line_px": 10, "gt_line_px": 1,
                              "known_px": 100})
        out = tool.totals_to_metrics(acc, n_frames=2, ms=[1.0, 2.0])
        assert out["line_precision"] == 0.5, "全局累加 = 10/20，不是逐帧平均"
        assert out["inference_ms_p95"] >= 1.0

    def test_per_group_evaluation_keeps_each_scene_separate(self, tmp_path):
        """分场景评估：每个场景自己是一份独立测量（方案 §10.2/A7）。

        这里证明的是**接线**：两个场景各 2 帧，各自出自己那份完整指标
        （不是从池化结果里切一块），池化仍按总帧数算；没有真值可召回的
        场景记 UNKNOWN（None）而不是 0。
        "坏场景不被均值抵消"本身由 `gates.scene_report` 的判定测试覆盖。
        """
        import numpy as np
        import torch
        tool = self._tool()
        from beamng_autopilot.vision.segmentation import SegUNet
        ckpt = tmp_path / "per_group.pt"
        torch.save({"state_dict": SegUNet(width=1.0).state_dict(),
                    "train_args": {"arch_args": {"width": 1.0}}}, ckpt)

        def frames(hit: bool):
            out = []
            for i in range(2):
                colour = np.zeros((30, 40, 3), np.uint8)
                label = np.zeros((30, 40), np.uint8)
                label[6:26, :] = 1
                label[15, :20] = 2
                if not hit:
                    label[:, :] = np.where(label == 2, 1, label)  # 场景 B 无标线
                out.append((f"f{i}.npz", colour, label))
            return out

        per_group = {"italy/ring_a": frames(True), "italy/ring_b": frames(False)}
        out = tool.evaluate_model_per_group(ckpt, per_group, device="cpu")
        assert set(out["per_group"]) == {"italy/ring_a", "italy/ring_b"}, out["per_group"]
        # 每个场景自己那一份都是完整指标（不是"总体里切出来的一个数"）
        for g in out["per_group"]:
            assert out["per_group"][g]["n_frames"] == 2, out["per_group"][g]
            assert "line_recall" in out["per_group"][g]
        # 场景 B 的真值标线被去掉 -> 它自己"没有真值标线可召回"（UNKNOWN 而非 0）
        assert out["per_group"]["italy/ring_b"]["line_recall"] is None,             out["per_group"]["italy/ring_b"]
        assert out["n_frames"] == 4, "总体仍按池化算"

    def test_model_argument_accepts_name_equals_path(self):
        from pathlib import Path
        tool = self._tool()
        name, path = tool.parse_model_arg("armA=logs/x/best.pt")
        assert name == "armA" and Path(path).name == "best.pt"
        name, path = tool.parse_model_arg("logs/y/best.pt")
        assert name == "best"
        assert Path(path).name == "best.pt" and Path(path).parent.name == "y"


class TestCheckpointDiff:
    """T14 阶段 B：逐位比较两个 checkpoint（续训 ≈ 未中断 的验收工具）。"""

    def _tool(self):
        import importlib.util
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "m5_seg_checkpoint_diff",
            root / "scripts" / "m5_seg_checkpoint_diff.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m5_seg_checkpoint_diff"] = mod
        spec.loader.exec_module(mod)
        return mod

    def _save(self, path, value, *, dataset_id="ds1", next_epoch=4):
        import torch
        torch.save({"state_dict": {"w": torch.full((3,), float(value))},
                    "optimizer": {}, "scheduler": {}, "next_epoch": next_epoch,
                    "dataset_id": dataset_id}, path)
        return path

    def test_identical_checkpoints_are_equal_and_differences_are_located(
            self, tmp_path):
        tool = self._tool()
        a = self._save(tmp_path / "a.pt", 1.0)
        b = self._save(tmp_path / "b.pt", 1.0)
        rep = tool.compare(a, b)
        assert rep["equal"] is True and rep["weights"]["n_diff"] == 0
        assert rep["a_dataset_id"] == rep["b_dataset_id"] == "ds1"
        c = self._save(tmp_path / "c.pt", 1.5)
        rep2 = tool.compare(a, c)
        assert rep2["equal"] is False and rep2["weights"]["n_diff"] == 1
        assert rep2["weights"]["max_key"] == "w"

    def test_missing_resume_fields_are_reported_not_assumed(
            self, tmp_path):
        import torch
        tool = self._tool()
        legacy = tmp_path / "legacy.pt"
        torch.save({"state_dict": {"w": torch.zeros(2)}, "optimizer": {},
                    "scheduler": {}, "next_epoch": 2}, legacy)
        rep = tool.compare(legacy, self._save(tmp_path / "b.pt", 0.0))
        assert "torch_rng" in rep["a_missing_extras"]
        assert "dataset_id" in rep["a_missing_extras"]


class TestResumeTolerance:
    """GPU 续训容差的统计是纯函数，先把它钉死，再谈实测数字。"""

    def _tool(self):
        import importlib.util
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "m5_seg_resume_tolerance",
            root / "scripts" / "m5_seg_resume_tolerance.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m5_seg_resume_tolerance"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_summary_reports_the_distribution_and_the_ratio(self):
        tool = self._tool()
        pairs = {
            "control": [{"max_abs_diff": 0.039, "n_diff": 92, "n_compared": 106,
                         "max_rel_diff": 0.0033},
                        {"max_abs_diff": 0.0014, "n_diff": 92,
                         "n_compared": 106, "max_rel_diff": 0.0001}],
            "resume": [{"max_abs_diff": 0.034, "n_diff": 92, "n_compared": 106,
                        "max_rel_diff": 0.0029},
                       {"max_abs_diff": 0.016, "n_diff": 92, "n_compared": 106,
                        "max_rel_diff": 0.0014}],
        }
        s = tool.summarize(pairs)
        assert s["control"]["max_abs_diff_max"] == 0.039
        assert s["control"]["max_abs_diff_median"] == (0.039 + 0.0014) / 2
        assert s["resume"]["max_abs_diff_max"] == 0.034
        assert s["tolerance"]["ratio"] == pytest.approx(0.034 / 0.039, rel=1e-6)
        assert s["control"]["n_diff_frac_max"] == pytest.approx(92 / 106)

    def test_an_unmeasured_group_is_missing_not_zero(self):
        tool = self._tool()
        s = tool.summarize({"control": [], "resume": []})
        assert s["control"]["n"] == 0 and "missing" in s["control"]
        assert "tolerance" not in s, "没有两组数据就不能给比值"


class TestThresholdProtocol:
    """冻结阈值：哈希对不上就是被改过；续训容差要有判据方法。"""

    def test_the_newest_version_wins_and_tampering_is_refused(self, tmp_path):
        import importlib.util
        import json as _json
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "m5_seg_autoloop_t", root / "scripts" / "m5_seg_autoloop.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m5_seg_autoloop_t"] = mod
        spec.loader.exec_module(mod)
        from beamng_autopilot.experiments.gates import Thresholds

        # 新版本号优先
        (tmp_path / "t14_thresholds.json").write_text("{}", encoding="utf-8")
        v2 = tmp_path / "t14_thresholds_v2.json"
        t = Thresholds()
        v2.write_text(_json.dumps({
            "thresholds": {**{k: v for k, v in
                              __import__("dataclasses").asdict(t).items()}},
            "config_hash": t.config_hash}), encoding="utf-8")
        mod.THRESHOLDS_DIR = tmp_path
        assert mod.newest_thresholds_file().name == "t14_thresholds_v2.json"
        assert mod.thresholds().config_hash == t.config_hash

        # 手改一个阈值 -> 哈希不符 -> 拒绝
        blob = _json.loads(v2.read_text(encoding="utf-8"))
        blob["thresholds"]["line_recall_min"] = 0.1
        bad = tmp_path / "t14_thresholds_v3.json"
        bad.write_text(_json.dumps(blob), encoding="utf-8")
        with pytest.raises(ValueError) as err:
            mod.thresholds(bad)
        assert "改过" in str(err.value)

    def test_the_resume_tolerance_has_a_verdict_helper(self):
        from beamng_autopilot.experiments.gates import Thresholds
        t = Thresholds()
        assert t.resume_max_rel_diff >= t.resume_control_max_rel_diff, \
            "容差必须不小于实测噪声，否则判据自相矛盾"
        ok = t.resume_within_tolerance(t.resume_control_max_rel_diff)
        assert ok["within_tolerance"] is True
        assert "measured" in ok["basis"]
        assert t.resume_within_tolerance(t.resume_max_rel_diff * 5)[
            "within_tolerance"] is False


class TestE0Baseline:
    """E0 基线工具：最差场景的方向不能搞反（越小越好的指标取最大值）。"""

    def _tool(self):
        import importlib.util
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "m5_e0_baseline", root / "scripts" / "m5_e0_baseline.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m5_e0_baseline"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_worst_scene_takes_the_right_direction(self):
        tool = self._tool()
        seeds = [
            {"seed": 42, "eval": {"per_scene": {
                "a": {"line_recall": 0.9, "offroad_false_frac_of_pred": 0.02},
                "b": {"line_recall": 0.3, "offroad_false_frac_of_pred": 0.7}}}},
            {"seed": 43, "eval": {"per_scene": {
                "a": {"line_recall": 0.5, "offroad_false_frac_of_pred": 0.01}}}},
        ]
        w = tool.worst_scene_table(seeds)
        # 召回越低越差 -> 取最小
        assert w["line_recall"]["value"] == 0.3 and w["line_recall"]["scene"] == "b"
        # 路外假线比例越高越差 -> 取**最大**（实测踩到：0.0018 被报成最差）
        assert w["offroad_false_frac_of_pred"]["value"] == 0.7, w
        assert w["offroad_false_frac_of_pred"]["lower_is_better"] is True
        # 缺测不参与（不能把"没测"当成最差或最好）
        seeds[0]["eval"]["per_scene"]["b"]["line_precision"] = None
        assert "line_precision" not in tool.worst_scene_table(seeds)
