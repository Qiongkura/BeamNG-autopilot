"""探针的可见性契约：缺相机 UNKNOWN、失败帧数与 P 帧口径。

方案 v2 §3.3–§3.5 与验收 T09/T11 的三件事（**本文件是这批新测试的唯一落点**；
``test_marking_identity_probe_counts.py`` 保持不变，继续管计数完备性）：

1. **缺相机 -> 逐帧 UNKNOWN**（T09/§3.4）：不是"处理过但为空"、更不是
   "无线"；这些帧不进 ``rows``/``frames_processed``/任何计数或 p50，逐帧
   条目在 ``unknowns`` 里可查，条目仍记真值有没有漆线像素（缺证据不许被
   改写成"无线"后排除，§3.3）。整 run 无相机时给出结构化 ``reason`` +
   真实帧数，绝不返回 frames=0 的"成功" summary。
2. **T11 失败可见**：探针自己装配失败、单帧失败、缺 ``counts`` 的帧都必须
   显式可见；帧数守恒 ``frames_total == frames_processed + frames_skipped +
   frames_unknown``；缺失计数绝不当 0 相加——"没测"与"实测 0"必须可区分。
3. **P_frames 只有一个口径**（协议 v5：真值**明确有漆线像素**，逐帧判断）：
   "引擎线能链成线"是另一个量，落在 ``P_frame_chained`` /
   ``counts_chained_engine_line`` / ``frames_with_chained_engine_line`` 上，
   不再覆盖 ``counts["P_frames"]``。真值有漆线却链不成线的帧**仍是 P 帧**，
   它的候选进覆盖率分母 ``C``（§3.3：真值有漆线却投影/链线失败属于缺证据，
   不能改成"无线"排除，否则覆盖率分母会只剩容易帧）。

确定性做法与计数测试一致：**不加载真实模型**（``HydraNet``/``SemanticHead``
用假对象替换），相机与像素由合成相机自己投影生成（几何与 oracle 同口径，
见 ``test_marking_identity_probe_counts.py::TestFixtureGeometry``）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import geometry as G  # noqa: E402

#: 合成相机：与 test_tool_contract / 计数测试的 meta 同构，front_main 前视。
META = {"cameras": {"front_main": {
    "offset": [0.0, 1.5, 1.4], "fwd": [0.0, 1.0, 0.0],
    "up": [0.0, 0.0, 1.0], "fov_deg": 65.0, "width": 536, "height": 403}}}

H, W = 403, 536
#: 地面在车原点下方这么多（oracle 自己的平面口径），合成像素必须用它。
GROUND_GAP = float(G.EGO_GROUND_GAP_M)


def _load(name: str, rel: Path):
    spec = importlib.util.spec_from_file_location(name, rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load("_m5_ident_probe_visibility",
                 ROOT / "scripts/m5_marking_identity_probe.py")


@pytest.fixture(scope="module")
def cam(probe):
    c = probe.camera_from_meta(META, "front_main")
    assert c is not None
    return c


def _ground_pixels(cam, lat_m: float, *, fwd0=6.0, fwd1=16.0, step=0.25):
    """前方地面一段等横向偏移的线 -> 像素 (u, v)，用作候选像素。"""
    fs = np.arange(float(fwd0), float(fwd1) + 1e-9, float(step))
    world = np.column_stack([fs, np.full(len(fs), float(lat_m)),
                             np.full(len(fs), -GROUND_GAP)])
    u, v, valid = cam.project(world, np.zeros(3), 0.0)
    return np.column_stack([u[valid], v[valid]])


def _line_mask(cam, lat_m: float, *, fwd0=4.0, fwd1=18.0, step=0.1):
    """引擎 line 类掩码：一段足长漆线（可被 engine_lines 链成一条线）。"""
    px = _ground_pixels(cam, lat_m, fwd0=fwd0, fwd1=fwd1, step=step)
    m = np.zeros((H, W), dtype=bool)
    for u, v in px:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < H and 0 <= ui < W:
            m[vi, ui] = True
            if ui + 1 < W:
                m[vi, ui + 1] = True
    return m


def _unchained_line_mask(cam):
    """有真值漆线**像素**但链不成线的掩码（覆盖率分母的边界用例）。

    三段实心漆线落在相邻的 forward bin（4–5.95 / 6–7.95 / 8–9.95 m），
    但横向每段跳 2.5 m > ``LINE_CHAIN_STEP_M + LINE_BIN_M *
    LINE_SLOPE_MAX`` = 2.0 m，且每段只覆盖 1 个 bin < ``LINE_MIN_BINS``：
    ``engine_lines`` 返回空（链不成线），而像素全在图像左半（engine_px_left
    >= ``MIN_REF_PX``，候选在该侧有可用参考）。
    """
    m = np.zeros((H, W), dtype=bool)
    for k in range(3):
        px = _ground_pixels(cam, 2.5 + 2.5 * k, fwd0=4.0 + 2.0 * k,
                            fwd1=5.95 + 2.0 * k, step=0.02)
        for u, v in px:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < H and 0 <= ui < W:
                m[vi, ui] = True
                if ui + 1 < W:
                    m[vi, ui + 1] = True
    return m


class _Marking:
    """假 LaneMarking：probe 只读 pixels/kind/color/meta 四个字段。"""

    def __init__(self, pixels, *, kind="solid", color="white", meta=None):
        self.pixels = np.asarray(pixels, dtype=float)
        self.kind = kind
        self.color = color
        self.meta = {} if meta is None else dict(meta)


def _mark(cam, lat_m: float, *, kind="solid", colour="white", learned=None):
    meta = {} if learned is None else {"learned_frac": float(learned)}
    return _Marking(_ground_pixels(cam, lat_m), kind=kind, color=colour,
                    meta=meta)


class _TaskOut:
    """假 TaskOutput：probe 只读 masks["line"] 与 meta["markings"]。"""

    def __init__(self, line=None, markings=()):
        self.masks = {} if line is None else {"line": np.asarray(line, bool)}
        self.meta = {"markings": list(markings)}


def _fake_net(monkeypatch, probe, outputs):
    """把 probe 的 HydraNet/SemanticHead 换成按顺序吐预置输出的假对象。"""

    class _FakeNet:
        def __init__(self, *_a, **_k):
            self._outputs = list(outputs)
            self.errors: dict = {}

        def add(self, head):
            self._head = head

        def run(self, ctx):
            # 预置输出用完还调用 = 这个帧本不该跑 head（例如缺相机）
            item = self._outputs.pop(0)
            return {} if item is None else {"semantic": item}

    monkeypatch.setattr(probe, "HydraNet", _FakeNet)
    monkeypatch.setattr(probe, "SemanticHead", lambda *a, **k: object())


def _write_frame(run_dir: Path, idx: int, colour, label) -> Path:
    p = run_dir / f"frame_{idx:05d}.npz"
    np.savez_compressed(p, colour=colour, label=label)
    return p


def _rgb():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _p_frame(probe, cam, *, engine_lat=2.0):
    """一帧真值：引擎漆线在 engine_lat（左正），返回 uint8 label。"""
    label = np.zeros((H, W), dtype=np.uint8)
    label[_line_mask(cam, engine_lat)] = probe.CLS_LINE
    return label


def _unchained_p_frame(probe, cam):
    """一帧真值：有漆线像素、但引擎线链不成线（缺证据边界）。"""
    label = np.zeros((H, W), dtype=np.uint8)
    label[_unchained_line_mask(cam)] = probe.CLS_LINE
    return label


class TestCameraMissingIsUnknown:
    """T09/§3.4：缺相机 = UNKNOWN，不是"处理过但为空"，也不是"无线"。"""

    def test_a_run_without_a_camera_model_is_refused_as_per_frame_unknown(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))       # 真值有漆线
        _write_frame(run, 1, _rgb(), np.zeros((H, W), np.uint8))  # 真值无线
        # outputs=[]：缺相机的帧绝不该跑 head；跑了会 IndexError -> errors
        _fake_net(monkeypatch, probe, [])
        res = probe.probe(run, None, view="front_main")
        assert "summary" not in res, \
            "缺相机不许返回'测过但为空'的成功 summary（T09/§3.4）"
        assert "reason" in res and "UNKNOWN" in res["reason"]
        assert "no camera model" in res["reason"] and "front_main" in res["reason"]
        # 真实帧数：没有一帧算"处理成功"；UNKNOWN 与失败分开记账
        assert res["frames_processed"] == 0 and res["frames_total"] == 2
        assert res["frames_unknown"] == 2
        assert res["frames_skipped"] == 0 and res["n_errors"] == 0
        assert res["camera_model_used"] is False
        assert "counts" not in res, "UNKNOWN 不产出计数：缺失不是 0"
        entries = res["unknowns"]
        assert [e["frame"] for e in entries] == \
            ["frame_00000.npz", "frame_00001.npz"]
        assert all(e["status"] == "unknown" and e["stage"] == "camera"
                   for e in entries), entries
        # 真值有没有漆线像素照记：缺相机 ≠ 无线（§3.3 不许改成"无线"排除）
        assert entries[0]["truth_has_line_px"] is True
        assert entries[0]["n_engine_px"] > 0
        assert entries[1]["truth_has_line_px"] is False

    def test_a_meta_without_the_requested_view_is_unknown_too(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [])
        res = probe.probe(run, META, view="side_left")
        assert "summary" not in res and "UNKNOWN" in res["reason"]
        assert res["frames_unknown"] == 1 and res["frames_processed"] == 0
        assert "side_left" in res["unknowns"][0]["reason"]

    def test_a_labelled_frame_with_a_camera_is_measured_normally(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0)])])
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        assert s["frames_processed"] == 1 and s["frames_with_counts"] == 1
        assert s["frames_unknown"] == 0 and s["unknown_frames"] == []
        assert s["frames_total"] == 1
        assert s["counts"]["P_frames"] == 1 and s["counts"]["C"] == 1
        assert res["rows"][0]["status"] == "measured"

    def test_the_label_check_still_wins_over_the_camera_check(
            self, probe, monkeypatch, tmp_path):
        """顺序护栏：没有 label 的 run 仍报"缺 label"，不是被相机检查抢先。"""
        run = tmp_path / "front_main"
        run.mkdir()
        np.savez_compressed(run / "frame_00000.npz", colour=_rgb())
        _fake_net(monkeypatch, probe, [])
        res = probe.probe(run, None, view="front_main")
        assert "no label array" in res["reason"]
        assert "no camera model" not in res["reason"]
        # 拒绝也带真实帧数与空账目：processed 字段不会缺省（T11）
        assert res["frames_processed"] == 0 and res["frames_total"] == 1
        assert res["frames_skipped"] == 0 and res["frames_unknown"] == 0


class TestFailureVisibility:
    """T11：探针异常/缺 counts 必须显式可见，帧数守恒，缺失不当 0 相加。"""

    def test_every_refusal_carries_the_real_frame_counts(self, probe, tmp_path):
        """T11：连"没有帧可测"的拒绝也带 processed/skipped/unknown/total，
        绝不出现"processed 字段缺失 -> 消费方默认 frames=0 -> 报成功"。"""
        missing = probe.probe(tmp_path / "does_not_exist", None)
        empty = probe.probe(tmp_path, None)
        for res in (missing, empty):
            assert "reason" in res and "summary" not in res
            assert res["frames_processed"] == 0
            assert res["frames_skipped"] == 0
            assert res["frames_unknown"] == 0
            assert res["frames_total"] == 0
            assert res["errors"] == [] and res["unknowns"] == []

    def test_a_probe_setup_failure_is_structured_not_a_zero_frame_success(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))

        class _BrokenNet:
            def __init__(self, *_a, **_k):
                raise RuntimeError("cuda oom while loading weights")

        monkeypatch.setattr(probe, "HydraNet", _BrokenNet)
        res = probe.probe(run, META, view="front_main")
        assert "summary" not in res, \
            "装配失败不许返回 frames=0 的成功 summary（T11）"
        assert "setup failed" in res["reason"] and "cuda oom" in res["reason"]
        # 真实处理量：0 帧测过，帧总数仍可见；不假装成功、不假装有 error
        assert res["frames_processed"] == 0 and res["frames_total"] == 1
        assert res["frames_skipped"] == 0 and res["n_errors"] == 0

    def test_failed_frames_keep_the_real_processed_count_and_closed_accounting(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        good = _p_frame(probe, cam)
        _write_frame(run, 0, _rgb(), good)
        # 形状不匹配的 label：单帧测量必然抛错（T11 反例）
        _write_frame(run, 1, _rgb(), np.zeros((20, 20), dtype=np.uint8))
        _write_frame(run, 2, _rgb(), good)
        marks = [_mark(cam, 2.0, learned=1.0)]
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool), marks),
            _TaskOut(np.zeros((H, W), bool), marks)])
        s = probe.probe(run, META, view="front_main")["summary"]
        assert s["frames_processed"] == 2 and s["frames_with_counts"] == 2
        assert s["frames_skipped"] == 1 and s["frames_unknown"] == 0
        assert s["frames_total"] == 3
        assert s["frames_total"] == (s["frames_processed"] + s["frames_skipped"]
                                     + s["frames_unknown"]), \
            "帧数必须守恒：processed + skipped + unknown == total"
        assert s["counts"]["P_frames"] == 2 and s["counts"]["C"] == 2, \
            "失败帧既没被算 0，也没把分母撑成 3"

    def test_a_missing_count_is_never_summed_as_a_measured_zero(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _write_frame(run, 1, _rgb(), _p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0)]),
            _TaskOut(None, []),                       # head 没给线掩码
        ])
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        assert s["frames_processed"] == 2
        assert s["frames_with_counts"] == 1, "没掩码的帧没有计数，不许补 0"
        assert s["frames_no_line_mask"] == 1 and s["frames_unknown"] == 0
        assert s["frames_skipped"] == 0 and s["n_errors"] == 0
        # 缺失帧对任何计数零贡献（不是"0 票"）：总量 == 唯一测到的那帧
        assert s["counts"] == {"P_frames": 1, "C": 1, "C_outside_P": 0,
                               "R": 1, "M": 1, "L": 1, "A": 1}, s["counts"]
        assert s["candidates_total"] == 1
        # 逐帧可区分：measured（有 counts）vs no_line_mask（无 counts 键）
        assert res["rows"][0]["status"] == "measured"
        assert "counts" in res["rows"][0]
        assert res["rows"][1]["status"] == "no_line_mask"
        assert "counts" not in res["rows"][1], "缺失不是 measured 0"
        assert "counts" not in res["unknowns"] and s["frames_unknown"] == 0

    def test_a_measured_zero_count_is_still_a_measured_zero(
            self, probe, cam, monkeypatch, tmp_path):
        """有线但空预测：counts 存在且为 0 -> 实测 0，与"缺失"可区分。"""
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [_TaskOut(np.zeros((H, W), bool), [])])
        s = probe.probe(run, META, view="front_main")["summary"]
        assert s["frames_with_counts"] == 1, "有线但空预测仍是实测（0），不是缺测"
        assert s["counts"]["P_frames"] == 1 and s["counts"]["C"] == 0
        assert s["counts"]["R"] == 0 and s["counts"]["M"] == 0
        assert s["candidates_total"] == 0
        # 缺失帧会是全 0 的另一个来源，所以 frames_with_counts 是区分键
        assert s["frames_no_line_mask"] == 0 and s["frames_unknown"] == 0


class TestPFrameSemantics:
    """P_frames 只有一个口径：真值明确有漆线**像素**（逐帧，协议 v5）。

    "引擎线能链成线"另立显式键。真值有漆线却链不成线的帧**仍是 P 帧**，
    它的候选进覆盖率分母 C——§3.3 明确：真值有漆线却投影/链线失败属于缺证据，
    不能改成"无线"后排除（否则覆盖率分母只剩容易帧）。链成线的严格口径在
    ``counts_chained_engine_line`` 里单独给，两个量不同键、可比但不混用。
    """

    def _two_frame_run(self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _write_frame(run, 1, _rgb(), _unchained_p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0)]),
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0),      # 左侧：该侧有参考
                      _mark(cam, -3.0, kind="thin",      # 右侧：真值无参考
                            learned=0.2)]),
        ])
        return probe.probe(run, META, view="front_main")

    def test_an_unchained_truth_line_is_a_p_frame_not_a_phantom(
            self, probe, cam, monkeypatch, tmp_path):
        res = self._two_frame_run(probe, cam, monkeypatch, tmp_path)
        s = res["summary"]
        # 链不成线的那帧真值仍有漆线像素 -> 契约 P 帧（不是 C_outside_P）
        assert res["rows"][0]["P_frame"] is True
        assert res["rows"][0]["P_frame_chained"] is True
        assert res["rows"][1]["P_frame"] is True
        assert res["rows"][1]["P_frame_chained"] is False
        assert res["rows"][1]["counts"]["P_frames"] == 1
        assert res["rows"][1]["counts"]["C_outside_P"] == 0, \
            "有真漆线像素的帧候选不许被当成'无线帧候选'排除（§3.3）"
        # 两个量两个键：主口径含链不成线的帧，诊断口径不含
        assert s["counts"]["P_frames"] == 2
        assert s["counts"]["C"] == 3 and s["counts"]["C_outside_P"] == 0
        assert s["counts"]["R"] == 2 and s["counts"]["M"] == 1
        assert s["counts_chained_engine_line"]["P_frames"] == 1
        assert s["counts_chained_engine_line"]["C"] == 1
        assert s["counts_chained_engine_line"]["C_outside_P"] == 2
        assert s["frames_with_engine_line"] == 2, \
            "frames_with_engine_line 与 counts['P_frames'] 同一个量"
        assert s["counts"]["P_frames"] == s["frames_with_engine_line"]
        assert s["frames_with_chained_engine_line"] == 1
        assert s["frames_truth_line_unchained"] == 1
        assert res["rows"][1]["counts_chained_engine_line"]["P_frames"] == 0
        assert res["rows"][1]["counts_chained_engine_line"]["C_outside_P"] == 2

    def test_coverage_uses_the_contract_p_set_not_the_chained_subset(
            self, probe, cam, monkeypatch, tmp_path):
        res = self._two_frame_run(probe, cam, monkeypatch, tmp_path)
        s = res["summary"]
        # 覆盖率 = R/C，分母是**契约 P 集**（含链不成线帧的候选）：
        # 2/3；链成线的严格子集是 1/1。两个口径必须落在不同键上。
        assert s["counts"]["C"] == 3 and s["counts"]["R"] == 2
        cov = s["counts"]["R"] / s["counts"]["C"]
        assert cov == pytest.approx(2 / 3, abs=1e-4)
        chained = s["counts_chained_engine_line"]
        assert chained["C"] == 1
        assert chained["R"] / chained["C"] == 1.0
        assert cov != pytest.approx(chained["R"] / chained["C"]), \
            "两个分母不同键、不同值：不许把链成线口径冒充覆盖率主口径"
        # 来源分解与主口径自洽（Σ各来源 == counts['C']）
        assert s["candidate_sources"]["total"] == s["counts"]["C"] == 3

    def test_a_run_whose_only_truth_line_is_unchained_stays_visible(
            self, probe, cam, monkeypatch, tmp_path):
        """整 run 唯一真值线链不成线：不是 not_applicable/无线，而是可查的缺证据。"""
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _unchained_p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0)])])
        s = probe.probe(run, META, view="front_main")["summary"]
        assert s["counts"]["P_frames"] == 1, \
            "真值有漆线像素就是 P 帧；P_frames=0 会被下游当成'确认真无线'"
        assert s["counts"]["C"] == 1 and s["counts"]["R"] == 1
        assert s["counts"]["M"] == 0, "链不成线时 M 实测为 0（不是缺测）"
        assert s["counts_chained_engine_line"]["P_frames"] == 0
        assert s["frames_truth_line_unchained"] == 1
        assert s["frames_with_chained_engine_line"] == 0
        assert s["frames_total"] == 1 and s["frames_processed"] == 1
