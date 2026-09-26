"""候选探针的计数完备性与失败可见性（方案 v2 §3.3 的 S2 验收）。

三件事：

1. 计数不变量：``M ≤ R ≤ C``、``A ≤ L ≤ M``，且同一帧的候选只进 ``C`` 或
   ``C_outside_P`` 中的一个口径（互斥）；
2. T11 型反例：单帧处理失败必须留下结构化 ``errors``/``n_errors``/
   ``frames_skipped``，被跳过的帧名可见，且该帧的计数**绝不以 0 混进
   成功帧**（失败帧不进 ``rows``，也不把 ``frames_processed`` 撑大）；
3. 候选来源分解必须存在、且与 ``counts["C"]`` 自洽（各来源之和 = C）：
   经典 CV / 虚线恢复 / 黄色先验这些"困难来源"一个都删不得。

确定性做法：**不加载真实模型**——``probe()`` 里的 ``HydraNet`` /
``SemanticHead`` 用假对象替换（monkeypatch），逐帧输出预置；相机与像素由
测试按 ``CameraModel`` 自己投影生成（合成相机可往返 oracle，见
``TestFixtureGeometry``）。计数与逐候选标志写在同一循环里（有意为之，
避免公式重复实现），所以计数部分只能通过跑完整 ``probe()`` 来测；来源
分解/合并与错误条目是纯函数，直接调用。真实 ``SemanticHead``/``detect_lines``
产出的候选分布、以及真实采集帧上的 oracle 误差不在本文件覆盖范围内。
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

#: 合成相机：与 test_tool_contract 的 meta 同构，front_main 前视。
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
    return _load("_m5_ident_probe_counts",
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


class _HeadFailure:
    """模拟 HydraNet 吞掉的 head 异常：run 返回 {} 但 net.errors 有记录。"""

    def __init__(self, message):
        self.message = str(message)


def _fake_net(monkeypatch, probe, outputs):
    """把 probe 的 HydraNet/SemanticHead 换成按顺序吐预置输出的假对象。"""

    class _FakeNet:
        def __init__(self, *_a, **_k):
            self._outputs = list(outputs)
            self.errors: dict = {}

        def add(self, head):
            self._head = head

        def run(self, ctx):
            item = self._outputs.pop(0)
            if isinstance(item, _HeadFailure):
                # 真实 HydraNet 正是这样：异常进 errors，run 本身不抛
                self.errors["semantic"] = item.message
                return {}
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


class TestFixtureGeometry:
    """合成相机必须与探针用的 oracle 同一套地面口径，否则下面全是假绿。"""

    def test_the_synthetic_camera_round_trips_through_the_oracle(self, probe,
                                                                 cam):
        for lat in (2.0, -3.0):
            px = _ground_pixels(cam, lat, fwd0=6.0, fwd1=6.0, step=1.0)
            assert px.shape == (1, 2)
            hit = probe.oracle_ground_point(float(px[0, 0]), float(px[0, 1]),
                                            cam)
            assert hit is not None, "合成像素必须打在地面上"
            assert abs(float(hit[0]) - 6.0) < 0.05
            assert abs(float(hit[1]) - lat) < 0.05

    def test_the_synthetic_engine_line_is_chained_and_refs_both_signs(
            self, probe, cam):
        lines = probe.engine_lines(_line_mask(cam, 2.0), cam)
        assert len(lines) == 1 and lines[0]["role"] == "near_left"
        assert abs(lines[0]["lat_m"] - 2.0) < 0.1
        cm = probe.compare_masks(_line_mask(cam, 2.0), _line_mask(cam, 2.0))
        assert cm["engine_px_left"] > probe.MIN_REF_PX
        assert cm["engine_px_right"] == 0, "左线不得出现在右半图"


class TestSourceBreakdown:
    """来源分解是纯函数：只计数，不删任何来源。"""

    def test_every_kind_colour_and_arm_is_counted_and_sums_match(self, probe):
        cands = [
            {"kind": "solid", "colour": "white", "learned_frac": 1.0},
            {"kind": "thin", "colour": "white", "learned_frac": 0.2},
            {"kind": "dashed", "colour": "white", "learned_frac": None},
            {"kind": "unknown", "colour": "yellow", "learned_frac": None},
            {"kind": "zigzag", "colour": "rainbow"},      # 未知取值也不许丢
        ]
        b = probe.candidate_source_breakdown(cands)
        assert b["total"] == 5
        assert sum(b["by_kind"].values()) == 5
        assert sum(b["by_colour"].values()) == 5
        assert sum(b["by_arm"].values()) == 5
        assert b["by_kind"]["zigzag"] == 1, "未知 kind 必须原样可查"
        assert b["by_colour"]["rainbow"] == 1
        assert b["by_arm"] == {"learned": 1, "cv_only": 1, "unattributed": 3}
        # 交叉口径：虚线恢复/黄色先验是无溯源臂的子集，不参与求和
        assert b["dashed_recovery"] == 1 and b["yellow_classic"] == 1

    def test_the_arm_threshold_is_the_frozen_half(self, probe):
        assert probe.candidate_source_of({"learned_frac": 0.5}) == "learned"
        assert probe.candidate_source_of({"learned_frac": 0.49}) == "cv_only"
        assert probe.candidate_source_of({"learned_frac": None}) \
            == "unattributed"
        # 坏取值不吞候选：仍然可查（归 unattributed），计数不为负/不消失
        assert probe.candidate_source_of({"learned_frac": "oops"}) \
            == "unattributed"

    def test_merging_adds_integers_and_keeps_the_frozen_arms(self, probe):
        b1 = probe.candidate_source_breakdown(
            [{"kind": "solid", "colour": "white", "learned_frac": 1.0}])
        b2 = probe.candidate_source_breakdown(
            [{"kind": "dashed", "colour": "yellow"}])
        m = probe.merge_source_breakdowns([b1, b2, None, {}])
        assert m["total"] == 2
        assert m["by_kind"] == {"solid": 1, "dashed": 1}
        assert m["by_colour"] == {"white": 1, "yellow": 1}
        assert m["by_arm"]["learned"] == 1
        assert m["by_arm"]["unattributed"] == 1
        assert set(probe.SOURCE_ARMS) <= set(m["by_arm"])
        assert m["dashed_recovery"] == 1 and m["yellow_classic"] == 1

    def test_a_frame_error_entry_carries_name_type_message_and_stage(self,
                                                                    probe):
        e = probe.frame_error_entry("frame_00007.npz", 7,
                                    ValueError("bad shape"))
        assert e["frame"] == "frame_00007.npz" and e["index"] == 7
        assert e["error"] == "ValueError" and "bad shape" in e["message"]
        assert e["skipped"] is True and e["stage"] == "measure"
        e2 = probe.frame_error_entry("f.npz", 0, OSError("disk full"),
                                     stage="overlay")
        assert e2["skipped"] is False and e2["stage"] == "overlay"


class TestCountInvariants:
    """逐帧计数只进 C 或 C_outside_P 中的一个；M⊆R⊆C、A⊆L⊆M。"""

    def test_m_le_r_le_c_and_a_le_l_le_m_and_p_frame_exclusivity(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        # 第二帧：真值没有任何漆线 -> 它的候选只能进 C_outside_P
        _write_frame(run, 1, _rgb(), np.zeros((H, W), dtype=np.uint8))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0),        # 左侧，有参考
                      _mark(cam, -3.0, kind="thin", learned=0.2)]),
            _TaskOut(np.zeros((H, W), bool), [_mark(cam, -3.0)]),
        ])
        res = probe.probe(run, META, view="front_main")
        rows = res["rows"]
        assert len(rows) == 2
        p, n = rows[0]["counts"], rows[1]["counts"]
        assert 0 <= p["M"] <= p["R"] <= p["C"], p
        assert 0 <= p["A"] <= p["L"] <= p["M"], p
        assert p["P_frames"] == 1 and p["C"] == 2 and p["C_outside_P"] == 0
        # 右侧候选的"该侧无参考"是 UNKNOWN，不得混进 R/M
        assert p["R"] == 1 and p["M"] == 1 and p["L"] == 1 and p["A"] == 1, p
        # 互斥：同一帧的候选只进一个口径
        assert min(p["C"], p["C_outside_P"]) == 0
        assert n["P_frames"] == 0 and n["C"] == 0 and n["C_outside_P"] == 1, n
        assert min(n["C"], n["C_outside_P"]) == 0
        s = res["summary"]
        assert s["counts"]["C"] == 2 and s["counts"]["C_outside_P"] == 1
        assert s["counts"]["P_frames"] == 1
        assert s["frames_processed"] == 2 and s["frames_skipped"] == 0
        assert s["n_errors"] == 0 and s["errors"] == []


class TestFailureVisibility:
    """T11 反例：失败必须留痕，且不得把失败帧当成 0 计数的成功帧。"""

    def test_t11_a_failed_frame_is_visible_and_never_counted_as_zero(
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
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        assert s["frames_processed"] == 2 and s["frames_with_counts"] == 2
        assert s["frames_skipped"] == 1 and s["n_errors"] == 1
        assert s["skipped_frames"] == ["frame_00001.npz"], s["skipped_frames"]
        assert [r["frame"] for r in res["rows"]] == [0, 2], \
            "失败帧不得进 rows（计数是'无'，不是 0）"
        # 计数只含两个成功帧：失败帧既没被算 0，也没把分母撑成 3
        assert s["counts"]["P_frames"] == 2
        assert s["counts"]["C"] == 2 and s["counts"]["M"] == 2
        e = s["errors"][0]
        assert e["frame"] == "frame_00001.npz" and e["index"] == 1
        assert e["error"] == "ValueError" and "does not match" in e["message"]
        assert e["skipped"] is True
        assert res["errors"] == s["errors"], "summary 内必须能查到全部错误"

    def test_an_all_failed_run_is_refused_with_its_errors(self, probe,
                                                          monkeypatch,
                                                          tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), np.zeros((20, 20), dtype=np.uint8))
        _write_frame(run, 1, _rgb(), np.zeros((30, 30), dtype=np.uint8))
        _fake_net(monkeypatch, probe, [])
        res = probe.probe(run, META, view="front_main")
        assert "summary" not in res, "全失败不许返回空 summary 冒充'测过'"
        assert "reason" in res and "could be measured" in res["reason"]
        assert res["n_errors"] == 2 and res["frames_skipped"] == 2
        assert [e["frame"] for e in res["errors"]] == \
            ["frame_00000.npz", "frame_00001.npz"]

    def test_a_swallowed_head_failure_is_not_reported_as_no_line_mask(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), np.zeros((H, W), dtype=np.uint8))
        _write_frame(run, 1, _rgb(), np.zeros((H, W), dtype=np.uint8))
        _fake_net(monkeypatch, probe, [
            _HeadFailure("cuda oom"),                 # head 异常被 HydraNet 吞
            _TaskOut(np.zeros((H, W), bool), []),
        ])
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        assert s["frames_skipped"] == 1 and s["n_errors"] == 1
        assert s["frames_no_line_mask"] == 0, \
            "head 失败不得伪装成'head 没给掩码'"
        assert "semantic head failed" in s["errors"][0]["message"]
        assert "cuda oom" in s["errors"][0]["message"]

    def test_a_head_without_a_line_mask_is_processed_not_an_error(
            self, probe, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), np.zeros((H, W), dtype=np.uint8))
        _fake_net(monkeypatch, probe, [_TaskOut(None, [])])
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        assert s["n_errors"] == 0 and s["frames_skipped"] == 0
        assert s["frames_processed"] == 1 and s["frames_no_line_mask"] == 1
        assert s["frames_with_counts"] == 0, "无掩码的帧没有计数，不许补 0"

    @pytest.mark.parametrize("broken", ["overlay_image", "write_overlay"])
    def test_a_decoration_failure_does_not_lose_the_measurement(
            self, probe, cam, monkeypatch, tmp_path, broken):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, 2.0, learned=1.0)])])

        def _boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(probe, broken, _boom)
        res = probe.probe(run, META, view="front_main",
                          overlay_dir=tmp_path / "ov")
        s = res["summary"]
        assert s["frames_processed"] == 1 and s["frames_with_counts"] == 1
        assert s["frames_skipped"] == 0, "装饰失败不许丢测量"
        assert s["n_errors"] == 1 and s["errors"][0]["stage"] == "overlay"
        assert s["errors"][0]["skipped"] is False


class TestUnreadableRunReasons:
    """'读不到' 与 '没有帧' 必须是两种不同的 reason（方案 v2 §3.3）。"""

    def test_a_missing_run_dir_says_unreadable_not_empty(self, probe,
                                                         tmp_path):
        res = probe.probe(tmp_path / "does_not_exist", None)
        assert "cannot read run dir" in res["reason"]
        assert "no frame_*.npz" not in res["reason"]

    def test_an_empty_dir_says_no_frames(self, probe, tmp_path):
        d = tmp_path / "front_main"
        d.mkdir()
        res = probe.probe(d, None)
        assert "no frame_*.npz" in res["reason"]

    def test_an_explicit_empty_frame_list_is_refused_too(self, probe,
                                                         tmp_path):
        d = tmp_path / "front_main"
        d.mkdir()
        res = probe.probe(d, None, frames=[])
        assert "no frame_*.npz" in res["reason"]


class TestSourcesInSummary:
    """来源分解要进 summary，且与整数计数自洽——困难来源不许被删。"""

    def test_candidate_sources_sum_to_C_and_hard_sources_stay_visible(
            self, probe, cam, monkeypatch, tmp_path):
        run = tmp_path / "front_main"
        run.mkdir()
        _write_frame(run, 0, _rgb(), _p_frame(probe, cam))
        _write_frame(run, 1, _rgb(), np.zeros((H, W), dtype=np.uint8))
        marks_p = [
            _mark(cam, 2.0, kind="solid", learned=1.0),          # learned
            _mark(cam, -3.0, kind="thin", learned=0.2),          # 经典 CV 边
            _mark(cam, -5.0, kind="dashed", learned=None),       # 虚线恢复链
            _mark(cam, 4.0, kind="thin", colour="yellow", learned=None),
        ]
        _fake_net(monkeypatch, probe, [
            _TaskOut(np.zeros((H, W), bool), marks_p),
            _TaskOut(np.zeros((H, W), bool),
                     [_mark(cam, -3.0, learned=0.2)]),
        ])
        res = probe.probe(run, META, view="front_main")
        s = res["summary"]
        src = s["candidate_sources"]
        # P 帧口径：各来源之和 = counts["C"]
        assert src["total"] == s["counts"]["C"] == 4
        assert sum(src["by_kind"].values()) == 4
        assert sum(src["by_colour"].values()) == 4
        assert sum(src["by_arm"].values()) == 4
        assert src["by_arm"] == {"learned": 1, "cv_only": 1,
                                 "unattributed": 2}, src
        assert src["dashed_recovery"] == 1 and src["yellow_classic"] == 1
        # 困难来源仍在 C 里（没有被任何过滤删掉）
        row0 = res["rows"][0]
        assert any(c["kind"] == "thin" and c["colour"] == "white"
                   and (c["learned_frac"] or 0) < probe.LEARNED_FRAC_MIN
                   for c in row0["candidates"])
        # 非 P 帧候选不在 C 口径，但全帧口径必须看得到（否则来源会"消失"）
        alls = s["candidate_sources_all"]
        assert alls["total"] == s["candidates_total"] == 5
        assert alls["by_arm"]["cv_only"] == 2
        # 逐帧也有分解，且能加总回 summary 的两个口径
        assert sum(r["candidate_sources"]["total"] for r in res["rows"]) == 5
