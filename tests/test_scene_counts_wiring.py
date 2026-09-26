"""逐场景计数的键空间与 R 下限（独立复核发现的真实缺陷的回归）。

实测缺陷（2026-09-26 独立复核 `docs/T14_T01_T16_REVIEW_20260926.md`）：
`rounds` 判定路径原来把 `identity_metrics()["per_group"]`（键是**比率名**）
当整数计数读（键 `P_frames/C/R/M/L/A`），于是每个场景都被读成 `P_frames=0`：
真实有线场景被判 `not_applicable`，R<30 的样本下限在真实 rounds 路径
**永不触发**（复核实测输入 P=3,C=100,R=62,M=9 得到全 0 判定）。
正确的整数计数一直在同一次返回的 `counts_by_group` 里（此前只落盘、未判定）。

本文件钉住三件事：
1. 两个字典的键空间不同（旧读法必然得 0，不能再犯）；
2. 下限按 **R** 计（不是 C）：R=62 过、R=9 不足、R=0 是 UNKNOWN、无线场景不适用；
3. 判定产物必须带 `scene_applicability`（否则看板只能显示"无数据"）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 探针在 scripts/ 下（`identity_metrics` 内部也会插一次；测试要先 import 它
# 才能替换 probe_fn，所以这里同样插一次，保持与生产同一份模块）
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_seg_autoloop", ROOT / "scripts" / "m5_seg_autoloop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_seg_autoloop"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frames(tmp_path, name: str, *, source_id: str) -> Path:
    """一个带身份的采集目录（身份从 meta.json 读，不从目录名猜）。"""
    d = tmp_path / "runs" / name / "front_main"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        np.savez(d / f"frame_{i:05d}.npz",
                 colour=np.full((20, 24, 3), 30 + i, np.uint8),
                 label=np.zeros((20, 24), np.uint8))
    (tmp_path / "runs" / name / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": source_id,
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(2)]}), encoding="utf-8")
    return d


#: 复核实测的输入：P=3, C=100, R=62, M=9（R<30 的场景另给 R=9）
FAKE_COUNTS = {
    "coll_big": {"P_frames": 3, "C": 100, "R": 62, "M": 9, "L": 0, "A": 0,
                 "C_outside_P": 0},
    "coll_small": {"P_frames": 3, "C": 12, "R": 9, "M": 4, "L": 4, "A": 4,
                   "C_outside_P": 0},
}


def _fake_probe(seen: list):
    def probe(run, meta, *, view=None, model_path=None):
        name = Path(run).parent.name
        seen.append(name)
        c = FAKE_COUNTS.get(name, {"P_frames": 0, "C": 0, "R": 0, "M": 0,
                                   "L": 0, "A": 0, "C_outside_P": 0})
        return {"summary": {
            "counts": dict(c), "n_candidates": c["C"],
            "n_candidates_with_reference": c["R"],
            "match_rate": 0.09, "match_rate_with_reference": 0.1452,
            "role_agreement_rate": None, "candidate_paint_recall": 0.5}}
    return probe


def test_counts_and_ratios_are_different_key_spaces(tmp_path, monkeypatch):
    """`counts_by_group`（整数）与 `per_group`（比率）不可互换读。"""
    loop = _load()
    import m5_marking_identity_probe as ip
    seen: list = []
    monkeypatch.setattr(ip, "probe", _fake_probe(seen))
    big = _frames(tmp_path, "coll_big", source_id="ring_big")
    small = _frames(tmp_path, "coll_small", source_id="ring_small")
    out = loop.identity_metrics(Path("model.pt"), [big, small])
    assert seen == ["coll_big", "coll_small"], seen

    cb = out["counts_by_group"]
    assert set(cb) == {"italy/ring_big", "italy/ring_small"}, sorted(cb)
    for g, c in cb.items():
        for k, v in c.items():
            assert isinstance(v, int), f"{g}.{k}={v!r} 不是整数计数"
    assert cb["italy/ring_big"]["R"] == 62 and cb["italy/ring_big"]["C"] == 100
    assert cb["italy/ring_small"]["R"] == 9

    # 比率字典是另一个键空间：**没有** P_frames/C/R/M/L/A
    rb = out["ratios_by_group"]["italy/ring_big"]
    assert "candidate_reference_coverage" in rb
    for k in ("P_frames", "C", "R", "M", "L", "A"):
        assert k not in rb, f"比率字典里不该有计数键 {k}"
    # 旧读法的必然结果：把 per_group 当计数读，每个场景都是 P_frames=0
    for g in out["per_group"]:
        assert int(out["per_group"][g].get("P_frames", 0)) == 0, \
            "这个 0 就是旧实现把有线场景判成 not_applicable 的原因"
    # 总体计数仍是整数累加（先加总再算比率）
    assert out["counts"]["R"] == 71 and out["counts"]["C"] == 112


def test_the_scene_floor_is_on_R_not_on_C():
    """下限对象是身份率的实际分母 R：C=100 但 R=9 仍然不足。"""
    from beamng_autopilot.experiments.gates import (
        Thresholds, scene_count_violations)

    t = Thresholds(per_scene_min_candidates=30)
    per_scene = {
        "big": dict(FAKE_COUNTS["coll_big"]),          # R=62 >= 30
        "small": dict(FAKE_COUNTS["coll_small"]),      # R=9  < 30（C=12）
        "unknown": {"P_frames": 2, "C": 10, "R": 0, "M": 0, "L": 0, "A": 0},
        "no_truth": {"P_frames": 0, "C": 7, "R": 0, "M": 0, "L": 0, "A": 0},
    }
    rep = scene_count_violations(per_scene, t)
    assert rep["scenes"]["big"]["applicability"] == "measured"
    assert rep["scenes"]["big"]["sample"] == "ok"
    assert rep["scenes"]["small"]["applicability"] == "measured"
    assert rep["scenes"]["small"]["sample"] == "insufficient"
    assert any("small" in m for m in rep["low_sample"]), rep["low_sample"]
    # R=0 且有真值：UNKNOWN（缺测），不是 0 分
    assert rep["scenes"]["unknown"]["applicability"] == "unknown"
    assert any("unknown" in m for m in rep["missing"]), rep["missing"]
    # 确认真无线：不适用——既不算通过也不算缺测（不能被下限/缺测挡住实验）
    assert rep["scenes"]["no_truth"]["applicability"] == "not_applicable"
    assert not any("no_truth" in m for m in rep["missing"] + rep["low_sample"]), \
        rep
    # L=0 且 R>0：角色率 UNKNOWN（单独一条），不能悄悄当"角色一致率 0"
    assert any("L=0" in m and "big" in m for m in rep["missing"]), rep["missing"]


def test_byte_identical_eval_dirs_do_not_double_count(tmp_path, monkeypatch):
    """S2 验收：评价路径也要消费**唯一**清单，不能按目录各自 glob。

    独立复核实测（`docs/T14_T01_T16_REVIEW_20260926.md` 发现 1）：两个字节
    相同的评价目录会让 `identity_metrics` 的 C 从 10 变 20，而标定侧（走
    manifest）报 10 —— 同一批帧两套数。这里钉住：审计给出唯一清单后，
    重复拷贝被排除，计数与唯一帧数一致。
    """
    from beamng_autopilot.experiments.manifest import DatasetManifest

    loop = _load()
    a = _frames(tmp_path, "coll_a", source_id="ring_a")
    # 第二个目录：与 coll_a 逐字节相同（同 RGB、同标签、同几何）
    b = tmp_path / "runs" / "coll_b" / "front_main"
    b.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.glob("frame_*.npz")):
        (b / f.name).write_bytes(f.read_bytes())
    (tmp_path / "runs" / "coll_b" / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_b",
        "map_name_source": "session.get_current().level",
        "frames": [{"i": i, "view": "front_main", "exposure": i,
                    "path": f"front_main/frame_{i:05d}.npz"}
                   for i in range(2)]}), encoding="utf-8")

    mf = DatasetManifest.build([a, b], root=tmp_path,
                               paint_sources={"front_main": "engine_annotation"})
    accepted = [r for r in mf.records if not r.reject_reason]
    rejected = [r for r in mf.records if r.reject_reason]
    assert len(accepted) == 2, [r.path for r in mf.records]
    assert rejected, "重复拷贝必须被审计排除（否则计数会翻倍）"
    frames_by_dir: dict = {}
    for r in accepted:
        _k = str(Path(r.path).parent).replace("\\", "/")
        frames_by_dir.setdefault(_k, []).append(r.path)

    def counting_probe(run, meta, *, view=None, model_path=None, frames=None):
        # 不给清单时按目录 glob（旧行为）；给了清单就只测清单里的帧
        fs = (sorted(Path(run).glob("frame_*.npz")) if frames is None
              else list(frames))
        n = len(fs)
        return {"summary": {
            "counts": {"P_frames": n, "C": 2 * n, "R": n, "M": 0, "L": 0,
                       "A": 0, "C_outside_P": 0},
            "n_candidates": 2 * n, "n_candidates_with_reference": n,
            "match_rate": 0.0, "match_rate_with_reference": 0.0,
            "role_agreement_rate": None, "candidate_paint_recall": None}}

    import m5_marking_identity_probe as ip
    monkeypatch.setattr(ip, "probe", counting_probe)
    out = loop.identity_metrics(Path("model.pt"), [a, b],
                                frames_by_dir=frames_by_dir)
    assert out["counts"]["P_frames"] == 2, out["counts"]
    assert out["counts"]["C"] == 4, out["counts"]
    # 不给唯一清单时是**未去重**口径（旧行为，保留但必须显式选择）
    out2 = loop.identity_metrics(Path("model.pt"), [a, b])
    assert out2["counts"]["C"] == 8, out2["counts"]


def test_identity_metrics_counts_by_group_matches_the_total(tmp_path,
                                                            monkeypatch):
    """逐场景计数之和必须等于总体计数（守恒，不允许两套数）。"""
    loop = _load()
    import m5_marking_identity_probe as ip

    monkeypatch.setattr(ip, "probe", _fake_probe([]))
    big = _frames(tmp_path, "coll_big", source_id="ring_big")
    small = _frames(tmp_path, "coll_small", source_id="ring_small")
    out = loop.identity_metrics(Path("model.pt"), [big, small])
    for k in ("P_frames", "C", "R", "M", "L", "A", "C_outside_P"):
        assert sum(int(v.get(k, 0)) for v in out["counts_by_group"].values()) \
            == int(out["counts"][k]), k


def test_a_failed_eval_run_is_visible_not_silently_dropped(tmp_path,
                                                           monkeypatch):
    """T11：缺 meta / 探针异常的 run 必须逐条可见，不许 `except: continue`。

    旧实现静默丢掉整个 run：计数为 0，读起来像"这个场景没有候选"。
    """
    loop = _load()
    good = _frames(tmp_path, "coll_good", source_id="ring_good")
    bad = tmp_path / "runs" / "coll_bad" / "front_main"
    bad.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        np.savez(bad / f"frame_{i:05d}.npz",
                 colour=np.full((20, 24, 3), 60 + i, np.uint8),
                 label=np.zeros((20, 24), np.uint8))
    # 没有 meta.json（也不在父目录）-> 旧实现直接 continue

    def probe(run, meta, *, view=None, model_path=None, frames=None):
        name = Path(run).parent.name
        if name == "coll_boom":
            raise RuntimeError("probe blew up")
        n = len(frames) if frames is not None else len(
            list(Path(run).glob("frame_*.npz")))
        return {"summary": {
            "counts": {"P_frames": n, "C": n, "R": n, "M": 0, "L": 0, "A": 0,
                       "C_outside_P": 0},
            "n_candidates": n, "n_candidates_with_reference": n,
            "match_rate": 0.0, "match_rate_with_reference": 0.0,
            "role_agreement_rate": None, "candidate_paint_recall": None}}

    boom = _frames(tmp_path, "coll_boom", source_id="ring_boom")
    import m5_marking_identity_probe as ip
    monkeypatch.setattr(ip, "probe", probe)
    out = loop.identity_metrics(Path("model.pt"), [good, bad, boom])
    assert out["n_eval_run_errors"] == 2, out["eval_run_errors"]
    whys = " | ".join(e["why"] for e in out["eval_run_errors"])
    assert "no meta.json" in whys and "probe blew up" in whys, whys
    # 好目录照常计数（不是被整批吞成 0）
    assert out["counts"]["P_frames"] == 2, out["counts"]
    assert out["counts_by_group"]["italy/ring_good"]["C"] == 2
