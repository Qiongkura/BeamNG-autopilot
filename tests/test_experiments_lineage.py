"""训练 lineage / 已见组关系（方案 v2 S1 §4–5）的判据测试。

钉住四条纪律：

* checkpoint 声明的 ``train_args.runs`` 含某组 → 该组帧是"已见"；
* 读不到 ``train_args`` / ``runs`` 为空 → **UNKNOWN**（``None``），
  ``lineage_unknown`` 记模型名——**绝不默认 False**；
* 内容重叠只看帧给的 ``content_sha16``（与训练目录 ``frame_*.npz`` 的 colour
  哈希比对；没给哈希就是 ``None``，不猜；不递归扫子目录）；
* ``summarize`` 计数守恒：True / False / None 是完整划分，逐组与顶层同口径。

checkpoint 用 ``torch.save`` 造极小文件（只关心 train_args），npz 用 2×2 图。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beamng_autopilot.experiments.lineage import (  # noqa: E402
    frame_training_relation,
    load_train_runs,
    summarize,
)

COUNT_KEYS = ("n_frames", "n_group_overlap", "n_content_overlap",
              "n_lineage_unknown", "n_parent_unknown")


def _img(value: int) -> np.ndarray:
    """2×2 的三通道图：内容只由像素值决定，够算哈希。"""
    return np.full((2, 2, 3), int(value), np.uint8)


def _sha16(arr: np.ndarray) -> str:
    """与 manifest.content_sha16 同一套定义（这里手算，测试才有独立性）。"""
    a = np.ascontiguousarray(np.asarray(arr, np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _run(root: Path, name: str, *, map_name: str = "italy",
         source_id: str | None = None, colours=()) -> Path:
    """造一个采集目录：meta.json 给身份（组键），frame_*.npz 给内容。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "map_name": map_name, "source_id": source_id or name,
        "map_name_source": "test"}), encoding="utf-8")
    for i, c in enumerate(colours):
        np.savez(d / f"frame_{i:05d}.npz", colour=c)
    return d


def _ckpt(path: Path, train_args: dict | None = None, **extra) -> Path:
    """极小 checkpoint。``train_args=None`` = 老 checkpoint（没有该字段）。"""
    blob = dict(extra)
    if train_args is not None:
        blob["train_args"] = train_args
    torch.save(blob, path)
    return path


def _frame(path: str, group: str, sha: str | None = None) -> dict:
    return {"path": path, "group": group, "content_sha16": sha}


def test_declared_train_group_is_seen_and_other_groups_are_not(tmp_path):
    """正例：runs 含某组 → 该组帧 True；全部可读且没命中 → 才是 False。"""
    run = _run(tmp_path, "ring_a", colours=[_img(7)])
    ck = _ckpt(tmp_path / "best.pt",
               {"runs": [str(run)], "n_train": 1, "epochs": 1})
    rel = frame_training_relation(
        [_frame("seen.npz", "italy/ring_a"),
         _frame("other.npz", "italy/ring_b")],
        [{"name": "best", "path": str(ck)}])
    seen, other = rel
    assert seen["train_group_overlap"] is True
    assert other["train_group_overlap"] is False
    assert seen["seen_in"] == ["best"] and other["seen_in"] == []
    assert seen["lineage_unknown"] == [] and other["lineage_unknown"] == []
    # 没给 content_sha16 -> 内容关系不判（None），不猜 False
    assert seen["train_content_overlap"] is None
    assert other["train_content_overlap"] is None


def test_unreadable_lineage_is_unknown_never_false(tmp_path):
    """反例：没有 train_args / runs 为空 → None + lineage_unknown，不是 False。"""
    legacy = _ckpt(tmp_path / "legacy.pt")                 # 老 checkpoint
    empty = _ckpt(tmp_path / "empty.pt", {"runs": []})
    for name, ck in (("legacy", legacy), ("empty", empty)):
        rel = frame_training_relation([_frame("f.npz", "italy/ring_a")],
                                      [{"name": name, "path": str(ck)}])
        r = rel[0]
        assert r["train_group_overlap"] is None, name       # UNKNOWN
        assert r["lineage_unknown"] == [name]
        assert r["seen_in"] == []
        assert load_train_runs(ck) is None


def test_the_production_checkpoint_shape_is_read_through_torch_load(tmp_path):
    """真实 checkpoint 的 train_args 形状（runs/n_train/epochs）能读出来。"""
    run = _run(tmp_path, "ring_a")
    ck = _ckpt(tmp_path / "best.pt",
               {"runs": [str(run)], "n_train": 24, "epochs": 12})
    assert load_train_runs(ck) == [str(run)]
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    assert blob["train_args"]["n_train"] == 24


def test_same_colour_in_a_declared_run_is_content_overlap(tmp_path):
    """同一张图（colour 相同）出现在训练目录与评价帧 → 内容重叠 True。"""
    pic = _img(3)
    run = _run(tmp_path, "ring_a", colours=[pic])
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(run)]})
    rel = frame_training_relation(
        [_frame("copy.npz", "italy/ring_b", sha=_sha16(pic)),      # 组不同，内容同
         _frame("new.npz", "italy/ring_b", sha=_sha16(_img(4))),   # 都不命中
         _frame("unchecked.npz", "italy/ring_b")],                 # 没给哈希
        [{"name": "m", "path": str(ck)}])
    copy, new, unchecked = rel
    assert copy["train_content_overlap"] is True
    assert copy["train_group_overlap"] is False
    # 内容命中就是"已见"（同图复制比组重叠更强）
    assert copy["seen_in"] == ["m"]
    assert new["train_content_overlap"] is False
    assert unchecked["train_content_overlap"] is None          # 不猜
    assert unchecked["seen_in"] == []


def test_content_scan_does_not_recurse_into_subdirectories(tmp_path):
    """只扫 checkpoint 声明目录本层的 frame_*.npz，不递归子目录。"""
    pic = _img(5)
    run = _run(tmp_path, "ring_a", colours=[_img(6)])   # 本层有帧 -> 扫描可判
    nested = run / "front_main"
    nested.mkdir()
    np.savez(nested / "frame_00000.npz", colour=pic)    # 只在子目录里的同图
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(run)]})
    r = frame_training_relation(
        [_frame("f.npz", "italy/ring_b", sha=_sha16(pic))],
        [{"name": "m", "path": str(ck)}])[0]
    assert r["train_content_overlap"] is False


def test_a_vanished_run_dir_cannot_rule_content_out(tmp_path):
    """声明的目录不在了 → 内容不可判（None）；组键仍按路径键算，不冒充命中。"""
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(tmp_path / "gone")]})
    r = frame_training_relation(
        [_frame("f.npz", "italy/ring_a", sha=_sha16(_img(1)))],
        [{"name": "m", "path": str(ck)}])[0]
    assert r["train_content_overlap"] is None
    # 无身份目录的键是 dir/<绝对路径>，不可能等于带身份的组键
    assert r["train_group_overlap"] is False
    assert r["lineage_unknown"] == []


def test_identity_less_train_dir_is_not_a_default_clean(tmp_path):
    """训练目录没有身份 → 组键是 ``dir/<名字>``，与带身份组键不同组（key 语义）。

    实测生产 ``best.pt`` 的 runs 多为无身份目录：相等判定只给 False，报告
    不能据此单独宣称 leak-free（方案 §5：继续查空间与生产训练来源）。
    """
    d = tmp_path / "dirt_road_labeled"        # 有目录、没有 meta.json、没有帧
    d.mkdir()
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(d)]})
    r = frame_training_relation(
        [_frame("f.npz", "italy/ring_a", sha=_sha16(_img(1)))],
        [{"name": "m", "path": str(ck)}])[0]
    assert r["train_group_overlap"] is False           # 键空间不同，不是"默认干净"
    assert r["train_content_overlap"] is None          # 空目录 -> 内容不可判
    assert r["lineage_unknown"] == []


def test_a_positive_hit_survives_an_unreadable_checkpoint(tmp_path):
    """一个 checkpoint 读不到 lineage，不推翻另一个的命中，但照样记 UNKNOWN。"""
    run = _run(tmp_path, "ring_a")
    good = _ckpt(tmp_path / "good.pt", {"runs": [str(run)]})
    legacy = _ckpt(tmp_path / "legacy.pt")
    r = frame_training_relation(
        [_frame("f.npz", "italy/ring_a")],
        [{"name": "good", "path": str(good)},
         {"name": "legacy", "path": str(legacy)}])[0]
    assert r["train_group_overlap"] is True
    assert r["seen_in"] == ["good"]
    assert r["lineage_unknown"] == ["legacy"]


def test_a_frame_without_a_group_is_unknown(tmp_path):
    """帧自己缺组键 → UNKNOWN（帧侧数据缺口，不是"没重叠"）。"""
    run = _run(tmp_path, "ring_a")
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(run)]})
    r = frame_training_relation([_frame("f.npz", "")],
                                [{"name": "m", "path": str(ck)}])[0]
    assert r["train_group_overlap"] is None
    assert r["lineage_unknown"] == []


def test_no_checkpoints_is_unknown_not_clean(tmp_path):
    """压根没给 checkpoint → 全部 UNKNOWN（绝不报"干净"）。"""
    r = frame_training_relation(
        [_frame("f.npz", "italy/ring_a", sha=_sha16(_img(1)))], [])[0]
    assert r["train_group_overlap"] is None
    assert r["train_content_overlap"] is None
    assert r["lineage_unknown"] == []
    assert r["parent_unknown"] is True


def test_parent_is_read_from_train_args_and_missing_is_unknown(tmp_path):
    """parent 取 init_from/resume_from/parent；没有 → None + parent_unknown。"""
    run = _run(tmp_path, "ring_a")
    ck = _ckpt(tmp_path / "m.pt",
               {"runs": [str(run)], "init_from": "logs/m5_seg/prev/best.pt"})
    r = frame_training_relation([_frame("f.npz", "italy/ring_a")],
                                [{"name": "m", "path": str(ck)}])[0]
    assert r["parent"] == "logs/m5_seg/prev/best.pt"
    assert r["parent_unknown"] is False

    legacy = _ckpt(tmp_path / "legacy.pt")
    r2 = frame_training_relation([_frame("f.npz", "italy/ring_a")],
                                 [{"name": "legacy", "path": str(legacy)}])[0]
    assert r2["parent"] is None and r2["parent_unknown"] is True


def test_conflicting_parents_are_not_named(tmp_path):
    """两个 checkpoint 的父来源不一致 → 不点名；一致 → 点名。"""
    a = _ckpt(tmp_path / "a.pt",
              {"runs": [str(_run(tmp_path, "r1"))], "parent": "p1"})
    b = _ckpt(tmp_path / "b.pt",
              {"runs": [str(_run(tmp_path, "r2"))], "parent": "p2"})
    r = frame_training_relation(
        [_frame("f.npz", "italy/r1")],
        [{"name": "a", "path": str(a)}, {"name": "b", "path": str(b)}])[0]
    assert r["parent"] is None and r["parent_unknown"] is True

    c = _ckpt(tmp_path / "c.pt",
              {"runs": [str(_run(tmp_path, "r3"))], "resume_from": "p0"})
    d = _ckpt(tmp_path / "d.pt",
              {"runs": [str(_run(tmp_path, "r4"))], "init_from": "p0"})
    r2 = frame_training_relation(
        [_frame("f.npz", "italy/r3")],
        [{"name": "c", "path": str(c)}, {"name": "d", "path": str(d)}])[0]
    assert r2["parent"] == "p0" and r2["parent_unknown"] is False


def test_boolean_resume_from_is_not_a_parent_name(tmp_path):
    """旧 checkpoint 的 resume_from 可能是 bool（"是不是续训"），不是路径。"""
    ck = _ckpt(tmp_path / "m.pt",
               {"runs": [str(_run(tmp_path, "r1"))], "resume_from": True})
    r = frame_training_relation([_frame("f.npz", "italy/r1")],
                                [{"name": "m", "path": str(ck)}])[0]
    assert r["parent"] is None and r["parent_unknown"] is True


def test_summarize_conserves_counts_and_lists_in_sample_frames(tmp_path):
    """True/False/None 完整划分且守恒；by_group 与顶层同口径。"""
    pic = _img(1)
    run = _run(tmp_path, "ring_a", colours=[pic])
    ck = _ckpt(tmp_path / "m.pt", {"runs": [str(run)]})
    rel = frame_training_relation(
        [_frame("a.npz", "italy/ring_a"),                        # 组重叠
         _frame("b.npz", "italy/ring_b"),                        # 不重叠
         _frame("c.npz", "italy/ring_b", sha=_sha16(pic))],      # 内容重叠
        [{"name": "m", "path": str(ck)}])
    s = summarize(rel)
    n = len(rel)
    assert s["n_frames"] == n
    assert s["n_group_overlap"] == 1
    assert s["n_content_overlap"] == 1
    assert s["n_lineage_unknown"] == 0
    assert s["n_parent_unknown"] == n          # 没记父来源 -> 逐帧 UNKNOWN
    # 守恒：三种取值各计一次，加起来 = 总数
    vals = [r["train_group_overlap"] for r in rel]
    assert sum(1 for v in vals if v is True) == s["n_group_overlap"]
    assert sum(1 for v in vals if v is False) == n - s["n_group_overlap"] \
        - s["n_lineage_unknown"]
    assert sum(1 for v in vals if v is None) == s["n_lineage_unknown"]
    assert sum(1 for v in vals if v in (True, False, None)) == n
    # 已见帧 = 组重叠或内容重叠
    assert s["in_sample_frames"] == ["a.npz", "c.npz"]
    # by_group 与顶层同口径
    assert sorted(s["by_group"]) == ["italy/ring_a", "italy/ring_b"]
    for key in COUNT_KEYS:
        assert sum(g[key] for g in s["by_group"].values()) == s[key]
    assert s["by_group"]["italy/ring_b"]["in_sample_frames"] == ["c.npz"]


def test_summarize_unknown_is_its_own_bucket(tmp_path):
    """UNKNOWN 单独成桶：不与"没重叠"混在一起（方案 §5）。"""
    legacy = _ckpt(tmp_path / "legacy.pt")
    rel = frame_training_relation(
        [_frame("x.npz", "italy/ring_a"), _frame("y.npz", "italy/ring_b")],
        [{"name": "legacy", "path": str(legacy)}])
    s = summarize(rel)
    assert s["n_frames"] == 2
    assert s["n_group_overlap"] == 0
    assert s["n_lineage_unknown"] == 2
    assert s["in_sample_frames"] == []
    assert s["by_group"]["italy/ring_a"]["n_lineage_unknown"] == 1
    assert s["n_group_overlap"] + s["n_lineage_unknown"] == s["n_frames"]


def test_load_train_runs_reports_unknown_as_none(tmp_path):
    """load_train_runs：可读给列表（忽略空串），不可读一律 None。"""
    run = _run(tmp_path, "ring_a")
    ok = _ckpt(tmp_path / "ok.pt", {"runs": [str(run), ""]})
    assert load_train_runs(ok) == [str(run)]
    assert load_train_runs(_ckpt(tmp_path / "legacy.pt")) is None
    assert load_train_runs(_ckpt(tmp_path / "empty.pt", {"runs": []})) is None
    assert load_train_runs(_ckpt(tmp_path / "noruns.pt", {"epochs": 3})) is None
    assert load_train_runs(tmp_path / "missing.pt") is None
    odd = tmp_path / "odd.pt"
    torch.save([1, 2, 3], odd)                 # 不是 dict
    assert load_train_runs(odd) is None
