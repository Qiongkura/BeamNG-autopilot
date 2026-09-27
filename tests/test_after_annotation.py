"""标注完成后的下一步入口（方案 §S6）：未就绪必须拒绝，就绪才生成两份配置。

动机：合格版 E1 与可晋级线训练都依赖**评价集之外**的 verified 标签；标注是人的
时间，所以"标注完了怎么跑"必须是一条命令，且**未标注的包绝不能被拿去跑训练**。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "m5_after_annotation", ROOT / "scripts" / "m5_after_annotation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m5_after_annotation"] = mod
    spec.loader.exec_module(mod)
    return mod


def _pack(tmp_path, name: str, *, line_px: int, label_source="human_revision",
          with_label=True) -> Path:
    d = tmp_path / name / "front_main"
    d.mkdir(parents=True)
    frames = []
    for i in range(3):
        kw = {"colour": np.full((12, 16, 3), 20 + i, np.uint8)}
        if with_label:
            lab = np.zeros((12, 16), np.uint8)
            if line_px:
                lab[5, :line_px] = 2
            kw["label"] = lab
        np.savez(d / f"frame_{i:05d}.npz", **kw)
        frames.append({"path": f"front_main/frame_{i:05d}.npz", "view": "front_main",
                       "pos": [1.0, 2.0, 3.0], "heading": 0.0,
                       "classes_painted": {"line": line_px, "road": 100,
                                           "background": 92 - line_px, "unknown": 0}})
    (d / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": f"ring_{name}",
        "label_source": label_source,
        "annotation": {"reviewer": "owner", "identity_missing": []},
        "frames": frames}), encoding="utf-8")
    return d


def test_unannotated_packs_refuse_to_generate_training_commands(tmp_path):
    mod = _load()
    raw = _pack(tmp_path, "raw", line_px=0, with_label=False,
                label_source="beamng_annotation (road dense; ...)")
    out = tmp_path / "out"
    rc = mod.main(["--negative-dir", str(raw), "--out-dir", str(out)])
    assert rc == 3, "未标注的包必须拒绝（不带未标注的包去跑训练）"
    assert not list(out.glob("loop_config_*.json")), "拒绝时不许留下任何配置"


def test_annotated_packs_generate_both_configs(tmp_path):
    mod = _load()
    neg = _pack(tmp_path, "neg", line_px=0)
    pos = _pack(tmp_path, "pos", line_px=4)
    pos2 = _pack(tmp_path, "pos2", line_px=4)   # 第二个正例视角：线通道要"加其余视角"
    out = tmp_path / "out"
    rc = mod.main(["--negative-dir", str(neg), "--positive-dir", str(pos),
                   "--positive-dir", str(pos2), "--out-dir", str(out)])
    assert rc == 0
    e1 = json.loads((out / "loop_config_e1_promotable.json").read_text(
        encoding="utf-8"))
    line = json.loads((out / "loop_config_line_promotable.json").read_text(
        encoding="utf-8"))
    prop = json.loads((out / "proposal_e1_promotable.json").read_text(
        encoding="utf-8"))
    # E1：负例臂 20%，两臂共享步数；标注目录按 human_revision；不加 research 旗标
    assert prop["proposals"][0]["factor"]["run_weights"] == {"0": 0.8, "1": 0.2}
    assert e1["equal_steps"] is True and "research_arm" not in e1
    assert any("neg/front_main=human_revision" in s for s in e1["paint_sources"])
    # 基座 = **verified 正例**（这样两臂标签都是 human_revision -> 判定可晋级）；
    # 若基座退回 agent 弱标签的 town，整轮会被判 research_only
    assert e1["baseline_runs"] == [str(pos).replace("\\", "/")], e1["baseline_runs"]
    assert all("human_revision" in s for s in e1["paint_sources"]), e1["paint_sources"]
    # 线通道：把**其余**正例组加进训练输入（基座那一个不重复加）
    assert line["baseline_runs"] == [str(pos).replace("\\", "/")]
    # `runs` 只放基座：其余视角由因子（add_runs）加进来，否则会被 factor_not_applied
    # 正确拒训（实测踩到）
    assert line["runs"] == [str(pos).replace("\\", "/")], line["runs"]
    assert json.loads((out / "proposal_line_promotable.json").read_text(
        encoding="utf-8"))["proposals"][0]["factor"]["add_runs"] == [
        str(pos2).replace("\\", "/")], "只加其余视角（基座不重复加）"
    assert json.loads((out / "proposal_line_promotable.json").read_text(
        encoding="utf-8"))["proposals"][0]["factor"]["add_runs"]


def test_a_line_bearing_dir_cannot_be_used_as_a_pure_negative(tmp_path):
    """把"含线帧"的目录当纯负例提交 -> 拒绝（脚本不替人改口径）。"""
    mod = _load()
    mixed = _pack(tmp_path, "mixed", line_px=4)
    rc = mod.main(["--negative-dir", str(mixed), "--out-dir", str(tmp_path / "o")])
    assert rc == 3
