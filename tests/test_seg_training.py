

def test_split_frames_train_only_run_stays_in_training() -> None:
    from scripts import m5_train_seg as tr
    import numpy as np
    frames = []
    per = {}
    for name, count in (("normal", 10), ("manual", 1)):
        start = len(frames)
        frames.extend([(np.zeros((4, 4, 3), np.uint8),
                        np.zeros((4, 4), np.uint8)) for _ in range(count)])
        per[name] = {"kept": count, "start": start, "end": len(frames)}
    train, val = tr.split_frames(frames, per, "per-run", 0.2,
                                  train_only_runs={"manual"})
    assert len(train) == 9
    assert len(val) == 2
    bounds = tr.train_run_bounds(frames, per, "per-run", 0.2,
                                 train_only_runs={"manual"})
    assert bounds[-1] == (8, 9)


def test_per_run_keys_are_unique_across_ring_collections(tmp_path) -> None:
    """Six ring collections all pass ``<collection>/front_main``.

    Keying ``per_run`` by the basename collapsed them into ONE entry: with
    ``--split by-map-scene`` the entry then trained on 25 of 173 frames
    (measured 2026-09-24).  The key must be unique per directory, so the
    split sees every collection as its own group.
    """
    from scripts import m5_train_seg as tr
    import numpy as np
    dirs = []
    for coll in ("coll_a", "coll_b", "coll_c"):
        d = tmp_path / coll / "front_main"
        d.mkdir(parents=True)
        for i in range(3):
            np.savez(d / f"frame_{i:05d}.npz",
                     colour=np.zeros((4, 4, 3), np.uint8),
                     label=np.zeros((4, 4), np.uint8))
        dirs.append(d)
    frames, per_run = tr.load_frames(dirs)
    assert len(per_run) == 3, "one entry PER DIRECTORY, not per basename"
    assert len(frames) == 9
    assert len(set(per_run)) == 3
    for key, rec in per_run.items():
        assert key.endswith("front_main"), key
        assert rec["dir_name"] == "front_main"


def test_a_ring_collection_reads_the_collection_meta_for_its_view(tmp_path) -> None:
    """The meta lives at the collection root; it must be read AND filtered.

    Reading it unfiltered would pull all eight views' frames into this
    run's identity (counts, exposures, wall clock) - the audit then
    describes 1258 frames while training uses 188.
    """
    from scripts import m5_train_seg as tr
    import json
    import numpy as np
    root = tmp_path / "coll"
    for view in ("front_main", "rear"):
        d = root / view
        d.mkdir(parents=True)
        for i in range(2):
            np.savez(d / f"frame_{i:05d}.npz",
                     colour=np.zeros((4, 4, 3), np.uint8),
                     label=np.zeros((4, 4), np.uint8))
    (root / "meta.json").write_text(json.dumps({
        "map_name": "italy", "source_id": "ring_x", "width": 4, "height": 4,
        "frames": [{"i": 0, "view": "front_main", "exposure": 0, "t_wall": 1.0},
                   {"i": 1, "view": "front_main", "exposure": 1, "t_wall": 2.0},
                   {"i": 0, "view": "rear", "exposure": 0, "t_wall": 1.0},
                   {"i": 1, "view": "rear", "exposure": 1, "t_wall": 2.0}],
    }), encoding="utf-8")
    frames, per_run = tr.load_frames([root / "front_main"])
    assert len(per_run) == 1
    rec = next(iter(per_run.values()))
    assert rec["meta_level"] == "parent"
    assert len(rec["meta"]["frames"]) == 2, "only THIS view's frames"
    assert {f["view"] for f in rec["meta"]["frames"]} == {"front_main"}


def test_a_non_finite_loss_fails_immediately(tmp_path) -> None:
    """方案 §4：NaN 立即失败，不能把非有限权重当"跑完了"。

    故障注入：``--line-weight nan`` 让类别权重变 NaN，第一步损失就非有限。
    """
    import json as _json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "runs" / "front_main"
    d.mkdir(parents=True)
    for i in range(4):
        colour = np.full((16, 20, 3), 40 + i * 5, np.uint8)
        label = np.zeros((16, 20), np.uint8)
        label[4:12, :] = 1
        label[8, :4] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    run_id = "pytest_nan"
    env = dict(os.environ)
    env["BEAMNG_LOGS_DIR"] = str(tmp_path / "logs")
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "m5_train_seg.py"),
         "--runs", str(tmp_path / "runs" / "front_main"), "--split", "tail",
         "--val-frac", "0.5", "--epochs", "1", "--batch", "2",
         "--line-weight", "nan", "--device", "cpu",
         "--out", str(tmp_path / "out"), "--metrics-run", run_id],
        capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0, f"非有限损失必须失败：{r.stdout[-800:]}"
    mfile = (tmp_path / "logs" / "experiments" / run_id / "metrics.jsonl")
    assert mfile.exists()
    recs = [_json.loads(ln) for ln in mfile.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    fails = [x for x in recs if x.get("status") == "failed"]
    assert fails, "失败状态必须落盘（否则看板停在 running）"
    assert "非有限" in fails[-1].get("error", ""), fails[-1]


def test_a_poisoned_label_fails_before_training(tmp_path) -> None:
    """标签污染（未定义类别）必须在训练前失败，而不是静默当背景。"""
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "runs" / "front_main"
    d.mkdir(parents=True)
    label = np.zeros((16, 20), np.uint8)
    label[4:12, :] = 1
    label[9, 3] = 7                      # 未定义类别
    np.savez(d / "frame_00000.npz", colour=np.full((16, 20, 3), 60, np.uint8),
             label=label)
    np.savez(d / "frame_00001.npz", colour=np.full((16, 20, 3), 70, np.uint8),
             label=label)
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "m5_train_seg.py"),
         "--runs", str(tmp_path / "runs" / "front_main"), "--epochs", "1",
         "--device", "cpu",
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "标签污染" in out and "7" in out, out[-500:]


def test_ignore_line_class_really_masks_the_paint_channel(tmp_path) -> None:
    """`--ignore-line-class` 必须**真的**屏蔽 line 监督，且留下证据。

    实测背景：引擎标注把可见漆线标成沥青，只有"已标注像素 ignore + 类别权重"
    时，未标注的漆线仍以负样本形式进入 softmax 分母（等于教"未标注的漆线=背景"）。
    跑一轮小训练，钉住两件事：(a) line 类没有拿到任何监督；
    (b) checkpoint 里记下"屏蔽了多少帧、依据哪个真值来源"，可回查。
    """
    import json as _json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "runs" / "front_main"
    d.mkdir(parents=True)
    for i in range(6):
        colour = np.full((16, 20, 3), 40 + i * 5, np.uint8)
        label = np.zeros((16, 20), np.uint8)
        label[4:12, :] = 1
        label[8, :4] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "m5_train_seg.py"),
         "--runs", str(d), "--split", "tail", "--val-frac", "0.34",
         "--epochs", "1", "--batch", "2", "--device", "cpu",
         "--ignore-line-class", "--line-tversky-weight", "0",
         "--line-cldice-weight", "0",
         "--out", str(out)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-600:]
    assert "--ignore-line-class" in r.stdout
    import torch
    # 部署链（torch>=2.6 默认 weights_only=True）必须能读；读不了就等于
    # 新 checkpoint 在线上加载失败（本项目踩过一次）
    ck = torch.load(out / "checkpoint_last.pt", map_location="cpu",
                    weights_only=True)
    ta = ck["train_args"]
    assert ta["ignore_line_class"] is True
    assert ta["paint_source"] == "engine_annotation"
    assert ta["line_ignored_frames"] > 0, \
        "引擎标注的帧必须被判为不可信并整通道屏蔽（否则等于没生效）"
    hist = ck["hist"]
    assert hist.get("line_ignored_frames"), "hist 里也要留下证据"
    # 指标侧同样屏蔽：屏蔽了 line 通道，就不能把"拿引擎参考算的标线 IoU"当测量
    assert all(v is None for v in hist.get("val_line_iou", [])),         f"line 通道被屏蔽时 val_line_iou 必须是 None，得到 {hist.get('val_line_iou')}"
    assert any(reason and "屏蔽" in str(reason)
               for reason in ck.get("not_applicable", {}).values()) or         ck.get("hist", {}).get("val_line_iou") == [None]


def _steps_per_epoch(n: int, b: int) -> int:
    return max(1, -(-int(n) // max(1, int(b))))


def test_capping_train_frames_keeps_every_run_represented(tmp_path) -> None:
    """等步数对照的截帧必须**按 run 配额**：直接砍前 N 帧会把新增场景全切掉。

    实测背景：候选臂"多了一组场景"时，若截帧只看位置，被保留的正好是原来那组，
    因子等于没生效。这里用一个 3 run 的合成划分钉住配额行为。
    """
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np

    from scripts import m5_train_seg as tr
    root = Path(__file__).resolve().parents[1]
    frames = [(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8))
              for _ in range(30)]
    bounds = [(0, 10), (10, 20), (20, 30)]        # 3 个 run 各 10 帧
    out, nb, note = tr.cap_train_frames(frames, bounds, 15, seed=42)
    assert note["applied"] and len(out) == 15
    assert len(nb) == 3, "每个 run 都要留边界"
    kept = [e - s for s, e in nb]
    assert all(k == 5 for k in kept), f"配额应均分到 3 个 run，得到 {kept}"
    # 同一 seed 可复现；不同 seed 抽样不同（但配额相同）
    out2, nb2, _ = tr.cap_train_frames(frames, bounds, 15, seed=42)
    assert [id(a) for a in out] == [id(b) for b in out2]
    out3, nb3, _ = tr.cap_train_frames(frames, bounds, 15, seed=7)
    assert [e - s for s, e in nb3] == kept
    # 不截的情形原样返回
    same, sb, note2 = tr.cap_train_frames(frames, bounds, 30, seed=1)
    assert note2["applied"] is False and same is frames

    # 端到端：截到 20 训练帧后，每 epoch 步数与基线一致（等步数的实现基础）
    d = tmp_path / "runs" / "front_main"
    d.mkdir(parents=True)
    for i in range(30):
        colour = np.full((16, 20, 3), 30 + i, np.uint8)
        label = np.zeros((16, 20), np.uint8)
        label[4:12, :] = 1
        label[8, :4] = 2
        np.savez(d / f"frame_{i:05d}.npz", colour=colour, label=label)
    out_dir = tmp_path / "out_cap"
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "m5_train_seg.py"),
         "--runs", str(d), "--split", "tail", "--val-frac", "0.2",
         "--epochs", "1", "--batch", "4", "--device", "cpu",
         "--max-train-frames", "20", "--out", str(out_dir)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-1000:]
    import torch
    ck = torch.load(out_dir / "checkpoint_last.pt", map_location="cpu",
                    weights_only=True)
    ta = ck["train_args"]
    assert ta["n_train"] == 20 and ta["max_train_frames"] == 20
    assert _steps_per_epoch(ta["n_train"], ta["batch"]) == 5


def test_run_weights_resample_without_changing_the_epoch_size() -> None:
    """困难样本采样：按 run 加权（超额有放回），但**每轮样本数不变**。

    为什么总样本数必须不变：数据因子不能顺带改变优化步数，否则"训练更久"会混进
    结论（3h 计划点名的纪律）。为什么超额要有放回：不重复就只能各 run 采满自己的
    帧，权重完全失效（实测：3 个 10 帧 run 权重 2/1/0.5 得到配额 [10,10,10]）。
    """
    import numpy as np

    from scripts import m5_train_seg as tr
    bounds = [(0, 10), (10, 20), (20, 30)]
    idx, note = tr.weighted_run_indices(bounds, {"0": 2.0, "1": 1.0, "2": 0.5},
                                        np.random.default_rng(0))
    assert note["quota"] == [17, 9, 4]
    assert sum(note["quota"]) == 30 == len(set(bounds[i][0] for i in range(3))
                                           ) * 10
    assert len(idx) == 30, "每轮样本数必须与未加权时相同"
    assert note["per_run"][0]["repeats"] == 7, "超额部分靠重复采样补齐"
    assert note["per_run"][2]["quota"] == 4, "降权的 run 少采"
    # 下标落在正确的 run 段里
    assert all(0 <= i < 10 for i in idx[:12])
    assert all(20 <= i < 30 for i in idx[-4:])
    # 等权时退化为"每 run 采满自己的帧"（与旧行为一致）
    idx_eq, note_eq = tr.weighted_run_indices(bounds, {},
                                              np.random.default_rng(0))
    assert note_eq["quota"] == [10, 10, 10] and len(idx_eq) == 30
    assert sorted(int(i) for i in idx_eq) == list(range(30))
