

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
