

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
