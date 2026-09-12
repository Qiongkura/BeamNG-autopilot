"""Grid labeler regressions: scan/store/picker plus HTTP round-trip (no game)."""

from __future__ import annotations

import base64
import json
import random
import threading
import urllib.request
from pathlib import Path

from beamng_autopilot.labeling.grid_labeler import (
    DEFAULT_TASK, BatchPicker, GridLabelApp, LabelStore, scan_images, serve)

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _make_tree(root: Path, n: int = 12) -> list[Path]:
    (root / "sub").mkdir(parents=True)
    for i in range(n):
        d = root / "sub" if i % 2 else root
        (d / f"f_{i:02d}.png").write_bytes(PNG_1PX)
    (root / "note.txt").write_bytes(b"skip me")
    return scan_images(root, recursive=True)


def test_scan_images_filters_and_sorts(tmp_path):
    all_imgs = _make_tree(tmp_path)
    assert len(all_imgs) == 12
    assert all(p.suffix == ".png" for p in all_imgs)
    names = [p.relative_to(tmp_path).as_posix() for p in all_imgs]
    assert names == sorted(names)
    flat = scan_images(tmp_path, recursive=False)
    assert len(flat) == 6 and all(p.parent == tmp_path for p in flat)


def test_label_store_roundtrip_and_resume(tmp_path):
    out = tmp_path / "labels.jsonl"
    store = LabelStore(out)
    store.append([{"path": "C:/a.png", "has_line": 1, "image": "a.png"},
                  {"path": "C:/b.png", "has_line": 0, "image": "b.png"}])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    reloaded = LabelStore(out)
    assert reloaded.labeled == {"C:/a.png": 1, "C:/b.png": 0}
    reloaded.append([{"path": "C:/a.png", "has_line": 0, "image": "a.png"}])
    assert LabelStore(out).labeled["C:/a.png"] == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_batch_picker_random_and_score():
    pending = [f"n{i:02d}.png" for i in range(20)]
    rng_picker = BatchPicker(rng=random.Random(0))
    got = rng_picker.pick(pending, 9)
    assert len(got) == 9 and len(set(got)) == 9
    assert all(n in pending for n in got)

    scored = BatchPicker(scores={"n05.png": 0.9, "n07.png": 0.8,
                                 "n11.png": 0.7},
                         epsilon=0.0)
    assert scored.pick(pending, 3) == ["n05.png", "n07.png", "n11.png"]
    explored = BatchPicker(scores={"n05.png": 0.9}, epsilon=1.0)
    got = explored.pick(pending, 5)
    assert len(set(got)) == 5 and all(n in pending for n in got)


def test_app_submit_flow(tmp_path):
    paths = _make_tree(tmp_path)
    out = tmp_path / "labels.jsonl"
    store = LabelStore(out)
    app = GridLabelApp(tmp_path, paths, store, batch_size=9, seed=0)

    batch = app.next_batch()
    assert len(batch["images"]) == 9
    assert batch["labeled"] == 0 and batch["total"] == 12
    assert batch["images"][0]["url"].startswith("/img?name=")

    labels = {im["name"]: i % 2 for i, im in enumerate(batch["images"])}
    res = app.submit(labels)
    assert res["accepted"] == 9 and not res["done"]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 9

    nxt = app.next_batch()
    assert len(nxt["images"]) == 3
    res = app.submit({im["name"]: 0 for im in nxt["images"]})
    assert res["done"] and res["labeled"] == 12 and res["total"] == 12
    assert app.next_batch()["images"] == []

    # 已标注帧默认跳过；未知名字被忽略
    app2 = GridLabelApp(tmp_path, paths, LabelStore(out), batch_size=9)
    assert app2.progress()["labeled"] == 12
    assert app2.submit({"ghost.png": 1})["accepted"] == 0


def test_app_http_roundtrip(tmp_path):
    # 系统代理（如 127.0.0.1:10809）会被 urllib 拾取，回环请求必须直连
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    paths = _make_tree(tmp_path)
    app = GridLabelApp(tmp_path, paths, LabelStore(tmp_path / "labels.jsonl"),
                       batch_size=9, seed=0)
    httpd = serve(app, port=0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        with opener.open(base + "/") as r:
            page = r.read().decode("utf-8")
            assert r.status == 200 and DEFAULT_TASK in page
            assert "提交" in page and "全选" in page and "全部无标线" in page

        with opener.open(base + "/api/batch") as r:
            batch = json.loads(r.read())
        assert len(batch["images"]) == 9

        im = batch["images"][0]
        with opener.open(base + im["url"]) as r:
            assert r.status == 200
            assert r.headers["Content-Type"] == "image/png"
            assert r.read() == PNG_1PX

        labels = {x["name"]: 1 for x in batch["images"]}
        req = urllib.request.Request(
            base + "/api/submit",
            data=json.dumps({"labels": labels}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with opener.open(req) as r:
            res = json.loads(r.read())
        assert res["ok"] and res["accepted"] == 9

        with opener.open(base + "/api/batch") as r:
            assert json.loads(r.read())["labeled"] == 9
    finally:
        httpd.shutdown()
        httpd.server_close()
