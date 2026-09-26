"""复核进度：还差多少帧、每类差多少（W1 §6.3 上量时用）。

用户正在按全量包标注，需要随时知道"标到哪了、还差多少"。本工具只读两个目录：

* 已标：``reviewed/``（起步包）与 ``reviewed_full/``（全量包）；
* 待标：``review_pack_full.json`` 的帧清单（与已标按**内容**去重，不按文件名）。

按类别汇总 done/total，并写出 ``review_status.json`` 供看板/报告引用。
帧的内容哈希用 colour（标注只改 label，colour 不变，所以同一帧标前后能对上）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

#: 类别 -> 人读名字（与复核包一致）
CATEGORY_LABEL = {
    "clear_paint": "清晰漆线",
    "curve_junction": "弯道与路口",
    "degraded_occluded": "退化与遮挡",
    "confusable_texture": "易混淆纹理",
    "no_line_pavement": "真无线铺装路",
    "dirt_shoulder": "铺装/土肩与纯土路",
    "already_reviewed": "已复核（并入）",
}


def _colour_hash(p: Path) -> str | None:
    """帧的 colour 内容哈希；读不到返回 None（不猜）。"""
    try:
        with np.load(p) as z:
            return hashlib.sha256(
                np.asarray(z["colour"]).tobytes()).hexdigest()[:16]
    except Exception:                                      # noqa: BLE001
        return None


def _category_of(path: str, cat_map: dict) -> str:
    return str(cat_map.get(str(path)) or "")


def status(pack_dir: Path) -> dict:
    """返回 ``{by_category, done, total, remaining, ...}``。"""
    pack_dir = Path(pack_dir)
    pack = pack_dir / "review_pack_full.json"
    rows: list = []
    cat_of: dict = {}
    if pack.is_file():
        blob = json.loads(pack.read_text(encoding="utf-8"))
        for f in blob.get("frames") or []:
            rows.append(str(f.get("path") or ""))
            cat_of[str(f.get("path") or "")] = str(f.get("category") or "")
    done: set = set()
    done_dirs = []
    for sub in ("reviewed", "reviewed_full"):
        for d in sorted((pack_dir / sub).glob("*/front_main")):
            n = 0
            for f in d.glob("frame_*.npz"):
                h = _colour_hash(f)
                if h:
                    done.add(h)
                    n += 1
            if n:
                done_dirs.append({"dir": str(d), "frames": n})
    total_by_cat: dict = {}
    for p in rows:
        c = _category_of(p, cat_of) or "unknown"
        total_by_cat[c] = total_by_cat.get(c, 0) + 1
    # 已标按类别：用帧的内容哈希把 pack 里的帧分成 done/remaining
    done_by_cat: dict = {}
    remaining = []
    for p in rows:
        h = _colour_hash(Path(p))
        c = _category_of(p, cat_of) or "unknown"
        if h and h in done:
            done_by_cat[c] = done_by_cat.get(c, 0) + 1
        else:
            remaining.append({"path": p, "category": c})
    by_category = {}
    for c, n in sorted(total_by_cat.items()):
        by_category[c] = {"label": CATEGORY_LABEL.get(c, c), "total": n,
                          "done": done_by_cat.get(c, 0),
                          "remaining": n - done_by_cat.get(c, 0)}
    return {"pack_dir": str(pack_dir), "n_total": len(rows),
            "n_done": sum(v["done"] for v in by_category.values()),
            "n_remaining": len(remaining), "by_category": by_category,
            "done_dirs": done_dirs,
            "remaining": remaining}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-dir",
                    default="logs/experiments/review_pack_20260926")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list-remaining", type=int, default=0,
                    help="列出前 N 个还没标的帧")
    args = ap.parse_args(argv)
    st = status(Path(args.pack_dir))
    print(f"[review-status] 全量清单 {st['n_total']} 帧：已标 {st['n_done']}、"
          f"还差 {st['n_remaining']}")
    print(f"{'类别':<22s} {'已标':>5s} {'总数':>5s} {'还差':>5s}")
    for c, v in st["by_category"].items():
        print(f"{v['label']:<22s} {v['done']:>5d} {v['total']:>5d} "
              f"{v['remaining']:>5d}")
    for d in st["done_dirs"]:
        print(f"[review-status]   已标目录 {d['dir']}：{d['frames']} 帧")
    out = Path(args.out) if args.out else (Path(args.pack_dir)
                                           / "review_status.json")
    out.write_text(json.dumps(st, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[review-status] -> {out}")
    if args.list_remaining:
        for r in st["remaining"][:int(args.list_remaining)]:
            print(f"[review-status]   待标 {CATEGORY_LABEL.get(r['category'], r['category'])}: {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
