"""把"该人工修订哪几帧"变成**可以直接打开的任务包**。

为什么需要它：本项目唯一挡住晋级的硬门是标线/身份指标，而它们现在是 UNKNOWN——
引擎标注不给 line 类（漆线在引擎眼里就是沥青），所以标线真值只能靠人工画。
一轮无人值守采集放完复核队列（``review_queue_collect_*.json``，按漆线像素降序）之后，
人还要自己从 120 帧里挑帧、自己拼命令行；这一步的摩擦就是"标线真值一直没补上"的
直接原因。本脚本把这一步做成两条命令。

它写出什么（``<out>/``）：

* ``<out>/<view>/frame_*.npz``：只带 ``colour`` 的帧副本。**故意不带 ``label``**：
  带 label 会命中标注器的"续标"分支（``_initial_label`` 优先用已有 label），
  于是 ``--prefill-model`` 不生效、人得从空白画；去掉 label 才能让模型预填。
* ``<out>/<view>/meta.json``：从采集目录复制的原 meta（``map_name``/``source_id``/
  每帧 ``pos``/``heading``/``exposure``）。身份必须**带着走**——3h 轮有 48 帧人工
  修订因为丢了身份被审计拒收且不可恢复，这条路不能再走一次。
* ``<out>/README.md``：选帧依据、键位、可直接粘贴的命令、回传什么、怎么自查身份。

用法::

    python scripts/m5_annotate_package.py \\
        --review-queue logs/experiments/<run>/review_queue_collect_<stamp>.json \\
        --out logs/m5_seg/annotate_pkg_<date> --per-view 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

#: 标注器的身份字段与 npz 身份封装（复用同一份实现，不另写一套口径）
from scripts.m5_annotate_manual import (  # noqa: E402
    identity_npz_extras, load_frame_dir,
)

DEFAULT_PREFILL = "logs/m5_seg/seg_model_v13b_ft_t13/best.pt"

KEY_HELP = """\
界面是中文，顶端有一条工具栏，**所有操作都能用鼠标点**（按钮上印着等价的键）：

| 顶端按钮 | 等价的键 | 作用 |
| --- | --- | --- |
| 标线 / 路面 / 擦除 | `1` / `2` / `3` | 像素类别：标线 / 路面 / 背景（擦除） |
| 沥青 / 碎石 / 路肩 | `7` / `8` / `9` | **路型**（单独一组，不和"路面"混）：逐像素记进 `road_type`（1/2/3）。沥青与纯土路算路面；**路肩不算路面**（写背景），按约束"有铺装时土肩不得算作道路" |
| 遮挡 / 模糊 / 未知 | `4` / `5` / `6` | 标记"这里为什么画不了"：写 255(ignore) + 原因 |
| 画笔 | `p` | 画笔：左键拖动，涂当前画笔 |
| 油漆桶 | `f`（或右键） | 填连通区域：先画闭合轮廓，再点内部 |
| 直线 | `l` | 拖动 A→B，或点两下（A 再 B）—— 画出来是**笔直**的 |
| 曲线 | `v` | 平滑曲线：拖动 = 自由手绘（自动去抖 + 平滑）；或点若干控制点后按 `ENTER`，曲线穿过每个点 |
| 撤回 | `u` | 画曲线时退掉最后一个控制点，否则退掉上一笔 |
| 清空 / 缩放 / 上一帧 / 保存并下一帧 | `c` / `z` / `a`(或 ←) / `s` | 对应操作；`q` 退出 |
| 拖动条 | — | 笔刷大小 1–40 |

草稿（还没落笔的直线/曲线）按 `ESC` 放弃，或再点一次当前工具按钮；
**草稿不会写进标签**，只有落笔才写。碎石与路肩在画面上有各自的颜色，
不存在"画了看不见"。
"""


def load_review_queue(path: Path) -> tuple:
    """``(collection_dir, frames)``；队列里没有的字段不猜。"""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    out_dir = str(blob.get("out_dir") or "")
    frames = [f for f in (blob.get("frames") or []) if f.get("path")]
    return out_dir, frames


def pick_per_view(frames: list, per_view: int) -> list:
    """按视角各取前 N 帧（队列本身按漆线像素降序，这里不重排、不插值）。"""
    seen: dict = {}
    picked: list = []
    for f in frames:
        view = str(f.get("view") or "")
        if not view or seen.get(view, 0) >= per_view:
            continue
        seen[view] = seen.get(view, 0) + 1
        picked.append(f)
    return picked


def build_package(*, collection: Path, frames: list, out: Path,
                  per_view: int = 8, prefill: str = DEFAULT_PREFILL,
                  seed: str = "") -> dict:
    """写任务包；返回 ``{view: {"frames": n, "identity_ok": n, ...}}``。"""
    collection = Path(collection)
    out = Path(out)
    # meta 的位置有两种（实测都遇到过）：
    #   * 采集级 ``<collection>/meta.json``（采集器写的，含全部视角）；
    #   * **视角级** ``<collection>/<view>/meta.json``（agent 标注池就是这种：
    #     每个视角目录各有一份，里面列着全部视角的帧）。
    # 只找采集级会在视角级池子上直接 FileNotFoundError（实测踩到）。
    meta_src = collection / "meta.json"
    picks = pick_per_view(frames, per_view)
    per: dict = {}
    for f in picks:
        view = str(f["view"])
        src = collection / str(f["path"])
        if not src.exists():
            per.setdefault(view, {"frames": 0, "missing": []})
            per[view]["missing"].append(str(src))
            continue
        d = out / view
        d.mkdir(parents=True, exist_ok=True)
        _meta_here = meta_src if meta_src.is_file() else (collection / view
                                                          / "meta.json")
        if not _meta_here.is_file():
            per.setdefault(view, {"frames": 0, "missing": []})
            per[view]["missing"].append(
                f"meta.json for {src} (tried {meta_src} and "
                f"{collection / view / 'meta.json'})")
            continue
        meta_src = _meta_here
        _blob = json.loads(meta_src.read_text(encoding="utf-8"))
        # 视角级 meta 列着**全部视角**的帧：只保留本包真正带的那些，避免
        # 包里的 meta 出现"并不存在的帧"（下游按 meta 记身份就会记错）。
        _names = {Path(str(x["path"])).name for x in picks
                  if str(x["view"]) == view}
        if _blob.get("frames"):
            # 必须按**视角+文件名**匹配：不同视角的同名帧（front_main/
            # frame_00000.npz 与 rear/frame_00000.npz）文件名相同，只按文件名
            # 过滤会把别的视角的帧一起留下（实测：4 帧的包留下 32 条记录）。
            _want = {f"{view}/{n}" for n in _names}

            def _keep(r) -> bool:
                rp = str(r.get("path") or "")
                if rp:
                    return rp in _want
                return (str(r.get("view") or "") == view
                        and Path(rp).name in _names)
            _cat = {Path(str(x.get("path") or "")).name: str(
                x.get("category") or "") for x in picks
                if str(x["view"]) == view}
            _kept = []
            for r in _blob["frames"]:
                if not _keep(r):
                    continue
                # 类别是**复核用**的候选归类：写进包内 meta，监督器直接读，
                # 不再靠工作表按源路径猜（包内是副本，按文件名会串味）。
                _c = _cat.get(Path(str(r.get("path") or "")).name)
                if _c:
                    r = {**r, "category": _c}
                _kept.append(r)
            _blob["frames"] = _kept
            _blob["views_in_package"] = [view]
        (d / "meta.json").write_text(
            json.dumps(_blob, ensure_ascii=False, indent=1), encoding="utf-8")
        with np.load(src) as z:
            payload = {"colour": np.asarray(z["colour"], dtype=np.uint8)}
            if "annotation_raw" in z.files:
                payload["annotation_raw"] = np.asarray(z["annotation_raw"],
                                                       dtype=np.uint8)
        # 身份：先用帧自己的 npz 字段，再用采集 meta 里按 basename 匹配的记录
        ident = {k: None for k in ("map_name", "source_id", "pos", "heading")}
        _blob2 = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        for rec in (_blob2.get("frames") or []):
            if Path(str(rec.get("path") or "")).name == src.name:
                ident.update({"map_name": rec.get("map_name"),
                              "source_id": rec.get("source_id"),
                              "pos": rec.get("pos"),
                              "heading": rec.get("heading")})
                break
        run_meta = _blob2
        ident["map_name"] = ident["map_name"] or run_meta.get("map_name")
        ident["source_id"] = ident["source_id"] or run_meta.get("source_id")
        np.savez_compressed(d / src.name, **payload,
                            **identity_npz_extras(ident))
        st = per.setdefault(view, {"frames": 0, "missing": [],
                                   "paint_px": 0, "pos_first": None})
        st["frames"] += 1
        st["paint_px"] += int(f.get("line_pixels") or 0)
        if st["pos_first"] is None:
            st["pos_first"] = f.get("pos")
    # 一帧都没拷到时给可读错误：原来会先写 README，而 out/<view> 没被创建 ->
    # FileNotFoundError 栈（实测踩到：队列里的路径与 collection 拼不上时）
    copied = sum(int(v.get("frames") or 0) for v in per.values())
    if copied == 0:
        print(f"[pkg] 一帧都没拷到：检查队列里的 path 与 --collection 能否拼上"
              f"（collection={collection}，前几个缺失："
              f"{[m for v in per.values() for m in (v.get('missing') or [])][:3]}）")
        return per
    out.mkdir(parents=True, exist_ok=True)
    _write_readme(out, collection=collection, per=per, prefill=prefill,
                  seed=seed, per_view=per_view)
    return per


def _write_readme(out: Path, *, collection: Path, per: dict, prefill: str,
                  seed: str, per_view: int) -> None:
    views = "、".join(sorted(per))
    lines = [
        "# 标线真值人工修订任务包",
        "",
        f"来源采集：`{collection}`" + (f"（{seed}）" if seed else ""),
        f"每个视角取漆线像素最多的 {per_view} 帧（复核队列按 `line_pixels` 降序，"
        f"没有重排）：{views}",
        "",
        "## 为什么是这几帧",
        "",
        "引擎标注里漆线被画成沥青（line 类为空），所以标线真值只能人工画。"
        "这些帧是**采到的数据里漆线像素最多**的帧——先修它们，性价比最高。",
        "包里的 npz 故意**不带 label**：这样标注器的 `--prefill-model` 会生效"
        "（带 label 会走续标分支，模型预填被跳过）。",
        "",
        "## 每帧有多少漆线像素（实测）",
        "",
    ]
    for view in sorted(per):
        st = per[view]
        lines.append(f"* `{view}`：{st['frames']} 帧，漆线像素合计 "
                     f"{st.get('paint_px', 0)}，起点 `{st.get('pos_first')}`"
                     + (f"，**缺 {len(st['missing'])} 个源文件**"
                        if st.get("missing") else ""))
    lines += [
        "",
        "## 怎么跑（PowerShell 7，逐视角跑；`--out` 用同一个目录 = 原地保存）",
        "",
        "```pwsh",
    ]
    for view in sorted(per):
        lines += [
            f".venv\\Scripts\\python.exe scripts\\m5_annotate_manual.py `",
            f"    --frames-dir {out.as_posix()}/{view} `",
            f"    --out {out.as_posix()}/{view} `",
            f"    --prefill-model {prefill}",
            "",
        ]
    lines += [
        "```",
        "",
        "## 键位",
        "",
        KEY_HELP,
        "## 画完回传",
        "",
        "1. 每帧按 `s` 保存后才算数（保存会写 `<view>/meta.json` 侧车 + 带身份的 npz）。",
        "2. 自查身份是否跟着走（这一步是硬要求：3h 轮有 48 帧因丢身份被判死）：",
        "",
        "```pwsh",
        f".venv\\Scripts\\python.exe scripts\\m5_seg_dataset_audit.py "
        f"--runs {out.as_posix()}/<view>",
        "```",
        "",
        "3. 审计里 `paint_valid_frames > 0`、`rejected = 0` 就说明这份标注可以被训练"
        "与硬门消费；`rejected` 不为 0 的帧不要删，留着看原因。",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-queue", required=True,
                    help="复核队列 JSON（review_queue_collect_*.json）")
    ap.add_argument("--collection", default=None,
                    help="采集目录（缺省用队列里记录的 out_dir）")
    ap.add_argument("--out", required=True, help="任务包输出目录")
    ap.add_argument("--per-view", type=int, default=8)
    ap.add_argument("--prefill-model", default=DEFAULT_PREFILL)
    args = ap.parse_args(argv)

    q_path = Path(args.review_queue)
    if not q_path.exists():
        print(f"[pkg] 复核队列不存在：{q_path}")
        return 2
    coll, frames = load_review_queue(q_path)
    collection = Path(args.collection or coll)
    if not collection.exists():
        print(f"[pkg] 采集目录不存在：{collection}（用 --collection 指定）")
        return 2
    if not frames:
        print("[pkg] 队列里没有帧：不产出空任务包")
        return 3
    out = Path(args.out)
    per = build_package(collection=collection, frames=frames, out=out,
                        per_view=int(args.per_view),
                        prefill=args.prefill_model, seed=q_path.stem)
    total = sum(v["frames"] for v in per.values())
    if not total:
        print("[pkg] 一帧都没复制成功：源文件不在？")
        return 3
    # 身份自查：用标注器自己的读法读回来，读不到身份就当失败（不是警告）
    print(f"[pkg] 任务包 -> {out}（{total} 帧，视角 {len(per)}）")
    bad = []
    for view in sorted(per):
        d = out / view
        _frames, idents, _r, _p = load_frame_dir(d)
        missing = [i for i, it in enumerate(idents)
                   if not (it.get("map_name") and it.get("source_id")
                           and it.get("pos") and it.get("heading"))]
        print(f"[pkg]   {view}: {per[view]['frames']} 帧，"
              f"身份齐全 {len(idents) - len(missing)}/{len(idents)}")
        if missing:
            bad.append(f"{view}:{len(missing)}")
    if bad:
        print(f"[pkg] 身份不齐：{bad} —— 先修包再让人画（画了也会被审计拒收）")
        return 4
    print(f"[pkg] README: {out / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
