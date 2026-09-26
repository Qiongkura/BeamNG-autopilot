"""W1 §6.3 复核包：把"该复核哪些帧"变成可打开的任务包 + 可查的工作表。

方案原文：「先建设一个可负担、可逐帧核查的工程验收集」「选择至少 6 个完整路段组，
避免把连续相邻帧当独立样本」「先用 20 帧混合包计时和统一标注规则，再估算余量，
**禁止承诺"十分钟审核全部 640 帧"**」。

本脚本产出三件东西（都在 ``--out`` 下）：

* ``review_pack.json``：**起步 20 帧混合包**的复核队列（直接喂给
  ``m5_annotate_package.py``，生成可打开的包）；每帧带类别、来源组、证据与
  "为什么选它"；
* ``review_worksheet.json`` / ``review_worksheet.md``：逐帧工作表，含空字段
  ``reviewer`` / ``reviewed_at`` / ``verdict`` / ``regions``——验收要求"能查询任意帧
  是谁、何时、对哪些类别和区域做了复核"，这张表就是那个查询的载体；
* ``review_plan.json``：**全量 120 帧**（6 类 × 20）的候选清单与**缺口**（某个类别
  在现有池子里凑不齐 20 帧时明确写出来，不拿别的类别凑数）。

类别与证据（**类别是候选归类，最终以复核人看到的为准**）：

| 类别 | 取材依据 |
| --- | --- |
| 清晰漆线 | meta 的 ``line_pixels`` 最高的一批 |
| 弯道与路口 | 弯道/路口采集组（``diverse_curve``、``t13_corner``、``t13_junction2``） |
| 退化与遮挡 | 未知像素占比高（agent 标签里 255 的比例） |
| 易混淆纹理 | E0 实测"路外假线"最重的场景（``plain`` 组） |
| 真无线铺装路 | ``line_pixels`` 近零但路面像素多的帧（**待复核确认"确实没有线"**） |
| 铺装/土肩与纯土路 | ``dirt_road_labeled`` 等土路目录 |

用法::

    .venv\\Scripts\\python.exe scripts\\m5_review_pack.py \\
        --out logs/experiments/review_pack_20260926 [--starter-per-category 4]
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

from beamng_autopilot.experiments.manifest import dir_group  # noqa: E402

#: 六个类别（方案 §6.3 的表）
CATEGORIES = (
    ("clear_paint", "清晰漆线", "左/右边缘、中央线、白/黄线、实/虚线"),
    ("curve_junction", "弯道与路口", "斜线、断线、交汇、不同远近尺度"),
    ("degraded_occluded", "退化与遮挡", "阴影、磨损、车辆遮挡、亮度变化"),
    ("confusable_texture", "易混淆纹理", "路缘石、排水槽、护栏反光、墙面、轮胎印"),
    ("no_line_pavement", "真无线铺装路", "能确认整片评价区域无线，保留边界信息"),
    ("dirt_shoulder", "铺装/土肩与纯土路", "两种道路类型分别标记，不能混成 road 一类"),
)

#: 路型取值与必填说明：方案要求"两种道路类型分别标记，不能混成 road 一类就结束"，
#: 而 worksheet 是复核结论的唯一载体，所以直接把 road_type 写进那几帧的 notes
#: （不另造文件，也不让审计去猜字段在哪）。
ROAD_TYPE_VALUES = ("gravel", "asphalt", "shoulder")
ROAD_TYPE_NOTE = "road_type=（gravel / asphalt / shoulder）"
ROAD_TYPE_CATEGORIES = ("dirt_shoulder", "no_line_pavement")


def worksheet_notes(category: str) -> str:
    """这一帧的 notes 预填什么：土路/无标线的帧必须补 road_type。

    抽成函数是为了能离线断言"哪几类被要求补路型"——这段逻辑藏在 main() 里时，
    少写一个类别不会有任何东西报警，而漏掉的正是方案点名的两类之一。
    """
    return ROAD_TYPE_NOTE if str(category) in ROAD_TYPE_CATEGORIES else ""


def worksheet_rows(starter: list) -> list:
    """逐帧工作表行：复核人填 reviewer/reviewed_at/verdict/regions(+notes)。"""
    return [{"path": f["path"], "view": f["view"], "group": f["group"],
             "category": f["category"], "why_selected": f["why"],
             "reviewer": "", "reviewed_at": "", "verdict": "",
             "regions": "", "notes": worksheet_notes(f["category"])}
            for f in starter]


def worksheet_table_md(rows: list) -> list:
    """工作表表头 + 逐帧行（含 notes 列：路型就写在那一列里）。"""
    md = ["| # | 帧 | 组 | 类别 | 选帧证据 | reviewer | reviewed_at | "
          "verdict | regions | notes |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | `{r['path']}` | `{r['group']}` | {r['category']} | "
                  f"{r['why_selected']} | | | | | {r['notes']} |")
    return md

#: 每个类别的取材目录（front_main 优先：评价主视角）
SOURCES = {
    "clear_paint": ("logs/m5_seg/line_truth_agent_full_20260925/town/front_main",),
    "curve_junction": ("logs/m5_seg/diverse_curve_20260924/front_main",
                       "logs/m5_seg/t13_junction2/front_main",
                       "logs/m5_seg/t13_corner/front_main"),
    # 退化与遮挡：除了宽/镇两组，也收 2026-09-26 两轮补采——砾石那轮有很重的
    # 树影（阴影/亮度变化），无标线铺装那轮有岩壁反光与背光。
    "degraded_occluded": ("logs/m5_seg/line_truth_agent_full_20260925/wide/front_main",
                          "logs/m5_seg/line_truth_agent_full_20260925/town/front_main",
                          "logs/m5_seg/collect_t14_collect_dirt_20260926_20260926_123625/front_main",
                          "logs/m5_seg/collect_t14_collect_plain2_20260926_20260926_123944/front_main"),
    # 2026-09-26 补采（都带身份、四视角全程零漆线像素）：
    #   * collect_..._plain2...（italy/ring_20260926_123950）：无标线**铺装**路，
    #     左护栏 + 右岩壁 + 砾石路肩 -> 同时喂"真无线铺装路"与"易混淆纹理"；
    #   * collect_..._dirt...（italy/ring_20260926_123638）：砾石土路 -> 土路类别。
    "confusable_texture": ("logs/m5_seg/diverse_plain_20260924/front_main",
                           "logs/m5_seg/line_truth_agent_full_20260925/plain/front_main",
                           "logs/m5_seg/collect_t14_collect_plain2_20260926_20260926_123944/front_main"),
    "no_line_pavement": ("logs/m5_seg/line_truth_agent_full_20260925/plain/front_main",
                         "logs/m5_seg/collect_t14_collect_plain2_20260926_20260926_123944/front_main"),
    # 土路：旧目录（dirt_road_labeled / dirt_road_pkg）**没有地图身份**，帧进不了
    # 评价集（审计拒收），所以 2026-09-26 在 italy 上沿 plain 区航向外推 400 m
    # 重采了一次（砾石路、四视角全程零漆线像素，身份 italy/ring_20260926_123638）。
    # 旧的 45 帧仍留在盘上作为"手涂 road 参照"，但不进评价包。
    "dirt_shoulder": ("logs/m5_seg/collect_t14_collect_dirt_20260926_20260926_123625/front_main",
                      "logs/m5_seg/dirt_road_labeled"),
}


def _unknown_frac(npz: Path) -> float | None:
    """未知像素占比（agent 标签用 255 表示未复核/未知）；读不到返回 None。"""
    try:
        z = np.load(npz)
        lab = np.asarray(z["label"]) if "label" in z else None
        if lab is None or lab.size == 0:
            return None
        return round(float((lab == 255).sum()) / float(lab.size), 4)
    except Exception:                                      # noqa: BLE001
        return None


def _frames_of(d: Path) -> list:
    """一个目录的逐帧记录：身份、视角、曝光、位姿、漆线像素、未知占比。"""
    meta_p = d / "meta.json"
    meta = {}
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            meta = {}
    elif (d.parent / "meta.json").is_file():
        try:
            meta = json.loads((d.parent / "meta.json").read_text(
                encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            meta = {}
    by_rel = {str(f.get("path") or ""): f for f in (meta.get("frames") or [])}
    n_views = len({str(f.get("view") or "") for f in (meta.get("frames") or [])})
    by_name = ({str(f.get("path") or "").split("/")[-1]: f
                for f in (meta.get("frames") or [])} if n_views <= 1 else {})
    out = []
    for f in sorted(d.glob("frame_*.npz")):
        rec = (by_rel.get(f"{d.name}/{f.name}") or by_rel.get(str(f))
               or by_name.get(f.name) or {})
        out.append({"view": str(rec.get("view") or d.name),
                    "path": str(f), "line_pixels": rec.get("line_pixels"),
                    "exposure": rec.get("exposure"), "pos": rec.get("pos"),
                    "heading": rec.get("heading"),
                    "group": dir_group(d),
                    "unknown_frac": _unknown_frac(f)})
    return out


def _pick(pool: list, n: int, *, key, reverse: bool = True,
          max_per_group: int = 2) -> list:
    """按 ``key`` 排序、**按组轮转**取 n 帧；单组不超过 ``max_per_group``。

    两条纪律（方案 §6.3）：
    * 「避免把连续相邻帧当独立样本」——所以**轮转**而不是"某一组取满"，
      让选出的帧尽量分散到不同组；
    * 缺测不静默：目录没有 meta（``line_pixels`` 全缺）时回退到文件名顺序并在
      ``fallback`` 里写明原因，而不是把整类静默丢成 0 帧（实测踩到：土路目录
      没有 meta，45 帧一帧都没选出来）。
    """
    keyed = [f for f in pool if key(f) is not None]
    fallback = ""
    if not keyed:
        fallback = ("this pool has no usable key (directory has no meta with "
                    "line_pixels): picked by path order")
        keyed = sorted(pool, key=lambda f: str(f.get("path")))
    else:
        keyed.sort(key=key, reverse=reverse)
    by_group: dict = {}
    for f in keyed:
        by_group.setdefault(f["group"], []).append(f)

    def _best(g):
        v = key(by_group[g][0])
        return 0.0 if v is None else float(v)

    order = sorted(by_group, key=lambda g: (_best(g), str(g)), reverse=True)
    out, counts, guard = [], {}, 0
    while len(out) < n and guard <= n + 2:
        progressed = False
        for g in order:
            if len(out) >= n:
                break
            idx = counts.get(g, 0)
            if idx >= max_per_group or idx >= len(by_group[g]):
                continue
            out.append(by_group[g][idx])
            counts[g] = idx + 1
            progressed = True
        guard += 1
        if not progressed:
            break
    for f in out:
        if fallback:
            f["fallback"] = fallback
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--starter-per-category", type=int, default=4,
                    help="起步混合包每类帧数（4 类 × ... 默认 4，凑 20 帧左右）")
    ap.add_argument("--plan-per-category", type=int, default=20,
                    help="全量清单每类帧数（方案 §6.3 建议 20）")
    ap.add_argument("--view", default="front_main", help="评价主视角")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pools: dict = {}
    for cat, _label, _need in CATEGORIES:
        frames: list = []
        for src in SOURCES[cat]:
            d = ROOT / src
            if d.is_dir():
                frames += [f for f in _frames_of(d)
                           if f["view"] == args.view or cat == "dirt_shoulder"]
        pools[cat] = frames

    def _max_per_group(pool: list, n: int) -> int:
        """全量清单的单组上限：把 n 帧尽量均分到池子里的组（至少 2）。"""
        n_groups = len({f["group"] for f in pool}) or 1
        return max(2, -(-int(n) // n_groups))

    def _sel(cat: str, n: int, *, for_plan: bool = False,
             pool: list | None = None) -> list:
        pool = pools[cat] if pool is None else pool
        mpg = _max_per_group(pool, n) if for_plan else 2
        if cat == "clear_paint":
            return _pick(pool, n, key=lambda f: f.get("line_pixels"),
                         max_per_group=mpg)
        if cat == "no_line_pavement":
            # 漆线像素近零、但要有路面（否则是空白帧）
            cand = [f for f in pool if (f.get("line_pixels") or 0) <= 10]
            return _pick(cand, n, key=lambda f: f.get("unknown_frac"),
                         reverse=False, max_per_group=mpg)
        if cat == "degraded_occluded":
            return _pick(pool, n, key=lambda f: f.get("unknown_frac"),
                         max_per_group=mpg)
        if cat == "confusable_texture":
            # 实测依据：E0 里这个场景的路外假线比 0.49–0.71（模型在这里画错线）
            return _pick(pool, n, key=lambda f: f.get("line_pixels"),
                         max_per_group=mpg)
        return _pick(pool, n, key=lambda f: f.get("line_pixels"),
                     max_per_group=mpg)

    starter: list = []
    plan: dict = {}
    # 跨类别去重：同一次采集（同 group）的**同一帧**可能被两个类别都选中
    # （实测：pkg_plain 与 pkg_diverse_plain_20260924 是同一次采集，同一帧进了
    # 两个包，审计按 colour 内容隔离了 1 帧，人工白标一遍）。
    _used_frames: set = set()

    def _key(f) -> tuple:
        """同一张底层帧的键：**组 + 曝光**。

        不能用文件名：同一次采集在不同目录里编号不同（实测：diverse_plain 的
        frame_00007 与 agent 池 plain 的同名帧其实是同一次采集的同一些帧，
        文件名一样但那是巧合；换一对目录就不一样了）。曝光缺失时才退回文件名。
        """
        e = f.get("exposure")
        if e is not None:
            return (str(f.get("group")), "e", int(e))
        return (str(f.get("group")), "n", Path(str(f.get("path"))).name)

    def _fresh(pool: list) -> list:
        """去掉已被别的类别选走的帧，并在**池内**去重。

        两处都要（实测都踩到）：
        * 类别之间：选完再删会让批次缩水（town 从 4 帧掉到 3 帧），所以先过滤；
        * 类别之内：一个类别的取材目录可能含**同一张帧的两份拷贝**
          （diverse_plain 与 agent 池的 plain 是同一次采集），池内不去重就会
          同批出现两份，人工白标一遍。
        """
        out, seen = [], set()
        for f in pool:
            k = _key(f)
            if k in _used_frames or k in seen:
                continue
            seen.add(k)
            out.append(f)
        return out

    def _mark(frames: list) -> list:
        for f in frames:
            _used_frames.add(_key(f))
        return frames

    # 只收**有身份**的帧（方案 §7.1：缺身份的帧进隔离队列，不从目录名猜地图）。
    # 实测踩到：土路目录没有 meta.json，选出来的帧没有 map/source_id——
    # 这种帧进不了评价集（审计会拒收），必须先排除并记缺口。
    for cat, _l, _n in CATEGORIES:
        keep, dropped = [], 0
        for f in pools[cat]:
            if str(f["group"]).startswith("dir/"):
                dropped += 1
                continue
            keep.append(f)
        pools[cat] = keep
        if dropped:
            pools.setdefault("_no_identity", {})[cat] = dropped
    # 选帧顺序按**池子大小升序**：多个类别共用同一批目录时，先满足稀缺的类别，
    # 否则靠后的类别会被前面的吃光（实测：真无线铺装路只剩 1/20，因为
    # 易混淆纹理先把它同一批目录里的帧选走了）。输出仍按 CATEGORIES 的规范顺序。
    _order = sorted(CATEGORIES, key=lambda c: (len(pools[c[0]]), c[0]))
    for cat, label, need in _order:
        n_starter = max(1, int(args.starter_per_category)) \
            if pools[cat] else 0
        picked = _mark(_sel(cat, n_starter, pool=_fresh(pools[cat])))
        for f in picked:
            starter.append({**f, "category": cat, "category_label": label,
                            "why": f"{label}（{need}）："
                                   f"line_px={f.get('line_pixels')} "
                                   f"unknown={f.get('unknown_frac')}"})
        full = _mark(_sel(cat, int(args.plan_per_category), for_plan=True,
                          pool=_fresh(pools[cat])))
        plan[cat] = {
            "label": label, "needs": need,
            "n_available": len(pools[cat]), "n_selected": len(full),
            "gap": max(0, int(args.plan_per_category) - len(full)),
            "gap_note": ("" if len(full) >= int(args.plan_per_category) else
                         f"现有池子只能提供 {len(full)} 帧（缺 "
                         f"{int(args.plan_per_category) - len(full)}）："
                         "需要新采集或复核其它视角才能补齐，**不拿别的类别凑数**")
                        + ((f"；另有 {pools.get('_no_identity', {}).get(cat, 0)} "
                            "帧因**缺地图身份**被排除（无 meta，不能进评价集）")
                           if pools.get("_no_identity", {}).get(cat) else ""),
            "frames": [{k: f.get(k) for k in
                        ("path", "view", "group", "line_pixels",
                         "unknown_frac", "exposure")} for f in full],
            "by_group": {g: sum(1 for f in full if f["group"] == g)
                         for g in sorted({f["group"] for f in full})},
            "n_groups": len({f["group"] for f in full}),
            # 方案 §6.3：「关键场景尽量覆盖两个独立组，达不到就**显式记录缺口**」。
            # 只有一个组时，这 20 帧里大部分是相邻帧，独立性弱——必须写出来，
            # 不能让它看起来像"这类场景已经覆盖充分"。
            "group_gap_note": ("" if len({f["group"] for f in full}) >= 2 else
                               "只有 1 个独立组：这类场景的帧多为相邻帧，"
                               "独立性弱，需显式记为缺口（补采同类的另一组，"
                               "或在报告里降级为研究档）"),
        }

    # 起步包按类别交错排（复核时不必连续看同一类）
    starter.sort(key=lambda f: (f["category"], str(f.get("path"))))
    starter_by_cat: dict = {}
    for f in starter:
        starter_by_cat[f["category"]] = starter_by_cat.get(f["category"], 0) + 1
    pack = {"why": "W1 §6.3 起步混合包：六类各若干帧，同组最多 2 帧"
                   "（避免把相邻帧当独立样本）。类别是**候选归类**，"
                   "最终以复核人看到的为准。",
            "view": args.view, "n_frames": len(starter),
            "by_category": starter_by_cat,
            "note": "起步包故意小：它的用途是**计时与统一标注规则**，"
                    "不是统计充分性；全量清单见 review_plan.json",
            "frames": [{k: f.get(k) for k in
                        ("view", "path", "line_pixels", "pos", "heading",
                         "exposure")} | {"category": f["category"],
                                         "group": f["group"]}
                       for f in starter]}
    (out / "review_pack.json").write_text(
        json.dumps(pack, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- 全量可标注包（方案 §6.3 的 6 类 × 20 = 120，实际 117）----------
    full_meta = []
    for cat, label, need in CATEGORIES:
        for f in plan[cat]["frames"]:
            full_meta.append({**f, "category": cat, "category_label": label})
    # 并入**已经复核过**的帧：它们是同一批池子里选出来的，没必要重标；
    # 按 (组, 曝光) 去重，并标 already_reviewed 供监督器提示。
    reviewed_dirs = sorted((Path(args.out) / "reviewed").glob("*/front_main"))
    starter_cat = {}
    ws = Path(args.out) / "review_worksheet.json"
    if ws.is_file():
        try:
            for row in (json.loads(ws.read_text(encoding="utf-8")).get("rows")
                        or []):
                starter_cat[str(row.get("path"))] = row.get("category")
        except Exception:                                  # noqa: BLE001
            starter_cat = {}
    have = {(str(f.get("group")), f.get("exposure")) for f in full_meta}
    n_done = 0
    for d in reviewed_dirs:
        for f in sorted(d.glob("frame_*.npz")):
            key = None
            cat = starter_cat.get(str(f), "")
            # 组与曝光：从所在包的 meta 里读（复核目录的 meta 是采集 meta 的子集）
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                rec = next((r for r in (meta.get("frames") or [])
                            if Path(str(r.get("path") or "")).name == f.name),
                           {})
                key = (str(meta.get("map_name") or "") + "/"
                       + str(meta.get("source_id") or ""), rec.get("exposure"))
            except Exception:                              # noqa: BLE001
                key = None
            if key and key in have:
                n_done += 1          # 已在全量清单里：算已完成
                continue
            full_meta.append({
                # 必须存**绝对路径**：打包工具按 collection/path 找源帧，相对路径
                # 会拼成 collection/logs/... 从而全部"缺失"（实测踩到）
                "path": str(Path(f).resolve()), "view": d.name,
                "group": (key or ("", ""))[0],
                "exposure": (key or (None, None))[1],
                "line_pixels": None, "pos": None, "heading": None,
                "category": cat or "already_reviewed",
                "category_label": "已复核（并入全量包）",
                "already_reviewed": True})
            if key:
                have.add(key)
    print(f"[review-pack] 已复核帧：{n_done} 帧已在全量清单里、"
          f"{len([f for f in full_meta if f.get('already_reviewed')])} 帧并入")
    full_pack = {
        "why": "W1 §6.3 **全量评价集**：6 类各 20 帧（实际 117，缺口见 "
               "review_plan.json）。同组帧按曝光去重；类别是候选归类，"
               "最终以复核人看到的为准。",
        "view": args.view, "n_frames": len(full_meta),
        "by_category": {c: plan[c]["n_selected"] for c, _l, _n in CATEGORIES},
        "frames": [{k: f.get(k) for k in
                    ("view", "path", "line_pixels", "pos", "heading",
                     "exposure")} | {"category": f["category"],
                                     "group": f["group"]}
                   for f in full_meta],
    }
    (out / "review_pack_full.json").write_text(
        json.dumps(full_pack, indent=1, ensure_ascii=False), encoding="utf-8")

    worksheet = {
        "why": "逐帧复核工作表：验收要求『能查询任意帧是谁、何时、对哪些类别和区域"
               "做了复核』——复核人填 reviewer/reviewed_at/verdict/regions，"
               "土路与无标线的帧还要在 notes 里补 road_type。本文件就是那个查询"
               "的载体（不要另造一份）。",
        "categories": {c: {"label": lab, "needs": need}
                       for c, lab, need in CATEGORIES},
        "road_type_values": list(ROAD_TYPE_VALUES),
        "road_type_note": ROAD_TYPE_NOTE,
        "rows": worksheet_rows(starter),
        "verdict_values": ["confirm", "relabel", "reject", "unsure"],
    }
    (out / "review_worksheet.json").write_text(
        json.dumps(worksheet, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# W1 §6.3 复核工作表（起步包）", "",
          f"共 {len(worksheet['rows'])} 帧；`verdict` 取值：confirm / relabel / "
          "reject / unsure；`regions` 写「你实际看过并确认的区域」（例如 "
          "`line_left,line_right,road_edge`）。", "",
          f"`dirt_shoulder`（土路/土肩）与 `no_line_pavement`（真无线铺装路）的帧，"
          f"notes 里已预填 `{ROAD_TYPE_NOTE}`，请改成实测值——这两类必须分别标出"
          "路型，不能只写 road 就结束（方案原文）。", ""]
    md += worksheet_table_md(worksheet["rows"])
    md += ["", "## 全量清单（6 类 × 20）的可用量与缺口", "",
           "| 类别 | 可用 | 已选 | 缺口 | 说明 |", "| --- | --- | --- | --- | --- |"]
    for cat, _l, _n in CATEGORIES:
        p = plan[cat]
        md.append(f"| {p['label']} | {p['n_available']} | {p['n_selected']} | "
                  f"{p['gap']} | {p['gap_note'] or '够'} |")
    md += ["", "## 复核时间怎么估（方案要求先计时再估余量）", "",
           "先用这 20 帧混合包实测一次：记下每帧耗时（含读图、判断、画线、"
           "保存），再乘全量帧数。**不要**在没测之前承诺"
           "『十分钟审核全部 640 帧』。", ""]
    (out / "review_worksheet.md").write_text("\n".join(md), encoding="utf-8")
    (out / "review_plan.json").write_text(
        json.dumps({"categories": plan,
                    "n_total_selected": sum(p["n_selected"]
                                            for p in plan.values())},
                   indent=1, ensure_ascii=False), encoding="utf-8")

    # 打包要**按采集分开**：标注包的身份是"一次采集一个 meta（map/source_id）"，
    # 而起步包跨了多个采集——合成一个包会让身份张冠李戴（标注工具写回的凭证
    # 也就无法归属）。所以逐采集写队列，并用现成的单采集工具各打一个包。
    def _collection_of(frame_path: str) -> Path:
        """帧路径 -> 它的采集目录（view 目录的父目录）。"""
        return Path(frame_path).parent.parent

    def _split_and_emit(frames: list, tag: str, why: str) -> tuple:
        """把一批帧按采集拆成队列 + 生成打包脚本（返回 (cmds, ps1)）。"""
        per_coll: dict = {}
        for f in frames:
            per_coll.setdefault(str(_collection_of(f["path"])), []).append(f)
        q_dir = out / f"queues{tag}"
        q_dir.mkdir(parents=True, exist_ok=True)
        cmds = []
        for coll, group in sorted(per_coll.items()):
            slug = Path(coll).name
            q = q_dir / f"queue_{slug}.json"
            q.write_text(json.dumps(
                {"why": f"{why}：来自 {coll}（一次采集一个包，身份才不会张冠李戴）",
                 "view": args.view, "n_frames": len(group),
                 "frames": [{**{k: f.get(k) for k in
                                 ("view", "path", "line_pixels", "pos",
                                  "heading", "exposure")},
                             "category": f.get("category", "")}
                            for f in group]},
                indent=1, ensure_ascii=False), encoding="utf-8")
            cmds.append((coll, slug, q))
        pkg_dir = out / f"packages{tag}"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        ps1 = out / f"build_packages{tag}.ps1"
        ps_lines = [f"# 逐采集打标注包（{why}；跨多次采集，必须分开打）",
                    "$ErrorActionPreference = 'Stop'",
                    "[Console]::OutputEncoding = [Text.Encoding]::UTF8"]
        for coll, slug, q in cmds:
            ps_lines.append(
                "& .venv" + chr(92) + "Scripts" + chr(92) + "python.exe "
                "scripts" + chr(92) + "m5_annotate_package.py "
                f"--review-queue '{q}' --collection '{coll}' "
                f"--per-view 80 "
                f"--out '{pkg_dir / ('pkg_' + slug)}'")
        ps1.write_text("\n".join(ps_lines) + "\n", encoding="utf-8")
        return cmds, ps1

    cmds, ps1 = _split_and_emit(starter, "", "起步包的一个子集")
    print(f"[review-pack] 起步包 {len(starter)} 帧 -> {out / 'review_pack.json'}")
    print(f"[review-pack]   按采集拆成 {len(cmds)} 个包（命令 -> {ps1}）")
    cmds_full, ps1_full = _split_and_emit(
        full_meta, "_full", "全量评价集的一个子集")
    print(f"[review-pack] 全量包 {len(full_meta)} 帧 -> "
          f"{out / 'review_pack_full.json'}")
    print(f"[review-pack]   按采集拆成 {len(cmds_full)} 个包"
          f"（命令 -> {ps1_full}）")
    for cat, label, _need in CATEGORIES:
        p = plan[cat]
        print(f"[review-pack]   {label}: 可用 {p['n_available']} / 已选 "
              f"{p['n_selected']}"
              + (f"（缺口 {p['gap']}）" if p["gap"] else "")
              + f" · 组数 {len(p['by_group'])}")
    print(f"[review-pack] 工作表 -> {out / 'review_worksheet.md'} / .json")
    print(f"[review-pack] 全量清单 -> {out / 'review_plan.json'}")
    print("[review-pack] 下一步（可直接粘贴，逐采集各打一个包）：")
    print(f"  # 全量 {len(full_meta)} 帧：")
    print(f"  pwsh -NoProfile -ExecutionPolicy Bypass -File {ps1_full}")
    print(f"  # 只要起步 {len(starter)} 帧：")
    print(f"  pwsh -NoProfile -ExecutionPolicy Bypass -File {ps1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
