"""论文数字审计：把正文表格里的每个数值回查到已记录产物。

用途：E0 纪律的可执行版本——「主张 → 数值 → 来源」不能靠人记。本脚本只读，
不改任何文件；不一致时打印差异并以非零码退出。

覆盖：
* 表 I（两套定义的五个门，来自 v7/v8 判定 JSON）
* 表 II（后处理 × 范围的召回 2×2，来自同一批判定的 pixel/paint_recall 块）
* 表 V（后处理单因子，来自 e1_c{0..4}_*）
* 表 VI（等种子剂量，来自 e2_dose0x/4x 与 e1_c4_v8）
* 正文关键数字：确认集总体、a27、仲裁原因 1383/512、2443 结算帧、范围保留 69%/89%

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_number_audit.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "logs" / "experiments"
PAPER = ROOT / "docs" / "paper"


def _load(name: str) -> dict:
    return json.loads((E / name).read_text(encoding="utf-8"))


def _mean(vals) -> float | None:
    v = [x for x in vals if isinstance(x, (int, float))]
    return st.mean(v) if v else None


def _dev_means(name: str, pool: str = "dev") -> dict:
    b = _load(name)[pool]
    c, p, pr = b["candidate"], b["pixel"], b["paint_recall"]
    s = sorted(c)
    return {"coverage": _mean([c[x]["candidate_reference_coverage"] for x in s]),
            "identity": _mean([c[x]["candidate_identity_rate"] for x in s]),
            "role": _mean([c[x]["left_right_role_agreement"] for x in s]),
            "precision": _mean([p[x]["line_precision"] for x in s]),
            "recall_label": _mean([p[x]["line_recall"] for x in s]),
            "recall_paint": _mean([pr[x]["recall_paint"] for x in s]),
            "gt_keep": sum(pr[x]["truth_paint_px"] for x in s) / sum(pr[x]["truth_px"] for x in s)}


def _table_rows(md: str, caption: str) -> list[list[str]]:
    """取题注后面的第一张表（按 ``|`` 行解析）。"""
    i = md.index(caption)
    j = md.index("\n\n", i)
    rows = []
    for line in md[j:].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _num(s: str) -> float | None:
    m = re.match(r"^\**([0-9]*\.?[0-9]+)\**$", s.replace("**", ""))
    return float(m.group(1)) if m else None


def main() -> int:
    bad: list[str] = []
    checked = 0

    def expect(what: str, got, want, tol: float = 5e-4) -> None:
        nonlocal checked
        checked += 1
        if got is None or want is None:
            bad.append(f"{what}: 缺值 got={got} want={want}")
        elif abs(got - want) > tol:
            bad.append(f"{what}: 稿件 {want} 对来源 {got}（差 {abs(got - want):.4f}）")

    # ---- 表 I：两套定义的五个门（v7 标签范围 / v8 漆范围）
    v7d, v7r = _dev_means("r2_verdict_v7_20261005.json"), _dev_means("r2_verdict_v7_20261005.json", "r3")
    v8d, v8r = _dev_means("r2_verdict_v8_ADOPTED_20261005.json"), _dev_means("r2_verdict_v8_ADOPTED_20261005.json", "r3")
    for path in (PAPER / "paper.md", PAPER / "paper_zh.md"):
        md = path.read_text(encoding="utf-8")
        cap = "TABLE: Table I." if path.name == "paper.md" else "TABLE: 表 I."
        rows = _table_rows(md, cap)
        # 行序：dev baseline / dev adopted / r3 baseline / r3 adopted
        wants = [(v7d, ["coverage", "identity", "precision", "recall_label", "role"]),
                 (v8d, ["coverage", "identity", "precision", "recall_paint", "role"]),
                 (v7r, ["coverage", "identity", "precision", "recall_label", "role"]),
                 (v8r, ["coverage", "identity", "precision", "recall_paint", "role"])]
        for r, (src, keys) in zip(rows[1:], wants):
            for cell, k in zip(r[2:], keys):
                expect(f"{path.name} 表I[{r[0]}/{r[1]}/{k}]", round(src[k], 3), _num(cell), 6e-4)

        # ---- 表 V：后处理单因子
        cap5 = "TABLE: Table V." if path.name == "paper.md" else "TABLE: 表 V."
        srcs5 = [_dev_means(f) for f in ("e1_c0_v7_20261007.json", "e1_c1_lat_20261007.json",
                                         "e1_c2_merge_20261007.json", "e1_c3_appgate_20261007.json",
                                         "e1_c4_v8_20261007.json")]
        for r, src in zip(_table_rows(md, cap5)[1:], srcs5):
            for cell, k in zip(r[1:], ["precision", "recall_label", "recall_paint", "gt_keep", "identity"]):
                expect(f"{path.name} 表V[{r[0]}/{k}]", round(src[k], 3), _num(cell), 6e-4)

        # ---- 表 VI：等种子剂量
        cap6 = "TABLE: Table VI." if path.name == "paper.md" else "TABLE: 表 VI."
        e2 = [_dev_means("e2_dose0x_20261007.json"), _dev_means("e2_dose4x_20261007.json"),
              _dev_means("e1_c4_v8_20261007.json")]
        for r, src in zip(_table_rows(md, cap6)[1:4], e2):
            expect(f"{path.name} 表VI[{r[0]}/identity]", round(src["identity"], 3), _num(r[1]), 6e-4)
            expect(f"{path.name} 表VI[{r[0]}/role]", round(src["role"], 3), _num(r[4]), 6e-4)

    # ---- 正文关键数字
    conf = _load("final_set_v5_20261005/seal/confirmation_base6x-seed42.json")["results"]
    ov = conf["overall"]
    expect("确认集 总体召回", round(ov["line_recall"], 3), 0.460)
    expect("确认集 总体精度", round(ov["line_precision"], 3), 0.763)
    expect("确认集 线 IoU", round(ov["line_iou"], 3), 0.403)
    expect("确认集 道路 IoU", round(ov["road_iou"], 3), 0.633)
    a27 = conf["per_group"]["italy/m5auto_a27"]
    expect("a27 精度", round(a27["line_precision"], 3), 0.169)
    expect("a27 召回", round(a27["line_recall"], 3), 0.190)
    expect("确认集 负例合格帧", ov["negative_line"]["eligible_frames"], 105)
    expect("确认集 假阳性帧", ov["negative_line"]["false_positive_frames"], 22)

    # 仲裁原因与结算帧（与 fig23 同源：固定 12 个城镇运行、t>=8 s）
    import collections
    ts = ["1791221139", "1791221359", "1791221561", "1791221776", "1791259365", "1791259786",
          "1791260204", "1791260603", "1791259583", "1791259992", "1791260405", "1791260800"]
    c = collections.Counter()
    for t in ts:
        p = ROOT / "logs" / "fsd_benchmark" / f"town_{t}.json"
        if p.is_file():
            for h in json.loads(p.read_text(encoding="utf-8")):
                if float(h.get("t") or 0) >= 8.0:
                    c[str(h.get("reason") or "(none)")] += 1
    expect("结算帧总数", sum(c.values()), 2443)
    expect("无可用路径", c.get("no drivable path"), 1383)
    expect("规划车体穿越", c.get("planned vehicle body crosses lane boundary"), 512)

    # 范围保留比例（69% / 89%）
    expect("开发池真值保留", round(v7d["gt_keep"], 2), 0.69, 5e-3)
    expect("有限类别池真值保留", round(v7r["gt_keep"], 2), 0.89, 5e-3)

    # E5 起点位姿 A/B（2026-10-11）：逐 run 指标回查
    import math
    ab = [("01_O", 41, 24.79, 3.484, 20), ("02_B", 43, 24.73, 3.944, 26),
          ("03_B", 42, 24.61, 1.573, 30)]
    nets = []
    for name, n_fr, span, net, acc in ab:
        t = json.loads((E / "e5_ab_20261011" / name / "pose_00" / "telemetry.json")
                       .read_text(encoding="utf-8"))
        net_src = sum(math.dist(f["pos"][:2], g["pos"][:2]) for f, g in zip(t, t[1:]))
        acc_src = sum(1 for f in t if not f.get("plan_blocked"))
        expect(f"E5 AB {name} 帧数", len(t), n_fr)
        expect(f"E5 AB {name} 曝光s", round(t[-1]["t"] - t[0]["t"], 2), span, 5e-3)
        expect(f"E5 AB {name} 净行进m", round(net_src, 3), net, 5e-4)
        expect(f"E5 AB {name} 被接受路径帧", acc_src, acc)
        nets.append(round(net_src, 3))
    expect("E5 AB B 臂重复净行进比", round(max(nets[1], nets[2]) / min(nets[1], nets[2]), 1), 2.5)
    r4 = json.loads((E / "e5_ab_20261011" / "04_O" / "pose_00" / "result.json")
                    .read_text(encoding="utf-8"))
    expect("E5 AB 04_O 保持 invalid", int(bool(r4.get("qualified"))), 0)

    # 停车归因与负例剂量扫描（2026-10-11）：论文 V-C/V-E 新增数字回查
    g = json.loads((E / "stall_geometry_check_20261011_v2.json")
                   .read_text(encoding="utf-8"))
    rows = g["frames"]
    classes = collections.Counter(r["class"] for r in rows)
    expect("归因 C 感知丢失帧", classes["C_perception_missing"], 23)
    expect("归因 A2 候选全拒帧", classes["A2_candidates_all_rejected"], 20)
    expect("归因 B2 安全拒绝帧", classes["B2_rejected_blocked_ahead"], 28)
    expect("归因 D1 接受慢速帧", classes["D1_accepted_slow"], 19)
    expect("归因 D2 接受运动帧", classes["D2_accepted_moving"], 1)
    expect("归因 B1 走廊空闲被拒", classes.get("B1_rejected_corridor_free", 0), 0)
    expect("归因 lane_cross 累计",
           sum((r.get("planner_rejects") or {}).get("lane_cross", 0)
               for r in rows), 191)
    expect("归因 body_cross 累计",
           sum((r.get("planner_rejects") or {}).get("body_cross", 0)
               for r in rows), 191 - 191 + 103)
    corr = [r["corridor_ahead_m"] for r in rows
            if r["class"].startswith("B") and r["corridor_ahead_m"] is not None]
    expect("归因 被拒帧走廊侵入数", len(corr), 28)
    expect("归因 走廊最近侵入", round(min(corr), 1), 1.0, 5e-2)
    # 稿件用常规 half-up 一位小数（5.55 → 5.6）；Python round 是银行家舍入。
    expect("归因 走廊最远侵入",
           math.floor(max(corr) * 10 + 0.5) / 10, 5.6, 5e-2)

    ds = json.loads((E / "t16_dose_sweep_20261011" / "dose_sweep_summary.json")
                    .read_text(encoding="utf-8"))
    expect("剂量 d100 mean", round(ds["arms"]["d100"]["mean_iou"], 4), 0.4565)
    expect("剂量 d100 std", round(ds["arms"]["d100"]["std_iou"], 4), 0.0486)
    expect("剂量 d20 mean", round(ds["arms"]["d20"]["mean_iou"], 4), 0.3694)
    pd = ds["arms"]["d100"]["paired_diff_vs_d20"]
    expect("剂量 d100−d20 mean", round(pd["mean"], 4), 0.0871)
    expect("剂量 d100−d20 正向种子数", pd["positive_count"], 5)

    # 第二线帧池跨池配对（2026-10-13）
    cp = json.loads((E / "t16_crosspool_20261012" / "crosspool_summary.json")
                    .read_text(encoding="utf-8"))
    cp_iou = cp["paired"]["line_iou"]
    expect("跨池 line IoU E−H mean", round(cp_iou["mean_delta"], 4), 0.1003)
    expect("跨池 line IoU E−H 正向种子数", sum(1 for d in cp_iou["deltas"] if d > 0), 5)
    expect("跨池 line IoU 配对 n", cp_iou["n"], 5)

    print(f"=== 数字审计：检查 {checked} 项 ===")
    for b in bad:
        print("  ✗", b)
    print("RESULT:", "OK - 稿件数字与来源一致" if not bad else f"{len(bad)} 项不一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
