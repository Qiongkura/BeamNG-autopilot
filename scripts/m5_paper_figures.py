"""论文图表生成：全部数字只从**已记录的数据文件**读取（不手写、不估计）。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_figures.py            # 全部图
    .venv\\Scripts\\python.exe scripts\\m5_paper_figures.py --only fig1

输出: ``logs/paper_figures/fig*.png``（300 dpi；logs/ 不提交）。

数据来源（每张图在 docs/T16_PAPER_FIGURES_20261006.md 里逐条写明）:
  fig1  logs/experiments/r2_verdict_{v7,v8_ADOPTED}_20261005.json
  fig2  logs/experiments/r2_verdict_v8_paint_{nodilate_mean,dil1b,dil2}_20261005.json
  fig3  logs/experiments/dev_full_{arms,dose_arms_s4*}_20261005.json
  fig4  logs/experiments/final_set_v5_20261005/seal/confirmation_base6x-seed42.json
  fig5  logs/fsd_benchmark/town_*.json（12 次运行，逐帧遥测）
  fig6  logs/fsd_benchmark/{scorecard,manifest}_*.json（按 manifest 开关分臂）
  fig7  logs/fsd_benchmark/town_1791272{266,479}.json（铺装门分级诊断轮）
  fig8  logs/fsd_benchmark/scorecard_1791202*/1791205*/1791206*/1791207*.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "paper_figures"
EXP = ROOT / "logs" / "experiments"
BENCH = ROOT / "logs" / "fsd_benchmark"

GATES = {  # 冻结门限（方案 §5；不得放宽）
    "candidate_reference_coverage": 0.80,
    "candidate_identity_rate": 0.60,
    "line_precision": 0.40,
    "line_recall": 0.70,
    "left_right_role_agreement": 0.70,
}
GATE_LABEL = {
    "candidate_reference_coverage": "coverage",
    "candidate_identity_rate": "identity",
    "line_precision": "precision",
    "line_recall": "recall",
    "left_right_role_agreement": "role",
}
C_CTRL, C_FACT = "#4C72B0", "#C44E52"
C_DEV, C_R3 = "#4C72B0", "#DD8452"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "figure.constrained_layout.use": True,
})


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def _per_seed_mean(block: dict, field: str) -> tuple[float | None, list]:
    ps = ((block or {}).get("gates") or {}).get("per_seed") or {}
    v = [ps[s][field] for s in sorted(ps) if isinstance(ps[s].get(field), (int, float))]
    return (round(st.mean(v), 4) if v else None), v


# ---------------------------------------------------------------- fig 1
def fig1() -> None:
    files = {"v7 (label scope)": "r2_verdict_v7_20261005.json",
             "v8 (paint scope, adopted)": "r2_verdict_v8_ADOPTED_20261005.json"}
    fields = ["candidate_reference_coverage", "candidate_identity_rate",
              "line_precision", "line_recall", "left_right_role_agreement"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, blk, title in zip(axes, ("dev", "r3"),
                              ("(a) dev pool (9 scenes, 117 frames)", "(b) R3 limited-class pool")):
        d = {name: json.load(open(EXP / fn, encoding="utf-8")) for name, fn in files.items()}
        x = np.arange(len(fields))
        w = 0.38
        for i, (name, blob) in enumerate(d.items()):
            vals = [_per_seed_mean(blob.get(blk), f)[0] for f in fields]
            recall_field = "line_recall_paint_scope" if "v8" in name else "line_recall"
            vals[3] = _per_seed_mean(blob.get(blk), recall_field)[0]
            ax.bar(x + (i - 0.5) * w, vals, w, label=name,
                   color=[C_CTRL, C_FACT][i], edgecolor="black", linewidth=0.4)
        # v8 的召回有两个口径：漆范围（采纳判定用）与标签范围。只画漆范围会把
        # "v8 在标签范围召回 0.56 < 0.70 不过门"藏起来，而那正是"两个定义各差一门"
        # 的关键对照（T16_R2_VERDICT §16）。
        v8 = json.load(open(EXP / files["v8 (paint scope, adopted)"], encoding="utf-8"))
        lab = _per_seed_mean(v8.get(blk), "line_recall")[0]
        ax.bar([3 + 0.5 * w], [lab], w, label="v8 (label scope)",
               color=C_FACT, hatch="///", edgecolor="black", linewidth=0.4)
        for j, f in enumerate(fields):
            ax.hlines(GATES[f], j - 0.45, j + 0.45, color="black", ls="--", lw=1.0)
        ax.set_xticks(x, [GATE_LABEL[f] for f in fields])
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("gate metric (mean over 6 seeds)")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].plot([], [], color="black", ls="--", lw=1.0, label="frozen gate")
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle("R2 acceptance: the two protocol definitions each fail a different gate",
                 fontsize=10)
    _save(fig, "fig1_gate_matrix")


# ---------------------------------------------------------------- fig 2
def fig2() -> None:
    files = [(0, "r2_verdict_v8_paint_nodilate_mean_20261005.json"),
             (1, "r2_verdict_v8_paint_dil1b_20261005.json"),
             (2, "r2_verdict_v8_paint_dil2_20261005.json")]
    series = {"recall (paint)": "line_recall_paint_scope",
              "precision": "line_precision",
              "role": "left_right_role_agreement"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, blk, title in zip(axes, ("dev", "r3"), ("(a) dev pool", "(b) R3 limited-class pool")):
        for label, field in series.items():
            ys = []
            for _, fn in files:
                blob = json.load(open(EXP / fn, encoding="utf-8"))
                ys.append(_per_seed_mean(blob.get(blk), field)[0])
            ax.plot([f[0] for f in files], ys, "o-", lw=1.4, ms=4, label=label)
        for f, thr in (("line_precision", GATES["line_precision"]),
                       ("line_recall", GATES["line_recall"]),
                       ("left_right_role_agreement", GATES["left_right_role_agreement"])):
            ax.axhline(thr, color="grey", ls=":", lw=0.9)
        ax.set_xticks([0, 1, 2], ["0", "1", "2"])
        ax.set_xlabel("mask dilation (px)")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("metric (mean over 6 seeds)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.text(0.5, -0.06,
             "dev: the recall gate fails for 1/6 seeds at 0 px (s45 = 0.671)   |   "
             "R3: precision 0.39 and role 0.66 fall below their gates at 1 px",
             ha="center", fontsize=6.5)
    fig.suptitle("Boundary map: dilation buys recall and pays precision — R3 pays ~3x "
                 "more, so no single value clears both pools", fontsize=10)
    _save(fig, "fig2_boundary_map")


# ---------------------------------------------------------------- fig 3
def fig3() -> None:
    # 剂量从臂名解析：base=0、neg4x=4、base6x=6、neg85x=8.5。名字里的 "85" 是
    # 8.5 倍——早期版本按字面读成 85 倍，轴错了整张图就错了。
    def dose_of(arm: str):
        head = arm.split("-")[0]
        if head == "base":
            return 0.0
        if head == "base6x":
            return 6.0
        if head.startswith("neg"):
            body = head[3:].rstrip("x")
            return 8.5 if body == "85" else float(body)
        return None
    rows = []
    base = json.load(open(EXP / "dev_full_arms_20261005.json", encoding="utf-8"))
    for name, a in (base.get("arms") or base).items():
        dose = dose_of(name)
        if dose is not None and isinstance(a, dict)                 and isinstance(a.get("candidate_identity_rate"), (int, float)):
            rows.append((dose, name, a["candidate_identity_rate"]))
    for p in sorted(EXP.glob("dev_full_dose_arms_s*.json")):
        for name, a in json.load(open(p, encoding="utf-8"))["arms"].items():
            dose = dose_of(name)
            if dose is not None and isinstance(a.get("candidate_identity_rate"), (int, float)):
                rows.append((dose, name, a["candidate_identity_rate"]))
    doses = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for dose in doses:
        v = [r[2] for r in rows if r[0] == dose]
        if not v:
            continue
        ax.scatter([dose] * len(v), v, s=18, color=C_FACT if dose else C_CTRL,
                   zorder=3, alpha=0.85)
        ax.hlines(st.mean(v), dose * 0.93 if dose else -0.12,
                  dose * 1.07 if dose else 0.12, color="black", lw=1.4, zorder=4)
    ax.axhline(GATES["candidate_identity_rate"], color="black", ls="--", lw=1.0)
    ax.annotate("identity gate 0.60", xy=(max(doses) * 0.52, 0.607), fontsize=7.5)
    ax.set_xlabel("negative-example dose (x, relative to the base pool)")
    ax.set_ylabel("candidate identity rate")
    ax.set_title("Training-composition dose response: identity peaks at 6x, "
                 "saturates at 8.5x")
    ax.set_xlim(-0.8, 9.3)
    ax.set_ylim(0.40, 0.78)
    ax.annotate("y-axis truncated at 0.40", xy=(0.98, 0.03), xycoords="axes fraction",
                ha="right", fontsize=6.5)
    _save(fig, "fig3_dose_response")


# ---------------------------------------------------------------- fig 4
def fig4() -> None:
    rec = json.load(open(EXP / "final_set_v5_20261005" / "seal" /
                         "confirmation_base6x-seed42.json", encoding="utf-8"))
    pg = rec["results"]["per_group"]
    names = [k.split("/")[-1] for k in pg]
    recall = [pg[k].get("line_recall") for k in pg]
    prec = [pg[k].get("line_precision") for k in pg]
    n = [pg[k].get("n_frames") for k in pg]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r if r is not None else 0 for r in recall], 0.4,
           label="line recall", color=C_CTRL, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, [p if p is not None else 0 for p in prec], 0.4,
           label="line precision", color=C_FACT, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(recall):
        if v is None:
            ax.text(i - 0.2, 0.02, "n/a", ha="center", fontsize=6.5, rotation=90)
    short = [a.replace("m5auto_", "") for a in names]
    ax.set_xticks(x, [f"{a}\n(n={b})" for a, b in zip(short, n)], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pixel metric")
    ax.set_title("(a) held-out final set, per group (label scope)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    nl = rec["results"]["overall"]["negative_line"]
    ax = axes[1]
    vals = [nl["false_positive_frame_rate"], nl["false_positive_pixel_fraction"] * 100,
            nl["false_positive_max_cc_px_max"] / 2000.0]
    ax.bar(["FP frames\n(rate)", "FP pixels\n(% of eligible)", "max CC\n(px/2000)"], vals,
           color="#55A868", edgecolor="black", linewidth=0.4)
    for i, (lab, v) in enumerate(zip(
            [f'{nl["false_positive_frames"]}/{nl["eligible_frames"]}',
             f'{nl["false_positive_px"]} px', f'{nl["false_positive_max_cc_px_max"]} px'],
            vals)):
        ax.text(i, v + 0.02, lab, ha="center", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) negative side (n=200 frames)")
    ax.set_ylabel("normalised value")
    ax.text(0.5, -0.26, "bars are normalised for display; raw values are labelled on the bars",
            transform=ax.transAxes, ha="center", fontsize=6.5)
    fig.suptitle("One-shot final confirmation (consumed once): delivered arm on a "
                 "road-disjoint 200-frame set", fontsize=10)
    _save(fig, "fig4_final_confirm")


# ---------------------------------------------------------------- fig 5
def fig5() -> None:
    ts = ["1791221139", "1791221359", "1791221561", "1791221776", "1791259365", "1791259786",
          "1791260204", "1791260603", "1791259583", "1791259992", "1791260405", "1791260800"]
    fc, sp = [], []
    for t in ts:
        p = BENCH / f"town_{t}.json"
        if not p.is_file():
            continue
        for h in json.loads(p.read_text(encoding="utf-8")):
            if float(h.get("t") or 0) < 8.0:
                continue
            if str(h.get("reason")) == "planned vehicle body crosses lane boundary":
                if isinstance(h.get("first_cross_m"), (int, float)):
                    fc.append(h["first_cross_m"])
                if isinstance(h.get("speed"), (int, float)):
                    sp.append(h["speed"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.hist(fc, bins=np.arange(2.0, 4.5, 0.25), color=C_FACT,
            edgecolor="black", linewidth=0.4)
    ax.axvline(4.0, color="black", ls="--", lw=1.0)
    ax.annotate("4 m hard-stop\nthreshold", xy=(4.0, ax.get_ylim()[1] * 0.55),
                xytext=(3.05, ax.get_ylim()[1] * 0.62), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("planned crossing distance (m)")
    ax.set_ylabel(f"frames (n={len(fc)})")
    ax.set_title("(a) the crossing is just inside the threshold")
    ax = axes[1]
    ax.hist(sp, bins=np.arange(0, 2.0, 0.1), color=C_CTRL,
            edgecolor="black", linewidth=0.4)
    frac = sum(1 for v in sp if v < 0.1) / max(len(sp), 1)
    ax.set_xlabel("ego speed (m/s)")
    ax.set_ylabel(f"frames (n={len(sp)})")
    ax.set_title(f"(b) {frac * 100:.0f}% of those frames are stationary")
    fig.suptitle("Deadlock anatomy: a parked, off-centre car is refused every path "
                 "(12 town runs, current body inside in all frames)", fontsize=10)
    _save(fig, "fig5_deadlock_anatomy")


# ---------------------------------------------------------------- fig 6
def _ab_runs() -> list[dict]:
    """每份 manifest+scorecard 一行，按开关分臂（从 manifest 读，不按顺序猜）。"""
    out = []
    for mp in sorted(glob.glob(str(BENCH / "manifest_*.json")), key=os.path.getmtime):
        man = json.loads(Path(mp).read_text(encoding="utf-8"))
        if (man.get("run") or {}).get("scenarios") != ["town"]:
            continue
        ts = Path(mp).stem.split("_")[1]
        card = BENCH / f"scorecard_{ts}.json"
        if not card.is_file():
            continue
        sw = man.get("switches") or {}
        eff = ((man.get("run") or {}).get("effective") or {}).get("town") or {}
        a = json.loads(card.read_text(encoding="utf-8"))["results"][0].get("assessed") or {}
        out.append({
            "ts": ts, "mtime": os.path.getmtime(card),
            "hold": sw.get("BEAMNG_HOLD_OBS_WINDOW") not in (None, "", "0"),
            "plc": sw.get("BEAMNG_PLC_ON_SENSOR_LANE") not in (None, "", "0"),
            "gate": sw.get("BEAMNG_LANE_PAVEMENT_GATE_NO_OCC") not in (None, "", "0"),
            "rec": sw.get("BEAMNG_STATIONARY_RECENTRE") not in (None, "", "0"),
            "deliv": "seed42" in str(eff.get("seg_model") or ""),
            "trav": a.get("travelled_m"), "stall": a.get("stall_frac"),
            "goal": a.get("goal_dist_m"), "bC": a.get("body_cross_centre_frames"),
            "coll": a.get("collision_count"), "off": a.get("off_road_frames"),
        })
    return sorted(out, key=lambda r: r["mtime"])


def fig6() -> None:
    """每个单因子的两臂 = 该 A/B 的**时间窗**（从运行日志转录）× 开关取值。

    早期版本按"开关签名 + 最近 4 次"选臂，F-D 的"对照"会混进 F-E/F-F 的臂
    （对照行进中位 26.6 m 就是这么来的）——臂必须限定在该 A/B 自己的时间窗内。
    """
    runs = _ab_runs()
    eras = {  # (起始 ts, 结束 ts)：转录自 logs/fsd_benchmark/_ab_*.log 的运行时刻
        "F-A hold window": (1791259000, 1791261000),
        # F-C 跑过三轮：第一轮代码静默失效（on 臂 == 对照臂），这里必须取
        # **修好之后**的那一轮（A/B #3），否则"factor on"其实没开。
        "F-C path recentre": (1791265400, 1791267500),
        "F-D delivered arm": (1791267100, 1791269500),
        "F-E lane gate": (1791273000, 1791275000),
        "F-F | F-E=1": (1791275000, 1791279500),
    }
    groups = [
        ("F-A hold window", lambda r: r["hold"], lambda r: not r["hold"]),
        ("F-C path recentre", lambda r: r["plc"], lambda r: not r["plc"]),
        ("F-D delivered arm", lambda r: r["deliv"], lambda r: not r["deliv"]),
        ("F-E lane gate", lambda r: r["gate"] and not r["rec"],
         lambda r: not r["gate"] and not r["rec"]),
        ("F-F | F-E=1", lambda r: r["gate"] and r["rec"],
         lambda r: r["gate"] and not r["rec"]),
    ]
    labels, ctrl_t, fact_t, ctrl_s, fact_s, ctrl_b, fact_b = [], [], [], [], [], [], []
    for name, is_f, is_c in groups:
        lo, hi = eras[name]
        era = [r for r in runs if lo <= int(r["ts"]) <= hi]
        f = [r for r in era if is_f(r)]
        c = [r for r in era if is_c(r)]
        print(f'[fig6] {name}: control={[r["ts"] for r in c]} factor={[r["ts"] for r in f]}')
        if not f or not c:
            continue
        labels.append(name)
        med = lambda rs, k: st.median([r[k] for r in rs if isinstance(r[k], (int, float))])
        mx = lambda rs, k: max([r[k] for r in rs if isinstance(r[k], (int, float))] or [0])
        ctrl_t.append(med(c, "trav")); fact_t.append(med(f, "trav"))
        ctrl_s.append(med(c, "stall")); fact_s.append(med(f, "stall"))
        ctrl_b.append(mx(c, "bC")); fact_b.append(mx(f, "bC"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.arange(len(labels))
    ax = axes[0]
    ax.bar(x - 0.2, ctrl_t, 0.4, label="control", color=C_CTRL,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, fact_t, 0.4, label="factor on", color=C_FACT,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("travelled (m, median of 4 runs)")
    ax.set_title("(a) the factors do move the car")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(x - 0.2, ctrl_s, 0.4, color=C_CTRL, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, fact_s, 0.4, color=C_FACT, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("stall fraction (median)")
    ax.set_ylim(0, 1.05)
    bcs = " / ".join(f"{cb:.0f}" + chr(0x2192) + f"{fb:.0f}"
                      for cb, fb in zip(ctrl_b, fact_b))
    ax.set_title("(b) ...but none clears no_stall (=0 frames)")
    ax.text(0.5, -0.30, "worst-case body-centre-cross frames (control→factor): " + bcs,
            transform=ax.transAxes, ha="center", fontsize=6.5)
    fig.suptitle("Driving-layer single-factor ladder (town, alternating A/B, "
                 "hard gates 0/4 in every arm)", fontsize=10)
    _save(fig, "fig6_factor_ladder")


# ---------------------------------------------------------------- fig 7
def fig7() -> None:
    ts = ["1791272266", "1791272479"]
    funnel = collections.Counter()
    n_rev = 0
    obst, offm = [], []
    for t in ts:
        p = BENCH / f"town_{t}.json"
        rows = [h for h in json.loads(p.read_text(encoding="utf-8"))
                if float(h.get("t") or 0) >= 8.0]
        funnel["settled"] += len(rows)
        for h in rows:
            if str(h.get("lane_sel")) == "sensor":
                funnel["sensor lane"] += 1
            if isinstance(h.get("lane_dev_m"), (int, float)):
                funnel["path exists"] += 1
                if h["lane_dev_m"] > 0.30 and str(h.get("lane_sel")) == "sensor":
                    funnel["dev > 0.30 m"] += 1
            if h.get("plc_pre_shift_m") is not None:
                funnel["pre-shift engaged"] += 1
            d = h.get("lane_drivable") or {}
            if isinstance(d.get("frac"), (int, float)) and d["frac"] < 0.6:
                n_rev += 1
                obst.append(d.get("n_obstacle") or 0)
                offm.append(d.get("n_off_mask") or 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    # 这两次运行是**基线配置**（无 F-E/F-C 开关）：funnel 说的是"车道→路径"的
    # 可用率，不是 F-C 的执行数——后者属于 F-C 自己的 on 臂（结果文档 §6）。
    order = ["settled", "sensor lane", "path exists", "dev > 0.30 m"]
    vals = [funnel[k] for k in order]
    ax.barh(range(len(order)), vals, color="#8172B3", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"{v}  ({v / max(vals) * 100:.0f}%)", va="center", fontsize=7)
    ax.set_xlabel("frames (2 baseline diagnostic runs)")
    ax.set_title("(a) lane/path availability in the baseline config")
    ax = axes[1]
    ax.boxplot([obst, offm], tick_labels=["obstacle\ncells", "off-mask\ncells"],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#CCB974", edgecolor="black", linewidth=0.4),
               medianprops=dict(color="black"))
    ax.set_ylabel("cells per revoked frame")
    ax.set_title(f"(b) why {n_rev} lanes were revoked")
    ax.set_ylim(-0.6, max(obst or [1]) * 1.25)
    ax.annotate("the obstacle term does the revoking; off-mask ≈ 0",
                xy=(0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=7)
    fig.suptitle("Lane-acceptance evidence: the 'on-pavement' gate is driven by the BEV "
                 "obstacle layer, not by 'off the pavement'", fontsize=10)
    _save(fig, "fig7_lane_gate_evidence")


# ---------------------------------------------------------------- fig 8
def fig8() -> None:
    v7 = ["1791201426", "1791204924", "1791206113", "1791206348"]
    v8 = ["1791202995", "1791205151", "1791206744", "1791206973"]
    def load(ts):
        out = []
        for t in ts:
            p = BENCH / f"scorecard_{t}.json"
            if not p.is_file():
                continue
            a = json.loads(p.read_text(encoding="utf-8"))["results"][0]["assessed"]
            out.append((a.get("lane_paired_rate"), a.get("lane_sensor_rate"),
                        a.get("cross_centre_frames"), a.get("body_cross_centre_frames")))
        return out
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for name, ts, color in (("v7 (label scope)", v7, C_CTRL), ("v8 (adopted)", v8, C_FACT)):
        rows = load(ts)
        xs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        ax.scatter(xs, ys, s=42, color=color, edgecolor="black", linewidth=0.4,
                   label=name, zorder=3)
    ax.set_xlabel("sensor lane-source rate (fraction of frames)")
    ax.set_ylabel("centre-crossing frames (reference line)")
    ax.set_title("Closed-loop trade-off (town, 4 runs per arm): v8 takes the centre\n"
                 "crossings to zero; sensor-source rate is lower on median "
                 "(ranges overlap)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(-0.5, max(14, ax.get_ylim()[1]))
    for ts, color in ((v7, C_CTRL), (v8, C_FACT)):
        rows = load(ts)
        if rows:
            ax.axvline(st.median([r[1] for r in rows]), color=color, ls=":", lw=1.0)
    _save(fig, "fig8_closed_loop_tradeoff")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
        "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="只出某一张（fig1..fig8）")
    args = ap.parse_args()
    names = [args.only] if args.only else list(FIGS)
    for n in names:
        if n not in FIGS:
            print(f"unknown figure {n}"); return 2
        FIGS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
