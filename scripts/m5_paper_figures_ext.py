"""论文图表集（扩展）：fig9–fig34。入口与样式复用 scripts/m5_paper_figures.py。

用法::

    .venv\\Scripts\\python.exe scripts\\m5_paper_figures_ext.py            # 全部扩展图
    .venv\\Scripts\\python.exe scripts\\m5_paper_figures_ext.py --only fig21

硬约束（与主脚本一致）：**所有数字只从已记录的数据文件读取**，不手写、不估计；
示意图（fig30+）在标题里明确标注为 schematic，并按代码/文档里的既有定义绘制。

数据来源逐条见 docs/T16_PAPER_FIGURES_20261006.md。
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

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # scripts/ 内的兄弟模块
sys.path.insert(0, str(_HERE.parents[1]))      # 仓库根（导入 beamng_autopilot）
from m5_paper_figures import (  # noqa: E402
    BENCH, C_CTRL, C_FACT, EXP, GATES, GATE_LABEL, OUT, _save,
)

ARROW = chr(0x2192)
NL = chr(10)


def _load(name: str) -> dict:
    return json.load(open(EXP / name, encoding="utf-8"))


def _dose_of(arm: str):
    head = arm.split("-")[0]
    if head == "base":
        return 0.0
    if head == "base6x":
        return 6.0
    if head.startswith("neg"):
        body = head[3:].rstrip("x")
        return 8.5 if body == "85" else float(body)
    return None


def _dose_rows(field: str):
    rows = []
    for name, a in (_load("dev_full_arms_20261005.json").get("arms") or {}).items():
        d = _dose_of(name)
        if d is not None and isinstance(a.get(field), (int, float)):
            rows.append((d, a[field]))
    for fn in sorted(EXP.glob("dev_full_dose_arms_s*.json")):
        for name, a in _load(fn.name)["arms"].items():
            d = _dose_of(name)
            if d is not None and isinstance(a.get(field), (int, float)):
                rows.append((d, a[field]))
    return rows



def _pixel_block(name: str, prefer: str = "base6x-s42") -> dict:
    """像素指标块：不同批次的键不同（dev/s42 或 frozen/base6x-s42）。"""
    d = _load(name)
    for blk, key in ((d.get("dev") or {}, "s42"),
                     (d.get("frozen") or {}, prefer),
                     (d.get("frozen") or {}, "s42"),
                     (d.get("dev") or {}, prefer)):
        if isinstance(blk.get(key), dict):
            return blk[key]
    raise KeyError(f"{name}: no pixel block (keys={list(d)})")


# ---------------------------------------------------------------- fig 9
def fig9() -> None:
    metrics = [("candidate_identity_rate", "identity"),
               ("left_right_role_agreement", "role"),
               ("candidate_reference_coverage", "coverage"),
               ("off_road_frac", "off-road candidate fraction"),
               ("merge_groups", "merge groups")]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.5), sharex=True)
    for ax, (field, label) in zip(axes, metrics):
        rows = _dose_rows(field)
        if not rows:
            ax.set_axis_off()
            continue
        for d in sorted({x for x, _ in rows}):
            v = [y for x, y in rows if x == d]
            ax.scatter([d] * len(v), v, s=14, color=C_FACT if d else C_CTRL, zorder=3)
            ax.hlines(st.mean(v), d - 0.25, d + 0.25, color="black", lw=1.2, zorder=4)
        if field in GATES:
            ax.axhline(GATES[field], color="black", ls="--", lw=0.9)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("dose (x)")
        ax.set_xlim(-0.8, 9.3)
    axes[0].set_ylabel("metric (per seed)")
    fig.suptitle("Dose response across five metrics (dev pool, per seed + mean)", fontsize=10)
    _save(fig, "fig9_dose_metrics")


# ---------------------------------------------------------------- fig 10
def fig10() -> None:
    arms = _load("t16_dual_scope_10seed.json")["arms"]
    pairs = {}
    for name, a in arms.items():
        fam, _, seed = name.partition("-")
        sfc = a.get("surface_scope") or {}
        if isinstance(a.get("candidate_identity_rate"), (int, float)) \
                and isinstance(sfc.get("candidate_identity_rate"), (int, float)):
            pairs.setdefault(fam, []).append(
                (seed, a["candidate_identity_rate"], sfc["candidate_identity_rate"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, fam in enumerate(sorted(pairs)):
        rows = pairs[fam]
        color = [C_CTRL, C_FACT][i % 2]
        for _seed, lab, surf in rows:
            ax.plot([0, 1], [lab, surf], "o-", color=color, lw=1.2, ms=4, alpha=0.9)
        ax.annotate(fam, xy=(0.5, st.mean([r[2] for r in rows]) + 0.012), ha="center",
                    fontsize=7.5, color=color)
    ax.axhline(GATES["candidate_identity_rate"], color="black", ls="--", lw=0.9)
    ax.set_xticks([0, 1], ["label scope", "surface scope"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("candidate identity rate")
    means = {fam: (st.mean([r[1] for r in rows]), st.mean([r[2] for r in rows]))
             for fam, rows in pairs.items()}
    base = means.get("base")
    six = means.get("base6x")
    if base and six:
        ax.text(0.02, 0.06, "mean shift: base %.3f" % base[0] + " -> %.3f" % base[1]
                + " | 6x %.3f" % six[0] + " -> %.3f" % six[1]
                + NL + "the two arms swap order between scopes",
                transform=ax.transAxes, fontsize=6.5, va="bottom")
    ax.set_title("Identity scope changes the reading, and can reverse the ranking"
                 + NL + "(same checkpoints, 10 seeds per arm; dashed = 0.60 gate)")
    _save(fig, "fig10_identity_scopes")


# ---------------------------------------------------------------- fig 11
def fig11() -> None:
    ps = _load("r2_verdict_v8_ADOPTED_20261005.json")["dev"]["gates"]["per_seed"]
    seeds = sorted(ps)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, [ps[s]["line_recall"] for s in seeds], 0.4, label="label scope",
           color=C_CTRL, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, [ps[s]["line_recall_paint_scope"] for s in seeds], 0.4,
           label="paint scope", color=C_FACT, edgecolor="black", linewidth=0.4)
    ax.axhline(GATES["line_recall"], color="black", ls="--", lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(x, [ps[s].get("label_nonpaint_frac") for s in seeds], "k^--", ms=4,
             lw=1.0, label="label non-paint frac")
    ax2.set_ylabel("label line pixels that are NOT paint")
    ax2.grid(False)
    ax.set_xticks(x, seeds)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("line recall")
    ax.set_title("Where the two recall scopes disagree (dev pool, 6 seeds)")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax2.legend(frameon=False, loc="upper right", fontsize=7.5)
    _save(fig, "fig11_recall_scopes")


# ---------------------------------------------------------------- fig 12
def fig12() -> None:
    scans = [("dev", "dev_scan_lateral_20261004.json"),
             ("R3", "r3b_scan_lateral_20261004.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, (pool, fn) in zip(axes, scans):
        rs = [r for r in _load(fn)["results"] if r.get("mode") != "baseline"]
        xs = [r["param"] for r in rs]
        ax.plot(xs, [r["candidate_identity_rate"] for r in rs], "o-", lw=1.3, ms=4,
                label="identity")
        ax.plot(xs, [r["left_right_role_agreement"] for r in rs], "s-", lw=1.3, ms=4,
                label="role")
        ax.axvline(5.5, color="grey", ls=":", lw=1.0)
        ax.annotate("adopted 5.5", xy=(5.5, 0.25), rotation=90, fontsize=6.5, ha="right")
        ax.set_xlabel("lateral scope threshold (m)")
        ax.set_title(f"({pool}) candidate lateral scope")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("metric")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Lateral-scope scan: identity/role vs threshold", fontsize=10)
    _save(fig, "fig12_lateral_scan")


# ---------------------------------------------------------------- fig 13
def fig13() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, (pool, fn) in zip(axes, [("dev", "dev_scan_parallel_20261004.json"),
                                     ("R3", "r3b_scan_parallel_20261004.json")]):
        rs = [r for r in _load(fn)["results"] if r.get("mode") != "baseline"]
        xs = [r["param"] for r in rs]
        ax.plot(xs, [r["merged_away"] for r in rs], "o-", lw=1.3, ms=4,
                color="#8172B3", label="merged away")
        ax.plot(xs, [r["kept_candidates"] for r in rs], "s-", lw=1.3, ms=4,
                color="#55A868", label="kept")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["candidate_identity_rate"] for r in rs], "k^--", ms=4, lw=1.0,
                 label="identity")
        ax2.grid(False)
        ax2.set_ylim(0, 1.0)
        ax.set_xlabel("parallel-merge radius (m)")
        ax.set_title(f"({pool}) parallel merge")
    axes[0].set_ylabel("candidates")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Parallel-merge scan: what merging removes, and what it costs", fontsize=10)
    _save(fig, "fig13_parallel_scan")


# ---------------------------------------------------------------- fig 14
def fig14() -> None:
    pairs = [("dev", "dev_pixel_base_20261004.json", "dev_pixel_appgate_20261004.json"),
             ("R3", "r3b_pixel_lat3_20261004.json", "r3b_pixel_appgate_20261004.json")]
    fields = [("line_iou", "line IoU"), ("line_precision", "line precision"),
              ("line_recall", "line recall")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
    for ax, (field, label) in zip(axes, fields):
        for i, (pool, base_fn, gate_fn) in enumerate(pairs):
            b = _pixel_block(base_fn)
            g = _pixel_block(gate_fn)
            ax.bar(i - 0.2, b.get(field) or 0, 0.4, color=C_CTRL, edgecolor="black",
                   linewidth=0.4, label="base" if i == 0 else None)
            ax.bar(i + 0.2, g.get(field) or 0, 0.4, color=C_FACT, edgecolor="black",
                   linewidth=0.4, label="appearance gate" if i == 0 else None)
        ax.set_xticks(np.arange(len(pairs)), [p[0] for p in pairs])
        ax.set_title(label, fontsize=8.5)
        ax.set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Appearance gate ablation on the pixel metrics (seed 42)", fontsize=10)
    _save(fig, "fig14_appearance_gate")


# ---------------------------------------------------------------- fig 15
def fig15() -> None:
    specs = [("keep", [(0.6, "r3b_pixel_keep0.6_20261004.json"),
                       (0.7, "r3b_pixel_keep0.7_20261004.json"),
                       (0.8, "r3b_pixel_keep0.8_20261004.json")]),
             ("elong", [(0.4, "r3b_pixel_elong0.4_20261004.json"),
                        (0.6, "r3b_pixel_elong0.6_20261004.json"),
                        (0.8, "r3b_pixel_elong0.8_20261004.json")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, (label, items) in zip(axes, specs):
        xs, fp, iou = [], [], []
        for v, fn in items:
            try:
                blk = _pixel_block(fn)
            except Exception:
                continue
            xs.append(v)
            fp.append((blk.get("negative_line") or {}).get("false_positive_frame_rate"))
            iou.append(blk.get("line_iou"))
        if any(v is not None for v in fp):
            ax.plot(xs, fp, "o-", lw=1.3, ms=4, color=C_FACT, label="FP frame rate")
        else:
            ax.annotate("negative side not measurable here" + NL
                        + "(R3 pool: no eligible negative frames)", xy=(0.5, 0.06),
                        xycoords="axes fraction", ha="center", fontsize=6.5)
        ax.plot(xs, iou, "s-", lw=1.3, ms=4, color=C_CTRL, label="line IoU")
        ax.set_xlabel(f"{label} threshold")
        ax.set_title(f"({label}) threshold sensitivity")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("metric (R3 pool, seed 42)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Threshold sensitivity of the mask post-processing", fontsize=10)
    _save(fig, "fig15_threshold_scans")


# ---------------------------------------------------------------- fig 16
def fig16() -> None:
    base = _load("r2_verdict_v8_ADOPTED_20261005.json")["dev"]["gates"]["per_seed"]
    seeds = sorted(base)
    try:
        swa = _load("r2_verdict_swa_20261005.json")["dev"]["gates"]["per_seed"]
    except Exception:
        swa = {}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, [base[s]["line_recall_paint_scope"] for s in seeds], 0.4,
           label="adopted (no SWA)", color=C_CTRL, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, [(swa.get(s) or {}).get("line_recall_paint_scope") or 0 for s in seeds],
           0.4, label="SWA (last-5-epoch mean)", color=C_FACT, edgecolor="black",
           linewidth=0.4)
    ax.axhline(GATES["line_recall"], color="black", ls="--", lw=1.0)
    ax.set_xticks(x, seeds)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall (paint scope)")
    ax.set_title("SWA single-factor result: no seed is rescued" + NL
                 + "(rejected; dashed = 0.70 recall gate)")
    ax.legend(frameon=False, fontsize=7.5)
    _save(fig, "fig16_swa")


# ---------------------------------------------------------------- fig 17
def fig17() -> None:
    pairs = [("base (dilate 1)", "dev9_v8_dilate1_20261005.json"),
             ("beta=0.8 (Tversky)", "dev9_beta08_v8d1_20261005.json")]
    fields = [("false_positive_frame_rate", "FP frames (rate)"),
              ("false_positive_px", "FP pixels"),
              ("false_positive_max_cc_px_max", "max CC (px)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for ax, (field, label) in zip(axes, fields):
        vals = []
        for _name, fn in pairs:
            nl = (_load(fn)["frozen"]["s42"].get("negative_line") or {})
            vals.append(nl.get(field))
        ax.bar([p[0] for p in pairs], [v or 0 for v in vals], color=[C_CTRL, C_FACT],
               edgecolor="black", linewidth=0.4)
        ax.set_title(label, fontsize=8.5)
        ax.tick_params(axis="x", labelrotation=15, labelsize=7)
        for i, v in enumerate(vals):
            ax.text(i, (v or 0) * 1.02, str(v), ha="center", fontsize=7)
    fig.suptitle("Tversky beta=0.8 arm vs base: the negative side gets worse "
                 "(R3 pool, seed 42)", fontsize=10)
    _save(fig, "fig17_beta08")


# ---------------------------------------------------------------- fig 18
def fig18() -> None:
    """R3 有限类别池逐种子：两套定义下的三个候选级门。

    数据源是 10-05 的两份判定文件——与表 I 同一批运行。早先版本读的是
    10-04 的 `r3_acceptance_20261004.json`（另一批运行），图和表因此对不上
    （2026-10-07 审查指出）。
    """
    srcs = [("v7 (baseline, label scope)", "r2_verdict_v7_20261005.json", C_CTRL),
            ("v8 (adopted, paint scope)", "r2_verdict_v8_ADOPTED_20261005.json", C_FACT)]
    fields = [("candidate_reference_coverage", "coverage (gate 0.80)", 0.80),
              ("candidate_identity_rate", "identity (gate 0.60)", 0.60),
              ("left_right_role_agreement", "role (gate 0.70)", 0.70)]
    blocks = {f: (_load(f).get("r3") or {}).get("candidate") or {} for _n, f, _c in srcs}
    seeds = sorted(blocks[srcs[0][1]])
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, (field, title, gate) in zip(axes, fields):
        for i, (_name, f, color) in enumerate(srcs):
            blk = blocks[f]
            vals = [blk[s].get(field) if isinstance(blk[s].get(field), (int, float)) else np.nan
                    for s in seeds]
            xs = x + (i - 0.5) * 0.38
            ax.bar(xs, vals, 0.36, color=color, edgecolor="black", linewidth=0.4,
                   label=_name)
            for xi, v in zip(xs, vals):
                if isinstance(v, float) and v < gate:
                    ax.text(xi, v + 0.02, "x", ha="center", fontsize=6.5)
        ax.axhline(gate, color="black", ls="--", lw=0.8)
        ax.set_xticks(x, seeds, fontsize=6.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("metric (per seed)")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.14), ncol=1)   # 放到面板下方，避开刻度与柱
    fig.suptitle("R3 limited-class pool, per seed: candidate-level gates under both "
                 "definitions (x = below gate)", fontsize=9)
    _save(fig, "fig18_r3_per_seed")


# ---------------------------------------------------------------- fig 19
def fig19() -> None:
    d = _load("r3_instance_ann_cov.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = {"near_left": C_CTRL, "far_left": "#8172B3", "near_right": C_FACT,
              "straddled": "#55A868"}
    for key, v in sorted(d.items()):
        role = key.split("/")[-1]
        ax.scatter(v["lateral_m"], v["ann_cov_mean"], s=28,
                   color=colors.get(role, "grey"), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("lateral offset of the reference instance (m)")
    ax.set_ylabel("annotation coverage (mean)")
    ax.set_ylim(-0.05, 1.1)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=r)
               for r, c in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=7.5)
    ax.set_title("Instance-level reference coverage vs lateral offset "
                 "(R3 certified scenes)")
    _save(fig, "fig19_instance_coverage")


# ---------------------------------------------------------------- fig 20
def fig20() -> None:
    rows = []
    for p in sorted(EXP.glob("*_pixel_*.json")):
        try:
            blk = _pixel_block(p.name)
            nl = blk.get("negative_line") or {}
            if nl.get("false_positive_frame_rate") is None:
                continue
            rows.append((p.name, nl["false_positive_frame_rate"],
                         (nl.get("false_positive_pixel_fraction") or 0) * 100,
                         nl.get("false_positive_max_cc_px_max") or 0))
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for _n, rate, pxpct, cc in rows:
        ax.scatter(rate, pxpct, s=20 + cc / 40, color="#8172B3",
                   edgecolor="black", linewidth=0.3)
    # 臂名很长且有两个点几乎重合：点内只放序号，名单另列，避免标签叠在一起
    for i, (_n, rate, pxpct, _cc) in enumerate(rows, start=1):
        off = (5, 4) if i % 2 else (5, -10)      # 相邻点序号上下交错，避免"14"叠一起
        ax.annotate(str(i), xy=(rate, pxpct), fontsize=6.5, xytext=off,
                    textcoords="offset points")
    if rows:
        listing = chr(10).join(
            f"{i} " + n.replace("_20261004.json", "").replace("_20261005.json", "")
            for i, (n, *_rest) in enumerate(rows, start=1))
        ax.text(0.26, 0.62, listing, transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left")
    ax.set_xlabel("false-positive frame rate")
    ax.set_ylabel("false-positive pixels (% of eligible)")
    ax.set_title(f"Negative-side diagnostics across {len(rows)} recorded arms "
                 "(dev, seed 42; marker size = max CC px)")
    _save(fig, "fig20_negatives_family")


# ---------------------------------------------------------------- fig 21
def fig21() -> None:
    """训练曲线：按 family 聚合（321 个 run 逐条画会把图例压死）。

    每个 family 取该 family 内所有 run 的 val_line_iou / train_loss 曲线，
    画**中位曲线 + IQR 带**，图例只列 run 数最多的前 10 个 family（放轴外）。
    """
    import collections as _c
    hists = []
    for p in sorted(EXP.rglob("train_hist.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("epoch"), list):
            cols = [k for k, v in d.items() if isinstance(v, list)]
            rows = [{k: d[k][i] for k in cols} for i in range(len(d["epoch"]))]
        elif isinstance(d, list):
            rows = [r for r in d if isinstance(r, dict) and "epoch" in r]
        else:
            rows = []
        if rows:
            hists.append((p.relative_to(EXP).parts[0], rows))
    by = _c.defaultdict(list)
    for fam, rows in hists:
        by[fam].append(rows)
    top = [f for f, _ in sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    cmap = plt.get_cmap("tab10")
    for k, fam in enumerate(top):
        color = cmap(k % 10)
        for ax, key in ((axes[0], "val_line_iou"), (axes[1], "train_loss")):
            grid = np.arange(0, 121, 5)
            curves = []
            for rows in by[fam]:
                xs = [r["epoch"] for r in rows]
                ys = [r.get(key) for r in rows]
                ok = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float))]
                if len(ok) < 2:
                    continue
                curves.append(np.interp(grid, [x for x, _ in ok], [y for _, y in ok]))
            if not curves:
                continue
            M = np.vstack(curves)
            med = np.median(M, axis=0)
            ax.plot(grid, med, lw=1.4, color=color, label=f"{fam} (n={len(curves)})")
            ax.fill_between(grid, np.percentile(M, 25, axis=0),
                            np.percentile(M, 75, axis=0), color=color, alpha=0.15, lw=0)
    axes[0].set_ylabel("val line IoU (median, IQR band)")
    axes[1].set_ylabel("train loss (median, IQR band)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 120)
    axes[1].set_ylim(0, 2.0)
    # 诚实性：line 通道被屏蔽的 run，其 val_line_iou 恒为 0（构造使然，不是"学不会"）
    try:
        masked = {}
        for r in _load("training_history.json")["runs"]:
            if r.get("line_ignored"):
                masked[r["family"]] = masked.get(r["family"], 0) + 1
        shown = [f"{f} ({masked[f]})" for f in top if f in masked]
        if shown:
            head = ", ".join(shown[:3]) + (" ..." if len(shown) > 3 else "")
            axes[0].annotate("line-masked runs: IoU = 0 by construction" + NL + head,
                             xy=(0.02, 0.03), xycoords="axes fraction", fontsize=6.5)
    except Exception:
        pass
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Training curves by family ({len(hists)} recorded runs, "
                 f"{len(by)} families; top {len(top)} shown)", fontsize=10)
    _save(fig, "fig21_training_curves")


def _bench_runs() -> list:
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
        d = json.loads(card.read_text(encoding="utf-8"))["results"][0]
        out.append({"ts": ts, "mtime": os.path.getmtime(card), "sw": sw,
                    "checks": d.get("checks") or {}, "unknown": d.get("unknown") or [],
                    "assessed": d.get("assessed") or {}})
    return sorted(out, key=lambda r: r["mtime"])


# ---------------------------------------------------------------- fig 22
def fig22() -> None:
    runs = _bench_runs()[-32:]
    order = ["has_frames", "no_reversing", "no_centre_crossing", "no_edge_crossing",
             "no_stall", "on_road", "no_collision", "reached_goal"]
    M = np.zeros((len(order), len(runs)))
    for j, r in enumerate(runs):
        for i, k in enumerate(order):
            M[i, j] = 1.0 if r["checks"].get(k) else 0.0
    fig, ax = plt.subplots(figsize=(12.0, 3.4))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    short = {"has_frames": "frames", "no_reversing": "no reversing",
             "no_centre_crossing": "no centre X", "no_edge_crossing": "no edge X",
             "no_stall": "no stall", "on_road": "on road",
             "no_collision": "no collision", "reached_goal": "goal reached"}
    ax.set_yticks(range(len(order)), [short[k] for k in order], fontsize=8)
    fig.subplots_adjust(left=0.13)
    ax.set_xticks(range(len(runs)), [r["ts"][-5:] for r in runs], rotation=90,
                  fontsize=5)
    ax.set_title(f"Driving hard-gate checklist over the last {len(runs)} town runs "
                 "(green = pass, red = fail; UNKNOWN counts as fail)")
    _save(fig, "fig22_gate_heatmap")


# ---------------------------------------------------------------- fig 23
def fig23() -> None:
    ts = ["1791221139", "1791221359", "1791221561", "1791221776", "1791259365",
          "1791259786", "1791260204", "1791260603", "1791259583", "1791259992",
          "1791260405", "1791260800"]
    c = collections.Counter()
    for t in ts:
        p = BENCH / f"town_{t}.json"
        if not p.is_file():
            continue
        for h in json.loads(p.read_text(encoding="utf-8")):
            if float(h.get("t") or 0) >= 8.0:
                c[str(h.get("reason") or "(none)")] += 1
    top = c.most_common(10)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(range(len(top)), [v for _, v in top], color="#8172B3",
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top)), [k for k, _ in top], fontsize=7)
    ax.invert_yaxis()
    for i, (_k, v) in enumerate(top):
        ax.text(v + 2, i, str(v), va="center", fontsize=7)
    ax.set_xlabel(f"frames (12 town runs, {sum(c.values())} settled frames)")
    ax.set_title("Why the car does not move: safety-arbitration reasons")
    _save(fig, "fig23_reason_hist")


# ---------------------------------------------------------------- fig 24
def fig24() -> None:
    pairs = [("baseline acceptance run", "1791221139"),
             ("F-E lane-gate run (factor on)", "1791273507")]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.0), sharex="col")
    for col, (label, ts) in enumerate(pairs):
        rows = [h for h in json.loads((BENCH / f"town_{ts}.json").read_text(encoding="utf-8"))]
        t = [h.get("t") for h in rows]
        axes[0][col].plot(t, [h.get("speed") for h in rows], lw=0.9, color=C_CTRL)
        axes[0][col].axhline(0.5, color="black", ls="--", lw=0.8)
        axes[0][col].set_title(f"{label} (ts {ts})", fontsize=8.5)
        axes[0][col].set_ylabel("speed (m/s)")
        dev = [h.get("lane_dev_m") for h in rows]
        axes[1][col].plot(t, dev, lw=0.8, color=C_FACT, marker="o", ms=1.5)
        axes[1][col].annotate(f"n measured = {sum(1 for v in dev if isinstance(v, (int, float)))}",
                              xy=(0.02, 0.9), xycoords="axes fraction", fontsize=6.5)
        axes[1][col].set_ylabel("lane_dev (m)")
        axes[1][col].set_xlabel("t (s)")
    fig.suptitle("Per-run time series: speed (top) and path-to-lane deviation (bottom); "
                 "dashed = 0.5 m/s stall threshold", fontsize=9.5)
    _save(fig, "fig24_time_series")


# ---------------------------------------------------------------- fig 25
def fig25() -> None:
    runs = _bench_runs()
    arms = {"F-A": lambda r: "on" if r["sw"].get("BEAMNG_HOLD_OBS_WINDOW") not in
            (None, "", "0") else "off",
            "F-C": lambda r: "on" if r["sw"].get("BEAMNG_PLC_ON_SENSOR_LANE") not in
            (None, "", "0") else "off",
            "F-E": lambda r: "on" if r["sw"].get("BEAMNG_LANE_PAVEMENT_GATE_NO_OCC") not in
            (None, "", "0") else "off"}
    eras = {"F-A": (1791259000, 1791261000), "F-C": (1791265400, 1791267500),
            "F-E": (1791273000, 1791275000)}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    labels, data_t, data_s = [], [], []
    for name, arm_of in arms.items():
        lo, hi = eras[name]
        for a in ("off", "on"):
            vals = [r["assessed"].get("travelled_m") for r in runs
                    if lo <= int(r["ts"]) <= hi and arm_of(r) == a]
            labels.append(f"{name}" + NL + a)
            data_t.append([v for v in vals if isinstance(v, (int, float))])
            vals_s = [r["assessed"].get("stall_frac") for r in runs
                      if lo <= int(r["ts"]) <= hi and arm_of(r) == a]
            data_s.append([v for v in vals_s if isinstance(v, (int, float))])
    for ax, data, ylab, title in ((axes[0], data_t, "travelled (m)", "(a) travelled"),
                                  (axes[1], data_s, "stall fraction", "(b) stall fraction")):
        bp = ax.boxplot(data, labels=labels, widths=0.5, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(C_FACT if i % 2 else C_CTRL)
            box.set_edgecolor("black")
            box.set_linewidth(0.4)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("Per-arm distributions (town A/Bs, 4 runs per arm)", fontsize=10)
    _save(fig, "fig25_arm_boxplots")


# ---------------------------------------------------------------- fig 26
def fig26() -> None:
    pairs = [("v7 closed-loop", ["1791201426", "1791204924", "1791206113", "1791206348"]),
             ("v8 closed-loop", ["1791202995", "1791205151", "1791206744", "1791206973"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (label, ts) in enumerate(pairs):
        vals = []
        for t in ts:
            p = BENCH / f"town_{t}.json"
            if not p.is_file():
                continue
            vals += [h.get("lane_dev_m") for h in json.loads(p.read_text(encoding="utf-8"))
                     if float(h.get("t") or 0) >= 8.0
                     and isinstance(h.get("lane_dev_m"), (int, float))]
        if vals:
            ax.scatter([i] * len(vals), vals, s=6, alpha=0.5,
                       color=[C_CTRL, C_FACT][i])
            ax.hlines(st.median(vals), i - 0.25, i + 0.25, color="black", lw=1.6)
    ax.set_xticks([0, 1], [p[0] for p in pairs])
    ax.set_ylabel("path-to-lane deviation (m)")
    ax.set_title("Lane-placement deviation per frame (closed-loop round; "
                 "black bar = median)")
    _save(fig, "fig26_lane_dev")


# ---------------------------------------------------------------- fig 27
def fig27() -> None:
    ts = ["1791221139", "1791221359", "1791221561", "1791221776"]
    close, ttc, occ = [], [], []
    for t in ts:
        p = BENCH / f"town_{t}.json"
        if not p.is_file():
            continue
        for h in json.loads(p.read_text(encoding="utf-8")):
            if float(h.get("t") or 0) < 8.0:
                continue
            for src, dst in (("closest_obs_m", close), ("min_ttc", ttc),
                             ("path_occ_frac", occ)):
                v = h.get(src)
                if isinstance(v, (int, float)):
                    dst.append(v)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    for ax, (vals, label, cap) in zip(axes, [(close, "closest obstacle (m)", 60.0),
                                        (ttc, "min TTC (s)", 15.0),
                                        (occ, "path occupancy fraction", 1.0)]):
        v = [x for x in vals if 0 <= x <= cap]
        ax.hist(v, bins=30, color="#55A868", edgecolor="black", linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("frames")
    fig.suptitle("Safety-margin distributions over 4 acceptance runs (settled frames)",
                 fontsize=10)
    _save(fig, "fig27_safety_margins")


# ---------------------------------------------------------------- fig 28
def fig28() -> None:
    """帧计数 vs 按行进量归一：同一个安全量在两种读法下的排序。"""
    runs = _bench_runs()
    groups = [("F-A", (1791259000, 1791261000), "BEAMNG_HOLD_OBS_WINDOW"),
              ("F-C", (1791265400, 1791267500), "BEAMNG_PLC_ON_SENSOR_LANE"),
              ("F-E", (1791273000, 1791275000), "BEAMNG_LANE_PAVEMENT_GATE_NO_OCC")]
    rows = []
    for name, (lo, hi), sw in groups:
        for arm in ("off", "on"):
            sel = [r for r in runs if lo <= int(r["ts"]) <= hi
                   and (r["sw"].get(sw) not in (None, "", "0")) == (arm == "on")]
            trav = [r["assessed"].get("travelled_m") or 0 for r in sel]
            bc = [r["assessed"].get("body_cross_centre_frames") or 0 for r in sel]
            if not sel:
                continue
            rows.append((f"{name}-{arm}", max(bc), 100.0 * max(bc) / max(st.median(trav), 0.1)))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=C_FACT,
           edgecolor="black", linewidth=0.4, label="raw max frames")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [max(r[2], 0.5) for r in rows], 0.4, color=C_CTRL,
            edgecolor="black", linewidth=0.4, label="per 100 m travelled")
    ax2.grid(False)
    ax2.set_yscale("log")
    ax2.tick_params(labelbottom=False)      # twin 会重复画一套 x 刻度标签，关掉
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r[1] + 1, str(r[1]), ha="center", fontsize=6)
        ax2.text(i + 0.2, max(r[2], 0.5) * 1.15, f"{r[2]:.0f}", ha="center", fontsize=6)
    ax.set_ylim(0, max([r[1] for r in rows] + [1]) * 1.18)   # 顶部数值标签别顶到标题
    ax2.set_ylim(top=max([max(r[2], 0.5) for r in rows] + [1]) * 3.0)
    ax.set_xticks(x, [r[0] for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("body-centre-cross frames (max)")
    ax2.set_ylabel("same, per 100 m travelled")
    ax.set_title("Why the safety rule needed re-specifying: raw frame counts scale with"
                 + NL + "how far the car moved (factor arms move 2x further)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5)
    _save(fig, "fig28_metric_scaling")


# ---------------------------------------------------------------- fig 29
def fig29() -> None:
    rec = _load("final_set_v5_20261005/seal/final_set_seal.json")
    groups = collections.Counter()
    for path in rec["digests"]:
        parts = Path(path).parts
        scene = [p for p in parts if p.startswith("m5auto_")][0]
        groups[scene.split("_", 1)[1].split("_", 1)[0]] += 1
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    keys = sorted(groups)
    ax.bar(keys, [groups[k] for k in keys], color="#8172B3", edgecolor="black",
           linewidth=0.4)
    for i, k in enumerate(keys):
        ax.text(i, groups[k] + 1, str(groups[k]), ha="center", fontsize=7)
    ax.set_ylabel("frames")
    ax.set_title(f"Sealed final set composition (n={rec['n_frames']}, "
                 f"digest {rec['digest']}, protocol {rec['protocol_hash']})")
    _save(fig, "fig29_final_set_composition")


# ============================ schematics ==================================
def _box(ax, x, y, w, h, text, fc="#DCE3F0", fontsize=7.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                               linewidth=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def fig30() -> None:
    """计数契约 v5（按 AGENTS.md / candidate_metrics 的定义绘制；schematic）。"""
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_axis_off()
    _box(ax, 0.02, 0.62, 0.22, 0.24, "P_frames\nframes whose annotation\ncarries a paint line")
    _box(ax, 0.30, 0.62, 0.22, 0.24, "C\nperception candidates\ninside P (evaluated)")
    _box(ax, 0.58, 0.62, 0.22, 0.24, "C_outside_P\ncandidates outside P\n(reported, not scored)",
         fc="#F3D9D9")
    _box(ax, 0.30, 0.30, 0.22, 0.24, "R ⊆ C\ncandidates whose own side\nhas a reference")
    _box(ax, 0.58, 0.30, 0.22, 0.24, "M ⊆ R\nmatched (frozen\ncondition)")
    _box(ax, 0.84, 0.30, 0.14, 0.24, "L ⊆ M\nboth sides\nrole-decidable")
    _box(ax, 0.84, 0.02, 0.14, 0.22, "A ⊆ L\nrole\nagrees")
    ax.annotate("", xy=(0.30, 0.74), xytext=(0.24, 0.74),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.30, 0.42), xytext=(0.24, 0.74),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.58, 0.42), xytext=(0.52, 0.42),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.84, 0.42), xytext=(0.80, 0.42),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.91, 0.24), xytext=(0.91, 0.30),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.02, 0.30, "gates (mean over seeds):" + NL
            + "  coverage = R / C      identity = M / R" + NL
            + "  role     = A / L      precision/recall = pixels",
            fontsize=7.5, va="top")
    ax.text(0.02, 0.14, "coverage is candidate-reference availability, not detection recall;"
            + NL + "no instance-level recall is claimed (a missed marking on a frame with"
            + NL + "no candidate enters pixel recall only)",
            fontsize=6.5, va="top")
    ax.set_title("Counting contract v5 (schematic): every ratio has an explicit "
                 "numerator and denominator", fontsize=9.5)
    _save(fig, "fig30_counting_contract")


def fig31() -> None:
    """候选口径流水线（schematic；数字取自本会话实测）。"""
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.set_axis_off()
    steps = [("mask\ncandidates", "#DCE3F0"),
             ("lateral scope\n|lat| <= 5.5 m", "#DCE3F0"),
             ("appearance gate\npaint-like", "#DCE3F0"),
             ("merge (same-side,\nr = 1.0 m)", "#DCE3F0"),
             ("final candidate set\n(v7/v8)", "#E8F0DC")]
    x = 0.02
    for i, (txt, fc) in enumerate(steps):
        _box(ax, x, 0.45, 0.17, 0.28, txt, fc=fc)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.19, 0.59), xytext=(x + 0.17, 0.59),
                        arrowprops=dict(arrowstyle="->", lw=0.8))
        x += 0.19
    ax.text(0.02, 0.32, "measured (2026-10-06, town driving): 34% of paired frames "
            "revoked by the on-pavement gate" + NL
            + "the revoking term was the BEV obstacle layer, not 'off the pavement' "
            "(202/202 frames)", fontsize=7.5, va="top")
    ax.set_title("Candidate-scope pipeline (schematic) and the measured revocation",
                 fontsize=9.5)
    _save(fig, "fig31_pipeline")


def fig32() -> None:
    """安全仲裁阶梯（按 safety_monitor.ARBITRATION_RULES 的顺序绘制）。"""
    import beamng_autopilot.safety_monitor as sm
    rules = list(sm.ARBITRATION_RULES)
    worst = [sm.RULE_WORST_LEVEL.get(r, "?") for r in rules]
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_axis_off()
    y = 0.94
    for r, w in zip(rules, worst):
        fc = {"degraded": "#FDF3D0", "minimal_risk": "#F3D9D9"}.get(w, "#DCE3F0")
        _box(ax, 0.02, y - 0.05, 0.62, 0.048, r, fc=fc, fontsize=7.5)
        ax.text(0.66, y - 0.028, w, fontsize=7, va="center")
        y -= 0.0565
    ax.text(0.02, 0.02, "soft rules accumulate caps; a minimal_risk rule stops the tick "
            "(order is the evaluated order, not severity)", fontsize=7)
    ax.set_title("Safety arbitration ladder (schematic, from ARBITRATION_RULES)",
                 fontsize=9.5)
    _save(fig, "fig32_arbitration_ladder")


def fig33() -> None:
    """协议开关矩阵（按 protocol.py 的定义与实测哈希绘制）。"""
    from beamng_autopilot.experiments.protocol import (
        PROTOCOL_VERSION, PROTOCOL_VERSION_V8, active_protocol_version, protocol_hash,
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.set_axis_off()
    rows = [("v7 (frozen)", "lat = 0 (off)", "merge-final off", "appearance off",
             PROTOCOL_VERSION),
            ("v8 (adopted)", "lat <= 5.5 m", "merge-final on", "appearance on",
             PROTOCOL_VERSION_V8)]
    for i, (name, a, b, c, ver) in enumerate(rows):
        y = 0.72 - i * 0.30
        _box(ax, 0.02, y, 0.20, 0.20, name, fc="#DCE3F0" if i == 0 else "#E8F0DC")
        _box(ax, 0.24, y, 0.17, 0.20, a)
        _box(ax, 0.43, y, 0.17, 0.20, b)
        _box(ax, 0.62, y, 0.17, 0.20, c)
        ax.text(0.81, y + 0.10, ver, fontsize=7.5, va="center")
    ax.text(0.02, 0.06, "active version = " + active_protocol_version()
            + "    protocol hash = " + protocol_hash() + NL
            + "one constant switches all three changes; the hash changes with the "
            + "version, so a seal/confirmation cannot be mislabelled", fontsize=7.5)
    ax.set_title("Protocol switch matrix (schematic) and the live hash", fontsize=9.5)
    _save(fig, "fig33_protocol_matrix")


def fig34() -> None:
    """一次性最终集流程（schematic）+ 实测封存/消费账。"""
    rec = _load("final_set_v5_20261005/seal/final_set_seal.json")
    led = (_load("final_set_v5_20261005/seal/final_set_ledger.jsonl")
           if (EXP / "final_set_v5_20261005/seal/final_set_ledger.jsonl").is_file()
           else None)
    cons = 0
    if led is not None:
        for ln in (EXP / "final_set_v5_20261005/seal/final_set_ledger.jsonl"
                   ).read_text(encoding="utf-8").splitlines():
            if ln.strip() and json.loads(ln).get("allowed"):
                cons += 1
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.set_axis_off()
    steps = [("compose\n(road-disjoint)", "#DCE3F0"), ("seal\n(content digest)", "#DCE3F0"),
             ("access gate\n(protocol hash)", "#FDF3D0"),
             ("evaluate once\n(pixels)", "#DCE3F0"),
             ("consumed\n(no re-read)", "#F3D9D9")]
    x = 0.02
    for i, (txt, fc) in enumerate(steps):
        _box(ax, x, 0.45, 0.18, 0.30, txt, fc=fc)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.20, 0.60), xytext=(x + 0.18, 0.60),
                        arrowprops=dict(arrowstyle="->", lw=0.8))
        x += 0.20
    ax.text(0.02, 0.30, f"sealed frames = {rec['n_frames']}   digest = {rec['digest']}"
            f"   protocol = {rec['protocol_hash']}   confirmations = {cons}",
            fontsize=8, va="top")
    ax.text(0.02, 0.12, "a refused access is recorded but does not consume; a consumed "
            "set is a diagnostic set (freeze a new one)", fontsize=7.5, va="top")
    ax.set_title("One-shot final confirmation flow (schematic) and the live seal",
                 fontsize=9.5)
    _save(fig, "fig34_final_set_flow")


FIGS = {
    "fig9": fig9, "fig10": fig10, "fig11": fig11, "fig12": fig12, "fig13": fig13,
    "fig14": fig14, "fig15": fig15, "fig16": fig16, "fig17": fig17, "fig18": fig18,
    "fig19": fig19, "fig20": fig20, "fig21": fig21, "fig22": fig22, "fig23": fig23,
    "fig24": fig24, "fig25": fig25, "fig26": fig26, "fig27": fig27, "fig28": fig28,
    "fig29": fig29, "fig30": fig30, "fig31": fig31, "fig32": fig32, "fig33": fig33,
    "fig34": fig34,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="只出某一张（fig9..fig34）")
    args = ap.parse_args()
    names = [args.only] if args.only else list(FIGS)
    fails = []
    for n in names:
        if n not in FIGS:
            print(f"unknown figure {n}")
            return 2
        try:
            FIGS[n]()
        except Exception as exc:                      # 单张失败不影响其余
            fails.append((n, f"{type(exc).__name__}: {exc}"))
            print(f"[fig] {n} FAILED: {type(exc).__name__}: {exc}")
    if fails:
        print("FAILED:", fails)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
