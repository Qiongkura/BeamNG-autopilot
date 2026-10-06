"""论文图表集（第三批，fig35–fig42）：把此前未成图的已记录数据全部用上。

    .venv\\Scripts\\python.exe scripts\\m5_paper_figures_extra.py [--only fig38]

来源（逐条对应 docs/T16_PAPER_FIGURES_20261006.md）：
  fig35 gpu_ledger_machine.json              —— 记账到的 GPU 分钟（按日期/任务）
  fig36 timing_retest_20260927.json          —— 推理延迟复测（静默前置 + 重复取较小值）
  fig37 pairing_v7v8_compare_20261005.json   —— 两套定义下的配对可用性（逐 run）
  fig38 lane_geometry_dev_20260928.json      —— 标线横向位置分布（人工修订池）
  fig39 t16_r3_density_dev/pool*_scenes.json —— 场景结构密度（R3 池筛选判据）
  fig40 e1_negative_pool_20260927.json       —— 负例池认证漏斗（合格/疑似/含线）
  fig41 collect_isolation_20260927.json      —— 空间隔离审计（与训练锚点的最小距离）
  fig42 final_pool_audit_20261005.json       —— 最终集来源包的内容级重叠审计
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5_paper_figures import C_CTRL, C_FACT, EXP, _save  # noqa: E402

NL = chr(10)


def _load(name: str) -> dict:
    return json.load(open(EXP / name, encoding="utf-8"))


# ---------------------------------------------------------------- fig 35
def fig35() -> None:
    led = _load("gpu_ledger_machine.json")
    dates = sorted(led)
    fams = collections.Counter()
    for d in dates:
        for k in (led[d].get("runs") or {}):
            fams[k.split("_")[0]] += 1
    top = [f for f, _ in fams.most_common(5)]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(dates))
    bottom = np.zeros(len(dates))
    for i, fam in enumerate(top + ["other"]):
        vals = []
        for d in dates:
            runs = led[d].get("runs") or {}
            v = sum(m for k, m in runs.items()
                    if (k.split("_")[0] == fam if fam != "other"
                        else k.split("_")[0] not in top))
            vals.append(float(v))
        ax.bar(x, vals, 0.55, bottom=bottom, label=fam,
               color=plt.get_cmap("tab10")(i % 10), edgecolor="black", linewidth=0.4)
        bottom += np.asarray(vals)
    for i, d in enumerate(dates):
        ax.text(i, bottom[i] + 4, f"{led[d].get('minutes', bottom[i]):.0f} min",
                ha="center", fontsize=7)
    ax.set_xticks(x, dates, fontsize=7.5)
    ax.set_ylabel("GPU minutes (recorded ledger)")
    ax.set_title("Recorded GPU-time ledger (three logged days) by task family")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    _save(fig, "fig35_gpu_ledger")


# ---------------------------------------------------------------- fig 36
def fig36() -> None:
    t = _load("timing_retest_20260927.json")
    models = t.get("models") or []
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    labels, p50s, p95s = [], [], []
    for m in models:
        name = Path(str(m.get("model", "?"))).parent.name or "model"
        for r in (m.get("repeats") or []):
            labels.append(f"{name[:14]}{NL}rep{r.get('i')}")
            p50s.append(r.get("p50"))
            p95s.append(r.get("p95"))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, p50s, 0.4, label="p50", color=C_CTRL, edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, p95s, 0.4, label="p95", color=C_FACT, edgecolor="black", linewidth=0.4)
    for i, (a, b) in enumerate(zip(p50s, p95s)):
        ax.text(i - 0.2, (a or 0) + 0.2, f"{a}", ha="center", fontsize=6)
        ax.text(i + 0.2, (b or 0) + 0.2, f"{b}", ha="center", fontsize=6)
    ax.set_xticks(x, labels, fontsize=6)
    ax.set_ylabel("inference latency (ms)")
    ax.set_ylim(0, max([v for v in p95s if v] or [1]) * 1.25)
    ax.set_title(f"Timing hygiene: repeated inference on a quiet machine "
                 f"(game running = {t.get('game_running')}, gpu util = {t.get('gpu_util')})")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "fig36_timing_retest")


# ---------------------------------------------------------------- fig 37
def fig37() -> None:
    d = _load("pairing_v7v8_compare_20261005.json")
    runs = d.get("runs") or []
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x = np.arange(len(runs))
    v7 = [r.get("v7_paired") for r in runs]
    v8 = [r.get("v8_paired") for r in runs]
    ax.bar(x - 0.2, v7, 0.4, label="baseline definition", color=C_CTRL,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, v8, 0.4, label="adopted definition", color=C_FACT,
           edgecolor="black", linewidth=0.4)
    for i, r in enumerate(runs):
        ax.text(i, max(v7[i] or 0, v8[i] or 0) + 2,
                f"-{len(r.get('lost') or [])}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"run {i + 1}{NL}({d.get('frames')} frames)" for i in range(len(runs))],
                  fontsize=7)
    ax.set_ylabel("frames with a paired lane")
    ax.set_title("Pairing availability under the two definitions "
                 "(label: frames lost by the adopted scope)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "fig37_pairing_compare")


# ---------------------------------------------------------------- fig 38
def fig38() -> None:
    g = _load("lane_geometry_dev_20260928.json")
    hist = (g.get("pooled") or {}).get("hist") or {}
    if not hist:
        raise SystemExit("lane_geometry: no pooled histogram")
    keys = list(hist)
    vals = [hist[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.bar(range(len(keys)), vals, color="#8172B3", edgecolor="black", linewidth=0.3)
    step = max(1, len(keys) // 10)
    ax.set_xticks(range(0, len(keys), step), [keys[i] for i in range(0, len(keys), step)],
                  rotation=45, ha="right", fontsize=6)
    pooled = g.get("pooled") or {}
    ax.set_ylabel("marking pixels")
    ax.set_title(f"Lateral position of line markings relative to the lane centre"
                 f" (n={pooled.get('n')}, median {pooled.get('p50')} m, "
                 f"|median| {pooled.get('abs_p50')} m)")
    _save(fig, "fig38_lane_geometry")


# ---------------------------------------------------------------- fig 39
def fig39() -> None:
    sets = [("dev (reviewed)", "t16_r3_density_dev.json"),
            ("R3 pool", "t16_r3_pool_density_scenes.json"),
            ("R3 pool B", "t16_r3_pool_b_density_scenes.json")]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for i, (label, fn) in enumerate(sets):
        try:
            rows = _load(fn)
        except Exception:
            continue
        vals = [r.get("density") for r in rows if isinstance(r.get("density"), (int, float))]
        if not vals:
            continue
        ax.scatter([i] * len(vals), vals, s=22, color=[C_CTRL, C_FACT, "#55A868"][i],
                   edgecolor="black", linewidth=0.3)
        ax.hlines(float(np.median(vals)), i - 0.25, i + 0.25, color="black", lw=1.4)
    ax.set_xticks(range(len(sets)), [s[0] for s in sets], fontsize=8)
    ax.set_ylabel("near-field structure density")
    ax.set_title("Scene screening criterion: structure density of the certified pools "
                 "(black bar = median)")
    _save(fig, "fig39_scene_density")


# ---------------------------------------------------------------- fig 40
def fig40() -> None:
    d = _load("e1_negative_pool_20260927.json")
    dirs = d.get("dirs") or {}
    eligible = sum(v.get("n_eligible", 0) for v in dirs.values())
    suspect = sum(v.get("n_suspect", 0) for v in dirs.values())
    total = sum(v.get("n_frames", 0) for v in dirs.values())
    line_like = sum(v.get("n_model_paint", 0) for v in dirs.values())
    unknown = max(0, total - eligible - suspect - line_like)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    names = ["certified\nnegative", "suspect\n(excluded)", "contains paint\n(excluded)",
             "undecided\n(excluded)"]
    vals = [eligible, suspect, line_like, unknown]
    ax.bar(names, vals, color=["#55A868", C_FACT, C_FACT, "#999999"],
           edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v}", ha="center", fontsize=7.5)
    crit = d.get("criteria") or {}
    ax.set_ylabel(f"frames (total {total})")
    ax.set_title(f"Negative-pool certification funnel "
                 f"(min paint px = {crit.get('min_paint_px')}, "
                 f"white v_min = {crit.get('white_v_min')})")
    _save(fig, "fig40_negative_audit")


# ---------------------------------------------------------------- fig 41
def fig41() -> None:
    d = _load("collect_isolation_20260927.json")
    cands = d.get("candidates") or {}
    labels, frames, verdicts = [], [], []
    for k, v in cands.items():
        labels.append(Path(k).name[:22])
        frames.append(v.get("frames_with_pose", 0))
        verdicts.append(str(v.get("verdict", "?")))
    order = np.argsort(frames)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    colors = {"isolated_other_map": C_CTRL, "isolated": "#55A868"}
    ax.barh(range(len(labels)), [frames[i] for i in order],
            color=[colors.get(verdicts[i], C_FACT) for i in order],
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(labels)), [labels[i] for i in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("frames with pose")
    ax.set_title(f"Spatial-isolation audit (buffer {d.get('buffer_m')} m; "
                 f"dev frames with pose = {d.get('dev_frames_with_pose')}; "
                 f"all isolated = {d.get('all_isolated')})")
    _save(fig, "fig41_isolation_audit")


# ---------------------------------------------------------------- fig 42
def fig42() -> None:
    d = _load("final_pool_audit_20261005.json")
    rows = d.get("dirs") or []
    labels = [Path(r["dir"]).parents[0].name[:26] for r in rows]
    frames = [r.get("frames", 0) for r in rows]
    overlap = [r.get("overlap", 0) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.bar(x - 0.2, frames, 0.4, label="frames", color=C_CTRL,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, overlap, 0.4, label="already used (content-level overlap)",
           color=C_FACT, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("frames")
    ax.set_title(f"Final-set provenance audit: content-level overlap with the used set "
                 f"({d.get('used_shas')} used frame hashes)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "fig42_overlap_audit")


FIGS = {"fig35": fig35, "fig36": fig36, "fig37": fig37, "fig38": fig38,
        "fig39": fig39, "fig40": fig40, "fig41": fig41, "fig42": fig42}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    names = [args.only] if args.only else list(FIGS)
    bad = []
    for n in names:
        if n not in FIGS:
            print(f"unknown figure {n}"); return 2
        try:
            FIGS[n]()
        except Exception as exc:
            bad.append((n, f"{type(exc).__name__}: {exc}"))
            print(f"[fig] {n} FAILED: {type(exc).__name__}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
