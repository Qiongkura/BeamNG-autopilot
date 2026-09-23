"""Freeze a locked holdout from the collected segmentation runs (T10).

The plan's rule: once a set has been used to tune thresholds, rules or
weights it is DEVELOPMENT data, and the final test set must be frozen
separately - "任何规则/权重/标签调整后，原测试集转为开发资料，最终测试重新冻结".
This tool builds that locked set from run directories, refusing to include
any run that development work already touched, and writes a manifest whose
digest a later comparison must reproduce (``check_frozen_testset``).

The refusal is the point: silently freezing a run that the threshold scan
or the front-end ablation already read would produce a "test set" that is
really development data.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_freeze_holdout.py \\
        --runs logs/m5_seg/run_a logs/m5_seg/run_b ... \\
        --dev-runs logs/m5_seg/manual_20260906_130230 ... \\
        --out logs/goal_20260921/frozen_holdout.json \\
        --json logs/goal_20260921/frozen_holdout_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.vision.dataset_split import (  # noqa: E402
    FrameRef, check_frozen_testset, coverage_digest, freeze_testset,
    split_by_group,
)


def _refs_for_run(run_dir: Path, start: int) -> tuple[list, int]:
    """FrameRefs for one run directory (index is the global position).

    The identity is ``<collection>/<view>`` (e.g. ``holdout_lines_20260923/
    front_main``), NOT the bare directory name: two collections recorded on
    different days both have a ``front_main`` directory, and a manifest that
    cannot tell them apart digests to the same value - measured, the first
    two freezes collided on the same digest.
    """
    fs = sorted(run_dir.glob("frame_*.npz"))
    identity = (f"{run_dir.parent.name}/{run_dir.name}"
                if run_dir.parent.name else run_dir.name)
    refs = [FrameRef(index=start + i, run=identity, t=float(i),
                     t_is_index=True) for i in range(len(fs))]
    return refs, start + len(fs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories that MAY be frozen")
    ap.add_argument("--dev-runs", nargs="*", default=[],
                    help="run names (or paths) already used for development; "
                         "freezing one of these is refused")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="refuse to freeze a set smaller than this: a "
                         "1-frame holdout is an artefact, not a test set "
                         "(measured: a failed walk froze exactly that)")
    ap.add_argument("--val-frac", type=float, default=0.35)
    ap.add_argument("--holdout-all", action="store_true",
                    help="the VALIDATION side is every frame of --runs "
                         "(a locked holdout is a whole held-out set, not a "
                         "fraction of the runs the model trained on)")
    ap.add_argument("--out", required=True, help="manifest path")
    ap.add_argument("--json", default=None, help="report path")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest")
    args = ap.parse_args(argv)

    dev = {Path(p).name for p in args.dev_runs}
    run_dirs = [Path(p) for p in args.runs]
    missing = [str(p) for p in run_dirs if not list(p.glob("frame_*.npz"))]
    if missing:
        print(f"[freeze] no frame_*.npz in: {missing}")
        return 2
    offenders = sorted({p.name for p in run_dirs} & dev)
    if offenders:
        print(f"[freeze] REFUSING: these runs were used for development and "
              f"cannot become a locked test set -> {offenders}")
        return 3
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[freeze] {out} exists; pass --force to replace a frozen set")
        return 4

    refs: list = []
    nxt = 0
    per_run: dict = {}
    for rd in run_dirs:
        r, nxt = _refs_for_run(rd, nxt)
        refs.extend(r)
        per_run[rd.name] = len(r)
    if len(refs) < int(args.min_frames) and not args.force:
        print(f"[freeze] REFUSING: only {len(refs)} frame(s); a set this "
              f"small is an artefact - pass --force if that is really "
              f"intended")
        return 5
    plan = split_by_group(refs, val_frac=float(args.val_frac))
    if args.holdout_all:
        # A locked holdout is the WHOLE held-out set: no frame of it may
        # appear on the training side, which is exactly what freezing the
        # validation side means.
        from beamng_autopilot.vision.dataset_split import SplitPlan
        plan = SplitPlan(train=[], val=list(refs),
                         groups_val=sorted({r.group for r in refs}),
                         notes=["holdout-all: validation is the whole set"])
    report = {
        "runs": per_run,
        "dev_runs_excluded": sorted(dev),
        "n_frames": len(refs),
        "n_train": len(plan.train),
        "n_val": len(plan.val),
        "groups_train": plan.groups_train,
        "groups_val": plan.groups_val,
        "coverage": coverage_digest(plan, refs),
        "notes": plan.notes,
    }
    frozen = freeze_testset(
        plan, out,
        note=("locked holdout for final comparisons; development runs "
              f"excluded: {sorted(dev)}"))
    report.update(frozen)
    report["check"] = check_frozen_testset(out, plan)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    print(f"[freeze] {frozen['n_frames']} val frames frozen, digest "
          f"{frozen['digest']}")
    print(f"[freeze] train {report['n_train']} frames from "
          f"{len(report['groups_train'])} group(s); val groups "
          f"{report['groups_val']}")
    print(f"[freeze] re-check: {report['check']['ok']} "
          f"({report['check']['reason']})")
    print("[freeze] NOTE: any later threshold/rule/weight change turns this "
          "set into development data - freeze a new one instead of reusing it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
