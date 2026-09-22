"""Time-slice annotation: one click per lane line instead of one per frame.

The plan (T10) proposes exactly this tool and states its limits, which are
part of the implementation:

* fixed image rows are stacked into a time slice (x = frame, y = image
  row), the annotator places CONTROL POINTS in that image, and the curve
  is interpolated along TIME between them, then snapped to the stroke in
  each frame;
* it only applies to a continuous, trackable identity - lane changes,
  occlusions and identity changes must BREAK the segment, so the tool
  takes explicit break frames and never bridges them;
* what it produces are ASSISTED curve labels marked
  ``inferred_extension`` with the control points they came from.  They are
  never written as measured paint (the schema refuses that), and the tool
  reports a per-frame human spot check instead of promising a speed-up.

Everything is file-driven (clicks and spot checks come from JSON), so the
arithmetic is testable without a GUI.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_timeslice_annotate.py \\
        --episode logs/live_runs/shadow_episodes/<ep>.npz \\
        --rows 150 190 --clicks clicks.json --breaks 40 41 \\
        --out labels.jsonl --json metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot.labeling import curve_schema as cs  # noqa: E402

#: Half-width (px) of the vertical window searched when snapping the
#: interpolated row to the stroke in one frame.
SNAP_WINDOW_PX = 14
#: A spot check counts as "within tolerance" at/below this row error.
SPOT_TOL_PX = 3.0


def build_timeslice(frames, rows, *, u0: int = 0, u1: int | None = None,
                    stat: str = "contrast"):
    """Stack the requested image rows over time.

    Each cell answers one question: **does the stroke cross this image row,
    inside this column window, in this frame?**  The default statistic is
    therefore the row's contrast (max - median) inside the window, not its
    mean: a mean over the whole row is dominated by the background gradient
    and threw away the stroke's column, which was measured on the first
    version of this tool.  ``u0``/``u1`` must bracket where the line is
    expected - that window IS the tool's assumption, and the spot check is
    how it is graded.
    """
    imgs = [np.asarray(f) for f in frames]
    if not imgs:
        return np.zeros((len(rows), 0), dtype=np.float32)
    h, w = imgs[0].shape[:2]
    lo = max(0, int(u0))
    hi = int(w if u1 is None else u1)
    out = np.zeros((len(rows), len(imgs)), dtype=np.float32)
    for j, img in enumerate(imgs):
        band = img[:, lo:hi]
        if band.ndim == 3:
            band = band.mean(axis=2)
        for i, r in enumerate(rows):
            r = int(np.clip(int(r), 0, h - 1))
            row = band[r].astype(np.float32)
            if row.size == 0:
                out[i, j] = 0.0
            elif stat == "mean":
                out[i, j] = float(np.mean(row))
            else:
                out[i, j] = float(np.max(row) - np.median(row))
    return out


def interpolate_rows(clicks, *, frame_start: int, frame_end: int,
                     breaks=()) -> dict:
    """Per-frame row for every frame in [start, end], piecewise in time.

    ``clicks`` is a list of ``[frame, row]`` control points.  Interpolation
    never crosses a BREAK: if a break falls between the two surrounding
    control points, the frames between them are UNKNOWN (omitted) instead
    of being bridged, and nothing is extrapolated beyond the last click.
    """
    ctrl = {int(f): float(r) for f, r in clicks}
    bset = {int(b) for b in breaks}
    keys = sorted(ctrl)
    out: dict[int, float] = {}
    for f in range(int(frame_start), int(frame_end) + 1):
        if f in ctrl:
            out[f] = ctrl[f]
            continue
        prev = [k for k in keys if k < f]
        nxt = [k for k in keys if k > f]
        if not prev or not nxt:
            continue
        f0, f1 = prev[-1], nxt[0]
        if any(f0 < b <= f1 for b in bset):
            continue
        w = (f - f0) / float(f1 - f0)
        out[f] = float(ctrl[f0] + w * (ctrl[f1] - ctrl[f0]))
    return out


def snap_to_stroke(frame, row_pred: float, *, window_px: int = SNAP_WINDOW_PX,
                   u0: int = 0, u1: int | None = None,
                   avoid: bool = False) -> dict:
    """Find the stroke's column near ``row_pred`` in one frame.

    The lane stroke is the brightest (white/yellow) run in the window, so
    the snap is a brightness argmax - it is a heuristic ASSIST, which is
    why the result is labelled inferred and spot-checked.  ``avoid=True``
    inverts it (a dark stroke), for markings that are darker than their
    surroundings.
    """
    img = np.asarray(frame)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    r = int(np.clip(round(float(row_pred)), 0, h - 1))
    lo = max(0, int(u0))
    hi = int(w if u1 is None else u1)
    row = img[r, lo:hi].astype(np.float32)
    if row.size == 0:
        return {"u": None, "v": float(row_pred), "score": None,
                "reason": "empty row window"}
    k = max(1, int(window_px))
    prof = np.convolve(row, np.ones(k) / k, mode="same")
    idx = int(np.argmin(prof) if avoid else np.argmax(prof))
    return {"u": float(lo + idx), "v": float(r), "score": float(prof[idx]),
            "reason": ""}


def timeslice_to_annotations(frames, clicks, *, rows, breaks=(),
                             role: str = "unknown",
                             curve_id: str = "c1", run: str | None = None,
                             map_name: str | None = None,
                             episode: str | None = None,
                             source_id: str | None = None,
                             window_px: int = SNAP_WINDOW_PX,
                             u0: int = 0, u1: int | None = None):
    """Turn control points into per-frame annotations (inferred segments).

    Returns ``(annotations, report)``.  Every annotation's segment is
    marked ``inferred_extension`` and carries ``derived_from`` ids of the
    control points, so nothing here can be mistaken for measured paint.
    """
    n = len(frames)
    if n == 0:
        return [], {"reason": "no frames"}
    predicted = interpolate_rows(clicks, frame_start=0, frame_end=n - 1,
                                 breaks=breaks)
    derived = [f"timeslice@{int(cf)}row{float(cr):.1f}"
               for cf, cr in sorted(clicks)]
    snapped: list[tuple[int, float, float | None]] = []
    runs: list[list] = []          # consecutive frames -> one segment each
    current: list = []
    for f in range(n):
        if f not in predicted:
            if current:
                runs.append(current)
                current = []
            continue
        hit = snap_to_stroke(frames[f], predicted[f], window_px=window_px,
                             u0=u0, u1=u1)
        snapped.append((f, float(predicted[f]), hit["u"]))
        current.append(cs.CurvePoint(
            u=(hit["u"] if hit["u"] is not None else -1.0),
            v=float(predicted[f]), frame=f))
        # a break is an identity/visibility boundary in the LABELS too, not
        # only in the interpolation: never let one segment span it
        if (f + 1) in {int(b) for b in breaks}:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    anns = []
    for pts in runs:
        seg = cs.CurveSegment(
            frame_start=pts[0].frame, frame_end=pts[-1].frame,
            source=cs.SRC_INFERRED, visible=True, derived_from=derived,
            points=pts)
        curve = cs.Curve(curve_id=curve_id, role=role, attributes={},
                         unknown=list(cs.ATTR_KEYS), segments=[seg])
        first = pts[0].frame
        anns.append(cs.FrameAnnotation(
            frame_index=first, image_w=int(np.asarray(frames[first]).shape[1]),
            image_h=int(np.asarray(frames[first]).shape[0]), curves=[curve],
            run=run, map_name=map_name, episode=episode, source_id=source_id,
            notes=["assisted time-slice label: inferred, not measured"]))

    report = {
        "frames": n,
        "records": len(anns),
        "runs": len(runs),
        "annotated_frames": int(sum(
            int(seg.frame_end) - int(seg.frame_start) + 1
            for a in anns for c in a.curves for seg in c.segments)),
        "control_points": len(clicks),
        "breaks": sorted(int(b) for b in breaks),
        "snapped": [{"frame": f, "v_pred": round(v, 2),
                     "u": (None if u is None else round(u, 2))}
                    for f, v, u in snapped],
        "limits": [
            "Assisted label: inferred_extension, never measured paint.",
            "Only valid on a continuous, trackable identity; breaks are "
            "mandatory at lane changes, occlusions and identity changes.",
            "The snap is a brightness heuristic; the spot check measures it.",
        ],
    }
    return anns, report


def spot_check(predicted_rows, manual_rows, *,
               tol_px: float = SPOT_TOL_PX) -> dict:
    """Compare assisted rows against per-frame human rows.

    ``predicted_rows`` and ``manual_rows`` map frame -> row.  Reports the
    error distribution and the share within tolerance; frames present in
    only one of the two are counted as UNKNOWN rather than scored.
    """
    common = sorted(set(predicted_rows) & set(manual_rows))
    if not common:
        return {"n": 0, "reason": "no frame has both rows",
                "unknown": sorted(set(predicted_rows) ^ set(manual_rows))}
    errs = np.array([abs(float(predicted_rows[f]) - float(manual_rows[f]))
                     for f in common], dtype=float)
    return {
        "n": len(common),
        "err_p50_px": round(float(np.median(errs)), 3),
        "err_max_px": round(float(errs.max()), 3),
        "within_tol_frac": round(float(np.mean(errs <= float(tol_px))), 4),
        "tol_px": float(tol_px),
        "unknown": sorted(set(predicted_rows) ^ set(manual_rows)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=str, required=True,
                    help="npz with an 'rgb' stack (HxWx3 per frame)")
    ap.add_argument("--rows", nargs="+", type=int, required=True,
                    help="image rows stacked into the time slice")
    ap.add_argument("--clicks", type=str, required=True,
                    help="JSON: [[frame, row], ...] control points")
    ap.add_argument("--breaks", nargs="*", type=int, default=[],
                    help="frames where the identity/visibility breaks")
    ap.add_argument("--role", type=str, default="unknown")
    ap.add_argument("--curve-id", type=str, default="c1")
    ap.add_argument("--u0", type=int, default=0)
    ap.add_argument("--u1", type=int, default=None)
    ap.add_argument("--spot-check", type=str, default=None,
                    help="JSON: [[frame, row], ...] human per-frame rows")
    ap.add_argument("--out", type=str, default=None, help="labels JSONL")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    z = np.load(args.episode, allow_pickle=True)
    frames = z["rgb"] if "rgb" in z.files else None
    if frames is None:
        print(f"[timeslice] {args.episode}: no 'rgb' stack")
        return 2
    clicks = [[float(a), float(b)] for a, b in json.loads(
        Path(args.clicks).read_text(encoding="utf-8"))]
    anns, report = timeslice_to_annotations(
        frames, clicks, rows=args.rows, breaks=args.breaks, role=args.role,
        curve_id=args.curve_id, episode=Path(args.episode).name,
        u0=args.u0, u1=args.u1)
    if args.spot_check:
        manual = {int(f): float(r) for f, r in json.loads(
            Path(args.spot_check).read_text(encoding="utf-8"))}
        predicted = {int(s["frame"]): float(s["v_pred"])
                     for s in report["snapped"]}
        report["spot_check"] = spot_check(predicted, manual)
    if args.out:
        cs.dump_jsonl(args.out, anns)
        report["labels_written"] = len(anns)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    print("[timeslice] " + json.dumps(
        {k: v for k, v in report.items() if k != "snapped"},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
