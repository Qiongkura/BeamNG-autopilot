"""Confirm line candidates against the ENGINE's own annotation (T08 prerequisite).

T08's acceptance needs candidates whose identity has been confirmed by an
independent source - "在人工/真值确认的候选身份上，减少错误关联".  The engine
renders an annotated frame from its own materials, so its line class is
independent of the perception model; this tool measures the perception's
line output against it in the IMAGE PLANE, per frame:

* precision = |perception_line AND engine_line| / |perception_line|
  (of what the model calls a line, how much the engine also calls a line);
* recall    = |AND| / |engine_line|
  (of the engine's line pixels, how much the model found);
* per-side coverage: engine line pixels left/right of the image centre,
  and how much of each side the model covers - a candidate set that only
  ever sees one side cannot support a two-sided reference.

Two limits are part of the result, not footnotes:

* the engine renders KNOWN line materials - worn or faded paint may be
  unlabelled, so low precision can mean "unlabelled", not "false";
* this is the image plane.  Per-CANDIDATE identity (which marking is the
  left boundary, which the divider) needs the camera model and a ground
  projection; the ring collector now saves the model per view, but a run
  recorded before that cannot answer the per-candidate question.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_marking_identity_probe.py \\
        --run logs/m5_seg/ident_probe_20260923/front_main \\
        --meta logs/m5_seg/ident_probe_20260923/meta.json \\
        --view front_main --json logs/goal_20260921/marking_identity.json
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

from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402
from beamng_autopilot.vision.projection import CameraModel  # noqa: E402

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from m5_geometry_audit import oracle_ground_point  # noqa: E402

CLS_LINE = 2

# --- ego-relative identity (vehicle frame), no lane/map assumption --------
#: |lateral| below this means the car is ON the line, not beside it.
STRADDLE_M = 0.5
#: A line this close on either side bounds the ego's lane.
NEAR_M = 5.0
#: Forward window in which a ground projection is believable for identity.
FWD_MIN_M = 2.0
FWD_MAX_M = 30.0
LAT_MAX_M = 12.0
#: Two lines are the same physical line when their lateral offsets differ
#: by less than this (the plane model's own error is a separate matter).
MATCH_M = 0.8
LINE_BIN_M = 2.0
LINE_LAT_GAP_M = 0.5
#: A straight line OBLIQUE to the ego axis drifts laterally by
#: ``LINE_BIN_M * tan(theta)`` per bin, so a fixed chain step is really a
#: fixed ANGLE limit.  Measured on an urban junction: the engine's own
#: painted line drifts ~1 m per 2 m bin (the first version used 0.6 m and
#: broke that line into a stub).  The step therefore allows the geometry:
#: ``LINE_CHAIN_STEP_M + LINE_BIN_M * LINE_SLOPE_MAX`` (~35 deg).
LINE_CHAIN_STEP_M = 0.6
LINE_SLOPE_MAX = 0.7
LINE_MIN_BINS = 3


def camera_from_meta(meta: dict, view: str):
    """CameraModel for ``view`` from a collector's ``cameras`` block."""
    cam = (meta or {}).get("cameras", {}).get(view)
    if not cam:
        return None
    try:
        return CameraModel(
            offset=np.asarray(cam["offset"], dtype=float),
            fwd_local=np.asarray(cam["fwd"], dtype=float),
            up_local=np.asarray(cam["up"], dtype=float),
            fov_deg=float(cam["fov_deg"]),
            width=int(cam["width"]),
            height=int(cam["height"]))
    except (KeyError, TypeError, ValueError):
        return None


def compare_masks(perception_line, engine_line, *, centre_col=None) -> dict:
    """Image-plane precision / recall / side coverage of a line mask.

    Returns UNKNOWN-style fields (``None``) where the comparison is not
    defined: a frame with no engine line pixels has no recall to report,
    and reporting 1.0 there would be the classic "no data reads as pass".
    """
    perc = np.asarray(perception_line, dtype=bool)
    eng = np.asarray(engine_line, dtype=bool)
    if perc.shape != eng.shape:
        return {"reason": f"shape mismatch {perc.shape} vs {eng.shape}"}
    n_perc = int(perc.sum())
    n_eng = int(eng.sum())
    inter = int((perc & eng).sum())
    out = {"n_perception_px": n_perc, "n_engine_px": n_eng, "n_intersection": inter,
           "precision": (None if n_perc == 0 else round(inter / n_perc, 4)),
           "recall": (None if n_eng == 0 else round(inter / n_eng, 4)),
           "iou": (None if (n_perc + n_eng - inter) == 0 else
                   round(inter / (n_perc + n_eng - inter), 4))}
    if centre_col is None:
        centre_col = perc.shape[1] // 2
    c = int(centre_col)
    for side, sl in (("left", slice(0, c)), ("right", slice(c, None))):
        e = int(eng[:, sl].sum())
        p = int(perc[:, sl].sum())
        i = int((perc[:, sl] & eng[:, sl]).sum())
        out[f"engine_px_{side}"] = e
        out[f"perception_px_{side}"] = p
        out[f"recall_{side}"] = (None if e == 0 else round(i / e, 4))
    out["engine_sides"] = int(out["engine_px_left"] > 0) + int(
        out["engine_px_right"] > 0)
    out["perception_sides"] = int(out["perception_px_left"] > 0) + int(
        out["perception_px_right"] > 0)
    return out


def project_pixels(us, vs, cam, *, stride: int = 1,
                   fwd_min: float = FWD_MIN_M, fwd_max: float = FWD_MAX_M,
                   lat_max: float = LAT_MAX_M):
    """Pixel list -> vehicle-frame ground points via the T05 oracle.

    The oracle is imported, not re-derived: it is the independent closed
    form validated against the pipeline (p50 0.0000 m) in the geometry
    work.  Its flat-ground assumption is stated in the limits - a slope
    moves the FORWARD distance, while the lateral SIGN (which side of the
    car a line is on) is what identity needs.
    """
    pts = []
    for u, v in zip(np.asarray(us).ravel()[::stride],
                    np.asarray(vs).ravel()[::stride]):
        hit = oracle_ground_point(float(u), float(v), cam)
        if hit is None:
            continue
        fwd, lat = float(hit[0]), float(hit[1])
        if fwd < float(fwd_min) or fwd > float(fwd_max):
            continue
        if abs(lat) > float(lat_max):
            continue
        pts.append((fwd, lat))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def role_of(lat_m: float) -> str:
    """Ego-relative role of ONE line, from its lateral offset alone."""
    if lat_m is None:
        return "unknown"
    if abs(float(lat_m)) <= STRADDLE_M:
        return "straddled"
    return "left" if float(lat_m) > 0.0 else "right"


def assign_roles(lines) -> list:
    """Role per line, with ``near_``/``far_`` for the same side.

    Only the nearest line on a side can bound the ego's lane, so the role
    vocabulary distinguishes it from the second line further out - the
    distinction a downstream consumer needs and a bare left/right cannot
    give.
    """
    out = [dict(ln) for ln in lines]          # never mutate the caller's view
    for side in ("left", "right"):
        same = [i for i, ln in enumerate(out) if ln["role"] == side]
        same.sort(key=lambda i: abs(float(out[i]["lat_m"])))
        for rank, i in enumerate(same):
            out[i]["role"] = f"{'near' if rank == 0 else 'far'}_{side}"
    return out


def engine_lines(engine_mask, cam, *, stride: int = 3) -> list:
    """The engine's own lines as vehicle-frame clusters with roles.

    Dense labelled pixels are binned forward, split laterally and chained
    across bins; a chain that does not cover ``LINE_MIN_BINS`` bins is not
    a line (it is a patch - the same discipline the perception side uses).
    """
    vs, us = np.nonzero(np.asarray(engine_mask, dtype=bool))
    if len(us) == 0:
        return []
    pts = project_pixels(us, vs, cam, stride=stride)
    if len(pts) == 0:
        return []
    bins: dict[int, list] = {}
    for k in range(len(pts)):
        bins.setdefault(int(pts[k, 0] // LINE_BIN_M), []).append(k)
    segments = []
    for b in sorted(bins):
        idx = np.asarray(bins[b], dtype=int)
        order = idx[np.argsort(pts[idx, 1])]
        cur = [int(order[0])]
        for j in order[1:]:
            if float(pts[j, 1]) - float(pts[cur[-1], 1]) <= LINE_LAT_GAP_M:
                cur.append(int(j))
                continue
            if len(cur) >= 2:
                segments.append((b, float(np.median(pts[cur, 1])), cur))
            cur = [int(j)]
        if len(cur) >= 2:
            segments.append((b, float(np.median(pts[cur, 1])), cur))
    segments.sort(key=lambda t: (t[0], t[1]))
    chains: list[dict] = []
    for b, lat, members in segments:
        best = None
        for ch in chains:
            if ch["last_bin"] != b - 1:
                continue
            jump = abs(lat - ch["lat"])
            step_max = LINE_CHAIN_STEP_M + LINE_BIN_M * LINE_SLOPE_MAX
            if jump <= step_max and (best is None or jump < best[0]):
                best = (jump, ch)
        if best is None:
            chains.append({"first_bin": b, "last_bin": b, "lat": lat,
                           "bins": 1, "members": list(members)})
        else:
            ch = best[1]
            ch["last_bin"] = b
            ch["lat"] = lat
            ch["bins"] += 1
            ch["members"].extend(members)
    lines = []
    for ch in chains:
        if ch["bins"] < LINE_MIN_BINS:
            continue
        sel = np.unique(np.asarray(ch["members"], dtype=int))
        fwd = pts[sel, 0]
        lat = float(np.median(pts[sel, 1]))
        lines.append({"lat_m": round(lat, 3), "role": role_of(lat),
                      "fwd_span_m": [round(float(fwd.min()), 2),
                                     round(float(fwd.max()), 2)],
                      "n_px": int(len(sel))})
    return assign_roles(lines)


def candidate_label_breakdown(pixels, label) -> dict:
    """Which ENGINE class does a candidate's own pixel set sit on?

    Three outcomes matter and they are not the same statement:
    ``on_line`` (confirmed paint), ``on_road`` (on the driving surface but
    not paint - a boundary-like edge, a seam, a shadow) and neither (off
    the road entirely).  Measured on an urban junction: every candidate was
    on_road and NONE was on_line, while the engine's paint sat 6-12 m away.
    """
    px = np.asarray(pixels)
    if px.ndim != 2 or len(px) == 0:
        return {}
    lab = np.asarray(label)
    ui = np.clip(np.round(px[:, 0]).astype(int), 0, lab.shape[1] - 1)
    vi = np.clip(np.round(px[:, 1]).astype(int), 0, lab.shape[0] - 1)
    vals = lab[vi, ui]
    n = int(len(vals))
    return {"n_px": n,
            "on_line_frac": round(float((vals == CLS_LINE).mean()), 4),
            "on_road_frac": round(float((vals == 1).mean()), 4),
            "off_road_frac": round(float((vals == 0).mean()), 4)}


def match_candidate(cand_lat: float, engine: list, *,
                    tol_m: float = MATCH_M):
    """The engine line a candidate would be the SAME marking as."""
    best = None
    for ln in engine:
        d = abs(float(cand_lat) - float(ln["lat_m"]))
        if d <= float(tol_m) and (best is None or d < best[0]):
            best = (d, ln)
    return None if best is None else best[1]


def probe(run_dir: Path, meta: dict | None = None, *, view: str = "front_main",
          limit: int | None = None, model_path: str | None = None,
          null_shift_m: float | None = None) -> dict:
    fs = sorted(run_dir.glob("frame_*.npz"))
    if not fs:
        return {"reason": f"no frame_*.npz in {run_dir}"}
    if limit:
        fs = fs[:int(limit)]
    cam = camera_from_meta(meta or {}, view)
    net = HydraNet()
    if model_path:
        # the checkpoint is chosen on the SEGMENTER, not on the head: the
        # head is the pipeline, the segmenter owns the weights
        from beamng_autopilot.vision.segmentation import Segmenter
        net.add(SemanticHead(segmenter=Segmenter(model_path=model_path)))
    else:
        net.add(SemanticHead())
    rows = []
    for i, f in enumerate(fs):
        z = np.load(f)
        colour = np.asarray(z["colour"])
        if "label" not in z.files:
            return {"reason": f"{f.name}: no label array (not a labelled run)"}
        label = np.asarray(z["label"])
        ctx = FrameContext(frame_rgb=colour, cam=cam, pos=np.zeros(3),
                           heading=0.0, ground_z=0.0, role=view)
        out = net.run(ctx).get("semantic")
        line_mask = None
        if out is not None and "line" in out.masks:
            line_mask = np.asarray(out.masks["line"], dtype=bool)
        if line_mask is None:
            rows.append({"frame": i, "reason": "no line mask from the head"})
            continue
        stats = compare_masks(line_mask, label == CLS_LINE)
        stats["frame"] = i
        # --- ground-projected identity (vehicle frame) -------------------
        if cam is not None:
            eng_lines = engine_lines(label == CLS_LINE, cam)
            cands = []
            cand_px = np.zeros(label.shape, dtype=bool)
            for mk in ((out.meta.get("markings") if out is not None else None)
                       or []):
                px = np.asarray(mk.pixels) if mk.pixels is not None                     else np.empty((0, 2))
                if px.size == 0:
                    continue
                _ui = np.clip(np.round(px[:, 0]).astype(int), 0,
                              label.shape[1] - 1)
                _vi = np.clip(np.round(px[:, 1]).astype(int), 0,
                              label.shape[0] - 1)
                cand_px[_vi, _ui] = True
                gp = project_pixels(px[:, 0], px[:, 1], cam, stride=2)
                if len(gp) == 0:
                    continue
                lat = float(np.median(gp[:, 1]))
                prov = dict(getattr(mk, "meta", None) or {})
                cands.append({"kind": mk.kind, "colour": mk.color,
                              "lat_m": round(lat, 3),
                              "role": role_of(lat), "n_px": int(len(gp)),
                              "learned_frac": prov.get("learned_frac"),
                              "on_road_frac": prov.get("on_road_frac"),
                              "aspect": prov.get("aspect"),
                              **candidate_label_breakdown(px, label)})
            cands = assign_roles(cands)
            null_shift = float(null_shift_m or 0.0)
            eng_null = ([{**ln, "lat_m": ln["lat_m"] - null_shift}
                         for ln in eng_lines] if null_shift else [])
            matched = agree = n_null = 0
            by_role = {}
            for c in cands:
                by_role.setdefault(c["role"], 0)
                by_role[c["role"]] += 1
                ln = match_candidate(c["lat_m"], eng_lines)
                c["engine_lat_m"] = None if ln is None else ln["lat_m"]
                c["engine_role"] = None if ln is None else ln["role"]
                c["matched"] = ln is not None
                if ln is not None:
                    matched += 1
                    if ln["role"] == c["role"]:
                        agree += 1
                if eng_null and match_candidate(c["lat_m"], eng_null) is not None:
                    n_null += 1
            stats["engine_lines"] = eng_lines
            stats["candidates"] = cands
            stats["n_candidates"] = len(cands)
            stats["n_candidates_matched"] = matched
            stats["n_role_agreement"] = agree
            stats["match_rate"] = (None if not cands
                                   else round(matched / len(cands), 4))
            stats["role_agreement_rate"] = (None if not matched
                                            else round(agree / matched, 4))
            stats["candidate_roles"] = by_role
            # Did the candidate SET cover the engine's paint at all?  This
            # is the question the per-candidate fractions cannot answer: a
            # set can be entirely off-paint even when one candidate
            # overlaps a few paint pixels.
            eng_mask = (label == CLS_LINE)
            n_eng = int(eng_mask.sum())
            stats["candidate_paint_recall"] = (
                None if n_eng == 0 else
                round(float((cand_px & eng_mask).sum()) / n_eng, 4))
            stats["candidates_learned_backed"] = sum(
                1 for c in cands
                if (c.get("learned_frac") or 0.0) >= 0.5)
            stats["candidates_cv_only"] = sum(
                1 for c in cands
                if (c.get("learned_frac") or 0.0) < 0.5)
            stats["line_candidate_gate"] = (out.meta.get("line_candidates")
                                            if out is not None else None)
            stats["n_candidates_matched_null"] = n_null
            stats["candidates_on_line"] = sum(
                1 for c in cands if (c.get("on_line_frac") or 0.0) >= 0.5)
            stats["candidates_on_road_only"] = sum(
                1 for c in cands
                if (c.get("on_line_frac") or 0.0) < 0.5
                and (c.get("on_road_frac") or 0.0) >= 0.5)
            stats["candidates_off_road"] = sum(
                1 for c in cands
                if (c.get("on_line_frac") or 0.0) < 0.5
                and (c.get("on_road_frac") or 0.0) < 0.5)
        rows.append(stats)
    def p50(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and r[key] is not None]
        return None if not vals else round(float(np.median(vals)), 4)
    n_engine_seen = sum(1 for r in rows if (r.get("n_engine_px") or 0) > 0)
    def total(key):
        """Sum an integer counter across rows (bool is not a counter)."""
        vals = [r.get(key) for r in rows]
        return int(sum(v for v in vals if isinstance(v, int) and not
                       isinstance(v, bool)))

    def total_len(key):
        """Number of ENTRIES across rows for a per-frame list field."""
        return int(sum(len(v) for v in (r.get(key) for r in rows)
                       if isinstance(v, (list, tuple))))

    n_cand = total("n_candidates")
    n_match = total("n_candidates_matched")
    n_agree = total("n_role_agreement")
    summary = {
        "frames": len(rows),
        "frames_with_engine_line": n_engine_seen,
        # ground-projected identity, vehicle frame
        "candidates_total": n_cand,
        "candidates_matched": n_match,
        # NULL control: the same measurement against engine lines shifted
        # sideways.  If the shifted rate is similar, "matches" are just
        # candidates landing inside the tolerance by chance.
        "null_shift_m": float(null_shift_m or 0.0),
        "candidates_matched_null": total("n_candidates_matched_null"),
        "match_rate_null": (None if not n_cand or not null_shift_m
                            else round(total("n_candidates_matched_null")
                                       / n_cand, 4)),
        "roles_agreeing": n_agree,
        "match_rate": (None if not n_cand else round(n_match / n_cand, 4)),
        "role_agreement_rate": (None if not n_match
                                else round(n_agree / n_match, 4)),
        "engine_lines_total": total_len("engine_lines"),
        # the three-valued statement per candidate, against the engine's
        # own classes: confirmed paint / on the road but unpainted / off it
        "candidates_on_engine_line": total("candidates_on_line"),
        "candidates_on_road_only": total("candidates_on_road_only"),
        "candidates_off_road": total("candidates_off_road"),
        "candidates_learned_backed": total("candidates_learned_backed"),
        "candidates_cv_only": total("candidates_cv_only"),
        "candidate_paint_recall_p50": p50("candidate_paint_recall"),
        "precision_p50": p50("precision"),
        "recall_p50": p50("recall"),
        "iou_p50": p50("iou"),
        "recall_left_p50": p50("recall_left"),
        "recall_right_p50": p50("recall_right"),
        "frames_both_engine_sides": sum(1 for r in rows
                                        if r.get("engine_sides") == 2),
        "frames_both_perception_sides": sum(1 for r in rows
                                            if r.get("perception_sides") == 2),
        "camera_model_used": cam is not None,
        "limits": [
            "Engine labels render KNOWN line materials: worn/faded paint may "
            "be unlabelled, so low precision can mean 'unlabelled'.",
            "Ground projection uses the T05 oracle's flat-plane model: a "
            "slope moves the FORWARD distance (0.9 m p50 at 8 deg); the "
            "lateral SIGN identity relies on is not affected by it.",
            "A candidate with no engine line nearby is 'unconfirmed', not "
            "'false': the engine renders known materials only, so faded or "
            "worn paint can be unlabelled.",
        ],
    }
    summary["model"] = str(model_path) if model_path else "default_model_path"
    return {"run": str(run_dir), "view": view, "summary": summary,
            "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with frame_*.npz")
    ap.add_argument("--meta", default=None, help="collector meta.json")
    ap.add_argument("--view", default="front_main")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--null-shift-m", type=float, default=None,
                    help="lateral shift applied to the engine lines as a "
                         "NULL control for the match rate")
    ap.add_argument("--model", default=None,
                    help="segmentation checkpoint (default: "
                         "segmentation.default_model_path())")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    meta = None
    if args.meta:
        mp = Path(args.meta)
        if not mp.is_file():
            print(f"[ident] no meta at {mp}; the camera model is unavailable, "
                  f"so only image-plane numbers will be produced")
        else:
            meta = json.loads(mp.read_text(encoding="utf-8"))
    res = probe(Path(args.run), meta, view=args.view, limit=args.limit,
                model_path=args.model, null_shift_m=args.null_shift_m)
    if "reason" in res:
        print(f"[ident] {res['reason']}")
        return 2
    s = res["summary"]
    print(f"[ident] {res['run']} view={res['view']} frames={s['frames']} "
          f"(engine line in {s['frames_with_engine_line']}) camera_model="
          f"{s['camera_model_used']}")
    if s.get("match_rate_null") is not None:
        print(f"[ident] NULL control (lines shifted {s['null_shift_m']} m): "
              f"match {s['candidates_matched_null']}/{s['candidates_total']} "
              f"= {s['match_rate_null']}")
    print(f"[ident] precision p50={s['precision_p50']} recall p50="
          f"{s['recall_p50']} iou p50={s['iou_p50']} | side recall L/R="
          f"{s['recall_left_p50']}/{s['recall_right_p50']} | both sides: "
          f"engine {s['frames_both_engine_sides']} / perception "
          f"{s['frames_both_perception_sides']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[ident] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
