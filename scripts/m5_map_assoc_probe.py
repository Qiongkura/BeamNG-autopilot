"""T08 shadow probe: real map prior vs real perception candidates, one frame.

Reads what actually exists - the map link under the ego from
``connector.read_current_road_rule`` and the perception candidates from the
semantic line pipeline - and prints the map-assisted association beside the
perception-only score.  Nothing is written back into the stack: this is the
falsifiable shadow comparison, not a wiring.

Usage::

    .venv\\Scripts\\python.exe scripts\\m5_map_assoc_probe.py --attach \\
        --json logs/goal_20260921/map_assoc_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402
from beamng_autopilot.labeling.tech_annotation import (  # noqa: E402
    prepare_annotation_sample,
)
from beamng_autopilot_tech.annotations import (  # noqa: E402
    annotation_palette,
)
from m5_marking_identity_probe import (  # noqa: E402
    candidate_label_breakdown,
)
from beamng_autopilot.lane.map_association import (  # noqa: E402
    DIR_HARD_DEG,
    DIR_TOL_DEG,
    SENSE_FLIP_DEG,
    MapLinkPrior,
    PerceptionCandidate,
    compare_arms,
    direction_delta_deg,
    direction_reference_rad,
)
from beamng_autopilot.runtime import build_camera_ring_provider  # noqa: E402
from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402


_JOINT_RULE = "not hard-conflicted AND on_line_frac >= 0.5"


def _new_counters() -> dict:
    """One arm's raw counters; the block builder turns them into evidence."""
    return {"n": 0, "accepted": 0, "hard": 0, "soft": 0, "abstain": 0,
            "rank_changed": 0, "sense_measured": 0, "sense_flipped": 0,
            "measured": 0, "judged": 0, "on_paint": 0, "on_road": 0,
            "off_road": 0, "false_acc": 0, "false_rej": 0,
            "paint_only_acc": 0, "paint_only_conflicted": 0,
            "joint_acc": 0, "joint_drop_paint": 0, "joint_drop_road": 0,
            "joint_drop_offroad": 0}


def _accumulate(c: dict, rows, by_id) -> None:
    """Add one frame's rows to an arm's counters.

    ``false_acceptance`` is the plan's required quantity: a candidate the
    prior did NOT reject that the ENGINE's own annotation does not confirm
    as paint.  ``judged`` is the denominator that belongs to it - abstained
    candidates are refusals, so they are neither an acceptance nor a
    rejection and must not enter the rate.  Rows without a paint fraction
    are not counted at all (unmeasured is not zero).
    """
    for row in rows:
        c["n"] += 1
        conf = row.get("conflicts") or []
        hard = any("direction_mismatch" in str(x) for x in conf)
        soft = any("direction_loose" in str(x) for x in conf)
        abstain = bool(row.get("abstain"))
        if abstain:
            c["abstain"] += 1
        elif hard:
            c["hard"] += 1
        elif soft:
            c["soft"] += 1
        else:
            c["accepted"] += 1
        cand = by_id.get(row["candidate_id"], {})
        sense = row.get("sense_delta_deg")
        if sense is not None:
            c["sense_measured"] += 1
            if float(sense) > SENSE_FLIP_DEG:
                c["sense_flipped"] += 1
        frac = cand.get("on_line_frac")
        if frac is None:
            continue
        frac = float(frac)
        road_frac = float(cand.get("on_road_frac") or 0.0)
        c["measured"] += 1
        if frac >= 0.5:
            c["on_paint"] += 1
        elif road_frac >= 0.5:
            c["on_road"] += 1
        else:
            c["off_road"] += 1
        if abstain:
            continue
        c["judged"] += 1
        if hard and frac >= 0.5:
            c["false_rej"] += 1
        if (not hard) and frac < 0.5:
            c["false_acc"] += 1
        if frac >= 0.5:
            # The PAINT-ONLY gate: what the engine's own paint alone would
            # keep, and how much of that the direction test condemns.  The
            # joint gate is only worth its price when these disagree.
            c["paint_only_acc"] += 1
            if hard:
                c["paint_only_conflicted"] += 1
        if not hard and frac >= 0.5:
            c["joint_acc"] += 1
        elif frac >= 0.5:
            c["joint_drop_paint"] += 1
        elif road_frac >= 0.5:
            c["joint_drop_road"] += 1
        else:
            c["joint_drop_offroad"] += 1


def _block_from(c: dict) -> dict:
    """The evidence block for one arm: what it keeps, and what that costs."""
    block = {"n_candidates": c["n"], "accepted": c["accepted"],
             "hard": c["hard"], "soft": c["soft"], "abstain": c["abstain"]}
    if c["sense_measured"]:
        block["direction_sense"] = {
            "measured": c["sense_measured"],
            "flipped": c["sense_flipped"],
            "flipped_frac": round(c["sense_flipped"] / c["sense_measured"], 4),
            "note": ("the map link's stored order runs against the ego's "
                     "travel for these candidates: reported, NOT penalised - "
                     "a lane marking has no arrowhead, so the axis is graded"),
        }
    if c["measured"]:
        block["paint"] = {
            "measured": c["measured"], "judged": c["judged"],
            "on_paint": c["on_paint"], "on_road_only": c["on_road"],
            "off_road": c["off_road"],
            "false_acceptance": c["false_acc"],
            "false_acceptance_rate": (round(c["false_acc"] / c["judged"], 4)
                                      if c["judged"] else None),
            "false_rejection": c["false_rej"],
            "note": ("false acceptance counts every candidate the prior did "
                     "NOT reject (soft pass included) that the engine's paint "
                     "does not confirm; the rate is over the judged rows, "
                     "never over the abstained ones"),
        }
        block["joint_gate"] = {
            "rule": _JOINT_RULE,
            "accepted": c["joint_acc"],
            "dropped_paint_confirmed": c["joint_drop_paint"],
            "dropped_road_only": c["joint_drop_road"],
            "dropped_off_road": c["joint_drop_offroad"],
            "judged": c["judged"],
            "buckets_sum_to_judged": (c["joint_acc"] + c["joint_drop_paint"]
                                      + c["joint_drop_road"]
                                      + c["joint_drop_offroad"]) == c["judged"],
            "note": ("false acceptance is 0 BY CONSTRUCTION here; the price is "
                     "that boundary-type candidates (on the road but not on "
                     "paint) are dropped, which is why the gate is only usable "
                     "for paint candidates"),
        }
        block["paint_only_gate"] = {
            "rule": "on_line_frac >= 0.5 (engine paint alone, no direction test)",
            "accepted": c["paint_only_acc"],
            "direction_conflicted": c["paint_only_conflicted"],
            "clean": c["paint_only_acc"] - c["paint_only_conflicted"],
            "note": ("the joint gate is only worth its price where this and "
                     "'joint_gate.accepted' disagree: 'direction_conflicted' "
                     "is what the direction test removes from the paint-only "
                     "set, and 'dropped_paint_confirmed' is what that removal "
                     "costs if the reference was wrong"),
        }
    return block


def _direction_stats(deltas, source=None) -> dict:
    """Median / tail / in-threshold share of |bearing - reference| (deg).

    The denominator travels with every number; no measured bearing gives an
    empty block (UNKNOWN is not zero).
    """
    vals = [float(d) for d in (deltas or []) if d is not None]
    block: dict = {"reference_source": source, "n": len(vals)}
    if not vals:
        return block
    arr = np.asarray(vals, dtype=float)
    block.update({
        "median_deg": round(float(np.median(arr)), 2),
        "p90_deg": round(float(np.percentile(arr, 90)), 2),
        "max_deg": round(float(arr.max()), 2),
        "within_tol_frac": round(float((arr <= DIR_TOL_DEG).mean()), 4),
        "within_hard_frac": round(float((arr <= DIR_HARD_DEG).mean()), 4),
        "tol_deg": DIR_TOL_DEG, "hard_deg": DIR_HARD_DEG,
    })
    return block


def _direction_error(cands, prior, *, use_tangent: bool = False,
                     direction_source: str | None = None) -> dict:
    """Per-arm direction deviation for ONE frame's candidate set.

    This is the quantity the instruction asks to re-measure after swapping
    the reference from the 3.5 m ``in/outRadius`` arc to the map graph's
    link polyline: the deviation of every candidate's OWN bearing from the
    reference each arm would grade against.  It is computed with the same
    resolver the acceptance test uses, so the reported median cannot drift
    from what the arm actually decided on.
    """
    ref, source = direction_reference_rad(prior, use_tangent=use_tangent,
                                          direction_source=direction_source)
    deltas = []
    for c in cands:
        d = direction_delta_deg(c.bearing_rad, ref)
        if d is not None:
            deltas.append(round(float(d), 3))
    block = _direction_stats(deltas, source=source)
    block["delta_deg"] = deltas
    return block


def _pool_direction(frames_out) -> dict:
    """Pooled per-arm direction deviation over the whole frame set."""
    pooled: dict = {}
    for fr in frames_out:
        for arm, blk in (fr.get("arms_direction") or {}).items():
            entry = pooled.setdefault(arm, {"deltas": [], "medians": [],
                                            "source": blk.get(
                                                "reference_source")})
            entry["deltas"].extend(blk.get("delta_deg") or [])
            if blk.get("median_deg") is not None:
                entry["medians"].append(blk["median_deg"])
    out: dict = {}
    for arm, entry in pooled.items():
        block = _direction_stats(entry["deltas"], source=entry["source"])
        block["frames_measured"] = len(entry["medians"])
        block["per_frame_median_deg"] = entry["medians"]
        if entry["medians"]:
            block["median_of_frame_medians_deg"] = round(
                float(np.median(np.asarray(entry["medians"], dtype=float))), 2)
        out[arm] = block
    return out


def _aggregate(frames_out) -> dict:
    """Counts the T08 acceptance wants: verdicts, paint confirmation, A/B.

    Top-level fields describe the PRIMARY arm of each frame (the old
    single-arm view, so existing evidence stays comparable).  ``arms``
    carries one block per reference for the same frames - the paired
    single-factor table: the chord, the arc tangent from ``in/outRadius``
    and the tangent from the map graph's polyline are graded against ONE
    candidate set, because two separate runs are not a controlled
    comparison (measured 32 vs 43 candidates between runs).
    """
    agg = {"frames": len(frames_out), "n_candidates": 0, "accepted": 0,
           "hard": 0, "soft": 0, "abstain": 0, "rank_changed_frames": 0,
           "direction_sources": {}, "paint": {}, "arms": {}, "arm_order": []}
    prim = _new_counters()
    per_arm: dict = {}
    for fr in frames_out:
        src = fr.get("direction_source") or "primary"
        agg["direction_sources"][src] = agg["direction_sources"].get(
            src, 0) + 1
        if fr["rank_changed"]:
            agg["rank_changed_frames"] += 1
        by_id = {c["cand_id"]: c for c in fr["candidates"]}
        _accumulate(prim, fr["rows"], by_id)
        arms = fr.get("arms") or {src: fr["rows"]}
        for name, rows in arms.items():
            if name not in per_arm:
                per_arm[name] = _new_counters()
                agg["arm_order"].append(name)
            from_rank = (fr.get("arms_rank_changed") or {}).get(name)
            if from_rank is None:
                from_rank = bool(fr.get("rank_changed")) if name == src \
                    else False
            per_arm[name]["rank_changed"] += int(bool(from_rank))
            _accumulate(per_arm[name], rows, by_id)
    block = _block_from(prim)
    agg["n_candidates"] = block["n_candidates"]
    for k in ("accepted", "hard", "soft", "abstain"):
        agg[k] = block[k]
    agg["paint"] = block.get("paint", {})
    if "joint_gate" in block:
        agg["joint_gate"] = block["joint_gate"]
    if "paint_only_gate" in block:
        agg["paint_only_gate"] = block["paint_only_gate"]
    if "direction_sense" in block:
        agg["direction_sense"] = block["direction_sense"]
    for name in agg["arm_order"]:
        arm_block = _block_from(per_arm[name])
        arm_block["rank_changed_frames"] = int(
            per_arm[name].get("rank_changed", 0))
        agg["arms"][name] = arm_block
    pooled = _pool_direction(frames_out)
    for name, dir_block in pooled.items():
        if name in agg["arms"]:
            agg["arms"][name]["direction"] = dir_block
        else:
            agg["arms"][name] = {"direction": dir_block}
    return agg


def roadnet_tangent_rad(rn, pos, heading: float, *, half_span_m: float = 10.0,
                        radius_m: float = 60.0):
    """Local road direction from the MAP GRAPH's centre-line polyline.

    ``inRadius/outRadius`` turned out to be 3.5 m on this map (measured; not
    a road arc radius), so the tangent is taken from the map's own node graph
    instead - the same data the routing uses, chained into polylines by
    ``RoadNetwork.nearby_polylines``.  The direction is the chord of that
    polyline within ``+-half_span_m`` of the nearest point, oriented AWAY
    from the car (the same convention the candidates use).  Returns
    ``(tangent_rad, span_m, offset_m)`` or ``(None, None, None)``.
    """
    if rn is None or not getattr(rn, "ready", False):
        return None, None, None
    p = np.asarray(pos, dtype=float)[:2]
    try:
        polys = rn.nearby_polylines(pos, radius=float(radius_m))
    except Exception:
        return None, None, None
    best = None
    for pl in polys or []:
        arr = np.asarray(pl, dtype=float)
        if arr.ndim != 2 or len(arr) < 2 or arr.shape[1] < 2:
            continue
        arr = arr[:, :2]
        d = np.linalg.norm(arr - p, axis=1)
        i = int(np.argmin(d))
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s_arc = np.concatenate([[0.0], np.cumsum(seg)])
        m = (s_arc >= s_arc[i] - float(half_span_m)) & \
            (s_arc <= s_arc[i] + float(half_span_m))
        q = arr[m]
        if len(q) < 2:
            # The window is measured on NODE arc positions: with a 10 m node
            # spacing a +-10 m window can hold a single node, so no chord can
            # be formed (measured: the bend case returned None).  Fall back
            # to the nearest node's immediate neighbours.
            q = arr[max(0, i - 1):min(len(arr), i + 2)]
        if len(q) < 2:
            continue
        v = q[-1] - q[0]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        u = v / n
        fwd = np.array([math.cos(float(heading)), math.sin(float(heading))])
        if float(u @ fwd) < 0.0:
            u = -u
        if best is None or float(d[i]) < best[0]:
            best = (float(d[i]), float(math.atan2(u[1], u[0])), n)
    if best is None:
        return None, None, None
    return best[1], best[2], best[0]


def _candidates_from_markings(markings, pos, heading) -> list[PerceptionCandidate]:
    """Perception candidates WITH identity (kind, side, bearing, span).

    The bearing is normalised to point AWAY from the car (forward).  The
    extractor does not guarantee a near->far point order, so an unnormalised
    ``world[-1] - world[0]`` is a coin flip in sign - measured live on two
    painted stretches: both candidates on a straight link came back as
    "179 deg" / "154 deg" direction conflicts, i.e. a convention artefact
    rather than two wrong markings.  Only the EGO heading is used here: no
    map, no lateral geometry.
    """
    out: list[PerceptionCandidate] = []
    p = np.asarray(pos, dtype=float)[:2]
    fwd = np.array([np.cos(float(heading)), np.sin(float(heading))])
    left = np.array([-np.sin(float(heading)), np.cos(float(heading))])
    for i, m in enumerate(markings or []):
        world = np.asarray(getattr(m, "world", None), dtype=float) \
            if getattr(m, "world", None) is not None else None
        if world is None or world.ndim != 2 or len(world) < 2:
            continue
        rel = world[:, :2] - p
        lateral = float(np.median(rel @ left))
        span = float(np.linalg.norm(world[-1, :2] - world[0, :2]))
        seg = world[-1, :2] - world[0, :2]
        if float(np.linalg.norm(seg)) > 1e-6 and float(seg @ fwd) < 0.0:
            seg = -seg                       # orient away from the car
        bearing = (float(np.arctan2(seg[1], seg[0]))
                   if float(np.linalg.norm(seg)) > 1e-6 else None)
        out.append(PerceptionCandidate(
            cand_id=f"mk{i}",
            side=("left" if lateral > 0.2 else
                  "right" if lateral < -0.2 else "centre"),
            kind=str(getattr(m, "kind", "") or "unknown"),
            bearing_rad=bearing, confidence=float(getattr(m, "conf", 0.5)
                                                  or 0.5),
            span_m=span, fresh=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="T08 shadow map-association probe")
    ap.add_argument("--runtime", choices=("auto", "steam", "tech"),
                    default="tech")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--map", type=str, default="italy")
    ap.add_argument("--vehicle", type=str, default="etk800")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--frames", type=int, default=1,
                    help="place the car forward and repeat: one frame gives a ''point'', "
                         "many frames give the false-acceptance distribution T08 asks for")
    ap.add_argument("--step-m", type=float, default=1.0)
    ap.add_argument("--annotations", action="store_true",
                    help="also grab the engine's annotated frame, so every candidate "
                         "can be checked against the engine's own paint")
    ap.add_argument("--direction",
                    choices=("tangent", "roadnet", "chord"), default="chord",
                    help="what the candidates are graded against: the raw link "
                         "CHORD (default, measured to agree best), the arc tangent "
                         "from in/outRadius (refuted on this map) or the tangent from "
                         "the ROADNET polyline of the local road")
    ap.add_argument("--model", default=None,
                    help="segmentation checkpoint for the candidates; the "
                         "default is segmentation.default_model_path()")
    ap.add_argument("--no-roadnet", action="store_true",
                    help="skip building the map graph; the roadnet tangent "
                         "arm is then absent from the paired comparison")
    args = ap.parse_args()

    conn = BeamNGConnector(
        args.map, args.vehicle,
        port=config.runtime_port(args.runtime),
        home=config.runtime_home(args.runtime))
    try:
        conn.open(launch=not args.attach)
        try:
            conn.attach_vehicle(already_open=True)
        except Exception:
            conn.load_scenario()
        rn = None
        if not args.no_roadnet:
            # The roadnet arm is part of the PAIRED comparison, not only of
            # --direction roadnet: the whole point is that all three
            # references are graded on one candidate set in one process.
            from beamng_autopilot.roadnet import RoadNetwork
            rn = RoadNetwork()
            _t0 = time.time()
            while not rn.ready and time.time() - _t0 < 120:
                try:
                    if rn.build(conn.bng):
                        break
                except Exception:
                    pass
                time.sleep(1)
            print(f"[map-assoc] roadnet ready={bool(rn.ready)} "
                  f"({time.time() - _t0:.1f}s)", flush=True)
        ring, mode = build_camera_ring_provider(
            conn, args.runtime, 320, 240,
            annotations=bool(args.annotations))
        pal = None
        if args.annotations:
            get_ann = getattr(conn.bng, "get_annotations", None)
            pal = annotation_palette(get_ann() if callable(get_ann) else None)
        net = None
        if ring is not None:
            net = HydraNet()
            if args.model:
                from beamng_autopilot.vision.segmentation import Segmenter
                net.add(SemanticHead(segmenter=Segmenter(
                    model_path=args.model)))
            else:
                net.add(SemanticHead())
        frames_out: list = []
        n_frames = max(1, int(args.frames))
        for k in range(n_frames):
            if k > 0 and float(args.step_m) > 0.0:
                st_prev = conn.get_state()
                hdg = float(st_prev.heading)
                p_prev = np.asarray(st_prev.pos, dtype=float)
                nx = float(p_prev[0]) + float(args.step_m) * float(np.cos(hdg))
                ny = float(p_prev[1]) + float(args.step_m) * float(np.sin(hdg))
                if not conn.safe_teleport(nx, ny,
                                          heading_deg=float(np.degrees(hdg))):
                    print(f"[map-assoc] frame {k}: placement failed; stopping")
                    break
                conn.step(8)
            st = conn.get_state()
            pos = np.asarray(st.pos, dtype=float)
            heading = float(st.heading)
            direction = np.asarray(getattr(st, "dir", (np.cos(heading),
                                                       np.sin(heading), 0.0)),
                                   dtype=float)
            prior = MapLinkPrior.from_road_rule(
                conn.read_current_road_rule(pos, direction), pos=pos)
            _rn_span = _rn_off = None
            _rn_tangent = None
            if rn is not None:
                _rn_tangent, _rn_span, _rn_off = roadnet_tangent_rad(
                    rn, pos, heading)
            cands: list[PerceptionCandidate] = []
            label = None
            if net is not None:
                snap = ring.grab_ring()
                role = ("front_main" if "front_main" in snap
                        else next(iter(snap)))
                frame, cam = snap[role]
                ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos,
                                   heading=heading,
                                   ground_z=G.ego_ground_z(pos), role=role)
                out = net.run(ctx).get("semantic")
                if out is not None:
                    cands = _candidates_from_markings(
                        out.meta.get("markings", []), pos, heading)
                    if args.annotations:
                        labels = ring.grab_ring_labels()
                        if labels:
                            r2 = (role if role in labels
                                  else next(iter(labels)))
                            colour, ann = labels[r2]
                            _s, _a = prepare_annotation_sample(
                                colour, ann, width=320, height=240, palette=pal)
                            label = np.asarray(_s["label"])
                            for c, m in zip(cands, out.meta.get("markings", [])):
                                pk = np.asarray(getattr(m, "pixels", None),
                                                dtype=float)
                                if pk.ndim == 2 and len(pk) and label is not None:
                                    c.__dict__.update(
                                        candidate_label_breakdown(pk, label))
            # PAIRED three-arm evaluation on the SAME frame: the chord, the
            # arc tangent (in/outRadius) and the roadnet-polyline tangent
            # are graded against one candidate set, so the comparison is a
            # single factor and not two separately-rendered runs (measured
            # 32 vs 43 candidates when the arms were run as separate
            # processes - that is not a paired comparison).
            arm_specs = [("chord", replace(prior, tangent_rad=None,
                                           direction_source="chord"),
                          False, "chord")]
            if prior.tangent_rad is not None:
                arm_specs.append(("tangent_arc", replace(
                    prior, direction_source="tangent_arc"), True,
                    "tangent_arc"))
            if _rn_tangent is not None:
                arm_specs.append(("tangent_roadnet", replace(
                    prior, tangent_rad=_rn_tangent,
                    direction_source="tangent_roadnet"), True,
                    "tangent_roadnet"))
            arms, arms_direction = {}, {}
            for _arm, _pr, _use_tan, _src in arm_specs:
                arms[_arm] = compare_arms(cands, _pr, use_tangent=_use_tan,
                                          direction_source=_src)
                arms_direction[_arm] = _direction_error(
                    cands, _pr, use_tangent=_use_tan,
                    direction_source=_src)
            primary = (args.direction if args.direction in arms
                       else next(iter(arms)))
            if args.direction == "roadnet" and "tangent_roadnet" not in arms:
                primary = next(iter(arms))
            elif args.direction == "tangent" and "tangent_arc" not in arms:
                primary = next(iter(arms))
            rep = arms[primary]
            frames_out.append({"frame": k, "link_id": prior.link_id,
                               "primary_arm": primary,
                               "direction_deg": (None if prior.direction_rad
                                                 is None else round(
                                                     np.degrees(
                                                         prior.direction_rad),
                                                     2)),
                               "tangent_deg": (None if prior.tangent_rad is None
                                               else round(np.degrees(
                                                   prior.tangent_rad), 2)),
                               "tangent_alt_deg": (
                                   None if getattr(prior, "tangent_alt_rad",
                                                   None) is None
                                   else round(np.degrees(
                                       prior.tangent_alt_rad), 2)),
                               "roadnet_tangent_deg": (
                                   None if _rn_tangent is None
                                   else round(np.degrees(_rn_tangent), 2)),
                               "direction_source": primary,
                               "tangent_span_m": _rn_span,
                               "roadnet_offset_m": _rn_off,
                               "arc_offset_m": prior.arc_offset_m,
                               "radius_m": prior.radius_m,
                               "n_candidates": len(cands),
                               "rows": rep["arm_b"],
                               "rank_changed": rep["rank_changed"],
                               "arms": {a: r["arm_b"] for a, r in arms.items()},
                               "arms_direction": arms_direction,
                               "arms_rank_changed": {
                                   a: r["rank_changed"]
                                   for a, r in arms.items()},
                               "candidates": [dict(c.__dict__) for c in cands]})
        rep = (frames_out[0] if frames_out else {})
        prior = MapLinkPrior.from_road_rule(None)
        payload = {"runtime": mode,
                   "map_link": {
                       "link_id": prior.link_id,
                       "direction_deg": (None if prior.direction_rad is None
                                         else round(np.degrees(
                                             prior.direction_rad), 2)),
                       "lanes_hint": prior.lanes_hint,
                       "one_way": prior.one_way,
                       "right_hand_drive": prior.right_hand_drive,
                       "curvature_1pm": (None if prior.curvature_1pm is None
                                         else round(prior.curvature_1pm, 5))},
                   "n_candidates": len(cands),
                   "candidates": [c.__dict__ for c in cands],
                   "comparison": rep}
        payload = {"runtime": mode, "frames": frames_out}
        agg = _aggregate(frames_out)
        payload["aggregate"] = agg
        for fr in frames_out[:1] if len(frames_out) == 1 else []:
            print(f"[map-assoc] link={fr['link_id']} "
                  f"dir={fr['direction_deg']} tangent={fr['tangent_deg']} "
                  f"alt={fr['tangent_alt_deg']} src={fr['direction_source']} "
                  f"arc_s={fr['arc_offset_m']} R={fr['radius_m']}")
        print(f"[map-assoc] {agg['frames']} frame(s) | candidates "
              f"{agg['n_candidates']} | direction_source "
              f"{agg['direction_sources']}")
        print(f"[map-assoc] candidate verdicts: accepted "
              f"{agg['accepted']} / hard-conflict {agg['hard']} / "
              f"soft {agg['soft']} / abstain {agg['abstain']}")
        if agg.get("paint", {}).get("measured"):
            p = agg["paint"]
            print(f"[map-assoc] engine-paint check ({p['measured']} candidates "
                  f"measured): on_paint {p['on_paint']} / on_road_only "
                  f"{p['on_road_only']} / off_road {p['off_road']}")
            print(f"[map-assoc] FALSE ACCEPTANCE: prior accepted and NOT "
                  f"paint-confirmed = {p['false_acceptance']}"
                  f" ({p['false_acceptance_rate']}) | FALSE REJECTION: "
                  f"hard-conflicted but paint-confirmed = "
                  f"{p['false_rejection']}")
        print(f"[map-assoc] rank changed frames: {agg['rank_changed_frames']}"
              f"/{agg['frames']}")
        if agg.get("arms"):
            print("[map-assoc] PAIRED ARMS (one candidate set per frame; "
                  "same frames for every row):")
            print("    arm              ref_source        n   med|d|  "
                  "<hard   paint  false_acc  joint_acc/drop_paint/drop_road/"
                  "drop_offroad  paint_only(conflicted)")
            for name in agg["arm_order"]:
                a = agg["arms"].get(name) or {}
                d = a.get("direction") or {}
                p = a.get("paint") or {}
                j = a.get("joint_gate") or {}
                po = a.get("paint_only_gate") or {}
                print(f"    {name:16s} {str(d.get('reference_source')):16s} "
                      f"{d.get('n', 0):3d} {str(d.get('median_deg')):>7s} "
                      f"{d.get('within_hard_frac')}   "
                      f"{(p.get('on_paint') if p else '-'):>5}  "
                      f"{p.get('false_acceptance', '-')}/{p.get('judged', '-')}"
                      f"{'=' + str(p.get('false_acceptance_rate')) if p else ''}"
                      f"   {j.get('accepted', '-')}/{j.get('dropped_paint_confirmed', '-')}"
                      f"/{j.get('dropped_road_only', '-')}"
                      f"/{j.get('dropped_off_road', '-')}"
                      f"   {po.get('accepted', '-')}"
                      f"({po.get('direction_conflicted', '-')})")
            for name in agg["arm_order"]:
                ds = (agg["arms"].get(name) or {}).get("direction_sense") or {}
                if ds:
                    print(f"    {name}: link sense against travel for "
                          f"{ds['flipped']}/{ds['measured']} candidates "
                          f"({ds['flipped_frac']}) - reported, not penalised")
            for name in agg["arm_order"]:
                d = (agg["arms"].get(name) or {}).get("direction") or {}
                print(f"    {name}: per-frame median |d| "
                      f"{d.get('per_frame_median_deg')} | pooled p90 "
                      f"{d.get('p90_deg')} max {d.get('max_deg')} | "
                      f"within tol({d.get('tol_deg')}) {d.get('within_tol_frac')}")
        if agg.get("joint_gate"):
            jg = agg["joint_gate"]
            print(f"[map-assoc] JOINT GATE ({jg['rule']}): accepted "
                  f"{jg['accepted']} | dropped paint-confirmed "
                  f"{jg['dropped_paint_confirmed']} | dropped road-only "
                  f"{jg['dropped_road_only']} | dropped off-road "
                  f"{jg['dropped_off_road']} | of {jg['judged']} judged")
        if agg.get("paint_only_gate"):
            po = agg["paint_only_gate"]
            print(f"[map-assoc] PAINT-ONLY GATE ({po['rule']}): accepted "
                  f"{po['accepted']} | direction-conflicted "
                  f"{po['direction_conflicted']} | clean {po['clean']}")
        for row in (frames_out[0]["rows"] if frames_out else []):
            print(f"    {row['candidate_id']:6s} score={row['score']:.3f} "
                  f"hyp={row['hypothesis_id']} abstain={row['abstain']} "
                  f"conflicts={row['conflicts']}")
        print("[map-assoc] shadow only: no geometry is created, moved or "
              "authorised by this comparison")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False,
                           default=str), encoding="utf-8")
            print(f"[map-assoc] -> {args.json}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
