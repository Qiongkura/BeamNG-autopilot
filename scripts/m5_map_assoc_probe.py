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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamng_autopilot import config  # noqa: E402
from beamng_autopilot import geometry as G  # noqa: E402
from beamng_autopilot.connector import BeamNGConnector  # noqa: E402
from beamng_autopilot.lane.map_association import (  # noqa: E402
    MapLinkPrior,
    PerceptionCandidate,
    compare_arms,
)
from beamng_autopilot.runtime import build_camera_ring_provider  # noqa: E402
from beamng_autopilot.vision.hydra import FrameContext, HydraNet  # noqa: E402
from beamng_autopilot.vision.heads.semantic import SemanticHead  # noqa: E402


def _candidates_from_markings(markings, pos, heading) -> list[PerceptionCandidate]:
    """Perception candidates WITH identity (kind, side, bearing, span)."""
    out: list[PerceptionCandidate] = []
    p = np.asarray(pos, dtype=float)[:2]
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
        st = conn.get_state()
        pos = np.asarray(st.pos, dtype=float)
        heading = float(st.heading)
        direction = np.asarray(getattr(st, "dir", (np.cos(heading),
                                                   np.sin(heading), 0.0)),
                               dtype=float)
        prior = MapLinkPrior.from_road_rule(
            conn.read_current_road_rule(pos, direction))
        ring, mode = build_camera_ring_provider(conn, args.runtime, 320, 240)
        cands: list[PerceptionCandidate] = []
        if ring is not None:
            net = HydraNet()
            net.add(SemanticHead())
            snap = ring.grab_ring()
            role = "front_main" if "front_main" in snap else next(iter(snap))
            frame, cam = snap[role]
            ctx = FrameContext(frame_rgb=frame, cam=cam, pos=pos,
                               heading=heading,
                               ground_z=G.ego_ground_z(pos), role=role)
            out = net.run(ctx).get("semantic")
            if out is not None:
                cands = _candidates_from_markings(
                    out.meta.get("markings", []), pos, heading)
        rep = compare_arms(cands, prior)
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
        print(f"[map-assoc] link={prior.link_id} lanes={prior.lanes_hint} "
              f"dir={payload['map_link']['direction_deg']} "
              f"oneway={prior.one_way} rht={prior.right_hand_drive}")
        print(f"[map-assoc] perception candidates: {len(cands)}")
        for row in rep["arm_b"]:
            print(f"    {row['candidate_id']:6s} score={row['score']:.3f} "
                  f"hyp={row['hypothesis_id']} abstain={row['abstain']} "
                  f"conflicts={row['conflicts']}")
        print(f"[map-assoc] rank changed={rep['rank_changed']} "
              f"score changes={rep['n_changed_scores']} "
              f"(with conflict: {rep['n_changed_with_conflict']})")
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
