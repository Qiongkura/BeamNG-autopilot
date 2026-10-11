"""Opt-in exact geometry at the production command boundary.

This is diagnostic evidence, never an extra lateral reference or a new gate.
Command receipts establish sending; vehicle execution remains unconfirmed.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np

from beamng_autopilot.experiments.metrics import sanitize_numbers
from beamng_autopilot.vehicle_body import (
    HALF_LENGTH_M, HALF_WIDTH_M, boundary_lateral,
    first_boundary_crossing_detail, footprint_corners,
)


def _points(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _raster(value):
    if value is None:
        return None
    arr = np.asarray(value)
    indices = np.flatnonzero(arr.ravel() != 0)
    return {"shape": list(arr.shape), "indices": indices.tolist(),
            "values": arr.ravel()[indices].astype(float).tolist()}


def capture_safety_assessment(scene, path, *, half_len, half_width,
                              max_cross_m, path_source, checks) -> dict:
    """Copy the inputs AT the safety check, before any control-path changes."""
    from .planning.constraints import _scene_boundaries

    left, right = _scene_boundaries(scene)
    return {"schema": "safety-assessment-geometry-v1",
            "coordinate_frame": "world_xy_metres",
            "pos": _points(scene.pos), "heading_rad": float(scene.heading),
            "lane_left": _points(left), "lane_right": _points(right),
            "path": _points(path), "path_source": str(path_source),
            "body_half_length_m": float(half_len),
            "body_half_width_m": float(half_width),
            "max_cross_m": float(max_cross_m), "checks": dict(checks)}


def capture_tick(out, *, pos, heading, selected_path, control_path,
                 command: dict, verdict) -> dict:
    """Copy geometry after the final command is sent; preserve missing evidence."""
    scene = getattr(out, "scene", None)
    grid = getattr(scene, "grid", None)
    snapshot = getattr(out, "snapshot", None)
    meta = getattr(out, "meta", {}) or {}
    payload = {
        "schema": "control-geometry-v1", "coordinate_frame": "world_xy_metres",
        "pos": _points(pos), "heading_rad": float(heading),
        "body_half_length_m": HALF_LENGTH_M, "body_half_width_m": HALF_WIDTH_M,
        "snapshot": snapshot.meta() if snapshot is not None else None,
        "lane_ref": _points(getattr(out, "lane_ref", None)),
        "scene_ref": _points(getattr(scene, "lane_ref", None)),
        "lane_left": _points(getattr(out, "lane_left", None)),
        "lane_right": _points(getattr(out, "lane_right", None)),
        "planner_path": _points(getattr(out, "best_path", None)),
        "selected_path": _points(selected_path), "control_path": _points(control_path),
        "candidates": getattr(out, "probe_candidates", None),
        "lane_source": meta.get("lane_src_sel"),
        "lane_drivable": meta.get("lane_drivable"),
        "lane_reject_reason": meta.get("lane_reject_reason"),
        "planner_rejects": meta.get("plan_rejects"),
        "head_sched": meta.get("head_sched"),
        "range_timing": meta.get("range_timing"),
        "range_age_basis": meta.get("range_age_basis"),
        "safety_assessment": getattr(verdict, "assessment_geometry", None),
        "safety_assessment_capture_ms": getattr(verdict, "assessment_geometry_ms", None),
        "command": dict(command, execution_status="unknown"),
        "verdict": {k: getattr(verdict, k, None) for k in
                    ("level", "reason", "body_cov_status", "body_cross_current",
                     "body_cross_planned", "first_crossing_distance_m",
                     "crossing_path_index", "crossing_boundary_side",
                     "rules_evaluated", "effective_rule")},
        "grid": (None if grid is None else {
            "origin": list(grid.origin), "heading_rad": float(grid.heading),
            "res_m": float(grid.res), "n_rows": int(grid.n_rows), "n_cols": int(grid.n_cols),
            **{key: _raster(getattr(grid, key, None)) for key in
               ("drivable", "observed", "obstacle", "occupancy")}}),
    }
    clean, nonfinite = sanitize_numbers(payload)
    clean["nonfinite_fields"] = nonfinite
    return clean


def replay_body(probe: dict) -> dict:
    """Recompute crossing witnesses from saved sensor geometry, game-free.

    A zero witness is not a feasibility or safety certificate. Current body
    coverage is explicit; swept tests keep their existing finite window.
    """
    if probe.get("schema") != "control-geometry-v1" or probe.get("nonfinite_fields"):
        raise ValueError("invalid/incomplete geometry record")
    left, right = probe.get("lane_left"), probe.get("lane_right")
    left = None if left is None else np.asarray(left, dtype=float)
    right = None if right is None else np.asarray(right, dtype=float)
    pos, heading = probe["pos"], probe["heading_rad"]
    corners = footprint_corners(pos, heading, probe["body_half_length_m"],
                               probe["body_half_width_m"])
    coverage, offsets = {}, {}
    for name, boundary in (("left", left), ("right", right)):
        values = ([boundary_lateral(*p, np.asarray(boundary), None) for p in corners]
                  if boundary is not None else [])
        coverage[name] = sum(bool(ok) for _, ok in values)
        offsets[name] = [value if covered else None for value, covered in values]
    current_crossing = (any(v is not None and v > 0.05 for v in offsets["left"])
                        or any(v is not None and v < -0.05 for v in offsets["right"]))
    current = (True if current_crossing else False
               if coverage == {"left": 4, "right": 4} else None)
    path = probe.get("control_path")
    distance, index, side = (first_boundary_crossing_detail(
        pos, path, left, right, half_len=probe["body_half_length_m"],
        half_width=probe["body_half_width_m"]) if path is not None else (0.0, -1, ""))
    return {"current_crossing": current, "corner_coverage": coverage,
            "corner_offsets_m": offsets, "planned_crossing_witness_m": distance,
            "planned_crossing_index": index, "planned_crossing_side": side,
            "safe_to_execute": None, "missing": "body replay alone does not establish feasibility"}


def replay_safety(probe: dict) -> dict:
    """Match safety predicates against THEIR saved inputs, separately from body truth.

    No detected crossing and missing boundary coverage can coexist. Matching
    the implementation is neither independent truth nor permission to drive.
    """
    from .planning.constraints import body_pose_crosses_lane, body_lane_cross_detail_m

    assessment = probe.get("safety_assessment")
    if (probe.get("nonfinite_fields") or not isinstance(assessment, dict)
            or assessment.get("schema") != "safety-assessment-geometry-v1"
            or assessment.get("error")):
        raise ValueError("safety assessment missing or invalid")
    scene = SimpleNamespace(pos=np.asarray(assessment["pos"], dtype=float),
                            heading=float(assessment["heading_rad"]),
                            lane_left=None if assessment["lane_left"] is None else
                            np.asarray(assessment["lane_left"], dtype=float),
                            lane_right=None if assessment["lane_right"] is None else
                            np.asarray(assessment["lane_right"], dtype=float))
    kwargs = {"half_len": assessment["body_half_length_m"],
              "half_width": assessment["body_half_width_m"],
              "max_cross_m": assessment["max_cross_m"]}
    now = body_pose_crosses_lane(scene, scene.pos, scene.heading, **kwargs)
    distance, index, side = body_lane_cross_detail_m(scene, assessment["path"], **kwargs)
    checks = assessment["checks"]
    result = {"body_cross_current": bool(now), "body_cross_planned": bool(distance > 0),
              "first_crossing_distance_m": float(distance) if distance > 0 else None,
              "crossing_path_index": int(index) if distance > 0 else None,
              "crossing_boundary_side": str(side) if distance > 0 else ""}
    mismatch = []
    for key, actual in result.items():
        if key not in checks:
            mismatch.append(key)
        elif key == "first_crossing_distance_m" and actual is not None:
            expected = checks[key]
            if expected is None or abs(float(expected) - actual) > 1e-8:
                mismatch.append(key)
        elif checks[key] != actual:
            mismatch.append(key)
        if (probe.get("verdict") or {}).get(key) != checks.get(key):
            mismatch.append("verdict." + key)
    # Reuse the physical witness coverage rule with these EXACT dimensions.
    witness = replay_body(dict(probe, lane_left=assessment["lane_left"],
                               lane_right=assessment["lane_right"],
                               pos=assessment["pos"], heading_rad=assessment["heading_rad"],
                               body_half_length_m=kwargs["half_len"],
                               body_half_width_m=kwargs["half_width"],
                               control_path=assessment["path"]))
    return {**result, "predicate_matches": not mismatch, "mismatched_fields": mismatch,
            "current_evidence": witness["current_crossing"],
            "corner_coverage": witness["corner_coverage"],
            "path_source": assessment["path_source"], "safe_to_execute": None}


def analyze_frames(frames: list[dict]) -> dict:
    """Audit saved command joins and refreshed source identities without a game."""
    counts = Counter()
    reasons = Counter()
    failures = []
    forced, consumed_new, pending = [], [], {}
    previous_source = {}
    for index, frame in enumerate(frames):
        reasons[str(frame.get("reason", "UNKNOWN"))] += 1
        probe = frame.get("geometry_probe")
        if not isinstance(probe, dict) or probe.get("error"):
            counts["geometry_unknown"] += 1
            counts["safety_assessment_unknown"] += 1
        else:
            try:
                body = replay_body(probe)
                counts["geometry_replayed"] += 1
                counts["body_coverage_unknown"] += body["current_crossing"] is None
                counts["current_crossing_witnesses"] += body["current_crossing"] is True
                counts["planned_crossing_witnesses"] += body["planned_crossing_witness_m"] > 0
                command = probe.get("command") or {}
                matches = (command.get("cmd_seq") == frame.get("cmd_seq")
                           and command.get("cmd_t") == frame.get("cmd_t"))
                for wire, column in (("throttle", "throttle"), ("brake", "brake"), ("steering", "steer")):
                    a, b = command.get(wire), frame.get(column)
                    matches = matches and a is not None and b is not None and abs(float(a) - float(b)) <= 0.000051
                counts["command_joins_matched" if matches else "command_joins_failed"] += 1
                if not matches:
                    failures.append({"frame": index, "reason": "command/probe mismatch"})
                if not probe.get("safety_assessment"):
                    counts["safety_assessment_unknown"] += 1
                else:
                    try:
                        safety = replay_safety(probe)
                        counts["safety_assessment_replayed"] += 1
                        counts["safety_coverage_unknown"] += safety["current_evidence"] is None
                        counts["safety_matches" if safety["predicate_matches"] else "safety_mismatches"] += 1
                        if not safety["predicate_matches"]:
                            failures.append({"frame": index, "reason": "safety predicate mismatch",
                                             "fields": safety["mismatched_fields"]})
                    except (ValueError, TypeError, KeyError, IndexError) as exc:
                        counts["safety_assessment_unknown"] += 1
                        failures.append({"frame": index, "reason": str(exc)})
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                counts["geometry_unknown"] += 1
                counts["safety_assessment_unknown"] += 1
                failures.append({"frame": index, "reason": str(exc)})
        for name, record in (frame.get("head_sched") or {}).items():
            if not isinstance(record, dict):
                continue
            if record.get("state") == "keepalive_forced":
                key = (name, record.get("source_seq"), record.get("eligible_t"))
                if key not in pending:
                    request = {"head": name, "frame": index, "source_seq": record.get("source_seq"),
                               "previous_consumed_source": previous_source.get(name)}
                    forced.append(request)
                    pending[key] = request
        for name, receipt in (frame.get("consumed") or {}).items():
            published = (frame.get("head_sched") or {}).get(name) or {}
            source = receipt.get("source_seq")
            if source is None:
                continue
            for key, request in list(pending.items()):
                prior = request["previous_consumed_source"]
                target = request["source_seq"]
                # A first observation has no previous source to compare. It
                # remains UNKNOWN instead of proving a forced replacement.
                if (key[0] == name and prior is not None and target is not None
                        and source > prior and source >= target
                        and receipt.get("result_seq") == published.get("result_seq")
                        and source == published.get("source_seq")
                        and receipt.get("publish_age_s") is not None
                        and frame.get("cmd_seq") is not None):
                    consumed_new.append({**request, "consumed_frame": index,
                                         "cmd_seq": frame["cmd_seq"], "new_source_seq": source,
                                         "result_seq": receipt.get("result_seq")})
                    del pending[key]
            previous_source[name] = source
    fields = ("geometry_replayed", "geometry_unknown", "body_coverage_unknown",
              "current_crossing_witnesses", "planned_crossing_witnesses",
              "command_joins_matched", "command_joins_failed",
              "safety_assessment_replayed", "safety_assessment_unknown",
              "safety_coverage_unknown", "safety_matches", "safety_mismatches")
    return {"schema": "geometry-audit-v1", "frames": len(frames),
            **{field: counts[field] for field in fields},
            "reasons_histogram": dict(reasons), "failures": failures,
            "forced_refresh_requests": forced, "forced_new_source_command_joins": consumed_new,
            "forced_unconfirmed": list(pending.values()),
            "actuator_execution": "unknown", "road_gate_truth_positive_negative": "unknown",
            "missing": "No independent road truth or actuator acknowledgement in these records"}
