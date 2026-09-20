# 2026-09-20 strict mode rides the road centre line

## Symptom

`scripts/m5_fsd_benchmark.py --attach --runtime tech --scenarios town
--strict --goal 868.3 744.9` drove the car onto the road centre line
(the divider between the two directions of travel) and stalled there.

`logs/fsd_benchmark/gate_on_3.log` (same session, prior arm):

* `painted-line lateral: mean=+1.18 m, p50=+1.04 m, min=-1.71, max=+5.03`
  - the car sat ~1 m to the RIGHT of the painted centre line for most
  of the run.  ``min=-1.71`` means it briefly slipped onto the
  *oncoming* side of the divider.
* `done: 166 frames, 75 stops` (45 % of frames the car was stationary)
* `town FAIL frames=166 lane=93% rev=0 crossC=0 crossR=0 off=0
  stall=127 dist=42.7m goal=48.54m FAILED: no_centre_crossing,no_edge_
  crossing,no_stall,reached_goal`

The user-visible frame is the screenshot pasted in chat at 21:16
(``clipboard-2026-09-20T13-16-08-265Z-187a7949.jpg``): the red hatchback
straddles the white divider at 0 mph.

## Root cause

`select_lane_reference` (``beamng_autopilot/lane/reference.py``) has
three sources in priority order:

1. **Sensor lane** - a paired perception read of the ego lane.
2. **Map-prior own lane** - a synthetic centre half a lane width
   RIGHT of the road centreline (right-hand traffic).
3. **BEV corridor fallback** - the whole-road free-space centre.

Two gates guard the sensor source: ``heading`` (the lane must HEAD the
same way as the route, rejecting junction-pair mislocks) and ``side``
(the lane centre must sit RIGHT of the route, rejecting oncoming
corridor reads).

The town scenario runs ``lane_mode="sensor" + strict=True``, which
sets ``strict_lane = True`` and triggers two
"permission" suppressions:

```python
# line 343-344 (pre-fix)
strict_lane = bool(lane_mode == "sensor" and strict_sensor)
map_lane = None if strict_lane else map_lane_override
```

```python
# line 453 (pre-fix)
elif not strict_lane and not lane_side_ok(
        lane_ref, route_ref, pos,
        left_max_m=(-0.4 if lane_mode == "map" else 0.5)):
    lane_rejected = True
```

The intent of ``strict_lane`` was to say "the perception lane leads,
the map lane may never become the navigator's reference".  The
implementation conflated that with "the SIDE gate may be bypassed".
Both suppressions switched on at the same flag, so in
``strict+sensor`` mode the SIDE gate was bypassed and an unpaired
sensor centre (typically the BEV whole-road corridor centre, which on
a two-way road IS the road centreline) flowed straight into the
planner as ``lane_ref``.

The existing comments in the file already documented the failure mode
from earlier runs:

> town runs 2026-08-22: the car rode the centre/oncoming lane end to
> end with lane_src=sensor

but the guard that should have stopped it was the very one being
bypassed.

A second path also rides the centre line when the perception fails
completely: ``arbiter.py:386`` falls back to ``route`` when
``lane_ref`` is None and ``strict_perception`` is False.  But in strict
mode ``choose_plan_route`` returns None (line 387) and
``constraints.py:175`` returns a 1e9 penalty (fail-closed).  The
"perception gone -> stop" branch was already wired correctly; only the
"perception gives a bad centre" branch was broken.

## Fix

`beamng_autopilot/lane/reference.py`:

```python
# SIDE gate now ALWAYS on - the last wall between an unpaired sensor
# centre and the planner.  Strict perception tightens the threshold
# to -0.2 m so off=0 (lane sitting on the road centre line) is
# rejected, not just off>0 (oncoming side).
elif not lane_side_ok(
        lane_ref, route_ref, pos,
        left_max_m=(-0.2 if strict_lane
                    else (-0.4 if lane_mode == "map" else 0.5))):
    lane_rejected = True
    side_bad = True
```

Why -0.2 and not 0.0: ``lane_side_ok`` returns ``off <= left_max_m``
and ``off`` is signed (negative = right/own side, positive =
left/oncoming side).  ``left_max_m=0.0`` still lets ``off=0`` through,
which is exactly the centreline case.  -0.2 forces the lane to sit at
least 0.2 m on the own side of the route.

Other modes keep their legacy tolerance:

* ``map`` : -0.4 (allow up to 0.4 m oncoming, corner apex)
* ``auto``/``sensor`` non-strict : 0.5 (loose perception-led)

## Tests

`tests/test_lane_reference.py` gains four regression tests:

* `test_strict_rejects_lane_sitting_on_the_route_centerline` -
  strict + centre at y=0 -> SRC_UNAVAILABLE, reason="side"
* `test_strict_rejects_lane_inside_the_centreline_band` -
  strict + centre at y=-0.1 -> rejected
* `test_non_strict_sensor_lane_on_centreline_is_still_published` -
  the relaxed tolerance for perception-led modes is preserved
* `test_map_mode_keeps_its_legacy_minus_0p4_tolerance` -
  the legacy map-mode band is unchanged

59 tests pass in ``test_lane_reference.py`` + ``test_run_manifest.py``
combined.

One unrelated pre-existing failure remains:
``test_fsd_closed_loop_recovers_from_a_body_crossing`` - the safety
monitor's body-crossing recovery test has been asserting on a single
apostrophe/character equivalence for a while; it does not touch
``lane/reference.py``.

## What the gate fix did NOT do: the envelope smuggled the lane back in

The gate change on its own was cosmetic.  ``lateral_reference``
(``planning/lateral_ref.py``) fell through to ``lane_envelope.center``
BEFORE the strict check, so the lane the gate had just refused came
straight back as the steering reference under a different label.

Measured on the offline closed-loop fixture, before vs after the gate
fix:

| | before | after |
| --- | --- | --- |
| ``lane_src`` | sensor | perception-unavailable |
| ``lateral_reference`` src | sensor | envelope |
| ref median y | **0.060** | **0.060** |
| ``best_path`` median y | **0.060** | **0.060** |

Same geometry, different tag - the car still drove the road centre
line.  The envelope centre is the whole-road corridor centre, which on
a two-way road IS the centre line: exactly what the gate exists to
refuse.

Fixed by moving the strict check ahead of the envelope fallback.  The
envelope still supplies hard BOUNDARIES; only its centre stopped being
a steering reference under strict perception.

## The deeper cause, still open: perception puts the centre on the centre line

The closed-loop fixture feeds markings at road lat 0 (yellow centre
line) and -3.5 (white right edge), so the lane centre is -1.75.  The
pairing derives the right answer in the car frame - ``center0 = -0.58``,
which is -1.77 in world coordinates - but ``lane_envelope.center``
comes out at world y = 0.060, i.e. the road centre line.

So perception is producing a lane centre that sits on the divider.
Reference-layer gates can refuse it; they cannot make it correct.
This is the actual source of the centre-line ride and is not fixed.

## Live verification: none

No live run completed after the fix.  The one attempt (started 21:29)
produced no scorecard because ``BeamNG.tech.x64.exe`` died mid-run and
the controller then hung at 0% CPU with its socket in ``SYN_SENT``,
waiting on a reply that never came.  The user called a stop to live
runs after that.

The two fixes are therefore verified by unit tests only - 59 passing
across ``test_lane_reference.py`` and ``test_run_manifest.py`` - plus
the offline closed-loop fixture, not on the car.

## Known consequence

Strict scenes whose perception cannot supply a trustworthy lane now
fail closed (no path, car stops) instead of driving the corridor
centre.  That is the intended contract, but it currently fails
``test_fsd_closed_loop_recovers_from_a_body_crossing``: that fixture
expects the car to creep back out of a body crossing, and with the
perception defect above there is no trustworthy lane to recover
along.  Fixing the perception centre is what makes both true at once.

## Aside: the town goal sits off the lane

Measured geometry at start node 22209: road centreline at Y=740.7,
right road edge at Y=744.9, half-width 4.2 m (LiDAR reads 4.15-4.3).
Lane width is 3.6 m, so there is 0.6 m of paved shoulder beyond the
lane.  Lane centre is Y=742.5 and the lane's right boundary Y=744.3,
but the town goal is ``(868.3, 744.9)`` - 0.6 m outside the lane, on
the shoulder, 2.4 m right of the lane centre.  Chasing that goal
pushes the car off the lane independently of everything above.
a separate design discussion, not part of this fix.