---
feature: pair-hold-centre
status: delivered
updated: 2026-09-14
branch: compose/pair-hold
commits: 7671b60..7e1111e
---

# Hold Centre-Paint Lanes Across Detection Gaps

## Report

**What was built** — When fusion has no detection, a last **paired
vision centre-paint** frame (left only) is held up to **12** miss ticks
(default 4 for other sources), so strict sensor keeps `lane_src=sensor`
across typical east_coast gaps.

**Verification** — `pytest tests/test_lane_fusion.py -q` → **6 passed**.
Live 90s at junction: longer early `sensor/P` run; later still planning
stops (`no drivable path` / graze) — not a hold bug.

**Journey log**
- spawn_gate “ok” long roads still fail strict sensor if pairing dies.
- Global `BEAMNG_LANE_HOLD_NONE_FRAMES` still available for A/B.

## [S1] Problem

Strict sensor mode stops when `lane_src` drops. east_coast runs show
`sensor/P` then `perception-unavailable` gaps of 5–17 frames even after
centre-paint pairing (`_centre_line_own_lane`). Default
`LANE_FUSION_HOLD_NONE_FRAMES=4` clears the hold mid-gap, so the runtime
sees no lane and fails closed. Env override exists but is not set in
production runs.

## [S2] Design

In `choose_sensor_lane` when `chosen is None`:

- If `state["last"]` is a **paired vision centre-paint** frame
  (`paired`, `left` present, `right` is None — our US yellow path),
  extend the hold window to **max(default, 12)** frames.
- Other miss reasons keep the existing `LANE_FUSION_HOLD_NONE_FRAMES`.
- Do not invent map lateral geometry; holding the last perception
  centre only.

## [S3] Out of Scope

- Planner no-path / BEV.
- Raising the global default for all sources.

## Tasks

- [x] T1: Extended hold for centre-paint frames — acceptance: unit test
      that a paired left-only frame survives >4 miss ticks (covers: S2)
- [x] T2: Default hold unchanged for other frames — acceptance: existing
      fusion tests pass (covers: S2; depends: T1)
