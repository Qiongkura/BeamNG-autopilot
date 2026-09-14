---
feature: pair-hold-centre
status: designed
updated: 2026-09-14
branch: compose/pair-hold
commits: 7671b60..HEAD
---

# Hold Centre-Paint Lanes Across Detection Gaps

## Report

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

- [ ] T1: Extended hold for centre-paint frames — acceptance: unit test
      that a paired left-only frame survives >4 miss ticks (covers: S2)
- [ ] T2: Default hold unchanged for other frames — acceptance: existing
      fusion tests pass (covers: S2; depends: T1)
