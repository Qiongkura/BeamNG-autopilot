---
feature: yellow-mask-live
status: designed
updated: 2026-09-14
branch: compose/yellow-lane-centre
commits: 4d3af36..HEAD
---

# Live Yellow-Paint Fusion for Painted-Line Geometry

## Report

## [S1] Problem

east_coast uses yellow centre paint. Placement still reports
`painted line not perceived` / `placed=False`. `vision/yellow_line_mask.py`
exists for pseudo-labels but is **never** unioned into the live LINE mask
that `painted_line_markings` / `painted_line_lane_center` consume — the
UNet (trained mostly on italy white lines) misses yellow, so geometry has
nothing to place onto even when HSV yellow is visible.

## [S2] Design

- `painted_line_markings(..., rgb=None)`: when `rgb` is provided, union
  `yellow_line_mask(rgb)` into the semantic LINE mask before
  back-projection.
- `painted_line_lane_center` / `painted_line_direction` forward `rgb`.
- Call sites in `fsd_drive` (`_prewarm_and_place`, `_painted_line_lat`)
  pass `rgb=out.frame`.
- Iron rule: still perception-only; yellow mask is a sensor colour prior,
  not a map offset.

## [S3] Out of Scope

- Training the UNet on yellow paint (data work).
- Planner no-path / graze gates.

## Tasks

- [ ] T1: rgb union in painted-line helpers + fsd_drive call sites
      — acceptance: unit test that yellow-only frame yields markings
      (covers: S2)
- [ ] T2: existing painted-line tests still pass (covers: S2; depends: T1)
