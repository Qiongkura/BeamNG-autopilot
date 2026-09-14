---
feature: yellow-lane-centre
status: delivered
updated: 2026-09-14
branch: compose/yellow-lane-centre
commits: cc0cc63..99755d7
---

# Double-Paint Own-Lane Centre (US yellow / double solid)

## Report

**What was built** — When opposite-side paint clusters hug the ego with
separation ≤1.2 m (double yellow / double solid), `painted_line_lane_center`
now uses the **midpoint** as the painted centre and places the ego
`lane_half_m` to the **right** (RHT). The old road-size straddle gate
(sep≥2.5 m) rejected this and left the car sitting on the paint at
east_coast junctions.

**Verification** — `pytest tests/test_painted_line_lane_center.py` → 18
passed (incl. new double-yellow test); reviewer ran full `tests/` →
**740 passed**. No critical.

**Journey log**
- Double-paint gate must run **before** the road-size straddle special case.
- Acceptance counts: file 18 / suite 740 (not the earlier 33 combined run).
- Three-cluster (double + left edge) still uses nearest-paint road-size path — out of scope.

## [S1] Problem

Live screenshot + S0.4 metrics: the ego parks **on the double yellow centre
lines** at an east_coast junction. `painted_line_lane_center` has a straddle
guard: if clusters exist on both sides of the ego it only accepts a
road-sized pair (sep 2.5–5 m). Double yellow paints sit ~0.3–0.6 m apart
around the ego → the guard **returns None**, no centre, no pull — the car
stays on the paint.

## [S2] Design

### [S2.1] Double-paint midpoint

When `has_left and has_right`:

1. **Double-paint** (new): both cluster means within ~1.2 m of ego and
   separation in **[0.15, 1.2] m** → treat
   `line_lat = 0.5 * (lat_near + lat_far)` as the painted centre, then
   `shift = clip(lane_half_m - line_lat, …)` (own lane to the **right**,
   same RHT rule as single centre line). Do **not` require sep≥2.5.
2. Keep the existing **road-size straddle** spawn case (sep 2.5–5).
3. Otherwise still `return None`.

Single-cluster near ego (merged double pixels) is unchanged: already
places `lane_half_m` to the right of that cluster.

### [S2.2] Out of Scope

- Planner `no drivable path` / `grazes obstacle`.
- Pairing flicker (`lane_src`).
- Junction geometry beyond lateral centre of the visible paint.

## Tasks

- [x] T1: Double-paint branch in `painted_line_lane_center` —
      acceptance: synthetic double-line clusters around ego yield target
      `lane_half_m` right of midpoint (covers: S2.1)
- [x] T2: Keep road-size straddle + single-line tests green —
      acceptance: existing painted-line tests pass (covers: S2.1)
- [x] T3: Targeted pytest — acceptance: green (depends: T1, T2)
