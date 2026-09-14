---
feature: plc-low-speed
status: delivered
updated: 2026-09-14
branch: compose/plc-low-speed
commits: 72452dd..0e5d916
---

# PLC Low-Speed Engagement

## Report

**What was built** — `PaintedLineLateralCorrector` now has three speed bands:
park (<0.05 m/s) freezes; crawl (0.05–0.5) integrates at `rate * 0.25`;
cruise (≥0.5) unchanged. A creeping ego can pull toward the painted-line
own-lane centre instead of pressing the line under the old `min_speed=0.5`
freeze. Policy gate (`painted_line_correction_active`) and hold/decay
behaviour are unchanged.

**Verification** —
`pytest tests/test_painted_line_corrector.py tests/test_painted_line_lane_center.py -q`
→ **32 passed**. Reviewer: no critical; defaults picked up at `fsd_drive`
call site without edit.

**Journey log**
- Old park test used speed=0.1, which is crawl after the band split — freeze tests must use true standstill (0.0).
- S0.4 p50 0.33 m/s explained PLC 59/216 engagement; this is the crawl-band fix.
- Pairing flicker / planner `grazes` still separate; re-run live drive after merge.
## [S1] Problem

On the S0.4 east_coast judgment run the car crawled with speed p50 ≈ 0.33 m/s
while `line_lat` reached +2.7 m and painted-line PLC was active on only
59/216 frames. `PaintedLineLateralCorrector.update` **returns early** when
`speed < min_speed_mps` (default 0.5), so crawl-speed frames never pull the
near path toward the perceived own-lane centre — the car presses / straddles
the painted line even when the LINE mask is live.

The freeze was meant to stop a **standstill** from accumulating a launch-y
offset, not to disable centring while the ego is already creeping.

## [S2] Design

### [S2.1] Park vs crawl

In `PaintedLineLateralCorrector`:

| Band | Condition | Behaviour |
| --- | --- | --- |
| Park | `speed < park_speed_mps` (default **0.05**) | Freeze `shift_m` (legacy intent) |
| Crawl | `park_speed_mps ≤ speed < min_speed_mps` | Integrate toward `_desired` at **`rate_m_s * crawl_rate_scale`** (default **0.25**), still clipped to `max_shift_m` |
| Cruise | `speed ≥ min_speed_mps` | Unchanged full `rate_m_s` |

- New ctor args: `park_speed_mps: float = 0.05`, `crawl_rate_scale: float = 0.25`.
- `min_speed_mps` keeps its meaning as the **full-rate** threshold (still 0.5).
- Hold/decay on missing `desired` unchanged.
- `painted_line_correction_active` policy unchanged: PLC still off when
  `lane_src_sel == "sensor"` (sensor lane owns lateral) or rule fallback.

### [S2.2] Contracts

- Park still never integrates (regression test with speed=0.0).
- Crawl with a positive desired **does** move `shift_m` (new test).
- Crawl step ≤ `rate_m_s * crawl_rate_scale * dt` (tighter than cruise).
- No map/nav centreline + fixed offset is introduced (iron rule).

### [S2.3] Out of Scope

- Pairing flicker / `lane_src` availability.
- Planner `no drivable path` / `grazes obstacle`.
- Live re-run of S0.4 (needs another game session after merge).

## Tasks

- [x] T1: Implement park/crawl bands in `PaintedLineLateralCorrector.update`
      — acceptance: park freezes; crawl integrates at scaled rate
      (covers: S2.1)
- [x] T2: Update/extend `tests/test_painted_line_corrector.py` —
      acceptance: park test uses true park speed; new crawl test fails
      before T1 and passes after (covers: S2.1; depends: T1)
- [x] T3: Run painted-line + nearby lane tests — acceptance: green
      (covers: S2.2; depends: T1, T2)
