---
feature: yellow-semantic-head
status: delivered
updated: 2026-09-14
branch: compose/line-recall
commits: daa8c73..4215dbf
---

# Yellow Paint in Semantic Head LINE Channel

## Report

**What was built** — SemanticHead unions HSV yellow into LINE before
evidence/markings. Line-evidence persistence raised (DECAY 0.25→0.15,
MAX_AGE 3→5 s) so sparse yellow survives dropouts. Live: placement
succeeded in 6.2 s; line_lat min improved toward 0.

**Verification** — 21 vision tests passed. Live 90s at junction.

**Journey log**
- Yellow only in painted_line_markings was not enough for pairing.
- Mid-run `no drivable path` remains when pairing fully expires (strict).

## [S1] Problem

`yellow_line_mask` only fused into `painted_line_markings` for placement.
The semantic LINE channel that drives **pairing + line evidence** still
comes from the UNet alone (white-line biased). US yellow centre paint is
often missing → pairing flickers → strict sensor stops.

## [S2] Design

After `Segmenter.predict` in `SemanticHead.run`, union
`yellow_line_mask(frame_rgb)` into `line` **before** evidence fusion and
`detect_lines`. Record `line_pixels_yellow` in meta. Failures of the
yellow helper are isolated (no crash).

## [S3] Out of Scope

- UNet retrain on yellow.
- Planner.

## Tasks

- [ ] T1: Yellow union in SemanticHead — acceptance: unit test with fake
      segmenter + yellow frame increases line mask (covers: S2)
- [ ] T2: Head / painted-line tests green (covers: S2; depends: T1)
