---
feature: yellow-semantic-head
status: designed
updated: 2026-09-14
branch: compose/line-recall
commits: daa8c73..HEAD
---

# Yellow Paint in Semantic Head LINE Channel

## Report

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
