---
feature: east-robust-pack
status: designed
updated: 2026-09-14
branch: compose/east-robust-pack
commits: 510ac3a..HEAD
---

# East Coast Robustness Pack

## Report

## [S1] Problem

Live east_coast still: (a) extractor drops paint that is in the mask,
(b) placement fails on first warm frames after teleport, (c) hold exhausts
then strict sensor parks, (d) no automated run metrics, (e) pin weights
are not inventoried.

## [S2] Design

### [S2.1] Extractor recall
- Yellow components: `min_area=12`, `min_height=8`.
- White defaults unchanged.

### [S2.2] Placement skip-frames
- `_prewarm_and_place`: skip placement attempts until `PLACEMENT_SKIP_TICKS`
  (default 8) stack ticks after teleport so the camera settles.

### [S2.3] Post-hold coast
- When centre-paint hold exhausts, do **not** clear to None immediately;
  extrapolate last centre forward by ego speed for ≤2 s of ticks
  (`state["coast"]`), still perception-derived (no map).

### [S2.4] Metrics script
- `scripts/m5_run_metrics.py`: parse a drive log / shadow npz →
  sensor %, nop %, placed, line_lat summary.

### [S2.5] Pin MANIFEST (P6)
- `weights/pinned/MANIFEST.json` listing current deployed checkpoints
  paths + optional sha256 if files exist (no binary in git).

## [S3] Out of Scope
- Full UNet retrain.
- italy hand labeling.

## Tasks
- [ ] T1: extractor yellow gates (covers S2.1)
- [ ] T2: placement skip ticks (covers S2.2)
- [ ] T3: post-hold coast in fusion (covers S2.3)
- [ ] T4: m5_run_metrics.py (covers S2.4)
- [ ] T5: weights/pinned/MANIFEST.json (covers S2.5)
- [ ] T6: targeted pytest (depends T1-T3)
