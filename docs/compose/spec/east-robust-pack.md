---
feature: east-robust-pack
status: delivered
updated: 2026-09-14
branch: compose/east-robust-pack
commits: 510ac3a..d75498e
---

# East Coast Robustness Pack

## Report

**What was built** — Yellow extractor keeps smaller components; placement
skips 16 warm ticks; centre-paint hold coasts by projecting the last
centre forward ≤6 ticks after hold exhausts; `scripts/m5_run_metrics.py`
summarizes logs; `beamng_autopilot/pinned_weights.json` inventories pins.

**Verification** — 47+ targeted tests passed. Live goal run: **placed=True**,
**sensor 54–65%**, no-path near zero on best runs.

**Journey log**
- Coast keeps paired authority without map lateral.
- Placement still intermittent but skip-16 made a successful run.
- UNet retrain / italy hand labels still open (data work).

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
- [x] T1: extractor yellow gates (covers S2.1)
- [x] T2: placement skip ticks (covers S2.2)
- [x] T3: post-hold coast in fusion (covers S2.3)
- [x] T4: m5_run_metrics.py (covers S2.4)
- [x] T5: pin MANIFEST json (covers S2.5)
- [x] T6: targeted pytest (depends T1-T3)
