---
feature: s0-experiment-trust
status: delivered
updated: 2026-09-13
branch: compose/s0-experiment-trust
commits: 5f99b58..3df2255
---

# S0 Experiment Trust (provenance + spawn gate)

## Report

**What was built** — Shadow episodes no longer hardcode `map=italy`: `fsd_drive`
resolves live level via `BeamNGConnector.current_env()` (`source=live`) and
falls back to `unknown` when attach has neither a verified level nor `--map`.
`data_contract` reads episode `meta.provenance` instead of inventing italy.
Photo tour scores every camera stop with `vision/spawn_gate.assess_spawn_frame`
(green / dark / road-like) and writes `ok`/`reasons`/metrics into `stops.json`;
`--no-require-ok` keeps all frames but default requires at least one legal stop.

**Verification** —
`pytest tests/test_s0_experiment_trust.py tests/test_data_contract.py tests/test_fsd_drive_helpers.py -q`
→ **55 passed**. Reviewer found critical attach-default-italy gap; fixed in
`3df2255` with live-only trust + three resolve tests.

**Journey log**
- Worktree create blocked by env; used `compose/s0-experiment-trust` on main worktree.
- Reviewer: `conn.map_name` is launch snapshot — attach without `--map` still italy.
- Fix: `current_env()["source"]`; provenance `unknown` unless live or explicit `--map`.
- `m5_shadow_drive` still hardcodes italy (follow-up, out of this feature).
- Spawn gate is heuristic (HSV); human photo review still required before live runs.

## [S1] Problem

Multi-map FSD experiments currently cannot trust their own artifacts:

1. **Shadow provenance lies about the map.** `fsd_drive` hardcodes
   `"map": "italy"` into every episode. `data_contract.make_npz_record`
   also hardcodes `"map": "italy"` for shadow/telemetry manifests. On
   east_coast (or any non-italy map) this mislabels every episode, breaks
   by_map routing audits, and made the 2026-09-13 bush-spawn episode look
   like italy data.
2. **Roadnet nodes are not legal camera/spawn points.** The east_coast
   judgment run teleported into vegetation. Photo tour still accepts any
   roadnode and writes it to `stops.json` with no visual legality check,
   so the next experiment can repeat the same failure.

## [S2] Design

### [S2.1] True map in provenance

- Call site: `fsd_drive` ShadowRecorder provenance uses
  `conn.map_name` (fallback `getattr(args, "map", None)` then `"unknown"`).
  Vehicle uses `conn.vehicle_model` when available.
- `data_contract.make_npz_record` reads episode `meta` JSON when present:
  `provenance.map` / `provenance.vehicle` / `provenance.runtime`.
  Missing fields stay absent or `"unknown"` — **do not invent `"italy"`**.
- `make_seg_record` keeps `meta.get("map", ...)` but the fallback becomes
  `"unknown"` instead of `"italy"` so missing metadata is visible.
- Seg runs that only exist as italy truth may still set `"italy"` in their
  own `meta.json`; the contract only stops guessing.

### [S2.2] Spawn / camera legality gate

New pure module `beamng_autopilot/vision/spawn_gate.py`:

```python
@dataclass
class SpawnAssessment:
    ok: bool
    reasons: list[str]
    green_frac: float
    dark_frac: float
    roadlike_frac: float
    mean_v: float

def assess_spawn_frame(rgb: np.ndarray) -> SpawnAssessment
```

Rules (offline, no game, no model):

| Metric | Definition | Reject when |
| --- | --- | --- |
| `green_frac` | `green_vegetation_mask` pixels / ROI pixels | `> 0.45` |
| `mean_v` | HSV V mean over ROI | `< 40` (too dark) |
| `roadlike_frac` | low-sat mid-V, non-green pixels in lower 2/3 | `< 0.08` |

ROI is lower 70% of the frame (road dominates; sky/tree tops are not the
criterion). Reasons are stable strings: `"too_green"`, `"too_dark"`,
`"no_road_like"`.

`scripts/m5_map_photo_tour.py` after each capture:

- `assess = assess_spawn_frame(rgb)`
- always write the stop entry with `ok`, `reasons`, and metric fields
- keep the PNG (dataset of bad stops is useful)
- `--require-ok` (default **true**): count only `ok` frames toward `n`
  printed as accepted; rejected stops still listed in `stops.json`
  (callers can filter). Without `--require-ok`, behavior matches old
  acceptance (all frames) but still records assessments.

No change to FSD drive loop in this feature (live gate can consume
`stops.json` later).

### [S2.3] Contracts

- Environment map in manifests must equal the connector/episode map or
  `"unknown"`.
- `stops.json` entries must include `ok` and `reasons` after this change.
- Iron rule unchanged: no navigation-line lateral offset added.

## [S3] Out of Scope

- Live spawn rejection inside `fsd_drive` placement.
- Retraining models / changing lane pairing.
- P6 `weights/pinned` MANIFEST.
- italy P1 paired-rate work.
- Deleting historical mislabeled episodes (they stay; manifests will
  show `"unknown"` or true map once re-scanned).

## Tasks

- [x] T1: Shadow provenance uses real map/vehicle — acceptance: unit test
      that `fsd_drive` helper (or extracted provenance builder) returns
      connector map, not `"italy"`; `make_npz_record` reads meta.provenance.map
      (covers: S2.1)
- [x] T2: `spawn_gate.assess_spawn_frame` + synthetic-frame tests —
      acceptance: green-dominated frame rejects `too_green`; dark frame
      rejects `too_dark`; bright low-veg road-like frame accepts
      (covers: S2.2)
- [x] T3: photo tour records assessments in `stops.json` — acceptance:
      tour code path writes `ok`/`reasons`/metrics; `--require-ok` flag
      exists; dry unit test of the log-builder function if extracted
      (covers: S2.2; depends: T2)
- [x] T4: targeted pytest for touched modules — acceptance: new tests +
      existing `test_data_contract` still pass
      (covers: S2.1, S2.2; depends: T1, T2, T3)
