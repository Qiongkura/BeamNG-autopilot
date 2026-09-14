---
feature: s0-shadow-hygiene
status: in-progress
updated: 2026-09-13
branch: compose/s0-shadow-hygiene
commits: a29a1ae..HEAD
---

# S0 Shadow Provenance Hygiene

## Report

## [S1] Problem

Review of `s0-experiment-trust` left two S0 holes:

1. **`m5_shadow_drive` still hardcodes `"map": "italy"`** into every episode
   (and constructs `BeamNGConnector("italy", …)` even when attaching to
   another level). Multi-map shadow collection therefore repeats the
   mislabeling bug fixed in `fsd_drive`.
2. **Bad bush episode is still a normal training candidate.**
   `shadow_fsd_1789298914_20260913_193100.npz` is vegetation-only; nothing
   in `data_contract.scan_project` / manifests marks it excluded, so a
   naive E2E dataset build will ingest it as a positive US episode.

## [S2] Design

### [S2.1] Shared live map resolution for shadow drive

- Reuse `fsd_drive.resolve_provenance_env` (already live-only trust).
- `m5_shadow_drive` gains `--map` / optional launch map; connector is
  constructed with `args.map or "italy"` only as **launch default**.
- After `conn.open` / attach / load, resolve provenance and **mutate**
  `rec.provenance["map"]` / `["vehicle"]` before recording frames
  (`ShadowRecorder` merges provenance at save time).
- If resolve returns `unknown` (attach + failed level query + no `--map`),
  record `unknown` — never invent italy.

### [S2.2] Episode exclusion list

- New repo file `data/excluded_shadow_episodes.json`:
  `{"version": 1, "episodes": {"<filename>": {"reason": "...", "added": "YYYY-MM-DD"}}}`
- Seed with the bush episode name + reason `camera_in_vegetation_20260913`.
- `data_contract.is_excluded_episode(path) -> bool` and
  `exclusion_reason(path) -> str | None`.
- `make_npz_record` sets `"split": "excluded"` and `"label.source":
  "excluded"` when the basename is listed; `scan_project` still lists the
  record (auditability) but consumers can filter `split != "excluded"`.
- Tests cover membership, reason, and that a normal episode is not excluded.

### [S2.3] Out of scope

- Live 120s judgment drive (needs BeamNG window).
- italy P1 paired-rate training.
- Rewriting historical episode bytes.

## Tasks

- [ ] T1: `m5_shadow_drive` post-connect live map/vehicle in provenance —
      acceptance: unit test for resolve reuse path or small helper; script
      no longer string-literal `"map": "italy"` in provenance (covers: S2.1)
- [ ] T2: exclusion list + data_contract hooks — acceptance: bush episode
      name in JSON; `make_npz_record` marks excluded; tests (covers: S2.2)
- [ ] T3: targeted pytest — acceptance: s0 + data_contract + shadow path
      tests pass (covers: S2.1, S2.2; depends: T1, T2)
