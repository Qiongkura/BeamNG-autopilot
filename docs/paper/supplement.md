---
title: "Supplementary Material: Lane Perception for Automated Driving with Human-Anchored, Engine-Certified Ground Truth"
authors: "Zheyu Yuan (袁哲宇)"
affiliation: "South China Agricultural University, Guangzhou, China"
journal: "Supplementary material to the main manuscript"
---

# Abstract

This supplement carries the sixteen figures whose readings are quoted in the main text but which are
not load-bearing for its argument. Every figure is regenerated from the same recorded artefacts as
the main text (per-run scorecards, per-frame telemetry traces, run manifests, gate verdict files and
sealed-set records) by `m5_paper_figures.py`, `m5_paper_figures_ext.py` and
`m5_paper_figures_extra.py`; no number in this supplement is transcribed by hand. Section S-A
collects the driving-layer evidence behind Section V-E of the main text (the single-factor ladder,
per-run traces, arm distributions, lane-placement deviation, safety margins and the pairing
comparison); S-B collects the composition and post-processing scans behind Section V-C (dose response
over five metrics, parallel-merge and threshold scans, SWA, the β=0.8 loss arm, instance coverage and
the negative-side family); S-C carries the training history; S-D carries the cost ledger and the
spatial-isolation audit. The reading conventions that apply to all of them are stated in the last
section, including the three quantities that are UNKNOWN rather than zero.

**Index Terms**—Supplementary figures, negative results, single-factor ablation, driving acceptance,
audit trail.

# S-A. Driving-layer evidence

The main text reports five pre-registered driving factors, all rejected, and localises the blocker to
a static body-sweep deadlock. The figures below are the per-arm and per-frame evidence behind those
statements.

![Driving single-factor ladder: travelled distance (a) and stall fraction (b) per arm, with worst-case body-centre-cross frames.](fig6_factor_ladder.png){#fig:s6}

Every arm in {@fig:s6} still fails the `no_stall` hard target (0 frames) and the hard gates stay 0/4;
the ladder is what justifies reporting the factors as rejected rather than as partial progress.

![Per-run time series (speed top, path-to-lane deviation bottom) for a baseline acceptance run and an F-E lane-gate run.](fig24_time_series.png){#fig:s24}

{@fig:s24} shows why the stall is an availability failure and not a control failure: speed stays at
zero for long plateaus while the deviation series shows the car parked off the lane centre, and the
F-E arm moves further only by travelling with a larger deviation.

![Per-arm distributions (travelled, stall fraction) for the F-A/F-C/F-E A/Bs.](fig25_arm_boxplots.png){#fig:s25}

The arm-level distributions in {@fig:s25} are the raw material of the pre-registered maximum rule:
verdicts were read as the worst run per arm, and the overlap of the ranges is why the delivered
perception arm produced no separation on path availability (0.179 → 0.200).

![Per-frame lane-placement deviation in the closed-loop round (v7 vs v8; black bar = median).](fig26_lane_dev.png){#fig:s26}

{@fig:s26} is the closed-loop counterpart of the definitional result: under the baseline definition
the median lane deviation is 0.86 m with the body crossing the centre line in 8/8/8/13 frames, and
under the adopted definition the crossings vanish at the cost of availability.

![Safety-margin distributions (closest obstacle, minimum time-to-collision, path occupancy) over four acceptance runs.](fig27_safety_margins.png){#fig:s27}

The margins in {@fig:s27} are what backs the statement that the acceptance failures are availability
failures, not safety failures: no collision, no off-road frame and no reversing frame occurred in any
of the four runs, and the measured margins stay inside their distributions' support rather than at
their limits.

![Pairing comparison: paired-frame and candidate counts per arm (dev and limited-class pools).](fig37_pairing_compare.png){#fig:s37}

{@fig:s37} quantifies the availability cost of the adopted definition at the pairing stage itself:
3–4 paired frames are lost per 117-frame run, which is the mechanism behind the closed-loop
availability drop reported in the main text.

# S-B. Composition and post-processing scans

These are the scans behind the adopted composition (negative dose 6×) and the adopted protocol
(lateral scope 5.5 m, merge radius 1.0 m, appearance gate on).

![Dose response across five metrics (identity, role, coverage, off-road candidate fraction, merge groups), per seed and mean.](fig9_dose_metrics.png){#fig:s9}

In {@fig:s9} identity and role move with the negative dose while coverage and the off-road candidate
fraction barely do, which is why the dose was varied as the composition axis instead of a loss-weight
knob.

![Parallel-merge scan: candidates merged away and kept, with the identity consequence.](fig13_parallel_scan.png){#fig:s13}

{@fig:s13} is why the adopted merge radius is 1.0 m rather than the aggressive end of the sweep:
tightening the radius removes near-duplicate same-side candidates (dev 477 → 166 kept from 6.0 m to
3.0 m) but costs identity.

![Threshold sensitivity of the mask post-processing (keep and elongation thresholds).](fig15_threshold_scans.png){#fig:s15}

The limited-class pool carries no eligible negative frames, so the false-positive series in
{@fig:s15} is not measurable there; the gap is marked in the figure and is not read as a zero.

![SWA single-factor result: no seed is rescued by last-five-epoch averaging.](fig16_swa.png){#fig:s16}

![Tversky β=0.8 arm versus base on the negative side (false-positive frames, false-positive pixels, largest connected component).](fig17_beta08.png){#fig:s17}

{@fig:s16} and {@fig:s17} are the two rejected single factors of Section V-C: weight averaging
rescued no seed, and the Tversky loss arm tripled the false-positive frame rate (11.5% → 51.9% of
eligible frames; 329 → 1730 false-positive pixels; largest component 320 → 808 px).

![Instance-level reference coverage versus lateral offset across the certified limited-class scenes, coloured by role.](fig19_instance_coverage.png){#fig:s19}

{@fig:s19} explains why the appearance gate can afford to be aggressive: across the certified
limited-class scenes the reference instances are covered by annotation at essentially 1.0 regardless
of lateral offset, so a surviving candidate is still measured against complete truth.

![Negative-side diagnostics across all recorded pixel arms (false-positive frame rate versus false-positive pixel fraction; marker size = largest connected component).](fig20_negatives_family.png){#fig:s20}

Across every recorded pixel arm the negative-side readings in {@fig:s20} cluster rather than spread,
which is the measurement that motivated varying the negative-example dose instead of a loss-weight
knob.

# S-C. Training history

![Training curves from the recorded `train_hist.json` runs (validation line IoU and training loss per epoch).](fig21_training_curves.png){#fig:s21}

The programme's training history comprises 321 recorded runs across 44 families; {@fig:s21} shows the
representative curves. Families whose line channel was masked sit at zero line IoU by construction
rather than by failure and are annotated as such in the figure, so their curves are not negative
results. Runs converge within 20–40 epochs of the 120-epoch budget.

# S-D. Cost and data-governance audits

![GPU cost ledger across the programme's recorded rounds.](fig35_gpu_ledger.png){#fig:s35}

{@fig:s35} is the machine-readable ledger kept for every unattended round; it records cost rather than
estimating it, and it is the artefact behind the statement that the reported rounds added no new
manual annotation (the labour was machine time).

![Spatial-isolation audit: every candidate collection against the development anchors under a 50 m buffer.](fig41_isolation_audit.png){#fig:s41}

{@fig:s41} is the isolation audit that admitted the reference and sealed sets: disjointness is checked
with pose under a 50 m buffer, not with directory names, and each candidate composition gets a
recorded verdict. The same machinery rejected the first two compositions of the sealed set.

# S-E. Reading conventions and stated gaps

1. **Unit of judgement.** The acceptance results are six-seed arms (mean read against the gate, every
   seed reported); the one-shot confirmation is a single checkpoint selected by index, not by score.
2. **The sealed set is consumed.** The confirmation figures and the main-text confirmation come from that
   single consumption; the instance-level probes (identity, role, reference coverage) were **not** run on it
   and are recorded as UNKNOWN, not as passes.
3. **Two definitions of "line".** The paint scope is used for the adopted verdict, the label scope
   keeps comparability with the baseline; dilation is a trade-off between the two sets (the 0 px row
   is a `mean` read, the 1/2 px rows are `strict` reads).
4. **Frame counts are confounded by motion.** `body_cross_centre_frames` scales with how far the car
   travelled; the F-C and F-E verdicts were taken under the pre-registered maximum-over-runs rule and
   a per-distance reading was pre-registered separately for later rounds.
5. **Unequal seed counts in the dose arms.** base 1, 4× 5, 6× 1, 8.5× 5 seeds; the 6× point is the
   delivered candidate's composition and rests on a single seed.
6. **No eligible negative frames on the limited-class pool**, so that pool's false-positive rate is
   not measurable (Figure S15) and is never reported as zero.
7. **Schematics are not measurements.** The five schematic figures in the main text are drawn from the
   code and document definitions, not from run data.
8. **Artefact retention.** Per-epoch snapshots and superseded run directories were removed in a
   storage pass (audit file `logs/_cleanup_20261006.txt`); every verdict file, scorecard, manifest,
   sealed frame and the delivered checkpoint are retained, so every figure in the main text and in
   this supplement is reproducible from the retained artefacts.
