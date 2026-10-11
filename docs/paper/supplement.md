---
title: "Supplementary Material: Lane Perception for Automated Driving with Human-Anchored, Engine-Certified Ground Truth"
authors: "Zheyu Yuan (袁哲宇)"
affiliation: "South China Agricultural University, Guangzhou, China"
journal: "Supplementary material to the main manuscript"
---

# Abstract

This supplement contains sixteen supplementary figures and 26 individual diagnostic or schematic
figures retained from the earlier layout. Measured plots read the same recorded artefacts as
the main text (per-run scorecards, per-frame telemetry traces, run manifests, gate verdict files and
sealed-set records) by `m5_paper_figures.py`, `m5_paper_figures_ext.py` and
`m5_paper_figures_extra.py`; schematics are labelled and numerical claims are tied to recorded inputs. Section S-A
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

The main text reports five pre-registered driving factors, of which four were implemented and rejected
and one was refuted by prior measurement before implementation, and localises the blocker to a static
body-sweep deadlock. The figures below are the per-arm and per-frame evidence behind those
statements.

![Driving single-factor ladder: travelled distance (a) and stall fraction (b) per arm, with worst-case body-centre-cross frames.](fig6_factor_ladder.png){#fig:s1}

Every arm in {@fig:s1} still fails the `no_stall` hard target (0 frames) and the hard gates stay 0/4;
the ladder is what justifies reporting the implemented factors as rejected rather than as partial
progress.

![Per-run time series (speed top, path-to-lane deviation bottom) for a baseline acceptance run and an F-E lane-gate run.](fig24_time_series.png){#fig:s2}

{@fig:s2} records long intervals of low speed and a car parked off the lane centre; the F-E arm
travels further with a larger deviation. These time series alone do not identify whether the
actuator responded correctly. The corresponding planning and arbitration evidence is reported
separately in the main text.

![Per-arm distributions (travelled, stall fraction) for the F-A/F-C/F-E A/Bs.](fig25_arm_boxplots.png){#fig:s3}

The arm-level distributions in {@fig:s3} are the raw material of the pre-registered maximum rule:
verdicts were read as the worst run per arm, and the overlap of the ranges is why the delivered
perception arm produced no separation on path availability (0.179 → 0.200).

![Per-frame lane-placement deviation in the closed-loop round (v7 vs v8; black bar = median).](fig26_lane_dev.png){#fig:s4}

{@fig:s4} is the closed-loop counterpart of the definitional result: under the baseline definition
the median lane deviation is 0.86 m with recorded reference-point centre crossings in 8/8/8/13
frames. No reference-point crossings were observed under the adopted definition, at a cost in
availability; this does not establish full-body lane safety.

![Safety-margin distributions (closest obstacle, minimum time-to-collision, path occupancy) over four acceptance runs.](fig27_safety_margins.png){#fig:s5}

{@fig:s5} describes recorded obstacle and path margins. The four formal acceptance runs recorded
zero collisions, off-road frames and reversing frames, but also failed full-body centre-line
constraints (12–24 crossing frames per run). The small amount of motion and these partial metrics
cannot establish driving safety or independently confirm paved-surface containment.

![Pairing comparison: paired-frame and candidate counts per arm (dev and limited-class pools).](fig37_pairing_compare.png){#fig:s6}

{@fig:s6} quantifies the availability cost of the adopted definition at the pairing stage itself:
3–4 paired frames are lost per 117-frame evaluation. These offline counts describe a loss of
reference availability; they do not by themselves establish the cause of the closed-loop
availability difference.

# S-B. Composition and post-processing scans

These are the scans behind the adopted composition (negative dose 6×) and the adopted protocol
(lateral scope 5.5 m, merge radius 1.0 m, appearance gate on).

![Dose response across five metrics (identity, role, coverage, off-road candidate fraction, merge groups), per seed and mean.](fig9_dose_metrics.png){#fig:s7}

{@fig:s7} is a historical multi-metric scan under each arm's recorded configuration. It is separate
from the paired six-seed re-evaluation in the main text and does not establish a dose effect under
that frozen protocol.

![Parallel-merge scan: candidates merged away and kept, with the identity consequence.](fig13_parallel_scan.png){#fig:s8}

{@fig:s8} reports candidates kept and merged under the recorded parallel-merge scan configuration.
It is separate from the lateral-scope sweep and does not establish an optimal radius. The adopted
radius is specified in the frozen protocol.

![Threshold sensitivity of the mask post-processing (keep and elongation thresholds).](fig15_threshold_scans.png){#fig:s9}

The limited-class pool carries no eligible negative frames, so the false-positive series in
{@fig:s9} is not measurable there; the gap is marked in the figure and is not read as a zero.

![SWA single-factor result: no seed is rescued by last-five-epoch averaging.](fig16_swa.png){#fig:s10}

![Tversky β=0.8 arm versus base on the negative side (false-positive frames, false-positive pixels, largest connected component).](fig17_beta08_20261009.png){#fig:s11}

{@fig:s10} and {@fig:s11} are the two rejected single factors of Section V-C: weight averaging
rescued no seed, and the Tversky loss arm raised the false-positive frame rate by a factor of 4.5 (11.5% → 51.9% of
eligible frames; 329 → 1730 false-positive pixels; largest component 320 → 808 px).

![Instance-level reference coverage versus lateral offset across the certified limited-class scenes, coloured by role.](fig19_instance_coverage.png){#fig:s12}

{@fig:s12} reports high annotation coverage for the observed limited-class reference instances,
including points below 1.0. This finite sample does not guarantee complete truth for every retained
candidate or for other maps.

![Negative-side diagnostics across all recorded pixel arms (false-positive frame rate versus false-positive pixel fraction; marker size = largest connected component).](fig20_negatives_family.png){#fig:s13}

{@fig:s13} describes negative-side metrics under each recorded pixel arm's own scope and
denominators. This plot is not a controlled loss-weight comparison and does not establish an
optimal negative-example dose; the paired dose experiment is reported separately.

# S-C. Training history

![Frozen R3 mechanical learning check: six training histories, two seeds per arm, 60 optimiser steps per run. Lines show median validation line IoU and training loss; bands show the interquartile range.](fig21_learning_r3_20261009.png){#fig:s14}

The six histories in {@fig:s14} are explicitly listed and SHA-256 pinned in
`figure_inputs_20261009.json`: baseline, round 0 and round 1, each with seeds 42 and 43 and a shared
initial checkpoint. These short runs verify actual optimiser execution and history recording, not
convergence or learning gain. Both candidate rounds were rejected by the independent development
gates and the experiment stopped at its two-round budget. Validation curves use each training
run's validation split; they are not the separate 112-frame development evaluation. This mechanical
check is separate from the earlier dose experiment and does not add seeds to that comparison.

# S-D. Cost and data-governance audits

![Historical GPU cost ledger for the three recorded days shown; it excludes the later R3 mechanical check.](fig35_gpu_ledger.png){#fig:s15}

{@fig:s15} reads the historical three-day machine ledger. The later R3 mechanical check separately
records 7.4 GPU wall-clock minutes in its own run ledger and is not included in this plot. Cost
records do not certify label provenance; the no-new-manual-annotation claim is scoped to the
reported rounds and their data credentials.

![Spatial-isolation audit: every candidate collection against the development anchors under a 50 m buffer.](fig41_isolation_audit.png){#fig:s16}

{@fig:s16} is the isolation audit that admitted the reference and sealed sets: disjointness is checked
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
5. **Dose response re-evaluated.** The earlier curve's identity values do not reproduce under the frozen
   protocol (the same 6× checkpoint reads 0.763 there and 0.667 here), so it was replaced by a paired-seed
   re-evaluation (seeds 42–47): 0× 0.658, 4× 0.661, 6× 0.665 — indistinguishable — while the recorded 8.5×
   value (0.701) is higher than 6×. The 8.5× checkpoints were removed in a storage pass, so that arm cannot
   be re-evaluated.
6. **No eligible negative frames on the limited-class pool**, so that pool's false-positive rate is
   not measurable (Figure {@fig:s9}) and is never reported as zero.
7. **Schematics are not measurements.** Schematic figures are drawn from code and document
   definitions, not from run data.
8. **Artefact retention.** Per-epoch snapshots and superseded run directories were removed in a
   storage pass (audit file `logs/_cleanup_20261006.txt`). Selected verdict files, scorecards,
   manifests, seal records and the delivered checkpoint remain locally. These summaries do not
   recover removed training states or establish complete reconstruction of every figure;
   retained input scopes and remaining gaps are recorded in the claim manifest.


# S-F. Individual diagnostic and schematic figures

The 26 figures below are retained individually from the earlier composite layout. Seven now appear
as full-width single figures in the main text; the others provide supplementary diagnostics and
schematics. Their scope is specified in the claim manifest.

![Candidate-scope pipeline (schematic) with the measured revocation rate.](fig31_pipeline.png){#fig:s17}

![Counting contract (schematic): every ratio has an explicit numerator and denominator, over the frozen counter definitions; coverage is candidate-reference availability, not detection recall.](fig30_counting_contract.png){#fig:s18}

![Protocol switch matrix (schematic) and the live evaluation hash.](fig33_protocol_matrix.png){#fig:s19}

![One-shot final-confirmation flow (schematic) with the live seal.](fig34_final_set_flow.png){#fig:s20}

![Gate metrics against frozen thresholds for the two protocol definitions, development pool (a) and limited-class pool (b).](fig1_gate_matrix.png){#fig:s21}

![R3 limited-class pool, per seed: the three candidate-level gates under both definitions, drawn from the same verdict files as Table I (x = below gate).](fig18_r3_per_seed.png){#fig:s22}

![Per-seed label-scope versus paint-scope recall with the non-paint fraction of label line pixels (development pool).](fig11_recall_scopes.png){#fig:s23}

![Identity scope changes the reading and can reverse the ranking: 20 checkpoints (two arms, ten seeds each) read in label and surface scope.](fig10_identity_scopes.png){#fig:s24}

![Boundary map: dilation buys recall and pays precision; the limited-class set pays about three times more.](fig2_boundary_map.png){#fig:s25}

![Appearance-gate ablation on the pixel metrics (line IoU, precision, recall) for both sets, seed 42.](fig14_appearance_gate.png){#fig:s26}

![Lateral-scope scan on both sets: identity and role versus threshold, with the number of retained candidates.](fig12_lateral_scan.png){#fig:s27}

![Dose response of the candidate identity rate (per seed and mean).](fig3_dose_response.png){#fig:s28}

![One-shot final confirmation on the sealed road-disjoint set: per-group metrics (a) and negative-side diagnostics (b).](fig4_final_confirm.png){#fig:s29}

![Sealed final-set composition by group, with digest and protocol hash.](fig29_final_set_composition.png){#fig:s30}

![Final-set provenance audit: content-level overlap of each source package with the used frame set.](fig42_overlap_audit.png){#fig:s31}

![Closed-loop trade-off: centre crossings versus sensor lane-source rate, four runs per arm.](fig8_closed_loop_tradeoff.png){#fig:s32}

![Historical diagnostic checklist over 32 explicitly frozen town runs, listed with input digests in `figure_inputs_20261009.json` (green = recorded pass, red = fail or UNKNOWN). Adding runs or changing modification times cannot change this selection. This checklist does not qualify these runs as new strict-driving evidence.](fig22_gate_heatmap_frozen_20261009.png){#fig:s33}

![Raw crossing counts versus the same quantity normalised per 100 m travelled (log scale).](fig28_metric_scaling.png){#fig:s34}

![Safety arbitration ladder (schematic, from the monitor's rule registry).](fig32_arbitration_ladder_20261009.png){#fig:s35}

![Deadlock anatomy: the planned crossing sits just inside the 4 m threshold (a) and the car is stationary in 96% of those frames (b).](fig5_deadlock_anatomy.png){#fig:s36}

![Why the car does not move: safety-arbitration reasons over 2443 settled frames.](fig23_reason_hist.png){#fig:s37}

![Lane-acceptance evidence: baseline availability funnel (a) and the evidence behind the revoked lanes (b).](fig7_lane_gate_evidence.png){#fig:s38}

![Scene screening criterion: near-field structure density of the certified pools (black bar = median).](fig39_scene_density.png){#fig:s39}

![Negative-pool certification funnel: frames admitted as certified negatives versus excluded categories.](fig40_negative_audit.png){#fig:s40}

![Timing hygiene: repeated inference measurements on a quiet machine.](fig36_timing_retest.png){#fig:s41}

![Lateral position of line markings relative to the lane centre, pooled over the human-revised development pool.](fig38_lane_geometry.png){#fig:s42}
