# Digital Twin Physics-Deepening: Stage 0-4 Findings

Date: 2026-09-01.

Scope: `llavagraph1.6/digital_twin/`. Open-loop only. This memo documents
a self-contained investigation of whether the paper's FOPDT<->Mamba-SSM
"structural correspondence" (draft_v2.tex Sec 3, currently presented as
motivation only) can be turned into something with actual evidence
behind it: a physics simulator whose gap to real measurements is
quantified, closed partly by deepening the physics and partly by a
learned correction, with the correction itself probed for physical
interpretability rather than treated as a black box.

## Goal and framing

Numerically closing the sim-to-real gap is not the point by itself --
the point is being able to say *why* a correction looks the way it does,
and to be honest about what does and doesn't fit a named physical
mechanism. Every stage below was designed so its result is independently
checkable, and negative/inconclusive results are reported as such rather
than smoothed over.

## Stage 0 -- data quality audit

**Finding:** of 489 audited-usable real open-loop measurements
(`Mamballm2/features_full.csv`, 493 raw rows minus 4 excluded per the
earlier leakage audit -- `github_data_audit_openloop.md`), exactly
**one**, `Pulse-9_absolute.csv`, is a sensor/interferometer artifact: its
raw trace jumps to ~456,000 nm within one sample, a physically
impossible >450 micron displacement for this 25mm/0.5mm disc PZT.

Two statistical outlier-detection methods were tried first and both
failed, which is itself worth recording: cross-file MAD-based z-scoring
and an intra-file jump-ratio both broke down because low-frequency
square/pulse recordings legitimately spend most of their duration flat
with rare large step edges -- statistically indistinguishable from a
glitch by those methods, which flagged large amounts of ordinary,
correct data. What worked instead: a simple, physically-derived
amplitude ceiling (10,000 nm, derived from K~380nm/V x ~5-10V drive with
generous margin) cleanly separates the one real glitch (456,213 nm) from
the next-highest legitimate value in the entire dataset (2,570 nm) --
almost three orders of magnitude of separation, so the exact ceiling
placement doesn't matter.

Result: **488 usable measurements**, used in every subsequent stage.
Code: `audit_raw_data.py`; output `raw_data_audit.csv`,
`artifact_measurements.json` (read automatically by
`feature_extract.py`, so every downstream script sees the cleaned set).

## Stage 1 -- hysteresis model upgrade

**Motivating evidence (real data):** within a controlled frequency band
(95-105 Hz, n=24 real sine measurements), hysteresis magnitude
(`hysteresis_nm`) correlates with drive amplitude (`peak_nm`) at
r=0.59 -- a real, moderate relationship. A naive global linear fit across
the full 2-500 Hz range fails (R^2=0.024), which is expected if the true
relationship is amplitude-dependent but frequency-coupled and nonlinear
(saturating), not a simple additive term.

**A single fixed-width hysteresis operator cannot reproduce this by
construction** -- its loop height is fixed regardless of amplitude. Two
model families were compared to fix this:

- Bouc-Wen (an ODE-based model, and what an independent Gemini
  consultation suggested): inherently adds rate/frequency-dependence,
  which is a mechanism the data does **not** yet give confirmed evidence
  for (the frequency-vs-amplitude confound above hasn't been
  disentangled). Also known to be numerically difficult to fit robustly.
- **Prandtl-Ishlinskii (PI)**, chosen instead: a weighted sum of several
  play (backlash) operators at different thresholds. Small-amplitude
  drive only engages small-threshold hysterons; larger amplitude engages
  more of them, so the *aggregate* loop height grows with amplitude and
  then saturates once amplitude exceeds every threshold -- exactly the
  "grows then plateaus" shape consistent with the r=0.59 / R^2=0.024
  pair of results above. It's also the natural extension of the
  single-operator model already in the code, not a rewrite, and stays
  rate-independent (no unconfirmed mechanism added).

**A genuine implementation bug was found and fixed while verifying this
choice.** The original play operator centered its clamping window on
the *previous output* (`clip(u[n], y[n-1]-w, y[n-1]+w)`), which
provably converges to a no-op as the simulation timestep gets finer --
verified numerically to be indistinguishable from the identity function
at the simulator's actual 200kHz internal rate for realistic command
frequencies. The corrected operator centers the window on the *current
input* instead (`max(u[n]-w, min(u[n]+w, y[n-1]))`), verified to produce
a constant loop height (2w) independent of timestep or amplitude, as an
ideal play operator should. This means the hysteresis stage had likely
been near-inert in the demo for as long as it existed, prior to this
fix.

A synthetic (not real-data) verification confirmed the PI model, once
correctly implemented, reproduces the intended qualitative behavior:
loop height 0.02 -> 0.07 -> 0.128 -> 0.168(saturated) as command
amplitude sweeps past each of 3 default thresholds (0.02/0.08/0.25 V).

Code: `physics_model.py` (`_play_operator`, `_prandtl_ishlinskii`,
`TwinParams`).

### Ablation: which piece of Stage 1+2 actually mattered

The combined 23.6% physics-only gap reduction (below, Stage 2) doesn't
say whether the bug fix, the PI structure, or the shared calibration did
the work. Five variants, evaluated on the identical held-out test split:

| variant | sine | square | ramp | pulse | overall |
|---|---|---|---|---|---|
| B0 old buggy operator, uncalibrated | 1.585 | 3.923 | 5.331 | 12.780 | 5.148 |
| B1 bug fixed, single operator, uncalibrated | 1.497 | 3.762 | 5.250 | 12.149 | 4.937 |
| B2 bug fixed + PI structure, uncalibrated | 1.408 | 3.651 | 5.276 | 11.746 | 4.806 |
| B3 bug fixed + PI + shared calibration (T1-v2) | 1.090 | 2.692 | 5.092 | 7.366 | **3.563** |
| B4 bug fixed, single operator, shared-calibrated | 1.496 | 3.733 | 5.206 | 12.203 | 4.929 |

(B4 forces the PI weights w2=w3=0 during calibration -- a fair
single-hysteron-equivalent test, not just "don't calibrate hysteresis at
all". The optimizer converged to w2=w3=0.0 exactly with w1=0.50, r1=0.02,
confirming the lock worked as intended, not that it merely preferred a
tiny w2/w3.)

**The result is a clean, load-bearing finding: B0->B1->B2 (bug fix, then
PI structure, both uncalibrated) improves the gap only modestly (5.148
-> 4.806, ~6.6%). B4 (a single operator, given the SAME shared
calibration budget as B3) barely moves at all from B1 (4.937 -> 4.929,
~0.2%) -- calibrating one operator's width and threshold essentially
doesn't help. Only B2->B3 (adding calibration ON TOP OF the PI
structure) produces the big jump (4.806 -> 3.563, ~25.9%).** In other
words: neither the bug fix nor the PI structure alone explains the
improvement, and neither does calibration alone (on a single operator).
**The gain comes from the interaction: PI's multiple thresholds give a
classical optimizer something worth calibrating.** This is a materially
stronger justification for Stage 1's design choice than the earlier
synthetic loop-height verification alone.

Code: `stage1_ablation.py`; output `stage1_ablation_b4.json`.

## Stage 2 -- shared (not per-waveform-type) recalibration

**Methodological correction from the original T1:** the original
classical-optimizer calibration fit a separate parameter set per
waveform type. This is physically indefensible on reflection -- K, tau,
and hysteresis are properties of the device, not of which command shape
you happen to be driving it with. Stage 2 instead fits **one shared
parameter set across sine/square/ramp/pulse simultaneously**
(`calibrate_shared()` in `calibrate.py`), using only the TRAIN split of
a freshly-built canonical split (`make_split.py` -> `stage_split.json`,
built off the Stage-0-cleaned 488 measurements; replaces an earlier,
now-stale split file that still contained the excluded Pulse-9 glitch in
its test set).

**Result** (40 real measurements, 10 per waveform type):

| parameter | fitted value |
|---|---|
| K | 142.0 nm/V |
| tau | 296.8 us |
| hysteresis thresholds (r1,r2,r3) | 0.020, 0.085, 0.250 V |
| hysteresis weights (w1,w2,w3) | 0.486, 0.357, 0.218 |
| saturation | 2207 nm |

Gap score 7.326 -> 6.173 (a smaller relative improvement than
independent per-type fits typically achieve -- expected, since one
shared description is a harder-constrained problem than four
independently-convenient ones; the tradeoff was made deliberately in
favor of physical honesty).

**Comparison against the old (buggy, default-parameter) T0**, on the
same held-out test measurements -- this isolates how much the Stage 1+2
physics work alone (no learning at all) closed the gap:

| waveform | old T0 (buggy hysteresis, uncalibrated) | T1-v2 (fixed + shared-calibrated) | change |
|---|---|---|---|
| sine | 1.604 | 1.111 | -30.7% |
| square | 3.841 | 2.692 | -29.9% |
| ramp | 4.872 | 4.941 | +1.4% (slightly worse) |
| pulse | 9.979 | 6.880 | -31.1% |
| **overall** | **4.513** | **3.448** | **-23.6%** |

**Headline result: fixing a real implementation bug, upgrading the
hysteresis model, and recalibrating with one shared, physically-honest
parameter set closed 23.6% of the sim-to-real gap with zero learning.**
Three of four waveform types improved substantially; ramp was
essentially unchanged.

Code: `stage2_calibrate_shared.py`; output `stage2_fitted_params.json`.

### Why ramp didn't improve

Investigated directly rather than left unexplained. Compared
feature-by-feature (old T0 vs T1-v2 vs real) for a representative real
ramp measurement: `ramp_rise_linearity` and `ramp_fall_linearity` are
1.0 for real data (a perfectly straight ramp) but ~0.22 and ~-0.41 for
BOTH old T0 and T1-v2 -- the hysteresis upgrade barely moves these at
all, so whatever's wrong isn't primarily a hysteresis-model problem.

Traced it to the simulator's ramp/sawtooth output around its reset edge:
the internal 200kHz signal is smooth there (as expected -- FOPDT can't
produce a sharp corner), but the 1kHz output signal (after resampling)
shows a clear **overshoot spike** right at the reset. Initially suspected
as a resampling bug (`scipy.signal.resample_poly`'s anti-alias filter
mishandling the fast transition) -- but a from-scratch alternative
(explicit zero-phase Butterworth low-pass + plain decimation) produces
the *same* overshoot. This means it isn't a resampling-implementation
bug: **any correct anti-aliasing filter will ring when band-limiting a
transition that's fast relative to the output Nyquist rate** -- which a
real ADC's own front-end would do too, so this isn't obviously wrong
behavior for the simulator to have.

Checked whether real ramp data shows the same overshoot to see if this
explains the mismatch -- inconclusive, blocked by the **same ADC
quantization-floor problem Stage 4 ran into independently**: the real
ramp measurement checked has a step near its reset that's only one ADC
count (~79nm) different from its neighbors, too close to the
quantization floor to tell a genuine smooth-vs-overshoot difference from
quantization noise. Not resolved with current data; noted as the same
class of data limitation as the Stage 4 dual-tau question, not a new,
separate problem to chase further right now.

Code/method: ad hoc, run interactively (not saved as a standalone
script) -- reproducible via `physics_model.simulate()`'s
`y_lin_sim`/`y_nm` outputs for a ramp `TwinParams` at low frequency.

## Stage 3 -- Mamba residual correction, retrained on T1-v2

Same architecture as before (a minimal, from-scratch, pure-PyTorch
selective-SSM -- see `mamba_twin_model.py`; deliberately not the
official `mamba-ssm` package, which needs CUDA kernels this machine
doesn't have, and deliberately not a fine-tuned language-model
checkpoint, which would be the wrong tool and would overfit 488
measurements badly). Output-layer near-zero-initialized so training
starts as "trust the physics completely" and only learns to deviate
where the data says to.

Retrained on (T1-v2 ideal, real) pairs instead of the old (buggy-T0
ideal, real) pairs. **Training-time result: uncorrected-T1-v2 val loss
0.14277 -> corrected 0.12596, an 11.8% reduction** -- much smaller than
the 70% reduction seen correcting the old buggy T0, which is the
expected and desired outcome: less residual "wrongness" is left over
once the physics itself is fixed, so there's proportionally less for the
learned part to explain.

**Test-set result** (same feature-based gap score, held-out
measurements):

| waveform | T1-v2 alone | T1-v2 + Mamba correction | change |
|---|---|---|---|
| sine | 1.111 | 1.800 | **+62% (worse)** |
| square | 2.692 | 2.103 | -22% |
| ramp | 4.941 | 4.254 | -14% |
| pulse | 6.880 | 3.300 | -52% |
| **overall** | **3.448** | **2.671** | **-22.5%** |

**This is a cleaner and more interpretable story than "the learned
model always wins":** where physics already explains the data well
(sine, now that Stage 1-2 fixed it), the learned correction adds nothing
and mildly overfits/hurts; where physics still leaves a real residual
(pulse especially), the learned correction contributes substantially.
That division of labor being sensible is itself evidence the setup is
doing what it's supposed to, more so than a uniform improvement would
be.

Code: `build_pairs_residual.py` (now defaults to T1-v2 via
`stage2_fitted_params.json`, falls back to bare T0 with a printed
warning if Stage 2 hasn't been run), `train_mamba_correction.py`,
`eval_mamba_correction.py`.

## Stage 4 -- probing what the correction learned

Treated the trained corrector as an unknown system and characterized it
the same way you'd characterize the real device: fed it controlled,
in-distribution synthetic inputs (T1-v2 simulations; no real per-sample
paired command data exists to probe with directly) and examined its own
response.

**Step-transient probe (pulse, several frequencies):** the correction's
shape around each step edge is a **smooth, single-hump, non-oscillatory
transient** -- not the multiple decaying oscillations a second-order
underdamped resonance would produce. This is a specific, informative
negative result: it argues against the resonance hypothesis originally
suspected from `pulse_ringing_ratio` outliers (which Stage 0 partly
attributed to the same sensor-glitch class as Pulse-9). Instead, the
shape is more consistent with an **additional slow settling
process/second, slower time constant** that the current single-tau
FOPDT doesn't capture -- a materially different hypothesis from
"resonance," reached by directly reading the trained model's behavior
rather than re-guessing.

**Frequency sweep (sine, 10-420 Hz), phase methodology fixed:** the
original phase measurement (discrete cross-correlation shift search) was
unreliable above ~100Hz -- its resolution is limited to
360/samples-per-period degrees, only 2-3 samples/period at high
frequency and 1kHz sampling, which is why it flatlined to exactly 0
degrees for every frequency 100-420Hz (a resolution floor, not a
finding). Replaced with a least-squares sin/cos fit at the exact known
drive frequency (`_fit_at_freq()`, same method
`sine_specific_features()` already uses for `sine_phase_lag_deg`) --
this has no such resolution limit.

**With the fix, a real and much more specific pattern emerges:** gain
(fundamental-only amplitude ratio) falls smoothly from 0.90 (10Hz) to
a sharp near-zero minimum at 90Hz (0.045), while phase sits essentially
flat near +/-180 degrees across that entire 10-90Hz range. Then, in the
narrow 90-100Hz band, gain jumps back up (0.15 at 100Hz) and **phase
flips by ~170 degrees** (from -165.5 deg at 90Hz to +6.8 deg at 100Hz),
after which gain rises to a second peak near 300Hz (0.92) with phase
staying small and stable (-5 to -11 deg) for the entire 100-420Hz range.

A gain MINIMUM coinciding with a ~180-degree phase flip is not the
signature a resonance would leave (resonance shows a gain PEAK at the
phase-crossing frequency, not a minimum) -- it's closer to a sign change
/ zero crossing in the correction's own transfer function. This
supersedes the earlier (bug-driven) reading of this plot and points away
from both "resonance" and toward a different, more specific open
question: **why does the correction's effective sign invert somewhere
around 90-100Hz** -- not yet investigated further.

**Attempted direct verification against real data (the dual-tau
hypothesis) -- inconclusive, and *why* is itself the finding:**

1. *Single-transition shape*: checked several real pulse step edges
   directly. None show a resolvable multi-sample transient at all --
   the signal jumps from one value to the next-settled value within a
   single 1ms sample. This is expected once you compute it: tau~300us
   is well under one ADC sample period (1ms) at 1kHz, so **the fast
   transient's shape is fundamentally unobservable with this ADC rate**,
   regardless of analysis method.
2. *Cross-cycle plateau drift* (a way to see a slow component without
   needing to resolve the fast transient): tried on ~8 real pulse
   recordings. Files with enough repeated cycles to fit a
   drift-vs-time slope had a dynamic range of only ~79 nm -- exactly one
   ADC quantization step -- so any real drift would be invisible under
   quantization noise. Files with enough dynamic range to see drift
   clearly had only 3-4 cycles in the whole recording -- not enough to
   fit a trend.

**Conclusion: the dual-time-constant hypothesis is a specific, falsifiable
claim generated directly from probing the trained model, but the current
dataset cannot confirm or refute it** -- not a modeling gap, a data
gap: no existing recording combines (a) high enough sampling rate to
resolve a sub-millisecond transient, (b) enough dynamic range to see
past ADC quantization, and (c) enough repeated cycles/duration to see a
slow trend. Recommended as a concrete future-data-collection item (see
below), not pursued further with the current dataset.

Code: `stage4_probe_corrector.py`; output `corrector_freq_response.png`,
`corrector_step_transients.png`.

## What this supports saying in the paper

- The FOPDT<->Mamba-SSM structural correspondence (draft_v2.tex Sec 3)
  can now be backed by more than an equation-shape analogy: a
  purpose-built, from-scratch selective-SSM, trained only as a residual
  corrector on top of a classically-calibrated physical model, produces
  a *quantifiable, defensible improvement* (22.5% additional gap
  reduction on top of a physics-only 23.6% reduction), and its benefit
  is concentrated exactly where physics alone is known to be weakest
  (pulse), not spread uniformly -- which is closer to actual
  informative evidence than "it usually helps."
- The ablation (B0-B4) is the strongest single piece of evidence for the
  hysteresis-model choice: neither the bug fix nor the PI structure nor
  calibration alone accounts for the improvement -- only PI structure
  *and* calibration together do (4.806 -> 3.563), while a calibrated
  single operator (B4) barely moves from its uncalibrated version (4.937
  -> 4.929). That's a mechanistic argument, not just an aggregate number.
- The main quantitative claims (data count 488/489, gap-score numbers
  above, the ablation table, the two negative results in Stage 0's
  outlier-detection and Stage 4's dual-tau verification attempt) are all
  reproducible from the scripts named throughout.
- Honest limitations worth stating explicitly, not hiding: ramp did not
  improve from the physics upgrade (investigated -- traced to an
  anti-alias-filter overshoot at the ramp's reset edge that may or may
  not also be present in real data, blocked from checking by the same
  ADC quantization floor Stage 4 hit; not the hysteresis model's fault);
  sine's learned correction is net negative; the corrected frequency
  sweep shows a specific unexplained sign-inversion around 90-100Hz, not
  a resonance; the dual-tau hypothesis is untested, not confirmed, for
  data-availability reasons.

## Future work (explicitly, not attempted here)

- **Data collection redesign** to make the dual-tau hypothesis (and the
  ramp overshoot question) testable: either a faster ADC for
  step-response-specific recordings, or longer recordings with many more
  repeated cycles at higher amplitude (above the ~79nm quantization
  floor) than the current dataset provides. This single data-collection
  gap now blocks three separate open questions (Stage 4's dual-tau,
  ramp's reset-edge shape, and whether the 90-100Hz sign inversion has a
  real-data counterpart) -- probably the single highest-leverage next
  step if new hardware time is available.
- Investigate the 90-100Hz gain-minimum/phase-inversion pattern in the
  corrected frequency sweep -- newly found, not yet explained.
- Wire the T1-v2/Mamba-correction comparison into a more thorough
  quantitative view inside `app.py` (currently a single overlay line;
  could add the corrected signal's own feature table / gap score next
  to the uncorrected one).
