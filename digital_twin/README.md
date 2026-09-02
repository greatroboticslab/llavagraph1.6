# digital_twin

Open-loop PZT actuator digital twin. Reframes the weakest part of
`draft_v2.tex` (the FOPDT <-> Mamba SSM "structural correspondence",
which the paper itself describes as motivation only, not a validated
claim) into something testable: a physics simulator whose gap to real
measurements is quantified, closed partly by deepening the physics and
partly by a learned correction, with the correction probed for physical
interpretability rather than treated as a black box.

Full narrative writeup (all stages, all numbers, what's still open):
[`PHYSICS_DEEPENING_WRITEUP.md`](PHYSICS_DEEPENING_WRITEUP.md) in this
folder (also kept at `newpaper_draft/docs/digital_twin_physics_deepening_20260901.md`
in the paper repo, for anyone working from that side instead).

## Core files

- `physics_model.py` -- forward simulator (**T0**). Hammerstein structure:
  command voltage -> hysteresis (Prandtl-Ishlinskii, 3 weighted play
  operators -- `_prandtl_ishlinskii()`) -> FOPDT linear dynamics (same
  equation/defaults as `batch_feature_extraction.py`) -> soft saturation
  -> measurement noise. Simulated at 200kHz, resampled to 1kHz to match
  the real ADC rate.
- `feature_extract.py` -- imports `mamballm/batch_feature_extraction.py`
  directly (never reimplements a formula) so simulated and real signals
  are always scored by identical code. Also resolves real measurement
  files (scattered across several folders on this machine) and applies
  every exclusion (leakage-audit + data-quality) so all downstream
  scripts see the same clean, current dataset automatically.
- `feature_docs.py` -- plain-English explanation of every feature, shown
  in `app.py`'s Glossary dialog (a button that opens a modal, not an
  inline table column).
- `calibrate.py` -- classical (Nelder-Mead) parameter fitting: `calibrate()`
  (one waveform type) and `calibrate_shared()` (one parameter set across
  multiple types -- Stage 2), plus `feature_gap()`, the shared sim-vs-real
  scoring function used everywhere in this project.
- `app.py` -- Streamlit demo: sliders for every physical parameter
  (hysteresis grouped under a "Prandtl-Ishlinskii, 3 hysterons" expander),
  live waveform plot, live feature table with real-device reference
  values and a Glossary button, frequency-matched real-measurement
  overlay, a T1 calibration panel, and an optional "T1-v2 + Mamba
  correction" overlay (uses Stage 2's fixed parameters regardless of the
  sliders above -- see its checkbox tooltip for why). Run with
  `python3 -m streamlit run app.py`.
- `mamba_twin_model.py` -- minimal, pure-PyTorch selective-SSM
  implementation (the actual Mamba recurrence, A/B/C/Delta -- not a
  conv+gating stand-in). No CUDA on this machine, so this deliberately
  isn't the official `mamba-ssm` package; see the file's docstring.

## Pipeline, in run order

1. `audit_raw_data.py` -- **Stage 0.** Flags physically-impossible real
   measurements (1 of 489: `Pulse-9_absolute.csv`, ~456,000nm).
   Output: `raw_data_audit.csv`, `artifact_measurements.json` (read
   automatically by `feature_extract.py`). Usable count: **488**.
2. `make_split.py` -- canonical stratified train/val/test split off the
   cleaned 488, saved to `stage_split.json`. Every later stage uses this
   (not the earlier, now-stale `pairs_data/meta.json`, which still had
   Pulse-9 in its test set).
3. `stage2_calibrate_shared.py` -- **Stage 2.** Fits ONE shared parameter
   set (K, tau, PI hysteresis thresholds+weights, saturation) across all
   4 waveform types at once, TRAIN split only. Output:
   `stage2_fitted_params.json` (T1-v2).
4. `stage1_ablation.py` -- decomposes the Stage 1+2 gap reduction into
   bug-fix-alone / PI-structure-alone / calibration-alone / PI+calibration
   contributions (B0-B4). Output: `stage1_ablation_b4.json`,
   `stage1_ablation.log`.
5. `build_pairs_residual.py` -- pairs T1-v2's simulated waveform with the
   real response (phase-aligned via `signal_align.py`). Uses
   `stage2_fitted_params.json` automatically if present (prints a warning
   and falls back to bare T0 defaults if Stage 2 hasn't been run).
6. `train_mamba_correction.py` / `eval_mamba_correction.py` -- trains
   `y_pred = y_ideal + MambaTwin(y_ideal)` (near-zero output-layer init,
   so it starts as "trust physics completely"), evaluates on held-out
   test data. Output: `mamba_correction.pt`, `correction_examples.png`.
7. `stage4_probe_corrector.py` -- **Stage 4.** Feeds the trained corrector
   controlled synthetic (T1-v2) inputs and examines its own step response
   and frequency response (phase via a least-squares sin/cos fit at the
   exact drive frequency, not a discrete-shift search -- the first version
   of this was unreliable above ~100Hz, see the file's `_fit_at_freq()`
   docstring). Output: `corrector_freq_response.png`,
   `corrector_step_transients.png`.

## Results summary (see the docs/ writeup for full detail and honest caveats)

- **Stage 1+2, physics only, no learning**: fixing a real bug (the
  original play operator was numerically inert at the simulator's actual
  timestep), upgrading hysteresis to Prandtl-Ishlinskii, and sharing one
  calibrated parameter set across all 4 waveform types closed **23.6%**
  of the sim-to-real gap vs. the old buggy/uncalibrated baseline.
- **Ablation (B0-B4)**: neither the bug fix nor the PI structure nor
  calibration alone explains that 23.6% -- a single operator, even
  shared-calibrated (B4), barely moves from its uncalibrated version.
  The gain is specifically PI structure *enabling* calibration to do
  something (B2->B3: 4.806 -> 3.563).
- **Stage 3, Mamba correction on top of T1-v2**: a further **22.5%**
  gap reduction overall, but *not* uniform -- helps square/ramp/pulse
  (pulse most, -52%), makes sine slightly worse (+62%, since T1-v2
  already explains sine well and the corrector has nothing useful left
  to learn there). That non-uniformity is itself evidence the
  physics/learned split is doing something real, not just "the neural
  net always wins."
- **Stage 4**: no evidence of second-order resonance (step transients
  are smooth, non-oscillatory). The fixed-methodology frequency sweep
  instead shows a gain minimum + ~170-degree phase flip around 90-100Hz
  -- a sign-inversion signature, not investigated further. A separate
  "does the corrector imply a second, slower time constant" hypothesis
  (from the step-transient shape) could not be confirmed or refuted
  against real data -- blocked by the SAME data limitation independently
  found while investigating why ramp didn't improve under T1-v2: real
  low-frequency recordings are either too short (too few repeated
  cycles) or too close to the ~79nm ADC quantization floor to resolve
  either question.

## Known discrepancy: Stage 3 as planned vs. as built

An earlier version of this file described two additional Stage 3
refinements -- feeding the corrector a `dy_ideal/dt` input channel
(rate/direction information for hysteresis) and a residual-magnitude
loss penalty (to keep the correction as small as possible) -- as if
already done. **They are not implemented.** `train_mamba_correction.py`
currently trains on `y_ideal` alone with plain MSE. Left here as an
explicit, checkable TODO rather than silently corrected, since the
results above are all measured against what's actually in the code, not
against that unbuilt version.

## What didn't become the main line (kept, not deleted)

- `build_pairs.py` / `train_mamba_twin.py` / `eval_mamba_twin.py` /
  `compare_T0_T1_T2.py`: an earlier attempt at training Mamba end-to-end
  on (reconstructed command -> real response) rather than as a physics
  correction. Result was mixed (won on square/pulse, lost on ramp) and,
  more importantly, isn't built around an interpretable physics/learned
  split. Kept as a documented negative-ish comparison, not the direction
  going forward.
