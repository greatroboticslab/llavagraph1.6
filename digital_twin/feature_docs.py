"""
feature_docs.py
================
Short, plain-English descriptions of every feature
mamballm/batch_feature_extraction.py can produce, for the "explanation"
column in app.py's live properties table. Exists so the demo can be
read without cross-referencing the extraction code -- the point of the
table is to let you judge whether each physical quantity is actually
one you need, not to memorize what 50 column names mean.
"""

_DESCRIPTIONS = {
    "n_samples": "Number of samples in the analyzed signal.",

    # -- fft_features() --
    "dominant_freq_hz": "Strongest frequency component. For sine/square/ramp/pulse this "
                         "should track the commanded drive frequency; unreliable for noise "
                         "(no single dominant frequency by design).",
    "dominant_amp_nm": "FFT amplitude at the dominant frequency, in nm.",
    "harmonic_ratio_2": "2nd-harmonic amplitude / dominant amplitude. How much energy leaked "
                         "into 2x the drive frequency -- one signature of nonlinear distortion.",
    "harmonic_ratio_3": "3rd-harmonic amplitude / dominant amplitude. Odd harmonics like this "
                         "are the classic signature of symmetric nonlinearities (e.g. hysteresis).",
    "harmonic_ratio_4": "4th-harmonic amplitude / dominant amplitude.",
    "harmonic_ratio_5": "5th-harmonic amplitude / dominant amplitude.",
    "odd_even_harmonic_ratio": "(3rd+5th harmonic energy) / (2nd+4th harmonic energy). Useful "
                                "to distinguish square-like distortion (odd-dominant) from "
                                "ramp-like distortion (both present).",
    "thd_pct": "Total Harmonic Distortion: combined energy of harmonics 2-5 relative to the "
               "fundamental, as a percentage. One overall 'how impure is this waveform' score. "
               "Note: square/ramp/pulse are naturally high-THD even for an ideal actuator -- "
               "compare against the real-device mean for that waveform type, not against 0.",
    "spectral_entropy": "Randomness of the power spectrum. Low = concentrated at one frequency "
                         "(sine, pulse); high = spread across many frequencies (noise).",
    "spectral_centroid_hz": "'Center of mass' of the power spectrum, in Hz.",
    "spectral_flatness": "Geometric mean / arithmetic mean of the spectrum. Near 1 = flat "
                          "spectrum (white noise); near 0 = tonal (concentrated at one or a "
                          "few frequencies).",
    "band_energy_ratio": "Energy in 1-100 Hz vs. 100-500 Hz. Sine tends to concentrate energy "
                          "at low frequency; noise spreads it out more evenly.",
    "num_spectral_peaks": "Count of distinct significant peaks in the spectrum.",

    # -- time_domain_features() --
    "rms_nm": "Root-mean-square displacement, in nm -- overall signal 'size'.",
    "peak_nm": "Peak absolute displacement, in nm.",
    "crest_factor": "Peak / RMS. High for spiky waveforms (pulse), low for waveforms that "
                     "spend most of their time near peak amplitude (square).",
    "skewness": "Asymmetry of the displacement distribution around its mean. 0 = symmetric.",
    "kurtosis": "'Peakedness' of the displacement distribution. Gaussian noise has kurtosis "
                "~3; sharply peaked signals (pulse) read much higher.",
    "zero_cross_rate": "How often the signal crosses zero, in crossings/sec -- a rough proxy "
                        "for frequency content that doesn't need FFT peak-picking.",
    "peak_density_hz": "Local-maxima count per second.",

    # -- sine_specific_features() --
    "sine_residual_pct": "RMS error between the signal and the best-fit pure sine at the "
                          "dominant frequency, as % of signal std. Directly measures 'how far "
                          "from an ideal sine' -- the single most direct nonlinearity gauge "
                          "for sine drive.",
    "sine_phase_lag_deg": "Phase lag (degrees) of the best-fit sine relative to a zero-phase "
                           "reference. Compare against fopdt_phase_lag_deg (below) -- the gap "
                           "between the two is the part FOPDT's linear model can't explain.",

    # -- square_specific_features() --
    "square_duty_cycle": "Fraction of time the signal is above zero. Should be near the "
                          "commanded duty cycle; drift indicates asymmetric rise/fall dynamics.",
    "square_edge_sharpness": "Variance of the derivative relative to signal variance -- high "
                              "= sharp edges (ideal square), low = edges rounded off by the "
                              "actuator's limited bandwidth (it can't slew fast enough).",

    # -- noise_specific_features() --
    "noise_kurtosis": "Kurtosis of the (normalized) signal. True Gaussian noise reads ~3.",
    "noise_gaussianity_err": "|kurtosis - 3|. How far the signal's distribution is from "
                              "Gaussian -- 0 = perfectly Gaussian noise.",
    "noise_autocorr_lag1": "Correlation between the signal and itself shifted by one sample. "
                            "Near 0 for true white noise; a nonzero value means consecutive "
                            "samples aren't independent (e.g. low-pass-filtered noise).",

    # -- ramp_specific_features() --
    "ramp_rise_linearity": "How straight the rising segments are (1 = perfectly linear). "
                            "Hysteresis/creep bends what should be a straight ramp into an "
                            "S-curve.",
    "ramp_fall_linearity": "Same, for falling segments.",
    "ramp_asymmetry": "Difference between rise-segment and fall-segment slopes, normalized. "
                       "0 = rise and fall are mirror images of each other.",

    # -- pulse_specific_features() --
    "pulse_crest_factor": "Peak / RMS for the pulse waveform specifically.",
    "pulse_duty_cycle": "Fraction of time the signal exceeds 50% of its peak.",
    "pulse_ringing_ratio": "RMS of the signal tail (10ms after the peak) relative to overall "
                            "RMS. High = the actuator 'rings'/oscillates after the pulse "
                            "instead of settling cleanly -- a resonance/underdamping signature.",

    # -- quarter_features() --
    "amplitude_drift_nm": "Max minus min peak amplitude across the four 65ms quarters of the "
                           "analysis window. Nonzero = the response amplitude isn't stable "
                           "over the recording (possible creep or drift).",
    "worst_quarter": "Which quarter (q1-q4) had the largest peak amplitude.",

    # -- fopdt_features() --
    "fopdt_phase_lag_deg": "Phase lag the LINEAR FOPDT model alone predicts at this frequency "
                            "(from K/tau/theta), ignoring hysteresis and other nonlinearities. "
                            "The reference value sine_phase_lag_deg is checked against.",
    "fopdt_attenuation": "Amplitude attenuation factor the linear FOPDT model predicts at "
                          "this frequency (1 = no attenuation).",
    "v_drive_est_v": "Drive voltage estimated by inverting the FOPDT model from the measured "
                      "peak displacement -- a sanity check on what voltage was likely applied.",

    # -- hysteresis_proxy() (sine only) --
    "hysteresis_nm": "Mean positive-peak magnitude minus mean negative-peak magnitude, in nm. "
                      "The direct, simplest measurement of hysteresis: how asymmetric the "
                      "positive and negative half-cycles are.",
    "half_cycle_asymmetry": "hysteresis_nm normalized by overall peak amplitude (a fraction "
                             "instead of an absolute nm value) -- lets you compare hysteresis "
                             "severity across signals of different amplitude.",
}

_QUARTER_SUFFIXES = {
    "peak_nm": "Peak absolute displacement within this quarter of the window, in nm.",
    "mean_nm": "Mean displacement within this quarter of the window, in nm.",
    "rms_nm": "RMS displacement within this quarter of the window, in nm.",
    "sine_residual_pct": "Sine-fit residual (see sine_residual_pct) computed within just this "
                          "quarter -- lets you see if distortion is worse in one part of the "
                          "recording than another.",
}


def explain(feature_name: str) -> str:
    if feature_name in _DESCRIPTIONS:
        return _DESCRIPTIONS[feature_name]
    for q in ("q1_", "q2_", "q3_", "q4_"):
        if feature_name.startswith(q):
            base = feature_name[len(q):]
            if base in _QUARTER_SUFFIXES:
                return f"Quarter {q[1]} ({int(q[1])}/4 of the window): " + _QUARTER_SUFFIXES[base]
    return ""
