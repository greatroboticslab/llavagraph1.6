"""
label_scheme.py
================
Defines the 17-category defect-classification scheme used by the
classification track (train_classifier.py / eval_classifier.py), and the
single-token label each category maps to.

The category for a sample is "which measured quantity deviates most from
its typical value, relative to how much that quantity normally varies" —
same z-score/argmax method used earlier to explore the data (see
conversation notes), now finalized into a fixed, deterministic function so
labels are computed identically at data-prep time and can be reproduced
for analysis later.

Every field used here is confirmed present in build_feature_text()'s
output for that waveform type (checked against generate_training_data.py's
_EXCLUDE sets) — a first pass mistakenly included pulse's crest factor
using the `pulse_crest_factor` CSV column, which is never actually shown
in the input text (only the general `crest_factor` field is); that would
have made the category unlearnable from the model's actual input. Fixed
before this was ever trained on.

MEAN/STD below are computed once from the full 493-sample
mamba_llm/features_full.csv and hardcoded — this is a fixed labeling
function, not something recomputed per split, so the same input always
maps to the same label regardless of which script calls it.
"""

import re

STATS = {
    "sine": {
        "phase_offset_deg": {"mean": 98.893352, "std": 52.816369},
        "harmonic_2_amp":   {"mean": 0.030602,  "std": 0.052885},
        "harmonic_3_amp":   {"mean": 0.013237,  "std": 0.026774},
        "hysteresis_ff_nm": {"mean": 13.459091, "std": 12.364511},
    },
    "square": {
        "harmonic_2_amp":         {"mean": 0.092711, "std": 0.172516},
        "duty_cycle_trim":        {"mean": 0.044208, "std": 0.064967},
        "edge_sharpness_deficit": {"mean": 1.034447, "std": 0.736856},
    },
    "ramp": {
        "rise_linearity_correction": {"mean": 0.179422, "std": 0.186583},
        "fall_linearity_correction": {"mean": 0.175784, "std": 0.181652},
        "asymmetry_correction":      {"mean": 0.017956, "std": 0.048124},
    },
    "pulse": {
        "ringing_suppression":  {"mean": 1.218786,  "std": 1.289877},
        "duty_cycle_deviation": {"mean": 0.406505,  "std": 0.103195},
        "amplitude_scale_dev":  {"mean": 4.383214,  "std": 11.929593},
        "crest_dev":            {"mean": 5.383213,  "std": 11.929592},
    },
    "noise": {
        "whitening_required": {"mean": 0.280484, "std": 0.046636},
        "gaussianity_error":  {"mean": 7.700809, "std": 22.218498},
        "autocorr_lag1":      {"mean": 0.689122, "std": 0.258781},
    },
}

# Single-token labels (verified single-token in AntonV/mamba2-780m-hf's
# tokenizer — see conversation notes). Order matches STATS insertion order.
LABELS = {}
_chars = list("0123456789ABCDEFG")
_i = 0
for _wt, _fields in STATS.items():
    for _field in _fields:
        LABELS[(_wt, _field)] = _chars[_i]
        _i += 1

LABEL_TO_CATEGORY = {v: k for k, v in LABELS.items()}
assert len(LABELS) == 17 and len(LABEL_TO_CATEGORY) == 17


def _get(input_text, pattern):
    m = re.search(pattern, input_text)
    return float(m.group(1)) if m else None


def parse_relevant_fields(input_text, waveform):
    """Pull just the fields this waveform type's category depends on."""
    f = {}
    if waveform == "sine":
        phase = _get(input_text, r"Phase lag \(deg\):\s*([-\d.]+)")
        if phase is None:
            phase = _get(input_text, r"FOPDT predicted phase lag \(deg\):\s*([-\d.]+)")
        f["phase_offset_deg"] = abs(phase) if phase is not None else 0.0
        h2 = _get(input_text, r"2nd harmonic ratio:\s*([\d.]+)") or 0.0
        h3 = _get(input_text, r"3rd harmonic ratio:\s*([\d.]+)") or 0.0
        hyst = _get(input_text, r"Hysteresis \(nm\):\s*([\d.]+)") or 0.0
        f["harmonic_2_amp"] = abs(h2)
        f["harmonic_3_amp"] = abs(h3)
        f["hysteresis_ff_nm"] = abs(hyst)
    elif waveform == "square":
        h2 = _get(input_text, r"2nd harmonic ratio:\s*([\d.]+)") or 0.0
        duty = _get(input_text, r"Duty cycle:\s*([\d.]+)")
        edge = _get(input_text, r"Edge sharpness:\s*([\d.]+)")
        f["harmonic_2_amp"] = abs(h2)
        f["duty_cycle_trim"] = abs(0.5 - duty) if duty is not None else 0.0
        f["edge_sharpness_deficit"] = abs(1.0 - edge) if edge is not None else 0.0
    elif waveform == "ramp":
        rise = _get(input_text, r"Rise linearity:\s*([\d.]+)")
        fall = _get(input_text, r"Fall linearity:\s*([\d.]+)")
        asym = _get(input_text, r"Rise/fall asymmetry:\s*([-\d.]+)") or 0.0
        f["rise_linearity_correction"] = abs(1.0 - rise) if rise is not None else 0.0
        f["fall_linearity_correction"] = abs(1.0 - fall) if fall is not None else 0.0
        f["asymmetry_correction"] = abs(asym)
    elif waveform == "pulse":
        ring = _get(input_text, r"Post-pulse ringing ratio:\s*([\d.]+)") or 0.0
        duty = _get(input_text, r"Pulse duty cycle:\s*([\d.]+)")
        peak = _get(input_text, r"Peak displacement \(nm\):\s*([\d.]+)") or 0.0
        rms = _get(input_text, r"RMS displacement \(nm\):\s*([\d.]+)") or 1.0
        crest = _get(input_text, r"Crest factor:\s*([\d.]+)") or 0.0
        f["ringing_suppression"] = abs(ring)
        f["duty_cycle_deviation"] = abs(0.5 - duty) if duty is not None else 0.0
        f["amplitude_scale_dev"] = abs(1.0 - peak / rms) if rms else 0.0
        f["crest_dev"] = abs(crest)
    elif waveform == "noise":
        flat = _get(input_text, r"Spectral flatness:\s*([\d.]+)")
        gauss = _get(input_text, r"Gaussianity error:\s*([\d.]+)") or 0.0
        auto = _get(input_text, r"Autocorrelation lag-1:\s*([\d.]+)") or 0.0
        f["whitening_required"] = abs(1.0 - flat) if flat is not None else 0.0
        f["gaussianity_error"] = abs(gauss)
        f["autocorr_lag1"] = abs(auto)
    return f


def compute_label(input_text, waveform):
    """Returns the single-character label token for this sample."""
    fields = parse_relevant_fields(input_text, waveform)
    stats = STATS[waveform]
    best_field, best_z = None, -1.0
    for field, val in fields.items():
        s = stats[field]
        z = abs((val - s["mean"]) / s["std"]) if s["std"] else 0.0
        if z > best_z:
            best_z, best_field = z, field
    return LABELS[(waveform, best_field)]


def majority_baseline(waveform_counts_by_label):
    """Given {label: count} for one waveform type, return the majority-class
    baseline accuracy (what you'd get by always predicting the most common
    label)."""
    total = sum(waveform_counts_by_label.values())
    return max(waveform_counts_by_label.values()) / total if total else 0.0
