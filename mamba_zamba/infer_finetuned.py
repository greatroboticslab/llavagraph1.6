"""
infer_finetuned.py
==================
Inference with fine-tuned Zamba2-7B-instruct (LoRA adapter loaded on top of base).
No few-shot examples in prompt — the model generates from a single input.
"""

import warnings
import torch
import json
import re
import math
import time
from pathlib import Path

# ── Zamba2 weight-tying monkey-patch ─────────────────────────────────────────
from transformers import PreTrainedModel
_orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys

def _patched_get_expanded(self, all_submodels=True):
    try:
        return _orig_get_expanded(self, all_submodels=all_submodels)
    except ValueError as e:
        warnings.warn(f"Weight tying validation skipped: {e}", stacklevel=2)
        return {}

PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID      = "Zyphra/Zamba2-7B-instruct"
ADAPTER_PATH  = "/projects/ya4v/llavagraph1.6/mamba_zamba/checkpoints/lora_adapter"
TEST_PATH     = "/projects/ya4v/llavagraph1.6/mamba_llm/data/test.jsonl"
MAX_NEW_TOKENS = 400
TEMPERATURE    = 0.2
TOP_P          = 0.9
REP_PENALTY    = 1.1
NUM_TEST       = 5

# ── load ──────────────────────────────────────────────────────────────────────

print(f"Loading tokenizer from adapter: {ADAPTER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def fix_zamba2_tying(model):
    tied = getattr(model.model, '_tied_weights_keys', {})
    fixed = 0
    for target_pattern, source_pattern in tied.items():
        parts_t = target_pattern.split('.')
        parts_s = source_pattern.split('.')
        if (len(parts_t) >= 3 and parts_t[0] == 'layers'
                and parts_t[2] == 'shared_transformer'):
            ti = int(parts_t[1])
            si = int(parts_s[1])
            tl = base_model.model.layers[ti]
            sl = base_model.model.layers[si]
            if hasattr(tl, 'shared_transformer') and hasattr(sl, 'shared_transformer'):
                tl.shared_transformer = sl.shared_transformer
                fixed += 1
    print(f"  Weight tying fixed: {fixed} layers.")

fix_zamba2_tying(base_model)

print(f"Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model ready.")

# ── correction vector (same analytical formulas as infer_zamba.py) ────────────

KNOWN_TYPES = {"sine", "square", "ramp", "pulse", "noise"}

def get_wtype(ex):
    wt = ex.get("waveform", "")
    if wt in KNOWN_TYPES:
        return wt
    m = re.search(r"COMMANDED WAVEFORM: (\w+)", ex.get("input", ""))
    return m.group(1) if m else "sine"

def parse_features(input_text):
    f = {}
    m = re.search(r"COMMANDED WAVEFORM: (\w+) at ([\d.]+) Hz", input_text)
    if m:
        f["waveform_type"] = m.group(1)
        v = float(m.group(2))
        f["freq"] = int(v) if v == int(v) else v
    m = re.search(r"target peak ±([\d.]+) nm", input_text)
    if m:
        f["target_peak"] = float(m.group(1))
    patterns = {
        "rms":             r"RMS displacement \(nm\):\s*([\d.]+)",
        "thd":             r"THD \(%\):\s*([\d.]+)",
        "h2_ratio":        r"2nd harmonic ratio:\s*([\d.]+)",
        "h3_ratio":        r"3rd harmonic ratio:\s*([\d.]+)",
        "phase_lag":       r"Phase lag \(deg\):\s*([-\d.]+)",
        "fopdt_lag":       r"FOPDT predicted phase lag \(deg\):\s*([-\d.]+)",
        "duty_cycle":      r"Duty cycle:\s*([\d.]+)",
        "edge_sharpness":  r"Edge sharpness:\s*([\d.]+)",
        "gaussianity_err": r"Gaussianity error:\s*([\d.]+)",
        "autocorr":        r"Autocorrelation lag-1:\s*([\d.]+)",
        "spectral_flat":   r"Spectral flatness:\s*([\d.]+)",
        "rise_linearity":  r"Rise linearity:\s*([\d.]+)",
        "fall_linearity":  r"Fall linearity:\s*([\d.]+)",
        "rise_fall_asym":  r"Rise/fall asymmetry:\s*([\d.]+)",
        "post_ringing":    r"Post-pulse ringing ratio:\s*([\d.]+)",
        "pulse_duty":      r"Pulse duty cycle:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m2 = re.search(pat, input_text)
        if m2:
            f[key] = float(m2.group(1))
    return f

def compute_cv(f):
    wtype       = f.get("waveform_type", "sine")
    target_peak = f.get("target_peak", 0.0)
    rms         = f.get("rms", 1.0)
    h2          = f.get("h2_ratio", 0.0)
    h3          = f.get("h3_ratio", 0.0)
    if wtype == "sine":
        phase_lag = f.get("phase_lag", f.get("fopdt_lag", 0.0))
        amp = round(target_peak / (math.sqrt(2) * rms), 4) if rms > 0 else 1.0
        return {"phase_offset_deg": round(-phase_lag, 3), "amplitude_scale": amp,
                "harmonic_2_amp": round(-h2, 4), "harmonic_3_amp": round(-h3, 4),
                "dc_offset_nm": 0.0, "hysteresis_ff_nm": 0.0}
    if wtype == "square":
        duty = f.get("duty_cycle", 0.5)
        edge = f.get("edge_sharpness", 1.0)
        return {"duty_cycle_trim": round(0.500 - duty, 4),
                "edge_sharpness_deficit": round(1.000 - edge, 5),
                "harmonic_2_amp": round(-h2, 5), "dc_offset_nm": 0.0}
    if wtype == "ramp":
        amp = round(target_peak / (math.sqrt(3) * rms), 4) if rms > 0 else 1.0
        return {"rise_linearity_correction": round(1.0 - f.get("rise_linearity", 1.0), 4),
                "fall_linearity_correction": round(1.0 - f.get("fall_linearity", 1.0), 4),
                "asymmetry_correction": round(-f.get("rise_fall_asym", 0.0), 4),
                "amplitude_scale": amp, "dc_offset_nm": 0.0}
    if wtype == "pulse":
        amp = round(target_peak / rms, 4) if rms > 0 else 1.0
        return {"ringing_suppression_target": f.get("post_ringing", 0.0),
                "pulse_duty_cycle_measured": f.get("pulse_duty", 1.0),
                "amplitude_scale": amp, "dc_offset_nm": 0.0}
    if wtype == "noise":
        return {"whitening_required": round(1.0 - f.get("spectral_flat", 1.0), 4),
                "rms_measured_nm": rms, "gaussianity_error": f.get("gaussianity_err", 0.0),
                "autocorr_lag1": f.get("autocorr", 0.0)}
    return {}

def format_cv(cv_dict):
    lines = ["CORRECTION VECTOR:"]
    for k, v in cv_dict.items():
        lines.append(f"  {k:<28} = {v:+.4f}")
    return "\n".join(lines)

# ── inference ─────────────────────────────────────────────────────────────────

STOP_STRINGS = ["CORRECTION VECTOR:", "<|im_end|>", "<|endoftext|>"]

def trim_at_stop(text):
    for s in STOP_STRINGS:
        if s in text:
            text = text[:text.index(s)]
    return text.rstrip()

def run_inference(ex):
    features = parse_features(ex["input"])
    cv       = compute_cv(features)

    messages = [
        {"role": "system", "content": ex["system"]},
        {"role": "user",   "content": ex["input"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "DIAGNOSIS:"

    input_ids   = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
    prompt_len  = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=REP_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed    = time.time() - t0
    new_tokens = output_ids.shape[1] - prompt_len
    print(f"  {new_tokens} tokens in {elapsed:.1f}s ({new_tokens/elapsed:.1f} tok/s)")

    generated = "DIAGNOSIS:" + tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    )
    generated = trim_at_stop(generated)
    return generated + "\n\n" + format_cv(cv)

# ── run on test samples ───────────────────────────────────────────────────────

test_data = [json.loads(l) for l in open(TEST_PATH) if l.strip()]
WAVEFORM_TYPES = ["sine", "square", "ramp", "pulse", "noise"]
SAMPLES_PER_TYPE = 1

buckets = {w: [] for w in WAVEFORM_TYPES}
for ex in test_data:
    wt = get_wtype(ex)
    if wt in buckets and len(buckets[wt]) < SAMPLES_PER_TYPE:
        buckets[wt].append(ex)

eval_samples = [ex for wt in WAVEFORM_TYPES for ex in buckets[wt]]
print(f"\nRunning on {len(eval_samples)} test samples (1 per waveform type)")

for idx, ex in enumerate(eval_samples):
    wtype = get_wtype(ex)
    print(f"\n{'='*65}")
    print(f"SAMPLE {idx}  |  waveform: {wtype}")
    print(f"{'='*65}")
    result = run_inference(ex)
    print("--- GENERATED (fine-tuned) ---")
    print(result)
    print("\n--- REFERENCE ---")
    print(ex["output"][:700])
