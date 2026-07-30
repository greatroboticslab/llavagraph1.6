"""
eval_zamba.py
=============
Full evaluation of Zamba2-7B-instruct on all 487 test samples.
Computes BLEU and ROUGE scores vs reference text.
Saves all generated outputs to results/eval_results.jsonl
"""

import warnings
import torch
import json
import re
import math
import random
import time
import os
from pathlib import Path
from collections import defaultdict

# ── Zamba2 weight-tying patch ─────────────────────────────────────────────────
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

MODEL_ID       = "Zyphra/Zamba2-7B-instruct"
TRAIN_PATH     = "/projects/ya4v/llavagraph1.6/mamba_llm/data/train.jsonl"
TEST_PATH      = "/projects/ya4v/llavagraph1.6/mamba_llm/data/test.jsonl"
RESULTS_DIR    = Path("/projects/ya4v/llavagraph1.6/mamba_zamba/results")
TEMPERATURE    = 0.3
TOP_P          = 0.85
MAX_NEW_TOKENS = 350
REP_PENALTY    = 1.15
NUM_SHOTS      = 1
RANDOM_SEED    = 42

RESULTS_DIR.mkdir(exist_ok=True)

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto",
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
            tl = model.model.layers[ti]
            sl = model.model.layers[si]
            if hasattr(tl, 'shared_transformer') and hasattr(sl, 'shared_transformer'):
                tl.shared_transformer = sl.shared_transformer
                fixed += 1
    print(f"  Weight tying fixed: {fixed} layers.")

fix_zamba2_tying(model)
model.eval()
print("Model ready.")

train_data = [json.loads(l) for l in open(TRAIN_PATH) if l.strip()]
test_data  = [json.loads(l) for l in open(TEST_PATH)  if l.strip()]
print(f"Train: {len(train_data)}  Test: {len(test_data)}")


# ── waveform type ─────────────────────────────────────────────────────────────

KNOWN_TYPES = {"sine", "square", "ramp", "pulse", "noise"}

def get_wtype(ex):
    wt = ex.get("waveform", "")
    if wt in KNOWN_TYPES:
        return wt
    m = re.search(r"COMMANDED WAVEFORM: (\w+)", ex.get("input", ""))
    return m.group(1) if m else "sine"


# ── feature parsing ──────────────────────────────────────────────────────────

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
        "peak_disp":       r"Peak displacement \(nm\):\s*([\d.]+)",
        "thd":             r"THD \(%\):\s*([\d.]+)",
        "ideal_thd":       r"THD \(theoretical\):\s*([\d.]+)%",
        "crest_factor":    r"Crest factor:\s*([\d.]+)",
        "h2_ratio":        r"2nd harmonic ratio:\s*([\d.]+)",
        "h3_ratio":        r"3rd harmonic ratio:\s*([\d.]+)",
        "phase_lag":       r"Phase lag \(deg\):\s*([-\d.]+)",
        "fopdt_lag":       r"FOPDT predicted phase lag \(deg\):\s*([-\d.]+)",
        "fopdt_atten":     r"FOPDT attenuation factor:\s*([\d.]+)",
        "amp_drift":       r"Amplitude drift Q1.Q4 \(nm\):\s*([-\d.]+)",
        "sine_resid":      r"Sine-fit residual \(%\):\s*([\d.]+)",
        "hysteresis":      r"Hysteresis \(nm\):\s*([\d.]+)",
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
        m = re.search(pat, input_text)
        if m:
            f[key] = float(m.group(1))
    return f


# ── correction vector ─────────────────────────────────────────────────────────

def compute_cv(f):
    wtype       = f.get("waveform_type", "sine")
    target_peak = f.get("target_peak", 0.0)
    rms         = f.get("rms", 1.0)
    h2          = f.get("h2_ratio", 0.0)
    h3          = f.get("h3_ratio", 0.0)

    if wtype == "sine":
        phase_lag = f.get("phase_lag") if "phase_lag" in f else f.get("fopdt_lag", 0.0)
        amp = round(target_peak / (math.sqrt(2) * rms), 4) if rms > 0 else 1.0
        return {
            "phase_offset_deg": round(-phase_lag, 3),
            "amplitude_scale":  amp,
            "harmonic_2_amp":   round(-h2, 4),
            "harmonic_3_amp":   round(-h3, 4),
            "dc_offset_nm":     0.0,
            "hysteresis_ff_nm": 0.0,
        }
    if wtype == "square":
        duty = f.get("duty_cycle", 0.5)
        edge = f.get("edge_sharpness", 1.0)
        return {
            "duty_cycle_trim":        round(0.500 - duty, 4),
            "edge_sharpness_deficit": round(1.000 - edge, 5),
            "harmonic_2_amp":         round(-h2, 5),
            "dc_offset_nm":           0.0,
        }
    if wtype == "ramp":
        amp  = round(target_peak / (math.sqrt(3) * rms), 4) if rms > 0 else 1.0
        rise = f.get("rise_linearity", 1.0)
        fall = f.get("fall_linearity", 1.0)
        asym = f.get("rise_fall_asym", 0.0)
        return {
            "rise_linearity_correction": round(1.0 - rise, 4),
            "fall_linearity_correction": round(1.0 - fall, 4),
            "asymmetry_correction":      round(-asym, 4),
            "amplitude_scale":           amp,
            "dc_offset_nm":              0.0,
        }
    if wtype == "pulse":
        amp = round(target_peak / rms, 4) if rms > 0 else 1.0
        return {
            "ringing_suppression_target": f.get("post_ringing", 0.0),
            "pulse_duty_cycle_measured":  f.get("pulse_duty", 1.0),
            "amplitude_scale":            amp,
            "dc_offset_nm":               0.0,
        }
    if wtype == "noise":
        return {
            "whitening_required": round(1.0 - f.get("spectral_flat", 1.0), 4),
            "rms_measured_nm":    rms,
            "gaussianity_error":  f.get("gaussianity_err", 0.0),
            "autocorr_lag1":      f.get("autocorr", 0.0),
        }
    phase_lag = f.get("phase_lag", f.get("fopdt_lag", 0.0))
    return {"phase_offset_deg": round(-phase_lag, 3),
            "harmonic_2_amp": round(-h2, 4), "harmonic_3_amp": round(-h3, 4)}


def format_cv(cv_dict):
    lines = ["CORRECTION VECTOR:"]
    for k, v in cv_dict.items():
        lines.append(f"  {k:<28} = {v:+.4f}")
    return "\n".join(lines)


# ── few-shot pool ─────────────────────────────────────────────────────────────

shot_pool: dict[str, list] = {}
for ex in train_data:
    shot_pool.setdefault(get_wtype(ex), []).append(ex)

random.seed(RANDOM_SEED)

def get_shots(wtype):
    pool = shot_pool.get(wtype) or train_data
    return random.sample(pool, min(NUM_SHOTS, len(pool)))

def strip_cv(output_text):
    idx = output_text.find("CORRECTION VECTOR:")
    return output_text[:idx].rstrip() if idx != -1 else output_text.rstrip()

def build_prompt(ex, shots):
    messages = [{"role": "system", "content": ex["system"]}]
    for shot in shots:
        messages.append({"role": "user",      "content": shot["input"]})
        messages.append({"role": "assistant", "content": strip_cv(shot["output"])})
    messages.append({"role": "user", "content": ex["input"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── post-processing ──────────────────────────────────────────────────────────

def fix_numbers(text, features):
    input_vals = {k: v for k, v in features.items()
                  if isinstance(v, float) and v != 0}
    def try_fix(match):
        try:
            num = float(match.group())
        except ValueError:
            return match.group()
        best_key, best_diff = None, float("inf")
        for key, val in input_vals.items():
            diff = abs(num - val) / abs(val)
            if diff < best_diff:
                best_diff, best_key = diff, key
        if best_diff < 0.20 and best_key is not None:
            exact    = input_vals[best_key]
            decimals = len(match.group().split(".")[-1]) if "." in match.group() else 0
            return f"{exact:.{decimals}f}"
        return match.group()
    return re.sub(r"\b\d+\.\d+\b", try_fix, text)

def clean_spurious(text):
    text = re.sub(r"\n\s*CORR(?!ECTION\b)[A-Z]+[^:\n]*[:\n].*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*CODI[:\s].*", "", text, flags=re.DOTALL)
    return text.strip()

STOP_STRINGS = ["CORRECTION VECTOR:", "<|im_end|>", "<|endoftext|>"]

def trim_at_stop(text, stops):
    for s in stops:
        if s in text:
            text = text[:text.index(s)]
    return text.rstrip()


# ── inference ────────────────────────────────────────────────────────────────

def run_fewshot(ex):
    features = parse_features(ex["input"])
    wtype    = features.get("waveform_type", get_wtype(ex))
    cv       = compute_cv(features)
    shots    = get_shots(wtype)

    base_prompt = build_prompt(ex, shots) + "DIAGNOSIS:"
    input_ids   = tokenizer(
        base_prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

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
    elapsed     = time.time() - t0
    new_tokens  = output_ids.shape[1] - input_ids.shape[1]

    llm_text = "DIAGNOSIS:" + tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    llm_text = trim_at_stop(llm_text, STOP_STRINGS)
    llm_text = clean_spurious(llm_text)
    llm_text = fix_numbers(llm_text, features)

    return llm_text + "\n\n" + format_cv(cv), elapsed, new_tokens


# ── metrics ──────────────────────────────────────────────────────────────────

def simple_bleu1(hyp, ref):
    hyp_tokens = hyp.lower().split()
    ref_tokens = set(ref.lower().split())
    if not hyp_tokens:
        return 0.0
    return sum(1 for t in hyp_tokens if t in ref_tokens) / len(hyp_tokens)

def rouge_l(hyp, ref):
    hyp_tokens = hyp.lower().split()
    ref_tokens = ref.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n
    rec  = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def extract_llm_text(full_output):
    idx = full_output.find("CORRECTION VECTOR:")
    return full_output[:idx].strip() if idx != -1 else full_output.strip()

def ref_llm_text(ref_output):
    idx = ref_output.find("CORRECTION VECTOR:")
    return ref_output[:idx].strip() if idx != -1 else ref_output.strip()


# ── run all test samples ─────────────────────────────────────────────────────

results_path = RESULTS_DIR / "eval_results.jsonl"
metrics_path = RESULTS_DIR / "metrics_summary.txt"

per_type_bleu   = defaultdict(list)
per_type_rouge  = defaultdict(list)
per_type_toks   = defaultdict(list)
per_type_secs   = defaultdict(list)

total_start = time.time()

with open(results_path, "w") as fout:
    for idx, ex in enumerate(test_data):
        wtype = get_wtype(ex)
        try:
            generated, elapsed, new_tokens = run_fewshot(ex)
            gen_text = extract_llm_text(generated)
            ref_text = ref_llm_text(ex.get("output", ""))
            b1   = simple_bleu1(gen_text, ref_text)
            rl   = rouge_l(gen_text, ref_text)

            per_type_bleu[wtype].append(b1)
            per_type_rouge[wtype].append(rl)
            per_type_toks[wtype].append(new_tokens)
            per_type_secs[wtype].append(elapsed)

            record = {
                "idx":       idx,
                "waveform":  wtype,
                "generated": generated,
                "reference": ex.get("output", ""),
                "bleu1":     round(b1, 4),
                "rouge_l":   round(rl, 4),
                "tokens":    new_tokens,
                "seconds":   round(elapsed, 2),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            tok_s = new_tokens / elapsed if elapsed > 0 else 0
            print(f"[{idx+1:3d}/487] {wtype:<7} | "
                  f"BLEU-1={b1:.3f} ROUGE-L={rl:.3f} | "
                  f"{new_tokens}tok {elapsed:.1f}s ({tok_s:.1f}tok/s)")

        except Exception as e:
            print(f"[{idx+1:3d}/487] {wtype:<7} | ERROR: {e}")
            fout.write(json.dumps({"idx": idx, "waveform": wtype, "error": str(e)}) + "\n")
            fout.flush()

total_elapsed = time.time() - total_start

# ── summary ───────────────────────────────────────────────────────────────────

lines = []
lines.append("=" * 60)
lines.append("EVALUATION SUMMARY — Zamba2-7B-instruct Few-Shot")
lines.append("=" * 60)
lines.append(f"Total samples: {len(test_data)}")
lines.append(f"Total time:    {total_elapsed/3600:.2f} hours\n")

all_bleu, all_rouge = [], []
for wtype in ["sine", "square", "ramp", "pulse", "noise"]:
    b  = per_type_bleu[wtype]
    r  = per_type_rouge[wtype]
    tk = per_type_toks[wtype]
    sc = per_type_secs[wtype]
    if not b:
        continue
    avg_tps = sum(tk) / sum(sc) if sum(sc) > 0 else 0
    lines.append(f"{wtype:<8} n={len(b):3d} | "
                 f"BLEU-1={sum(b)/len(b):.3f}  "
                 f"ROUGE-L={sum(r)/len(r):.3f}  "
                 f"Speed={avg_tps:.1f} tok/s")
    all_bleu.extend(b)
    all_rouge.extend(r)

lines.append("-" * 60)
lines.append(f"{'OVERALL':<8} n={len(all_bleu):3d} | "
             f"BLEU-1={sum(all_bleu)/len(all_bleu):.3f}  "
             f"ROUGE-L={sum(all_rouge)/len(all_rouge):.3f}")
lines.append("=" * 60)

summary = "\n".join(lines)
print("\n" + summary)
with open(metrics_path, "w") as f:
    f.write(summary + "\n")

print(f"\nResults saved to {results_path}")
print(f"Metrics saved to {metrics_path}")
