"""
sanity_check.py
================
De-risks the community-converted AntonV/mamba2-780m-hf checkpoint before we
invest in fine-tuning it. Zamba2 taught us that a subtly broken weight
conversion (the shared_transformer weight-tying bug) can silently produce
degraded output that looks plausible at a glance — so this script checks:

  1. The checkpoint loads without shape/key-mismatch errors.
  2. It generates coherent (not garbage/repeating) text on a plain prompt.
  3. It generates a sane response on one real feature-text example from this
     project, so we have a first-look sample before building the full pipeline.
  4. Whether the fast CUDA SSM kernels (mamba_ssm + causal_conv1d) are active —
     without them, transformers falls back to a much slower pure-PyTorch path,
     and any speed numbers we measure would not reflect Mamba's real potential.

Run on the cluster (needs a GPU): see run_sanity_check.sbatch
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "AntonV/mamba2-780m-hf"

# One real example, taken from this project's feature-text format
# (mamba_llm/generate_training_data.py::build_feature_text), so the first
# look at this model's behavior is on data that actually matters to us.
SAMPLE_FEATURE_TEXT = """COMMANDED WAVEFORM: sine at 100.0 Hz, target peak +/-495.8 nm

IDEAL REFERENCE for this waveform type:
  THD: 0%
  crest factor: 1.414

MEASURED FEATURES (480-740 ms window):
  Dominant frequency (Hz): 100.0
  RMS displacement (nm):   206.7
  Peak displacement (nm):  495.8
  THD (%):                 3.631
  Phase lag (deg):         76.1

DIAGNOSIS:"""


def check_fast_kernels():
    try:
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model: {MODEL_ID}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s on {model.device}")

    fast_kernels = check_fast_kernels()
    print(f"\nFast SSM kernels (mamba_ssm + causal_conv1d) available: {fast_kernels}")
    if not fast_kernels:
        print("  WARNING: falling back to pure-PyTorch SSM scan — speed numbers")
        print("  measured without these installed will NOT reflect Mamba's real")
        print("  potential. Install mamba-ssm and causal-conv1d before benchmarking.")

    # ── Test 1: plain-language coherence check ─────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1: plain prompt coherence")
    print("=" * 60)
    prompt = "The main advantage of state space models over transformers is"
    run_generation(model, tokenizer, prompt, max_new_tokens=40)

    # ── Test 2: domain feature-text sample ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: project feature-text sample (zero-shot, base model)")
    print("=" * 60)
    print("NOTE: this is a base (non-instruction-tuned) model with zero")
    print("domain fine-tuning yet, so weak/off-topic output here is expected")
    print("and is NOT evidence of a broken checkpoint. We're only checking")
    print("that generation runs and produces real words, not garbage tokens.")
    run_generation(model, tokenizer, SAMPLE_FEATURE_TEXT, max_new_tokens=60)


def run_generation(model, tokenizer, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy — deterministic, easiest to sanity-check
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = output_ids.shape[1] - input_ids.shape[1]

    text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    tok_s = new_tokens / elapsed if elapsed > 0 else 0

    print(f"\nPrompt:\n{prompt}")
    print(f"\nGenerated ({new_tokens} tokens, {elapsed:.2f}s, {tok_s:.1f} tok/s):")
    print(text)


if __name__ == "__main__":
    main()
