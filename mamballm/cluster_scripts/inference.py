"""
inference.py
============
Test the fine-tuned LoRA adapter on the test set and print sample outputs.

Usage (via SLURM — see submit_inference.sh):
    python inference.py --adapter_dir ./checkpoints/final_adapter --data_dir ./data
"""

import argparse
import json
import types
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "nvidia/Nemotron-H-8B-Base-8K"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(example: dict) -> str:
    return (
        f"### System:\n{example['system']}\n\n"
        f"### Instruction:\n{example['input']}\n\n"
        f"### Response:\n"
    )




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="./checkpoints/final_adapter")
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--n_samples",   type=int, default=10)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit NF4 (Mamba mixer in bfloat16 for CUDA kernel compat)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Mirror the training patch: CUDA SSM kernels break with 4-bit quantized weights.
    # Force torch_forward on every Mamba mixer, exactly as done in finetune.py.
    def _torch_forward_only(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

    n_patched = 0
    for module in model.modules():
        if hasattr(module, "torch_forward") and hasattr(module, "cuda_kernels_forward"):
            module.forward = types.MethodType(_torch_forward_only, module)
            n_patched += 1
    print(f"  Patched {n_patched} Mamba mixer(s): forcing torch_forward")

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = load_jsonl(Path(args.data_dir) / "test.jsonl")
    samples = test_data[:args.n_samples]

    print(f"\nRunning inference on {len(samples)} test samples...\n{'='*60}")

    for i, example in enumerate(samples):
        prompt = format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"\n[{i+1}] Waveform: {example['waveform']}")
        print(f"  REFERENCE :\n{example['output']}")
        print(f"\n  GENERATED :\n{generated}")
        print("\n" + "-"*60)


if __name__ == "__main__":
    main()
