"""
inference.py
============
Run the fine-tuned Nemotron-H-8B LoRA adapter on a data split.
Loads the model in bf16 (no quantization) so Mamba CUDA kernels
run at full speed, matching the training setup.

Saves results to results/predictions_{split}.jsonl — one JSON per sample
containing the reference text, generated text, and inference time (ms).

Usage (via SLURM — see run_inference.sbatch):
    python inference.py --adapter_dir ./checkpoints/final_adapter
    python inference.py --adapter_dir ./checkpoints/final_adapter --split val --n_samples 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID       = "nvidia/Nemotron-H-8B-Base-8K"
MAX_NEW_TOKENS = 512


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(ex: dict) -> str:
    return (
        f"### System:\n{ex['system']}\n\n"
        f"### Instruction:\n{ex['input']}\n\n"
        f"### Response:\n"
    )


def load_model(adapter_dir: str):
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def run_inference(model, tokenizer, samples: list[dict]) -> list[dict]:
    results  = []
    total_ms = 0.0

    for i, ex in enumerate(samples):
        prompt = format_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms  += elapsed_ms

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "idx":       i,
            "waveform":  ex.get("waveform", "unknown"),
            "reference": ex["output"],
            "generated": generated,
            "ms":        round(elapsed_ms, 2),
        })

        if i == 0:
            print("--- First sample preview ---")
            print(f"[REFERENCE]\n{ex['output'][:300]}...\n")
            print(f"[GENERATED]\n{generated[:300]}...\n")
            print("-" * 60)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:4d}/{len(samples)}]  avg {total_ms/(i+1):.1f} ms/sample",
                  flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="./checkpoints/final_adapter")
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--output_dir",  default="./results")
    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_samples",   type=int, default=None)
    parser.add_argument("--start_idx",   type=int, default=0)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.adapter_dir)

    data    = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")
    end     = args.start_idx + args.n_samples if args.n_samples else len(data)
    samples = data[args.start_idx:end]
    print(f"Running inference on {len(samples)} {args.split} samples\n")

    results  = run_inference(model, tokenizer, samples)
    out_path = Path(args.output_dir) / f"predictions_{args.split}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    avg_ms = sum(r["ms"] for r in results) / len(results)
    print(f"\nDone: {len(results)} samples | avg {avg_ms:.1f} ms/sample")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
