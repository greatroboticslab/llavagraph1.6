"""
finetune.py
===========
LoRA fine-tuning of Nemotron-H-8B-Base-8K on piezoelectric waveform
description data. Runs on A100 (80 GB) in bf16 — no 4-bit quantization,
so Mamba CUDA kernels run at full speed without any patching.

Key design choices:
  - bf16 only (no 4-bit): Mamba SSM CUDA kernels work normally
  - Response-only loss: model only learns to predict the answer, not the prompt
  - batch_size=4, grad_accum=4 -> effective batch 16, stable gradients
  - eval every 100 steps for fast feedback
  - adamw_torch_fused optimizer
  - EarlyStoppingCallback (patience=3) to stop automatically

Usage (via SLURM — see run_finetune.sbatch):
    python finetune.py --data_dir ./data --output_dir ./checkpoints
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    logging as hf_logging,
)
from trl import SFTConfig, SFTTrainer
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DataCollatorForCompletionOnlyLM:
    """Mask loss on everything before the response template."""
    response_template: str
    tokenizer: Any
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Strip any labels SFTTrainer may have pre-added — we recompute them.
        for f in features:
            f.pop("labels", None)

        # Manual padding (avoids tokenizer.pad() nesting issues)
        input_ids_list = [f["input_ids"] for f in features]
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        padded, masks = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded.append(ids + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)
        labels = input_ids.clone()

        # Find response template and mask prompt tokens
        tmpl_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        tlen = len(tmpl_ids)
        for i in range(labels.shape[0]):
            seq = labels[i].tolist()
            start = None
            for j in range(len(seq) - tlen):
                if seq[j: j + tlen] == tmpl_ids:
                    start = j + tlen
                    break
            labels[i, : (start if start is not None else len(seq))] = self.ignore_index

        labels[input_ids == pad_id] = self.ignore_index

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

hf_logging.set_verbosity_info()

MODEL_ID      = "nvidia/Nemotron-H-8B-Base-8K"
MAX_SEQ_LEN   = 1024
RESPONSE_TMPL = "### Response:\n"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(ex: dict) -> str:
    return (
        f"### System:\n{ex['system']}\n\n"
        f"### Instruction:\n{ex['input']}\n\n"
        f"### Response:\n{ex['output']}"
    )


def make_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_dict({"text": [format_example(e) for e in examples]})


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # flash_attention_2 applies to the transformer blocks inside Nemotron-H.
    # Falls back to eager if flash-attn is not installed.
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("  flash_attention_2 enabled for transformer blocks")
    except ImportError:
        print("  flash_attn not found — using eager attention")

    print(f"Loading {model_id} in bf16 (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded — {total:.1f}B parameters")
    return model, tokenizer


def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            # Transformer attention + feed-forward
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # Mamba SSM input/output projections
            "in_proj", "out_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Logging callback ──────────────────────────────────────────────────────────

class StepLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(
                f"  step {state.global_step:5d} | "
                f"loss {logs['loss']:.4f} | "
                f"lr {logs.get('learning_rate', 0):.2e}",
                flush=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--eval_steps", type=int,   default=100)
    parser.add_argument("--patience",   type=int,   default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data = load_jsonl(Path(args.data_dir) / "train.jsonl")
    val_data   = load_jsonl(Path(args.data_dir) / "val.jsonl")
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

    train_ds = make_dataset(train_data)
    val_ds   = make_dataset(val_data)

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = apply_lora(model)

    # Response-only loss: mask everything before "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TMPL,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[
            StepLogger(),
            EarlyStoppingCallback(early_stopping_patience=args.patience),
        ],
    )

    print(f"\nStarting training — effective batch size: {args.batch_size * args.grad_accum}")
    trainer.train()

    final_dir = Path(args.output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
