"""
finetune.py
===========
QLoRA fine-tuning of Nemotron-H-47B-Reasoning-128K on piezoelectric
waveform description data.

Usage (via SLURM — see submit_finetune.sh):
    python finetune.py --data_dir ./data --output_dir ./checkpoints
"""

import argparse
import json
import os
import math
import types
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    logging as hf_logging,
)
from trl import SFTTrainer, SFTConfig

hf_logging.set_verbosity_info()

# Auto-approve trust_remote_code in non-interactive (SLURM) environments
def _patch_trust_remote_code():
    try:
        import transformers.dynamic_module_utils as _dmu
        _orig = _dmu.resolve_trust_remote_code
        def _auto(trust_remote_code, model_name, has_local_code, has_remote_code):
            if has_remote_code and trust_remote_code is None:
                return True
            return _orig(trust_remote_code, model_name, has_local_code, has_remote_code)
        _dmu.resolve_trust_remote_code = _auto
    except Exception:
        pass

_patch_trust_remote_code()

MODEL_ID = "nvidia/Nemotron-H-8B-Base-8K"
MAX_SEQ_LEN = 1024  # v3: full-feature inputs; p90 ≈ 1024 tokens, fits with grad checkpointing

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(example: dict) -> str:
    """
    Format a single example into the instruction template the model will train on.
    The model learns to predict everything after '### Response:\n'.
    """
    return (
        f"### System:\n{example['system']}\n\n"
        f"### Instruction:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def make_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_dict({"text": [format_example(e) for e in examples]})


# ─────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit NF4 for all layers (~4GB). No gradient checkpointing needed at this size.
    print("Loading model in 4-bit NF4 (all layers)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # The mixer forward checks module-level `is_fast_path_available` (not an instance attr),
    # so we can't patch it via setattr. Instead replace each mixer's forward method directly
    # to always call torch_forward, which uses self.out_proj(x) compatible with bitsandbytes 4-bit.
    def _torch_forward_only(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

    n_patched = 0
    for module in model.modules():
        if hasattr(module, 'torch_forward') and hasattr(module, 'cuda_kernels_forward'):
            module.forward = types.MethodType(_torch_forward_only, module)
            n_patched += 1
    print(f"  Patched {n_patched} Mamba mixer(s): forcing torch_forward")

    # Gradient checkpointing required: torch_forward creates large intermediate SSM tensors.
    # With grad checkpointing, only 1 layer's activations in memory at a time (~4GB total).
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    return model, tokenizer




def apply_lora(model) -> object:
    """
    Apply LoRA to the model.
    Targeting both transformer attention layers and Mamba SSM projection layers
    to cover Nemotron-H's hybrid architecture.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            # Transformer attention layers only (Mamba SSM layers have non-standard shapes)
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    return model


# ─────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "")
            lr   = logs.get("learning_rate", "")
            print(f"  step {step:5d} | loss {loss:.4f} | lr {lr:.2e}" if loss else "", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--output_dir",  default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=16)  # effective batch = 16
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--warmup_ratio",type=float, default=0.05)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────
    train_data = load_jsonl(Path(args.data_dir) / "train.jsonl")
    val_data   = load_jsonl(Path(args.data_dir) / "val.jsonl")
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

    train_dataset = make_dataset(train_data)
    val_dataset   = make_dataset(val_data)

    # ── Load model ─────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = apply_lora(model)

    # ── Training args ──────────────────────────────────────────
    total_steps = math.ceil(len(train_data) / (args.batch_size * args.grad_accum)) * args.epochs
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=0.3,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
    )

    # ── Trainer ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[LogCallback()],
    )

    print(f"\nStarting training — {total_steps} total steps")
    trainer.train()

    # ── Save final adapter ──────────────────────────────────────
    final_dir = Path(args.output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
