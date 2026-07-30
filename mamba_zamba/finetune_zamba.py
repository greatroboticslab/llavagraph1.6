"""
finetune_zamba.py
=================
LoRA fine-tuning of Zamba2-7B-instruct on piezoelectric waveform diagnostic data.
Trains the model to generate DIAGNOSIS + CORRECTION + CORRECTION VECTOR from
measured waveform features.

Architecture: Zamba2-7B-instruct (Mamba-2 + Transformer hybrid, Zyphra)
Method: LoRA (PEFT) — only attention/MLP projection layers are adapted
Data: 1468 train / 490 val samples from mamba_llm/data/
"""

import warnings
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

# ── Zamba2 weight-tying monkey-patch ─────────────────────────────────────────
# Must be applied before any transformers import that touches PreTrainedModel.
from transformers import PreTrainedModel
_orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys

def _patched_get_expanded(self, all_submodels=True):
    try:
        return _orig_get_expanded(self, all_submodels=all_submodels)
    except ValueError as e:
        warnings.warn(f"Weight tying validation skipped: {e}", stacklevel=2)
        return {}

PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── paths ─────────────────────────────────────────────────────────────────────

MODEL_ID    = "Zyphra/Zamba2-7B-instruct"
TRAIN_PATH  = "/projects/ya4v/llavagraph1.6/mamba_llm/data/train.jsonl"
VAL_PATH    = "/projects/ya4v/llavagraph1.6/mamba_llm/data/val.jsonl"
OUTPUT_DIR  = "/projects/ya4v/llavagraph1.6/mamba_zamba/checkpoints"

# ── hyperparameters ───────────────────────────────────────────────────────────

MAX_SEQ_LEN = 1024
LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROP   = 0.05
BATCH_SIZE  = 2
GRAD_ACCUM  = 8        # effective batch = 16
LR          = 2e-4
EPOCHS      = 3
WARMUP_RATIO= 0.05
SEED        = 42

# ── load tokenizer & model ────────────────────────────────────────────────────

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ── fix Zamba2 shared_transformer weight tying ────────────────────────────────
# The monkey-patch bypasses validation but doesn't perform the actual tying.
# This manually shares the Python object so tied layers use the same weights.

def fix_zamba2_tying(model):
    tied = getattr(model.model, '_tied_weights_keys', {})
    if not tied:
        print("  No tied weights found — skipping tying fix.")
        return
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
    print(f"  Weight tying fixed: {fixed} layers now share shared_transformer objects.")

fix_zamba2_tying(model)
model.enable_input_require_grads()

# ── apply LoRA ────────────────────────────────────────────────────────────────
# Target the transformer attention and MLP linear layers inside shared_transformer.
# Mamba SSM layers (in_proj, out_proj, x_proj) are left frozen.

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROP,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── dataset ───────────────────────────────────────────────────────────────────

def build_conversation(ex):
    """Format one example as chat messages and return (prompt_text, full_text)."""
    messages = [
        {"role": "system",    "content": ex["system"]},
        {"role": "user",      "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt only (no assistant turn) — used to compute where labels begin
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text, full_text


class WaveformDataset(Dataset):
    def __init__(self, path):
        self.examples = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text, full_text = build_conversation(ex)

        full_ids   = tokenizer(full_text,   add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN).input_ids

        # Labels: -100 for prompt tokens (don't train on them), actual ids for response
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Ensure same length after truncation
        if len(labels) > MAX_SEQ_LEN:
            labels = labels[:MAX_SEQ_LEN]

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }


# ── data collator ─────────────────────────────────────────────────────────────

@dataclass
class CompletionCollator:
    tokenizer: object
    max_length: int = MAX_SEQ_LEN

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        input_ids      = []
        attention_masks = []
        labels         = []
        pad_id         = self.tokenizer.pad_token_id

        for f in features:
            seq   = f["input_ids"]
            attn  = f["attention_mask"]
            lbl   = f["labels"]
            pad_n = max_len - len(seq)

            input_ids.append(seq + [pad_id] * pad_n)
            attention_masks.append(attn + [0] * pad_n)
            labels.append(lbl + [-100] * pad_n)

        return {
            "input_ids":      torch.tensor(input_ids,       dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels":         torch.tensor(labels,          dtype=torch.long),
        }


# ── training ──────────────────────────────────────────────────────────────────

train_dataset = WaveformDataset(TRAIN_PATH)
val_dataset   = WaveformDataset(VAL_PATH)
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=CompletionCollator(tokenizer),
)

print("\nStarting fine-tuning...")
trainer.train()

# ── save LoRA adapter ─────────────────────────────────────────────────────────

adapter_path = Path(OUTPUT_DIR) / "lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nLoRA adapter saved to: {adapter_path}")
