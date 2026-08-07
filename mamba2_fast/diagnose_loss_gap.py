"""
diagnose_loss_gap.py
=====================
Controlled experiment to find out WHY finetune_mamba2.py's logged
train_loss (~6.7 at convergence) sits so far above eval_loss (~1.7) —
normally these should be close, not 4x apart.

Leading hypothesis: GRAD_ACCUM=4 is inflating the *logged* training loss
(a known category of issue in some transformers versions), without
actually meaning the model performs worse on train than on eval data.

Method: load the already fine-tuned checkpoint (already converged, so its
loss on a training batch should be stable and representative) and run a
handful of steps with GRAD_ACCUM=1 instead of 4 — same model, same data,
only that one variable changed. If the logged loss drops to ~1.7-2 range,
the hypothesis is confirmed: it's a logging artifact, not a real
train/eval performance gap. If it stays ~6-7, something else is wrong.

Usage: python diagnose_loss_gap.py
"""

import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

MODEL_PATH = "/projects/ya4v/llavagraph1.6/mamba2_fast/checkpoints/final"
TRAIN_PATH = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/train.jsonl"
MAX_SEQ_LEN = 768
BATCH_SIZE = 4

print(f"Loading tokenizer + fine-tuned model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model.to("cuda" if torch.cuda.is_available() else "cpu")


def build_texts(ex):
    prompt_text = f"{ex['system']}\n\n{ex['input']}\n\n"
    return prompt_text, prompt_text + ex["output"]


class WaveformDataset(Dataset):
    def __init__(self, path):
        self.examples = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text, full_text = build_texts(ex)
        full_ids = tokenizer(full_text, add_special_tokens=False,
                             truncation=True, max_length=MAX_SEQ_LEN).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN).input_ids
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


@dataclass
class CompletionCollator:
    tokenizer: object
    max_length: int = MAX_SEQ_LEN

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            attention_masks.append(f["attention_mask"] + [0] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


train_dataset = WaveformDataset(TRAIN_PATH)

for grad_accum in [4, 1]:
    print("\n" + "=" * 60)
    print(f"TEST: gradient_accumulation_steps = {grad_accum}")
    print("=" * 60)

    args = TrainingArguments(
        output_dir="/tmp/loss_diag",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        learning_rate=0,          # freeze weights — we only want to READ the
                                    # loss number, not actually train further
        weight_decay=0.01,
        label_smoothing_factor=0.05,   # same as the real training run
        logging_steps=1,
        max_steps=5,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                       data_collator=CompletionCollator(tokenizer))
    trainer.train()
    logged = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"\nLogged loss values (grad_accum={grad_accum}): {logged}")

print("\n" + "=" * 60)
print("If the grad_accum=1 losses are close to eval_loss (~1.7-2.0) while")
print("grad_accum=4 losses are close to what training logged before (~6.7-9),")
print("the hypothesis is CONFIRMED: it's a logging artifact under gradient")
print("accumulation, not a real train/eval performance gap.")
print("=" * 60)
