"""
finetune_mamba2.py
===================
Full-parameter fine-tuning of AntonV/mamba2-780m-hf on the short-format
piezoelectric waveform diagnostic data (mamba2_fast/data/).

Method: full fine-tuning, not LoRA. At 780M parameters this is cheap enough
on an A100 (~9-10GB for weights+gradients+Adam states in bf16/fp32 mix) that
full fine-tuning is affordable, and it avoids relying on PEFT/LoRA support
for the Mamba2 architecture, which is newer and less battle-tested than
LoRA-on-attention-layers. If GPU memory becomes a constraint, switch to
LoRA with target_modules=["in_proj", "out_proj"] (the Mamba2 mixer's linear
projections) — see the transformers Mamba2 docs.

Unlike finetune_zamba.py, there is no chat template here: this checkpoint
is a BASE model (confirmed by sanity_check.py), not instruction-tuned, so
prompts are built as plain concatenated text instead of
tokenizer.apply_chat_template().

Data: 1468 train / 490 val samples from ./data/ (see prepare_data.py)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ── paths ─────────────────────────────────────────────────────────────────────

MODEL_ID    = "AntonV/mamba2-780m-hf"
TRAIN_PATH  = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/train.jsonl"
VAL_PATH    = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/val.jsonl"
OUTPUT_DIR  = "/projects/ya4v/llavagraph1.6/mamba2_fast/checkpoints"

# ── hyperparameters ───────────────────────────────────────────────────────────

MAX_SEQ_LEN = 768   # inputs here are shorter than Zamba2's (no few-shot example)
BATCH_SIZE  = 4
GRAD_ACCUM  = 4      # effective batch = 16
LR          = 2e-5   # full fine-tune, not LoRA — an order of magnitude lower
                      # than finetune_zamba.py's 2e-4 LoRA rate, since every
                      # parameter moves now, not just a small adapter
EPOCHS      = 10      # ceiling, not a target — early stopping (below) picks
                      # the actual stopping point from val loss. At 780M
                      # params vs. 1468 train examples, letting this run the
                      # full 10 epochs unconditionally would very likely
                      # overfit/memorize well before reaching it.
EARLY_STOP_PATIENCE = 3   # stop after 3 epochs with no val-loss improvement
WEIGHT_DECAY = 0.01       # L2 penalty on the weights — standard full-finetune
                          # default, pulls parameters back toward the
                          # pretrained values instead of letting them drift
                          # freely to fit the 1468 training examples exactly
LABEL_SMOOTHING = 0.05    # same value finetune_zamba.py uses — softens the
                          # training targets so the model isn't pushed to
                          # 100% confidence on any single token, which is
                          # itself a form of overfitting on small data
WARMUP_RATIO= 0.05
SEED        = 42

# ── load tokenizer & model ────────────────────────────────────────────────────

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# NOTE: training the model to emit EOS at the end of the target (via an
# appended eos_token_id + left-padding, per Mamba2's own docs) was tried
# twice and both times collapsed training into predicting EOS as literally
# the first generated token, despite the training data itself being
# verified correct (see DEBUG block below). Root cause unresolved — rather
# than keep guessing at training-time fixes, "when to stop" is instead
# handled at generation time in eval_mamba2.py (trim at the first blank
# line), which is proven to work on top of this exact training config.

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params/1e6:.0f}M (full fine-tune — all trainable)")


# ── dataset ───────────────────────────────────────────────────────────────────
# Plain text concatenation (no chat template — base model).
#   prompt_text = system + input, ending right before the answer
#   full_text   = prompt_text + output
# Labels are -100 over the prompt so loss is only computed on the generated
# DIAGNOSIS/CORRECTION text, exactly like finetune_zamba.py's collator does.

def build_texts(ex: dict):
    prompt_text = f"{ex['system']}\n\n{ex['input']}\n\n"
    full_text = prompt_text + ex["output"]
    return prompt_text, full_text


class WaveformDataset(Dataset):
    def __init__(self, path):
        self.examples = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text, full_text = build_texts(ex)

        full_ids   = tokenizer(full_text,   add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN).input_ids

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if len(labels) > MAX_SEQ_LEN:
            labels = labels[:MAX_SEQ_LEN]

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }


@dataclass
class CompletionCollator:
    tokenizer: object
    max_length: int = MAX_SEQ_LEN

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        pad_id  = self.tokenizer.pad_token_id

        input_ids, attention_masks, labels = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            attention_masks.append(f["attention_mask"] + [0] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)

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
    weight_decay=WEIGHT_DECAY,
    label_smoothing_factor=LABEL_SMOOTHING,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=torch.cuda.is_available(),
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # NOTE: selects the checkpoint with lowest next-token-prediction loss,
    # not the one with the best BLEU-1/ROUGE-L. That's a real gap (same one
    # finetune_zamba.py has) — Trainer's built-in loop can't easily compute
    # generation-based metrics for a plain causal LM. After training, run
    # eval_mamba2.py (task 5) on the last 1-2 saved checkpoints and pick by
    # actual BLEU-1/ROUGE-L if they disagree with the eval_loss ranking.
    metric_for_best_model="eval_loss",
    greater_is_better=False,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

print("\nStarting fine-tuning...")
trainer.train()

# ── over/underfitting diagnostic ────────────────────────────────────────────
# Read train_loss and eval_loss back out of the Trainer's own log history so
# the fit quality is visible at a glance, without having to parse the raw
# stdout log by hand.

# CONFIRMED via diagnose_loss_gap.py (2026-08-06): this transformers version
# logs training loss inflated by ~GRAD_ACCUM before averaging/reporting —
# same model + same data, only grad_accum changed 4->1, dropped the logged
# loss from ~6.8 to ~1.7 (matching eval_loss almost exactly). So the raw
# logged "loss" is divided back down here to get the true per-token loss.
train_losses = [(h["epoch"], h["loss"] / GRAD_ACCUM) for h in trainer.state.log_history if "loss" in h]
eval_losses  = [(h["epoch"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]

# Save as plain CSV too — make_figures.py (and any other plotting) reads this
# instead of scraping the stdout log, which is fragile (log format can change,
# and the values here are already the exact numbers Trainer computed).
results_dir = Path("/projects/ya4v/llavagraph1.6/mamba2_fast/results")
results_dir.mkdir(exist_ok=True)
with open(results_dir / "training_history.csv", "w") as f:
    f.write("epoch,train_loss,eval_loss\n")
    for ep, ev in eval_losses:
        nearest_train = min((t for t in train_losses if t[0] <= ep),
                             key=lambda t: abs(t[0] - ep), default=None)
        tr = nearest_train[1] if nearest_train else ""
        f.write(f"{ep},{tr},{ev}\n")

print("\n" + "=" * 60)
print("FIT DIAGNOSTIC")
print("=" * 60)
print(f"{'epoch':>6}  {'train_loss':>12}  {'eval_loss':>12}")
for ep, ev in eval_losses:
    nearest_train = min((t for t in train_losses if t[0] <= ep), key=lambda t: abs(t[0]-ep), default=None)
    tr_str = f"{nearest_train[1]:.4f}" if nearest_train else "n/a"
    print(f"{ep:>6.1f}  {tr_str:>12}  {ev:>12.4f}")

if len(eval_losses) >= 2:
    first_gap = train_losses[0][1] - eval_losses[0][1] if train_losses else None
    last_ev   = eval_losses[-1][1]
    best_ev   = min(ev for _, ev in eval_losses)
    if last_ev > best_ev * 1.05:
        print("\n  -> eval_loss rose noticeably after its best point: classic "
              "overfitting shape. Early stopping should have already rolled "
              "back to the best checkpoint (load_best_model_at_end=True) — "
              "confirm the saved 'final' model matches the best epoch above, "
              "not the last one.")
    elif eval_losses[-1][1] > eval_losses[0][1] * 0.9 and len(eval_losses) == EPOCHS:
        print("\n  -> Hit the 10-epoch ceiling AND eval_loss was still "
              "improving at the end (never triggered early stopping): "
              "this is the underfitting signature — the model likely hadn't "
              "finished learning. Consider raising EPOCHS and re-running.")
    else:
        print("\n  -> eval_loss decreased and then plateaued/stopped "
              "improving — looks like a normal, healthy convergence.")
print("=" * 60)

# ── save ──────────────────────────────────────────────────────────────────────

final_path = Path(OUTPUT_DIR) / "final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"\nFine-tuned model saved to: {final_path}")
