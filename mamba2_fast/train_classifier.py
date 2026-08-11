"""
train_classifier.py
====================
Fine-tunes mamba2-780m-hf as a single-token defect-category classifier
instead of a paragraph generator. Reuses finetune_mamba2.py's proven
training setup (full-parameter, right-padding, early stopping, weight
decay, label smoothing — see that file's docstring for why each of those
choices was made) since that config is known to train this model
successfully; the only structural change is the target: one label
character instead of a multi-sentence answer, which sidesteps the
EOS-training problem entirely — there's no "when to stop" to learn,
generation is always exactly 1 token.

UPDATE (after first real eval run): the plain (unweighted) version trained
successfully — all 5 waveform types beat their majority-class baseline —
but per-class prediction counts showed pulse's C/D and noise's F were
never predicted at all (0 times), despite existing in the true labels.
square and ramp showed no such gap. That's the real majority-collapse
signal the plain-first approach was watching for, so class weighting is
now added: inverse-frequency weight per label token, computed from the
actual training set, capped at 5x (pulse's D has only 7 training examples
— its uncapped weight would be ~12x, which risks the opposite failure,
unstable training that overfits those 7 examples rather than learning a
generalizable pattern).

Data: data/{train,val}_cls.jsonl (see prepare_classifier_data.py)
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

MODEL_ID   = "AntonV/mamba2-780m-hf"
TRAIN_PATH = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/train_cls.jsonl"
VAL_PATH   = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/val_cls.jsonl"
OUTPUT_DIR = "/projects/ya4v/llavagraph1.6/mamba2_fast/checkpoints_cls_weighted3x"

MAX_SEQ_LEN = 768
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
LR          = 2e-5
EPOCHS      = 10
EARLY_STOP_PATIENCE = 3
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
WARMUP_RATIO = 0.05
SEED = 42
# 2nd attempt: cap=5.0 fixed noise's F and partially fixed pulse's C, but
# overcorrected on ramp (pushed it below its own majority baseline) and hurt
# sine — overall accuracy fell 78.6%->76.2%. Lower cap: ramp's minority
# classes sit at ~3.2-3.3x uncapped, so 3.0 barely touches them (should undo
# most of that regression) while still meaningfully upweighting the truly
# rare classes (pulse/noise's rarest were 5-12x uncapped).
CLASS_WEIGHT_CAP = 3.0

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params/1e6:.0f}M (full fine-tune — all trainable)")


def build_prompt(ex):
    return f"{ex['system']}\n\n{ex['input']}\n\nCategory:"


class ClassifierDataset(Dataset):
    def __init__(self, path):
        self.examples = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text = build_prompt(ex)
        label_id = tokenizer(ex["label"], add_special_tokens=False).input_ids
        assert len(label_id) == 1, f"label {ex['label']!r} is not a single token"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN - 1).input_ids
        full_ids = prompt_ids + label_id
        labels = [-100] * len(prompt_ids) + label_id

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class PadCollator:
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


train_dataset = ClassifierDataset(TRAIN_PATH)
val_dataset = ClassifierDataset(VAL_PATH)
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

# ── class weights (inverse frequency, capped) ────────────────────────────────
# Computed from the actual training set label ids so this stays correct if
# the data changes, rather than hardcoding the numbers worked out by hand.
label_id_counts = Counter()
for ex in train_dataset.examples:
    tid = tokenizer(ex["label"], add_special_tokens=False).input_ids[0]
    label_id_counts[tid] += 1

vocab_size = model.get_input_embeddings().weight.shape[0]
class_weights = torch.ones(vocab_size, dtype=torch.float32)
N, K = len(train_dataset), len(label_id_counts)
for tid, count in label_id_counts.items():
    class_weights[tid] = min(N / (K * count), CLASS_WEIGHT_CAP)
print(f"Class weights computed for {K} label tokens "
      f"(range {class_weights[list(label_id_counts)].min():.2f}-"
      f"{class_weights[list(label_id_counts)].max():.2f}, cap={CLASS_WEIGHT_CAP})")


class WeightedTrainer(Trainer):
    """Overrides the default loss to apply per-class weights. Necessary
    because HF's built-in label_smoothing_factor path (LabelSmoother) has
    no support for class weights, so this replicates it manually with
    plain nn.functional.cross_entropy, which supports both at once."""

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Mamba2ForCausalLM's own loss shifts internally (see its docs);
        # replicating that shift here since compute_loss bypasses it.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=torch.cuda.is_available(),
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=PadCollator(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    class_weights=class_weights,
    label_smoothing=LABEL_SMOOTHING,
)

print("\nStarting fine-tuning...")
trainer.train()

# Same GRAD_ACCUM logging correction confirmed necessary for the generation
# track (diagnose_loss_gap.py) — applies identically here since it's the
# same transformers version / same accumulation mechanism, not specific to
# what the labels look like.
train_losses = [(h["epoch"], h["loss"] / GRAD_ACCUM) for h in trainer.state.log_history if "loss" in h]
eval_losses = [(h["epoch"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]

results_dir = Path("/projects/ya4v/llavagraph1.6/mamba2_fast/results")
results_dir.mkdir(exist_ok=True)
with open(results_dir / "training_history_cls_weighted3x.csv", "w") as f:
    f.write("epoch,train_loss,eval_loss\n")
    for ep, ev in eval_losses:
        nearest_train = min((t for t in train_losses if t[0] <= ep),
                             key=lambda t: abs(t[0] - ep), default=None)
        tr = nearest_train[1] if nearest_train else ""
        f.write(f"{ep},{tr},{ev}\n")

print("\n" + "=" * 60)
print("FIT DIAGNOSTIC")
print("=" * 60)
for ep, ev in eval_losses:
    nearest_train = min((t for t in train_losses if t[0] <= ep), key=lambda t: abs(t[0]-ep), default=None)
    tr_str = f"{nearest_train[1]:.4f}" if nearest_train else "n/a"
    print(f"{ep:>6.1f}  train={tr_str:>10}  eval={ev:>10.4f}")
print("=" * 60)

final_path = Path(OUTPUT_DIR) / "final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"\nFine-tuned classifier saved to: {final_path}")
