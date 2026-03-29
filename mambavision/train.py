"""
train.py
--------
Train MambaVision for piezo waveform classification.

Usage:
    python train.py [--model_id nvidia/MambaVision-T-1K]
                    [--epochs 30]
                    [--batch_size 32]
                    [--lr 3e-4]
                    [--strategy full|head_only|partial]
                    [--resume checkpoints/best.pth]

All defaults come from config.py.
"""

import os
import sys
import argparse
import logging
import time
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                       StepLR,
                                       ReduceLROnPlateau)
from tqdm import tqdm

import config
from dataset import build_dataloaders, compute_class_weights, discover_files
from model import build_model, save_checkpoint, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Warmup scheduler wrapper
# ──────────────────────────────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Linear warm-up for the first `warmup_epochs`, then hand off to `base_scheduler`.
    """
    def __init__(self, optimizer, warmup_epochs: int, base_scheduler):
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.base_scheduler  = base_scheduler
        self.base_lrs        = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch          = 0

    def step(self, val_loss=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            alpha = self._epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * alpha
        else:
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                self.base_scheduler.step(val_loss)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────────────────────
# One epoch of training / validation
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model: nn.Module,
              loader,
              criterion: nn.Module,
              optimizer=None,
              device: str = config.DEVICE,
              split: str = "train") -> Dict[str, float]:
    """
    Run one full pass over the dataset.

    Returns dict with keys: loss, acc, top1, top3
    """
    is_train = (split == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct1   = 0
    correct3   = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        bar = tqdm(loader, desc=f"  {split:5s}", leave=False, unit="batch")
        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total      += bs

            # Top-1 and Top-3 accuracy
            _, pred1 = logits.topk(1, dim=1)
            correct1 += pred1.squeeze(1).eq(labels).sum().item()

            k = min(3, config.NUM_CLASSES)
            _, pred3 = logits.topk(k, dim=1)
            correct3 += pred3.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            bar.set_postfix(loss=f"{loss.item():.4f}")

    n = max(total, 1)
    return {
        "loss": total_loss / n,
        "acc":  correct1  / n,
        "top3": correct3  / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    logger = setup_logger(config.LOG_FILE)
    logger.info("=" * 60)
    logger.info("  Piezo Waveform Classifier – Training")
    logger.info("=" * 60)
    logger.info(f"  Model      : {args.model_id}")
    logger.info(f"  Strategy   : {args.strategy}")
    logger.info(f"  Device     : {args.device}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Batch size : {args.batch_size}")
    logger.info(f"  LR         : {args.lr}")
    logger.info("=" * 60)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    split_json = os.path.join(config.RESULTS_DIR, "split.json")
    loaders = build_dataloaders(
        config.DATA_DIR,
        save_split_json=split_json
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # Class weights (for imbalanced datasets)
    # Get labels from the training dataset
    train_labels = loaders["train"].dataset.labels
    class_weights = compute_class_weights(train_labels).to(args.device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_id=args.model_id,
        finetune_strategy=args.strategy
    ).to(args.device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use different LRs for backbone vs. head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # backbone: 10× smaller
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=config.WEIGHT_DECAY)

    if args.strategy == "head_only":
        # Single LR is fine when backbone is frozen
        optimizer = AdamW(head_params, lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    steps_after_warmup = args.epochs - config.WARMUP_EPOCHS
    if config.LR_SCHEDULER == "cosine":
        base_sched = CosineAnnealingLR(optimizer, T_max=steps_after_warmup, eta_min=1e-6)
    elif config.LR_SCHEDULER == "step":
        base_sched = StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    else:   # plateau
        base_sched = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    scheduler = WarmupScheduler(optimizer, config.WARMUP_EPOCHS, base_sched)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_val_acc = load_checkpoint(
            args.resume, model, optimizer, device=args.device
        )
        start_epoch += 1
        logger.info(f"  Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        logger.info(f"\nEpoch [{epoch+1:03d}/{args.epochs}]")

        train_metrics = run_epoch(model, train_loader, criterion,
                                   optimizer, args.device, split="train")
        val_metrics   = run_epoch(model, val_loader,   criterion,
                                   device=args.device, split="val")

        # Step scheduler
        scheduler.step(val_loss=val_metrics["loss"])
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        logger.info(
            f"  train_loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['acc']:.4f}  |  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  "
            f"val_top3={val_metrics['top3']:.4f}  |  "
            f"lr={current_lr:.2e}  elapsed={elapsed:.1f}s"
        )

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Save best checkpoint
        val_acc = val_metrics["acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)
            logger.info(f"  ✓ New best val_acc={best_val_acc:.4f}  "
                        f"→ saved to {ckpt_path}")
        else:
            patience_counter += 1

        # Also save latest
        last_path = os.path.join(config.CHECKPOINT_DIR, "last.pth")
        save_checkpoint(model, optimizer, epoch, val_acc, last_path)

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            logger.info(f"\n  Early stopping triggered after "
                        f"{config.EARLY_STOP_PATIENCE} epochs without improvement.")
            break

    logger.info(f"\n  Training complete. Best val_acc = {best_val_acc:.4f}")

    # Save training history for plotting
    import json
    hist_path = os.path.join(config.RESULTS_DIR, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"  History saved to {hist_path}")

    _plot_history(history, config.RESULTS_DIR)
    return model, history


def _plot_history(history: dict, out_dir: str):
    """Save loss and accuracy curves."""
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], label="train")
        axes[1].plot(epochs, history["val_acc"],   label="val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training curves saved to {path}")
    except Exception as e:
        print(f"  [WARN] Could not save training curves: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaVision on piezo waveforms.")
    parser.add_argument("--model_id", default=config.MODEL_ID,
                        help="HuggingFace model ID for MambaVision.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--strategy", default=config.FINETUNE_STRATEGY,
                        choices=["full", "head_only", "partial"])
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Sync batch_size / model_id with module-level config for dataloaders
    config.BATCH_SIZE = args.batch_size
    config.MODEL_ID   = args.model_id
    train(args)
