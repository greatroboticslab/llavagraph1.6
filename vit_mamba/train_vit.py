#!/usr/bin/env python3
"""
train_vit.py  —  ViT (google/vit-base-patch16-224-in21k) fine-tuning
                 for 5-class piezo waveform classification.

Uses ViTModel (backbone only) + custom head to avoid key-naming
mismatches introduced in transformers 5.x when using
ViTForImageClassification.from_pretrained directly.

Usage:
    python train_vit.py
    python train_vit.py --epochs 50 --batch_size 32 --strategy full
"""

import sys
import os
import argparse
import time
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from transformers import ViTModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VIT_MODEL_ID  = "google/vit-base-patch16-224-in21k"


# ── Model ─────────────────────────────────────────────────────────────────────

class ViTClassifier(nn.Module):
    def __init__(self, backbone: ViTModel, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.config.hidden_size  # 768 for vit-base
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor):
        # CLS token from last hidden state → (B, 768)
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.head(cls)


def build_vit_model(strategy: str = "full", unfreeze_last_n: int = 2,
                    dropout: float = 0.3) -> ViTClassifier:
    print(f"\nLoading backbone: {VIT_MODEL_ID}")
    backbone = ViTModel.from_pretrained(VIT_MODEL_ID, add_pooling_layer=False)
    print("  Using ViT backbone from HuggingFace (ViTModel)")

    model = ViTClassifier(backbone, num_classes=cfg.NUM_CLASSES, dropout=dropout)
    _apply_strategy(model, strategy, unfreeze_last_n)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters → total: {total/1e6:.1f}M  "
          f"trainable: {trainable/1e6:.1f}M  "
          f"frozen: {(total-trainable)/1e6:.1f}M")
    return model


def _apply_strategy(model: ViTClassifier, strategy: str, unfreeze_last_n: int):
    if strategy == "full":
        for p in model.parameters():
            p.requires_grad = True
        print("  Fine-tune strategy: FULL  (all parameters trainable)")

    elif strategy == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        print("  Fine-tune strategy: HEAD_ONLY  (backbone frozen)")

    else:  # partial
        for p in model.parameters():
            p.requires_grad = False

        # Auto-detect encoder layer list (name changed across transformers versions)
        enc_layers = None
        for attr_path in ("encoder.layer", "layers", "encoder.layers"):
            obj = model.backbone
            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                enc_layers = obj
                break

        if enc_layers is not None:
            n = len(enc_layers)
            for block in list(enc_layers)[max(0, n - unfreeze_last_n):]:
                for p in block.parameters():
                    p.requires_grad = True
            # Also unfreeze the final layernorm
            ln = getattr(model.backbone, "layernorm", None)
            if ln is not None:
                for p in ln.parameters():
                    p.requires_grad = True
            print(f"  Fine-tune strategy: PARTIAL  "
                  f"(last {unfreeze_last_n}/{n} encoder blocks + head)")
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("  Fine-tune strategy: PARTIAL requested but layers not found "
                  "— falling back to FULL")

        for p in model.head.parameters():
            p.requires_grad = True


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_transform(split: str):
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class WaveformDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.transform = get_transform(split)
        self.paths, self.labels = [], []
        for i, cls in enumerate(cfg.CLASSES):
            cls_dir = os.path.join(root, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith(".png"):
                    self.paths.append(os.path.join(cls_dir, f))
                    self.labels.append(i)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, split, use_amp=False):
    is_train = split == "train"
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {split:5s}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += bs

    n = max(total, 1)
    return {"loss": total_loss / n, "acc": correct / n}


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    print("=" * 60)
    print(f"  ViT — Piezo Waveform Training")
    print(f"  Data    : {cfg.DATA_DIR}")
    print(f"  Device  : {args.device}")
    print(f"  Epochs  : {args.epochs}  BS: {args.batch_size}  LR: {args.lr}")
    print(f"  Strategy: {args.strategy}")
    print("=" * 60)

    train_ds = WaveformDataset(cfg.DATA_DIR, "train")
    val_ds   = WaveformDataset(cfg.DATA_DIR, "val")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    model = build_vit_model(strategy=args.strategy).to(args.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith("head.") and p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = AdamW([{"params": backbone_params, "lr": args.lr * 0.1},
                       {"params": head_params,     "lr": args.lr}],
                      weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    if use_amp:
        print("  AMP: enabled (bfloat16)")

    best_acc, patience = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer,
                       args.device, "train", use_amp)
        vl = run_epoch(model, val_loader,   criterion, None,
                       args.device, "val",   use_amp)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f}  |  "
              f"val_loss={vl['loss']:.4f} acc={vl['acc']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss",   vl["loss"]), ("val_acc",   vl["acc"])]:
            history[k].append(v)

        if vl["acc"] > best_acc:
            best_acc, patience = vl["acc"], 0
            torch.save({
                "epoch":   epoch,
                "val_acc": vl["acc"],
                "model":   model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config":  {"model_id": VIT_MODEL_ID,
                            "num_classes": cfg.NUM_CLASSES,
                            "classes": cfg.CLASSES},
            }, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  ✓ best val_acc={best_acc:.4f} saved")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                print("  Early stopping.")
                break

    print(f"\n  Best val_acc = {best_acc:.4f}")
    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    _save_curves(history, args.results_dir)
    return best_acc


def _save_curves(history, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(e, history["train_loss"], label="train")
        axes[0].plot(e, history["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(e, history["train_acc"], label="train")
        axes[1].plot(e, history["val_acc"],   label="val")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
        plt.close()
    except Exception as ex:
        print(f"[warn] curves: {ex}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=cfg.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--strategy",    default="partial",
                   choices=["full", "head_only", "partial"])
    p.add_argument("--num_workers", type=int,   default=cfg.NUM_WORKERS)
    p.add_argument("--device",      default=cfg.DEVICE)
    p.add_argument("--results_dir", default=cfg.RESULTS_DIR_VIT)
    p.add_argument("--ckpt_dir",    default=cfg.CKPT_DIR_VIT)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
