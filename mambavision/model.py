"""
model.py
--------
Build a MambaVision model fine-tuned for piezo waveform classification.

Three fine-tuning strategies are supported (set in config.py):
  • "full"      – update all parameters (recommended for small datasets too,
                  with a low LR)
  • "head_only" – freeze the entire backbone; only train the new classifier
  • "partial"   – freeze early stages, train the last N stages + classifier

Architecture:
  MambaVision backbone (HuggingFace AutoModel)
      │
      ▼
  Global Average Pooling  (already done by the backbone → 1-D feature vector)
      │
      ▼
  Dropout → Linear(feat_dim, NUM_CLASSES)

Note: If MambaVision dependencies (mamba-ssm) are not available, falls back to
      a Vision Transformer (vit_small_patch16_224) from timm.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import config


# ──────────────────────────────────────────────────────────────────────────────
# Custom classifier head
# ──────────────────────────────────────────────────────────────────────────────

class MambaVisionClassifier(nn.Module):
    """
    MambaVision backbone + lightweight classification head.

    Args:
        backbone:    HuggingFace AutoModel (returns (avg_pool_feats, stage_feats))
        feat_dim:    dimension of the avg-pool output vector
        num_classes: number of target classes
        dropout:     dropout probability before the linear layer
    """

    def __init__(self,
                 backbone: nn.Module,
                 feat_dim: int,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (B, C, H, W) float tensor, normalised

        Returns:
            logits: (B, num_classes)
        """
        # MambaVision AutoModel returns (avg_pool_feats, list_of_stage_feats)
        # timm ViT returns just the features
        out = self.backbone(pixel_values)
        if isinstance(out, (tuple, list)):
            avg_pool_feats, _ = out
        else:
            avg_pool_feats = out
        logits = self.head(avg_pool_feats)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_id: str = config.MODEL_ID,
                pretrained: bool = config.PRETRAINED,
                finetune_strategy: str = config.FINETUNE_STRATEGY,
                unfreeze_last_n: int = config.UNFREEZE_LAST_N_STAGES,
                dropout: float = 0.4) -> MambaVisionClassifier:
    """
    Load MambaVision from HuggingFace and attach the custom classifier head.
    Falls back to timm's ViT if MambaVision dependencies are unavailable.

    Args:
        model_id:           HuggingFace model identifier.
        pretrained:         Whether to load ImageNet pre-trained weights.
        finetune_strategy:  "full" | "head_only" | "partial"
        unfreeze_last_n:    For "partial" strategy, number of final stages to unfreeze.
        dropout:            Dropout rate for the classifier head.

    Returns:
        MambaVisionClassifier instance
    """
    print(f"\nLoading backbone: {model_id}  (pretrained={pretrained})")

    try:
        if pretrained:
            backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            backbone = AutoModel.from_config(cfg, trust_remote_code=True)
        print("  Using MambaVision backbone from HuggingFace")
    except ImportError as e:
        print(f"  MambaVision not available ({e}), falling back to timm ViT")
        import timm
        backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
    except Exception as e:
        print(f"  Failed to load MambaVision ({e}), falling back to timm ViT")
        import timm
        backbone = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)

    # Infer feature dimension from config or a dummy forward pass
    feat_dim = _infer_feat_dim(backbone)
    print(f"  Backbone feature dim: {feat_dim}")

    model = MambaVisionClassifier(backbone, feat_dim=feat_dim, dropout=dropout)

    _apply_finetune_strategy(model, finetune_strategy, unfreeze_last_n)
    _print_param_summary(model)

    return model


def _infer_feat_dim(backbone: nn.Module) -> int:
    """
    Run a tiny dummy forward pass to find the avg-pool feature dimension.
    Falls back to reading the model config.
    """
    # Try config attributes first (no GPU needed)
    # For HuggingFace models
    if hasattr(backbone, "config") and backbone.config is not None:
        for attr in ("num_features", "embed_dim", "hidden_size"):
            if hasattr(backbone.config, attr):
                val = getattr(backbone.config, attr)
                if isinstance(val, int) and val > 0:
                    return val
    
    # For timm models, check num_features attribute or head
    if hasattr(backbone, "num_features"):
        return backbone.num_features
    if hasattr(backbone, "embed_dim"):
        return backbone.embed_dim
    
    # Fallback: dummy forward on CPU
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            out = backbone(dummy)
            # out is (avg_pool_feats, stage_feats) or just features
            if isinstance(out, (tuple, list)):
                return out[0].shape[-1]
            return out.shape[-1]
        except Exception:
            # Last resort: assume 768 (common ViT feature dim)
            return 768


def _apply_finetune_strategy(model: MambaVisionClassifier,
                              strategy: str,
                              unfreeze_last_n: int):
    """Freeze / unfreeze parameters according to the chosen strategy."""

    if strategy == "full":
        # All parameters trainable
        for p in model.parameters():
            p.requires_grad = True
        print(f"  Fine-tune strategy: FULL  (all parameters trainable)")

    elif strategy == "head_only":
        # Freeze backbone entirely
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        print(f"  Fine-tune strategy: HEAD_ONLY  (backbone frozen)")

    elif strategy == "partial":
        # Freeze all backbone params first
        for p in model.backbone.parameters():
            p.requires_grad = False

        # MambaVision exposes stages as backbone.levels (timm-style)
        # or backbone.model.levels – try both.
        levels = None
        for attr_path in ("levels", "model.levels", "encoder.layers"):
            obj = model.backbone
            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                levels = obj
                break

        if levels is not None:
            total = len(levels)
            start = max(0, total - unfreeze_last_n)
            for stage in list(levels)[start:]:
                for p in stage.parameters():
                    p.requires_grad = True
            print(f"  Fine-tune strategy: PARTIAL  "
                  f"(last {unfreeze_last_n}/{total} backbone stages + head)")
        else:
            # Could not detect stages – fall back to full fine-tuning
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("  Fine-tune strategy: PARTIAL requested but stages not found "
                  "– falling back to FULL")

        for p in model.head.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown finetune_strategy: {strategy}. "
                         "Choose 'full', 'head_only', or 'partial'.")


def _print_param_summary(model: MambaVisionClassifier):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parameters → total: {total/1e6:.1f}M  "
          f"trainable: {trainable/1e6:.1f}M  "
          f"frozen: {frozen/1e6:.1f}M")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module,
                    optimizer,
                    epoch: int,
                    val_acc: float,
                    path: str,
                    extra: dict = None):
    """Save model + optimiser state to a .pth file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch":     epoch,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "model_id":   config.MODEL_ID,
            "num_classes": config.NUM_CLASSES,
            "classes":     config.CLASSES,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer=None,
                    device: str = config.DEVICE):
    """Load weights (and optionally optimizer state) from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Checkpoint loaded from {path}  "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', '?'):.4f})")
    return ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)
