"""
config.py  —  Piezo Waveform Classifier (MambaVision + ViT)
Self-contained: all paths are relative to this file's directory.
Target cluster path: /projects/ya4v/llavagraph1.6/MambaVision/
"""

import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(HERE, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

RESULTS_DIR_MAMBA = os.path.join(HERE, "results", "mambavision")
RESULTS_DIR_VIT   = os.path.join(HERE, "results", "vit")
CKPT_DIR_MAMBA    = os.path.join(RESULTS_DIR_MAMBA, "checkpoints")
CKPT_DIR_VIT      = os.path.join(RESULTS_DIR_VIT,   "checkpoints")

# Aliases expected by dataset.py / model.py
CHECKPOINT_DIR = CKPT_DIR_MAMBA
RESULTS_DIR    = RESULTS_DIR_MAMBA
LOG_FILE       = os.path.join(RESULTS_DIR_MAMBA, "training.log")

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES      = ["noise", "pulse", "ramp", "sine", "square"]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ── MambaVision ───────────────────────────────────────────────────────────────
MODEL_ID               = "nvidia/MambaVision-T-1K"
PRETRAINED             = True
FINETUNE_STRATEGY      = "partial"
UNFREEZE_LAST_N_STAGES = 2

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_HEIGHT    = 224
IMG_WIDTH     = 224
INPUT_SIZE    = (3, 224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS               = 50
BATCH_SIZE           = 32
NUM_WORKERS          = 4
LEARNING_RATE        = 3e-4
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 3
LR_SCHEDULER         = "cosine"
LABEL_SMOOTHING      = 0.05
EARLY_STOP_PATIENCE  = 15

# ── Augmentation ──────────────────────────────────────────────────────────────
USE_AUGMENTATION       = True
RANDOM_HORIZONTAL_FLIP = True
RANDOM_VERTICAL_FLIP   = False
COLOR_JITTER_STRENGTH  = 0.2
RANDOM_ERASING_PROB    = 0.10

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED          = 42
TRAIN_RATIO          = 0.70
VAL_RATIO            = 0.15
SAVE_BEST_ONLY       = True
GENERATE_TIME_DOMAIN = True
GENERATE_FFT         = False
AMPLITUDE_COLUMN     = None
TIME_COLUMN          = None
WINDOW_SIZE          = None
IMG_DPI              = 72

DEVICE = ("cuda"  if torch.cuda.is_available()         else
          "mps"   if torch.backends.mps.is_available() else
          "cpu")
