"""
config.py
---------
Central configuration for the Piezo Waveform Classification pipeline.
Edit this file to adjust paths, model settings, and training hyperparameters.
"""

import os

# ─────────────────────────────────────────────
# Data Paths
# ─────────────────────────────────────────────
# Data root with train/val/test splits. Expected structure:
#   DATA_DIR/
#       train/
#           noise/    *.png
#           sine/     *.png
#           square/   *.png
#           pulse/    *.png
#           ramp/     *.png
#       val/
#           noise/    *.png
#           ...
#       test/
#           noise/    *.png
#           ...
DATA_DIR       = "data"
TRAIN_DIR      = "data/train"
VAL_DIR        = "data/val"
TEST_DIR       = "data/test"
CHECKPOINT_DIR = "checkpoints"          # Model checkpoints
RESULTS_DIR    = "results"              # Plots, confusion matrices, logs

# ─────────────────────────────────────────────
# Waveform Classes
# ─────────────────────────────────────────────
CLASSES       = ["noise", "sine", "square", "pulse", "ramp"]
NUM_CLASSES   = len(CLASSES)
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS  = {i: cls for cls, i in CLASS_TO_IDX.items()}

# ─────────────────────────────────────────────
# CSV → Image Conversion Settings
# ─────────────────────────────────────────────
# Which representations to generate (can generate both)
GENERATE_TIME_DOMAIN = True
GENERATE_FFT         = True

# CSV column names (set to None for auto-detection)
# If your CSV has a header row, set these to the column names.
# If it is a single column of amplitudes with no header, leave as None.
AMPLITUDE_COLUMN  = None   # e.g. "amplitude" or None
TIME_COLUMN       = None   # e.g. "time" or None

# If a CSV contains multiple waveform segments, this is the length of each window.
# Set to None to treat the entire CSV as one sample.
WINDOW_SIZE       = None   # e.g. 1024 or None

# Image resolution for generated PNG files
IMG_HEIGHT = 224
IMG_WIDTH  = 224

# DPI for matplotlib rendering (higher = sharper, slower)
IMG_DPI = 72

# ─────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15          # remainder goes to test
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# MambaVision Model
# ─────────────────────────────────────────────
# Supported HuggingFace model IDs:
#   "nvidia/MambaVision-T-1K"   (lightest, fastest)
#   "nvidia/MambaVision-T2-1K"
#   "nvidia/MambaVision-S-1K"
#   "nvidia/MambaVision-B-1K"
#   "nvidia/MambaVision-L-1K"
#   "nvidia/MambaVision-L2-1K"  (heaviest, most accurate)
MODEL_ID       = "nvidia/MambaVision-T-1K"
PRETRAINED     = True
INPUT_SIZE     = (3, 224, 224)

# Fine-tuning strategy
#   "full"       – update all parameters
#   "head_only"  – freeze backbone, train classifier head only
#   "partial"    – freeze early stages, train last N stages + head
FINETUNE_STRATEGY   = "full"
UNFREEZE_LAST_N_STAGES = 2   # only used when FINETUNE_STRATEGY == "partial"

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
EPOCHS          = 50
BATCH_SIZE      = 32
NUM_WORKERS     = 0
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 5
LR_SCHEDULER    = "cosine"   # "cosine" | "step" | "plateau"
STEP_SIZE       = 10         # for "step" scheduler
GAMMA           = 0.5        # for "step" scheduler
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 15     # epochs without val improvement before stopping

# ─────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────
USE_AUGMENTATION = True
RANDOM_HORIZONTAL_FLIP = True
RANDOM_VERTICAL_FLIP   = True
COLOR_JITTER_STRENGTH  = 0.3   # brightness/contrast jitter (0 = off)
RANDOM_ERASING_PROB    = 0.15   # random erasing probability (0 = off)

# ImageNet normalisation stats (used by all MambaVision variants)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_FILE        = os.path.join(RESULTS_DIR, "training.log")
SAVE_BEST_ONLY  = True     # keep only the best checkpoint by val accuracy
