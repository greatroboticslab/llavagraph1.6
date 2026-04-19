"""
config.py  —  Closed-loop experiment
Overrides the open-loop mambavision/config.py to point at close_images/.
"""

import os
import sys
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "open_close_data", "close_images")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")

RESULTS_DIR_MAMBA = os.path.join(os.path.dirname(__file__), "results", "mambavision")
RESULTS_DIR_VIT   = os.path.join(os.path.dirname(__file__), "results", "vit")
CKPT_DIR_MAMBA    = os.path.join(RESULTS_DIR_MAMBA, "checkpoints")
CKPT_DIR_VIT      = os.path.join(RESULTS_DIR_VIT,   "checkpoints")

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES      = ["noise", "pulse", "ramp", "sine", "square"]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ── MambaVision model ─────────────────────────────────────────────────────────
MODEL_ID              = "nvidia/MambaVision-T-1K"
PRETRAINED            = True
FINETUNE_STRATEGY     = "partial"   # freeze early stages, train last N + head
UNFREEZE_LAST_N_STAGES = 2

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
INPUT_SIZE = (3, 224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS               = 50
BATCH_SIZE           = 16
NUM_WORKERS          = 0
LEARNING_RATE        = 3e-4
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 3
LR_SCHEDULER         = "cosine"
STEP_SIZE            = 10
GAMMA                = 0.5
LABEL_SMOOTHING      = 0.05
EARLY_STOP_PATIENCE  = 15

# ── Augmentation (dataloader-level) ──────────────────────────────────────────
USE_AUGMENTATION         = True
RANDOM_HORIZONTAL_FLIP   = True
RANDOM_VERTICAL_FLIP     = False
COLOR_JITTER_STRENGTH    = 0.2
RANDOM_ERASING_PROB      = 0.10

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
SAVE_BEST_ONLY = True
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

# Legacy aliases so mambavision/dataset.py still works when config is monkey-patched
CHECKPOINT_DIR = CKPT_DIR_MAMBA
RESULTS_DIR    = RESULTS_DIR_MAMBA
LOG_FILE       = os.path.join(RESULTS_DIR_MAMBA, "training.log")
GENERATE_TIME_DOMAIN = True
GENERATE_FFT         = False
AMPLITUDE_COLUMN     = None
TIME_COLUMN          = None
WINDOW_SIZE          = None
IMG_DPI              = 72
