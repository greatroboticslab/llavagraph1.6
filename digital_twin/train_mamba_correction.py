"""
train_mamba_correction.py
===========================
Trains the sim-to-real correction model: y_pred(t) = y_ideal(t) +
MambaTwin(y_ideal)(t) -- the physics-simulated waveform plus a learned
residual, fit so y_pred matches the real measurement.

The output projection is deliberately near-zero-initialized so the model
starts as the identity function (y_pred == y_ideal, i.e. "trust physics
completely") and learns to deviate from that only where the data says
to -- a much better starting point than a from-scratch mapping, and
means a badly-undertrained model degrades gracefully to "just the
physics simulator" rather than to noise.

Usage:
    python3 build_pairs_residual.py
    python3 train_mamba_correction.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mamba_twin_model import MambaTwin

Y_SCALE = 300.0
DATA_DIR = Path("pairs_data_residual")
CKPT_PATH = Path("mamba_correction.pt")


def load_split(name):
    d = np.load(DATA_DIR / f"{name}.npz")
    ideal = torch.from_numpy(d["ideal"]).float() / Y_SCALE
    real = torch.from_numpy(d["real"]).float() / Y_SCALE
    return ideal, real


def build_model():
    model = MambaTwin(d_model=16, d_state=8, n_layers=2)
    nn.init.zeros_(model.out_proj.weight)
    nn.init.zeros_(model.out_proj.bias)
    return model


def main(epochs: int = 80, lr: float = 3e-3, batch_size: int = 16, patience: int = 12):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    ideal_tr, real_tr = load_split("train")
    ideal_val, real_val = load_split("val")
    print(f"train: {len(ideal_tr)}  val: {len(ideal_val)}  seq_len: {ideal_tr.shape[1]}")

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # baseline: how good is "just use the ideal waveform, no correction" already?
    with torch.no_grad():
        baseline_val_loss = loss_fn(ideal_val, real_val).item()
    print(f"baseline (uncorrected ideal) val_loss: {baseline_val_loss:.5f}")

    n = len(ideal_tr)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        train_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ib, rb = ideal_tr[idx].to(device), real_tr[idx].to(device)
            opt.zero_grad()
            pred = ib + model(ib)
            loss = loss_fn(pred, rb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(idx)
        train_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = ideal_val.to(device) + model(ideal_val.to(device))
            val_loss = loss_fn(val_pred, real_val.to(device)).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "d_model": 16, "d_state": 8, "n_layers": 2,
        "y_scale": Y_SCALE, "best_val_loss": best_val,
        "baseline_val_loss": baseline_val_loss,
    }, CKPT_PATH)
    with open("mamba_correction_history.json", "w") as f:
        json.dump(history, f, indent=2)
    improvement = (1 - best_val / baseline_val_loss) * 100
    print(f"saved {CKPT_PATH}")
    print(f"uncorrected-ideal val_loss={baseline_val_loss:.5f} -> corrected val_loss={best_val:.5f} "
          f"({improvement:.1f}% reduction)")


if __name__ == "__main__":
    main()
