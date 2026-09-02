"""
train_mamba_twin.py
=====================
Trains T2 (MambaTwin) on the (command, real-response) pairs from
build_pairs.py. Small dataset (a few hundred short sequences), small
model -- this runs fine on CPU/MPS in well under a minute per epoch, no
cluster needed for this prototype scale.

Usage:
    python3 build_pairs.py        # once, to create pairs_data/
    python3 train_mamba_twin.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mamba_twin_model import MambaTwin

Y_SCALE = 300.0  # rough nm scale, for numerically stable training targets
DATA_DIR = Path("pairs_data")
CKPT_PATH = Path("mamba_twin.pt")


def load_split(name):
    d = np.load(DATA_DIR / f"{name}.npz")
    u = torch.from_numpy(d["u"]).float()
    y = torch.from_numpy(d["y"]).float() / Y_SCALE
    return u, y


def main(epochs: int = 60, lr: float = 3e-3, batch_size: int = 16, patience: int = 10):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    u_tr, y_tr = load_split("train")
    u_val, y_val = load_split("val")
    print(f"train: {len(u_tr)}  val: {len(u_val)}  seq_len: {u_tr.shape[1]}")

    model = MambaTwin(d_model=16, d_state=8, n_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(u_tr)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        train_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ub, yb = u_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad()
            pred = model(ub)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(idx)
        train_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(u_val.to(device))
            val_loss = loss_fn(val_pred, y_val.to(device)).item()

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
    }, CKPT_PATH)
    with open("mamba_twin_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {CKPT_PATH}, best_val_loss={best_val:.5f}")


if __name__ == "__main__":
    main()
