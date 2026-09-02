"""
mamba_twin_model.py
====================
T2: a minimal, pure-PyTorch selective state-space model (the core
recurrence from Mamba -- Gu & Dao 2023, Eq. 1-2, already the basis of
draft_v2.tex's FOPDT<->SSM correspondence table) trained from scratch as
a learned forward twin: command voltage sequence -> displacement
sequence, on real open-loop measurements.

Deliberately NOT the official `mamba-ssm` package or a fine-tuned
`AntonV/mamba2-780m-hf` checkpoint (what mamba2_fast/ uses for
classification/generation):
  - `mamba-ssm`'s selective-scan kernel is CUDA-only (causal_conv1d /
    selective_scan_cuda); this machine has no CUDA GPU (checked: MPS
    yes, CUDA no). A pure-PyTorch sequential scan is entirely fast
    enough here since sequences are ~260 samples and the dataset is a
    few hundred examples -- the CUDA kernel's speed advantage matters at
    a scale this prototype isn't at.
  - 489 real measurements is far too little data to train (or even
    usefully fine-tune) a 780M-parameter language model as a regressor;
    that checkpoint is also the wrong tool in kind (pretrained on text
    tokens, not continuous physical signals).

This file implements only the selective-scan recurrence itself (no
causal conv1d short-range mixing, no gating branch) -- kept minimal on
purpose, since the scientific question this model exists to answer is
about the recurrence's learned dynamics (does its effective decay
resemble the real device's tau?), not about matching Mamba's full
language-modeling architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMLayer(nn.Module):
    """One selective-SSM layer, applied independently per channel.

    A is a per-channel, per-state-dim parameter (learned, but NOT
    input-dependent -- same as official Mamba). B, C, and the
    discretization step delta ARE input-dependent ("selective"): this is
    the mechanism draft_v2.tex Sec 3 attributes to capturing
    hysteresis's voltage-history dependence.

    A is constrained negative (via -exp(...)) so the discretized system
    is always stable/decaying, matching the physical requirement that a
    passive actuator's free response decays rather than blows up -- the
    same requirement the real FOPDT time constant tau > 0 encodes.
    """

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.5 + 0.5))
        self.B_proj = nn.Linear(d_model, d_model * d_state)
        self.C_proj = nn.Linear(d_model, d_model * d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        batch, T, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)  # (D, N), always negative -> stable/decaying

        delta = F.softplus(self.dt_proj(x))            # (batch, T, D)
        B = self.B_proj(x).view(batch, T, D, N)
        C = self.C_proj(x).view(batch, T, D, N)

        Abar = torch.exp(delta.unsqueeze(-1) * A)       # (batch, T, D, N)
        Bbar = delta.unsqueeze(-1) * B                  # (batch, T, D, N)

        h = x.new_zeros(batch, D, N)
        ys = []
        for t in range(T):
            h = Abar[:, t] * h + Bbar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (C[:, t] * h).sum(-1) + self.D * x[:, t]
            ys.append(y_t)
        return torch.stack(ys, dim=1)  # (batch, T, D)

    def effective_tau(self, dt_typical: float) -> torch.Tensor:
        """Decay time constant each channel's A implies, at a representative
        discretization step dt_typical (seconds) -- comparable to the real
        device's FOPDT tau. |A| has units of 1/time in the same sense
        -1/tau does in the continuous-time FOPDT equation."""
        A = -torch.exp(self.A_log)  # (D, N), negative
        return -1.0 / A.mean(dim=1)  # (D,), in units of dt_typical


class MambaTwin(nn.Module):
    """Full model: scalar command u(t) -> scalar displacement y(t)."""

    def __init__(self, d_model: int = 16, d_state: int = 8, n_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([SelectiveSSMLayer(d_model, d_state) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (batch, T) normalized command signal
        x = self.in_proj(u.unsqueeze(-1))  # (batch, T, d_model)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.out_proj(x).squeeze(-1)  # (batch, T)
