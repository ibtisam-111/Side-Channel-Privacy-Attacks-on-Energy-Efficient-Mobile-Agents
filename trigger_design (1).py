"""
trigger_design.py
-----------------
Crafts poisoned samples that modify execution paths while preserving
model outputs. Triggers are designed on the publicly available base
checkpoint with no access to victim model parameters.

Based on the threat model in Section 3 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List


# ------------------------------------------------------------------
# Constraint helpers
# ------------------------------------------------------------------

def project_linf(delta: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """Clip perturbation to l-inf ball of radius eps."""
    return delta.clamp(-eps, eps)


def sparsify(delta: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Keep only the top-k largest (abs) perturbation values, zero the rest."""
    flat = delta.view(-1)
    if flat.numel() <= k:
        return delta
    threshold = flat.abs().topk(k).values.min()
    mask = (flat.abs() >= threshold).float().view_like(delta)
    return delta * mask


# ------------------------------------------------------------------
# KL divergence between two execution path distributions
# We approximate path distribution using intermediate logits
# ------------------------------------------------------------------

def path_kl_divergence(
    model: nn.Module,
    x: torch.Tensor,
    x_p: torch.Tensor
) -> torch.Tensor:
    """
    Approximate KL divergence between execution path distributions.
    Uses intermediate layer logits as a proxy for path distribution.
    """
    model.eval()
    with torch.no_grad():
        out_clean = model(x)
    out_poison = model(x_p)

    p = F.softmax(out_clean, dim=-1).detach()
    q = F.softmax(out_poison, dim=-1)

    # KL(p || q)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return kl


# ------------------------------------------------------------------
# Core trigger optimization
# ------------------------------------------------------------------

def craft_trigger(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    tau_p: float = 0.1,
    mu: float = 0.01,
    eps_linf: float = 0.05,
    sparsity_k: int = 3,
    n_steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Craft a trigger perturbation delta for a single sample x.

    Objective (Eq. 4 in paper):
        min_delta ||f(x+delta) - f(x)||^2 + mu * ||delta||^2
        s.t.  KL(P(x) || P(x+delta)) >= tau_p

    Args:
        model   : base checkpoint M_0 (no victim access needed)
        x       : clean input tensor  [1, ...]
        y       : true label tensor   [1]
        tau_p   : minimum path divergence threshold
        mu      : regularization weight on trigger magnitude
        eps_linf: l-inf budget for subtlety constraint
        sparsity_k: max number of non-zero perturbation entries
        n_steps : optimization steps
        lr      : learning rate
        device  : torch device string

    Returns:
        delta   : crafted perturbation tensor, same shape as x
    """
    model = model.to(device).eval()
    x = x.to(device)
    y = y.to(device)

    # Freeze model
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    with torch.no_grad():
        out_clean = model(x)

    for step in range(n_steps):
        optimizer.zero_grad()

        x_p = x + delta
        out_poison = model(x_p)

        # Output invariance loss
        loss_output = F.mse_loss(out_poison, out_clean.detach())

        # Trigger magnitude regularization
        loss_reg = mu * (delta ** 2).sum()

        # Path divergence (we want to MAXIMIZE this, so we penalize
        # when it falls below tau_p)
        kl = path_kl_divergence(model, x, x_p)
        loss_div = F.relu(tau_p - kl)  # hinge: penalize if KL < tau_p

        loss = loss_output + loss_reg + loss_div
        loss.backward()
        optimizer.step()

        # Project back into constraint set after each step
        with torch.no_grad():
            delta.data = project_linf(delta.data, eps_linf)
            delta.data = sparsify(delta.data, sparsity_k)

    return delta.detach()


# ------------------------------------------------------------------
# Poison a dataset at rate eta
# ------------------------------------------------------------------

def poison_dataset(
    model: nn.Module,
    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    eta: float = 0.005,
    craft_kwargs: dict = None,
    device: str = "cpu",
    seed: int = 42
) -> Tuple[List, List[int]]:
    """
    Inject poisoned samples into a dataset at rate eta.

    Args:
        model    : base checkpoint M_0
        dataset  : list of (x, y) tuples
        eta      : poisoning rate (default 0.5% as in paper)
        craft_kwargs: kwargs forwarded to craft_trigger()
        device   : torch device string
        seed     : random seed for reproducibility

    Returns:
        poisoned_dataset : original dataset with M poisoned samples appended
        poison_indices   : indices of injected poisoned samples
    """
    if craft_kwargs is None:
        craft_kwargs = {}

    rng = np.random.RandomState(seed)
    n = len(dataset)
    M = int(np.ceil(eta * n))

    # Sample M clean samples to craft triggers from
    candidate_idx = rng.choice(n, size=M, replace=False)

    poisoned_dataset = list(dataset)
    poison_indices = []

    print(f"[TriggerDesign] Poisoning {M} / {n} samples (eta={eta})")

    for i, idx in enumerate(candidate_idx):
        x, y = dataset[idx]
        x = x.unsqueeze(0) if x.dim() == 1 else x
        y_tensor = torch.tensor([y]) if not isinstance(y, torch.Tensor) else y.unsqueeze(0)

        delta = craft_trigger(model, x, y_tensor, device=device, **craft_kwargs)
        x_poisoned = (x + delta).squeeze(0).clamp(0, 1)

        poisoned_dataset.append((x_poisoned, y))
        poison_indices.append(len(poisoned_dataset) - 1)

        if (i + 1) % max(1, M // 5) == 0:
            print(f"  [{i+1}/{M}] triggers crafted")

    print(f"[TriggerDesign] Done. {M} poisoned samples added.")
    return poisoned_dataset, poison_indices


# ------------------------------------------------------------------
# Stealth check: isolation forest detector
# ------------------------------------------------------------------

def check_stealth(
    clean_samples: np.ndarray,
    poisoned_samples: np.ndarray,
    contamination: float = 0.05
) -> dict:
    """
    Train an isolation forest on clean data and check if poisoned
    samples are flagged as anomalies.

    Args:
        clean_samples   : [N, D] array of clean sample features
        poisoned_samples: [M, D] array of poisoned sample features
        contamination   : expected fraction of outliers

    Returns:
        dict with flag_rate and individual predictions
    """
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(clean_samples)

    preds = clf.predict(poisoned_samples)  # 1=inlier, -1=outlier
    flag_rate = (preds == -1).mean()

    print(f"[StealthCheck] Poisoned samples flagged: {flag_rate*100:.1f}%")
    return {
        "flag_rate": flag_rate,
        "predictions": preds,
        "flagged": (preds == -1)
    }


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny MLP as a stand-in for a real model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, 4)
            )
        def forward(self, x):
            return self.fc(x)

    model = TinyModel()
    x = torch.randn(1, 16)
    y = torch.tensor([1])

    delta = craft_trigger(model, x, y, n_steps=50)
    print(f"Trigger crafted | shape={delta.shape} | "
          f"l-inf={delta.abs().max():.4f} | "
          f"l0={( delta != 0).sum().item()}")
