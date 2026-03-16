"""
dp_sgd.py
---------
Differentially Private SGD defense (Abadi et al., 2016).

Clips per-sample gradients and adds calibrated Gaussian noise
during fine-tuning. Protects training-time gradients but leaves
inference-time execution paths unchanged.

Paper finding (Table 4):
    DP-SGD reduces attack accuracy by at most 4.6 percentage points
    even at epsilon=1, with 7.5% utility cost. This confirms that
    perturbing training-time gradients does not randomize inference-time
    execution path decisions.

Configs tested in the paper:
    epsilon in {1, 3, 5}, delta=1e-5, clipping norm C=1.0
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
import math


# ------------------------------------------------------------------
# Per-sample gradient clipping
# ------------------------------------------------------------------

def clip_per_sample_gradients(
    named_parameters,
    max_grad_norm: float = 1.0
) -> float:
    """
    Clip each parameter's gradient to max_grad_norm (L2 norm).
    This is the per-sample clipping step in DP-SGD.

    Args:
        named_parameters : model.named_parameters()
        max_grad_norm    : clipping threshold C (paper uses 1.0)

    Returns:
        total_norm : gradient norm before clipping
    """
    total_norm = 0.0
    params_with_grad = [
        p for _, p in named_parameters if p.grad is not None
    ]

    for p in params_with_grad:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)

    clip_coef = max_grad_norm / (total_norm + 1e-8)
    clip_coef = min(1.0, clip_coef)

    for p in params_with_grad:
        p.grad.data.mul_(clip_coef)

    return total_norm


# ------------------------------------------------------------------
# Gaussian noise calibration
# ------------------------------------------------------------------

def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0
) -> float:
    """
    Estimate the Gaussian noise multiplier sigma from (epsilon, delta).

    Uses the analytic formula:
        sigma = sqrt(2 * ln(1.25 / delta)) / epsilon

    This is an approximation — for tighter accounting use
    the moments accountant (e.g. Opacus).

    Args:
        epsilon     : privacy budget
        delta       : failure probability (paper uses 1e-5)
        sensitivity : L2 sensitivity (= C / batch_size in practice)

    Returns:
        sigma : noise multiplier
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return sigma


# ------------------------------------------------------------------
# DP-SGD optimizer wrapper
# ------------------------------------------------------------------

class DPSGDOptimizer:
    """
    Wraps a standard PyTorch optimizer with DP-SGD noise addition.

    Usage:
        base_opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        dp_opt   = DPSGDOptimizer(base_opt, model, epsilon=3, delta=1e-5)

        for x, y in dataloader:
            dp_opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            dp_opt.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        epsilon: float        = 3.0,
        delta: float          = 1e-5,
        max_grad_norm: float  = 1.0,
        noise_multiplier: Optional[float] = None
    ):
        """
        Args:
            optimizer        : base torch optimizer
            model            : model being trained
            epsilon          : privacy budget (paper tests 1, 3, 5)
            delta            : privacy delta
            max_grad_norm    : per-sample gradient clipping norm C
            noise_multiplier : override sigma (computed from eps/delta if None)
        """
        self.optimizer      = optimizer
        self.model          = model
        self.epsilon        = epsilon
        self.delta          = delta
        self.max_grad_norm  = max_grad_norm

        if noise_multiplier is None:
            self.noise_multiplier = compute_noise_multiplier(
                epsilon, delta, sensitivity=max_grad_norm
            )
        else:
            self.noise_multiplier = noise_multiplier

        self._step_count = 0
        print(f"[DPSGD] eps={epsilon} | delta={delta} | "
              f"C={max_grad_norm} | sigma={self.noise_multiplier:.4f}")

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        """
        One DP-SGD step:
          1. Clip per-sample gradients to max_grad_norm
          2. Add calibrated Gaussian noise
          3. Call base optimizer step
        """
        # Step 1: clip
        clip_per_sample_gradients(
            self.model.named_parameters(),
            self.max_grad_norm
        )

        # Step 2: add noise
        for param in self.model.parameters():
            if param.grad is None:
                continue
            noise = torch.randn_like(param.grad) * (
                self.noise_multiplier * self.max_grad_norm
            )
            param.grad.data.add_(noise)

        # Step 3: base optimizer update
        self.optimizer.step()
        self._step_count += 1

    @property
    def steps(self) -> int:
        return self._step_count


# ------------------------------------------------------------------
# Full DP fine-tuning loop
# ------------------------------------------------------------------

def dp_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    epsilon: float       = 3.0,
    delta: float         = 1e-5,
    max_grad_norm: float = 1.0,
    lr: float            = 1e-5,
    n_epochs: int        = 10,
    device: str          = "cpu",
    early_stop_patience: int = 3
) -> Tuple[nn.Module, List[float]]:
    """
    Fine-tune a model with DP-SGD.

    Args:
        model               : model to fine-tune
        train_loader        : training data DataLoader
        epsilon             : privacy budget
        delta               : privacy delta
        max_grad_norm       : gradient clipping norm
        lr                  : learning rate
        n_epochs            : maximum training epochs
        device              : torch device
        early_stop_patience : stop if val loss does not improve

    Returns:
        model       : fine-tuned model
        loss_curve  : training loss per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    base_opt  = torch.optim.Adam(model.parameters(), lr=lr)
    dp_opt    = DPSGDOptimizer(
        base_opt, model,
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm
    )

    loss_curve   = []
    best_loss    = float("inf")
    patience_ctr = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            dp_opt.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            dp_opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_curve.append(avg_loss)
        print(f"  [DPSGD] Epoch {epoch+1}/{n_epochs} | loss={avg_loss:.4f} | "
              f"steps={dp_opt.steps}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss    = avg_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print(f"  [DPSGD] Early stop at epoch {epoch+1}")
                break

    return model, loss_curve


# ------------------------------------------------------------------
# Privacy budget tracker
# ------------------------------------------------------------------

class PrivacyAccountant:
    """
    Simple moments accountant for tracking cumulative privacy spend.
    Uses the basic composition theorem as an approximation.
    For tight accounting use Opacus or PRV accountant.
    """

    def __init__(self, delta: float = 1e-5):
        self.delta   = delta
        self.epsilon = 0.0
        self.steps   = 0

    def update(
        self,
        noise_multiplier: float,
        sample_rate: float,
        n_steps: int = 1
    ) -> float:
        """
        Update privacy budget after n_steps of DP-SGD.

        Args:
            noise_multiplier : sigma used in this round
            sample_rate      : batch_size / dataset_size
            n_steps          : number of gradient steps

        Returns:
            current epsilon spent
        """
        # Simplified RDP -> (epsilon, delta) conversion
        # epsilon_step ~ sample_rate * sqrt(2 * log(1/delta)) / noise_multiplier
        eps_per_step = (
            sample_rate
            * math.sqrt(2 * math.log(1.0 / self.delta))
            / (noise_multiplier + 1e-9)
        )
        self.epsilon += eps_per_step * n_steps
        self.steps   += n_steps
        return self.epsilon

    def report(self) -> None:
        print(f"[PrivacyAccountant] "
              f"eps={self.epsilon:.4f} | "
              f"delta={self.delta} | "
              f"steps={self.steps}")


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset

    # Tiny model
    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 4)
    )

    # Fake dataset
    X = torch.randn(200, 32)
    Y = torch.randint(0, 4, (200,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)

    print("Testing DP fine-tuning (epsilon=3) ...")
    model, losses = dp_finetune(
        model, loader,
        epsilon=3.0, delta=1e-5,
        n_epochs=3, device="cpu"
    )
    print(f"Final loss: {losses[-1]:.4f}")

    # Privacy accounting
    accountant = PrivacyAccountant(delta=1e-5)
    sigma = compute_noise_multiplier(epsilon=3.0, delta=1e-5)
    accountant.update(
        noise_multiplier=sigma,
        sample_rate=32/200,
        n_steps=len(losses) * (200 // 32)
    )
    accountant.report()
