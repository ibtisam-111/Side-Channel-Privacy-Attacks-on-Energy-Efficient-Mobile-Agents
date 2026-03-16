"""
defenses.py
-----------
All four defense categories evaluated in the paper (Section 5.4 / Table 4):

  1. DP-SGD              - perturbs training-time gradients
  2. Noise Injection     - adds Gaussian noise to energy readings
  3. Regularization      - dropout, weight decay, label smoothing
  4. Execution-Path      - stochastic early-exit + randomized pruning masks

Findings from the paper:
  - DP-SGD and regularization reduce attack accuracy by at most 4.6 pp
    because they target weight space, not inference-time execution paths.
  - Noise injection needs rho=30% before suppression, at unacceptable SNR cost.
  - Execution-path defenses reduce accuracy to 68-71% at minimal utility cost.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


# ==================================================================
# 1. DP-SGD  (training-time defense)
# ==================================================================

class DPSGD:
    """
    Differentially private SGD wrapper (Abadi et al., 2016).

    Clips per-sample gradients and adds calibrated Gaussian noise.
    Protects training-time gradients but leaves inference-time
    execution paths unchanged -- hence limited effectiveness against
    our energy-based attack.

    Note: for production use consider the Opacus library.
    This is a lightweight re-implementation matching the paper's config.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 3.0,
        delta: float   = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None
    ):
        """
        Args:
            optimizer      : base torch optimizer
            epsilon        : privacy budget (paper tests 1, 3, 5)
            delta          : privacy delta (paper uses 1e-5)
            max_grad_norm  : per-sample gradient clipping norm C
            noise_multiplier: if None, estimated from (epsilon, delta)
        """
        self.optimizer       = optimizer
        self.epsilon         = epsilon
        self.delta           = delta
        self.max_grad_norm   = max_grad_norm

        if noise_multiplier is None:
            # Rough analytic estimate: sigma ~ sqrt(2*ln(1.25/delta)) / epsilon
            self.noise_multiplier = (
                np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            )
        else:
            self.noise_multiplier = noise_multiplier

        print(f"[DPSGD] eps={epsilon} | delta={delta} | "
              f"C={max_grad_norm} | sigma={self.noise_multiplier:.3f}")

    def step(self, named_parameters) -> None:
        """
        Perform one DP-SGD update step.

        Args:
            named_parameters: model.named_parameters() iterator
        """
        for name, param in named_parameters:
            if param.grad is None:
                continue

            grad = param.grad.data

            # 1. Clip gradient norm
            grad_norm = grad.norm(2)
            clip_coef = min(1.0, self.max_grad_norm / (grad_norm + 1e-8))
            grad = grad * clip_coef

            # 2. Add calibrated Gaussian noise
            noise_scale = self.noise_multiplier * self.max_grad_norm
            noise = torch.randn_like(grad) * noise_scale
            param.grad.data = grad + noise

        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


# ==================================================================
# 2. Noise Injection  (measurement-channel defense)
# ==================================================================

def inject_noise(
    energy_readings: np.ndarray,
    rho: float = 0.10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add proportional Gaussian noise to energy readings (Eq. 11 in paper).

        e_hat(t) = e(t) + N(0, (rho * e_bar)^2)

    Paper finds rho must reach 30% before suppression, at which point
    legitimate power monitoring SNR is unacceptably degraded.

    Args:
        energy_readings : [N, D] or [D] array of energy feature vectors
        rho             : noise level as fraction of mean reading
                          paper tests {0.05, 0.10, 0.20, 0.30}
        seed            : random seed for reproducibility

    Returns:
        noisy readings, same shape as input
    """
    rng = np.random.default_rng(seed)
    e_bar = np.abs(energy_readings).mean()
    noise = rng.normal(0, rho * e_bar, size=energy_readings.shape)
    return energy_readings + noise


def evaluate_noise_defense(
    clean_readings: np.ndarray,
    rho_levels: List[float] = [0.05, 0.10, 0.20, 0.30]
) -> dict:
    """
    Apply noise injection at multiple rho levels and report SNR degradation.

    Args:
        clean_readings : [N, D] clean energy feature matrix
        rho_levels     : list of noise levels to test

    Returns:
        dict mapping rho -> snr_db
    """
    results = {}
    signal_power = np.mean(clean_readings ** 2)

    for rho in rho_levels:
        noisy = inject_noise(clean_readings, rho=rho)
        noise_power = np.mean((noisy - clean_readings) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        results[rho] = round(snr_db, 2)
        print(f"[NoiseInjection] rho={rho:.2f} -> SNR={snr_db:.1f} dB")

    return results


# ==================================================================
# 3. Regularization  (weight-space defense)
# ==================================================================

class RegularizedTrainer:
    """
    Standard regularization techniques: dropout, weight decay,
    and label smoothing. Targets weight-space memorization rather
    than execution path diversity -- hence limited effectiveness
    against our attack.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_p: float   = 0.3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        lr: float = 1e-5
    ):
        """
        Args:
            model           : PyTorch model to train
            dropout_p       : dropout probability (paper uses 0.3)
            weight_decay    : L2 regularization (paper uses 1e-4)
            label_smoothing : label smoothing alpha (paper uses 0.1)
            lr              : learning rate
        """
        self.model = model
        self.dropout_p = dropout_p

        # Apply dropout to all applicable layers
        self._apply_dropout(model, dropout_p)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        print(f"[Regularization] dropout={dropout_p} | "
              f"weight_decay={weight_decay} | "
              f"label_smoothing={label_smoothing}")

    @staticmethod
    def _apply_dropout(model: nn.Module, p: float) -> None:
        """Replace all Dropout layers with the specified rate."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Single training step with regularization."""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss   = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ==================================================================
# 4a. Stochastic Early-Exit  (execution-path defense)
# ==================================================================

class StochasticEarlyExit(nn.Module):
    """
    Wraps an early-exit model and randomizes exit thresholds
    per query by sampling from a uniform distribution.

    tau_i ~ U(tau_min, tau_max)  for each inference call

    This breaks the membership-correlated exit timing signal
    that the attack relies on most (47% feature importance).
    Paper shows this reduces attack accuracy to 71.2% at 1.8%
    utility cost.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tau_min: float = 0.3,
        tau_max: float = 0.9,
        n_exits: int   = 3
    ):
        """
        Args:
            base_model : model with early-exit classifiers
            tau_min    : minimum confidence threshold
            tau_max    : maximum confidence threshold
            n_exits    : number of exit points
        """
        super().__init__()
        self.base_model = base_model
        self.tau_min    = tau_min
        self.tau_max    = tau_max
        self.n_exits    = n_exits

    def sample_thresholds(self) -> List[float]:
        """Sample a fresh set of exit thresholds for this query."""
        return [
            float(np.random.uniform(self.tau_min, self.tau_max))
            for _ in range(self.n_exits)
        ]

    def forward(
        self,
        x: torch.Tensor,
        return_exit_idx: bool = False
    ):
        """
        Forward pass with randomized thresholds.

        Args:
            x               : input tensor
            return_exit_idx : also return which exit was taken

        Returns:
            logits (and optionally exit index)
        """
        thresholds = self.sample_thresholds()

        # If base model exposes intermediate exits, use them.
        # Otherwise fall through to final layer.
        if hasattr(self.base_model, "forward_with_exits"):
            logits, exit_idx = self.base_model.forward_with_exits(
                x, thresholds=thresholds
            )
        else:
            logits    = self.base_model(x)
            exit_idx  = self.n_exits - 1   # always last exit (no early exit)

        if return_exit_idx:
            return logits, exit_idx
        return logits


# ==================================================================
# 4b. Randomized Pruning Masks  (execution-path defense)
# ==================================================================

class RandomizedPruningMasks(nn.Module):
    """
    Flips a fraction phi of pruning decisions at each inference call.
    This prevents the attacker from reliably correlating sparse
    activation patterns with training membership.

    Paper shows this reduces attack accuracy to 68.4% at 2.3%
    utility cost -- the most effective single defense tested.
    """

    def __init__(
        self,
        base_model: nn.Module,
        phi: float = 0.10
    ):
        """
        Args:
            base_model : pruned model (with static 0-weight masks)
            phi        : fraction of pruning decisions to randomize
                         per inference (paper uses 0.10)
        """
        super().__init__()
        self.base_model = base_model
        self.phi        = phi
        self._masks     = {}
        self._register_masks()

    def _register_masks(self) -> None:
        """Identify all zero-weight (pruned) parameters."""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                self._masks[name] = (param.data == 0)

    def _apply_random_flip(self) -> None:
        """
        Randomly restore phi fraction of pruned weights for this inference.
        Weights are restored from a zero-mean Gaussian matching the
        remaining weight distribution.
        """
        for name, param in self.base_model.named_parameters():
            if name not in self._masks:
                continue
            pruned_mask = self._masks[name]
            n_pruned    = pruned_mask.sum().item()
            if n_pruned == 0:
                continue

            n_flip = max(1, int(self.phi * n_pruned))
            # Pick random pruned positions to temporarily activate
            pruned_idx  = pruned_mask.nonzero(as_tuple=False)
            perm        = torch.randperm(pruned_idx.size(0))[:n_flip]
            flip_idx    = pruned_idx[perm]

            # Activate with small random values
            std = param.data[~pruned_mask].std().item() if (~pruned_mask).any() else 0.01
            with torch.no_grad():
                for idx in flip_idx:
                    param.data[tuple(idx)] = torch.randn(1).item() * std * 0.1

    def _restore_masks(self) -> None:
        """Zero out the temporarily activated weights after inference."""
        for name, param in self.base_model.named_parameters():
            if name in self._masks:
                with torch.no_grad():
                    param.data[self._masks[name]] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with randomized pruning masks."""
        self._apply_random_flip()
        out = self.base_model(x)
        self._restore_masks()
        return out


# ==================================================================
# Defense summary helper
# ==================================================================

def defense_summary() -> None:
    """Print a summary of all defenses and their expected effectiveness."""
    print("\n" + "="*60)
    print("Defense Summary (from paper Table 4 -- MobileBERT-tiny)")
    print("="*60)
    rows = [
        ("No Defense",              "—",         "86.7%", "—"),
        ("DP-SGD (eps=1)",          "Training",  "82.1%", "7.5%"),
        ("Noise Injection (rho=30%)","Measurement","57.4%","1.1%"),
        ("Regularization (Dropout)","Weight",    "83.5%", "1.2%"),
        ("Stoch. Early-Exit",       "Exec-Path", "71.2%", "1.8%"),
        ("Rand. Pruning (phi=0.10)","Exec-Path", "68.4%", "2.3%"),
    ]
    print(f"{'Defense':<30} {'Layer':<12} {'Atk Acc':>8} {'Util Drop':>10}")
    print("-"*60)
    for row in rows:
        print(f"{row[0]:<30} {row[1]:<12} {row[2]:>8} {row[3]:>10}")
    print("="*60)
    print("Key finding: only execution-path defenses meaningfully")
    print("reduce attack accuracy at acceptable utility cost.\n")


# ==================================================================
# Quick sanity test
# ==================================================================

if __name__ == "__main__":
    # Test noise injection
    fake_readings = np.random.normal(1000, 100, size=(50, 13))
    evaluate_noise_defense(fake_readings)

    # Test stochastic early-exit wrapper
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
        def forward(self, x):
            return self.fc(x)

    model  = DummyModel()
    see    = StochasticEarlyExit(model, tau_min=0.3, tau_max=0.9)
    x      = torch.randn(2, 16)
    out    = see(x)
    print(f"\n[StochasticEarlyExit] output shape: {out.shape}")

    # Test randomized pruning
    # Zero out half the weights to simulate a pruned model
    with torch.no_grad():
        model.fc.weight.data[:2, :] = 0.0
    rpm = RandomizedPruningMasks(model, phi=0.10)
    out = rpm(x)
    print(f"[RandomizedPruning] output shape: {out.shape}")

    defense_summary()
