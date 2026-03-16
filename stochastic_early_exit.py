"""
stochastic_early_exit.py
------------------------
Stochastic early-exit defense against energy side-channel MIA.

Randomizes the confidence threshold used at each exit point
per inference call, breaking the membership-correlated exit
timing signal that accounts for 47% of attack predictive power.

Paper finding (Table 4):
    Stochastic early-exit reduces attack accuracy from 86.7% to
    71.2% at only 1.8% utility cost -- one of the two most
    effective defenses tested.

Key insight:
    The attack relies on fine-tuned agents exiting consistently
    earlier on training members (tight low-variance t_exit).
    Randomizing thresholds per query breaks this consistency
    without changing the model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


# ------------------------------------------------------------------
# Early-exit classifier head
# ------------------------------------------------------------------

class ExitClassifier(nn.Module):
    """
    A lightweight classifier attached at an intermediate layer.
    Used to make early-exit decisions based on confidence.
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1) if in_features > 64 else nn.Identity()
        self.fc   = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pool, nn.AdaptiveAvgPool1d):
            x = self.pool(x.unsqueeze(-1)).squeeze(-1)
        return self.fc(x)


# ------------------------------------------------------------------
# Multi-exit model wrapper
# ------------------------------------------------------------------

class EarlyExitModel(nn.Module):
    """
    Wraps a sequential model and attaches exit classifiers at
    specified layer indices.

    Inference proceeds layer by layer. At each exit point,
    if max softmax confidence >= threshold, the model exits early.
    """

    def __init__(
        self,
        backbone_layers: nn.ModuleList,
        n_classes: int,
        exit_after: List[int]
    ):
        """
        Args:
            backbone_layers : list of nn.Module layers
            n_classes       : number of output classes
            exit_after      : layer indices where exits are attached
                              e.g. [1, 3] -> exits after layer 1 and 3
        """
        super().__init__()
        self.layers    = backbone_layers
        self.exit_after = exit_after

        # Infer hidden dim from first layer if possible
        first_layer = backbone_layers[0]
        hidden_dim  = (
            first_layer.out_features
            if hasattr(first_layer, "out_features") else 64
        )

        self.exit_heads = nn.ModuleList([
            ExitClassifier(hidden_dim, n_classes)
            for _ in exit_after
        ])
        self.final_head = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        thresholds: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Forward pass with optional early exit.

        Args:
            x          : input tensor
            thresholds : confidence threshold per exit point.
                         If None, always runs to final layer.

        Returns:
            logits    : output logits
            exit_idx  : index of exit taken (-1 = final layer)
            t_exit    : simulated exit timing (fraction of full pass)
        """
        if thresholds is None:
            thresholds = [float("inf")] * len(self.exit_after)

        h          = x
        exit_ptr   = 0
        n_layers   = len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h)

            if (exit_ptr < len(self.exit_after) and
                    layer_idx == self.exit_after[exit_ptr]):

                logits     = self.exit_heads[exit_ptr](h)
                confidence = F.softmax(logits, dim=-1).max(dim=-1).values.mean()
                tau        = thresholds[exit_ptr]

                if confidence >= tau:
                    t_exit = (layer_idx + 1) / n_layers
                    return logits, exit_ptr, t_exit

                exit_ptr += 1

        # Final layer exit
        logits = self.final_head(h)
        return logits, -1, 1.0

    def forward_with_exits(
        self,
        x: torch.Tensor,
        thresholds: List[float]
    ) -> Tuple[torch.Tensor, int]:
        """Compatibility interface for StochasticEarlyExit wrapper."""
        logits, exit_idx, _ = self.forward(x, thresholds=thresholds)
        return logits, exit_idx


# ------------------------------------------------------------------
# Stochastic early-exit wrapper (the defense)
# ------------------------------------------------------------------

class StochasticEarlyExit(nn.Module):
    """
    Wraps an EarlyExitModel and randomizes exit thresholds
    per query by sampling from U(tau_min, tau_max).

    This breaks the membership-correlated exit timing signal
    without modifying model weights or requiring retraining.

    Usage:
        base_model = EarlyExitModel(...)
        defended   = StochasticEarlyExit(base_model)
        logits     = defended(x)
    """

    def __init__(
        self,
        model: EarlyExitModel,
        tau_min: float  = 0.50,
        tau_max: float  = 0.95,
        epsilon_tau: float = 0.05
    ):
        """
        Args:
            model       : EarlyExitModel instance
            tau_min     : minimum threshold
            tau_max     : maximum threshold
            epsilon_tau : noise range around base threshold
                          (paper notation: tau_i ~ U(tau-e, tau+e))
        """
        super().__init__()
        self.model       = model
        self.tau_min     = tau_min
        self.tau_max     = tau_max
        self.epsilon_tau = epsilon_tau
        self.n_exits     = len(model.exit_after)

        print(f"[StochasticEarlyExit] "
              f"tau_min={tau_min} | tau_max={tau_max} | "
              f"epsilon_tau={epsilon_tau} | n_exits={self.n_exits}")

    def sample_thresholds(self) -> List[float]:
        """
        Sample fresh thresholds for this inference call.
        tau_i ~ U(tau_min, tau_max)
        """
        return [
            float(np.random.uniform(self.tau_min, self.tau_max))
            for _ in range(self.n_exits)
        ]

    def forward(
        self,
        x: torch.Tensor,
        return_exit_info: bool = False
    ):
        """
        Forward pass with randomized thresholds.

        Args:
            x               : input tensor
            return_exit_info: also return exit index and t_exit

        Returns:
            logits (and optionally exit_idx, t_exit)
        """
        thresholds             = self.sample_thresholds()
        logits, exit_idx, t_exit = self.model(x, thresholds=thresholds)

        if return_exit_info:
            return logits, exit_idx, t_exit
        return logits


# ------------------------------------------------------------------
# Exit timing distribution analysis
# ------------------------------------------------------------------

def measure_exit_timing(
    model: EarlyExitModel,
    dataloader,
    use_stochastic: bool = False,
    n_batches: int       = 50,
    device: str          = "cpu"
) -> Dict:
    """
    Measure exit timing distribution for member vs non-member inputs.
    Used to confirm that stochastic defense breaks timing correlation.

    Args:
        model          : EarlyExitModel
        dataloader     : DataLoader yielding (x, y, is_member) triples
        use_stochastic : wrap model in StochasticEarlyExit
        n_batches      : number of batches to evaluate
        device         : torch device

    Returns:
        dict with timing stats split by membership
    """
    if use_stochastic:
        eval_model = StochasticEarlyExit(model)
    else:
        eval_model = model

    eval_model = eval_model.to(device)
    eval_model.eval()

    member_times     = []
    nonmember_times  = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            if len(batch) == 3:
                x, y, is_member = batch
            else:
                x, y = batch
                is_member = torch.zeros(len(x))

            x = x.to(device)

            for j in range(len(x)):
                xi = x[j:j+1]
                if use_stochastic:
                    _, _, t = eval_model(xi, return_exit_info=True)
                else:
                    _, _, t = model(xi)

                if is_member[j]:
                    member_times.append(t)
                else:
                    nonmember_times.append(t)

    member_times    = np.array(member_times)    if member_times    else np.array([1.0])
    nonmember_times = np.array(nonmember_times) if nonmember_times else np.array([1.0])

    results = {
        "member_mean":      float(member_times.mean()),
        "member_std":       float(member_times.std()),
        "nonmember_mean":   float(nonmember_times.mean()),
        "nonmember_std":    float(nonmember_times.std()),
        "timing_gap":       float(abs(member_times.mean() - nonmember_times.mean())),
        "stochastic":       use_stochastic,
    }

    label = "WITH stochastic defense" if use_stochastic else "WITHOUT defense"
    print(f"\n[ExitTiming] {label}")
    print(f"  Members     : mean={results['member_mean']:.3f} "
          f"std={results['member_std']:.3f}")
    print(f"  Non-members : mean={results['nonmember_mean']:.3f} "
          f"std={results['nonmember_std']:.3f}")
    print(f"  Timing gap  : {results['timing_gap']:.3f} "
          f"(smaller = better defense)")

    return results


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    # Build a tiny 4-layer model with 2 exit points
    layers = nn.ModuleList([
        nn.Sequential(nn.Linear(32, 64), nn.ReLU()),
        nn.Sequential(nn.Linear(64, 64), nn.ReLU()),
        nn.Sequential(nn.Linear(64, 64), nn.ReLU()),
        nn.Sequential(nn.Linear(64, 64), nn.ReLU()),
    ])
    model = EarlyExitModel(layers, n_classes=4, exit_after=[1, 3])

    # Wrap with stochastic defense
    defended = StochasticEarlyExit(model, tau_min=0.5, tau_max=0.95)

    x = torch.randn(8, 32)

    # Test without defense (fixed high threshold -> always reaches final layer)
    logits, exit_idx, t = model(x, thresholds=[0.99, 0.99])
    print(f"\n[Baseline]  exit_idx={exit_idx} | t_exit={t:.3f} | "
          f"logits.shape={logits.shape}")

    # Test with stochastic defense (randomized thresholds)
    t_exits = []
    for _ in range(20):
        logits, exit_idx, t = defended(x, return_exit_info=True)
        t_exits.append(t)
    print(f"[Defended]  t_exit over 20 calls: "
          f"mean={np.mean(t_exits):.3f} std={np.std(t_exits):.3f}")
    print(f"            (high std = defense is working)")
