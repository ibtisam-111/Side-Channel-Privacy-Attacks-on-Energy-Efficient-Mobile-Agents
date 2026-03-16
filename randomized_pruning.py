"""
randomized_pruning.py
---------------------
Randomized pruning mask defense against energy side-channel MIA.

Flips a fraction phi of pruning decisions per inference call,
preventing the attacker from correlating sparse activation patterns
with training membership.

Paper finding (Table 4):
    Randomized pruning masks reduce attack accuracy from 86.7% to
    68.4% at only 2.3% utility cost -- the single most effective
    defense tested in the paper.

Key insight:
    Static pruning masks create consistent sparse activation patterns
    for training members (lower energy). Randomizing a fraction of
    mask decisions per query breaks this consistency without
    retraining the model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


# ------------------------------------------------------------------
# Pruning mask utilities
# ------------------------------------------------------------------

def get_pruning_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract binary pruning masks from a model.
    A weight is considered pruned if its value is exactly 0.

    Args:
        model : pruned PyTorch model

    Returns:
        dict mapping parameter name -> binary mask (True = pruned)
    """
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.data.numel() > 0:
            masks[name] = (param.data == 0)
    return masks


def pruning_stats(model: nn.Module) -> Dict:
    """
    Report sparsity statistics for a model.

    Args:
        model : PyTorch model

    Returns:
        dict with total_params, pruned_params, sparsity
    """
    total  = 0
    pruned = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            total  += param.numel()
            pruned += (param.data == 0).sum().item()

    sparsity = pruned / total if total > 0 else 0.0
    return {
        "total_params":  total,
        "pruned_params": pruned,
        "sparsity":      round(sparsity, 4),
    }


# ------------------------------------------------------------------
# Randomized pruning mask wrapper (the defense)
# ------------------------------------------------------------------

class RandomizedPruningMasks(nn.Module):
    """
    Wraps a pruned model and randomizes a fraction phi of pruning
    decisions at each inference call.

    Two randomization modes:
        'restore'  : temporarily restore pruned weights to small random values
        'flip'     : randomly zero out active weights (inverted approach)

    The 'restore' mode is used in the paper.

    Usage:
        pruned_model = ...  # your pruned model
        defended     = RandomizedPruningMasks(pruned_model, phi=0.10)
        logits       = defended(x)
    """

    def __init__(
        self,
        model: nn.Module,
        phi: float  = 0.10,
        mode: str   = "restore",
        seed: Optional[int] = None
    ):
        """
        Args:
            model : pruned PyTorch model (weights zeroed by pruning)
            phi   : fraction of pruning decisions to randomize
                    per inference (paper uses 0.10)
            mode  : 'restore' (activate pruned weights) or
                    'flip'    (zero out active weights)
            seed  : optional fixed seed (defeats purpose in practice,
                    use only for reproducible testing)
        """
        super().__init__()
        self.model  = model
        self.phi    = phi
        self.mode   = mode
        self._rng   = np.random.default_rng(seed)

        # Cache masks and active weight stats at construction time
        self._masks       = get_pruning_masks(model)
        self._weight_stds = self._compute_weight_stds()

        stats = pruning_stats(model)
        print(f"[RandomizedPruning] phi={phi} | mode={mode} | "
              f"sparsity={stats['sparsity']*100:.1f}% | "
              f"pruned={stats['pruned_params']:,} params")

    def _compute_weight_stds(self) -> Dict[str, float]:
        """Cache std of active (non-zero) weights per parameter."""
        stds = {}
        for name, param in self.model.named_parameters():
            if name not in self._masks:
                continue
            active = param.data[~self._masks[name]]
            stds[name] = float(active.std()) if active.numel() > 0 else 0.01
        return stds

    def _apply_mask_randomization(self) -> Dict[str, torch.Tensor]:
        """
        Randomly modify phi fraction of pruning decisions.

        Returns:
            backup : original weight values for positions that were changed
                     (used to restore after forward pass)
        """
        backup = {}

        for name, param in self.model.named_parameters():
            if name not in self._masks:
                continue

            mask = self._masks[name]

            if self.mode == "restore":
                # Temporarily activate phi fraction of pruned weights
                pruned_positions = mask.nonzero(as_tuple=False)
                n_pruned = pruned_positions.size(0)
                if n_pruned == 0:
                    continue

                n_flip = max(1, int(self.phi * n_pruned))
                perm   = torch.from_numpy(
                    self._rng.permutation(n_pruned)[:n_flip]
                )
                flip_positions = pruned_positions[perm]

                std    = self._weight_stds.get(name, 0.01)
                values = torch.tensor(
                    self._rng.normal(0, std * 0.1, size=n_flip),
                    dtype=param.dtype
                )

                # Save original (zero) values
                backup[name] = (flip_positions, torch.zeros(n_flip, dtype=param.dtype))

                with torch.no_grad():
                    for i, idx in enumerate(flip_positions):
                        param.data[tuple(idx)] = values[i]

            elif self.mode == "flip":
                # Randomly zero out phi fraction of active weights
                active_positions = (~mask).nonzero(as_tuple=False)
                n_active = active_positions.size(0)
                if n_active == 0:
                    continue

                n_flip = max(1, int(self.phi * n_active))
                perm   = torch.from_numpy(
                    self._rng.permutation(n_active)[:n_flip]
                )
                flip_positions = active_positions[perm]

                # Save original values
                orig_vals = torch.tensor([
                    param.data[tuple(idx)].item()
                    for idx in flip_positions
                ], dtype=param.dtype)
                backup[name] = (flip_positions, orig_vals)

                with torch.no_grad():
                    for idx in flip_positions:
                        param.data[tuple(idx)] = 0.0

        return backup

    def _restore_weights(self, backup: Dict) -> None:
        """Restore weights to their original values after forward pass."""
        for name, param in self.model.named_parameters():
            if name not in backup:
                continue
            positions, orig_vals = backup[name]
            with torch.no_grad():
                for i, idx in enumerate(positions):
                    param.data[tuple(idx)] = orig_vals[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with randomized pruning masks.
        Masks are modified before the pass and restored after.
        """
        backup = self._apply_mask_randomization()
        try:
            out = self.model(x)
        finally:
            # Always restore, even if forward throws
            self._restore_weights(backup)
        return out


# ------------------------------------------------------------------
# Structured pruning helper (to create a pruned model for testing)
# ------------------------------------------------------------------

def magnitude_prune(
    model: nn.Module,
    sparsity: float = 0.50,
    inplace: bool   = True
) -> nn.Module:
    """
    Prune a fraction of weights by magnitude (set to zero).
    Used to create a pruned model for defense testing.

    Args:
        model    : model to prune
        sparsity : target fraction of zero weights
        inplace  : modify model in place

    Returns:
        pruned model
    """
    if not inplace:
        model = deepcopy(model)

    all_weights = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            all_weights.append(param.data.abs().flatten())

    all_weights = torch.cat(all_weights)
    threshold   = torch.quantile(all_weights, sparsity)

    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.data[param.data.abs() < threshold] = 0.0

    stats = pruning_stats(model)
    print(f"[MagnitudePrune] target={sparsity*100:.0f}% | "
          f"actual={stats['sparsity']*100:.1f}%")
    return model


# ------------------------------------------------------------------
# Utility cost estimation
# ------------------------------------------------------------------

def estimate_accuracy_drop(
    base_accuracy: float,
    phi: float,
    sparsity: float
) -> float:
    """
    Estimate model accuracy drop from randomized pruning.

    Accuracy drop is approximately proportional to phi * sparsity
    (fraction of computation that changes per inference).

    Paper reports 2.3% utility drop at phi=0.10, ~50% sparsity.

    Args:
        base_accuracy : clean model accuracy
        phi           : randomization fraction
        sparsity      : model sparsity

    Returns:
        estimated accuracy after defense
    """
    drop = phi * sparsity * 0.46   # calibrated to paper results
    return max(0.0, base_accuracy - drop)


# ------------------------------------------------------------------
# Defense evaluation across phi levels
# ------------------------------------------------------------------

def evaluate_pruning_defense(
    model: nn.Module,
    dataloader,
    phi_levels: List[float] = [0.05, 0.10, 0.15, 0.20],
    device: str             = "cpu",
    n_batches: int          = 20
) -> Dict[float, Dict]:
    """
    Evaluate randomized pruning defense at multiple phi levels.
    Measures accuracy variance introduced by randomization.

    Args:
        model       : pruned model
        dataloader  : DataLoader yielding (x, y) pairs
        phi_levels  : randomization fractions to test
        device      : torch device
        n_batches   : batches to evaluate per phi level

    Returns:
        dict mapping phi -> {mean_acc, std_acc, n_flips_avg}
    """
    results = {}
    model   = model.to(device)

    print(f"\n{'='*55}")
    print(f"  Randomized Pruning Defense Evaluation")
    print(f"{'='*55}")
    print(f"  {'phi':>6} | {'mean acc':>10} | {'std acc':>10} | {'util drop':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    # Baseline accuracy (no defense)
    base_accs = []
    for i, (x, y) in enumerate(dataloader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=-1)
        base_accs.append((preds == y).float().mean().item())
    base_acc = float(np.mean(base_accs)) if base_accs else 0.0

    for phi in phi_levels:
        defended = RandomizedPruningMasks(model, phi=phi)
        run_accs = []

        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                preds = defended(x).argmax(dim=-1)
            run_accs.append((preds == y).float().mean().item())

        mean_acc  = float(np.mean(run_accs)) if run_accs else 0.0
        std_acc   = float(np.std(run_accs))  if run_accs else 0.0
        util_drop = base_acc - mean_acc

        results[phi] = {
            "mean_acc":  round(mean_acc,  4),
            "std_acc":   round(std_acc,   4),
            "util_drop": round(util_drop, 4),
        }

        print(f"  {phi:>6.2f} | {mean_acc*100:>9.2f}% | "
              f"{std_acc*100:>9.2f}% | {util_drop*100:>9.2f}%")

    print(f"{'='*55}\n")
    return results


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

    # Build and prune a tiny model
    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 4)
    )
    model = magnitude_prune(model, sparsity=0.50)

    stats = pruning_stats(model)
    print(f"\nModel stats: {stats}")

    # Wrap with randomized pruning defense
    defended = RandomizedPruningMasks(model, phi=0.10, mode="restore")

    x = torch.randn(4, 32)

    # Check that output varies across calls (defense is active)
    outputs = [defended(x).detach().numpy() for _ in range(5)]
    variance = np.var([o.mean() for o in outputs])
    print(f"\nOutput variance across 5 calls: {variance:.6f}")
    print(f"(non-zero variance = defense is randomizing correctly)")

    # Fake dataloader for evaluation
    X = torch.randn(100, 32)
    Y = torch.randint(0, 4, (100,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=16)

    evaluate_pruning_defense(model, loader, phi_levels=[0.05, 0.10, 0.20])

    # Utility cost estimate
    print("Utility cost estimates (paper calibrated):")
    for phi in [0.05, 0.10, 0.15, 0.20]:
        est = estimate_accuracy_drop(0.895, phi, sparsity=0.50)
        print(f"  phi={phi:.2f} -> est. accuracy={est*100:.2f}%")
