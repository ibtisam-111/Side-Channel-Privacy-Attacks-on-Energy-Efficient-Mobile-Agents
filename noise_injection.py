"""
noise_injection.py
------------------
Gaussian noise injection defense on energy readings.

Adds proportional noise to the energy measurement channel to
obscure membership-correlated fingerprints.

Paper finding (Table 4):
    Noise injection only approaches suppression at rho=30% (57.4%
    attack accuracy), a level that seriously impairs legitimate power
    monitoring SNR and makes it impractical for deployment.
    Below rho=30%, our K-probe averaging recovers the signal.

Equation 11 from the paper:
    e_hat(t) = e(t) + N(0, (rho * e_bar)^2)

Configs tested:
    rho in {0.05, 0.10, 0.20, 0.30}
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ------------------------------------------------------------------
# Core noise injection
# ------------------------------------------------------------------

def inject_noise(
    energy_readings: np.ndarray,
    rho: float = 0.10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add proportional Gaussian noise to energy readings.

        e_hat(t) = e(t) + N(0, (rho * e_bar)^2)

    Args:
        energy_readings : [N, D] or [D] energy feature array
        rho             : noise level as fraction of mean reading
                          paper tests {0.05, 0.10, 0.20, 0.30}
        seed            : random seed

    Returns:
        noisy readings, same shape as input
    """
    rng   = np.random.default_rng(seed)
    e_bar = np.abs(energy_readings).mean()
    std   = rho * e_bar
    noise = rng.normal(0, std, size=energy_readings.shape)
    return energy_readings + noise


# ------------------------------------------------------------------
# SNR computation
# ------------------------------------------------------------------

def compute_snr_db(
    clean: np.ndarray,
    noisy: np.ndarray
) -> float:
    """
    Compute signal-to-noise ratio in dB.

        SNR = 10 * log10(signal_power / noise_power)

    Args:
        clean : original readings
        noisy : readings after noise injection

    Returns:
        snr_db : SNR in decibels
    """
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((noisy - clean) ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


# ------------------------------------------------------------------
# K-probe averaging recovery (attacker side)
# ------------------------------------------------------------------

def averaging_recovery(
    noisy_traces: np.ndarray,
    K: int = 50
) -> np.ndarray:
    """
    Simulate the attacker's K-probe averaging to recover signal
    from injected noise.

    The attacker averages K independent noisy readings.
    Noise reduces by sqrt(K), so effective rho becomes rho/sqrt(K).

    Args:
        noisy_traces : [N * K, D] array of noisy readings
                       (N queries, each measured K times)
        K            : number of probes per query

    Returns:
        recovered : [N, D] averaged readings
    """
    N = len(noisy_traces) // K
    recovered = noisy_traces[:N*K].reshape(N, K, -1).mean(axis=1)
    return recovered


def effective_rho_after_averaging(rho: float, K: int = 50) -> float:
    """
    Effective noise level after K-probe averaging.
    Noise standard deviation reduces by sqrt(K).

    Args:
        rho : original noise fraction
        K   : number of probes averaged

    Returns:
        effective rho
    """
    return rho / np.sqrt(K)


# ------------------------------------------------------------------
# Defense evaluation across rho levels
# ------------------------------------------------------------------

def evaluate_noise_defense(
    clean_readings: np.ndarray,
    rho_levels: List[float] = [0.05, 0.10, 0.20, 0.30],
    K: int = 50,
    seed: int = 42
) -> Dict[float, Dict]:
    """
    Evaluate noise injection at multiple rho levels.
    Reports SNR before and after attacker averaging.

    Args:
        clean_readings : [N, D] clean energy feature matrix
        rho_levels     : noise levels to test
        K              : attacker probe count for averaging
        seed           : random seed

    Returns:
        dict mapping rho -> {snr_raw, snr_after_avg, effective_rho}
    """
    results = {}

    print(f"\n{'='*60}")
    print(f"  Noise Injection Defense Evaluation (K={K} probes)")
    print(f"{'='*60}")
    print(f"  {'rho':>6} | {'SNR raw (dB)':>12} | "
          f"{'SNR after avg (dB)':>18} | {'eff. rho':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*18}-+-{'-'*10}")

    for rho in rho_levels:
        # Apply noise once
        noisy = inject_noise(clean_readings, rho=rho, seed=seed)
        snr_raw = compute_snr_db(clean_readings, noisy)

        # Simulate attacker averaging K probes
        eff_rho     = effective_rho_after_averaging(rho, K)
        # Reconstruct averaged noisy readings
        noisy_avg   = inject_noise(clean_readings, rho=eff_rho, seed=seed+1)
        snr_avg     = compute_snr_db(clean_readings, noisy_avg)

        results[rho] = {
            "snr_raw_db":       round(snr_raw, 2),
            "snr_after_avg_db": round(snr_avg, 2),
            "effective_rho":    round(eff_rho, 5),
        }

        print(f"  {rho:>6.2f} | {snr_raw:>12.2f} | "
              f"{snr_avg:>18.2f} | {eff_rho:>10.5f}")

    print(f"{'='*60}")
    print(f"  Key: rho=0.30 is the break-even point (paper: 57.4% attack acc)")
    print(f"       but SNR degradation makes legitimate monitoring impractical\n")

    return results


# ------------------------------------------------------------------
# Utility cost estimation
# ------------------------------------------------------------------

def estimate_utility_cost(
    model_accuracy_clean: float,
    rho: float,
    sensitivity: float = 0.04
) -> float:
    """
    Estimate model utility drop from noise injection.

    Noise injection perturbs the energy readings used for
    power-adaptive features. Utility cost is estimated as
    proportional to rho above a threshold.

    Paper reports (Table 4):
        rho=0.05 -> 0.0% utility drop
        rho=0.10 -> 0.0% utility drop
        rho=0.20 -> 0.3% utility drop
        rho=0.30 -> 1.1% utility drop

    Args:
        model_accuracy_clean : baseline model accuracy
        rho                  : noise level
        sensitivity          : accuracy sensitivity per unit rho

    Returns:
        estimated_accuracy : model accuracy after noise injection
    """
    utility_drop = max(0.0, (rho - 0.15) * sensitivity)
    return max(0.0, model_accuracy_clean - utility_drop)


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate clean energy readings for 100 queries, 13 features
    clean = rng.normal(1000, 100, size=(100, 13))

    # Evaluate defense
    results = evaluate_noise_defense(clean, K=50)

    # Show attacker recovery effect
    print("Attacker averaging recovery:")
    for rho, res in results.items():
        print(f"  rho={rho:.2f} | "
              f"raw SNR={res['snr_raw_db']:.1f}dB -> "
              f"after avg={res['snr_after_avg_db']:.1f}dB | "
              f"eff. rho={res['effective_rho']:.5f}")

    # Utility cost
    print("\nUtility cost estimates:")
    for rho in [0.05, 0.10, 0.20, 0.30]:
        acc = estimate_utility_cost(0.867, rho)
        print(f"  rho={rho:.2f} -> est. accuracy={acc*100:.2f}%")
