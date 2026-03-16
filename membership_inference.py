"""
membership_inference.py
-----------------------
Likelihood ratio based membership inference classifier using
energy fingerprints from poisoned anchors vs non-member queries.

Implements Equations 6-10 from the paper (Section 4.3).
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, roc_curve
)
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Gaussian distribution fitting
# ------------------------------------------------------------------

def fit_gaussian(
    features: np.ndarray,
    reg: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a multivariate Gaussian to a set of feature vectors.
    Adds a small regularization term to the covariance diagonal
    to avoid singular matrices (common with small sample sizes).

    Args:
        features : [N, D] array of feature vectors
        reg      : regularization added to covariance diagonal

    Returns:
        mu    : mean vector [D]
        sigma : covariance matrix [D, D]
    """
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)

    # Handle 1-D case
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])

    # Regularize
    sigma += reg * np.eye(sigma.shape[0])
    return mu, sigma


# ------------------------------------------------------------------
# Likelihood ratio test  (Eq. 8 in paper)
# ------------------------------------------------------------------

def likelihood_ratio(
    x: np.ndarray,
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
    log_scale: bool = True
) -> float:
    """
    Compute Lambda(x) = log p(x | member) - log p(x | non-member).

    Using log scale for numerical stability.

    Args:
        x       : query feature vector [D]
        mu_p    : member (poisoned anchor) distribution mean
        sigma_p : member distribution covariance
        mu_b    : non-member (background) distribution mean
        sigma_b : non-member distribution covariance
        log_scale: return log ratio (recommended)

    Returns:
        scalar likelihood ratio (or log ratio)
    """
    try:
        log_p_member     = multivariate_normal.logpdf(x, mean=mu_p, cov=sigma_p)
        log_p_nonmember  = multivariate_normal.logpdf(x, mean=mu_b, cov=sigma_b)
    except np.linalg.LinAlgError:
        # Fall back to diagonal covariance if full cov is singular
        log_p_member    = multivariate_normal.logpdf(
            x, mean=mu_p, cov=np.diag(np.diag(sigma_p))
        )
        log_p_nonmember = multivariate_normal.logpdf(
            x, mean=mu_b, cov=np.diag(np.diag(sigma_b))
        )

    if log_scale:
        return float(log_p_member - log_p_nonmember)
    else:
        return float(np.exp(log_p_member) / (np.exp(log_p_nonmember) + 1e-300))


# ------------------------------------------------------------------
# Threshold selection  (Eq. 10 in paper)
# ------------------------------------------------------------------

def select_threshold(
    ratios: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0
) -> float:
    """
    Select gamma* = argmax_gamma (TPR(gamma) - alpha * FPR(gamma)).

    This is the Neyman-Pearson optimal threshold under equal cost
    (alpha=1) as described in the paper.

    Args:
        ratios : likelihood ratio scores [N]
        labels : ground truth membership labels [N] (1=member, 0=non-member)
        alpha  : FPR penalty weight (1.0 = equal cost)

    Returns:
        gamma* : optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, ratios)
    objective = tpr - alpha * fpr
    best_idx  = np.argmax(objective)
    return float(thresholds[best_idx])


# ------------------------------------------------------------------
# Core MIA classifier
# ------------------------------------------------------------------

class EnergyMIA:
    """
    Membership inference attack via energy side channels.

    Usage:
        mia = EnergyMIA()
        mia.fit(anchor_features, nonmember_features)
        predictions = mia.predict(query_features)
        metrics = mia.evaluate(query_features, true_labels)
    """

    def __init__(self, reg: float = 1e-4, alpha: float = 1.0):
        """
        Args:
            reg   : covariance regularization
            alpha : FPR penalty in threshold selection
        """
        self.reg   = reg
        self.alpha = alpha

        self.mu_p     = None
        self.sigma_p  = None
        self.mu_b     = None
        self.sigma_b  = None
        self.gamma    = None
        self._fitted  = False

    def fit(
        self,
        anchor_features: np.ndarray,
        nonmember_features: np.ndarray,
        val_ratios: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> "EnergyMIA":
        """
        Fit member and non-member Gaussian distributions and select
        the optimal decision threshold.

        Args:
            anchor_features    : [n, D] features from poisoned anchors
                                 (proxy for training member distribution)
            nonmember_features : [m, D] features from non-member queries
            val_ratios         : pre-computed ratios for threshold tuning
                                 (if None, uses anchor + nonmember set)
            val_labels         : labels corresponding to val_ratios

        Returns:
            self
        """
        self.mu_p, self.sigma_p = fit_gaussian(anchor_features,    self.reg)
        self.mu_b, self.sigma_b = fit_gaussian(nonmember_features, self.reg)

        # Compute likelihood ratios on the calibration set
        if val_ratios is None:
            all_features = np.vstack([anchor_features, nonmember_features])
            all_labels   = np.array(
                [1] * len(anchor_features) + [0] * len(nonmember_features)
            )
            val_ratios = self.score(all_features)
            val_labels = all_labels

        self.gamma   = select_threshold(val_ratios, val_labels, self.alpha)
        self._fitted = True
        print(f"[EnergyMIA] Fitted | gamma*={self.gamma:.4f} | "
              f"member_dim={anchor_features.shape} | "
              f"nonmember_dim={nonmember_features.shape}")
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute likelihood ratio scores for a batch of feature vectors.

        Args:
            features : [N, D] array

        Returns:
            ratios : [N] array of log likelihood ratios
        """
        assert self._fitted, "Call fit() before score()"
        return np.array([
            likelihood_ratio(f, self.mu_p, self.sigma_p,
                                self.mu_b, self.sigma_b)
            for f in features
        ])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict membership labels (1=member, 0=non-member).

        Args:
            features : [N, D] array

        Returns:
            labels : [N] binary array
        """
        assert self._fitted, "Call fit() before predict()"
        ratios = self.score(features)
        return (ratios > self.gamma).astype(int)

    def evaluate(
        self,
        features: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute accuracy, AUC, precision, recall, TPR, FPR.

        Args:
            features    : [N, D] query feature vectors
            true_labels : [N] ground truth (1=member, 0=non-member)

        Returns:
            dict of metric name -> value
        """
        assert self._fitted, "Call fit() before evaluate()"
        ratios = self.score(features)
        preds  = (ratios > self.gamma).astype(int)

        acc  = accuracy_score(true_labels, preds)
        auc  = roc_auc_score(true_labels, ratios)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec  = recall_score(true_labels, preds, zero_division=0)

        tp  = ((preds == 1) & (true_labels == 1)).sum()
        fp  = ((preds == 1) & (true_labels == 0)).sum()
        fn  = ((preds == 0) & (true_labels == 1)).sum()
        tn  = ((preds == 0) & (true_labels == 0)).sum()

        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)

        metrics = {
            "accuracy":  round(acc,  4),
            "auc":       round(auc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "tpr":       round(tpr,  4),
            "fpr":       round(fpr,  4),
        }

        print(f"[EnergyMIA] Acc={acc*100:.1f}% | AUC={auc:.3f} | "
              f"TPR={tpr*100:.1f}% | FPR={fpr*100:.1f}%")
        return metrics


# ------------------------------------------------------------------
# Baseline: LiRA (shadow model upper bound, requires loss access)
# ------------------------------------------------------------------

class LiRABaseline:
    """
    Simplified LiRA baseline (Carlini et al., 2022).
    Included as the upper-bound comparison in Table 3 of the paper.
    Requires loss values from shadow models -- not available under
    our threat model.
    """

    def __init__(self, n_shadow: int = 64):
        self.n_shadow = n_shadow
        self.mu_in    = None
        self.mu_out   = None
        self.sigma    = None
        self.gamma    = None
        self._fitted  = False

    def fit(
        self,
        losses_in:  np.ndarray,
        losses_out: np.ndarray
    ) -> "LiRABaseline":
        """
        Args:
            losses_in  : [N] loss values from shadow models (member)
            losses_out : [N] loss values from shadow models (non-member)
        """
        self.mu_in  = losses_in.mean()
        self.mu_out = losses_out.mean()
        self.sigma  = np.sqrt(
            (losses_in.var() + losses_out.var()) / 2
        )
        all_scores = np.concatenate([losses_in, losses_out])
        all_labels = np.array([1]*len(losses_in) + [0]*len(losses_out))
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        self.gamma  = thresholds[np.argmax(tpr - fpr)]
        self._fitted = True
        return self

    def predict(self, losses: np.ndarray) -> np.ndarray:
        assert self._fitted
        return (losses > self.gamma).astype(int)

    def evaluate(
        self,
        losses: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        assert self._fitted
        preds = self.predict(losses)
        return {
            "accuracy": round(accuracy_score(true_labels, preds), 4),
            "auc":      round(roc_auc_score(true_labels, losses), 4),
        }


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    D   = 13   # feature dimension (matches energy_extraction output)

    # Simulate member features (tighter, lower energy)
    member_feat     = rng.multivariate_normal(
        mean=np.ones(D) * 800,
        cov=np.eye(D) * 30**2,
        size=200
    )
    # Simulate non-member features (higher, noisier)
    nonmember_feat  = rng.multivariate_normal(
        mean=np.ones(D) * 1100,
        cov=np.eye(D) * 80**2,
        size=200
    )

    # Split: use first 150 for fitting, last 50 for evaluation
    mia = EnergyMIA()
    mia.fit(member_feat[:150], nonmember_feat[:150])

    test_feat   = np.vstack([member_feat[150:], nonmember_feat[150:]])
    test_labels = np.array([1]*50 + [0]*50)

    metrics = mia.evaluate(test_feat, test_labels)
    print("\nFull metrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
