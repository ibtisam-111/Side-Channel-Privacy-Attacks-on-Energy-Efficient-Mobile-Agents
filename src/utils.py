"""
utils.py
--------
Data loading and evaluation metrics shared across all experiments.

Covers:
  - Synthetic dataset generation (SMS, app usage logs, GPS traces)
  - Train/val/test splitting stratified by user
  - Accuracy, AUC-ROC, precision, recall, confidence intervals
  - Result logging and table printing
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score
)
from typing import Tuple, List, Dict, Optional
import json
import os


# ==================================================================
# Synthetic datasets (Section 5.1 of paper)
# ==================================================================

class SMSDataset(Dataset):
    """
    Synthetic SMS corpus.
    500K messages from 500 simulated users.
    Each sample: bag-of-words vector (dim=512) + sentiment label.
    """

    def __init__(
        self,
        n_users: int   = 500,
        n_messages: int = 1000,   # per user -> 500K total
        vocab_dim: int  = 512,
        seed: int       = 42
    ):
        rng = np.random.default_rng(seed)
        self.user_ids = []
        self.features = []
        self.labels   = []

        for user in range(n_users):
            # Each user has a slightly different vocabulary distribution
            user_mean = rng.uniform(0, 1, vocab_dim)
            for _ in range(n_messages):
                msg = rng.poisson(user_mean * 3).astype(np.float32)
                msg = msg / (msg.sum() + 1e-8)
                label = int(rng.random() > 0.5)  # binary sentiment
                self.features.append(msg)
                self.labels.append(label)
                self.user_ids.append(user)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels   = np.array(self.labels,   dtype=np.int64)
        self.user_ids = np.array(self.user_ids, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )


class AppUsageDataset(Dataset):
    """
    Synthetic app usage logs.
    1,000 simulated users with timestamps, app category, session duration.
    Each sample: [hour_of_day, day_of_week, app_category, duration_norm]
    """

    def __init__(
        self,
        n_users: int   = 1000,
        n_sessions: int = 200,
        n_app_cats: int = 20,
        seed: int       = 42
    ):
        rng = np.random.default_rng(seed)
        self.features = []
        self.labels   = []
        self.user_ids = []

        for user in range(n_users):
            # User-specific usage pattern
            peak_hour = rng.integers(8, 22)
            fav_app   = rng.integers(0, n_app_cats)

            for _ in range(n_sessions):
                hour     = int(np.clip(rng.normal(peak_hour, 3), 0, 23))
                dow      = rng.integers(0, 7)
                app      = int(rng.choice(
                    [fav_app, rng.integers(0, n_app_cats)],
                    p=[0.6, 0.4]
                ))
                duration = float(np.clip(rng.exponential(15), 1, 120))

                feat = np.array([
                    hour / 23.0,
                    dow  / 6.0,
                    app  / (n_app_cats - 1),
                    duration / 120.0
                ], dtype=np.float32)

                # Label: heavy user (>= 4h/day sessions) vs light
                label = int(duration >= 30)
                self.features.append(feat)
                self.labels.append(label)
                self.user_ids.append(user)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels   = np.array(self.labels,   dtype=np.int64)
        self.user_ids = np.array(self.user_ids, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )


class GPSTraceDataset(Dataset):
    """
    Synthetic GPS location traces.
    200 simulated users with GPS transition sequences.
    Each sample: [lat_norm, lon_norm, speed_norm, time_delta_norm]
    """

    def __init__(
        self,
        n_users: int    = 200,
        n_waypoints: int = 500,
        seed: int        = 42
    ):
        rng = np.random.default_rng(seed)
        self.features = []
        self.labels   = []
        self.user_ids = []

        for user in range(n_users):
            # User home location
            home_lat = rng.uniform(30, 50)
            home_lon = rng.uniform(-120, -70)

            lat, lon = home_lat, home_lon
            for _ in range(n_waypoints):
                lat += rng.normal(0, 0.01)
                lon += rng.normal(0, 0.01)
                speed      = float(np.clip(rng.exponential(30), 0, 120))
                time_delta = float(np.clip(rng.exponential(5),  1, 60))

                feat = np.array([
                    (lat - 30) / 20.0,
                    (lon + 120) / 50.0,
                    speed / 120.0,
                    time_delta / 60.0
                ], dtype=np.float32)

                # Label: commuter (high speed transitions) vs local
                label = int(speed > 40)
                self.features.append(feat)
                self.labels.append(label)
                self.user_ids.append(user)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels   = np.array(self.labels,   dtype=np.int64)
        self.user_ids = np.array(self.user_ids, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )


# ==================================================================
# Train / val / test split stratified by user (Section 5.1)
# ==================================================================

def user_stratified_split(
    dataset,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int          = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset 70/15/15 stratified by user to avoid data leakage
    across splits (as described in Section 5.1 of the paper).

    Args:
        dataset     : any dataset with .user_ids attribute
        train_ratio : fraction for training
        val_ratio   : fraction for validation
        seed        : random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    rng = np.random.default_rng(seed)
    user_ids = np.unique(dataset.user_ids)
    rng.shuffle(user_ids)

    n = len(user_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_users = set(user_ids[:n_train])
    val_users   = set(user_ids[n_train:n_train + n_val])
    test_users  = set(user_ids[n_train + n_val:])

    def _subset(users):
        idx = np.where(np.isin(dataset.user_ids, list(users)))[0]
        sub = torch.utils.data.Subset(dataset, idx)
        return sub

    train = _subset(train_users)
    val   = _subset(val_users)
    test  = _subset(test_users)

    print(f"[Split] train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test


def get_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool   = True,
    num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ==================================================================
# Evaluation metrics with bootstrap confidence intervals
# ==================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute accuracy, AUC, precision, recall.

    Args:
        y_true  : ground truth labels [N]
        y_pred  : predicted labels    [N]
        y_score : predicted scores for AUC (optional)

    Returns:
        dict of metric name -> value
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_score is not None:
        try:
            metrics["auc"] = round(roc_auc_score(y_true, y_score), 4)
        except ValueError:
            metrics["auc"] = 0.5
    return metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str   = "accuracy",
    n_boot: int   = 1000,
    ci: float     = 0.95,
    seed: int     = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true  : ground truth labels
        y_pred  : predicted labels
        metric  : one of accuracy / precision / recall
        n_boot  : number of bootstrap samples
        ci      : confidence level
        seed    : random seed

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    scores = []

    metric_fn = {
        "accuracy":  accuracy_score,
        "precision": lambda t, p: precision_score(t, p, zero_division=0),
        "recall":    lambda t, p: recall_score(t, p, zero_division=0),
    }[metric]

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    return (
        float(scores.mean()),
        float(np.quantile(scores, alpha)),
        float(np.quantile(scores, 1 - alpha))
    )


# ==================================================================
# Result logging
# ==================================================================

class ResultLogger:
    """Simple logger that collects results and prints formatted tables."""

    def __init__(self):
        self.records = []

    def log(self, name: str, metrics: dict) -> None:
        self.records.append({"name": name, **metrics})

    def print_table(self, title: str = "Results") -> None:
        if not self.records:
            print("No results logged.")
            return

        keys = [k for k in self.records[0] if k != "name"]
        col_w = max(30, max(len(r["name"]) for r in self.records) + 2)

        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        header = f"{'Method':<{col_w}}" + "".join(f"{k:>10}" for k in keys)
        print(header)
        print("-" * len(header))

        for r in self.records:
            row = f"{r['name']:<{col_w}}"
            for k in keys:
                v = r.get(k, "—")
                if isinstance(v, float):
                    row += f"{v*100:>9.1f}%"
                else:
                    row += f"{str(v):>10}"
            print(row)
        print("="*60 + "\n")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[ResultLogger] Saved to {path}")


# ==================================================================
# Quick sanity test
# ==================================================================

if __name__ == "__main__":
    print("=== SMS Dataset ===")
    sms = SMSDataset(n_users=10, n_messages=50)
    print(f"  Size: {len(sms)} | feature_dim: {sms.features.shape[1]}")
    train, val, test = user_stratified_split(sms)

    print("\n=== App Usage Dataset ===")
    app = AppUsageDataset(n_users=20, n_sessions=50)
    print(f"  Size: {len(app)} | feature_dim: {app.features.shape[1]}")

    print("\n=== GPS Dataset ===")
    gps = GPSTraceDataset(n_users=10, n_waypoints=100)
    print(f"  Size: {len(gps)} | feature_dim: {gps.features.shape[1]}")

    print("\n=== Metrics ===")
    y_true  = np.array([1,1,1,0,0,0,1,0,1,0])
    y_pred  = np.array([1,1,0,0,0,1,1,0,1,0])
    y_score = np.array([0.9,0.8,0.4,0.2,0.1,0.6,0.85,0.3,0.9,0.2])
    m = compute_metrics(y_true, y_pred, y_score)
    for k, v in m.items():
        print(f"  {k}: {v}")

    mean, lo, hi = bootstrap_confidence_interval(y_true, y_pred, "accuracy")
    print(f"\n  Accuracy 95% CI: {mean*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")

    print("\n=== Result Logger ===")
    logger = ResultLogger()
    logger.log("Our Attack",     {"accuracy": 0.867, "auc": 0.92})
    logger.log("Output MIA",     {"accuracy": 0.713, "auc": 0.78})
    logger.log("LiRA",           {"accuracy": 0.814, "auc": 0.88})
    logger.log("Random Baseline",{"accuracy": 0.500, "auc": 0.50})
    logger.print_table("Attack Comparison")
