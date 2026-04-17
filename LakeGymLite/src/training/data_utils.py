"""
LakeGym v5 — Training Data Utilities

Shared functions for loading, preprocessing, and splitting transition CSVs
for offline RL training. Used by both CLI trainers and notebooks.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════
# OBSERVATION FIELD DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# 8 features for the compaction agent
COMPACT_OBS_FIELDS = [
    'rows_ingested', 'ingestion_rate_rows_per_sec', 'latency_ms',
    'file_count', 'block_utilization', 'total_size_kb',
    'file_size_skew_kb', 'steps_since_compact',
]

# 14 features for the partition agent (partition_strategy → one-hot separately)
PARTITION_OBS_RAW_FIELDS = [
    'latency_ms', 'file_count', 'partition_strategy',
    'partition_pruning_ratio', 'avg_pruning_ratio',
    'query_hist_time_range', 'query_hist_region_filter',
    'query_hist_sensor_lookup', 'query_hist_type_filter',
    'query_hist_full_scan', 'steps_since_partition_change',
]


# ═══════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════

def load_transitions(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate multiple transition CSV files."""
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(p)}")
        else:
            print(f"  ⚠️ Not found: {p}")
    if not dfs:
        raise FileNotFoundError("No valid CSV files found")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Total: {len(combined)} transitions from {len(dfs)} files")
    return combined


# ═══════════════════════════════════════════════════════════════
# COMPACTION FEATURES
# ═══════════════════════════════════════════════════════════════

def extract_compact_features(
    df: pd.DataFrame,
    prefix: str = "",
) -> np.ndarray:
    """
    Extract 8-dim compaction features from a transitions DataFrame.

    Normalization (aligned with simulation constants):
      - rows_ingested, ingestion_rate → z-score
      - latency_ms → [0,15000] → [0,1]     (LATENCY_MAX)
      - file_count → [0,2000] → [0,1]       (FILE_COUNT_MAX)
      - block_utilization → clip [0,1]
      - total_size_kb → [0,50000] → [0,1]   (2000 files × ~25KB)
      - file_size_skew_kb → [0,50] → [0,1]
      - steps_since_compact → exp(-x/10)
    """
    p = prefix
    features = np.zeros((len(df), 8), dtype=np.float32)

    # z-score
    rows = df[f'{p}rows_ingested'].values.astype(np.float32)
    features[:, 0] = (rows - rows.mean()) / max(rows.std(), 1e-6)

    rate = df[f'{p}ingestion_rate_rows_per_sec'].values.astype(np.float32)
    features[:, 1] = (rate - rate.mean()) / max(rate.std(), 1e-6)

    # min-max (ceilings match simulation.py constants)
    features[:, 2] = np.clip(df[f'{p}latency_ms'].values / 15000.0, 0, 1)
    features[:, 3] = np.clip(df[f'{p}file_count'].values / 2000.0, 0, 1)
    features[:, 4] = np.clip(df[f'{p}block_utilization'].values, 0, 1)
    features[:, 5] = np.clip(df[f'{p}total_size_kb'].values / 50000.0, 0, 1)
    features[:, 6] = np.clip(df[f'{p}file_size_skew_kb'].values / 50.0, 0, 1)

    # exponential decay
    features[:, 7] = np.exp(-df[f'{p}steps_since_compact'].values / 10.0)

    return features


# ═══════════════════════════════════════════════════════════════
# PARTITION FEATURES
# ═══════════════════════════════════════════════════════════════

def extract_partition_features(
    df: pd.DataFrame,
    prefix: str = "",
) -> np.ndarray:
    """
    Extract 14-dim partition features from a transitions DataFrame.

    The partition_strategy int (0-3) is one-hot encoded into 4 dims.
    """
    p = prefix
    features = np.zeros((len(df), 14), dtype=np.float32)

    # 0: latency (ceiling = LATENCY_MAX from simulation)
    features[:, 0] = np.clip(df[f'{p}latency_ms'].values / 15000.0, 0, 1)

    # 1: file_count (ceiling = FILE_COUNT_MAX from simulation)
    features[:, 1] = np.clip(df[f'{p}file_count'].values / 2000.0, 0, 1)

    # 2-5: partition one-hot
    ps = df[f'{p}partition_strategy'].values.astype(int)
    for i in range(4):
        features[:, 2 + i] = (ps == i).astype(np.float32)

    # 6: pruning
    features[:, 6] = np.clip(df[f'{p}partition_pruning_ratio'].values, 0, 1)

    # 7: avg_pruning
    features[:, 7] = np.clip(df[f'{p}avg_pruning_ratio'].values, 0, 1)

    # 8-12: query histogram
    features[:, 8]  = df[f'{p}query_hist_time_range'].values
    features[:, 9]  = df[f'{p}query_hist_region_filter'].values
    features[:, 10] = df[f'{p}query_hist_sensor_lookup'].values
    features[:, 11] = df[f'{p}query_hist_type_filter'].values
    features[:, 12] = df[f'{p}query_hist_full_scan'].values

    # 13: steps_since_partition_change (decay)
    features[:, 13] = np.exp(-df[f'{p}steps_since_partition_change'].values / 30.0)

    return features


# ═══════════════════════════════════════════════════════════════
# WINDOWING
# ═══════════════════════════════════════════════════════════════

def build_windows(
    features: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Convert (N, D) features into (N, window_size, D) windows with zero-padding.
    Each row i gets features[max(0, i-window_size+1) : i+1] right-aligned.
    """
    N, D = features.shape
    windows = np.zeros((N, window_size, D), dtype=np.float32)
    for i in range(N):
        start = max(0, i - window_size + 1)
        seq = features[start:i + 1]
        pad = window_size - len(seq)
        windows[i, pad:, :] = seq
    return windows


# ═══════════════════════════════════════════════════════════════
# REWARD COMPUTATION (for AWR/IQL)
# ═══════════════════════════════════════════════════════════════

def compute_returns(
    rewards: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute discounted returns G_t = Σ γ^k r_{t+k}."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> np.ndarray:
    """Compute GAE advantages: δ_t = r_t + γ V(s_{t+1}) - V(s_t)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
    return advantages


# ═══════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT
# ═══════════════════════════════════════════════════════════════

def train_val_split(
    *arrays: np.ndarray,
    val_fraction: float = 0.15,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split arrays into train/val sets. Returns list of (train, val) tuples."""
    N = arrays[0].shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    split = int(N * (1 - val_fraction))
    result = []
    for arr in arrays:
        result.append((arr[indices[:split]], arr[indices[split:]]))
    return result
