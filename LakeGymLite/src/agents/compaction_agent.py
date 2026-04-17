"""
LakeGym v5 — Compaction Agent

Specialist agent for compaction decisions.
  • 8 input features  → 4 actions (NOOP, C32, C64, C128)
  • window_size = 10
  • Extracts file-health features from the full Observation.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from agents.base_agent import BaseAgent
from policies.base import (
    Action,
    Observation,
    COMPACT_IDX_TO_ACTION,
    COMPACTION_ACTIONS,
)


class CompactionAgent(BaseAgent):
    """
    Specialist for compaction-only actions.

    Feature vector (8 dims):
      0  rows_ingested             — z-score normalized
      1  ingestion_rate            — z-score normalized
      2  latency_ms                — min-max [0, 15000] → [0, 1]
      3  file_count                — min-max [0, 2000]  → [0, 1]
      4  block_utilization         — already [0, 1]
      5  total_size_kb             — min-max [0, 50000] → [0, 1]
      6  file_size_skew_kb         — min-max [0, 50]    → [0, 1]
      7  steps_since_compact       — exponential decay τ=10
    """

    NUM_FEATURES = 8
    NUM_ACTIONS  = 4
    WINDOW_SIZE  = 10

    # Normalization — must match training notebook ceilings
    # (aligned with simulation.py RewardCalculator constants)
    LATENCY_MAX    = 15000.0   # was 2000  — training used 15000
    FILE_MAX       = 2000.0    # was 100   — training used 2000
    SIZE_MAX       = 50000.0   # was 5000  — training used 50000
    SKEW_MAX       = 50.0
    DECAY_TAU      = 10.0

    # For z-score normalization (running estimates, updated externally or fixed)
    _rows_mean: float = 5.0
    _rows_std: float  = 3.0
    _rate_mean: float = 10.0
    _rate_std: float  = 8.0

    def __init__(self, weights_path: Optional[str] = None, model_type: str = "attentive_ppo"):
        super().__init__(
            name="CompactionAgent",
            num_actions=self.NUM_ACTIONS,
            num_features=self.NUM_FEATURES,
            window_size=self.WINDOW_SIZE,
            weights_path=weights_path,
            model_type=model_type,
        )

    def extract_features(self, obs: Observation) -> np.ndarray:
        """Extract 8 compaction-relevant features."""
        f = np.zeros(self.NUM_FEATURES, dtype=np.float32)

        # 0: rows_ingested (z-score)
        f[0] = (obs.rows_ingested - self._rows_mean) / max(self._rows_std, 1e-6)

        # 1: ingestion_rate (z-score)
        f[1] = (obs.ingestion_rate_rows_per_sec - self._rate_mean) / max(self._rate_std, 1e-6)

        # 2: latency (min-max)
        f[2] = min(max(obs.latency_ms, 0.0) / self.LATENCY_MAX, 1.0)

        # 3: file_count (min-max)
        f[3] = min(obs.file_count / self.FILE_MAX, 1.0)

        # 4: block_utilization (already [0,1])
        f[4] = min(max(obs.block_utilization, 0.0), 1.0)

        # 5: total_size_kb (min-max)
        f[5] = min(obs.total_size_kb / self.SIZE_MAX, 1.0)

        # 6: file_size_skew (min-max)
        f[6] = min(obs.file_size_skew_kb / self.SKEW_MAX, 1.0)

        # 7: steps_since_compact (exponential decay)
        f[7] = np.exp(-obs.steps_since_compact / self.DECAY_TAU)

        return f

    def get_action(self, obs: Observation) -> Tuple[Action, np.ndarray, float]:
        """
        Returns (global_action, probabilities, value_estimate).
        """
        local_idx, probs, value = self.predict(obs)
        action = COMPACT_IDX_TO_ACTION[local_idx]
        return action, probs, value

    def get_action_stochastic(self, obs: Observation) -> Tuple[Action, np.ndarray, float]:
        local_idx, probs, value = self.predict_stochastic(obs)
        action = COMPACT_IDX_TO_ACTION[local_idx]
        return action, probs, value
