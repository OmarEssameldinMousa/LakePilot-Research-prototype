"""
LakeGym v5 — Partition Agent

Specialist agent for partition decisions.
  • 14 input features → 5 actions (NOOP, HOUR, REGION, EVENT_TYPE, REMOVE)
  • window_size = 20
  • Extracts query-workload features from the full Observation.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from agents.base_agent import BaseAgent
from policies.base import (
    Action,
    Observation,
    PARTITION_IDX_TO_ACTION,
    PARTITION_ACTIONS,
)


class PartitionAgent(BaseAgent):
    """
    Specialist for partition-only actions.

    Feature vector (14 dims):
      0   latency_ms                → min-max [0, 15000] → [0, 1]
      1   file_count                → min-max [0, 2000]  → [0, 1]
      2-5 partition_one_hot[4]      → one-hot of current partition strategy
      6   partition_pruning_ratio   → already [0, 1]
      7   avg_pruning_ratio         → already [0, 1]
      8   query_hist_time_range     → already [0, 1]
      9   query_hist_region_filter  → already [0, 1]
      10  query_hist_sensor_lookup  → already [0, 1]
      11  query_hist_type_filter    → already [0, 1]
      12  query_hist_full_scan      → already [0, 1]
      13  steps_since_partition_change → exponential decay τ=30
    """

    NUM_FEATURES = 14
    NUM_ACTIONS  = 5
    WINDOW_SIZE  = 20

    # Normalization — must match training notebook ceilings
    # (aligned with simulation.py RewardCalculator constants)
    LATENCY_MAX = 15000.0   # was 2000 — training used 15000
    FILE_MAX    = 2000.0    # was 100  — training used 2000
    DECAY_TAU   = 30.0
    NUM_PARTITION_STRATEGIES = 4  # 0=none, 1=hour, 2=region, 3=event_type

    def __init__(self, weights_path: Optional[str] = None, model_type: str = "attentive_ppo"):
        super().__init__(
            name="PartitionAgent",
            num_actions=self.NUM_ACTIONS,
            num_features=self.NUM_FEATURES,
            window_size=self.WINDOW_SIZE,
            weights_path=weights_path,
            model_type=model_type,
        )

    def extract_features(self, obs: Observation) -> np.ndarray:
        """Extract 14 partition-relevant features."""
        f = np.zeros(self.NUM_FEATURES, dtype=np.float32)

        # 0: latency (min-max)
        f[0] = min(max(obs.latency_ms, 0.0) / self.LATENCY_MAX, 1.0)

        # 1: file_count (min-max)
        f[1] = min(obs.file_count / self.FILE_MAX, 1.0)

        # 2-5: partition strategy one-hot
        ps = int(obs.partition_strategy)
        if 0 <= ps < self.NUM_PARTITION_STRATEGIES:
            f[2 + ps] = 1.0

        # 6: partition_pruning_ratio
        f[6] = min(max(obs.partition_pruning_ratio, 0.0), 1.0)

        # 7: avg_pruning_ratio
        f[7] = min(max(obs.avg_pruning_ratio, 0.0), 1.0)

        # 8-12: query histogram
        f[8]  = obs.query_hist_time_range
        f[9]  = obs.query_hist_region_filter
        f[10] = obs.query_hist_sensor_lookup
        f[11] = obs.query_hist_type_filter
        f[12] = obs.query_hist_full_scan

        # 13: steps_since_partition_change (exponential decay)
        f[13] = np.exp(-obs.steps_since_partition_change / self.DECAY_TAU)

        return f

    def get_action(self, obs: Observation) -> Tuple[Action, np.ndarray, float]:
        """
        Returns (global_action, probabilities, value_estimate).
        """
        local_idx, probs, value = self.predict(obs)
        action = PARTITION_IDX_TO_ACTION[local_idx]
        return action, probs, value

    def get_action_stochastic(self, obs: Observation) -> Tuple[Action, np.ndarray, float]:
        local_idx, probs, value = self.predict_stochastic(obs)
        action = PARTITION_IDX_TO_ACTION[local_idx]
        return action, probs, value
