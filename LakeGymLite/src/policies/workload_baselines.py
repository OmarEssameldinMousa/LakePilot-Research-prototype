"""
LakeGym v5 — Workload-Aware Baseline Policies

Advanced baselines that use workload features and/or partition actions.
These generate training data for the partition agent.

Policies:
  • AdaptiveCompactionPolicy      — picks C32/C64/C128 based on ingestion rate
  • WorkloadAwareThresholdPolicy  — compacts + partitions based on query histogram
  • PartitionExplorationPolicy    — cycles through partition strategies for training data
"""

from __future__ import annotations

import random
from typing import Optional

from policies.base import Action, Observation, BasePolicy


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE COMPACTION
# ═══════════════════════════════════════════════════════════════

class AdaptiveCompactionPolicy(BasePolicy):
    """
    Picks compaction target based on ingestion rate:
      • High ingestion (> HIGH_THRESH rows/s) → C32  (cheap, frequent)
      • Medium ingestion                      → C64  (optimal)
      • Low ingestion (< LOW_THRESH rows/s)   → C128 (thorough, infrequent)

    Only compacts when file_count > file_threshold.
    """

    HIGH_THRESH = 200.0    # rows/sec (burst ingestion)
    LOW_THRESH  = 80.0     # rows/sec (warm-up / wind-down)
    FILE_THRESH = 10       # minimum files to trigger compaction

    def __init__(self, file_threshold: int = 10):
        self.FILE_THRESH = file_threshold
        super().__init__(
            name="AdaptiveCompaction",
            description="Picks C32/C64/C128 based on ingestion rate.",
        )

    def get_action(self, obs: Observation) -> Action:
        if obs.file_count <= self.FILE_THRESH:
            action = Action.NOOP
        elif obs.ingestion_rate_rows_per_sec > self.HIGH_THRESH:
            action = Action.COMPACT_32KB
        elif obs.ingestion_rate_rows_per_sec < self.LOW_THRESH:
            action = Action.COMPACT_128KB
        else:
            action = Action.COMPACT_64KB
        self._record_action(action)
        return action


# ═══════════════════════════════════════════════════════════════
# WORKLOAD-AWARE THRESHOLD
# ═══════════════════════════════════════════════════════════════

class WorkloadAwareThresholdPolicy(BasePolicy):
    """
    Combines compaction (threshold-based on accumulated files) with partition
    decisions based on which query type dominates the recent workload.

    Partition logic:
      • If TIME_RANGE > 40% of recent queries → partition by HOUR
      • If REGION_FILTER > 40%              → partition by REGION
      • If TYPE_FILTER > 40%                → partition by EVENT_TYPE
      • Otherwise → keep current or REMOVE_PARTITION

    Only re-partitions every PARTITION_INTERVAL steps.
    Compacts when accumulated new files > COMPACT_THRESHOLD.
    """

    COMPACT_THRESHOLD  = 10
    DOMINANT_FRACTION  = 0.40
    PARTITION_INTERVAL = 50   # minimum steps between partition changes

    def __init__(self):
        self._step_count = 0
        self._last_partition_step = 0
        self._baseline_file_count = 0
        self._just_compacted = False
        super().__init__(
            name="WorkloadAwareThreshold",
            description="Compacts at threshold + partitions by dominant query type.",
        )

    def get_action(self, obs: Observation) -> Action:
        self._step_count += 1

        # After compaction, record post-compaction file count as new baseline
        if self._just_compacted:
            self._baseline_file_count = obs.file_count
            self._just_compacted = False

        new_files = obs.file_count - self._baseline_file_count

        # ── Compaction check (higher priority) ──
        if new_files >= self.COMPACT_THRESHOLD:
            self._just_compacted = True
            action = Action.COMPACT_64KB
            self._record_action(action)
            return action

        # ── Partition check ──
        if (self._step_count - self._last_partition_step) >= self.PARTITION_INTERVAL:
            partition_action = self._pick_partition(obs)
            if partition_action is not None:
                self._last_partition_step = self._step_count
                self._record_action(partition_action)
                return partition_action

        action = Action.NOOP
        self._record_action(action)
        return action

    def _pick_partition(self, obs: Observation) -> Optional[Action]:
        """Decide partition based on dominant query type."""
        hist = {
            'time_range':    obs.query_hist_time_range,
            'region_filter': obs.query_hist_region_filter,
            'type_filter':   obs.query_hist_type_filter,
        }
        dominant = max(hist, key=hist.get)
        dominant_frac = hist[dominant]

        if dominant_frac < self.DOMINANT_FRACTION:
            return None  # no clear winner

        partition_map = {
            'time_range':    (Action.PARTITION_HOUR, 1),
            'region_filter': (Action.PARTITION_REGION, 2),
            'type_filter':   (Action.PARTITION_EVENT_TYPE, 3),
        }
        target_action, target_ps = partition_map[dominant]

        # Don't re-apply the same partition
        if obs.partition_strategy == target_ps:
            return None

        return target_action

    def reset(self) -> None:
        super().reset()
        self._step_count = 0
        self._last_partition_step = 0
        self._baseline_file_count = 0
        self._just_compacted = False


# ═══════════════════════════════════════════════════════════════
# PARTITION EXPLORATION
# ═══════════════════════════════════════════════════════════════

class PartitionExplorationPolicy(BasePolicy):
    """
    Systematically explores all partition strategies to generate
    diverse training data for the partition agent.

    Cycle:
      1. Start UNPARTITIONED for N steps
      2. Switch to HOUR for N steps
      3. Compact (to rewrite data into new layout)
      4. Switch to REGION for N steps
      5. Compact
      6. Switch to EVENT_TYPE for N steps
      7. Compact
      8. REMOVE partition for N steps
      9. Repeat

    Also compacts when file_count > threshold.
    """

    PHASE_LENGTH = 50         # steps per partition strategy
    COMPACT_THRESHOLD = 15    # new files since last compaction
    COMPACT_TARGET = Action.COMPACT_64KB

    PARTITION_CYCLE = [
        None,                     # UNPARTITIONED phase
        Action.PARTITION_HOUR,
        Action.PARTITION_REGION,
        Action.PARTITION_EVENT_TYPE,
        Action.REMOVE_PARTITION,  # back to unpartitioned
    ]

    def __init__(self):
        self._step = 0
        self._phase_idx = 0
        self._phase_start = 0
        self._just_switched = False
        self._baseline_file_count = 0
        self._just_compacted = False
        super().__init__(
            name="PartitionExploration",
            description="Cycles through all partition strategies for data collection.",
        )

    def get_action(self, obs: Observation) -> Action:
        self._step += 1
        steps_in_phase = self._step - self._phase_start

        # After compaction, update baseline
        if self._just_compacted:
            self._baseline_file_count = obs.file_count
            self._just_compacted = False

        new_files = obs.file_count - self._baseline_file_count

        # ── Phase transition ──
        if steps_in_phase >= self.PHASE_LENGTH:
            self._phase_idx = (self._phase_idx + 1) % len(self.PARTITION_CYCLE)
            self._phase_start = self._step
            partition_action = self.PARTITION_CYCLE[self._phase_idx]
            if partition_action is not None:
                self._just_switched = True
                self._record_action(partition_action)
                return partition_action

        # ── Compact after partition switch (to rewrite data) ──
        if self._just_switched and obs.file_count >= 2:
            self._just_switched = False
            self._just_compacted = True
            self._record_action(self.COMPACT_TARGET)
            return self.COMPACT_TARGET

        # ── Regular compaction when new files accumulate ──
        if new_files >= self.COMPACT_THRESHOLD:
            self._just_compacted = True
            self._record_action(self.COMPACT_TARGET)
            return self.COMPACT_TARGET

        self._record_action(Action.NOOP)
        return Action.NOOP

    def reset(self) -> None:
        super().reset()
        self._step = 0
        self._phase_idx = 0
        self._phase_start = 0
        self._just_switched = False
        self._baseline_file_count = 0
        self._just_compacted = False


# ═══════════════════════════════════════════════════════════════
# RANDOM PARTITION
# ═══════════════════════════════════════════════════════════════

class RandomPartitionPolicy(BasePolicy):
    """
    Uniformly random over partition actions only:
      NOOP, PARTITION_HOUR, PARTITION_REGION, PARTITION_EVENT_TYPE, REMOVE_PARTITION.

    Also compacts when accumulated files > threshold to prevent runaway file count.
    Provides stochastic exploration data for partition agent training.
    """

    COMPACT_THRESHOLD = 15

    PARTITION_CHOICES = [
        Action.NOOP,
        Action.PARTITION_HOUR,
        Action.PARTITION_REGION,
        Action.PARTITION_EVENT_TYPE,
        Action.REMOVE_PARTITION,
    ]

    def __init__(self):
        self._baseline_file_count = 0
        self._just_compacted = False
        super().__init__(
            name="Random_Partition",
            description="Random partition actions + threshold compaction.",
        )

    def get_action(self, obs: Observation) -> Action:
        # After compaction, update baseline
        if self._just_compacted:
            self._baseline_file_count = obs.file_count
            self._just_compacted = False

        new_files = obs.file_count - self._baseline_file_count

        # Compact if too many files accumulate (prevent runaway)
        if new_files >= self.COMPACT_THRESHOLD:
            self._just_compacted = True
            action = Action.COMPACT_64KB
            self._record_action(action)
            return action

        # Random partition action
        action = random.choice(self.PARTITION_CHOICES)
        self._record_action(action)
        return action

    def reset(self) -> None:
        super().reset()
        self._baseline_file_count = 0
        self._just_compacted = False


# ═══════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════

WORKLOAD_BASELINE_REGISTRY = {
    "AdaptiveCompaction":       AdaptiveCompactionPolicy,
    "WorkloadAwareThreshold":   WorkloadAwareThresholdPolicy,
    "PartitionExploration":     PartitionExplorationPolicy,
    "Random_Partition":         RandomPartitionPolicy,
}
