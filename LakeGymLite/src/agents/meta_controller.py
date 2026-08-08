"""
LakeGym v5 — Meta-Controller

Rule-based arbitrator that decides WHICH specialist agent acts each step.
Computes urgency scores for compaction and partitioning, then delegates.

Design: rule-based (not learned). The meta-controller is transparent and
interpretable — important for research reproducibility and paper presentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from policies.base import Action, Observation


# ═══════════════════════════════════════════════════════════════
# DECISION ENUM
# ═══════════════════════════════════════════════════════════════

class Delegation:
    """Which specialist the meta-controller delegates to."""
    NOOP       = "noop"
    COMPACTION = "compaction"
    PARTITION  = "partition"


@dataclass
class MetaDecision:
    """Result of the meta-controller's arbitration."""
    delegation: str                   # Delegation constant
    compact_urgency: float = 0.0     # [0, 1]
    partition_urgency: float = 0.0   # [0, 1]
    reason: str = ""


# ═══════════════════════════════════════════════════════════════
# META-CONTROLLER
# ═══════════════════════════════════════════════════════════════

class MetaController:
    """
    Rule-based meta-controller.

    Urgency scores:
        compact_urgency = f(file_count, block_utilization)
        partition_urgency = f(avg_pruning_ratio, query_diversity, steps_since_change)

    Cooldowns:
        • Compaction: wait COMPACT_COOLDOWN steps after last compaction
        • Partition: wait PARTITION_COOLDOWN steps after last partition change

    Thresholds:
        • High file_count → high compaction urgency
        • Low avg_pruning + diverse workload → high partition urgency
        • Only delegate if urgency > MIN_URGENCY
    """

    # Cooldowns (steps)
    COMPACT_COOLDOWN   = 4
    PARTITION_COOLDOWN = 20

    # Urgency thresholds
    MIN_URGENCY = 0.35

    # Compaction urgency curve (file_count → urgency)
    FILE_CRITICAL = 96      # → urgency = 0.95
    FILE_HIGH     = 36      # → urgency = 0.60
    FILE_LOW      = 12      # → urgency = 0.10

    # Block utilization thresholds
    UTIL_LOW = 0.55         # low utilization → urgency boost

    # Partition urgency factors
    PRUNING_LOW = 0.25      # avg pruning below this → consider re-partition
    DIVERSITY_THRESHOLD = 0.45  # if dominant query type fraction > this → focused workload

    # Utilisation-boost magnitude (was hard-coded inline as 0.15)
    UTIL_BOOST = 0.15

    def __init__(self):
        # Sensitivity-analysis overrides (revision Phase 3). Defaults reproduce the
        # published constants exactly, so behaviour is unchanged unless a caller
        # explicitly sets these environment variables.
        self.MIN_URGENCY = float(os.getenv('LAKEGYM_META_THETA', MetaController.MIN_URGENCY))
        self.UTIL_BOOST = float(os.getenv('LAKEGYM_META_UTIL_BOOST', MetaController.UTIL_BOOST))
        self.COMPACT_COOLDOWN = int(os.getenv('LAKEGYM_META_COMPACT_COOLDOWN',
                                              MetaController.COMPACT_COOLDOWN))
        self.PARTITION_COOLDOWN = int(os.getenv('LAKEGYM_META_PARTITION_COOLDOWN',
                                                MetaController.PARTITION_COOLDOWN))

        self._step = 0
        self._last_compact_step = 0
        self._last_partition_step = 0
        self._last_decision: Optional[MetaDecision] = None

    def config(self) -> Dict[str, Any]:
        """Active constants, for run manifests."""
        return {'min_urgency': self.MIN_URGENCY, 'util_boost': self.UTIL_BOOST,
                'compact_cooldown': self.COMPACT_COOLDOWN,
                'partition_cooldown': self.PARTITION_COOLDOWN}

    def decide(self, obs: Observation) -> MetaDecision:
        """
        Evaluate urgency scores and decide delegation.

        Args:
            obs: Full observation from the simulation.

        Returns:
            MetaDecision with delegation target and urgency scores.
        """
        self._step += 1

        # ── Compaction urgency ──
        cu = self._compaction_urgency(obs)

        # ── Partition urgency ──
        pu = self._partition_urgency(obs)

        # ── Cooldown masking ──
        compact_ok = obs.steps_since_compact >= self.COMPACT_COOLDOWN
        partition_ok = obs.steps_since_partition_change >= self.PARTITION_COOLDOWN

        if not compact_ok:
            cu = 0.0
        if not partition_ok:
            pu = 0.0

        # ── Arbitration ──
        if cu < self.MIN_URGENCY and pu < self.MIN_URGENCY:
            decision = MetaDecision(
                delegation=Delegation.NOOP,
                compact_urgency=cu,
                partition_urgency=pu,
                reason="Both urgencies below threshold",
            )
        elif cu >= pu:
            decision = MetaDecision(
                delegation=Delegation.COMPACTION,
                compact_urgency=cu,
                partition_urgency=pu,
                reason=f"Compaction urgency {cu:.2f} >= partition {pu:.2f}",
            )
        else:
            decision = MetaDecision(
                delegation=Delegation.PARTITION,
                compact_urgency=cu,
                partition_urgency=pu,
                reason=f"Partition urgency {pu:.2f} > compaction {cu:.2f}",
            )

        self._last_decision = decision
        return decision

    def _compaction_urgency(self, obs: Observation) -> float:
        """Compute compaction urgency from file_count and block_util."""
        fc = obs.file_count

        # Piecewise linear urgency
        if fc >= self.FILE_CRITICAL:
            u = 0.95
        elif fc >= self.FILE_HIGH:
            # Linear interpolation from 0.60 to 0.95
            t = (fc - self.FILE_HIGH) / (self.FILE_CRITICAL - self.FILE_HIGH)
            u = 0.60 + t * 0.35
        elif fc >= self.FILE_LOW:
            # Linear interpolation from 0.10 to 0.60
            t = (fc - self.FILE_LOW) / (self.FILE_HIGH - self.FILE_LOW)
            u = 0.10 + t * 0.50
        else:
            u = 0.0

        # Boost if block utilization is very low
        if obs.block_utilization < self.UTIL_LOW and fc > 3:
            u = min(u + self.UTIL_BOOST, 1.0)

        return u

    def _partition_urgency(self, obs: Observation) -> float:
        """Compute partition urgency from pruning and workload focus."""
        # If pruning is already good, no urgency
        if obs.avg_pruning_ratio > 0.5:
            return 0.0

        u = 0.0

        # Low pruning → some urgency
        if obs.avg_pruning_ratio < self.PRUNING_LOW:
            u += 0.40

        # Focused workload (one query type dominates) → partition could help
        hist = [
            obs.query_hist_time_range,
            obs.query_hist_region_filter,
            obs.query_hist_sensor_lookup,
            obs.query_hist_type_filter,
            obs.query_hist_full_scan,
        ]
        max_frac = max(hist) if hist else 0.0
        if max_frac > self.DIVERSITY_THRESHOLD:
            u += 0.30  # focused workload → partition likely beneficial

        # Long time since partition change → might be stale
        if obs.steps_since_partition_change > 100:
            u += 0.10

        return min(u, 1.0)

    def notify_action(self, action: Action) -> None:
        """Update internal tracking after an action is taken."""
        if action in (Action.COMPACT_32KB, Action.COMPACT_64KB, Action.COMPACT_128KB):
            self._last_compact_step = self._step
        elif action in (Action.PARTITION_HOUR, Action.PARTITION_REGION,
                        Action.PARTITION_EVENT_TYPE, Action.REMOVE_PARTITION):
            self._last_partition_step = self._step

    def reset(self) -> None:
        self._step = 0
        self._last_compact_step = 0
        self._last_partition_step = 0
        self._last_decision = None

    @property
    def last_decision(self) -> Optional[MetaDecision]:
        return self._last_decision

    def get_info(self) -> Dict[str, Any]:
        d = self._last_decision
        return {
            'step': self._step,
            'delegation': d.delegation if d else None,
            'compact_urgency': d.compact_urgency if d else 0.0,
            'partition_urgency': d.partition_urgency if d else 0.0,
            'reason': d.reason if d else "",
        }
