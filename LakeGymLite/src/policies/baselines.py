"""
LakeGym v5 — Baseline Policies

Classic baseline policies for experimental comparison.
Each policy extends BasePolicy and maps Observation → Action.

Experiment Design (11 unique runs):
  Group A — Fix target (C64), vary timing:
    • NoMaintenance        — always NOOP (control)
    • AlwaysCompact_C32    — compact every step, 32 KB  (~350 files, ~40% util)
    • AlwaysCompact_C64    — compact every step, 64 KB  (~160 files, ~85% util)
    • AlwaysCompact_C128   — compact every step, 128 KB (~91 files, ~140% util)
    • Periodic5_C64        — compact every 5 steps
    • Periodic10_C64       — compact every 10 steps
    • Threshold10_C64      — compact when files > 10
    • Threshold20_C64      — compact when files > 20
    • Random               — uniform random (control)

  Group B — Fix timing (Periodic10), vary target:
    • Periodic10_C32       — every 10 steps, 32 KB target
    • Periodic10_C64       — (shared with Group A)
    • Periodic10_C128      — every 10 steps, 128 KB target

  Group C — Fix timing (Threshold10), vary target:
    • Threshold10_C32      — files > 10, 32 KB target
    • Threshold10_C64      — (shared with Group A)
    • Threshold10_C128     — files > 10, 128 KB target
"""

from __future__ import annotations

import random
from typing import List

from policies.base import Action, Observation, BasePolicy, COMPACTION_ACTIONS


# ═══════════════════════════════════════════════════════════════
# NO MAINTENANCE
# ═══════════════════════════════════════════════════════════════

class NoMaintenancePolicy(BasePolicy):
    """Never performs any maintenance — pure control baseline."""

    def __init__(self):
        super().__init__(
            name="No_Maintenance",
            description="Always NOOP. Measures natural degradation.",
        )

    def get_action(self, obs: Observation) -> Action:
        self._record_action(Action.NOOP)
        return Action.NOOP


# ═══════════════════════════════════════════════════════════════
# ALWAYS COMPACT
# ═══════════════════════════════════════════════════════════════

class AlwaysCompactPolicy(BasePolicy):
    """Compact every step with a fixed target size."""

    def __init__(self, target: Action = Action.COMPACT_64KB):
        self._target = target
        super().__init__(
            name=f"AlwaysCompact_{target.name}",
            description=f"Compact every step with {target.name}.",
        )

    def get_action(self, obs: Observation) -> Action:
        self._record_action(self._target)
        return self._target


# ═══════════════════════════════════════════════════════════════
# PERIODIC
# ═══════════════════════════════════════════════════════════════

class PeriodicPolicy(BasePolicy):
    """Compact every N steps using the specified target."""

    def __init__(self, period: int = 5, target: Action = Action.COMPACT_64KB):
        self._period = period
        self._target = target
        super().__init__(
            name=f"Periodic{period}_{target.name}",
            description=f"Compact every {period} steps with {target.name}.",
        )

    def get_action(self, obs: Observation) -> Action:
        action = self._target if (self._action_count + 1) % self._period == 0 else Action.NOOP
        self._record_action(action)
        return action


# ═══════════════════════════════════════════════════════════════
# THRESHOLD
# ═══════════════════════════════════════════════════════════════

class ThresholdPolicy(BasePolicy):
    """Compact when newly accumulated files since last compaction exceed a threshold."""

    def __init__(
        self,
        threshold: int = 10,
        target: Action = Action.COMPACT_64KB,
        name: str = "",
    ):
        self._threshold = threshold
        self._target = target
        self._baseline_file_count = 0
        self._just_compacted = False
        auto_name = name or f"Threshold{threshold}_{target.name}"
        super().__init__(
            name=auto_name,
            description=f"Compact ({target.name}) when {threshold} new files accumulate.",
        )

    def reset(self) -> None:
        super().reset()
        self._baseline_file_count = 0
        self._just_compacted = False

    def get_action(self, obs: Observation) -> Action:
        # After compaction, record the post-compaction file count as new baseline
        if self._just_compacted:
            self._baseline_file_count = obs.file_count
            self._just_compacted = False

        new_files = obs.file_count - self._baseline_file_count
        if new_files >= self._threshold:
            self._just_compacted = True
            action = self._target
        else:
            action = Action.NOOP
        self._record_action(action)
        return action


class ThresholdC32Policy(ThresholdPolicy):
    def __init__(self, threshold: int = 10):
        super().__init__(threshold=threshold, target=Action.COMPACT_32KB, name=f"Threshold{threshold}_C32")


class ThresholdC64Policy(ThresholdPolicy):
    def __init__(self, threshold: int = 10):
        super().__init__(threshold=threshold, target=Action.COMPACT_64KB, name=f"Threshold{threshold}_C64")


class ThresholdC128Policy(ThresholdPolicy):
    def __init__(self, threshold: int = 10):
        super().__init__(threshold=threshold, target=Action.COMPACT_128KB, name=f"Threshold{threshold}_C128")


# ═══════════════════════════════════════════════════════════════
# RANDOM
# ═══════════════════════════════════════════════════════════════

class RandomCompactPolicy(BasePolicy):
    """Uniformly random over compaction actions only (NOOP, C32, C64, C128)."""

    def __init__(self):
        self._actions = list(COMPACTION_ACTIONS)
        super().__init__(
            name="Random_Compact",
            description="Uniform random over NOOP/C32/C64/C128.",
        )

    def get_action(self, obs: Observation) -> Action:
        action = random.choice(self._actions)
        self._record_action(action)
        return action


# ═══════════════════════════════════════════════════════════════
# REGISTRY: name → constructor
# ═══════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    # ── Controls ──
    "No_Maintenance":     NoMaintenancePolicy,
    "Random_Compact":     RandomCompactPolicy,

    # ── Group A: Fix target (C64), vary timing ──
    "AlwaysCompact_C64":  lambda: AlwaysCompactPolicy(Action.COMPACT_64KB),
    "AlwaysCompact_C32":  lambda: AlwaysCompactPolicy(Action.COMPACT_32KB),
    "AlwaysCompact_C128": lambda: AlwaysCompactPolicy(Action.COMPACT_128KB),
    "Periodic5_C64":      lambda: PeriodicPolicy(period=5, target=Action.COMPACT_64KB),
    "Periodic10_C64":     lambda: PeriodicPolicy(period=10, target=Action.COMPACT_64KB),
    "Threshold10_C64":    lambda: ThresholdC64Policy(threshold=10),
    "Threshold20_C64":    lambda: ThresholdC64Policy(threshold=20),

    # ── Group B: Fix timing (Periodic10), vary target ──
    "Periodic10_C32":     lambda: PeriodicPolicy(period=10, target=Action.COMPACT_32KB),
    # Periodic10_C64 already in Group A
    "Periodic10_C128":    lambda: PeriodicPolicy(period=10, target=Action.COMPACT_128KB),

    # ── Group C: Fix timing (Threshold10), vary target ──
    "Threshold10_C32":    lambda: ThresholdC32Policy(threshold=10),
    # Threshold10_C64 already in Group A
    "Threshold10_C128":   lambda: ThresholdC128Policy(threshold=10),
}
