"""
LakeGym v5 — Base Policy Interface

Defines the shared vocabulary for the entire system:
  • Action     — unified discrete action enum (8 actions)
  • Observation — full environment observation dataclass
  • BasePolicy — abstract base class all policies implement

Every other module imports Action / Observation from here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, Any, Optional, List


# ═══════════════════════════════════════════════════════════════
# ACTION SPACE
# ═══════════════════════════════════════════════════════════════

class Action(IntEnum):
    """
    Unified action space for all agents and policies.

    Compaction actions (0-3):
        NOOP          — do nothing
        COMPACT_32KB  — compact to 32 KB target file size
        COMPACT_64KB  — compact to 64 KB target file size
        COMPACT_128KB — compact to 128 KB target file size

    Partition actions (4-7):
        PARTITION_HOUR       — partition by hours(event_ts)
        PARTITION_REGION     — partition by identity(region)
        PARTITION_EVENT_TYPE — partition by identity(event_type)
        REMOVE_PARTITION     — drop current partition field
    """
    NOOP               = 0
    COMPACT_32KB       = 1
    COMPACT_64KB       = 2
    COMPACT_128KB      = 3
    PARTITION_HOUR      = 4
    PARTITION_REGION    = 5
    PARTITION_EVENT_TYPE = 6
    REMOVE_PARTITION    = 7


# Convenience subsets — used by specialist agents
COMPACTION_ACTIONS: List[Action] = [
    Action.NOOP,
    Action.COMPACT_32KB,
    Action.COMPACT_64KB,
    Action.COMPACT_128KB,
]

PARTITION_ACTIONS: List[Action] = [
    Action.NOOP,
    Action.PARTITION_HOUR,
    Action.PARTITION_REGION,
    Action.PARTITION_EVENT_TYPE,
    Action.REMOVE_PARTITION,
]

# Mapping from compaction-agent local index → global Action
COMPACT_IDX_TO_ACTION: Dict[int, Action] = {
    0: Action.NOOP,
    1: Action.COMPACT_32KB,
    2: Action.COMPACT_64KB,
    3: Action.COMPACT_128KB,
}

# Mapping from partition-agent local index → global Action
PARTITION_IDX_TO_ACTION: Dict[int, Action] = {
    0: Action.NOOP,
    1: Action.PARTITION_HOUR,
    2: Action.PARTITION_REGION,
    3: Action.PARTITION_EVENT_TYPE,
    4: Action.REMOVE_PARTITION,
}

# Action costs (used by reward calculator)
ACTION_COSTS: Dict[Action, float] = {
    Action.NOOP:               0.0,
    Action.COMPACT_32KB:       0.9,
    Action.COMPACT_64KB:       1.0,
    Action.COMPACT_128KB:      1.1,
    Action.PARTITION_HOUR:      2.0,
    Action.PARTITION_REGION:    2.0,
    Action.PARTITION_EVENT_TYPE: 2.0,
    Action.REMOVE_PARTITION:    0.7,
}

MAINTENANCE_ACTIONS = {
    Action.COMPACT_32KB,
    Action.COMPACT_64KB,
    Action.COMPACT_128KB,
    Action.PARTITION_HOUR,
    Action.PARTITION_REGION,
    Action.PARTITION_EVENT_TYPE,
    Action.REMOVE_PARTITION,
}


# ═══════════════════════════════════════════════════════════════
# OBSERVATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """
    Full observation returned by the simulation each step.

    Fields are grouped into:
      • Table physical state  — file count, sizes, utilization
      • Partition state       — current strategy, change tracking
      • Query workload        — histogram + pruning metrics
    """

    # ── Table physical state ──
    rows_ingested: int = 0
    ingestion_rate_rows_per_sec: float = 0.0
    latency_ms: float = 0.0
    file_count: int = 0
    block_utilization: float = 0.0
    total_size_kb: float = 0.0
    file_size_skew_kb: float = 0.0

    # ── Partition state ──
    partition_strategy: int = 0          # 0=none, 1=hour, 2=region, 3=event_type
    steps_since_compact: int = 0
    steps_since_partition_change: int = 0

    # ── Query workload features ──
    partition_pruning_ratio: float = 0.0   # pruning for the latest query
    avg_pruning_ratio: float = 0.0         # rolling average over last 20 queries
    query_hist_time_range: float = 0.0     # fraction of recent queries
    query_hist_region_filter: float = 0.0
    query_hist_sensor_lookup: float = 0.0
    query_hist_type_filter: float = 0.0
    query_hist_full_scan: float = 0.0

    # ── Convenience ──
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        """Create Observation from a dictionary (ignores unknown keys)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# BASE POLICY
# ═══════════════════════════════════════════════════════════════

class BasePolicy(ABC):
    """
    Abstract base class that every policy must implement.

    A policy maps an Observation to an Action.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._action_count: int = 0
        self._last_action: Optional[Action] = None

    # ── Core interface ──

    @abstractmethod
    def get_action(self, obs: Observation) -> Action:
        """Select an action given the current observation."""
        ...

    def reset(self) -> None:
        """Reset internal state (called at episode start)."""
        self._action_count = 0
        self._last_action = None

    # ── Bookkeeping ──

    def _record_action(self, action: Action) -> None:
        self._action_count += 1
        self._last_action = action

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def last_action(self) -> Optional[Action]:
        return self._last_action

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'action_count': self._action_count,
            'last_action': self._last_action.name if self._last_action else None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
