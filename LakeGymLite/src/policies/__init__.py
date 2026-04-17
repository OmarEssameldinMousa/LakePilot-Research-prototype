"""
LakeGym v5 — Policies Package

Exposes all policies, registries, and base classes.
"""

from policies.base import (
    Action,
    Observation,
    BasePolicy,
    COMPACTION_ACTIONS,
    PARTITION_ACTIONS,
    COMPACT_IDX_TO_ACTION,
    PARTITION_IDX_TO_ACTION,
    ACTION_COSTS,
    MAINTENANCE_ACTIONS,
)
from policies.baselines import (
    NoMaintenancePolicy,
    AlwaysCompactPolicy,
    PeriodicPolicy,
    ThresholdPolicy,
    ThresholdC32Policy,
    ThresholdC64Policy,
    ThresholdC128Policy,
    RandomCompactPolicy,
    BASELINE_REGISTRY,
)
from policies.workload_baselines import (
    AdaptiveCompactionPolicy,
    WorkloadAwareThresholdPolicy,
    PartitionExplorationPolicy,
    RandomPartitionPolicy,
    WORKLOAD_BASELINE_REGISTRY,
)
from policies.multi_agent_policy import MultiAgentPolicy, CompactOnlyPolicy, PartitionOnlyPolicy

# ── Standard weight paths (auto-detected if present) ──
import os

def _find_weights(agent: str, model_type: str):
    """Look for model-specific weights, falling back to generic ones."""
    specific = f"trained_models/{agent}_{model_type}.weights.h5"
    generic  = f"trained_models/{agent}.weights.h5"
    if os.path.exists(specific):
        return specific
    elif model_type == "attentive_ppo" and os.path.exists(generic):
        return generic
    return None

_cw = _find_weights("compaction_agent", "attentive_ppo")
_pw = _find_weights("partition_agent", "attentive_ppo")

# Unified registry of all policies
ALL_POLICIES = {**BASELINE_REGISTRY, **WORKLOAD_BASELINE_REGISTRY}

# ── AttentivePPO (default) ──
ALL_POLICIES["MultiAgent"]    = lambda: MultiAgentPolicy(compact_weights=_cw, partition_weights=_pw)
ALL_POLICIES["CompactOnly"]   = lambda: CompactOnlyPolicy(compact_weights=_cw)
ALL_POLICIES["PartitionOnly"] = lambda: PartitionOnlyPolicy(partition_weights=_pw)

# ── DDQN variants ──
_cw_ddqn = _find_weights("compaction_agent", "ddqn")
_pw_ddqn = _find_weights("partition_agent", "ddqn")
ALL_POLICIES["MultiAgent_DDQN"]    = lambda: MultiAgentPolicy(
    compact_weights=_cw_ddqn, partition_weights=_pw_ddqn, model_type="ddqn", name="MultiAgent_DDQN"
)
ALL_POLICIES["CompactOnly_DDQN"]   = lambda: CompactOnlyPolicy(compact_weights=_cw_ddqn, model_type="ddqn")
ALL_POLICIES["PartitionOnly_DDQN"] = lambda: PartitionOnlyPolicy(partition_weights=_pw_ddqn, model_type="ddqn")

# ── MLP-PPO variants ──
_cw_mlp = _find_weights("compaction_agent", "mlp_ppo")
_pw_mlp = _find_weights("partition_agent", "mlp_ppo")
ALL_POLICIES["MultiAgent_MlpPPO"]    = lambda: MultiAgentPolicy(
    compact_weights=_cw_mlp, partition_weights=_pw_mlp, model_type="mlp_ppo", name="MultiAgent_MlpPPO"
)
ALL_POLICIES["CompactOnly_MlpPPO"]   = lambda: CompactOnlyPolicy(compact_weights=_cw_mlp, model_type="mlp_ppo")
ALL_POLICIES["PartitionOnly_MlpPPO"] = lambda: PartitionOnlyPolicy(partition_weights=_pw_mlp, model_type="mlp_ppo")

__all__ = [
    # Base
    "Action", "Observation", "BasePolicy",
    "COMPACTION_ACTIONS", "PARTITION_ACTIONS",
    "COMPACT_IDX_TO_ACTION", "PARTITION_IDX_TO_ACTION",
    "ACTION_COSTS", "MAINTENANCE_ACTIONS",
    # Baselines
    "NoMaintenancePolicy", "AlwaysCompactPolicy", "PeriodicPolicy",
    "ThresholdPolicy", "ThresholdC32Policy", "ThresholdC64Policy", "ThresholdC128Policy",
    "RandomPolicy", "BASELINE_REGISTRY",
    # Workload
    "AdaptiveCompactionPolicy", "WorkloadAwareThresholdPolicy", "PartitionExplorationPolicy",
    "WORKLOAD_BASELINE_REGISTRY",
    # Multi-agent
    "MultiAgentPolicy", "CompactOnlyPolicy", "PartitionOnlyPolicy",
    # Registry
    "ALL_POLICIES",
]

