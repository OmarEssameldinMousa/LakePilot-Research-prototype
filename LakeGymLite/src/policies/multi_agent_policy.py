"""
LakeGym v5 — Multi-Agent Policy

Wraps the meta-controller + compaction agent + partition agent as a single
BasePolicy, so it can be used in experiments just like any baseline.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from policies.base import Action, Observation, BasePolicy
from agents.compaction_agent import CompactionAgent
from agents.partition_agent import PartitionAgent
from agents.meta_controller import MetaController, Delegation


class MultiAgentPolicy(BasePolicy):
    """
    Hierarchical multi-agent policy:
      1. Meta-controller evaluates urgency → delegates to specialist
      2. Specialist agent (compaction or partition) picks the action
      3. Action is returned

    Compatible with BasePolicy interface for use in ExperimentManager.
    """

    def __init__(
        self,
        compact_weights: Optional[str] = None,
        partition_weights: Optional[str] = None,
        stochastic: bool = False,
        name: str = "MultiAgent",
        model_type: str = "attentive_ppo",
    ):
        super().__init__(
            name=name,
            description=f"Hierarchical: meta-controller → compaction / partition agents ({model_type}).",
        )
        self._meta = MetaController()
        self._compact_agent = CompactionAgent(weights_path=compact_weights, model_type=model_type)
        self._partition_agent = PartitionAgent(weights_path=partition_weights, model_type=model_type)
        self._stochastic = stochastic
        self._model_type = model_type

        # For transparency / UI
        self._last_compact_probs = None
        self._last_partition_probs = None
        self._last_compact_value = 0.0
        self._last_partition_value = 0.0

    def get_action(self, obs: Observation) -> Action:
        """
        Run hierarchical decision:
          meta → specialist → action.
        """
        decision = self._meta.decide(obs)

        if decision.delegation == Delegation.COMPACTION:
            if self._stochastic:
                action, probs, value = self._compact_agent.get_action_stochastic(obs)
            else:
                action, probs, value = self._compact_agent.get_action(obs)
            self._last_compact_probs = probs
            self._last_compact_value = value

        elif decision.delegation == Delegation.PARTITION:
            if self._stochastic:
                action, probs, value = self._partition_agent.get_action_stochastic(obs)
            else:
                action, probs, value = self._partition_agent.get_action(obs)
            self._last_partition_probs = probs
            self._last_partition_value = value

        else:
            action = Action.NOOP

        self._meta.notify_action(action)
        self._record_action(action)
        return action

    def reset(self) -> None:
        super().reset()
        self._meta.reset()
        self._compact_agent.reset()
        self._partition_agent.reset()
        self._last_compact_probs = None
        self._last_partition_probs = None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        meta_info = self._meta.get_info()
        info.update({
            'meta_delegation': meta_info.get('delegation'),
            'meta_compact_urgency': meta_info.get('compact_urgency', 0.0),
            'meta_partition_urgency': meta_info.get('partition_urgency', 0.0),
            'meta_reason': meta_info.get('reason', ''),
            'compact_agent': self._compact_agent.get_info(),
            'partition_agent': self._partition_agent.get_info(),
            'compact_probs': self._last_compact_probs.tolist() if self._last_compact_probs is not None else None,
            'partition_probs': self._last_partition_probs.tolist() if self._last_partition_probs is not None else None,
        })
        return info

    @property
    def meta_controller(self) -> MetaController:
        return self._meta

    @property
    def compaction_agent(self) -> CompactionAgent:
        return self._compact_agent

    @property
    def partition_agent(self) -> PartitionAgent:
        return self._partition_agent


# ═══════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ═══════════════════════════════════════════════════════════════

class CompactOnlyPolicy(BasePolicy):
    """
    Ablation: only the compaction agent acts.
    Partition decisions are always NOOP.
    Proves that partition agent adds value.
    """

    def __init__(self, compact_weights: Optional[str] = None, stochastic: bool = False, model_type: str = "attentive_ppo"):
        super().__init__(
            name="CompactOnly",
            description=f"Ablation: compaction agent only, no partition agent ({model_type}).",
        )
        self._meta = MetaController()
        self._compact_agent = CompactionAgent(weights_path=compact_weights, model_type=model_type)
        self._stochastic = stochastic
        self._model_type = model_type
        self._last_compact_probs = None
        self._last_compact_value = 0.0

    def get_action(self, obs: Observation) -> Action:
        decision = self._meta.decide(obs)

        # Let the compaction agent act when:
        #   1. Meta explicitly delegates to COMPACTION, OR
        #   2. Meta delegates to PARTITION (which we can't do) but
        #      compaction urgency is above threshold — don't waste the step.
        should_compact = (
            decision.delegation == Delegation.COMPACTION
            or (
                decision.delegation == Delegation.PARTITION
                and decision.compact_urgency >= MetaController.MIN_URGENCY
            )
        )

        if should_compact:
            if self._stochastic:
                action, probs, value = self._compact_agent.get_action_stochastic(obs)
            else:
                action, probs, value = self._compact_agent.get_action(obs)
            self._last_compact_probs = probs
            self._last_compact_value = value
        else:
            action = Action.NOOP
        self._meta.notify_action(action)
        self._record_action(action)
        return action

    def reset(self) -> None:
        super().reset()
        self._meta.reset()
        self._compact_agent.reset()
        self._last_compact_probs = None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        meta_info = self._meta.get_info()
        info.update({
            'meta_delegation': meta_info.get('delegation'),
            'meta_compact_urgency': meta_info.get('compact_urgency', 0.0),
            'meta_partition_urgency': meta_info.get('partition_urgency', 0.0),
            'meta_reason': meta_info.get('reason', ''),
            'compact_probs': self._last_compact_probs.tolist() if self._last_compact_probs is not None else None,
            'partition_probs': None,
        })
        return info


class PartitionOnlyPolicy(BasePolicy):
    """
    Ablation: only the partition agent acts.
    Compaction decisions are always NOOP.
    Proves that compaction agent adds value.
    """

    def __init__(self, partition_weights: Optional[str] = None, stochastic: bool = False, model_type: str = "attentive_ppo"):
        super().__init__(
            name="PartitionOnly",
            description=f"Ablation: partition agent only, no compaction agent ({model_type}).",
        )
        self._meta = MetaController()
        self._partition_agent = PartitionAgent(weights_path=partition_weights, model_type=model_type)
        self._stochastic = stochastic
        self._model_type = model_type
        self._last_partition_probs = None
        self._last_partition_value = 0.0

    def get_action(self, obs: Observation) -> Action:
        decision = self._meta.decide(obs)

        # Let the partition agent act when:
        #   1. Meta explicitly delegates to PARTITION, OR
        #   2. Meta delegates to COMPACTION (which we can't do) but
        #      partition urgency is above threshold — don't waste the step.
        should_partition = (
            decision.delegation == Delegation.PARTITION
            or (
                decision.delegation == Delegation.COMPACTION
                and decision.partition_urgency >= MetaController.MIN_URGENCY
            )
        )

        if should_partition:
            if self._stochastic:
                action, probs, value = self._partition_agent.get_action_stochastic(obs)
            else:
                action, probs, value = self._partition_agent.get_action(obs)
            self._last_partition_probs = probs
            self._last_partition_value = value
        else:
            action = Action.NOOP
        self._meta.notify_action(action)
        self._record_action(action)
        return action

    def reset(self) -> None:
        super().reset()
        self._meta.reset()
        self._partition_agent.reset()
        self._last_partition_probs = None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        meta_info = self._meta.get_info()
        info.update({
            'meta_delegation': meta_info.get('delegation'),
            'meta_compact_urgency': meta_info.get('compact_urgency', 0.0),
            'meta_partition_urgency': meta_info.get('partition_urgency', 0.0),
            'meta_reason': meta_info.get('reason', ''),
            'compact_probs': None,
            'partition_probs': self._last_partition_probs.tolist() if self._last_partition_probs is not None else None,
        })
        return info
