"""
LakeGym v5 — Agents Package

Exposes: AttentivePPO, CompactionAgent, PartitionAgent, MetaController
         + Model Registry (BaseModel, build_model, MODEL_REGISTRY)
"""

from agents.model import AttentivePPO, build_and_init
from agents.model_registry import BaseModel, build_model, MODEL_REGISTRY
from agents.base_agent import BaseAgent
from agents.compaction_agent import CompactionAgent
from agents.partition_agent import PartitionAgent
from agents.meta_controller import MetaController, MetaDecision, Delegation

__all__ = [
    "AttentivePPO",
    "build_and_init",
    "BaseModel",
    "build_model",
    "MODEL_REGISTRY",
    "BaseAgent",
    "CompactionAgent",
    "PartitionAgent",
    "MetaController",
    "MetaDecision",
    "Delegation",
]
