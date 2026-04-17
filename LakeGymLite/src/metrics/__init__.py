"""
LakeGym v5 — Offline Metrics Collector

Records transitions (s, a, r, s') as they occur during experiments.
Exports to CSV for offline RL training.
"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from threading import Lock
from typing import List, Dict, Any, Optional

from policies.base import Action, Observation


class OfflineMetricsCollector:
    """
    Collects transitions during simulation runs for offline RL training.

    Each transition is:
        (observation, action, compact_reward, partition_reward, global_reward, next_observation, done)

    Observations are stored as flat dicts with all 17 fields from Observation.
    """

    # Column names for the transition CSV
    OBS_FIELDS = [f.name for f in Observation.__dataclass_fields__.values()]
    NEXT_OBS_FIELDS = [f"next_{f}" for f in OBS_FIELDS]

    HEADER = (
        ['step', 'policy_name', 'action', 'action_name',
         'compact_reward', 'partition_reward', 'global_reward', 'done']
        + OBS_FIELDS
        + NEXT_OBS_FIELDS
    )

    def __init__(self):
        self._lock = Lock()
        self._transitions: List[Dict[str, Any]] = []
        self._prev_obs: Optional[Dict[str, Any]] = None
        self._step_count = 0
        self._policy_name = "unknown"

    def set_policy(self, name: str) -> None:
        with self._lock:
            self._policy_name = name

    def record_step(
        self,
        obs_dict: Dict[str, Any],
        action: Action,
        compact_reward: float,
        partition_reward: float,
        global_reward: float,
        next_obs_dict: Dict[str, Any],
        done: bool = False,
    ) -> None:
        """Record one (s, a, r, s') transition."""
        with self._lock:
            self._step_count += 1
            row: Dict[str, Any] = {
                'step': self._step_count,
                'policy_name': self._policy_name,
                'action': int(action),
                'action_name': action.name,
                'compact_reward': round(compact_reward, 6),
                'partition_reward': round(partition_reward, 6),
                'global_reward': round(global_reward, 6),
                'done': int(done),
            }
            for f in self.OBS_FIELDS:
                row[f] = obs_dict.get(f, 0)
            for f in self.OBS_FIELDS:
                row[f"next_{f}"] = next_obs_dict.get(f, 0)
            self._transitions.append(row)

    def record_from_result(
        self,
        prev_obs: Dict[str, Any],
        action: Action,
        step_result: Any,
        next_obs: Dict[str, Any],
        done: bool = False,
    ) -> None:
        """Convenience: record using a StepResult object."""
        self.record_step(
            obs_dict=prev_obs,
            action=action,
            compact_reward=step_result.compact_reward,
            partition_reward=step_result.partition_reward,
            global_reward=step_result.global_reward,
            next_obs_dict=next_obs,
            done=done,
        )

    def export_transitions(self, filepath: str) -> str:
        """Export all transitions to CSV."""
        with self._lock:
            if not self._transitions:
                return "No transitions to export"
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADER)
                writer.writeheader()
                writer.writerows(self._transitions)
            return f"✅ Exported {len(self._transitions)} transitions → {filepath}"

    def export_episode_stats(self, filepath: str) -> str:
        """Export per-step summary statistics."""
        with self._lock:
            if not self._transitions:
                return "No data"
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            # Build per-step stats
            stats = []
            for t in self._transitions:
                stats.append({
                    'step': t['step'],
                    'policy_name': t['policy_name'],
                    'action': t['action_name'],
                    'global_reward': t['global_reward'],
                    'compact_reward': t['compact_reward'],
                    'partition_reward': t['partition_reward'],
                    'latency_ms': t.get('latency_ms', 0),
                    'file_count': t.get('file_count', 0),
                    'block_utilization': t.get('block_utilization', 0),
                    'partition_pruning_ratio': t.get('partition_pruning_ratio', 0),
                })
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=stats[0].keys())
                writer.writeheader()
                writer.writerows(stats)
            return f"✅ Exported {len(stats)} stats → {filepath}"

    def clear(self) -> None:
        with self._lock:
            self._transitions.clear()
            self._prev_obs = None
            self._step_count = 0

    @property
    def transition_count(self) -> int:
        with self._lock:
            return len(self._transitions)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._transitions:
                return {'count': 0}
            rewards = [t['global_reward'] for t in self._transitions]
            actions = [t['action_name'] for t in self._transitions]
            from collections import Counter
            return {
                'count': len(self._transitions),
                'avg_global_reward': round(sum(rewards) / len(rewards), 4),
                'total_global_reward': round(sum(rewards), 4),
                'action_distribution': dict(Counter(actions)),
            }
