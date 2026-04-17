"""
LakeGym v5 — Experiment Manager

Runs policy evaluation experiments: for each policy, run N episodes of M steps,
collect transitions, compute stats, export CSVs.
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from threading import Event

from policies.base import Action, Observation, BasePolicy
from simulation import LakeSimulator
from metrics import OfflineMetricsCollector


class ExperimentManager:
    """
    Manages experiment runs for policy comparison.

    Usage:
        em = ExperimentManager(simulator)
        em.run_experiment(
            policy=some_policy,
            num_episodes=5,
            steps_per_episode=1000,
            output_dir="results/PolicyName",
        )
    """

    def __init__(self, simulator: LakeSimulator):
        self._sim = simulator
        self._stop_event = Event()
        self._is_running = False
        self._current_policy: Optional[str] = None
        self._progress: Dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def progress(self) -> Dict[str, Any]:
        return dict(self._progress)

    def stop(self) -> None:
        self._stop_event.set()

    def run_experiment(
        self,
        policy: BasePolicy,
        num_episodes: int = 5,
        steps_per_episode: int = 1000,
        output_dir: str = "results",
        workload_plan: Optional[Dict] = None,
        on_step: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run a full experiment: multiple episodes with the given policy.

        Args:
            policy:             Policy to evaluate
            num_episodes:       Number of episodes to run
            steps_per_episode:  Steps per episode
            output_dir:         Directory for result CSVs
            workload_plan:      Optional workload plan JSON dict
            on_step:            Optional callback(step_result) for UI updates

        Returns:
            Summary dict with statistics.
        """
        self._is_running = True
        self._stop_event.clear()
        self._current_policy = policy.name
        os.makedirs(output_dir, exist_ok=True)

        # Load workload plan if provided
        if workload_plan:
            self._sim.scenario_manager.load_plan(workload_plan)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_transitions = OfflineMetricsCollector()
        all_transitions.set_policy(policy.name)
        episode_summaries = []

        try:
            for ep in range(num_episodes):
                if self._stop_event.is_set():
                    break

                print(f"\n{'='*60}")
                print(f"  Episode {ep+1}/{num_episodes} — Policy: {policy.name}")
                print(f"{'='*60}")

                # Reset
                self._sim.reset()
                policy.reset()
                ep_collector = OfflineMetricsCollector()
                ep_collector.set_policy(policy.name)

                prev_obs = self._sim.get_observation()
                ep_start = time.time()

                for step in range(1, steps_per_episode + 1):
                    if self._stop_event.is_set():
                        break

                    self._progress = {
                        'policy': policy.name,
                        'episode': ep + 1,
                        'total_episodes': num_episodes,
                        'step': step,
                        'total_steps': steps_per_episode,
                        'pct': (ep * steps_per_episode + step) / (num_episodes * steps_per_episode) * 100,
                    }

                    # Get action
                    obs = Observation.from_dict(prev_obs)
                    action = policy.get_action(obs)

                    # Execute step
                    result = self._sim.run_step(action)
                    next_obs = self._sim.get_observation()

                    # Record transition
                    done = (step == steps_per_episode)
                    ep_collector.record_from_result(prev_obs, action, result, next_obs, done)
                    all_transitions.record_from_result(prev_obs, action, result, next_obs, done)

                    if on_step:
                        on_step(result)

                    # Progress logging (mirrors UI log output)
                    if step % 20 == 0:
                        pct = (ep * steps_per_episode + step) / (num_episodes * steps_per_episode) * 100
                        print(
                            f'  [Ep{ep+1} S{step:4d}] {result.action_taken:20s} | '
                            f'lat={result.latency_ms:6.0f}ms  files={result.file_count:4d}  '
                            f'util={result.block_utilization:.2f}  '
                            f'R={result.global_reward:+.3f}  ({pct:.0f}%)'
                        )
                    if step % 100 == 0:
                        ingest_desc = self._sim.scenario_manager.get_ingestion_description(step) or 'N/A'
                        print(
                            f'    \U0001f4cb S{step}: profile={result.workload_profile}  '
                            f'rows={result.rows_ingested}  rate={result.ingestion_rate_rows_per_sec:.0f}r/s'
                        )
                        print(
                            f'    \U0001f4e6 S{step}: ingestion phase="{ingest_desc}"'
                        )

                    prev_obs = next_obs

                ep_time = time.time() - ep_start

                # Export episode data
                ep_trans_path = os.path.join(output_dir, f"transitions_{timestamp}_ep{ep+1}.csv")
                ep_collector.export_transitions(ep_trans_path)

                ep_summary = {
                    'episode': ep + 1,
                    'steps': step,
                    'time_sec': round(ep_time, 1),
                    **self._sim.get_summary_stats(),
                    **ep_collector.get_summary(),
                }
                episode_summaries.append(ep_summary)
                print(f"  Episode {ep+1} done in {ep_time:.0f}s — "
                      f"avg_latency: {ep_summary.get('avg_latency', 0):.0f}ms, "
                      f"avg_reward: {ep_summary.get('avg_global_reward', 0):.4f}")

            # Export combined transitions
            combined_path = os.path.join(output_dir, f"transitions_{timestamp}.csv")
            all_transitions.export_transitions(combined_path)

            # Export episode stats
            stats_path = os.path.join(output_dir, f"episode_stats_{timestamp}.csv")
            self._export_episode_stats(episode_summaries, stats_path)

            summary = {
                'policy': policy.name,
                'num_episodes': len(episode_summaries),
                'total_transitions': all_transitions.transition_count,
                'episodes': episode_summaries,
                'transitions_file': combined_path,
                'stats_file': stats_path,
            }
            return summary

        finally:
            self._is_running = False
            self._current_policy = None
            self._sim.scenario_manager.clear()

    def run_batch(
        self,
        policies: List[BasePolicy],
        num_episodes: int = 5,
        steps_per_episode: int = 1000,
        base_dir: str = "results",
        workload_plan: Optional[Dict] = None,
        on_step: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Run experiments for multiple policies sequentially."""
        results = []
        for i, policy in enumerate(policies):
            if self._stop_event.is_set():
                break
            print(f"\n{'#'*60}")
            print(f"  Policy {i+1}/{len(policies)}: {policy.name}")
            print(f"{'#'*60}")

            out_dir = os.path.join(base_dir, policy.name.replace(' ', '_'))
            summary = self.run_experiment(
                policy=policy,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                output_dir=out_dir,
                workload_plan=workload_plan,
                on_step=on_step,
            )
            results.append(summary)
        return results

    def _export_episode_stats(self, summaries: List[Dict], filepath: str):
        """Export episode-level stats to CSV."""
        import csv
        if not summaries:
            return
        keys = summaries[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
